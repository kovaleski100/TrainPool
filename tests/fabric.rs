use std::sync::Arc;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::{TcpListener, UdpSocket},
};
use trainpool::{
    config::Config,
    memory::{
        block::MemoryBlockHandle,
        remote_ram::{read_chunk, write_chunk},
    },
    protocol::{Request, Response},
    runtime::Runtime,
    scheduler::planner::TrainingPlan,
    transport::{
        Transport, read_frame,
        udp::{Kind, MAX_DATAGRAM, Packet},
        write_frame,
    },
};
use uuid::Uuid;

async fn start() -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    start_with_limit(32 * 1024 * 1024).await
}
async fn start_with_limit(limit: u64) -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    start_with_options(limit, 128).await
}
async fn start_with_options(
    limit: u64,
    udp_max_sessions: usize,
) -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    start_with_transport(limit, udp_max_sessions, "udp").await
}
async fn start_with_transport(
    limit: u64,
    udp_max_sessions: usize,
    data_transport: &str,
) -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = Config {
        listen: listener.local_addr().unwrap(),
        discovery_enabled: false,
        ram_limit_bytes: Some(limit),
        chunk_bytes: 4096,
        cluster_secret: Some("integration secret".into()),
        udp_max_sessions,
        data_transport: data_transport.into(),
        ..Default::default()
    };
    let runtime = Runtime::new(config, Uuid::new_v4()).await.unwrap();
    let r = runtime.clone();
    let task = tokio::spawn(async move {
        r.serve(listener).await.unwrap();
    });
    (runtime, task)
}

async fn routed_write(
    runtime: &Runtime,
    address: std::net::SocketAddr,
    handle: &MemoryBlockHandle,
    transfer_id: Uuid,
    offset: u64,
    payload: &[u8],
) {
    let mut stream = runtime.transport.connect(address).await.unwrap();
    write_frame(
        &mut stream,
        &Request::WriteChunk {
            handle: handle.clone(),
            transfer_id,
            offset,
            length: payload.len(),
            total_size: handle.size,
            checksum: blake3::hash(payload).to_hex().to_string(),
            direct: false,
        },
    )
    .await
    .unwrap();
    read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .check()
        .unwrap();
    stream.write_all(payload).await.unwrap();
    read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .check()
        .unwrap();
}

async fn routed_read(
    runtime: &Runtime,
    address: std::net::SocketAddr,
    handle: &MemoryBlockHandle,
    offset: u64,
    length: usize,
) -> Vec<u8> {
    let transfer_id = Uuid::new_v4();
    let mut stream = runtime.transport.connect(address).await.unwrap();
    write_frame(
        &mut stream,
        &Request::ReadChunk {
            handle: handle.clone(),
            transfer_id,
            offset,
            length,
            direct: false,
        },
    )
    .await
    .unwrap();
    let metadata: serde_json::Value = read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let mut payload = vec![0; length];
    stream.read_exact(&mut payload).await.unwrap();
    assert_eq!(
        metadata["checksum"],
        blake3::hash(&payload).to_hex().as_str()
    );
    payload
}

async fn udp_send(socket: &UdpSocket, address: std::net::SocketAddr, packet: &Packet) {
    socket
        .send_to(&packet.encode(Some("integration secret")).unwrap(), address)
        .await
        .unwrap();
}

async fn udp_exchange(
    socket: &UdpSocket,
    address: std::net::SocketAddr,
    packet: &Packet,
) -> Packet {
    udp_send(socket, address, packet).await;
    let mut wire = [0_u8; MAX_DATAGRAM];
    let (size, source) = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        socket.recv_from(&mut wire),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(source, address);
    Packet::decode(&wire[..size], Some("integration secret")).unwrap()
}

fn udp_start(handle: &MemoryBlockHandle, transfer: Uuid, payload: &[u8]) -> Packet {
    let mut packet = Packet::for_handle(Kind::Start, handle, transfer);
    packet.sequence_id = payload.len().div_ceil(1200) as u32;
    packet.payload = (payload.len() as u64).to_be_bytes().to_vec();
    packet
        .payload
        .extend_from_slice(blake3::hash(payload).as_bytes());
    packet
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn remote_payload_uses_reliable_udp_while_local_owner_uses_fast_path() {
    // 270 KiB admits the later 4 KiB local allocation (including relay
    // headroom), but not the 16 KiB object, which must fall back to remote RAM.
    let (a, ta) = start_with_limit(270_000).await;
    let (b, tb) = start().await;
    a.exchange(b.local().await.network.control_address)
        .await
        .unwrap();
    b.exchange(a.local().await.network.control_address)
        .await
        .unwrap();
    let address = a.local().await.network.control_address;
    let plan: TrainingPlan = a
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let payload: Vec<_> = (0..16_000).map(|index| (index % 251) as u8).collect();
    let remote: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Allocate {
                size: payload.len() as u64,
                job_id: plan.job_id,
                compute_node: a.node_id,
                preferred_node: Some(b.node_id),
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let transfer = Uuid::new_v4();
    for (index, chunk) in payload.chunks(4096).enumerate() {
        routed_write(&a, address, &remote, transfer, (index * 4096) as u64, chunk).await;
    }
    let committed: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Commit {
                handle: remote,
                checksum: blake3::hash(&payload).to_hex().to_string(),
                direct: false,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let mut restored = vec![];
    for offset in (0..payload.len()).step_by(4096) {
        restored.extend(
            routed_read(
                &a,
                address,
                &committed,
                offset as u64,
                4096.min(payload.len() - offset),
            )
            .await,
        );
    }
    assert_eq!(restored, payload);
    let metrics = a.metrics.lock().await;
    let job = &metrics.jobs[&plan.job_id];
    assert!(job.udp_datagrams_sent > 0);
    assert!(job.udp_datagrams_received > 0);
    assert!(job.udp_payload_bytes >= payload.len() as u64 * 2);
    assert_eq!(job.tcp_connections_opened, 0);
    drop(metrics);

    let local: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Allocate {
                size: 4096,
                job_id: plan.job_id,
                compute_node: a.node_id,
                preferred_node: Some(a.node_id),
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let before = a.metrics.lock().await.jobs[&plan.job_id].udp_datagrams_sent;
    routed_write(&a, address, &local, Uuid::new_v4(), 0, &[3; 4096]).await;
    let after = a.metrics.lock().await.jobs[&plan.job_id].udp_datagrams_sent;
    assert_eq!(
        before, after,
        "local owner must not use daemon-to-daemon UDP"
    );
    ta.abort();
    tb.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tcp_reference_reuses_authenticated_data_connection() {
    let (a, ta) = start_with_transport(270_000, 128, "tcp").await;
    let (b, tb) = start_with_transport(32 * 1024 * 1024, 128, "tcp").await;
    a.exchange(b.local().await.network.control_address)
        .await
        .unwrap();
    b.exchange(a.local().await.network.control_address)
        .await
        .unwrap();
    let address = a.local().await.network.control_address;
    let plan: TrainingPlan = a
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let payload = vec![31_u8; 16_000];
    let handle: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Allocate {
                size: payload.len() as u64,
                job_id: plan.job_id,
                compute_node: a.node_id,
                preferred_node: Some(b.node_id),
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(handle.owner_node, b.node_id);
    let transfer = Uuid::new_v4();
    for (index, chunk) in payload.chunks(4096).enumerate() {
        routed_write(&a, address, &handle, transfer, (index * 4096) as u64, chunk).await;
    }
    let committed: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Commit {
                handle,
                checksum: blake3::hash(&payload).to_hex().to_string(),
                direct: false,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let mut restored = Vec::new();
    for offset in (0..payload.len()).step_by(4096) {
        restored.extend(
            routed_read(
                &a,
                address,
                &committed,
                offset as u64,
                4096.min(payload.len() - offset),
            )
            .await,
        );
    }
    assert_eq!(restored, payload);
    let metrics = a.metrics.lock().await;
    let metrics = &metrics.jobs[&plan.job_id];
    assert_eq!(metrics.tcp_connections_opened, 1);
    assert!(metrics.tcp_connections_reused >= 7);
    assert_eq!(metrics.udp_payload_bytes, 0);
    ta.abort();
    tb.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn udp_receiver_reconstructs_reorder_rejects_stale_and_applies_backpressure() {
    let (runtime, task) = start_with_options(32 * 1024 * 1024, 1).await;
    let address = runtime.local().await.network.control_address;
    let plan: TrainingPlan = runtime
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let allocate = |size| Request::Allocate {
        size,
        job_id: plan.job_id,
        compute_node: runtime.node_id,
        preferred_node: None,
        tensor: None,
    };
    let handle: MemoryBlockHandle = runtime
        .transport
        .control(address, &allocate(4096))
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
    let payload: Vec<_> = (0..4096).map(|index| (index % 251) as u8).collect();
    let transfer = Uuid::new_v4();
    assert_eq!(
        udp_exchange(&socket, address, &udp_start(&handle, transfer, &payload))
            .await
            .kind,
        Kind::StartAck
    );

    let data = |sequence: usize| {
        let begin = sequence * 1200;
        let end = (begin + 1200).min(payload.len());
        let mut packet = Packet::for_handle(Kind::Data, &handle, transfer);
        packet.sequence_id = sequence as u32;
        packet.offset = begin as u64;
        packet.payload = payload[begin..end].to_vec();
        packet
    };
    // Intentionally deliver sequence 1 before 0, then duplicate 0.
    udp_send(&socket, address, &data(1)).await;
    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
    assert_eq!(
        udp_exchange(&socket, address, &data(0)).await.kind,
        Kind::Ack
    );
    assert_eq!(
        udp_exchange(&socket, address, &data(0)).await.kind,
        Kind::Ack
    );
    udp_send(&socket, address, &data(2)).await;
    assert_eq!(
        udp_exchange(&socket, address, &data(3)).await.kind,
        Kind::Ack
    );
    let mut fin = Packet::for_handle(Kind::Fin, &handle, transfer);
    fin.sequence_id = 4;
    fin.payload = blake3::hash(&payload).as_bytes().to_vec();
    assert_eq!(
        udp_exchange(&socket, address, &fin).await.kind,
        Kind::Complete
    );
    assert_eq!(
        udp_exchange(&socket, address, &data(0)).await.kind,
        Kind::Ack,
        "late retransmission after FIN must be idempotent"
    );
    let committed: MemoryBlockHandle = runtime
        .transport
        .control(
            address,
            &Request::Commit {
                handle,
                checksum: blake3::hash(&payload).to_hex().to_string(),
                direct: false,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(
        routed_read(&runtime, address, &committed, 0, payload.len()).await,
        payload
    );

    let mut stale = Packet::for_handle(Kind::Data, &committed, Uuid::new_v4());
    stale.payload = vec![1];
    assert_eq!(
        udp_exchange(&socket, address, &stale).await.kind,
        Kind::Error
    );

    let second: MemoryBlockHandle = runtime
        .transport
        .control(address, &allocate(4096))
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let third: MemoryBlockHandle = runtime
        .transport
        .control(address, &allocate(4096))
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let second_transfer = Uuid::new_v4();
    assert_eq!(
        udp_exchange(
            &socket,
            address,
            &udp_start(&second, second_transfer, &[2; 4096]),
        )
        .await
        .kind,
        Kind::StartAck
    );
    let mut incomplete_fin = Packet::for_handle(Kind::Fin, &second, second_transfer);
    incomplete_fin.payload = blake3::hash(&[2; 4096]).as_bytes().to_vec();
    assert_eq!(
        udp_exchange(&socket, address, &incomplete_fin).await.kind,
        Kind::Nack
    );
    let third_transfer = Uuid::new_v4();
    let backpressure = udp_exchange(
        &socket,
        address,
        &udp_start(&third, third_transfer, &[3; 4096]),
    )
    .await;
    assert_eq!(backpressure.kind, Kind::Error);
    assert!(String::from_utf8_lossy(&backpressure.payload).contains("BACKPRESSURE"));
    let abort = Packet::for_handle(Kind::Abort, &second, second_transfer);
    udp_send(&socket, address, &abort).await;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    assert_eq!(
        udp_exchange(
            &socket,
            address,
            &udp_start(&third, third_transfer, &[3; 4096]),
        )
        .await
        .kind,
        Kind::StartAck
    );
    udp_send(
        &socket,
        address,
        &Packet::for_handle(Kind::Abort, &third, third_transfer),
    )
    .await;
    tokio::time::sleep(std::time::Duration::from_millis(20)).await;
    let metrics = runtime.metrics.lock().await;
    let metrics = &metrics.jobs[&plan.job_id];
    assert!(metrics.out_of_order_datagrams >= 1);
    assert!(metrics.duplicate_datagrams >= 1);
    assert!(metrics.nack_count >= 1);
    assert_eq!(metrics.active_transfer_sessions, 0);
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn direct_transfer_migration_checksums_and_cpu_only_memory() {
    let (a, ta) = start().await;
    let (b, tb) = start().await;
    let (c, tc) = start_with_limit(65536).await;
    let runtimes = [a.clone(), b.clone(), c.clone()];
    for r in &runtimes {
        for other in &runtimes {
            if r.node_id != other.node_id {
                r.exchange(other.local().await.network.control_address)
                    .await
                    .unwrap();
            }
        }
    }
    let leader = a.leadership().await.leader_id.unwrap();
    for r in &runtimes {
        assert_eq!(r.leadership().await.leader_id, Some(leader));
    }
    let address = a.local().await.network.control_address;
    let plan: TrainingPlan = a
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    // Occupy local RAM while allocating the test tensor so strict locality
    // falls back to b. Release it before pressure migration so a becomes the
    // valid destination.
    let local_blocker: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Allocate {
                size: 32 * 1024 * 1024 - 270_000,
                job_id: plan.job_id,
                compute_node: a.node_id,
                preferred_node: Some(a.node_id),
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let bytes: Vec<u8> = (0..12345).map(|i| ((i * 17 + 3) % 251) as u8).collect();
    let handle: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Allocate {
                size: bytes.len() as u64,
                job_id: plan.job_id,
                compute_node: a.node_id,
                preferred_node: Some(b.node_id),
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(handle.owner_node, b.node_id);
    assert_eq!(b.ram.accounting.used(), bytes.len() as u64);
    let transfer = Uuid::new_v4();
    let owner = b.local().await.network.control_address;
    for (i, chunk) in bytes.chunks(4096).enumerate() {
        write_chunk(
            &a.transport,
            owner,
            &handle,
            transfer,
            (i * 4096) as u64,
            chunk,
        )
        .await
        .unwrap();
    }
    let committed: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Commit {
                handle: handle.clone(),
                checksum: blake3::hash(&bytes).to_hex().to_string(),
                direct: false,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let mut restored = vec![];
    for offset in (0..bytes.len()).step_by(4096) {
        restored.extend(
            read_chunk(
                &a.transport,
                owner,
                &committed,
                offset as u64,
                4096.min(bytes.len() - offset),
            )
            .await
            .unwrap(),
        );
    }
    assert_eq!(bytes, restored);
    let rejected = a
        .transport
        .control(
            owner,
            &Request::Migrate {
                handle: committed.clone(),
                destination: c.node_id,
                leadership: a.leadership().await,
            },
        )
        .await
        .unwrap();
    assert!(
        !rejected.ok,
        "destination without headroom must reject migration"
    );
    assert_eq!(b.ram.accounting.used(), bytes.len() as u64);
    assert_eq!(
        read_chunk(&a.transport, owner, &committed, 0, 4096)
            .await
            .unwrap(),
        bytes[..4096]
    );
    a.transport
        .control(
            address,
            &Request::Free {
                handle: local_blocker,
                direct: false,
            },
        )
        .await
        .unwrap()
        .check()
        .unwrap();
    // Wrong lease and writes to committed objects must be rejected.
    let mut wrong = committed.clone();
    wrong.lease_token = Uuid::new_v4();
    assert!(
        read_chunk(&a.transport, owner, &wrong, 0, 10)
            .await
            .is_err()
    );
    assert!(
        write_chunk(&a.transport, owner, &committed, transfer, 0, &bytes[..10])
            .await
            .is_err()
    );
    // Memory-pressure event causes leader -> source control, then source -> destination bytes.
    let result = a
        .transport
        .control(
            address,
            &Request::MemoryPressure {
                node_id: b.node_id,
                currently_owned: bytes.len() as u64,
                new_budget: 0,
                excess_bytes: bytes.len() as u64,
            },
        )
        .await
        .unwrap();
    result.check().unwrap();
    let migrated: MemoryBlockHandle = a
        .transport
        .control(
            address,
            &Request::Resolve {
                id: handle.id,
                lease_token: handle.lease_token,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    assert_ne!(migrated.owner_node, b.node_id);
    assert_eq!(migrated.generation, 1);
    assert_eq!(b.ram.accounting.used(), 0);
    // Old SDK handles resolve to the new owner without routing bytes through the leader.
    let mut stream = a.transport.connect(address).await.unwrap();
    write_frame(
        &mut stream,
        &Request::ReadChunk {
            handle: committed,
            transfer_id: Uuid::new_v4(),
            offset: 0,
            length: 4096,
            direct: false,
        },
    )
    .await
    .unwrap();
    read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .check()
        .unwrap();
    let mut chunk = vec![0; 4096];
    stream.read_exact(&mut chunk).await.unwrap();
    assert_eq!(chunk, bytes[..4096]);
    a.transport
        .control(
            address,
            &Request::Free {
                handle: migrated,
                direct: false,
            },
        )
        .await
        .unwrap()
        .check()
        .unwrap();
    for r in &runtimes {
        assert_eq!(r.ram.accounting.used(), 0);
    }
    ta.abort();
    tb.abort();
    tc.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn checksum_failure_does_not_commit_and_frame_limits_are_enforced() {
    let (r, task) = start().await;
    let address = r.local().await.network.control_address;
    let plan: TrainingPlan = r
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let h: MemoryBlockHandle = r
        .transport
        .control(
            address,
            &Request::Allocate {
                size: 4096,
                job_id: plan.job_id,
                compute_node: r.node_id,
                preferred_node: None,
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let mut stream = r.transport.connect(address).await.unwrap();
    write_frame(
        &mut stream,
        &Request::WriteChunk {
            handle: h.clone(),
            transfer_id: Uuid::new_v4(),
            offset: 0,
            length: 4096,
            total_size: 4096,
            checksum: "bad".into(),
            direct: true,
        },
    )
    .await
    .unwrap();
    read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .check()
        .unwrap();
    stream.write_all(&vec![7; 4096]).await.unwrap();
    assert!(!read_frame::<Response>(&mut stream).await.unwrap().ok);
    let response = r
        .transport
        .control(
            address,
            &Request::Commit {
                handle: h,
                checksum: "bad".into(),
                direct: true,
            },
        )
        .await
        .unwrap();
    assert!(!response.ok);
    let mut stream = r.transport.connect(address).await.unwrap();
    stream.write_u32(u32::MAX).await.unwrap();
    assert!(!read_frame::<Response>(&mut stream).await.unwrap().ok);
    let mut wrong = r.transport.clone();
    wrong.config.cluster_secret = Some("wrong secret".into());
    assert!(wrong.connect(address).await.is_err());
    task.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn interrupted_upload_cannot_commit_and_job_failure_is_explicit() {
    let (r, task) = start().await;
    let address = r.local().await.network.control_address;
    let plan: TrainingPlan = r
        .transport
        .control(address, &Request::Plan { stages: vec![] })
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let handle: MemoryBlockHandle = r
        .transport
        .control(
            address,
            &Request::Allocate {
                size: 4096,
                job_id: plan.job_id,
                compute_node: r.node_id,
                preferred_node: None,
                tensor: None,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    let payload = vec![19; 4096];
    let checksum = blake3::hash(&payload).to_hex().to_string();
    let mut stream = r.transport.connect(address).await.unwrap();
    write_frame(
        &mut stream,
        &Request::WriteChunk {
            handle: handle.clone(),
            transfer_id: Uuid::new_v4(),
            offset: 0,
            length: 4096,
            total_size: 4096,
            checksum: checksum.clone(),
            direct: true,
        },
    )
    .await
    .unwrap();
    read_frame::<Response>(&mut stream)
        .await
        .unwrap()
        .check()
        .unwrap();
    stream.write_all(&payload[..128]).await.unwrap();
    drop(stream);
    let result = r
        .transport
        .control(
            address,
            &Request::Commit {
                handle: handle.clone(),
                checksum: checksum.clone(),
                direct: true,
            },
        )
        .await
        .unwrap();
    assert!(!result.ok, "partial transfer must never become visible");
    assert!(
        read_chunk(&r.transport, address, &handle, 0, 4096)
            .await
            .is_err()
    );
    // A complete retransmission can safely overwrite an uncommitted partial chunk.
    write_chunk(&r.transport, address, &handle, Uuid::new_v4(), 0, &payload)
        .await
        .unwrap();
    let committed: MemoryBlockHandle = r
        .transport
        .control(
            address,
            &Request::Commit {
                handle,
                checksum,
                direct: true,
            },
        )
        .await
        .unwrap()
        .into_data()
        .unwrap();
    assert_eq!(
        read_chunk(&r.transport, address, &committed, 0, 4096)
            .await
            .unwrap(),
        payload
    );
    r.jobs.lock().await.fail(
        plan.job_id,
        "TRAINPOOL_DATA_LOST: injected stress fault".into(),
    );
    let failed = r
        .transport
        .control(
            address,
            &Request::JobStatus {
                job_id: plan.job_id,
            },
        )
        .await
        .unwrap();
    assert!(!failed.ok);
    assert!(failed.error.unwrap().contains("TRAINPOOL_DATA_LOST"));
    let allocation = r
        .transport
        .control(
            address,
            &Request::Allocate {
                size: 1,
                job_id: plan.job_id,
                compute_node: r.node_id,
                preferred_node: None,
                tensor: None,
            },
        )
        .await
        .unwrap();
    assert!(!allocation.ok);
    r.ram
        .free(committed.id, committed.lease_token)
        .await
        .unwrap();
    task.abort();
}
