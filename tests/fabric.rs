use std::sync::Arc;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpListener,
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
    transport::{Transport, read_frame, write_frame},
};
use uuid::Uuid;

async fn start() -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    start_with_limit(32 * 1024 * 1024).await
}
async fn start_with_limit(limit: u64) -> (Arc<Runtime>, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let config = Config {
        listen: listener.local_addr().unwrap(),
        discovery_enabled: false,
        ram_limit_bytes: Some(limit),
        chunk_bytes: 4096,
        cluster_secret: Some("integration secret".into()),
        ..Default::default()
    };
    let runtime = Runtime::new(config, Uuid::new_v4()).await.unwrap();
    let r = runtime.clone();
    let task = tokio::spawn(async move {
        r.serve(listener).await.unwrap();
    });
    (runtime, task)
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
