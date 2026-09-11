mod control;
mod data;
mod monitor;
mod udp;

use crate::{
    cluster::{
        discovery::{MulticastDiscovery, advertised_address},
        election::Leadership,
        membership::Membership,
    },
    config::Config,
    jobs::Jobs,
    memory::{local_ram::LocalRam, residency::ResidencyTable},
    metrics::Metrics,
    node::NodeCapabilities,
    protocol::{ClusterStatus, LogicalTrainingMemory, Request, Response},
    scheduler::topology::Topology,
    transport::{TcpTransport, Transport, read_frame, server_auth, write_frame},
};
use anyhow::{Result, ensure};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{
    net::{TcpListener, TcpStream},
    sync::{Mutex, RwLock, Semaphore},
    task::JoinHandle,
};
use uuid::Uuid;

pub struct Runtime {
    pub config: Config,
    pub node_id: Uuid,
    pub membership: RwLock<Membership>,
    pub ram: LocalRam,
    pub residency: RwLock<ResidencyTable>,
    pub topology: RwLock<Topology>,
    pub jobs: Mutex<Jobs>,
    pub metrics: Mutex<Metrics>,
    pub transport: TcpTransport,
    pub system: std::sync::Mutex<sysinfo::System>,
    pub migration_gate: Mutex<()>,
    pub active_transfers: std::sync::atomic::AtomicU32,
    pub staging:
        Mutex<std::collections::BTreeMap<Uuid, (crate::memory::allocator::Reservation, u64)>>,
    pub udp_server: udp::UdpServerState,
    pub tcp_data_pool: Mutex<std::collections::HashMap<SocketAddr, Vec<TcpStream>>>,
}
impl Runtime {
    pub async fn new(config: Config, node_id: Uuid) -> Result<Arc<Self>> {
        config.validate()?;
        let address = advertised_address(&config)?;
        let local = NodeCapabilities::detect(node_id, address, &config).await;
        let ram = LocalRam::default();
        ram.accounting.budget.store(
            local.memory.trainpool_ram_budget,
            std::sync::atomic::Ordering::SeqCst,
        );
        Ok(Arc::new(Self {
            transport: TcpTransport {
                config: config.clone(),
            },
            config,
            node_id,
            membership: RwLock::new(Membership::new(local)),
            ram,
            residency: RwLock::new(ResidencyTable::default()),
            topology: RwLock::new(Topology::default()),
            jobs: Mutex::new(Jobs::default()),
            metrics: Mutex::new(Metrics::default()),
            system: std::sync::Mutex::new(sysinfo::System::new()),
            migration_gate: Mutex::new(()),
            active_transfers: std::sync::atomic::AtomicU32::new(0),
            staging: Mutex::new(std::collections::BTreeMap::new()),
            udp_server: udp::UdpServerState::default(),
            tcp_data_pool: Mutex::new(std::collections::HashMap::new()),
        }))
    }
    pub async fn local(&self) -> NodeCapabilities {
        self.membership.read().await.peers[&self.node_id]
            .capabilities
            .clone()
    }
    pub async fn leadership(&self) -> Leadership {
        self.membership.read().await.leader()
    }
    pub async fn address(&self, node: Uuid) -> Result<SocketAddr> {
        self.membership
            .read()
            .await
            .peers
            .get(&node)
            .map(|p| p.capabilities.network.control_address)
            .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_NODE_UNAVAILABLE: {node}"))
    }
    pub async fn leader_address(&self) -> Result<SocketAddr> {
        self.address(
            self.leadership()
                .await
                .leader_id
                .ok_or_else(|| anyhow::anyhow!("no leader"))?,
        )
        .await
    }
    pub async fn validate_leadership(&self, expected: &Leadership) -> Result<()> {
        ensure!(
            &self.leadership().await == expected,
            "TRAINPOOL_STALE_LEADER: membership/epoch changed"
        );
        Ok(())
    }
    pub async fn status(&self) -> ClusterStatus {
        self.refresh_memory().await;
        let membership = self.membership.read().await;
        let nodes = membership.nodes();
        let mut logical_memory = LogicalTrainingMemory::from_nodes(&nodes);
        if let Some(local) = nodes.iter().find(|node| node.node_id == self.node_id) {
            logical_memory.local_physical_ram = local.memory.physical_ram_total;
            logical_memory.local_os_available_ram = local.memory.os_available_ram;
            logical_memory.local_pool_ram_budget = local.memory.trainpool_ram_budget;
            logical_memory.local_pool_ram_used = local.memory.trainpool_ram_used;
            logical_memory.local_pool_ram_allocatable = local.memory.trainpool_ram_available;
            logical_memory.local_ram_safety_reserve = local.memory.safety_reserve;
        }
        logical_memory.remote_pool_ram_used = nodes
            .iter()
            .filter(|node| node.node_id != self.node_id)
            .map(|node| node.memory.trainpool_ram_used)
            .sum();
        let metrics = self.metrics.lock().await;
        if let Some(latest) = metrics
            .jobs
            .values()
            .filter(|job| job.sdk_metrics_timestamp_ms.is_some())
            .max_by_key(|job| job.sdk_metrics_timestamp_ms)
        {
            logical_memory.sdk_metrics_timestamp_ms = latest.sdk_metrics_timestamp_ms;
            logical_memory.sdk_metrics_age_ms = latest
                .sdk_metrics_timestamp_ms
                .map(|timestamp| crate::now_ms().saturating_sub(timestamp));
            logical_memory.driver_used_vram_bytes = latest.driver_used_vram_bytes;
            logical_memory.sdk_physical_vram_bytes = latest.physical_vram_bytes;
            logical_memory.driver_free_vram_bytes = latest.driver_free_vram_bytes;
            logical_memory.torch_allocated_bytes = latest.torch_allocated_bytes;
            logical_memory.torch_reserved_bytes = latest.torch_reserved_bytes;
            logical_memory.torch_reclaimable_bytes = latest.torch_reclaimable_bytes;
            logical_memory.trainpool_resident_vram_bytes = Some(latest.current_vram_resident_bytes);
            logical_memory.safe_vram_allocatable_bytes = latest.safe_vram_allocatable_bytes;
            logical_memory.configured_usable_vram_ceiling_bytes =
                latest.configured_usable_vram_ceiling_bytes;
            logical_memory.configured_vram_safety_reserve_bytes =
                latest.configured_vram_safety_reserve_bytes;
        }
        drop(metrics);
        ClusterStatus {
            local_node: self.node_id,
            leadership: membership.leader(),
            logical_memory,
            nodes,
            topology: self.topology.read().await.clone(),
            chunk_bytes: self.config.chunk_bytes,
            lease_seconds: self.config.lease_seconds,
        }
    }
    pub async fn forward_leader(&self, request: &Request) -> Result<Response> {
        self.transport
            .control(self.leader_address().await?, request)
            .await
    }
    pub async fn serve(self: Arc<Self>, listener: TcpListener) -> Result<()> {
        let permits = Arc::new(Semaphore::new(64));
        let udp_permits = Arc::new(Semaphore::new(256));
        let udp = Arc::new(tokio::net::UdpSocket::bind(listener.local_addr()?).await?);
        loop {
            tokio::select! {
                accepted = listener.accept() => {
                    let (stream, peer) = accepted?;
                    let Ok(permit) = permits.clone().try_acquire_owned() else {
                        drop(stream);
                        continue;
                    };
                    let this = self.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        match tokio::time::timeout(Duration::from_secs(120), this.connection(stream)).await
                        {
                            Ok(Ok(())) => {}
                            Ok(Err(error)) => tracing::debug!(%peer, %error, "request failed"),
                            Err(_) => tracing::warn!(%peer, "connection deadline exceeded"),
                        }
                    });
                }
                received = async {
                    let mut buffer = vec![0_u8; crate::transport::udp::MAX_DATAGRAM];
                    let (size, source) = udp.recv_from(&mut buffer).await?;
                    buffer.truncate(size);
                    Ok::<_, std::io::Error>((buffer, source))
                } => {
                    let (datagram, source) = received?;
                    // Concurrency hides per-datagram integrity and metrics costs,
                    // while the semaphore provides a hard bound on queued tasks
                    // and naturally backpressures the kernel receive queue.
                    let permit = udp_permits.clone().acquire_owned().await?;
                    let this = self.clone();
                    let socket = udp.clone();
                    tokio::spawn(async move {
                        let _permit = permit;
                        this.udp_datagram(socket, datagram, source).await;
                    });
                }
            }
        }
    }
    async fn connection(&self, mut stream: TcpStream) -> Result<()> {
        stream.set_nodelay(true)?;
        if let Err(error) = server_auth(&mut stream, &self.config).await {
            let _ = write_frame(&mut stream, &Response::error(&error)).await;
            return Err(error);
        }
        loop {
            let request = match read_frame::<Request>(&mut stream).await {
                Ok(request) => request,
                Err(error)
                    if error.downcast_ref::<std::io::Error>().is_some_and(|io| {
                        matches!(
                            io.kind(),
                            std::io::ErrorKind::UnexpectedEof
                                | std::io::ErrorKind::ConnectionReset
                                | std::io::ErrorKind::BrokenPipe
                        )
                    }) =>
                {
                    return Ok(());
                }
                Err(error) => {
                    write_frame(&mut stream, &Response::error(&error)).await?;
                    return Err(error);
                }
            };
            if matches!(
                request,
                Request::WriteChunk { .. } | Request::ReadChunk { .. } | Request::Probe { .. }
            ) {
                if let Err(error) = self.data(&mut stream, request).await {
                    let _ = write_frame(&mut stream, &Response::error(&error)).await;
                    return Err(error);
                }
            } else {
                let response = self.control(request).await.unwrap_or_else(Response::error);
                write_frame(&mut stream, &response).await?;
            }
        }
    }
}

/// A running local TrainPool node. Dropping the handle stops all background work.
/// This is shared by the foreground daemon command and the transparent launcher.
pub struct RuntimeHandle {
    pub runtime: Arc<Runtime>,
    server: JoinHandle<Result<()>>,
    monitor: JoinHandle<()>,
    receiver: Option<JoinHandle<()>>,
}

impl RuntimeHandle {
    pub async fn shutdown(mut self) {
        self.server.abort();
        self.monitor.abort();
        if let Some(receiver) = &mut self.receiver {
            receiver.abort();
        }
        let _ = (&mut self.server).await;
        let _ = (&mut self.monitor).await;
        if let Some(receiver) = &mut self.receiver {
            let _ = receiver.await;
        }
    }
}

impl Drop for RuntimeHandle {
    fn drop(&mut self) {
        self.server.abort();
        self.monitor.abort();
        if let Some(receiver) = &self.receiver {
            receiver.abort();
        }
    }
}

pub async fn start_runtime(config: Config, id: Uuid) -> Result<RuntimeHandle> {
    let listener = TcpListener::bind(config.listen).await?;
    let runtime = Runtime::new(config.clone(), id).await?;
    let discovery = if config.discovery_enabled {
        match MulticastDiscovery::bind(&config) {
            Ok(d) => Some(Arc::new(d)),
            Err(error) => {
                tracing::warn!(%error, "multicast unavailable; using seed/known peers");
                None
            }
        }
    } else {
        None
    };
    tracing::info!(node_id = %id, address = %runtime.local().await.network.control_address, gpu_compute = runtime.local().await.runtime.gpu_compute, "TrainPool daemon ready; disk spill disabled");
    let monitor = tokio::spawn(runtime.clone().monitor(discovery.clone()));
    let receiver = discovery.map(|d| tokio::spawn(runtime.clone().discover(d)));
    let server = tokio::spawn(runtime.clone().serve(listener));
    Ok(RuntimeHandle {
        runtime,
        server,
        monitor,
        receiver,
    })
}

pub async fn daemon(config: Config, id: Uuid) -> Result<()> {
    let mut handle = start_runtime(config, id).await?;
    tokio::select! {
        result = &mut handle.server => { result??; },
        result = tokio::signal::ctrl_c() => { result?; },
    }
    Ok(())
}
