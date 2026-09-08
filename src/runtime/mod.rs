mod control;
mod data;
mod monitor;

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
        ClusterStatus {
            local_node: self.node_id,
            leadership: membership.leader(),
            logical_memory: LogicalTrainingMemory::from_nodes(&nodes),
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
        loop {
            let (stream, peer) = listener.accept().await?;
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
    }
    async fn connection(&self, mut stream: TcpStream) -> Result<()> {
        stream.set_nodelay(true)?;
        if let Err(error) = server_auth(&mut stream, &self.config).await {
            let _ = write_frame(&mut stream, &Response::error(&error)).await;
            return Err(error);
        }
        let request = match read_frame::<Request>(&mut stream).await {
            Ok(r) => r,
            Err(e) => {
                write_frame(&mut stream, &Response::error(&e)).await?;
                return Err(e);
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
        Ok(())
    }
}

pub async fn daemon(config: Config, id: Uuid) -> Result<()> {
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
    tokio::select! {
        result = runtime.clone().serve(listener) => { result?; },
        result = tokio::signal::ctrl_c() => { result?; },
    }
    monitor.abort();
    if let Some(r) = receiver {
        r.abort();
    }
    Ok(())
}
