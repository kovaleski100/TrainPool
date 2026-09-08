use super::Runtime;
use crate::{
    cluster::{
        discovery::{Announcement, Discovery, MulticastDiscovery},
        election::Leadership,
    },
    memory::block::{BlockState, MemoryBlockHandle},
    node::NodeCapabilities,
    protocol::{Request, Response},
    scheduler::{
        placement::{CapacityPlacement, PlacementPolicy},
        topology::{LinkEstimate, Topology},
    },
    transport::{Transport, read_frame, write_frame},
};
use anyhow::{Result, ensure};
use std::{
    collections::BTreeSet,
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use tokio::io::AsyncReadExt;
use uuid::Uuid;

impl Runtime {
    pub async fn refresh_memory(&self) {
        let mut membership = self.membership.write().await;
        let local = &mut membership
            .peers
            .get_mut(&self.node_id)
            .unwrap()
            .capabilities;
        local.network.active_transfers = self.active_transfers.load(Ordering::Relaxed);
        local.refresh_memory(
            &mut self.system.lock().expect("system monitor lock"),
            &self.config,
            self.ram.accounting.used(),
        );
        self.ram
            .accounting
            .budget
            .store(local.memory.trainpool_ram_budget, Ordering::SeqCst);
    }
    pub async fn accept_peer(
        &self,
        node: NodeCapabilities,
        inventory: Vec<MemoryBlockHandle>,
    ) -> Result<()> {
        ensure!(
            node.node_id != self.node_id,
            "duplicate node_id: use distinct data directories per installation"
        );
        ensure!(
            node.network.chunk_bytes >= 4096 && node.network.chunk_bytes <= 64 * 1024 * 1024,
            "invalid peer chunk size"
        );
        ensure!(inventory.len() <= 100_000, "peer inventory exceeds limit");
        let mut members = self.membership.write().await;
        let before = members.leader();
        let restarted = members
            .peers
            .get(&node.node_id)
            .is_some_and(|p| p.capabilities.incarnation != node.incarnation);
        if restarted {
            self.jobs.lock().await.lose_compute(node.node_id);
            let jobs = self.residency.write().await.lose_node(node.node_id);
            for job in jobs {
                self.jobs.lock().await.fail(
                    job,
                    format!("TRAINPOOL_DATA_LOST: node {} restarted", node.node_id),
                );
            }
        }
        for handle in inventory {
            ensure!(
                handle.owner_node == node.node_id && handle.owner_incarnation == node.incarnation,
                "invalid inventory owner"
            );
            self.residency.write().await.register(handle);
        }
        members.update(node);
        self.rotate_leader(&mut members, before);
        Ok(())
    }
    fn rotate_leader(
        &self,
        members: &mut crate::cluster::membership::Membership,
        before: Leadership,
    ) {
        let after = members.leader();
        if after.leader_id != before.leader_id {
            if after.leader_id == Some(self.node_id) {
                members
                    .peers
                    .get_mut(&self.node_id)
                    .unwrap()
                    .capabilities
                    .leader_generation = Uuid::new_v4();
            }
            tracing::info!(leader_id = ?members.leader().leader_id, epoch = %members.leader().leader_epoch, "leader changed");
        }
    }
    pub async fn exchange(&self, address: std::net::SocketAddr) -> Result<()> {
        let started = Instant::now();
        let response: serde_json::Value = self
            .transport
            .control(
                address,
                &Request::Exchange {
                    node: self.local().await,
                    inventory: self.ram.inventory_page(crate::now_ms() / 2000).await,
                },
            )
            .await?
            .into_data()?;
        let node: NodeCapabilities = serde_json::from_value(response["node"].clone())?;
        let id = node.node_id;
        let inventory = serde_json::from_value(response["inventory"].clone())?;
        self.accept_peer(node, inventory).await?;
        self.topology.write().await.update(LinkEstimate {
            source: self.node_id,
            destination: id,
            latency_ms: started.elapsed().as_secs_f64() * 1000.0,
            bytes_per_second: None,
            sampled_at_ms: crate::now_ms(),
            active_transfers: 0,
        });
        Ok(())
    }
    pub async fn discover(self: Arc<Self>, discovery: Arc<MulticastDiscovery>) {
        loop {
            match discovery.receive().await {
                Ok(a) if a.node_id != self.node_id => {
                    let address = std::net::SocketAddr::new(a.control_address, a.control_port);
                    if let Err(error) =
                        tokio::time::timeout(Duration::from_secs(2), self.exchange(address))
                            .await
                            .unwrap_or_else(|e| Err(e.into()))
                    {
                        tracing::debug!(%error, %address, "discovered peer handshake failed");
                    }
                }
                Ok(_) => {}
                Err(error) => tracing::debug!(%error, "ignored discovery packet"),
            }
        }
    }
    pub async fn monitor(self: Arc<Self>, discovery: Option<Arc<MulticastDiscovery>>) {
        let mut interval = tokio::time::interval(Duration::from_secs(2));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut cycle = 0_u64;
        loop {
            interval.tick().await;
            cycle += 1;
            self.staging
                .lock()
                .await
                .retain(|_, (_, deadline)| *deadline > crate::now_ms());
            self.ram.expire().await;
            self.refresh_memory().await;
            {
                let mut members = self.membership.write().await;
                let before = members.leader();
                for id in members.expire(Instant::now()) {
                    self.jobs.lock().await.lose_compute(id);
                    tracing::warn!(node_id = %id, "peer unavailable; single-copy blocks lost");
                    let affected = self.residency.write().await.lose_node(id);
                    for job in affected {
                        self.jobs.lock().await.fail(
                            job,
                            format!("TRAINPOOL_DATA_LOST: data stored exclusively on node {id}"),
                        );
                    }
                }
                self.rotate_leader(&mut members, before);
            }
            let local = self.local().await;
            if let Some(d) = &discovery
                && let Err(error) = d.advertise(&Announcement::new(&local, &self.config)).await
            {
                tracing::warn!(%error, "advertisement failed");
            }
            let mut addresses: BTreeSet<_> = self.config.seeds.iter().copied().collect();
            for n in self
                .membership
                .read()
                .await
                .nodes()
                .iter()
                .filter(|n| n.node_id != self.node_id)
            {
                addresses.insert(n.network.control_address);
            }
            let mut exchanges = tokio::task::JoinSet::new();
            for address in addresses.into_iter().take(64) {
                let this = self.clone();
                exchanges.spawn(async move {
                    let _ =
                        tokio::time::timeout(Duration::from_millis(1500), this.exchange(address))
                            .await;
                });
            }
            while exchanges.join_next().await.is_some() {}
            if local.memory.excess() > 0 {
                tracing::warn!(node_id = %self.node_id, owned = local.memory.trainpool_ram_used, budget = local.memory.trainpool_ram_budget, excess = local.memory.excess(), "MemoryPressure");
                let affected: BTreeSet<_> = self
                    .ram
                    .inventory()
                    .await
                    .iter()
                    .map(|b| b.job_id)
                    .collect();
                for job in affected {
                    self.metrics.lock().await.job(job).ram_pressure_events += 1;
                }
                let this = self.clone();
                tokio::spawn(async move {
                    if let Err(error) = this
                        .forward_leader(&Request::MemoryPressure {
                            node_id: this.node_id,
                            currently_owned: local.memory.trainpool_ram_used,
                            new_budget: local.memory.trainpool_ram_budget,
                            excess_bytes: local.memory.excess(),
                        })
                        .await
                    {
                        tracing::warn!(%error, "pressure notification failed; retaining data");
                    }
                });
            }
            if cycle.is_multiple_of(5) {
                let gpus = crate::node::gpu::detect(&self.config).await;
                let mut members = self.membership.write().await;
                let local = &mut members.peers.get_mut(&self.node_id).unwrap().capabilities;
                // Static inventory/score never changes due to transient driver query failures.
                for old in &mut local.gpus {
                    if let Some(new) = gpus.iter().find(|g| g.uuid == old.uuid) {
                        let allocated = old.trainpool_allocated_vram;
                        *old = new.clone();
                        old.trainpool_allocated_vram = allocated;
                    }
                }
            }
        }
    }
    pub async fn relieve_pressure(&self, source: Uuid, excess: u64) -> Result<u64> {
        let Ok(_gate) = self.migration_gate.try_lock() else {
            return Ok(0);
        };
        let leadership = self.leadership().await;
        let mut moved = 0;
        let mut blocks: Vec<_> = self
            .residency
            .read()
            .await
            .blocks
            .values()
            .filter(|b| {
                b.owner_node == source
                    && b.state == BlockState::Ready
                    && b.lease_expires_ms > crate::now_ms()
            })
            .cloned()
            .collect();
        blocks.sort_by_key(|b| (b.size, b.id));
        for block in blocks {
            if moved >= excess {
                break;
            }
            let nodes: Vec<_> = self
                .membership
                .read()
                .await
                .nodes()
                .into_iter()
                .filter(|n| n.node_id != source)
                .collect();
            let ranked = CapacityPlacement.rank_ram(
                &nodes,
                source,
                block.size,
                &*self.topology.read().await,
            );
            for target in ranked {
                let result = self
                    .transport
                    .control(
                        self.address(source).await?,
                        &Request::Migrate {
                            handle: block.clone(),
                            destination: target.node_id,
                            leadership: leadership.clone(),
                        },
                    )
                    .await;
                match result.and_then(|r| r.check()) {
                    Ok(()) => {
                        moved += block.size;
                        break;
                    }
                    Err(error) => {
                        tracing::warn!(%error, block_id = %block.id, "migration failed; retaining source copy")
                    }
                }
            }
        }
        if moved < excess {
            tracing::warn!(
                unresolved_bytes = excess - moved,
                "insufficient remote capacity; rejecting new allocations, retaining existing data"
            );
        }
        Ok(moved)
    }
    pub async fn measure(&self, destination: Uuid, bytes: usize) -> Result<LinkEstimate> {
        ensure!(
            bytes <= 64 * 1024 * 1024,
            "benchmark capped at 64 MiB per directed link"
        );
        ensure!(destination != self.node_id, "cannot benchmark self");
        let address = self.address(destination).await?;
        let start = Instant::now();
        self.transport
            .control(address, &Request::Ping)
            .await?
            .check()?;
        let latency = start.elapsed().as_secs_f64() * 1000.0;
        let mut stream = self.transport.connect(address).await?;
        write_frame(&mut stream, &Request::Probe { bytes }).await?;
        read_frame::<Response>(&mut stream).await?.check()?;
        let start = Instant::now();
        let mut remaining = bytes;
        let mut buffer = [0_u8; 8192];
        while remaining > 0 {
            let n = remaining.min(buffer.len());
            stream.read_exact(&mut buffer[..n]).await?;
            remaining -= n;
        }
        let estimate = LinkEstimate {
            source: self.node_id,
            destination,
            latency_ms: latency,
            bytes_per_second: if bytes > 0 {
                Some(bytes as f64 / start.elapsed().as_secs_f64().max(0.000001))
            } else {
                None
            },
            sampled_at_ms: crate::now_ms(),
            active_transfers: 0,
        };
        self.topology.write().await.update(estimate.clone());
        Ok(estimate)
    }
    pub async fn benchmark(&self, bytes: usize) -> Result<Topology> {
        ensure!(
            bytes <= 64 * 1024 * 1024,
            "benchmark capped at 64 MiB per directed link"
        );
        let nodes = self.membership.read().await.nodes();
        for source in &nodes {
            for target in &nodes {
                if source.node_id == target.node_id {
                    continue;
                }
                let sample: LinkEstimate = self
                    .transport
                    .control(
                        source.network.control_address,
                        &Request::Measure {
                            destination: target.node_id,
                            bytes,
                        },
                    )
                    .await?
                    .into_data()?;
                self.topology.write().await.update(sample);
            }
        }
        Ok(self.topology.read().await.clone())
    }
}
