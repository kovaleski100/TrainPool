use super::Runtime;
use crate::{
    jobs::Job,
    memory::block::{BlockState, Location, MemoryBlockHandle},
    protocol::{Request, Response},
    scheduler::{
        placement::{CapacityPlacement, PlacementPolicy},
        planner::{CapacityPlanner, TrainingPlanner},
    },
    transport::Transport,
};
use anyhow::{Result, ensure};
use uuid::Uuid;

impl Runtime {
    pub async fn resolve(&self, handle: &MemoryBlockHandle) -> Result<MemoryBlockHandle> {
        self.forward_leader(&Request::Resolve {
            id: handle.id,
            lease_token: handle.lease_token,
        })
        .await?
        .into_data()
    }
    pub async fn control(&self, request: Request) -> Result<Response> {
        // Only metadata operations are forwarded to the leader. Payloads use data().
        let leader_only = matches!(
            request,
            Request::Plan { .. }
                | Request::JobStatus { .. }
                | Request::Allocate { .. }
                | Request::Resolve { .. }
                | Request::Register { .. }
                | Request::MemoryPressure { .. }
                | Request::Benchmark { .. }
                | Request::Topology
        );
        if leader_only && self.leadership().await.leader_id != Some(self.node_id) {
            return self.forward_leader(&request).await;
        }
        match request {
            Request::Ping => Ok(Response::data(crate::now_ms())),
            Request::ReserveStaging { bytes } => {
                ensure!(
                    bytes <= self.config.chunk_bytes as u64,
                    "SDK staging reservation exceeds chunk size"
                );
                self.refresh_memory().await;
                let reserved = self.ram.accounting.reserve(bytes)?;
                let id = Uuid::new_v4();
                self.staging.lock().await.insert(
                    id,
                    (reserved, crate::now_ms() + self.config.lease_seconds * 1000),
                );
                Ok(Response::data(id))
            }
            Request::ReleaseStaging { reservation_id } => {
                self.staging.lock().await.remove(&reservation_id);
                Ok(Response::data(()))
            }
            Request::Status => Ok(Response::data(self.status().await)),
            Request::Capabilities => Ok(Response::data(self.local().await)),
            Request::Exchange { node, inventory } => {
                self.accept_peer(node, inventory).await?;
                Ok(Response::data(
                    serde_json::json!({ "node": self.local().await, "inventory": self.ram.inventory_page(crate::now_ms() / 2000).await }),
                ))
            }
            Request::Inventory { page } => Ok(Response::data(self.ram.inventory_page(page).await)),
            Request::Metrics => {
                let mut all = self.metrics.lock().await;
                for (id, accounting) in self
                    .ram
                    .accounting
                    .jobs
                    .lock()
                    .expect("RAM accounting lock")
                    .iter()
                {
                    let m = all.job(*id);
                    m.peak_ram_residency =
                        accounting.peak.load(std::sync::atomic::Ordering::Relaxed);
                    m.current_ram_residency =
                        accounting.owned.load(std::sync::atomic::Ordering::SeqCst);
                }
                Ok(Response::data(
                    serde_json::json!({ "node_id": self.node_id, "jobs": all.jobs,
                    "owned_ram": self.ram.accounting.used(), "peak_ram": self.ram.accounting.peak.load(std::sync::atomic::Ordering::Relaxed) }),
                ))
            }
            Request::ReportMetrics { job_id, metrics } => {
                let mut all = self.metrics.lock().await;
                let m = all.job(job_id);
                m.bytes_local_ram_to_gpu += metrics.bytes_local_ram_to_gpu;
                m.bytes_gpu_to_local_ram += metrics.bytes_gpu_to_local_ram;
                m.prefetch_hits += metrics.prefetch_hits;
                m.prefetch_misses += metrics.prefetch_misses;
                m.gpu_wait_for_data_ms += metrics.gpu_wait_for_data_ms;
                m.peak_gpu_residency = m.peak_gpu_residency.max(metrics.peak_gpu_residency);
                m.current_gpu_residency = metrics.current_gpu_residency;
                m.gpu_id = metrics.gpu_id.clone();
                drop(all);
                if let Some(id) = metrics.gpu_id {
                    let mut members = self.membership.write().await;
                    if let Some(gpu) = members
                        .peers
                        .get_mut(&self.node_id)
                        .unwrap()
                        .capabilities
                        .gpus
                        .iter_mut()
                        .find(|g| g.uuid == id)
                    {
                        gpu.trainpool_allocated_vram = metrics.current_gpu_residency;
                    }
                }
                Ok(Response::data(()))
            }
            Request::Plan { stages } => {
                ensure!(stages.len() <= 4096, "too many stages");
                let membership = self.membership.read().await;
                let topology = self.topology.read().await;
                let plan = CapacityPlanner.plan(
                    &membership.nodes(),
                    &membership.leader(),
                    &topology,
                    &stages,
                )?;
                drop(topology);
                drop(membership);
                self.jobs.lock().await.entries.insert(
                    plan.job_id,
                    Job {
                        plan: plan.clone(),
                        error: None,
                    },
                );
                Ok(Response::data(plan))
            }
            Request::JobStatus { job_id } => {
                let epoch = self.leadership().await.leader_epoch;
                let jobs = self.jobs.lock().await;
                let job = jobs.entries.get(&job_id).ok_or_else(|| {
                    anyhow::anyhow!("TRAINPOOL_JOB_LOST: unknown job; leadership may have changed")
                })?;
                ensure!(
                    job.plan.leader_epoch == epoch,
                    "TRAINPOOL_STALE_LEADER: recreate job after leadership change"
                );
                if let Some(error) = &job.error {
                    anyhow::bail!("{error}");
                }
                Ok(Response::data(job))
            }
            Request::Allocate {
                size,
                job_id,
                compute_node,
                preferred_node,
                tensor,
            } => {
                if let Some(meta) = &tensor {
                    meta.validate(size)?;
                }
                ensure!(
                    size > 0,
                    "zero-length blocks are represented locally by the SDK"
                );
                let epoch = self.leadership().await.leader_epoch;
                {
                    let jobs = self.jobs.lock().await;
                    let job = jobs
                        .entries
                        .get(&job_id)
                        .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_JOB_LOST"))?;
                    ensure!(
                        job.error.is_none(),
                        "{}",
                        job.error.as_deref().unwrap_or("")
                    );
                    ensure!(job.plan.leader_epoch == epoch, "TRAINPOOL_STALE_LEADER");
                }
                self.refresh_memory().await;
                let nodes = self.membership.read().await.nodes();
                let candidates = CapacityPlacement.rank_ram(
                    &nodes,
                    compute_node,
                    size,
                    &*self.topology.read().await,
                );
                let leadership = self.leadership().await;
                let mut last_error = "no eligible memory nodes".to_owned();
                for node in candidates
                    .into_iter()
                    .filter(|n| preferred_node.is_none_or(|id| n.node_id == id))
                {
                    let handle = MemoryBlockHandle {
                        id: Uuid::new_v4(),
                        job_id,
                        size,
                        owner_node: node.node_id,
                        owner_incarnation: node.incarnation,
                        location_type: Location::Ram {
                            node_id: node.node_id,
                        },
                        checksum: None,
                        state: BlockState::Writing,
                        lease_token: Uuid::new_v4(),
                        lease_expires_ms: crate::now_ms() + self.config.lease_seconds * 1000,
                        generation: 0,
                        tensor: tensor.clone(),
                    };
                    let response = self
                        .transport
                        .control(
                            node.network.control_address,
                            &Request::AllocateLocal {
                                handle: handle.clone(),
                                leadership: leadership.clone(),
                            },
                        )
                        .await;
                    match response.and_then(|r| r.check()) {
                        Ok(()) => {
                            self.residency.write().await.register(handle.clone());
                            return Ok(Response::data(handle));
                        }
                        Err(error) => {
                            last_error = error.to_string();
                        }
                    }
                }
                anyhow::bail!("TRAINPOOL_OUT_OF_CAPACITY: {last_error}")
            }
            Request::AllocateLocal { handle, leadership } => {
                self.validate_leadership(&leadership).await?;
                let local = self.local().await;
                ensure!(
                    handle.owner_node == self.node_id
                        && handle.owner_incarnation == local.incarnation,
                    "wrong allocation owner"
                );
                ensure!(
                    matches!(handle.location_type, Location::Ram { node_id } if node_id == self.node_id),
                    "only RAM allocations are supported"
                );
                ensure!(
                    handle.state == BlockState::Writing && handle.checksum.is_none(),
                    "allocation must start in writing state"
                );
                if let Some(meta) = &handle.tensor {
                    meta.validate(handle.size)?;
                }
                self.refresh_memory().await;
                let free = self
                    .ram
                    .accounting
                    .budget
                    .load(std::sync::atomic::Ordering::SeqCst)
                    .saturating_sub(self.ram.accounting.used());
                ensure!(
                    handle.size.saturating_add(262144) <= free,
                    "TRAINPOOL_OUT_OF_CAPACITY: allocation must leave 256 KiB relay headroom"
                );
                self.ram.allocate(handle.clone()).await?;
                let mut metrics = self.metrics.lock().await;
                let m = metrics.job(handle.job_id);
                m.tensor_allocations += 1;
                Ok(Response::data(handle))
            }
            Request::Resolve { id, lease_token } => {
                let table = self.residency.read().await;
                let handle = table
                    .blocks
                    .get(&id)
                    .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_DATA_LOST: block {id} not found"))?
                    .clone();
                drop(table);
                ensure!(handle.lease_token == lease_token, "TRAINPOOL_LEASE_DENIED");
                ensure!(
                    handle.state != BlockState::Lost,
                    "TRAINPOOL_DATA_LOST: block {id} was stored exclusively on node {}",
                    handle.owner_node
                );
                ensure!(
                    handle.lease_expires_ms > crate::now_ms(),
                    "TRAINPOOL_LEASE_EXPIRED"
                );
                self.address(handle.owner_node).await?;
                Ok(Response::data(handle))
            }
            Request::Register {
                handle,
                previous_generation,
                leadership,
            } => {
                self.validate_leadership(&leadership).await?;
                let mut table = self.residency.write().await;
                if let Some(previous) = previous_generation {
                    ensure!(
                        table.blocks.get(&handle.id).is_some_and(
                            |b| b.generation == previous && b.lease_token == handle.lease_token
                        ),
                        "TRAINPOOL_MIGRATION_CONFLICT"
                    );
                    ensure!(
                        handle.generation == previous + 1 && handle.state == BlockState::Ready,
                        "invalid migration commit"
                    );
                }
                table.register(handle);
                Ok(Response::data(()))
            }
            Request::Publish { handle, leadership } => {
                self.validate_leadership(&leadership).await?;
                let block = self.ram.get(handle.id).await?;
                let mut b = block.lock().await;
                b.authorize(handle.lease_token)?;
                ensure!(
                    b.handle.generation == handle.generation && b.handle.state == BlockState::Ready,
                    "destination is not ready"
                );
                b.published = true;
                Ok(Response::data(()))
            }
            Request::Commit {
                handle,
                checksum,
                direct,
            } => {
                if !direct {
                    let h = self.resolve(&handle).await?;
                    return self
                        .transport
                        .control(
                            self.address(h.owner_node).await?,
                            &Request::Commit {
                                handle: h,
                                checksum,
                                direct: true,
                            },
                        )
                        .await;
                }
                let block = self.ram.get(handle.id).await?;
                let mut b = block.lock().await;
                b.authorize(handle.lease_token)?;
                ensure!(b.written == b.bytes.len(), "incomplete upload");
                ensure!(
                    b.hash.finalize().to_hex().as_str() == checksum,
                    "TRAINPOOL_CHECKSUM_MISMATCH"
                );
                b.handle.state = BlockState::Ready;
                b.handle.checksum = Some(checksum);
                let committed = b.handle.clone();
                drop(b);
                // A migrated destination is published only by the source's CAS.
                if committed.generation == 0 {
                    self.forward_leader(&Request::Register {
                        handle: committed.clone(),
                        previous_generation: None,
                        leadership: self.leadership().await,
                    })
                    .await?
                    .check()?;
                }
                Ok(Response::data(committed))
            }
            Request::Renew { handle, direct } => {
                if !direct {
                    let h = self.resolve(&handle).await?;
                    return self
                        .transport
                        .control(
                            self.address(h.owner_node).await?,
                            &Request::Renew {
                                handle: h,
                                direct: true,
                            },
                        )
                        .await;
                }
                let block = self.ram.get(handle.id).await?;
                let mut b = block.lock().await;
                b.authorize(handle.lease_token)?;
                b.handle.lease_expires_ms = crate::now_ms() + self.config.lease_seconds * 1000;
                let renewed = b.handle.clone();
                drop(b);
                self.forward_leader(&Request::Register {
                    handle: renewed.clone(),
                    previous_generation: None,
                    leadership: self.leadership().await,
                })
                .await?
                .check()?;
                Ok(Response::data(renewed))
            }
            Request::Free { handle, direct } => {
                if !direct {
                    let h = self.resolve(&handle).await?;
                    let response = self
                        .transport
                        .control(
                            self.address(h.owner_node).await?,
                            &Request::Free {
                                handle: h.clone(),
                                direct: true,
                            },
                        )
                        .await?;
                    if response.ok {
                        let mut expired = h;
                        expired.lease_expires_ms = 0;
                        self.forward_leader(&Request::Register {
                            handle: expired,
                            previous_generation: None,
                            leadership: self.leadership().await,
                        })
                        .await?
                        .check()?;
                    }
                    return Ok(response);
                }
                self.ram.free(handle.id, handle.lease_token).await?;
                Ok(Response::data(()))
            }
            Request::MemoryPressure {
                node_id,
                currently_owned: _,
                new_budget: _,
                excess_bytes,
            } => {
                let moved = self.relieve_pressure(node_id, excess_bytes).await?;
                Ok(Response::data(
                    serde_json::json!({ "requested_bytes": excess_bytes, "migrated_bytes": moved, "unresolved_bytes": excess_bytes.saturating_sub(moved) }),
                ))
            }
            Request::Migrate {
                handle,
                destination,
                leadership,
            } => {
                self.validate_leadership(&leadership).await?;
                Ok(Response::data(
                    crate::memory::migration::migrate(self, handle, destination, leadership)
                        .await?,
                ))
            }
            Request::Benchmark { bytes } => Ok(Response::data(self.benchmark(bytes).await?)),
            Request::Measure { destination, bytes } => {
                Ok(Response::data(self.measure(destination, bytes).await?))
            }
            Request::Topology => Ok(Response::data(self.topology.read().await.clone())),
            _ => anyhow::bail!("data operation on control channel"),
        }
    }
}
