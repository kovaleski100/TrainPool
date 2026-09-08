use crate::{
    cluster::election::Leadership,
    memory::{
        block::{BlockState, Location, MemoryBlockHandle},
        remote_ram,
    },
    protocol::Request,
    runtime::Runtime,
    transport::Transport,
};
use anyhow::{Result, ensure};
use uuid::Uuid;

/// Source keeps its allocation pinned until destination verification AND metadata CAS.
pub async fn migrate(
    runtime: &Runtime,
    handle: MemoryBlockHandle,
    destination: Uuid,
    leadership: Leadership,
) -> Result<MemoryBlockHandle> {
    ensure!(
        destination != runtime.node_id,
        "migration destination equals source"
    );
    let block = runtime.ram.get(handle.id).await?;
    let b = block.lock().await;
    b.authorize(handle.lease_token)?;
    ensure!(
        b.handle.generation == handle.generation && b.handle.state == BlockState::Ready,
        "migration requires current committed block"
    );
    let node = runtime
        .membership
        .read()
        .await
        .peers
        .get(&destination)
        .ok_or_else(|| anyhow::anyhow!("destination unavailable"))?
        .capabilities
        .clone();
    let mut target = b.handle.clone();
    target.owner_node = destination;
    target.owner_incarnation = node.incarnation;
    target.location_type = Location::Ram {
        node_id: destination,
    };
    target.generation += 1;
    target.state = BlockState::Writing;
    target.checksum = None;
    target.lease_expires_ms = crate::now_ms() + runtime.config.lease_seconds * 1000;
    let address = node.network.control_address;
    runtime
        .transport
        .control(
            address,
            &Request::AllocateLocal {
                handle: target.clone(),
                leadership: leadership.clone(),
            },
        )
        .await?
        .check()?;
    let start = std::time::Instant::now();
    let copy: Result<MemoryBlockHandle> = async {
        let transfer = Uuid::new_v4();
        let chunk = runtime.config.chunk_bytes.min(node.network.chunk_bytes);
        for (index, bytes) in b.bytes.chunks(chunk).enumerate() {
            remote_ram::write_chunk(
                &runtime.transport,
                address,
                &target,
                transfer,
                (index * chunk) as u64,
                bytes,
            )
            .await?;
        }
        runtime
            .transport
            .control(
                address,
                &Request::Commit {
                    handle: target.clone(),
                    checksum: b.handle.checksum.clone().expect("ready block checksum"),
                    direct: true,
                },
            )
            .await?
            .into_data()
    }
    .await;
    let committed = match copy {
        Ok(h) => h,
        Err(error) => {
            let _ = runtime
                .transport
                .control(
                    address,
                    &Request::Free {
                        handle: target,
                        direct: true,
                    },
                )
                .await;
            runtime
                .metrics
                .lock()
                .await
                .job(handle.job_id)
                .failed_transfers += 1;
            return Err(error);
        }
    };
    // After sending CAS, a lost ACK is ambiguous: retain BOTH copies, never delete destination.
    runtime
        .forward_leader(&Request::Register {
            handle: committed.clone(),
            previous_generation: Some(handle.generation),
            leadership: leadership.clone(),
        })
        .await?
        .check()?;
    runtime
        .transport
        .control(
            address,
            &Request::Publish {
                handle: committed.clone(),
                leadership,
            },
        )
        .await?
        .check()?;
    drop(b);
    drop(block);
    runtime.ram.free(handle.id, handle.lease_token).await?;
    let mut metrics = runtime.metrics.lock().await;
    let m = metrics.job(handle.job_id);
    m.tensor_migrations += 1;
    m.bytes_local_to_remote_ram += handle.size;
    m.migration_latency_ms += start.elapsed().as_secs_f64() * 1000.0;
    tracing::info!(block_id = %handle.id, source = %runtime.node_id, %destination, bytes = handle.size, "RAM migration committed");
    Ok(committed)
}
