use crate::{node::NodeCapabilities, scheduler::topology::Topology};
use uuid::Uuid;

pub trait PlacementPolicy: Send + Sync {
    fn rank_ram<'a>(
        &self,
        nodes: &'a [NodeCapabilities],
        compute: Uuid,
        size: u64,
        topology: &Topology,
    ) -> Vec<&'a NodeCapabilities>;
}
pub struct CapacityPlacement;
impl PlacementPolicy for CapacityPlacement {
    fn rank_ram<'a>(
        &self,
        nodes: &'a [NodeCapabilities],
        compute: Uuid,
        size: u64,
        topology: &Topology,
    ) -> Vec<&'a NodeCapabilities> {
        let candidates: Vec<_> = nodes
            .iter()
            .filter(|n| {
                n.runtime.ram_provider
                    && n.memory.trainpool_ram_available >= size
                    && n.memory.excess() == 0
            })
            .collect();
        let (mut local, mut remote): (Vec<_>, Vec<_>) = candidates
            .into_iter()
            .partition(|node| node.node_id == compute);
        remote.sort_by(|a, b| {
            let cost = |n: &NodeCapabilities| {
                topology.transfer_cost(compute, n.node_id, size)
                    * (1.0 + n.network.active_transfers as f64)
            };
            cost(a)
                .total_cmp(&cost(b))
                .then_with(|| {
                    b.memory
                        .trainpool_ram_available
                        .cmp(&a.memory.trainpool_ram_available)
                })
                .then(a.node_id.cmp(&b.node_id))
        });
        // Locality is a tier boundary, not merely a small cost preference.
        // Remote cost/capacity/topology ranking only runs after local RAM.
        local.append(&mut remote);
        local
    }
}
