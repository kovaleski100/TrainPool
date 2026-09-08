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
        let mut candidates: Vec<_> = nodes
            .iter()
            .filter(|n| {
                n.runtime.ram_provider
                    && n.memory.trainpool_ram_available >= size
                    && n.memory.excess() == 0
            })
            .collect();
        candidates.sort_by(|a, b| {
            let cost = |n: &NodeCapabilities| {
                if n.node_id == compute {
                    -1.0
                } else {
                    topology.transfer_cost(compute, n.node_id, size)
                        * (1.0 + n.network.active_transfers as f64)
                }
            };
            cost(a).total_cmp(&cost(b)).then(a.node_id.cmp(&b.node_id))
        });
        candidates
    }
}
