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
        let mut local = Vec::new();
        let mut remote = Vec::new();
        for node in nodes.iter().filter(|n| {
            n.runtime.ram_provider
                && n.memory.trainpool_ram_available >= size
                && n.memory.excess() == 0
        }) {
            if node.node_id == compute {
                local.push(node);
            } else {
                let cost = topology.transfer_cost(compute, node.node_id, size)
                    * (1.0 + node.network.active_transfers as f64);
                remote.push((node, cost));
            }
        }
        remote.sort_by(|(a, a_cost), (b, b_cost)| {
            a_cost
                .total_cmp(b_cost)
                .then_with(|| {
                    b.memory
                        .trainpool_ram_available
                        .cmp(&a.memory.trainpool_ram_available)
                })
                .then(a.node_id.cmp(&b.node_id))
        });
        // Locality is a tier boundary, not merely a small cost preference.
        // Remote cost/capacity/topology ranking only runs after local RAM.
        local.extend(remote.into_iter().map(|(node, _)| node));
        local
    }
}
