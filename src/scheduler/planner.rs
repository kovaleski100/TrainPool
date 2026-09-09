use crate::{cluster::election::Leadership, node::NodeCapabilities, scheduler::topology::Topology};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GpuAssignment {
    pub node_id: Uuid,
    pub gpu_id: String,
    pub usable_bytes: u64,
    pub capacity_share: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageSpec {
    pub name: String,
    pub working_set_bytes: u64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StageAssignment {
    pub stage: String,
    pub node_id: Uuid,
    pub gpu_id: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainingPlan {
    pub job_id: Uuid,
    pub leader_id: Uuid,
    pub leader_epoch: Uuid,
    pub strategy: String,
    pub compute_nodes: Vec<Uuid>,
    pub gpu_assignments: Vec<GpuAssignment>,
    pub memory_nodes: Vec<Uuid>,
    pub tensor_placement_policy: String,
    pub topology: Topology,
    pub memory_budgets: BTreeMap<Uuid, u64>,
    pub stages: Vec<StageAssignment>,
    pub execution_supported: bool,
}
pub trait TrainingPlanner {
    fn plan(
        &self,
        nodes: &[NodeCapabilities],
        leader: &Leadership,
        topology: &Topology,
        stages: &[StageSpec],
    ) -> Result<TrainingPlan>;
}
pub struct CapacityPlanner;
impl TrainingPlanner for CapacityPlanner {
    fn plan(
        &self,
        nodes: &[NodeCapabilities],
        leader: &Leadership,
        topology: &Topology,
        stages: &[StageSpec],
    ) -> Result<TrainingPlan> {
        let mut gpus: Vec<_> = nodes
            .iter()
            .filter(|n| n.runtime.gpu_compute)
            .flat_map(|n| {
                n.gpus
                    .iter()
                    .filter(|g| g.usable_vram > 0)
                    .map(|g| GpuAssignment {
                        node_id: n.node_id,
                        gpu_id: g.uuid.clone(),
                        usable_bytes: g.usable_vram,
                        capacity_share: 0.0,
                    })
            })
            .collect();
        let total: u64 = gpus.iter().map(|g| g.usable_bytes).sum();
        for gpu in &mut gpus {
            gpu.capacity_share = gpu.usable_bytes as f64 / total as f64;
        }
        gpus.sort_by(|a, b| {
            b.usable_bytes
                .cmp(&a.usable_bytes)
                .then(a.gpu_id.cmp(&b.gpu_id))
                .then(a.node_id.cmp(&b.node_id))
        });
        // Automatic capacity execution currently uses exactly one compute GPU.
        // Additional GPUs remain cluster resources, but are not presented as if
        // their VRAM participated in this job.
        if stages.is_empty() && !gpus.is_empty() {
            gpus.truncate(1);
            gpus[0].capacity_share = 1.0;
        }
        let mut loads = vec![0_u64; gpus.len()];
        let mut assignments = vec![];
        for stage in stages {
            let candidate = gpus
                .iter()
                .enumerate()
                .filter(|(_, g)| g.usable_bytes >= stage.working_set_bytes)
                .min_by(|(ai, a), (bi, b)| {
                    (loads[*ai] as f64 / a.usable_bytes as f64)
                        .total_cmp(&(loads[*bi] as f64 / b.usable_bytes as f64))
                });
            let Some((index, gpu)) = candidate else {
                anyhow::bail!(
                    "TRAINPOOL_STAGE_TOO_LARGE: {} fits no usable GPU",
                    stage.name
                );
            };
            loads[index] += stage.working_set_bytes;
            assignments.push(StageAssignment {
                stage: stage.name.clone(),
                node_id: gpu.node_id,
                gpu_id: gpu.gpu_id.clone(),
            });
        }
        let mut compute_nodes: Vec<_> = gpus.iter().map(|g| g.node_id).collect();
        compute_nodes.sort();
        compute_nodes.dedup();
        let count = gpus.len();
        Ok(TrainingPlan {
            job_id: Uuid::new_v4(),
            leader_id: leader
                .leader_id
                .ok_or_else(|| anyhow::anyhow!("no leader"))?,
            leader_epoch: leader.leader_epoch,
            strategy: match count {
                0 => "MemoryOnly",
                1 => "SingleGpuDistributedMemory",
                _ => "ExplicitStagePartitioningExperimental",
            }
            .into(),
            compute_nodes,
            gpu_assignments: gpus,
            memory_nodes: nodes
                .iter()
                .filter(|n| n.runtime.ram_provider)
                .map(|n| n.node_id)
                .collect(),
            tensor_placement_policy: "local-ram-then-transfer-cost".into(),
            topology: topology.clone(),
            memory_budgets: nodes
                .iter()
                .map(|n| (n.node_id, n.memory.trainpool_ram_budget))
                .collect(),
            stages: assignments,
            execution_supported: count == 1,
        })
    }
}
pub fn require_cuda(plan: &TrainingPlan, node: Uuid) -> Result<()> {
    ensure!(
        plan.gpu_assignments.iter().any(|g| g.node_id == node),
        "TRAINPOOL_NO_CUDA: node {node} is a memory provider only"
    );
    Ok(())
}
