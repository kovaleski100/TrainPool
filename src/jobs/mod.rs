use crate::scheduler::planner::TrainingPlan;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Job {
    pub plan: TrainingPlan,
    pub error: Option<String>,
}
#[derive(Default)]
pub struct Jobs {
    pub entries: BTreeMap<Uuid, Job>,
}
impl Jobs {
    pub fn lose_compute(&mut self, node: Uuid) {
        for job in self
            .entries
            .values_mut()
            .filter(|j| j.plan.compute_nodes.contains(&node))
        {
            job.error = Some(format!(
                "TRAINPOOL_COMPUTE_LOST: GPU node {node} disappeared"
            ));
        }
    }
    pub fn fail(&mut self, id: Uuid, error: String) {
        if let Some(job) = self.entries.get_mut(&id) {
            job.error = Some(error);
        }
    }
}
