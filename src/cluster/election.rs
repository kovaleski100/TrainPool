use crate::node::NodeCapabilities;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Leadership {
    pub leader_id: Option<Uuid>,
    /// A leader-issued generation token, not a consensus term.
    pub leader_epoch: Uuid,
    pub election_score: u64,
}

pub fn elect<'a>(nodes: impl Iterator<Item = &'a NodeCapabilities>) -> Leadership {
    match nodes.max_by_key(|n| (n.election_score(), n.node_id)) {
        Some(n) => Leadership {
            leader_id: Some(n.node_id),
            leader_epoch: n.leader_generation,
            election_score: n.election_score(),
        },
        None => Leadership::default(),
    }
}
