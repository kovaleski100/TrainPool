use crate::{
    cluster::election::{Leadership, elect},
    node::NodeCapabilities,
};
use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};
use uuid::Uuid;

pub const PEER_TIMEOUT: Duration = Duration::from_secs(7);
pub struct Peer {
    pub capabilities: NodeCapabilities,
    pub last_seen: Instant,
}
pub struct Membership {
    pub local_id: Uuid,
    pub peers: BTreeMap<Uuid, Peer>,
}
impl Membership {
    pub fn new(local: NodeCapabilities) -> Self {
        let mut this = Self {
            local_id: local.node_id,
            peers: BTreeMap::new(),
        };
        this.update(local);
        this
    }
    pub fn update(&mut self, capabilities: NodeCapabilities) {
        self.peers.insert(
            capabilities.node_id,
            Peer {
                capabilities,
                last_seen: Instant::now(),
            },
        );
    }
    pub fn expire(&mut self, now: Instant) -> Vec<Uuid> {
        let dead: Vec<_> = self
            .peers
            .iter()
            .filter(|(id, p)| {
                **id != self.local_id && now.saturating_duration_since(p.last_seen) > PEER_TIMEOUT
            })
            .map(|(id, _)| *id)
            .collect();
        for id in &dead {
            self.peers.remove(id);
        }
        dead
    }
    pub fn leader(&self) -> Leadership {
        elect(self.peers.values().map(|p| &p.capabilities))
    }
    pub fn nodes(&self) -> Vec<NodeCapabilities> {
        self.peers
            .values()
            .map(|p| p.capabilities.clone())
            .collect()
    }
}
