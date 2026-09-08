use super::block::{BlockState, MemoryBlockHandle};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Default)]
pub struct ResidencyTable {
    pub blocks: BTreeMap<Uuid, MemoryBlockHandle>,
}
impl ResidencyTable {
    pub fn register(&mut self, mut handle: MemoryBlockHandle) {
        if let Some(old) = self.blocks.get_mut(&handle.id) {
            if old.lease_expires_ms == 0 {
                return;
            }
            if handle.generation == old.generation && handle.lease_expires_ms != 0 {
                old.lease_expires_ms = old.lease_expires_ms.max(handle.lease_expires_ms);
                handle.lease_expires_ms = old.lease_expires_ms;
                if old.state == BlockState::Ready && handle.state == BlockState::Writing {
                    return;
                }
            }
        }
        if self
            .blocks
            .get(&handle.id)
            .is_none_or(|old| handle.generation >= old.generation)
        {
            self.blocks.insert(handle.id, handle);
        }
    }
    pub fn lose_node(&mut self, node: Uuid) -> Vec<Uuid> {
        let mut jobs = vec![];
        for b in self.blocks.values_mut().filter(|b| b.owner_node == node) {
            b.state = BlockState::Lost;
            jobs.push(b.job_id);
        }
        jobs
    }
}
