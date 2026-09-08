use super::{
    allocator::{RamAccounting, Reservation},
    block::{BlockState, MemoryBlockHandle},
};
use anyhow::{Result, ensure};
use std::{collections::BTreeMap, sync::Arc};
use tokio::sync::Mutex;
use uuid::Uuid;

pub struct RamBlock {
    pub handle: MemoryBlockHandle,
    pub bytes: Vec<u8>,
    pub written: usize,
    pub transfer_id: Option<Uuid>,
    pub hash: blake3::Hasher,
    pub published: bool,
    _reservation: Reservation,
}
impl RamBlock {
    pub fn authorize(&self, token: Uuid) -> Result<()> {
        ensure!(token == self.handle.lease_token, "TRAINPOOL_LEASE_DENIED");
        ensure!(
            self.handle.lease_expires_ms > crate::now_ms(),
            "TRAINPOOL_LEASE_EXPIRED"
        );
        Ok(())
    }
}
#[derive(Default)]
pub struct LocalRam {
    pub accounting: Arc<RamAccounting>,
    blocks: Mutex<BTreeMap<Uuid, Arc<Mutex<RamBlock>>>>,
}
impl LocalRam {
    pub async fn allocate(&self, handle: MemoryBlockHandle) -> Result<()> {
        let mut blocks = self.blocks.lock().await;
        ensure!(!blocks.contains_key(&handle.id), "block already exists");
        ensure!(blocks.len() < 100_000, "block count limit reached");
        let size: usize = handle.size.try_into()?;
        let reservation = self
            .accounting
            .reserve_payload(handle.job_id, handle.size)?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(size)?;
        bytes.resize(size, 0);
        blocks.insert(
            handle.id,
            Arc::new(Mutex::new(RamBlock {
                published: handle.generation == 0,
                handle,
                bytes,
                written: 0,
                transfer_id: None,
                hash: blake3::Hasher::new(),
                _reservation: reservation,
            })),
        );
        Ok(())
    }
    pub async fn get(&self, id: Uuid) -> Result<Arc<Mutex<RamBlock>>> {
        self.blocks.lock().await.get(&id).cloned().ok_or_else(|| {
            anyhow::anyhow!("TRAINPOOL_DATA_LOST: block {id} is absent on this node")
        })
    }
    pub async fn free(&self, id: Uuid, token: Uuid) -> Result<()> {
        let block = self.blocks.lock().await.get(&id).cloned();
        if let Some(block) = block {
            ensure!(
                block.lock().await.handle.lease_token == token,
                "TRAINPOOL_LEASE_DENIED"
            );
            let mut blocks = self.blocks.lock().await;
            if blocks
                .get(&id)
                .is_some_and(|current| Arc::ptr_eq(current, &block))
            {
                blocks.remove(&id);
            }
        }
        Ok(())
    }
    pub async fn inventory(&self) -> Vec<MemoryBlockHandle> {
        let blocks: Vec<_> = self.blocks.lock().await.values().cloned().collect();
        let mut handles = Vec::with_capacity(blocks.len());
        // Busy transfers must never stop heartbeats. The leader retains prior entries.
        for block in blocks {
            if let Ok(b) = block.try_lock()
                && b.published
            {
                handles.push(b.handle.clone());
            }
        }
        handles
    }
    /// Bounded rotating metadata pages keep large inventories out of heartbeats.
    pub async fn inventory_page(&self, page: u64) -> Vec<MemoryBlockHandle> {
        let blocks = self.blocks.lock().await;
        let pages = blocks.len().div_ceil(512).max(1);
        let selected: Vec<_> = blocks
            .values()
            .skip((page as usize % pages) * 512)
            .take(512)
            .cloned()
            .collect();
        drop(blocks);
        let mut handles = vec![];
        for block in selected {
            if let Ok(b) = block.try_lock()
                && b.published
            {
                handles.push(b.handle.clone());
            }
        }
        handles
    }
    pub async fn expire(&self) {
        let mut blocks = self.blocks.lock().await;
        let mut expired = vec![];
        for (id, block) in blocks.iter() {
            if let Ok(b) = block.try_lock()
                && b.handle.lease_expires_ms <= crate::now_ms()
                && b.handle.state != BlockState::Migrating
            {
                expired.push(*id);
            }
        }
        for id in expired {
            blocks.remove(&id);
        }
    }
}
