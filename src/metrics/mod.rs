use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use uuid::Uuid;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct JobMetrics {
    pub tensor_allocations: u64,
    pub tensor_migrations: u64,
    pub bytes_local_ram_to_gpu: u64,
    pub bytes_gpu_to_local_ram: u64,
    pub bytes_remote_ram_to_local: u64,
    pub bytes_local_to_remote_ram: u64,
    pub migration_latency_ms: f64,
    pub prefetch_hits: u64,
    pub prefetch_misses: u64,
    pub gpu_wait_for_data_ms: f64,
    pub ram_pressure_events: u64,
    pub network_bytes: u64,
    pub network_wait_ms: f64,
    pub peak_gpu_residency: u64,
    pub current_gpu_residency: u64,
    pub peak_vram_resident_bytes: u64,
    pub current_vram_resident_bytes: u64,
    pub peak_local_ram_backing_bytes: u64,
    pub current_local_ram_backing_bytes: u64,
    pub peak_remote_ram_backing_bytes: u64,
    pub current_remote_ram_backing_bytes: u64,
    pub gpu_to_local_bytes: u64,
    pub local_to_gpu_bytes: u64,
    pub gpu_to_remote_bytes: u64,
    pub remote_to_gpu_bytes: u64,
    pub remote_to_local_bytes: u64,
    pub local_to_remote_bytes: u64,
    pub eviction_count: u64,
    pub prefetch_count: u64,
    pub gpu_id: Option<String>,
    pub peak_ram_residency: u64,
    pub current_ram_residency: u64,
    pub failed_transfers: u64,
}
#[derive(Default)]
pub struct Metrics {
    pub jobs: BTreeMap<Uuid, JobMetrics>,
}
impl Metrics {
    pub fn job(&mut self, id: Uuid) -> &mut JobMetrics {
        self.jobs.entry(id).or_default()
    }
}
