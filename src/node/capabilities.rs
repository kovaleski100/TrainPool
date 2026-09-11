use super::{cpu::CpuCapabilities, gpu::GpuCapabilities, memory::MemoryCapabilities};
use crate::config::Config;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use uuid::Uuid;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RuntimeCapabilities {
    pub gpu_compute: bool,
    pub ram_provider: bool,
    pub disk_spill: bool,
    pub version: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NetworkCapabilities {
    pub control_address: SocketAddr,
    #[serde(default)]
    pub data_address: Option<SocketAddr>,
    pub transport: String,
    #[serde(default)]
    pub data_transport: String,
    pub chunk_bytes: usize,
    pub active_transfers: u32,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeCapabilities {
    pub node_id: Uuid,
    /// Changes on each daemon start, distinguishing restarts from retained data.
    pub incarnation: Uuid,
    pub leader_generation: Uuid,
    pub hostname: String,
    pub os: String,
    pub architecture: String,
    pub cpu: CpuCapabilities,
    pub memory: MemoryCapabilities,
    pub gpus: Vec<GpuCapabilities>,
    pub network: NetworkCapabilities,
    pub runtime: RuntimeCapabilities,
}
impl NodeCapabilities {
    pub fn election_score(&self) -> u64 {
        self.gpus
            .iter()
            .fold(self.memory.physical_ram_total, |sum, gpu| {
                sum.saturating_add(gpu.vram_total)
            })
    }
    pub fn refresh_memory(&mut self, system: &mut sysinfo::System, config: &Config, owned: u64) {
        system.refresh_memory();
        let mut available = system.available_memory();
        // Respect container headroom as well as physical host headroom.
        if let Some(limits) = system.cgroup_limits() {
            available = available.min(limits.free_memory);
        }
        self.memory = MemoryCapabilities::calculate(
            self.memory.physical_ram_total,
            available,
            owned,
            config.ram_fraction,
            config.ram_limit_bytes,
            config.ram_reserve_bytes,
            config.ram_reserve_fraction,
        );
    }
    pub async fn detect(id: Uuid, address: SocketAddr, config: &Config) -> Self {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        system.refresh_cpu_all();
        let gpus = super::gpu::detect(config).await;
        let architecture = std::env::consts::ARCH.to_owned();
        let mut node = Self {
            node_id: id,
            incarnation: Uuid::new_v4(),
            leader_generation: Uuid::new_v4(),
            hostname: sysinfo::System::host_name().unwrap_or_else(|| id.to_string()),
            os: std::env::consts::OS.into(),
            architecture: architecture.clone(),
            cpu: CpuCapabilities {
                model: system
                    .cpus()
                    .first()
                    .map(|c| c.brand())
                    .unwrap_or("unknown")
                    .into(),
                physical_cores: sysinfo::System::physical_core_count().unwrap_or(0),
                logical_cores: system.cpus().len(),
                architecture,
            },
            memory: MemoryCapabilities {
                physical_ram_total: system.total_memory(),
                ..Default::default()
            },
            runtime: RuntimeCapabilities {
                gpu_compute: !gpus.is_empty(),
                ram_provider: true,
                disk_spill: false,
                version: env!("CARGO_PKG_VERSION").into(),
            },
            gpus,
            network: NetworkCapabilities {
                control_address: address,
                data_address: Some(address),
                transport: "framed-tcp-control".into(),
                data_transport: config.data_transport.clone(),
                chunk_bytes: config.chunk_bytes,
                active_transfers: 0,
            },
        };
        node.refresh_memory(&mut system, config, 0);
        node
    }
}
