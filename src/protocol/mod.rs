use crate::{
    cluster::election::Leadership,
    memory::block::{MemoryBlockHandle, TensorMetadata},
    metrics::JobMetrics,
    node::NodeCapabilities,
    scheduler::{planner::StageSpec, topology::Topology},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const VERSION: u32 = 1;
pub const MAX_FRAME: usize = 8 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wire<T> {
    pub protocol_version: u32,
    pub message: T,
}
#[derive(Serialize, Deserialize)]
pub struct Challenge {
    pub nonce: Uuid,
}
#[derive(Serialize, Deserialize)]
pub struct Authentication {
    pub cluster_name: String,
    pub nonce: Uuid,
    pub mac: Option<String>,
}
#[derive(Serialize, Deserialize)]
pub struct Authenticated {
    pub mac: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    Ping,
    Status,
    Capabilities,
    Exchange {
        node: NodeCapabilities,
        inventory: Vec<MemoryBlockHandle>,
    },
    Inventory {
        page: u64,
    },
    Metrics,
    ReserveStaging {
        bytes: u64,
    },
    ReleaseStaging {
        reservation_id: Uuid,
    },
    ReportMetrics {
        job_id: Uuid,
        metrics: JobMetrics,
    },
    Plan {
        stages: Vec<StageSpec>,
    },
    JobStatus {
        job_id: Uuid,
    },
    Allocate {
        size: u64,
        job_id: Uuid,
        compute_node: Uuid,
        preferred_node: Option<Uuid>,
        tensor: Option<TensorMetadata>,
    },
    AllocateLocal {
        handle: MemoryBlockHandle,
        leadership: Leadership,
    },
    Resolve {
        id: Uuid,
        lease_token: Uuid,
    },
    Register {
        handle: MemoryBlockHandle,
        previous_generation: Option<u64>,
        leadership: Leadership,
    },
    Publish {
        handle: MemoryBlockHandle,
        leadership: Leadership,
    },
    WriteChunk {
        handle: MemoryBlockHandle,
        transfer_id: Uuid,
        offset: u64,
        length: usize,
        total_size: u64,
        checksum: String,
        direct: bool,
    },
    Commit {
        handle: MemoryBlockHandle,
        checksum: String,
        direct: bool,
    },
    ReadChunk {
        handle: MemoryBlockHandle,
        transfer_id: Uuid,
        offset: u64,
        length: usize,
        direct: bool,
    },
    Renew {
        handle: MemoryBlockHandle,
        direct: bool,
    },
    Free {
        handle: MemoryBlockHandle,
        direct: bool,
    },
    MemoryPressure {
        node_id: Uuid,
        currently_owned: u64,
        new_budget: u64,
        excess_bytes: u64,
    },
    Migrate {
        handle: MemoryBlockHandle,
        destination: Uuid,
        leadership: Leadership,
    },
    Benchmark {
        bytes: usize,
    },
    Probe {
        bytes: usize,
    },
    Measure {
        destination: Uuid,
        bytes: usize,
    },
    Topology,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Response {
    pub ok: bool,
    #[serde(default)]
    pub data: serde_json::Value,
    pub error: Option<String>,
}
impl Response {
    pub fn data(value: impl Serialize) -> Self {
        Self {
            ok: true,
            data: serde_json::to_value(value).expect("serializable response"),
            error: None,
        }
    }
    pub fn error(error: impl ToString) -> Self {
        Self {
            ok: false,
            data: serde_json::Value::Null,
            error: Some(error.to_string()),
        }
    }
    pub fn into_data<T: serde::de::DeserializeOwned>(self) -> anyhow::Result<T> {
        anyhow::ensure!(
            self.ok,
            "{}",
            self.error.unwrap_or_else(|| "protocol failure".into())
        );
        Ok(serde_json::from_value(self.data)?)
    }
    pub fn check(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.ok,
            "{}",
            self.error.as_deref().unwrap_or("protocol failure")
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClusterStatus {
    pub local_node: Uuid,
    pub leadership: Leadership,
    pub nodes: Vec<NodeCapabilities>,
    pub logical_memory: LogicalTrainingMemory,
    pub topology: Topology,
    pub chunk_bytes: usize,
    pub lease_seconds: u64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LogicalTrainingMemory {
    pub physical_ram: u64,
    pub pool_ram_budget: u64,
    pub pool_ram_allocatable: u64,
    pub allocated_ram: u64,
    pub physical_vram: u64,
    pub free_vram: u64,
    pub usable_vram: u64,
    pub allocated_vram: u64,
    pub logical_training_capacity: u64,
    pub cluster_physical_vram: u64,
    pub cluster_usable_vram: u64,
    pub primary_gpu_physical_vram: u64,
    pub primary_gpu_usable_vram: u64,
    pub pool_ram_owned: u64,
    pub current_job_backing_capacity: u64,
    pub current_job_remaining_capacity: u64,
    pub disk_spill: bool,
    pub local_physical_ram: u64,
    pub local_os_available_ram: u64,
    pub local_pool_ram_budget: u64,
    pub local_pool_ram_used: u64,
    pub local_pool_ram_allocatable: u64,
    pub local_ram_safety_reserve: u64,
    pub remote_pool_ram_used: u64,
    pub sdk_metrics_timestamp_ms: Option<u64>,
    pub sdk_metrics_age_ms: Option<u64>,
    pub sdk_physical_vram_bytes: Option<u64>,
    pub driver_used_vram_bytes: Option<u64>,
    pub driver_free_vram_bytes: Option<u64>,
    pub torch_allocated_bytes: Option<u64>,
    pub torch_reserved_bytes: Option<u64>,
    pub torch_reclaimable_bytes: Option<u64>,
    pub trainpool_resident_vram_bytes: Option<u64>,
    pub safe_vram_allocatable_bytes: Option<u64>,
    pub configured_usable_vram_ceiling_bytes: Option<u64>,
    pub configured_vram_safety_reserve_bytes: Option<u64>,
}
impl LogicalTrainingMemory {
    pub fn from_nodes(nodes: &[NodeCapabilities]) -> Self {
        let mut m = Self::default();
        for n in nodes {
            m.physical_ram += n.memory.physical_ram_total;
            if n.runtime.ram_provider {
                m.pool_ram_budget += n.memory.trainpool_ram_budget;
                m.pool_ram_allocatable += n.memory.trainpool_ram_available;
                m.allocated_ram += n.memory.trainpool_ram_used;
            }
            for g in &n.gpus {
                m.physical_vram += g.vram_total;
                m.free_vram += g.vram_free;
                m.usable_vram += g.usable_vram;
                m.allocated_vram += g.trainpool_allocated_vram;
            }
        }
        m.cluster_physical_vram = m.physical_vram;
        m.cluster_usable_vram = m.usable_vram;
        m.pool_ram_owned = m.allocated_ram;
        let primary = nodes
            .iter()
            .filter(|n| n.runtime.gpu_compute)
            .flat_map(|n| {
                n.gpus
                    .iter()
                    .filter(|g| g.usable_vram > 0)
                    .map(move |g| (n, g))
            })
            .min_by(|(an, a), (bn, b)| {
                b.usable_vram
                    .cmp(&a.usable_vram)
                    .then(a.uuid.cmp(&b.uuid))
                    .then(an.node_id.cmp(&bn.node_id))
            });
        if let Some((_, gpu)) = primary {
            m.primary_gpu_physical_vram = gpu.vram_total;
            m.primary_gpu_usable_vram = gpu.usable_vram;
        }
        // Snapshot for a new single-GPU job. Owned RAM is already inside budget.
        m.current_job_backing_capacity = m.primary_gpu_usable_vram + m.pool_ram_budget;
        m.current_job_remaining_capacity = m.primary_gpu_usable_vram + m.pool_ram_allocatable;
        // Retain the old wire field, with corrected remaining-capacity semantics.
        m.logical_training_capacity = m.current_job_remaining_capacity;
        m
    }
}
