use trainpool::{
    config::Config,
    node::{
        NetworkCapabilities, NodeCapabilities, RuntimeCapabilities, cpu::CpuCapabilities,
        gpu::GpuCapabilities, memory::MemoryCapabilities,
    },
};
use uuid::Uuid;
pub const GIB: u64 = 1024 * 1024 * 1024;
pub fn node(id: u128, ram_gib: u64, gpu_gib: Option<u64>) -> NodeCapabilities {
    let gpus = gpu_gib
        .map(|size| {
            vec![GpuCapabilities {
                uuid: format!("GPU-{id}"),
                model: "mock NVIDIA".into(),
                vram_total: size * GIB,
                vram_free: size * GIB,
                usable_vram: size * GIB - 96 * 1024 * 1024,
                ..Default::default()
            }]
        })
        .unwrap_or_default();
    NodeCapabilities {
        node_id: Uuid::from_u128(id),
        incarnation: Uuid::from_u128(id + 100),
        leader_generation: Uuid::from_u128(id + 1000),
        hostname: format!("node-{id}"),
        os: "mock".into(),
        architecture: "x86_64".into(),
        cpu: CpuCapabilities::default(),
        memory: MemoryCapabilities::calculate(ram_gib * GIB, ram_gib * GIB, 0, 0.5, None),
        network: NetworkCapabilities {
            control_address: format!("127.0.0.1:{}", 7432 + id).parse().unwrap(),
            transport: "mock".into(),
            chunk_bytes: Config::default().chunk_bytes,
            active_transfers: 0,
        },
        runtime: RuntimeCapabilities {
            gpu_compute: !gpus.is_empty(),
            ram_provider: true,
            disk_spill: false,
            version: "test".into(),
        },
        gpus,
    }
}
