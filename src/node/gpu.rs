use crate::config::{Config, MIB};
use serde::{Deserialize, Serialize};
use std::time::Duration;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GpuCapabilities {
    pub uuid: String,
    pub model: String,
    pub vram_total: u64,
    pub vram_free: u64,
    pub vram_used: u64,
    pub usable_vram: u64,
    pub trainpool_allocated_vram: u64,
    pub utilization: Option<f64>,
    pub cuda_capability: Option<String>,
    pub driver_version: String,
    pub temperature: Option<f64>,
}

pub fn usable_vram(total: u64, free: u64, reserve: u64, fraction: f64) -> u64 {
    free.saturating_sub(reserve.max((total as f64 * fraction) as u64))
}

/// nvidia-smi is optional; no driver is linked into the runtime.
pub async fn detect(config: &Config) -> Vec<GpuCapabilities> {
    let base = "--query-gpu=uuid,name,memory.total,memory.free,memory.used,utilization.gpu,driver_version,temperature.gpu";
    for query in [format!("{base},compute_cap"), base.to_owned()] {
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            tokio::process::Command::new("nvidia-smi")
                .args([&query, "--format=csv,noheader,nounits"])
                .kill_on_drop(true)
                .output(),
        )
        .await;
        let Ok(Ok(output)) = result else {
            return vec![];
        };
        if output.status.success() {
            return parse_csv(&String::from_utf8_lossy(&output.stdout), config);
        }
    }
    vec![]
}

pub fn parse_csv(output: &str, config: &Config) -> Vec<GpuCapabilities> {
    output
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::trim).collect();
            if !(8..=9).contains(&fields.len()) {
                return None;
            }
            let total = fields[2].parse::<u64>().ok()? * MIB;
            let free = fields[3].parse::<u64>().ok()? * MIB;
            Some(GpuCapabilities {
                uuid: fields[0].into(),
                model: fields[1].into(),
                vram_total: total,
                vram_free: free,
                vram_used: fields[4].parse::<u64>().ok()? * MIB,
                usable_vram: usable_vram(
                    total,
                    free,
                    config.vram_reserve_bytes,
                    config.vram_reserve_fraction,
                ),
                trainpool_allocated_vram: 0,
                utilization: fields[5].parse().ok(),
                driver_version: fields[6].into(),
                temperature: fields[7].parse().ok(),
                cuda_capability: fields
                    .get(8)
                    .and_then(|s| s.parse::<f64>().ok().map(|_| (*s).into())),
            })
        })
        .collect()
}
