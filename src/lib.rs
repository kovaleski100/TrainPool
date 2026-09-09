pub mod cli;
pub mod cluster;
pub mod config;
pub mod jobs;
pub mod launcher;
pub mod memory;
pub mod metrics;
pub mod node;
pub mod protocol;
pub mod runtime;
pub mod scheduler;
pub mod transport;

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
