use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinkEstimate {
    pub source: Uuid,
    pub destination: Uuid,
    pub latency_ms: f64,
    pub bytes_per_second: Option<f64>,
    pub sampled_at_ms: u64,
    pub active_transfers: u32,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Topology {
    pub links: Vec<LinkEstimate>,
}
impl Topology {
    pub fn update(&mut self, mut sample: LinkEstimate) {
        if let Some(old) = self
            .links
            .iter_mut()
            .find(|l| l.source == sample.source && l.destination == sample.destination)
        {
            sample.latency_ms = 0.25 * sample.latency_ms + 0.75 * old.latency_ms;
            sample.bytes_per_second = match (sample.bytes_per_second, old.bytes_per_second) {
                (Some(a), Some(b)) => Some(0.25 * a + 0.75 * b),
                (a, b) => a.or(b),
            };
            *old = sample;
        } else {
            self.links.push(sample);
        }
    }
    pub fn transfer_cost(&self, source: Uuid, destination: Uuid, size: u64) -> f64 {
        self.links
            .iter()
            .find(|l| l.source == source && l.destination == destination)
            .map(|l| {
                l.latency_ms / 1000.0
                    + size as f64 / l.bytes_per_second.unwrap_or(10_000_000.0).max(1.0)
                        * (1.0 + l.active_transfers as f64)
            })
            .unwrap_or(0.01 + size as f64 / 10_000_000.0)
    }
}
