use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    pub physical_ram_total: u64,
    pub os_available_ram: u64,
    pub trainpool_ram_used: u64,
    pub trainpool_ram_budget: u64,
    pub trainpool_ram_available: u64,
}
impl MemoryCapabilities {
    pub fn calculate(
        total: u64,
        available: u64,
        owned: u64,
        fraction: f64,
        limit: Option<u64>,
    ) -> Self {
        let effective = available.saturating_add(owned).min(total);
        let budget = ((effective as f64 * fraction) as u64).min(limit.unwrap_or(u64::MAX));
        Self {
            physical_ram_total: total,
            os_available_ram: available,
            trainpool_ram_used: owned,
            trainpool_ram_budget: budget,
            trainpool_ram_available: budget.saturating_sub(owned),
        }
    }
    pub fn excess(&self) -> u64 {
        self.trainpool_ram_used
            .saturating_sub(self.trainpool_ram_budget)
    }
}
