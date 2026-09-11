use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MemoryCapabilities {
    pub physical_ram_total: u64,
    pub os_available_ram: u64,
    pub trainpool_ram_used: u64,
    pub trainpool_ram_budget: u64,
    pub trainpool_ram_available: u64,
    pub safety_reserve: u64,
    pub safe_local_ram_allocatable_now: u64,
}
impl MemoryCapabilities {
    pub fn calculate(
        total: u64,
        available: u64,
        owned: u64,
        fraction: f64,
        limit: Option<u64>,
        reserve_bytes: u64,
        reserve_fraction: f64,
    ) -> Self {
        let effective = available.saturating_add(owned).min(total);
        let reserve = reserve_bytes.max((total as f64 * reserve_fraction) as u64);
        let safe_capacity = effective.saturating_sub(reserve);
        let contribution_ceiling = (total as f64 * fraction) as u64;
        let budget = safe_capacity
            .min(contribution_ceiling)
            .min(limit.unwrap_or(u64::MAX));
        let allocatable = budget.saturating_sub(owned);
        Self {
            physical_ram_total: total,
            os_available_ram: available,
            trainpool_ram_used: owned,
            trainpool_ram_budget: budget,
            trainpool_ram_available: allocatable,
            safety_reserve: reserve,
            safe_local_ram_allocatable_now: allocatable,
        }
    }
    pub fn excess(&self) -> u64 {
        self.trainpool_ram_used
            .saturating_sub(self.trainpool_ram_budget)
    }
}
