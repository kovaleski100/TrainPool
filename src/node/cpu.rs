use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CpuCapabilities {
    pub model: String,
    pub physical_cores: usize,
    pub logical_cores: usize,
    pub architecture: String,
}
