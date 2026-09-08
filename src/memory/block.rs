use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Location {
    Gpu {
        node_id: Uuid,
        gpu_id: String,
    },
    Ram {
        node_id: Uuid,
    },
    /// Reserved wire representation. No disk allocator exists.
    Disk {
        node_id: Uuid,
    },
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BlockState {
    Writing,
    Ready,
    Migrating,
    Lost,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorMetadata {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub layout: String,
    pub byte_length: u64,
}
impl TensorMetadata {
    pub fn validate(&self, size: u64) -> Result<()> {
        ensure!(
            self.layout == "contiguous" && self.shape.len() <= 32,
            "unsupported tensor layout or rank"
        );
        let width = match self.dtype.as_str() {
            "uint8" | "int8" | "bool" => 1,
            "float16" | "bfloat16" | "int16" => 2,
            "float32" | "int32" => 4,
            "float64" | "int64" | "complex64" => 8,
            "complex128" => 16,
            _ => anyhow::bail!("unsupported tensor dtype"),
        };
        let bytes = self
            .shape
            .iter()
            .try_fold(width, |n: u64, dim| n.checked_mul(*dim));
        ensure!(
            bytes == Some(size) && size == self.byte_length,
            "tensor shape/dtype/byte_length mismatch"
        );
        Ok(())
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemoryBlockHandle {
    pub id: Uuid,
    pub job_id: Uuid,
    pub size: u64,
    pub owner_node: Uuid,
    pub owner_incarnation: Uuid,
    pub location_type: Location,
    pub checksum: Option<String>,
    pub state: BlockState,
    pub lease_token: Uuid,
    pub lease_expires_ms: u64,
    pub generation: u64,
    pub tensor: Option<TensorMetadata>,
}
