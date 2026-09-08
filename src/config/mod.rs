use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    net::{Ipv4Addr, SocketAddr},
    path::{Path, PathBuf},
};
use uuid::Uuid;

pub const MIB: u64 = 1024 * 1024;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub cluster_name: String,
    pub cluster_secret: Option<String>,
    pub listen: SocketAddr,
    pub advertise_ip: Option<Ipv4Addr>,
    pub multicast_group: Ipv4Addr,
    pub discovery_port: u16,
    pub multicast_interface: Ipv4Addr,
    pub discovery_enabled: bool,
    pub seeds: Vec<SocketAddr>,
    pub ram_fraction: f64,
    /// Optional absolute contribution ceiling, also useful for safe local demos.
    pub ram_limit_bytes: Option<u64>,
    pub chunk_bytes: usize,
    pub lease_seconds: u64,
    pub vram_reserve_bytes: u64,
    pub vram_reserve_fraction: f64,
    pub disk: DiskConfig,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DiskConfig {
    pub enabled: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            cluster_name: "trainpool".into(),
            cluster_secret: None,
            listen: "0.0.0.0:7432".parse().unwrap(),
            advertise_ip: None,
            multicast_group: Ipv4Addr::new(239, 255, 74, 32),
            discovery_port: 7433,
            multicast_interface: Ipv4Addr::UNSPECIFIED,
            discovery_enabled: true,
            seeds: vec![],
            ram_fraction: 0.50,
            ram_limit_bytes: None,
            chunk_bytes: (64 * MIB) as usize,
            lease_seconds: 300,
            vram_reserve_bytes: 512 * MIB,
            vram_reserve_fraction: 0.05,
            disk: DiskConfig::default(),
        }
    }
}

impl Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.10..=0.90).contains(&self.ram_fraction),
            "ram-fraction must be between 0.10 and 0.90"
        );
        ensure!(
            (4096..=64 * MIB as usize).contains(&self.chunk_bytes),
            "chunk-bytes must be 4096..67108864"
        );
        ensure!(
            (10..=86400).contains(&self.lease_seconds),
            "lease-seconds must be 10..86400"
        );
        ensure!(
            (0.0..=0.90).contains(&self.vram_reserve_fraction),
            "invalid VRAM reserve fraction"
        );
        ensure!(
            self.multicast_group.is_multicast(),
            "discovery address must be multicast"
        );
        ensure!(
            !self.cluster_name.is_empty() && self.cluster_name.len() <= 128,
            "invalid cluster name"
        );
        ensure!(
            !self.disk.enabled,
            "experimental disk spill: not implemented"
        );
        ensure!(
            self.cluster_secret.as_ref().is_none_or(|s| !s.is_empty()),
            "cluster-secret cannot be empty"
        );
        Ok(())
    }
    pub fn load(dir: &Path) -> Result<Self> {
        let path = dir.join("config.toml");
        let mut config: Self = if path.exists() {
            toml::from_str(&std::fs::read_to_string(path)?)?
        } else {
            Self::default()
        };
        if let Ok(secret) = std::env::var("TRAINPOOL_CLUSTER_SECRET") {
            config.cluster_secret = Some(secret);
        }
        config.validate()?;
        Ok(config)
    }
    pub fn save(&self, dir: &Path) -> Result<()> {
        self.validate()?;
        std::fs::create_dir_all(dir)?;
        std::fs::write(dir.join("config.toml"), toml::to_string_pretty(self)?)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(
                dir.join("config.toml"),
                std::fs::Permissions::from_mode(0o600),
            )?;
        }
        Ok(())
    }
}

pub fn default_dir() -> PathBuf {
    std::env::var_os("TRAINPOOL_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("HOME").unwrap_or_else(|| ".".into())).join(".trainpool")
        })
}

pub fn node_id(dir: &Path) -> Result<Uuid> {
    use std::io::Write;
    std::fs::create_dir_all(dir)?;
    let path = dir.join("node_id");
    if path.exists() {
        return Ok(std::fs::read_to_string(path)?.trim().parse()?);
    }
    let id = Uuid::new_v4();
    match std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut f) => {
            writeln!(f, "{id}")?;
            Ok(id)
        }
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            Ok(std::fs::read_to_string(path)?.trim().parse()?)
        }
        Err(e) => Err(e.into()),
    }
}
