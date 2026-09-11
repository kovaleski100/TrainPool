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
    pub ram_reserve_bytes: u64,
    pub ram_reserve_fraction: f64,
    /// Optional absolute contribution ceiling, also useful for safe local demos.
    pub ram_limit_bytes: Option<u64>,
    pub chunk_bytes: usize,
    /// Inter-node payload transport. Control and local SDK traffic remain TCP.
    pub data_transport: String,
    pub udp_payload_bytes: usize,
    pub udp_window_packets: usize,
    pub udp_initial_rto_ms: u64,
    pub udp_pacing_micros: u64,
    pub udp_max_retries: u32,
    pub udp_max_sessions: usize,
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
            ram_fraction: 0.90,
            ram_reserve_bytes: 1024 * MIB,
            ram_reserve_fraction: 0.10,
            ram_limit_bytes: None,
            chunk_bytes: (64 * MIB) as usize,
            data_transport: "udp".into(),
            udp_payload_bytes: 1200,
            udp_window_packets: 64,
            udp_initial_rto_ms: 50,
            udp_pacing_micros: 20,
            udp_max_retries: 8,
            udp_max_sessions: 128,
            lease_seconds: 300,
            // A small non-zero floor protects CUDA context/workspaces on small
            // GPUs; the fractional component grows adaptively on larger cards.
            vram_reserve_bytes: 96 * MIB,
            vram_reserve_fraction: 0.02,
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
            (0.0..=0.90).contains(&self.ram_reserve_fraction),
            "invalid RAM reserve fraction"
        );
        ensure!(
            (4096..=64 * MIB as usize).contains(&self.chunk_bytes),
            "chunk-bytes must be 4096..67108864"
        );
        ensure!(
            matches!(self.data_transport.as_str(), "tcp" | "udp"),
            "data-transport must be tcp or udp"
        );
        ensure!(
            (512..=1200).contains(&self.udp_payload_bytes),
            "udp-payload-bytes must be 512..1200"
        );
        ensure!(
            (2..=256).contains(&self.udp_window_packets),
            "udp-window-packets must be 2..256"
        );
        ensure!(
            (10..=5000).contains(&self.udp_initial_rto_ms),
            "udp-initial-rto-ms must be 10..5000"
        );
        ensure!(
            (1..=100_000).contains(&self.udp_pacing_micros),
            "udp-pacing-micros must be 1..100000"
        );
        ensure!(
            (1..=64).contains(&self.udp_max_retries),
            "udp-max-retries must be 1..64"
        );
        ensure!(
            (1..=4096).contains(&self.udp_max_sessions),
            "udp-max-sessions must be 1..4096"
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
