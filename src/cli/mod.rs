use crate::{
    config::{Config, default_dir, node_id},
    protocol::{ClusterStatus, Request},
    transport::{TcpTransport, Transport},
};
use anyhow::{Result, ensure};
use clap::{Parser, Subcommand};
use std::{
    ffi::OsString,
    net::{Ipv4Addr, SocketAddr},
    path::PathBuf,
};

#[derive(Parser)]
#[command(version, about = "Capacity-first distributed logical training memory")]
pub struct Cli {
    #[arg(long, global = true, default_value_os_t = default_dir())]
    pub data_dir: PathBuf,
    #[arg(long, global = true, env = "TRAINPOOL_ADDRESS")]
    pub address: Option<SocketAddr>,
    #[arg(long, global = true)]
    pub json: bool,
    #[command(subcommand)]
    pub command: Command,
}
#[derive(Subcommand)]
pub enum Command {
    Daemon {
        #[arg(long)]
        listen: Option<SocketAddr>,
        #[arg(long)]
        advertise_ip: Option<Ipv4Addr>,
        #[arg(long)]
        multicast_interface: Option<Ipv4Addr>,
        #[arg(long)]
        discovery_port: Option<u16>,
        #[arg(long)]
        no_discovery: bool,
        #[arg(long)]
        seed: Vec<SocketAddr>,
        #[arg(long)]
        ram_limit_mib: Option<u64>,
        #[arg(long, value_parser = ["tcp", "udp"])]
        data_transport: Option<String>,
        #[arg(long)]
        enable_disk_spill: bool,
    },
    Nodes,
    Status,
    Leader,
    Memory,
    Topology,
    Metrics,
    Benchmark {
        #[arg(long, default_value_t = 16)]
        mib: usize,
    },
    Plan,
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Run {
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        argv: Vec<OsString>,
    },
    #[command(external_subcommand)]
    External(Vec<OsString>),
}
#[derive(Subcommand)]
pub enum ConfigCommand {
    Set { key: String, value: String },
    Show,
}

pub async fn execute(cli: Cli) -> Result<()> {
    let mut config = Config::load(&cli.data_dir)?;
    if let Ok(cluster) = std::env::var("TRAINPOOL_CLUSTER_NAME") {
        config.cluster_name = cluster;
    }
    if let Command::Config { command } = cli.command {
        match command {
            ConfigCommand::Show => {
                config.cluster_secret = config.cluster_secret.map(|_| "<redacted>".into());
                println!("{}", toml::to_string_pretty(&config)?);
            }
            ConfigCommand::Set { key, value } => {
                match key.as_str() {
                    "ram-fraction" => config.ram_fraction = value.parse()?,
                    "ram-reserve-bytes" => config.ram_reserve_bytes = value.parse()?,
                    "ram-reserve-fraction" => config.ram_reserve_fraction = value.parse()?,
                    "ram-limit-bytes" => config.ram_limit_bytes = Some(value.parse()?),
                    "cluster-name" => config.cluster_name = value,
                    "cluster-secret" => config.cluster_secret = Some(value),
                    "chunk-bytes" => config.chunk_bytes = value.parse()?,
                    "data-transport" => config.data_transport = value,
                    "udp-payload-bytes" => config.udp_payload_bytes = value.parse()?,
                    "udp-window-packets" => config.udp_window_packets = value.parse()?,
                    "udp-initial-rto-ms" => config.udp_initial_rto_ms = value.parse()?,
                    "udp-pacing-micros" => config.udp_pacing_micros = value.parse()?,
                    "udp-max-retries" => config.udp_max_retries = value.parse()?,
                    "lease-seconds" => config.lease_seconds = value.parse()?,
                    "vram-reserve-bytes" => config.vram_reserve_bytes = value.parse()?,
                    "vram-reserve-fraction" => config.vram_reserve_fraction = value.parse()?,
                    "listen" => config.listen = value.parse()?,
                    _ => anyhow::bail!("unknown configuration key: {key}"),
                }
                config.save(&cli.data_dir)?;
                println!("Configuration saved. Restart the daemon to apply it.");
            }
        }
        return Ok(());
    }
    if let Command::Daemon {
        listen,
        advertise_ip,
        multicast_interface,
        discovery_port,
        no_discovery,
        seed,
        ram_limit_mib,
        data_transport,
        enable_disk_spill,
    } = cli.command
    {
        ensure!(
            !enable_disk_spill,
            "experimental disk spill: not implemented"
        );
        if let Some(address) = listen {
            config.listen = address;
        }
        if advertise_ip.is_some() {
            config.advertise_ip = advertise_ip;
        }
        if let Some(interface) = multicast_interface {
            config.multicast_interface = interface;
        }
        if let Some(port) = discovery_port {
            config.discovery_port = port;
        }
        if no_discovery {
            config.discovery_enabled = false;
        }
        config.seeds.extend(seed);
        if let Some(limit) = ram_limit_mib {
            config.ram_limit_bytes = Some(
                limit
                    .checked_mul(1024 * 1024)
                    .ok_or_else(|| anyhow::anyhow!("RAM limit overflow"))?,
            );
        }
        if let Some(transport) = data_transport {
            config.data_transport = transport;
        }
        config.validate()?;
        return crate::runtime::daemon(config, node_id(&cli.data_dir)?).await;
    }
    let address = cli.address.unwrap_or_else(|| {
        if config.listen.ip().is_unspecified() {
            SocketAddr::new(Ipv4Addr::LOCALHOST.into(), config.listen.port())
        } else {
            config.listen
        }
    });
    let transport = TcpTransport {
        config: config.clone(),
    };
    if let Command::Run { argv } | Command::External(argv) = &cli.command {
        let code =
            crate::launcher::launch(argv.clone(), config.clone(), &cli.data_dir, address).await?;
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }
    let request = match &cli.command {
        Command::Topology => Request::Topology,
        Command::Metrics => Request::Metrics,
        Command::Benchmark { mib } => Request::Benchmark {
            bytes: mib
                .checked_mul(1024 * 1024)
                .ok_or_else(|| anyhow::anyhow!("benchmark size overflow"))?,
        },
        Command::Plan => Request::Plan { stages: vec![] },
        Command::Run { .. } | Command::External(_) => unreachable!("launches were handled above"),
        _ => Request::Status,
    };
    let response = transport.control(address, &request).await?;
    response.check()?;
    if cli.json || matches!(cli.command, Command::Metrics | Command::Plan) {
        println!("{}", serde_json::to_string_pretty(&response.data)?);
        return Ok(());
    }
    if matches!(cli.command, Command::Topology | Command::Benchmark { .. }) {
        let topology: crate::scheduler::topology::Topology = serde_json::from_value(response.data)?;
        println!(
            "SOURCE                               DESTINATION                          LATENCY    BANDWIDTH"
        );
        for link in topology.links {
            println!(
                "{} {} {:8.2} ms  {}",
                link.source,
                link.destination,
                link.latency_ms,
                link.bytes_per_second
                    .map(|b| format!("{:.1} Mbps", b * 8.0 / 1e6))
                    .unwrap_or_else(|| "unmeasured (run benchmark)".into())
            );
        }
        return Ok(());
    }
    let status: ClusterStatus = serde_json::from_value(response.data)?;
    if matches!(cli.command, Command::Nodes | Command::Status) {
        println!("NODE             ROLE    CPU     RAM         POOL RAM    GPU / PHYSICAL VRAM");
        for node in &status.nodes {
            let gpu = if node.gpus.is_empty() {
                "none (RAM provider)".into()
            } else {
                node.gpus
                    .iter()
                    .map(|g| format!("{} / {}", g.model, format_bytes(g.vram_total)))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            println!(
                "{:<16} {:<7} {:>3}c    {:<11} {:<11} {}",
                node.hostname,
                if Some(node.node_id) == status.leadership.leader_id {
                    "leader"
                } else {
                    "peer"
                },
                node.cpu.logical_cores,
                format_bytes(node.memory.physical_ram_total),
                format_bytes(node.memory.trainpool_ram_available),
                gpu
            );
            println!("  {}  {}", node.node_id, node.network.control_address);
        }
    }
    if matches!(cli.command, Command::Leader | Command::Status) {
        println!(
            "Leader: {}",
            status
                .leadership
                .leader_id
                .map(|id| id.to_string())
                .unwrap_or_else(|| "none".into())
        );
        println!(
            "Election score: {}\nReason: highest total physical RAM + physical VRAM; largest UUID breaks ties\nLeader epoch: {}",
            format_bytes(status.leadership.election_score),
            status.leadership.leader_epoch
        );
    }
    if matches!(cli.command, Command::Memory | Command::Status) {
        let m = status.logical_memory;
        for (label, value) in [
            ("Cluster physical VRAM (inventory)", m.cluster_physical_vram),
            ("GPU driver free (daemon snapshot)", m.free_vram),
            (
                "GPU driver used (daemon snapshot)",
                m.physical_vram.saturating_sub(m.free_vram),
            ),
            ("Cluster usable VRAM (inventory)", m.cluster_usable_vram),
            ("Primary GPU physical VRAM", m.primary_gpu_physical_vram),
            ("Primary GPU usable ceiling", m.primary_gpu_usable_vram),
            ("Job backing capacity", m.current_job_backing_capacity),
            ("Physical system RAM", m.physical_ram),
            ("Pool RAM budget", m.pool_ram_budget),
            ("Allocated pool RAM", m.allocated_ram),
            ("Currently allocatable pool RAM", m.pool_ram_allocatable),
            (
                "Job remaining logical capacity",
                m.logical_training_capacity,
            ),
        ] {
            println!("{label:<39} {}", format_bytes(value));
        }
        println!("\nLOCAL COMPUTE-NODE RAM");
        for (label, value) in [
            ("Physical RAM local", m.local_physical_ram),
            ("OS available RAM local", m.local_os_available_ram),
            ("TrainPool local RAM budget", m.local_pool_ram_budget),
            ("TrainPool local RAM used", m.local_pool_ram_used),
            ("Local RAM safety reserve", m.local_ram_safety_reserve),
            (
                "TrainPool local RAM allocatable now",
                m.local_pool_ram_allocatable,
            ),
            ("TrainPool remote RAM used", m.remote_pool_ram_used),
        ] {
            println!("{label:<39} {}", format_bytes(value));
        }
        println!("\nPYTORCH SDK CUDA SNAPSHOT");
        let age = m.sdk_metrics_age_ms;
        for (label, value) in [
            ("GPU physical VRAM", m.sdk_physical_vram_bytes),
            ("GPU driver used", m.driver_used_vram_bytes),
            ("GPU driver free", m.driver_free_vram_bytes),
            ("PyTorch allocated", m.torch_allocated_bytes),
            ("PyTorch reserved", m.torch_reserved_bytes),
            ("PyTorch reclaimable", m.torch_reclaimable_bytes),
            (
                "TrainPool resident handles",
                m.trainpool_resident_vram_bytes,
            ),
            ("Safe VRAM allocatable now", m.safe_vram_allocatable_bytes),
            (
                "Configured usable VRAM ceiling",
                m.configured_usable_vram_ceiling_bytes,
            ),
            (
                "Configured VRAM safety reserve",
                m.configured_vram_safety_reserve_bytes,
            ),
        ] {
            println!("{label:<39} {}", format_sdk_metric(value, age));
        }
        println!(
            "Disk spill: DISABLED\nLogical capacity combines distinct RAM and VRAM tiers; it is not one CUDA allocation or VRAM-speed memory."
        );
    }
    Ok(())
}

fn format_sdk_metric(value: Option<u64>, age_ms: Option<u64>) -> String {
    match (value, age_ms) {
        (Some(_), Some(age)) if age > 5_000 => {
            format!("STALE (age {:.2} s)", age as f64 / 1000.0)
        }
        (Some(value), Some(age)) => format!("{} (age {age} ms)", format_bytes(value)),
        _ => "UNKNOWN".into(),
    }
}
fn format_bytes(bytes: u64) -> String {
    const UNITS: [(&str, u64); 4] = [
        ("TB", 1_000_000_000_000),
        ("GB", 1_000_000_000),
        ("MB", 1_000_000),
        ("KB", 1_000),
    ];
    for (unit, divisor) in UNITS {
        if bytes >= divisor {
            return format!("{:.2} {unit}", bytes as f64 / divisor as f64);
        }
    }
    format!("{bytes} B")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn external_program_is_captured_verbatim() {
        let cli =
            Cli::try_parse_from(["trainpool", "python", "train.py", "--epochs", "100"]).unwrap();
        let Command::External(argv) = cli.command else {
            panic!("expected external command");
        };
        assert_eq!(
            argv,
            ["python", "train.py", "--epochs", "100"].map(OsString::from)
        );
    }

    #[test]
    fn built_in_command_still_wins() {
        let cli = Cli::try_parse_from(["trainpool", "nodes"]).unwrap();
        assert!(matches!(cli.command, Command::Nodes));
    }

    #[test]
    fn legacy_run_remains_available() {
        let cli = Cli::try_parse_from(["trainpool", "run", "--", "python", "train.py"]).unwrap();
        assert!(matches!(cli.command, Command::Run { .. }));
    }

    #[test]
    fn byte_sizes_use_familiar_decimal_units() {
        assert_eq!(format_bytes(999), "999 B");
        assert_eq!(format_bytes(1_000), "1.00 KB");
        assert_eq!(format_bytes(1_500_000), "1.50 MB");
        assert_eq!(format_bytes(16_659_828_736), "16.66 GB");
        assert_eq!(format_bytes(2_500_000_000_000), "2.50 TB");
    }

    #[test]
    fn stale_sdk_metrics_are_never_rendered_as_zero() {
        assert_eq!(format_sdk_metric(None, None), "UNKNOWN");
        assert_eq!(
            format_sdk_metric(Some(0), Some(12_400)),
            "STALE (age 12.40 s)"
        );
        assert_eq!(format_sdk_metric(Some(0), Some(320)), "0 B (age 320 ms)");
    }
}
