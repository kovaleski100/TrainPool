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
                    "ram-limit-bytes" => config.ram_limit_bytes = Some(value.parse()?),
                    "cluster-name" => config.cluster_name = value,
                    "cluster-secret" => config.cluster_secret = Some(value),
                    "chunk-bytes" => config.chunk_bytes = value.parse()?,
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
    let request = match &cli.command {
        Command::Topology => Request::Topology,
        Command::Metrics => Request::Metrics,
        Command::Benchmark { mib } => Request::Benchmark {
            bytes: mib
                .checked_mul(1024 * 1024)
                .ok_or_else(|| anyhow::anyhow!("benchmark size overflow"))?,
        },
        Command::Plan | Command::Run { .. } => Request::Plan { stages: vec![] },
        _ => Request::Status,
    };
    let response = transport.control(address, &request).await?;
    response.check()?;
    if let Command::Run { argv } = cli.command {
        let mut command = tokio::process::Command::new(&argv[0]);
        command
            .args(&argv[1..])
            .env("TRAINPOOL_ADDRESS", address.to_string())
            .env("TRAINPOOL_CLUSTER_NAME", &config.cluster_name)
            .env(
                "TRAINPOOL_JOB_ID",
                response.data["job_id"].as_str().unwrap_or(""),
            );
        if let Some(secret) = &config.cluster_secret {
            command.env("TRAINPOOL_CLUSTER_SECRET", secret);
        }
        let status = command.status().await?;
        ensure!(status.success(), "training process failed: {status}");
        return Ok(());
    }
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
                    .map(|g| format!("{} / {}", g.model, gib(g.vram_total)))
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
                gib(node.memory.physical_ram_total),
                gib(node.memory.trainpool_ram_available),
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
            gib(status.leadership.election_score),
            status.leadership.leader_epoch
        );
    }
    if matches!(cli.command, Command::Memory | Command::Status) {
        let m = status.logical_memory;
        for (label, value) in [
            ("Physical GPU VRAM", m.physical_vram),
            ("Currently free GPU VRAM", m.free_vram),
            ("Usable GPU VRAM", m.usable_vram),
            ("TrainPool allocated VRAM (SDK reports)", m.allocated_vram),
            ("Physical system RAM", m.physical_ram),
            ("Pool RAM budget", m.pool_ram_budget),
            ("Allocated pool RAM", m.allocated_ram),
            ("Currently allocatable pool RAM", m.pool_ram_allocatable),
            (
                "Logical training capacity available",
                m.logical_training_capacity,
            ),
        ] {
            println!("{label:<39} {}", gib(value));
        }
        println!(
            "Disk spill: DISABLED\nLogical capacity combines distinct RAM and VRAM tiers; it is not one CUDA allocation or VRAM-speed memory."
        );
    }
    Ok(())
}
fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}
