use crate::{
    config::Config,
    node::NodeCapabilities,
    protocol::VERSION,
    transport::{signature, verify},
};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use socket2::{Domain, Protocol, Socket, Type};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4};
use tokio::net::UdpSocket;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Announcement {
    pub protocol_version: u32,
    pub trainpool_version: String,
    pub cluster_name: String,
    pub node_id: Uuid,
    pub hostname: String,
    pub os: String,
    pub architecture: String,
    pub control_address: std::net::IpAddr,
    pub control_port: u16,
    pub leader_election_score: u64,
    pub timestamp: u64,
}
impl Announcement {
    pub fn new(node: &NodeCapabilities, config: &Config) -> Self {
        Self {
            protocol_version: VERSION,
            trainpool_version: env!("CARGO_PKG_VERSION").into(),
            cluster_name: config.cluster_name.clone(),
            node_id: node.node_id,
            hostname: node.hostname.clone(),
            os: node.os.clone(),
            architecture: node.architecture.clone(),
            control_address: node.network.control_address.ip(),
            control_port: node.network.control_address.port(),
            leader_election_score: node.election_score(),
            timestamp: crate::now_ms(),
        }
    }
}
#[derive(Serialize, Deserialize)]
struct SignedAnnouncement {
    payload: String,
    mac: Option<String>,
}
#[async_trait]
pub trait Discovery: Send + Sync {
    async fn advertise(&self, announcement: &Announcement) -> Result<()>;
    async fn receive(&self) -> Result<Announcement>;
}
pub struct MulticastDiscovery {
    socket: UdpSocket,
    config: Config,
}
impl MulticastDiscovery {
    pub fn bind(config: &Config) -> Result<Self> {
        let socket = Socket::new(Domain::IPV4, Type::DGRAM, Some(Protocol::UDP))?;
        socket.set_reuse_address(true)?;
        #[cfg(unix)]
        socket.set_reuse_port(true)?;
        socket.bind(&SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, config.discovery_port).into())?;
        socket.join_multicast_v4(&config.multicast_group, &config.multicast_interface)?;
        socket.set_multicast_if_v4(&config.multicast_interface)?;
        socket.set_multicast_loop_v4(true)?;
        socket.set_multicast_ttl_v4(1)?;
        socket.set_nonblocking(true)?;
        Ok(Self {
            socket: UdpSocket::from_std(socket.into())?,
            config: config.clone(),
        })
    }
}
#[async_trait]
impl Discovery for MulticastDiscovery {
    async fn advertise(&self, announcement: &Announcement) -> Result<()> {
        let payload = serde_json::to_string(announcement)?;
        let packet = SignedAnnouncement {
            mac: signature(self.config.cluster_secret.as_deref(), payload.as_bytes()),
            payload,
        };
        self.socket
            .send_to(
                &serde_json::to_vec(&packet)?,
                SocketAddrV4::new(self.config.multicast_group, self.config.discovery_port),
            )
            .await?;
        Ok(())
    }
    async fn receive(&self) -> Result<Announcement> {
        let mut bytes = [0_u8; 4096];
        let (len, _) = self.socket.recv_from(&mut bytes).await?;
        let signed: SignedAnnouncement = serde_json::from_slice(&bytes[..len])?;
        let announcement: Announcement = serde_json::from_str(&signed.payload)?;
        ensure!(
            announcement.cluster_name == self.config.cluster_name,
            "different cluster"
        );
        ensure!(
            announcement.protocol_version == VERSION,
            "TRAINPOOL_PROTOCOL_VERSION: incompatible discovery packet"
        );
        ensure!(
            crate::now_ms().abs_diff(announcement.timestamp) < 30_000,
            "stale discovery packet (check clock synchronization)"
        );
        verify(
            self.config.cluster_secret.as_deref(),
            signed.payload.as_bytes(),
            signed.mac.as_deref(),
        )?;
        ensure!(
            !announcement.control_address.is_unspecified() && announcement.control_port != 0,
            "invalid peer address"
        );
        Ok(announcement)
    }
}

pub fn advertised_address(config: &Config) -> Result<SocketAddr> {
    if let Some(ip) = config.advertise_ip {
        return Ok(SocketAddr::new(ip.into(), config.listen.port()));
    }
    if !config.listen.ip().is_unspecified() {
        return Ok(config.listen);
    }
    // UDP connect selects an interface without sending packets to the destination.
    let socket = std::net::UdpSocket::bind("0.0.0.0:0")?;
    let route =
        config.seeds.first().copied().unwrap_or_else(|| {
            SocketAddr::new(config.multicast_group.into(), config.discovery_port)
        });
    socket.connect(route)?;
    Ok(SocketAddr::new(
        socket.local_addr()?.ip(),
        config.listen.port(),
    ))
}
