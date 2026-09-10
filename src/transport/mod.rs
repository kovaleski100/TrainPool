use crate::{
    config::Config,
    protocol::{
        Authenticated, Authentication, Challenge, MAX_FRAME, Request, Response, VERSION, Wire,
    },
};
use anyhow::{Result, ensure};
use async_trait::async_trait;
use hmac::{Hmac, Mac};
use serde::{Serialize, de::DeserializeOwned};
use sha2::Sha256;
use std::{net::SocketAddr, time::Duration};
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt},
    net::TcpStream,
};
use uuid::Uuid;

pub async fn write_frame<T: Serialize>(
    stream: &mut (impl AsyncWrite + Unpin),
    message: &T,
) -> Result<()> {
    let bytes = serde_json::to_vec(&Wire {
        protocol_version: VERSION,
        message,
    })?;
    ensure!(bytes.len() <= MAX_FRAME, "control frame exceeds limit");
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;
    Ok(())
}
pub async fn read_frame<T: DeserializeOwned>(stream: &mut (impl AsyncRead + Unpin)) -> Result<T> {
    let size = stream.read_u32().await? as usize;
    ensure!(size <= MAX_FRAME, "control frame exceeds limit");
    let mut bytes = vec![0; size];
    stream.read_exact(&mut bytes).await?;
    // Inspect version before attempting to decode version-specific fields.
    let value: Wire<serde_json::Value> = serde_json::from_slice(&bytes)?;
    ensure!(
        value.protocol_version == VERSION,
        "TRAINPOOL_PROTOCOL_VERSION: expected {VERSION}, received {}",
        value.protocol_version
    );
    Ok(serde_json::from_value(value.message)?)
}
pub fn signature(secret: Option<&str>, bytes: &[u8]) -> Option<String> {
    secret.map(|s| {
        let mut mac = Hmac::<Sha256>::new_from_slice(s.as_bytes()).expect("HMAC accepts any key");
        mac.update(bytes);
        hex::encode(mac.finalize().into_bytes())
    })
}
pub fn verify(secret: Option<&str>, bytes: &[u8], supplied: Option<&str>) -> Result<()> {
    match (secret, supplied) {
        (None, None) => Ok(()),
        (Some(key), Some(tag)) => {
            let mut mac = Hmac::<Sha256>::new_from_slice(key.as_bytes())?;
            mac.update(bytes);
            mac.verify_slice(&hex::decode(tag)?)
                .map_err(|_| anyhow::anyhow!("TRAINPOOL_AUTH_FAILED"))
        }
        _ => anyhow::bail!("TRAINPOOL_AUTH_FAILED: cluster secret mismatch"),
    }
}
fn transcript(role: &str, server: Uuid, client: Uuid, cluster: &str) -> String {
    format!("{role}|{server}|{client}|{cluster}")
}
pub async fn server_auth(stream: &mut TcpStream, config: &Config) -> Result<()> {
    let server = Uuid::new_v4();
    write_frame(stream, &Challenge { nonce: server }).await?;
    let auth: Authentication = read_frame(stream).await?;
    ensure!(
        auth.cluster_name == config.cluster_name,
        "TRAINPOOL_CLUSTER_MISMATCH"
    );
    verify(
        config.cluster_secret.as_deref(),
        transcript("client", server, auth.nonce, &config.cluster_name).as_bytes(),
        auth.mac.as_deref(),
    )?;
    write_frame(
        stream,
        &Authenticated {
            mac: signature(
                config.cluster_secret.as_deref(),
                transcript("server", server, auth.nonce, &config.cluster_name).as_bytes(),
            ),
        },
    )
    .await
}

#[async_trait]
pub trait Transport: Send + Sync {
    type Stream: AsyncRead + AsyncWrite + Unpin + Send;
    async fn connect(&self, address: SocketAddr) -> Result<Self::Stream>;
    async fn control(&self, address: SocketAddr, request: &Request) -> Result<Response>;
}
#[derive(Clone)]
pub struct TcpTransport {
    pub config: Config,
}
#[async_trait]
impl Transport for TcpTransport {
    type Stream = TcpStream;
    async fn connect(&self, address: SocketAddr) -> Result<TcpStream> {
        // Authentication can contend with model execution and batched lease
        // renewal on CPU-only workers.  Five seconds was short enough for a
        // healthy peer to be mistaken for a failed one during DeepLab stress.
        tokio::time::timeout(Duration::from_secs(30), async {
            let mut stream = TcpStream::connect(address).await?;
            stream.set_nodelay(true)?;
            let challenge: Challenge = read_frame(&mut stream).await?;
            let client = Uuid::new_v4();
            write_frame(
                &mut stream,
                &Authentication {
                    cluster_name: self.config.cluster_name.clone(),
                    nonce: client,
                    mac: signature(
                        self.config.cluster_secret.as_deref(),
                        transcript("client", challenge.nonce, client, &self.config.cluster_name)
                            .as_bytes(),
                    ),
                },
            )
            .await?;
            let accepted: Authenticated = read_frame(&mut stream).await?;
            verify(
                self.config.cluster_secret.as_deref(),
                transcript("server", challenge.nonce, client, &self.config.cluster_name).as_bytes(),
                accepted.mac.as_deref(),
            )?;
            Ok(stream)
        })
        .await?
    }
    async fn control(&self, address: SocketAddr, request: &Request) -> Result<Response> {
        tokio::time::timeout(Duration::from_secs(120), async {
            let mut stream = self.connect(address).await?;
            write_frame(&mut stream, request).await?;
            read_frame(&mut stream).await
        })
        .await?
    }
}
