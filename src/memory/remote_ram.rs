use crate::{
    memory::block::MemoryBlockHandle,
    protocol::{Request, Response},
    transport::{TcpTransport, Transport, read_frame, write_frame},
};
use anyhow::{Result, ensure};
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use uuid::Uuid;

pub async fn write_chunk(
    transport: &TcpTransport,
    address: SocketAddr,
    handle: &MemoryBlockHandle,
    transfer_id: Uuid,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    let mut stream = transport.connect(address).await?;
    write_frame(
        &mut stream,
        &Request::WriteChunk {
            handle: handle.clone(),
            transfer_id,
            offset,
            length: bytes.len(),
            total_size: handle.size,
            checksum: blake3::hash(bytes).to_hex().to_string(),
            direct: true,
        },
    )
    .await?;
    read_frame::<Response>(&mut stream).await?.check()?;
    stream.write_all(bytes).await?;
    read_frame::<Response>(&mut stream).await?.check()
}
pub async fn read_chunk(
    transport: &TcpTransport,
    address: SocketAddr,
    handle: &MemoryBlockHandle,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>> {
    let mut stream = transport.connect(address).await?;
    write_frame(
        &mut stream,
        &Request::ReadChunk {
            handle: handle.clone(),
            transfer_id: Uuid::new_v4(),
            offset,
            length,
            direct: true,
        },
    )
    .await?;
    let metadata: serde_json::Value = read_frame::<Response>(&mut stream).await?.into_data()?;
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes).await?;
    ensure!(
        metadata["checksum"].as_str() == Some(blake3::hash(&bytes).to_hex().as_str()),
        "TRAINPOOL_CHECKSUM_MISMATCH"
    );
    Ok(bytes)
}
