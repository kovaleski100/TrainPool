use crate::{
    memory::block::MemoryBlockHandle,
    protocol::{Request, Response},
    transport::{TcpTransport, Transport, read_frame, write_frame},
};
use anyhow::{Result, ensure};
use std::net::SocketAddr;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};
use uuid::Uuid;

pub(crate) async fn write_chunk_to(
    stream: &mut TcpStream,
    handle: &MemoryBlockHandle,
    transfer_id: Uuid,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    write_frame(
        stream,
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
    read_frame::<Response>(stream).await?.check()?;
    stream.write_all(bytes).await?;
    read_frame::<Response>(stream).await?.check()
}

pub(crate) async fn read_chunk_from(
    stream: &mut TcpStream,
    handle: &MemoryBlockHandle,
    transfer_id: Uuid,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>> {
    write_frame(
        stream,
        &Request::ReadChunk {
            handle: handle.clone(),
            transfer_id,
            offset,
            length,
            direct: true,
        },
    )
    .await?;
    let metadata: serde_json::Value = read_frame::<Response>(stream).await?.into_data()?;
    let mut bytes = vec![0; length];
    stream.read_exact(&mut bytes).await?;
    ensure!(
        metadata["checksum"].as_str() == Some(blake3::hash(&bytes).to_hex().as_str()),
        "TRAINPOOL_CHECKSUM_MISMATCH"
    );
    Ok(bytes)
}

pub async fn write_chunk(
    transport: &TcpTransport,
    address: SocketAddr,
    handle: &MemoryBlockHandle,
    transfer_id: Uuid,
    offset: u64,
    bytes: &[u8],
) -> Result<()> {
    let mut stream = transport.connect(address).await?;
    write_chunk_to(&mut stream, handle, transfer_id, offset, bytes).await
}
pub async fn read_chunk(
    transport: &TcpTransport,
    address: SocketAddr,
    handle: &MemoryBlockHandle,
    offset: u64,
    length: usize,
) -> Result<Vec<u8>> {
    let mut stream = transport.connect(address).await?;
    read_chunk_from(&mut stream, handle, Uuid::new_v4(), offset, length).await
}
