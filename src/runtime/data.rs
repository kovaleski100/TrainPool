use super::Runtime;
use crate::{
    memory::block::BlockState,
    protocol::{Request, Response},
    transport::{Transport, read_frame, write_frame},
};
use anyhow::{Result, ensure};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

impl Runtime {
    pub async fn data(&self, stream: &mut TcpStream, request: Request) -> Result<()> {
        let _active = ActiveTransfer::new(&self.active_transfers);
        match request {
            Request::WriteChunk {
                handle,
                transfer_id,
                offset,
                length,
                total_size,
                checksum,
                direct,
            } => {
                ensure!(
                    length > 0 && length <= self.config.chunk_bytes,
                    "chunk size exceeds negotiated limit"
                );
                ensure!(
                    offset
                        .checked_add(length as u64)
                        .is_some_and(|end| end <= total_size)
                        && total_size == handle.size,
                    "invalid chunk range"
                );
                if !direct {
                    let h = self.resolve(&handle).await?;
                    let mut remote = self
                        .transport
                        .connect(self.address(h.owner_node).await?)
                        .await?;
                    write_frame(
                        &mut remote,
                        &Request::WriteChunk {
                            handle: h.clone(),
                            transfer_id,
                            offset,
                            length,
                            total_size,
                            checksum,
                            direct: true,
                        },
                    )
                    .await?;
                    let ready: Response = read_frame(&mut remote).await?;
                    ready.check()?;
                    // Relay uses a fixed 64 KiB reservation, never a whole tensor copy.
                    let reservation = self.ram.accounting.reserve(65536)?;
                    write_frame(stream, &ready).await?;
                    let start = std::time::Instant::now();
                    relay(stream, &mut remote, length).await?;
                    drop(reservation);
                    let response: Response = read_frame(&mut remote).await?;
                    if response.ok && h.owner_node != self.node_id {
                        let mut metrics = self.metrics.lock().await;
                        let m = metrics.job(h.job_id);
                        m.bytes_local_to_remote_ram += length as u64;
                        m.network_wait_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                    return write_frame(stream, &response).await;
                }
                let block = self.ram.get(handle.id).await?;
                let mut b = block.lock().await;
                b.authorize(handle.lease_token)?;
                ensure!(
                    b.handle.state == BlockState::Writing
                        && b.handle.generation == handle.generation,
                    "block is immutable or stale"
                );
                ensure!(
                    b.written as u64 == offset && b.handle.size == total_size,
                    "chunks must be sequential"
                );
                ensure!(
                    b.transfer_id.is_none_or(|id| id == transfer_id),
                    "transfer id mismatch"
                );
                write_frame(stream, &Response::data(())).await?;
                let begin = offset as usize;
                let end = begin + length;
                stream.read_exact(&mut b.bytes[begin..end]).await?;
                ensure!(
                    blake3::hash(&b.bytes[begin..end]).to_hex().as_str() == checksum,
                    "TRAINPOOL_CHECKSUM_MISMATCH"
                );
                let crate::memory::local_ram::RamBlock { bytes, hash, .. } = &mut *b;
                hash.update(&bytes[begin..end]);
                b.transfer_id = Some(transfer_id);
                b.written = end;
                self.metrics.lock().await.job(handle.job_id).network_bytes += length as u64;
                write_frame(stream, &Response::data(())).await
            }
            Request::ReadChunk {
                handle,
                transfer_id,
                offset,
                length,
                direct,
            } => {
                ensure!(
                    length > 0 && length <= self.config.chunk_bytes,
                    "chunk size exceeds negotiated limit"
                );
                ensure!(
                    offset
                        .checked_add(length as u64)
                        .is_some_and(|end| end <= handle.size),
                    "invalid read range"
                );
                if !direct {
                    let h = self.resolve(&handle).await?;
                    let mut remote = self
                        .transport
                        .connect(self.address(h.owner_node).await?)
                        .await?;
                    write_frame(
                        &mut remote,
                        &Request::ReadChunk {
                            handle: h.clone(),
                            transfer_id,
                            offset,
                            length,
                            direct: true,
                        },
                    )
                    .await?;
                    let metadata: Response = read_frame(&mut remote).await?;
                    metadata.check()?;
                    let reservation = self.ram.accounting.reserve(65536)?;
                    write_frame(stream, &metadata).await?;
                    let start = std::time::Instant::now();
                    relay(&mut remote, stream, length).await?;
                    drop(reservation);
                    if h.owner_node != self.node_id {
                        let mut metrics = self.metrics.lock().await;
                        let m = metrics.job(h.job_id);
                        m.bytes_remote_ram_to_local += length as u64;
                        m.network_wait_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                    return Ok(());
                }
                let block = self.ram.get(handle.id).await?;
                let b = block.lock().await;
                b.authorize(handle.lease_token)?;
                ensure!(
                    b.handle.state == BlockState::Ready && b.handle.generation == handle.generation,
                    "block not ready or handle stale"
                );
                ensure!(
                    offset + length as u64 <= b.handle.size,
                    "read exceeds allocation"
                );
                let bytes = &b.bytes[offset as usize..offset as usize + length];
                write_frame(stream, &Response::data(serde_json::json!({ "transfer_id": transfer_id, "object_id": handle.id,
                    "offset": offset, "length": length, "total_size": b.handle.size, "checksum": blake3::hash(bytes).to_hex().to_string() }))).await?;
                stream.write_all(bytes).await?;
                self.metrics.lock().await.job(handle.job_id).network_bytes += length as u64;
                Ok(())
            }
            Request::Probe { bytes } => {
                ensure!(bytes <= 64 * 1024 * 1024, "probe capped at 64 MiB");
                write_frame(stream, &Response::data(bytes)).await?;
                let buffer = [0_u8; 8192];
                let mut remaining = bytes;
                while remaining > 0 {
                    let n = remaining.min(buffer.len());
                    stream.write_all(&buffer[..n]).await?;
                    remaining -= n;
                }
                Ok(())
            }
            _ => anyhow::bail!("not a data request"),
        }
    }
}
struct ActiveTransfer<'a>(&'a std::sync::atomic::AtomicU32);
impl<'a> ActiveTransfer<'a> {
    fn new(count: &'a std::sync::atomic::AtomicU32) -> Self {
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(count)
    }
}
impl Drop for ActiveTransfer<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    }
}
async fn relay(source: &mut TcpStream, destination: &mut TcpStream, length: usize) -> Result<()> {
    let mut buffer = vec![0_u8; 65536];
    let mut remaining = length;
    while remaining > 0 {
        let n = remaining.min(buffer.len());
        source.read_exact(&mut buffer[..n]).await?;
        destination.write_all(&buffer[..n]).await?;
        remaining -= n;
    }
    Ok(())
}
