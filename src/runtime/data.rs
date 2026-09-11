use super::Runtime;
use crate::{
    memory::block::BlockState,
    protocol::{Request, Response},
    transport::{Transport, read_frame, write_frame},
};
use anyhow::{Result, ensure};
use std::net::SocketAddr;
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    net::TcpStream,
};

impl Runtime {
    async fn acquire_tcp_data(&self, address: SocketAddr) -> Result<(TcpStream, bool)> {
        if let Some(stream) = self
            .tcp_data_pool
            .lock()
            .await
            .get_mut(&address)
            .and_then(Vec::pop)
        {
            return Ok((stream, true));
        }
        Ok((self.transport.connect(address).await?, false))
    }

    async fn release_tcp_data(&self, address: SocketAddr, stream: TcpStream) {
        let mut pool = self.tcp_data_pool.lock().await;
        let connections = pool.entry(address).or_default();
        if connections.len() < 8 {
            connections.push(stream);
        }
    }

    pub(crate) async fn tcp_write_chunk(
        &self,
        address: SocketAddr,
        handle: &crate::memory::block::MemoryBlockHandle,
        transfer_id: uuid::Uuid,
        offset: u64,
        payload: &[u8],
    ) -> Result<()> {
        let (mut stream, reused) = self.acquire_tcp_data(address).await?;
        {
            let mut all = self.metrics.lock().await;
            let metrics = all.job(handle.job_id);
            metrics.tcp_connections_opened += u64::from(!reused);
            metrics.tcp_connections_reused += u64::from(reused);
        }
        let result = async {
            write_frame(
                &mut stream,
                &Request::WriteChunk {
                    handle: handle.clone(),
                    transfer_id,
                    offset,
                    length: payload.len(),
                    total_size: handle.size,
                    checksum: blake3::hash(payload).to_hex().to_string(),
                    direct: true,
                },
            )
            .await?;
            read_frame::<Response>(&mut stream).await?.check()?;
            stream.write_all(payload).await?;
            read_frame::<Response>(&mut stream).await?.check()
        }
        .await;
        if result.is_ok() {
            self.release_tcp_data(address, stream).await;
        }
        result
    }

    pub(crate) async fn tcp_read_chunk(
        &self,
        address: SocketAddr,
        handle: &crate::memory::block::MemoryBlockHandle,
        transfer_id: uuid::Uuid,
        offset: u64,
        length: usize,
    ) -> Result<Vec<u8>> {
        let (mut stream, reused) = self.acquire_tcp_data(address).await?;
        {
            let mut all = self.metrics.lock().await;
            let metrics = all.job(handle.job_id);
            metrics.tcp_connections_opened += u64::from(!reused);
            metrics.tcp_connections_reused += u64::from(reused);
        }
        let result = async {
            write_frame(
                &mut stream,
                &Request::ReadChunk {
                    handle: handle.clone(),
                    transfer_id,
                    offset,
                    length,
                    direct: true,
                },
            )
            .await?;
            let metadata: serde_json::Value =
                read_frame::<Response>(&mut stream).await?.into_data()?;
            let mut payload = vec![0; length];
            stream.read_exact(&mut payload).await?;
            ensure!(
                metadata["checksum"].as_str() == Some(blake3::hash(&payload).to_hex().as_str()),
                "TRAINPOOL_CHECKSUM_MISMATCH"
            );
            Ok(payload)
        }
        .await;
        if result.is_ok() {
            self.release_tcp_data(address, stream).await;
        }
        result
    }

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
                    // The SDK-to-local-daemon hop remains TCP. Inter-node payload
                    // forwarding uses the selected data plane, while local owners
                    // write directly into LocalRam without loopback networking.
                    let reservation = self.ram.accounting.reserve(length as u64)?;
                    write_frame(stream, &Response::data(())).await?;
                    let mut payload = vec![0; length];
                    stream.read_exact(&mut payload).await?;
                    ensure!(
                        blake3::hash(&payload).to_hex().as_str() == checksum,
                        "TRAINPOOL_CHECKSUM_MISMATCH"
                    );
                    if h.owner_node == self.node_id {
                        let block = self.ram.get(h.id).await?;
                        let mut b = block.lock().await;
                        b.authorize(h.lease_token)?;
                        ensure!(
                            b.handle.state == BlockState::Writing
                                && b.handle.generation == h.generation
                                && b.written as u64 == offset,
                            "block is immutable, stale or non-sequential"
                        );
                        let begin = offset as usize;
                        let end = begin + length;
                        b.bytes[begin..end].copy_from_slice(&payload);
                        b.hash.update(&payload);
                        b.transfer_id = Some(transfer_id);
                        b.written = end;
                    } else {
                        let start = std::time::Instant::now();
                        let node = self
                            .membership
                            .read()
                            .await
                            .peers
                            .get(&h.owner_node)
                            .map(|peer| peer.capabilities.clone())
                            .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_NODE_UNAVAILABLE"))?;
                        if self.config.data_transport == "udp"
                            && node.network.data_transport == "udp"
                        {
                            let address = node
                                .network
                                .data_address
                                .unwrap_or(node.network.control_address);
                            let mut all = self.metrics.lock().await;
                            let metrics = all.job(h.job_id);
                            metrics.transfer_sessions += 1;
                            metrics.active_transfer_sessions += 1;
                            drop(all);
                            let result = self
                                .reliable_udp()
                                .write(address, &h, transfer_id, offset, &payload)
                                .await;
                            let mut all = self.metrics.lock().await;
                            let metrics = all.job(h.job_id);
                            metrics.active_transfer_sessions =
                                metrics.active_transfer_sessions.saturating_sub(1);
                            match result {
                                Ok(stats) => self.apply_udp_stats(metrics, &stats),
                                Err(error) => {
                                    metrics.failed_transfers += 1;
                                    return Err(error);
                                }
                            }
                        } else {
                            self.tcp_write_chunk(
                                node.network.control_address,
                                &h,
                                transfer_id,
                                offset,
                                &payload,
                            )
                            .await?;
                        }
                        let mut metrics = self.metrics.lock().await;
                        let m = metrics.job(h.job_id);
                        m.bytes_local_to_remote_ram += length as u64;
                        m.local_to_remote_bytes += length as u64;
                        m.network_bytes += length as u64;
                        m.network_wait_ms += start.elapsed().as_secs_f64() * 1000.0;
                    }
                    drop(reservation);
                    return write_frame(stream, &Response::data(())).await;
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
                    let reservation = self.ram.accounting.reserve(length as u64)?;
                    let start = std::time::Instant::now();
                    let payload = if h.owner_node == self.node_id {
                        let block = self.ram.get(h.id).await?;
                        let b = block.lock().await;
                        b.authorize(h.lease_token)?;
                        ensure!(
                            b.handle.state == BlockState::Ready
                                && b.handle.generation == h.generation,
                            "block not ready or handle stale"
                        );
                        let begin = offset as usize;
                        let end = begin + length;
                        ensure!(end <= b.bytes.len(), "read exceeds allocation");
                        b.bytes[begin..end].to_vec()
                    } else {
                        let node = self
                            .membership
                            .read()
                            .await
                            .peers
                            .get(&h.owner_node)
                            .map(|peer| peer.capabilities.clone())
                            .ok_or_else(|| anyhow::anyhow!("TRAINPOOL_NODE_UNAVAILABLE"))?;
                        let payload = if self.config.data_transport == "udp"
                            && node.network.data_transport == "udp"
                        {
                            let address = node
                                .network
                                .data_address
                                .unwrap_or(node.network.control_address);
                            let mut all = self.metrics.lock().await;
                            let metrics = all.job(h.job_id);
                            metrics.transfer_sessions += 1;
                            metrics.active_transfer_sessions += 1;
                            drop(all);
                            let result = self
                                .reliable_udp()
                                .read(address, &h, transfer_id, offset, length)
                                .await;
                            let mut all = self.metrics.lock().await;
                            let metrics = all.job(h.job_id);
                            metrics.active_transfer_sessions =
                                metrics.active_transfer_sessions.saturating_sub(1);
                            match result {
                                Ok((payload, stats)) => {
                                    self.apply_udp_stats(metrics, &stats);
                                    payload
                                }
                                Err(error) => {
                                    metrics.failed_transfers += 1;
                                    return Err(error);
                                }
                            }
                        } else {
                            self.tcp_read_chunk(
                                node.network.control_address,
                                &h,
                                transfer_id,
                                offset,
                                length,
                            )
                            .await?
                        };
                        let mut metrics = self.metrics.lock().await;
                        let m = metrics.job(h.job_id);
                        m.bytes_remote_ram_to_local += length as u64;
                        m.remote_to_local_bytes += length as u64;
                        m.network_bytes += length as u64;
                        m.network_wait_ms += start.elapsed().as_secs_f64() * 1000.0;
                        payload
                    };
                    let metadata = serde_json::json!({
                        "transfer_id": transfer_id,
                        "object_id": h.id,
                        "offset": offset,
                        "length": length,
                        "total_size": h.size,
                        "checksum": blake3::hash(&payload).to_hex().to_string(),
                    });
                    write_frame(stream, &Response::data(metadata)).await?;
                    stream.write_all(&payload).await?;
                    drop(reservation);
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
