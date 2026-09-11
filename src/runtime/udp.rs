use super::Runtime;
use crate::{
    memory::{allocator::Reservation, block::BlockState},
    transport::udp::{
        ACK_BITS, Kind, Packet, ReliableUdpDataTransport, SendPayload, bitmap_payload, send_payload,
    },
};
use anyhow::{Result, ensure};
use std::{
    collections::{HashMap, HashSet},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::net::UdpSocket;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct SessionKey {
    source: SocketAddr,
    transfer: Uuid,
    block: Uuid,
}

struct IncomingSession {
    object: Uuid,
    lease: Uuid,
    generation: u64,
    base_offset: u64,
    block_size: u64,
    payload_size: usize,
    packet_count: usize,
    checksum: [u8; 32],
    buffer: Vec<u8>,
    received: Vec<bool>,
    received_count: usize,
    last_sequence: Option<usize>,
    last_activity: Instant,
    _reservation: Reservation,
}

#[derive(Default)]
pub struct UdpServerState {
    incoming: tokio::sync::Mutex<HashMap<SessionKey, IncomingSession>>,
    finishing: tokio::sync::Mutex<HashSet<SessionKey>>,
    completed: tokio::sync::Mutex<HashMap<(SessionKey, u64), Instant>>,
    active_reads: tokio::sync::Mutex<usize>,
}

fn response(request: &Packet, kind: Kind, payload: Vec<u8>) -> Packet {
    Packet {
        kind,
        flags: request.flags,
        transfer_id: request.transfer_id,
        object_id: request.object_id,
        block_id: request.block_id,
        lease_token: request.lease_token,
        generation: request.generation,
        sequence_id: request.sequence_id,
        offset: request.offset,
        total_size: request.total_size,
        payload,
    }
}

fn completed_data_ack(request: &Packet, window: usize) -> Packet {
    let base = (request.sequence_id as usize / ACK_BITS) * ACK_BITS;
    response(
        request,
        Kind::Ack,
        bitmap_payload(base as u32, &vec![true; ACK_BITS], window),
    )
}

impl Runtime {
    pub async fn udp_datagram(
        self: &Arc<Self>,
        socket: Arc<UdpSocket>,
        datagram: Vec<u8>,
        source: SocketAddr,
    ) {
        let decoded = Packet::decode(&datagram, self.config.cluster_secret.as_deref());
        let packet = match decoded {
            Ok(packet) => packet,
            Err(error) => {
                tracing::debug!(%source, %error, "discarding invalid UDP datagram");
                return;
            }
        };
        {
            let mut all = self.metrics.lock().await;
            let metrics = all.job(packet.object_id);
            metrics.udp_datagrams_received += 1;
        }
        let result = self
            .handle_udp_packet(socket.clone(), &packet, source)
            .await;
        if let Err(error) = result {
            let reply = response(&packet, Kind::Error, error.to_string().into_bytes());
            if let Ok(wire) = reply.encode(self.config.cluster_secret.as_deref()) {
                let _ = socket.send_to(&wire, source).await;
                let mut all = self.metrics.lock().await;
                all.job(packet.object_id).udp_datagrams_sent += 1;
            }
        }
    }

    async fn send_udp_response(
        &self,
        socket: &UdpSocket,
        destination: SocketAddr,
        packet: Packet,
    ) -> Result<()> {
        let job = packet.object_id;
        socket
            .send_to(
                &packet.encode(self.config.cluster_secret.as_deref())?,
                destination,
            )
            .await?;
        self.metrics.lock().await.job(job).udp_datagrams_sent += 1;
        Ok(())
    }

    async fn handle_udp_packet(
        self: &Arc<Self>,
        socket: Arc<UdpSocket>,
        packet: &Packet,
        source: SocketAddr,
    ) -> Result<()> {
        let key = SessionKey {
            source,
            transfer: packet.transfer_id,
            block: packet.block_id,
        };
        let incoming_expiry = Duration::from_millis(
            self.config
                .udp_initial_rto_ms
                .saturating_mul(self.config.udp_max_retries as u64 + 2)
                .max(30_000),
        );
        let completed_expiry = Duration::from_secs(120);
        self.udp_server
            .completed
            .lock()
            .await
            .retain(|_, completed| completed.elapsed() < completed_expiry);
        match packet.kind {
            Kind::ReadRequest => {
                let length = packet.sequence_id as usize;
                ensure!(
                    length > 0 && length <= self.config.chunk_bytes,
                    "invalid UDP read length"
                );
                let block = self.ram.get(packet.block_id).await?;
                let b = block.lock().await;
                b.authorize(packet.lease_token)?;
                ensure!(
                    b.handle.job_id == packet.object_id
                        && b.handle.generation == packet.generation
                        && b.handle.state == BlockState::Ready,
                    "stale UDP read handle"
                );
                let begin: usize = packet.offset.try_into()?;
                let end = begin
                    .checked_add(length)
                    .ok_or_else(|| anyhow::anyhow!("UDP range overflow"))?;
                ensure!(end <= b.bytes.len(), "UDP read exceeds block");
                {
                    let mut active = self.udp_server.active_reads.lock().await;
                    ensure!(
                        *active < self.config.udp_max_sessions,
                        "TRAINPOOL_UDP_BACKPRESSURE: receiver read-session limit"
                    );
                    *active += 1;
                }
                // Reserve a bounded read session before cloning the requested
                // payload so rejected work cannot consume untracked memory.
                let payload = b.bytes[begin..end].to_vec();
                let handle = b.handle.clone();
                drop(b);
                let config = self.config.clone();
                let runtime = self.clone();
                let destination = source;
                let transfer_id = packet.transfer_id;
                let offset = packet.offset;
                tokio::spawn(async move {
                    let result = async {
                        let sender = UdpSocket::bind(if destination.is_ipv4() {
                            "0.0.0.0:0"
                        } else {
                            "[::]:0"
                        })
                        .await?;
                        send_payload(
                            &sender,
                            destination,
                            &config,
                            SendPayload {
                                handle: &handle,
                                transfer_id,
                                offset,
                                payload: &payload,
                                read_response: true,
                            },
                        )
                        .await
                    }
                    .await;
                    let mut all = runtime.metrics.lock().await;
                    let metrics = all.job(handle.job_id);
                    match result {
                        Ok(stats) => runtime.apply_udp_stats(metrics, &stats),
                        Err(error) => {
                            metrics.failed_transfers += 1;
                            tracing::warn!(%error, "UDP read response failed");
                        }
                    }
                    metrics.active_transfer_sessions =
                        metrics.active_transfer_sessions.saturating_sub(1);
                    drop(all);
                    let mut active = runtime.udp_server.active_reads.lock().await;
                    *active = active.saturating_sub(1);
                });
                let mut all = self.metrics.lock().await;
                let metrics = all.job(packet.object_id);
                metrics.transfer_sessions += 1;
                metrics.active_transfer_sessions += 1;
            }
            Kind::Start if packet.flags == 0 => {
                ensure!(packet.payload.len() == 40, "invalid UDP START metadata");
                let length = u64::from_be_bytes(packet.payload[..8].try_into()?) as usize;
                ensure!(
                    length > 0 && length <= self.config.chunk_bytes,
                    "invalid UDP session length"
                );
                let packet_count = packet.sequence_id as usize;
                ensure!(
                    packet_count == length.div_ceil(self.config.udp_payload_bytes),
                    "invalid UDP packet count"
                );
                ensure!(
                    packet
                        .offset
                        .checked_add(length as u64)
                        .is_some_and(|end| end <= packet.total_size),
                    "UDP session exceeds block"
                );
                let block = self.ram.get(packet.block_id).await?;
                let b = block.lock().await;
                b.authorize(packet.lease_token)?;
                ensure!(
                    b.handle.job_id == packet.object_id
                        && b.handle.generation == packet.generation
                        && b.handle.size == packet.total_size
                        && b.handle.state == BlockState::Writing
                        && b.written as u64 == packet.offset,
                    "stale or non-sequential UDP write session"
                );
                drop(b);
                let mut sessions = self.udp_server.incoming.lock().await;
                let expired_objects: Vec<_> = sessions
                    .values()
                    .filter(|session| session.last_activity.elapsed() >= incoming_expiry)
                    .map(|session| session.object)
                    .collect();
                sessions.retain(|_, session| session.last_activity.elapsed() < incoming_expiry);
                if !expired_objects.is_empty() {
                    let mut all = self.metrics.lock().await;
                    for object in expired_objects {
                        let metrics = all.job(object);
                        metrics.active_transfer_sessions =
                            metrics.active_transfer_sessions.saturating_sub(1);
                        metrics.failed_transfers += 1;
                    }
                }
                if !sessions.contains_key(&key) {
                    ensure!(
                        sessions.len() < self.config.udp_max_sessions,
                        "TRAINPOOL_UDP_BACKPRESSURE: receiver session limit"
                    );
                    let reservation = self.ram.accounting.reserve(length as u64)?;
                    sessions.insert(
                        key,
                        IncomingSession {
                            object: packet.object_id,
                            lease: packet.lease_token,
                            generation: packet.generation,
                            base_offset: packet.offset,
                            block_size: packet.total_size,
                            payload_size: self.config.udp_payload_bytes,
                            packet_count,
                            checksum: packet.payload[8..].try_into()?,
                            buffer: vec![0; length],
                            received: vec![false; packet_count],
                            received_count: 0,
                            last_sequence: None,
                            last_activity: Instant::now(),
                            _reservation: reservation,
                        },
                    );
                    let mut all = self.metrics.lock().await;
                    let metrics = all.job(packet.object_id);
                    metrics.transfer_sessions += 1;
                    metrics.active_transfer_sessions += 1;
                }
                drop(sessions);
                self.send_udp_response(
                    &socket,
                    source,
                    response(
                        packet,
                        Kind::StartAck,
                        (self.config.udp_window_packets.min(ACK_BITS) as u16)
                            .to_be_bytes()
                            .to_vec(),
                    ),
                )
                .await?;
            }
            Kind::Data => {
                let completed_base = packet
                    .offset
                    .checked_sub(packet.sequence_id as u64 * self.config.udp_payload_bytes as u64);
                let already_completed = if let Some(base) = completed_base {
                    self.udp_server
                        .completed
                        .lock()
                        .await
                        .contains_key(&(key, base))
                } else {
                    false
                };
                if already_completed {
                    // A retransmitted data packet may already be queued when FIN
                    // commits and removes the live session. Acknowledge it
                    // idempotently instead of turning successful completion into
                    // a spurious stale-transfer failure.
                    return self
                        .send_udp_response(
                            &socket,
                            source,
                            completed_data_ack(packet, self.config.udp_window_packets),
                        )
                        .await;
                }
                let mut duplicate = false;
                let mut out_of_order = false;
                let mut ack_payload = None;
                {
                    let mut sessions = self.udp_server.incoming.lock().await;
                    let Some(session) = sessions.get_mut(&key) else {
                        drop(sessions);
                        if self.udp_server.finishing.lock().await.contains(&key) {
                            return self
                                .send_udp_response(
                                    &socket,
                                    source,
                                    completed_data_ack(packet, self.config.udp_window_packets),
                                )
                                .await;
                        }
                        anyhow::bail!("stale UDP transfer ID");
                    };
                    ensure!(
                        packet.object_id == session.object
                            && packet.lease_token == session.lease
                            && packet.generation == session.generation
                            && packet.total_size == session.block_size,
                        "UDP session metadata changed"
                    );
                    let sequence = packet.sequence_id as usize;
                    session.last_activity = Instant::now();
                    ensure!(
                        sequence < session.packet_count,
                        "UDP sequence outside session"
                    );
                    let begin = sequence * session.payload_size;
                    let expected = session.base_offset + begin as u64;
                    ensure!(packet.offset == expected, "UDP packet offset mismatch");
                    ensure!(
                        begin + packet.payload.len() <= session.buffer.len(),
                        "UDP payload range mismatch"
                    );
                    if session.received[sequence] {
                        duplicate = true;
                    } else {
                        out_of_order = session.last_sequence.is_some_and(|last| sequence < last);
                        session.buffer[begin..begin + packet.payload.len()]
                            .copy_from_slice(&packet.payload);
                        session.received[sequence] = true;
                        session.received_count += 1;
                        session.last_sequence = Some(sequence);
                    }
                    if duplicate
                        || out_of_order
                        || session.received_count.is_multiple_of(8)
                        || session.received_count == session.packet_count
                    {
                        let base = (sequence / ACK_BITS) * ACK_BITS;
                        let bits: Vec<_> = (base..base + ACK_BITS)
                            .map(|index| session.received.get(index).copied().unwrap_or(false))
                            .collect();
                        ack_payload = Some(bitmap_payload(
                            base as u32,
                            &bits,
                            self.config.udp_window_packets,
                        ));
                    }
                }
                {
                    let mut all = self.metrics.lock().await;
                    let metrics = all.job(packet.object_id);
                    metrics.udp_payload_bytes += if duplicate {
                        0
                    } else {
                        packet.payload.len() as u64
                    };
                    metrics.duplicate_datagrams += u64::from(duplicate);
                    metrics.out_of_order_datagrams += u64::from(out_of_order);
                }
                if let Some(payload) = ack_payload {
                    self.send_udp_response(&socket, source, response(packet, Kind::Ack, payload))
                        .await?;
                    self.metrics.lock().await.job(packet.object_id).ack_count += 1;
                }
            }
            Kind::Fin => {
                if self
                    .udp_server
                    .completed
                    .lock()
                    .await
                    .contains_key(&(key, packet.offset))
                {
                    return self
                        .send_udp_response(
                            &socket,
                            source,
                            response(packet, Kind::Complete, vec![]),
                        )
                        .await;
                }
                let session = {
                    let mut sessions = self.udp_server.incoming.lock().await;
                    let Some(session) = sessions.get(&key) else {
                        drop(sessions);
                        if self.udp_server.finishing.lock().await.contains(&key) {
                            return Ok(());
                        }
                        anyhow::bail!("stale UDP transfer ID");
                    };
                    if session.received_count != session.packet_count {
                        let first_missing = session
                            .received
                            .iter()
                            .position(|received| !*received)
                            .unwrap_or(0);
                        let base = (first_missing / ACK_BITS) * ACK_BITS;
                        let bits: Vec<_> = (base..base + ACK_BITS)
                            .map(|index| !session.received.get(index).copied().unwrap_or(true))
                            .collect();
                        let payload =
                            bitmap_payload(base as u32, &bits, self.config.udp_window_packets);
                        drop(sessions);
                        self.send_udp_response(
                            &socket,
                            source,
                            response(packet, Kind::Nack, payload),
                        )
                        .await?;
                        self.metrics.lock().await.job(packet.object_id).nack_count += 1;
                        return Ok(());
                    }
                    ensure!(
                        packet.payload == session.checksum,
                        "UDP FIN checksum changed"
                    );
                    ensure!(
                        blake3::hash(&session.buffer).as_bytes() == &session.checksum,
                        "TRAINPOOL_CHECKSUM_MISMATCH: UDP transfer"
                    );
                    self.udp_server.finishing.lock().await.insert(key);
                    sessions.remove(&key).expect("session checked while locked")
                };
                let finalized: Result<()> = async {
                    let block = self.ram.get(packet.block_id).await?;
                    let mut b = block.lock().await;
                    b.authorize(packet.lease_token)?;
                    ensure!(
                        b.handle.generation == session.generation
                            && b.handle.state == BlockState::Writing
                            && b.written as u64 == session.base_offset,
                        "UDP destination changed before completion"
                    );
                    let begin: usize = session.base_offset.try_into()?;
                    let end = begin + session.buffer.len();
                    b.bytes[begin..end].copy_from_slice(&session.buffer);
                    b.hash.update(&session.buffer);
                    b.transfer_id = Some(packet.transfer_id);
                    b.written = end;
                    Ok(())
                }
                .await;
                if let Err(error) = finalized {
                    self.udp_server.finishing.lock().await.remove(&key);
                    return Err(error);
                }
                self.udp_server
                    .completed
                    .lock()
                    .await
                    .insert((key, packet.offset), Instant::now());
                {
                    let mut completed = self.udp_server.completed.lock().await;
                    let completed_limit = self.config.udp_max_sessions.saturating_mul(16);
                    if completed.len() > completed_limit
                        && let Some(oldest) = completed
                            .iter()
                            .min_by_key(|(_, completed_at)| **completed_at)
                            .map(|(key, _)| *key)
                    {
                        completed.remove(&oldest);
                    }
                }
                self.udp_server.finishing.lock().await.remove(&key);
                let mut all = self.metrics.lock().await;
                all.job(packet.object_id).active_transfer_sessions = all
                    .job(packet.object_id)
                    .active_transfer_sessions
                    .saturating_sub(1);
                drop(all);
                self.send_udp_response(&socket, source, response(packet, Kind::Complete, vec![]))
                    .await?;
            }
            Kind::Abort if self.udp_server.incoming.lock().await.remove(&key).is_some() => {
                let mut all = self.metrics.lock().await;
                let metrics = all.job(packet.object_id);
                metrics.active_transfer_sessions =
                    metrics.active_transfer_sessions.saturating_sub(1);
                metrics.failed_transfers += 1;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn apply_udp_stats(
        &self,
        metrics: &mut crate::metrics::JobMetrics,
        stats: &crate::transport::udp::TransferStats,
    ) {
        metrics.udp_datagrams_sent += stats.datagrams_sent;
        metrics.udp_datagrams_received += stats.datagrams_received;
        metrics.udp_payload_bytes += stats.payload_bytes;
        metrics.udp_retransmitted_datagrams += stats.retransmitted_datagrams;
        metrics.udp_retransmitted_bytes += stats.retransmitted_bytes;
        metrics.duplicate_datagrams += stats.duplicate_datagrams;
        metrics.out_of_order_datagrams += stats.out_of_order_datagrams;
        metrics.ack_count += stats.ack_count;
        metrics.nack_count += stats.nack_count;
        // A receive-only half of a transfer has no RTT/RTO sample. Preserve
        // the latest authoritative sender sample instead of replacing it with
        // a misleading zero when a read completes.
        if stats.estimated_rtt_ms > 0.0 {
            metrics.estimated_rtt_ms = stats.estimated_rtt_ms;
        }
        if stats.retransmission_timeout_ms > 0.0 {
            metrics.retransmission_timeout_ms = stats.retransmission_timeout_ms;
        }
        metrics.packet_loss_estimate = if metrics.udp_datagrams_sent == 0 {
            0.0
        } else {
            metrics.udp_retransmitted_datagrams as f64 / metrics.udp_datagrams_sent as f64
        };
        metrics.effective_throughput_mbps = stats.throughput_mbps();
        if metrics.udp_datagrams_sent > 0 {
            metrics.average_payload_size =
                metrics.udp_payload_bytes as f64 / metrics.udp_datagrams_sent as f64;
        }
    }

    pub fn reliable_udp(&self) -> ReliableUdpDataTransport {
        ReliableUdpDataTransport::new(self.config.clone())
    }
}
