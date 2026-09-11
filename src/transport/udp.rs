//! Reliable, unordered, bounded UDP data plane for inter-node tensor payloads.
//!
//! Control stays on authenticated TCP. Every UDP datagram is independently
//! integrity checked and, when a cluster secret exists, authenticated. Payload
//! transfers use a bounded selective-repeat window and bitmap acknowledgements.

use crate::{config::Config, memory::block::MemoryBlockHandle};
use anyhow::{Result, ensure};
use hmac::{Hmac, Mac};
use sha2::Sha256;
use std::{net::SocketAddr, time::Duration};
use tokio::{net::UdpSocket, time::Instant};
use uuid::Uuid;

pub const MAGIC: &[u8; 4] = b"TPU1";
pub const VERSION: u8 = 1;
pub const HEADER_LEN: usize = 166;
pub const MAX_DATAGRAM: usize = 1400;
pub const ACK_BITS: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum Kind {
    Start = 1,
    StartAck = 2,
    Data = 3,
    Ack = 4,
    Nack = 5,
    Fin = 6,
    Complete = 7,
    Abort = 8,
    ReadRequest = 9,
    Error = 10,
}

impl TryFrom<u8> for Kind {
    type Error = anyhow::Error;
    fn try_from(value: u8) -> Result<Self> {
        Ok(match value {
            1 => Self::Start,
            2 => Self::StartAck,
            3 => Self::Data,
            4 => Self::Ack,
            5 => Self::Nack,
            6 => Self::Fin,
            7 => Self::Complete,
            8 => Self::Abort,
            9 => Self::ReadRequest,
            10 => Self::Error,
            _ => anyhow::bail!("unknown UDP packet kind {value}"),
        })
    }
}

#[derive(Clone, Debug)]
pub struct Packet {
    pub kind: Kind,
    pub flags: u16,
    pub transfer_id: Uuid,
    /// Tensor/session owner. The current block protocol uses the job id.
    pub object_id: Uuid,
    pub block_id: Uuid,
    pub lease_token: Uuid,
    pub generation: u64,
    pub sequence_id: u32,
    pub offset: u64,
    pub total_size: u64,
    pub payload: Vec<u8>,
}

impl Packet {
    pub fn for_handle(kind: Kind, handle: &MemoryBlockHandle, transfer_id: Uuid) -> Self {
        Self {
            kind,
            flags: 0,
            transfer_id,
            object_id: handle.job_id,
            block_id: handle.id,
            lease_token: handle.lease_token,
            generation: handle.generation,
            sequence_id: 0,
            offset: 0,
            total_size: handle.size,
            payload: vec![],
        }
    }

    pub fn encode(&self, secret: Option<&str>) -> Result<Vec<u8>> {
        ensure!(
            self.payload.len() + HEADER_LEN <= MAX_DATAGRAM,
            "UDP datagram exceeds MTU-safe envelope"
        );
        let mut bytes = Vec::with_capacity(HEADER_LEN + self.payload.len());
        bytes.extend_from_slice(MAGIC);
        bytes.push(VERSION);
        bytes.push(self.kind as u8);
        bytes.extend_from_slice(&self.flags.to_be_bytes());
        for id in [
            self.transfer_id,
            self.object_id,
            self.block_id,
            self.lease_token,
        ] {
            bytes.extend_from_slice(id.as_bytes());
        }
        bytes.extend_from_slice(&self.generation.to_be_bytes());
        bytes.extend_from_slice(&self.sequence_id.to_be_bytes());
        bytes.extend_from_slice(&self.offset.to_be_bytes());
        bytes.extend_from_slice(&self.total_size.to_be_bytes());
        bytes.extend_from_slice(&(self.payload.len() as u16).to_be_bytes());
        bytes.extend_from_slice(blake3::hash(&self.payload).as_bytes());
        let tag = authentication_tag(secret, &bytes, &self.payload)?;
        bytes.extend_from_slice(&tag);
        bytes.extend_from_slice(&self.payload);
        Ok(bytes)
    }

    pub fn decode(datagram: &[u8], secret: Option<&str>) -> Result<Self> {
        ensure!(
            (HEADER_LEN..=MAX_DATAGRAM).contains(&datagram.len()),
            "invalid UDP datagram length"
        );
        ensure!(&datagram[..4] == MAGIC, "invalid UDP magic");
        ensure!(datagram[4] == VERSION, "unsupported UDP protocol version");
        let kind = Kind::try_from(datagram[5])?;
        let flags = u16::from_be_bytes(datagram[6..8].try_into()?);
        let transfer_id = Uuid::from_bytes(datagram[8..24].try_into()?);
        let object_id = Uuid::from_bytes(datagram[24..40].try_into()?);
        let block_id = Uuid::from_bytes(datagram[40..56].try_into()?);
        let lease_token = Uuid::from_bytes(datagram[56..72].try_into()?);
        let generation = u64::from_be_bytes(datagram[72..80].try_into()?);
        let sequence_id = u32::from_be_bytes(datagram[80..84].try_into()?);
        let offset = u64::from_be_bytes(datagram[84..92].try_into()?);
        let total_size = u64::from_be_bytes(datagram[92..100].try_into()?);
        let payload_len = u16::from_be_bytes(datagram[100..102].try_into()?) as usize;
        ensure!(
            datagram.len() == HEADER_LEN + payload_len,
            "UDP payload length mismatch"
        );
        let payload = &datagram[HEADER_LEN..];
        ensure!(
            &datagram[102..134] == blake3::hash(payload).as_bytes(),
            "TRAINPOOL_CHECKSUM_MISMATCH: UDP datagram"
        );
        let expected = authentication_tag(secret, &datagram[..134], payload)?;
        ensure!(
            datagram[134..166] == expected,
            "TRAINPOOL_AUTH_FAILED: UDP datagram"
        );
        Ok(Self {
            kind,
            flags,
            transfer_id,
            object_id,
            block_id,
            lease_token,
            generation,
            sequence_id,
            offset,
            total_size,
            payload: payload.to_vec(),
        })
    }
}

fn authentication_tag(secret: Option<&str>, header: &[u8], payload: &[u8]) -> Result<[u8; 32]> {
    let Some(secret) = secret else {
        return Ok([0; 32]);
    };
    let mut mac = Hmac::<Sha256>::new_from_slice(secret.as_bytes())?;
    mac.update(header);
    mac.update(payload);
    Ok(mac.finalize().into_bytes().into())
}

pub fn bitmap_payload(base: u32, bits: &[bool], window: usize) -> Vec<u8> {
    let mut payload = vec![0; 4 + ACK_BITS / 8 + 2];
    payload[..4].copy_from_slice(&base.to_be_bytes());
    for (index, value) in bits.iter().take(ACK_BITS).enumerate() {
        if *value {
            payload[4 + index / 8] |= 1 << (index % 8);
        }
    }
    payload[36..38].copy_from_slice(&(window.min(u16::MAX as usize) as u16).to_be_bytes());
    payload
}

pub fn parse_bitmap(payload: &[u8]) -> Result<(u32, [u8; 32], usize)> {
    ensure!(payload.len() == 38, "invalid ACK/NACK bitmap");
    let base = u32::from_be_bytes(payload[..4].try_into()?);
    let bitmap = payload[4..36].try_into()?;
    let window = u16::from_be_bytes(payload[36..38].try_into()?) as usize;
    Ok((base, bitmap, window))
}

pub fn bitmap_contains(bitmap: &[u8; 32], index: usize) -> bool {
    index < ACK_BITS && bitmap[index / 8] & (1 << (index % 8)) != 0
}

#[derive(Clone, Debug, Default)]
pub struct TransferStats {
    pub datagrams_sent: u64,
    pub datagrams_received: u64,
    pub payload_bytes: u64,
    pub retransmitted_datagrams: u64,
    pub retransmitted_bytes: u64,
    pub ack_count: u64,
    pub nack_count: u64,
    pub duplicate_datagrams: u64,
    pub out_of_order_datagrams: u64,
    pub estimated_rtt_ms: f64,
    pub retransmission_timeout_ms: f64,
    pub elapsed_ms: f64,
}

impl TransferStats {
    pub fn loss_estimate(&self) -> f64 {
        if self.datagrams_sent == 0 {
            0.0
        } else {
            self.retransmitted_datagrams as f64 / self.datagrams_sent as f64
        }
    }
    pub fn throughput_mbps(&self) -> f64 {
        if self.elapsed_ms <= 0.0 {
            0.0
        } else {
            self.payload_bytes as f64 * 8.0 / self.elapsed_ms / 1000.0
        }
    }
}

#[derive(Clone)]
pub struct ReliableUdpDataTransport {
    config: Config,
}

pub struct SendPayload<'a> {
    pub handle: &'a MemoryBlockHandle,
    pub transfer_id: Uuid,
    pub offset: u64,
    pub payload: &'a [u8],
    pub read_response: bool,
}

impl ReliableUdpDataTransport {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    pub async fn write(
        &self,
        destination: SocketAddr,
        handle: &MemoryBlockHandle,
        transfer_id: Uuid,
        offset: u64,
        payload: &[u8],
    ) -> Result<TransferStats> {
        let socket = UdpSocket::bind(if destination.is_ipv4() {
            "0.0.0.0:0"
        } else {
            "[::]:0"
        })
        .await?;
        send_payload(
            &socket,
            destination,
            &self.config,
            SendPayload {
                handle,
                transfer_id,
                offset,
                payload,
                read_response: false,
            },
        )
        .await
    }

    pub async fn read(
        &self,
        destination: SocketAddr,
        handle: &MemoryBlockHandle,
        transfer_id: Uuid,
        offset: u64,
        length: usize,
    ) -> Result<(Vec<u8>, TransferStats)> {
        ensure!(
            length <= self.config.chunk_bytes,
            "UDP read exceeds configured chunk size"
        );
        let socket = UdpSocket::bind(if destination.is_ipv4() {
            "0.0.0.0:0"
        } else {
            "[::]:0"
        })
        .await?;
        let mut request = Packet::for_handle(Kind::ReadRequest, handle, transfer_id);
        request.offset = offset;
        request.sequence_id = length.try_into()?;
        let wire = request.encode(self.config.cluster_secret.as_deref())?;
        let started = Instant::now();
        let mut retries = 0;
        loop {
            socket.send_to(&wire, destination).await?;
            let mut datagram = [0_u8; MAX_DATAGRAM];
            match tokio::time::timeout(
                Duration::from_millis(self.config.udp_initial_rto_ms),
                socket.recv_from(&mut datagram),
            )
            .await
            {
                Ok(Ok((size, source))) => {
                    let start =
                        Packet::decode(&datagram[..size], self.config.cluster_secret.as_deref())?;
                    if start.transfer_id == transfer_id && start.kind == Kind::Start {
                        return receive_payload(
                            &socket,
                            source,
                            handle,
                            start,
                            &self.config,
                            started,
                        )
                        .await;
                    }
                    if start.transfer_id == transfer_id && start.kind == Kind::Error {
                        anyhow::bail!(String::from_utf8_lossy(&start.payload).into_owned());
                    }
                }
                _ => {
                    retries += 1;
                    ensure!(
                        retries <= self.config.udp_max_retries,
                        "TRAINPOOL_TRANSFER_TIMEOUT"
                    );
                }
            }
        }
    }
}

pub async fn send_payload(
    socket: &UdpSocket,
    destination: SocketAddr,
    config: &Config,
    request: SendPayload<'_>,
) -> Result<TransferStats> {
    let result = send_payload_inner(socket, destination, config, &request).await;
    if result.is_err() {
        let mut abort = Packet::for_handle(Kind::Abort, request.handle, request.transfer_id);
        abort.flags = u16::from(request.read_response);
        abort.offset = request.offset;
        if let Ok(wire) = abort.encode(config.cluster_secret.as_deref()) {
            let _ = socket.send_to(&wire, destination).await;
        }
    }
    result
}

fn update_rto(sample_ms: f64, srtt: &mut Option<f64>, rttvar: &mut f64) -> Duration {
    if let Some(previous) = *srtt {
        *rttvar = 0.75 * *rttvar + 0.25 * (previous - sample_ms).abs();
        *srtt = Some(0.875 * previous + 0.125 * sample_ms);
    } else {
        *srtt = Some(sample_ms);
        *rttvar = sample_ms / 2.0;
    }
    let milliseconds = (srtt.unwrap_or(sample_ms) + 4.0 * *rttvar).clamp(10.0, 2_000.0);
    Duration::from_secs_f64(milliseconds / 1_000.0)
}

async fn send_payload_inner(
    socket: &UdpSocket,
    destination: SocketAddr,
    config: &Config,
    request: &SendPayload<'_>,
) -> Result<TransferStats> {
    let handle = request.handle;
    let transfer_id = request.transfer_id;
    let offset = request.offset;
    let payload = request.payload;
    let read_response = request.read_response;
    ensure!(!payload.is_empty(), "zero-length UDP transfer");
    ensure!(
        payload.len() <= config.chunk_bytes,
        "UDP transfer exceeds configured chunk size"
    );
    ensure!(
        offset
            .checked_add(payload.len() as u64)
            .is_some_and(|end| end <= handle.size),
        "UDP transfer range exceeds block"
    );
    let started = Instant::now();
    let mut stats = TransferStats::default();
    let packet_count = payload.len().div_ceil(config.udp_payload_bytes);
    let mut start = Packet::for_handle(Kind::Start, handle, transfer_id);
    start.flags = u16::from(read_response);
    start.sequence_id = packet_count.try_into()?;
    start.offset = offset;
    start.payload = (payload.len() as u64).to_be_bytes().to_vec();
    start
        .payload
        .extend_from_slice(blake3::hash(payload).as_bytes());
    let start_wire = start.encode(config.cluster_secret.as_deref())?;
    let mut rto = Duration::from_millis(config.udp_initial_rto_ms);
    let mut negotiated_window = config.udp_window_packets.min(ACK_BITS);
    let mut srtt = None;
    let mut rttvar = 0.0;
    let handshake_started = Instant::now();
    let mut retries = 0;
    loop {
        socket.send_to(&start_wire, destination).await?;
        stats.datagrams_sent += 1;
        let mut datagram = [0_u8; MAX_DATAGRAM];
        match tokio::time::timeout(rto, socket.recv_from(&mut datagram)).await {
            Ok(Ok((size, source))) if source == destination => {
                let packet = Packet::decode(&datagram[..size], config.cluster_secret.as_deref())?;
                stats.datagrams_received += 1;
                if packet.transfer_id == transfer_id && packet.kind == Kind::StartAck {
                    stats.ack_count += 1;
                    if packet.payload.len() == 2 {
                        let receiver_window =
                            u16::from_be_bytes(packet.payload[..2].try_into()?) as usize;
                        negotiated_window = negotiated_window.min(receiver_window.max(2));
                    }
                    if retries == 0 {
                        let sample = handshake_started.elapsed().as_secs_f64() * 1000.0;
                        rto = update_rto(sample, &mut srtt, &mut rttvar);
                        stats.estimated_rtt_ms = sample;
                    }
                    break;
                }
                if packet.transfer_id == transfer_id && packet.kind == Kind::Error {
                    anyhow::bail!(String::from_utf8_lossy(&packet.payload).into_owned());
                }
            }
            _ => {
                retries += 1;
                stats.retransmitted_datagrams += 1;
                ensure!(
                    retries <= config.udp_max_retries,
                    "TRAINPOOL_TRANSFER_TIMEOUT: UDP START"
                );
                rto = (rto * 2).min(Duration::from_secs(2));
            }
        }
    }

    let mut acknowledged = vec![false; packet_count];
    let mut base = 0;
    while base < packet_count {
        let end = (base + negotiated_window).min(packet_count);
        let mut round = 0;
        while acknowledged[base..end].iter().any(|value| !*value) {
            let round_started = Instant::now();
            let mut sampled_round = false;
            let mut sent_in_round = 0_u64;
            let loss_backoff = 1_u64 << round.min(4);
            let pacing_interval = config.udp_pacing_micros.saturating_mul(loss_backoff);
            for (sequence, is_acknowledged) in acknowledged.iter().enumerate().take(end).skip(base)
            {
                if *is_acknowledged {
                    continue;
                }
                let begin = sequence * config.udp_payload_bytes;
                let finish = (begin + config.udp_payload_bytes).min(payload.len());
                let mut packet = Packet::for_handle(Kind::Data, handle, transfer_id);
                packet.flags = u16::from(read_response);
                packet.sequence_id = sequence.try_into()?;
                packet.offset = offset + begin as u64;
                packet.payload = payload[begin..finish].to_vec();
                let wire = packet.encode(config.cluster_secret.as_deref())?;
                socket.send_to(&wire, destination).await?;
                stats.datagrams_sent += 1;
                if round > 0 {
                    stats.retransmitted_datagrams += 1;
                    stats.retransmitted_bytes += packet.payload.len() as u64;
                } else {
                    stats.payload_bytes += packet.payload.len() as u64;
                }
                sent_in_round += 1;
                // Pace bounded bursts against a monotonic target. Sub-millisecond
                // sleeps are commonly rounded up, so one timer per datagram would
                // accidentally throttle the LAN by orders of magnitude.
                if sent_in_round.is_multiple_of(32) {
                    let target =
                        Duration::from_micros(pacing_interval.saturating_mul(sent_in_round));
                    if let Some(delay) = target.checked_sub(round_started.elapsed()) {
                        tokio::time::sleep(delay).await;
                    }
                }
            }
            let target = Duration::from_micros(pacing_interval.saturating_mul(sent_in_round));
            if let Some(delay) = target.checked_sub(round_started.elapsed()) {
                tokio::time::sleep(delay).await;
            }
            let deadline = Instant::now() + rto;
            while Instant::now() < deadline && acknowledged[base..end].iter().any(|value| !*value) {
                let mut datagram = [0_u8; MAX_DATAGRAM];
                let remaining = deadline.saturating_duration_since(Instant::now());
                let Ok(Ok((size, source))) =
                    tokio::time::timeout(remaining, socket.recv_from(&mut datagram)).await
                else {
                    break;
                };
                if source != destination {
                    continue;
                }
                let packet = Packet::decode(&datagram[..size], config.cluster_secret.as_deref())?;
                stats.datagrams_received += 1;
                if packet.transfer_id != transfer_id {
                    continue;
                }
                if packet.kind == Kind::Error {
                    anyhow::bail!(String::from_utf8_lossy(&packet.payload).into_owned());
                }
                if matches!(packet.kind, Kind::Ack | Kind::Nack) {
                    let (ack_base, bitmap, receiver_window) = parse_bitmap(&packet.payload)?;
                    negotiated_window = negotiated_window.min(receiver_window.max(2));
                    if packet.kind == Kind::Ack {
                        stats.ack_count += 1;
                        if round == 0 && !sampled_round {
                            let sample = round_started.elapsed().as_secs_f64() * 1000.0;
                            rto = update_rto(sample, &mut srtt, &mut rttvar);
                            stats.estimated_rtt_ms = srtt.unwrap_or(sample);
                            sampled_round = true;
                        }
                        for index in 0..ACK_BITS {
                            let sequence = ack_base as usize + index;
                            if sequence < packet_count && bitmap_contains(&bitmap, index) {
                                acknowledged[sequence] = true;
                            }
                        }
                    } else {
                        stats.nack_count += 1;
                        // NACK bits denote missing packets; every clear bit in
                        // the reported range is known received. This preserves
                        // selective retransmission rather than replaying a window.
                        for index in 0..ACK_BITS {
                            let sequence = ack_base as usize + index;
                            if sequence >= base
                                && sequence < end
                                && !bitmap_contains(&bitmap, index)
                            {
                                acknowledged[sequence] = true;
                            }
                        }
                        break;
                    }
                }
            }
            round += 1;
            ensure!(
                round <= config.udp_max_retries,
                "TRAINPOOL_TRANSFER_TIMEOUT: bounded UDP retry exhausted"
            );
            if round > 1 {
                rto = (rto * 2).min(Duration::from_secs(2));
                negotiated_window = (negotiated_window / 2).max(2);
            }
        }
        base = end;
    }

    let mut fin = Packet::for_handle(Kind::Fin, handle, transfer_id);
    fin.flags = u16::from(read_response);
    fin.offset = offset;
    fin.sequence_id = packet_count.try_into()?;
    fin.payload = blake3::hash(payload).as_bytes().to_vec();
    let fin_wire = fin.encode(config.cluster_secret.as_deref())?;
    for attempt in 0..=config.udp_max_retries {
        socket.send_to(&fin_wire, destination).await?;
        stats.datagrams_sent += 1;
        if attempt > 0 {
            stats.retransmitted_datagrams += 1;
        }
        let mut datagram = [0_u8; MAX_DATAGRAM];
        if let Ok(Ok((size, source))) =
            tokio::time::timeout(rto, socket.recv_from(&mut datagram)).await
            && source == destination
        {
            let packet = Packet::decode(&datagram[..size], config.cluster_secret.as_deref())?;
            stats.datagrams_received += 1;
            if packet.transfer_id == transfer_id && packet.kind == Kind::Complete {
                stats.ack_count += 1;
                stats.retransmission_timeout_ms = rto.as_secs_f64() * 1000.0;
                stats.estimated_rtt_ms = srtt.unwrap_or(stats.estimated_rtt_ms);
                stats.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                return Ok(stats);
            }
            if packet.transfer_id == transfer_id && packet.kind == Kind::Error {
                anyhow::bail!(String::from_utf8_lossy(&packet.payload).into_owned());
            }
            if packet.transfer_id == transfer_id && packet.kind == Kind::Nack {
                stats.nack_count += 1;
            }
        }
        rto = (rto * 2).min(Duration::from_secs(2));
    }
    anyhow::bail!("TRAINPOOL_TRANSFER_TIMEOUT: UDP FIN acknowledgement")
}

async fn receive_payload(
    socket: &UdpSocket,
    source: SocketAddr,
    handle: &MemoryBlockHandle,
    start: Packet,
    config: &Config,
    started: Instant,
) -> Result<(Vec<u8>, TransferStats)> {
    let packet_count = start.sequence_id as usize;
    ensure!(start.payload.len() == 40, "invalid UDP START metadata");
    let length = u64::from_be_bytes(start.payload[..8].try_into()?).try_into()?;
    ensure!(
        packet_count > 0 && length <= config.chunk_bytes,
        "invalid UDP read session"
    );
    let mut buffer = vec![0; length];
    let mut received = vec![false; packet_count];
    let mut stats = TransferStats::default();
    let mut ack = Packet::for_handle(Kind::StartAck, handle, start.transfer_id);
    ack.flags = 1;
    ack.payload = (config.udp_window_packets.min(ACK_BITS) as u16)
        .to_be_bytes()
        .to_vec();
    socket
        .send_to(&ack.encode(config.cluster_secret.as_deref())?, source)
        .await?;
    stats.datagrams_sent += 1;
    let mut last_sequence = None;
    let mut retries = 0;
    loop {
        let mut datagram = [0_u8; MAX_DATAGRAM];
        let received_packet = tokio::time::timeout(
            Duration::from_millis(config.udp_initial_rto_ms.max(20) * 4),
            socket.recv_from(&mut datagram),
        )
        .await;
        let Ok(Ok((size, peer))) = received_packet else {
            retries += 1;
            ensure!(
                retries <= config.udp_max_retries,
                "TRAINPOOL_TRANSFER_TIMEOUT: UDP read"
            );
            continue;
        };
        if peer != source {
            continue;
        }
        let packet = Packet::decode(&datagram[..size], config.cluster_secret.as_deref())?;
        stats.datagrams_received += 1;
        if packet.transfer_id != start.transfer_id {
            continue;
        }
        match packet.kind {
            Kind::Data => {
                let sequence = packet.sequence_id as usize;
                ensure!(sequence < packet_count, "UDP sequence outside transfer");
                let begin = sequence * config.udp_payload_bytes;
                ensure!(
                    begin + packet.payload.len() <= buffer.len(),
                    "UDP data range invalid"
                );
                if received[sequence] {
                    stats.duplicate_datagrams += 1;
                } else {
                    if last_sequence.is_some_and(|previous| sequence < previous) {
                        stats.out_of_order_datagrams += 1;
                    }
                    buffer[begin..begin + packet.payload.len()].copy_from_slice(&packet.payload);
                    received[sequence] = true;
                    stats.payload_bytes += packet.payload.len() as u64;
                    last_sequence = Some(sequence);
                }
                if received.iter().filter(|value| **value).count() % 8 == 0
                    || received.iter().all(|value| *value)
                {
                    let base = (sequence / ACK_BITS) * ACK_BITS;
                    let bits: Vec<_> = (base..base + ACK_BITS)
                        .map(|index| received.get(index).copied().unwrap_or(false))
                        .collect();
                    let mut response = Packet::for_handle(Kind::Ack, handle, start.transfer_id);
                    response.payload =
                        bitmap_payload(base as u32, &bits, config.udp_window_packets);
                    socket
                        .send_to(&response.encode(config.cluster_secret.as_deref())?, source)
                        .await?;
                    stats.datagrams_sent += 1;
                    stats.ack_count += 1;
                }
            }
            Kind::Fin if received.iter().all(|value| *value) => {
                ensure!(
                    packet.payload == start.payload[8..],
                    "UDP FIN checksum changed"
                );
                ensure!(
                    blake3::hash(&buffer).as_bytes() == &start.payload[8..],
                    "TRAINPOOL_CHECKSUM_MISMATCH: UDP transfer"
                );
                let complete = Packet::for_handle(Kind::Complete, handle, start.transfer_id);
                socket
                    .send_to(&complete.encode(config.cluster_secret.as_deref())?, source)
                    .await?;
                stats.datagrams_sent += 1;
                stats.ack_count += 1;
                stats.elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
                return Ok((buffer, stats));
            }
            Kind::Fin => {
                let bits: Vec<_> = received.iter().map(|value| !*value).collect();
                let mut nack = Packet::for_handle(Kind::Nack, handle, start.transfer_id);
                nack.payload = bitmap_payload(0, &bits, config.udp_window_packets);
                socket
                    .send_to(&nack.encode(config.cluster_secret.as_deref())?, source)
                    .await?;
                stats.datagrams_sent += 1;
                stats.nack_count += 1;
            }
            Kind::Abort | Kind::Error => anyhow::bail!("TRAINPOOL_TRANSFER_ABORTED"),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::block::{BlockState, Location};
    use std::collections::HashSet;

    fn packet(kind: Kind, payload: &[u8]) -> Packet {
        Packet {
            kind,
            flags: 0,
            transfer_id: Uuid::from_u128(1),
            object_id: Uuid::from_u128(2),
            block_id: Uuid::from_u128(3),
            lease_token: Uuid::from_u128(4),
            generation: 5,
            sequence_id: 6,
            offset: 7,
            total_size: 8,
            payload: payload.to_vec(),
        }
    }

    fn handle(size: usize) -> MemoryBlockHandle {
        MemoryBlockHandle {
            id: Uuid::new_v4(),
            job_id: Uuid::new_v4(),
            size: size as u64,
            owner_node: Uuid::new_v4(),
            owner_incarnation: Uuid::new_v4(),
            location_type: Location::Ram {
                node_id: Uuid::new_v4(),
            },
            checksum: None,
            state: BlockState::Writing,
            lease_token: Uuid::new_v4(),
            lease_expires_ms: u64::MAX,
            generation: 0,
            tensor: None,
        }
    }

    #[test]
    fn datagram_is_mtu_safe_authenticated_and_corruption_is_rejected() {
        let encoded = packet(Kind::Data, &[9; 1200])
            .encode(Some("secret"))
            .unwrap();
        assert!(encoded.len() <= MAX_DATAGRAM);
        assert_eq!(
            Packet::decode(&encoded, Some("secret"))
                .unwrap()
                .payload
                .len(),
            1200
        );
        assert!(Packet::decode(&encoded, Some("wrong")).is_err());
        let mut corrupt = encoded;
        *corrupt.last_mut().unwrap() ^= 1;
        assert!(Packet::decode(&corrupt, Some("secret")).is_err());
    }

    #[test]
    fn bitmap_represents_sparse_out_of_order_delivery() {
        let mut bits = vec![false; ACK_BITS];
        for index in [0, 1, 2, 4, 5, 7, 8, 9] {
            bits[index] = true;
        }
        let payload = bitmap_payload(0, &bits, 64);
        let (base, bitmap, window) = parse_bitmap(&payload).unwrap();
        assert_eq!(base, 0);
        assert_eq!(window, 64);
        assert!(bitmap_contains(&bitmap, 7));
        assert!(!bitmap_contains(&bitmap, 3));
        assert!(!bitmap_contains(&bitmap, 6));
    }

    async fn lossy_receiver(
        socket: UdpSocket,
        config: Config,
        loss_every: Option<u32>,
        drop_fin_and_complete: bool,
    ) -> (Vec<u8>, usize) {
        let mut start = None;
        let mut source = None;
        let mut received = Vec::new();
        let mut present = Vec::new();
        let mut dropped_once = HashSet::new();
        let mut dropped_fin = false;
        let mut dropped_complete = false;
        loop {
            let mut wire = [0_u8; MAX_DATAGRAM];
            let (size, peer) = socket.recv_from(&mut wire).await.unwrap();
            let packet = Packet::decode(&wire[..size], config.cluster_secret.as_deref()).unwrap();
            match packet.kind {
                Kind::Start => {
                    let length =
                        u64::from_be_bytes(packet.payload[..8].try_into().unwrap()) as usize;
                    received = vec![0; length];
                    present = vec![false; packet.sequence_id as usize];
                    source = Some(peer);
                    start = Some(packet.clone());
                    let reply = response_packet(
                        &packet,
                        Kind::StartAck,
                        (config.udp_window_packets as u16).to_be_bytes().to_vec(),
                    );
                    socket
                        .send_to(
                            &reply.encode(config.cluster_secret.as_deref()).unwrap(),
                            peer,
                        )
                        .await
                        .unwrap();
                }
                Kind::Data => {
                    let sequence = packet.sequence_id;
                    if loss_every.is_some_and(|every| sequence.is_multiple_of(every))
                        && dropped_once.insert(sequence)
                    {
                        continue;
                    }
                    let begin = (packet.offset - start.as_ref().unwrap().offset) as usize;
                    received[begin..begin + packet.payload.len()].copy_from_slice(&packet.payload);
                    present[sequence as usize] = true;
                    let base = (sequence as usize / ACK_BITS) * ACK_BITS;
                    let bits: Vec<_> = (base..base + ACK_BITS)
                        .map(|index| present.get(index).copied().unwrap_or(false))
                        .collect();
                    let reply = response_packet(
                        &packet,
                        Kind::Ack,
                        bitmap_payload(base as u32, &bits, config.udp_window_packets),
                    );
                    socket
                        .send_to(
                            &reply.encode(config.cluster_secret.as_deref()).unwrap(),
                            peer,
                        )
                        .await
                        .unwrap();
                }
                Kind::Fin => {
                    if drop_fin_and_complete && !dropped_fin {
                        dropped_fin = true;
                        continue;
                    }
                    assert!(present.iter().all(|value| *value));
                    assert_eq!(packet.payload, blake3::hash(&received).as_bytes());
                    if drop_fin_and_complete && !dropped_complete {
                        dropped_complete = true;
                        continue;
                    }
                    let reply = response_packet(&packet, Kind::Complete, vec![]);
                    socket
                        .send_to(
                            &reply.encode(config.cluster_secret.as_deref()).unwrap(),
                            peer,
                        )
                        .await
                        .unwrap();
                    assert_eq!(source, Some(peer));
                    return (received, dropped_once.len());
                }
                Kind::Abort => panic!("successful transfer was aborted"),
                _ => {}
            }
        }
    }

    fn response_packet(request: &Packet, kind: Kind, payload: Vec<u8>) -> Packet {
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

    #[tokio::test]
    async fn selective_repeat_recovers_loss_and_dropped_completion_without_replaying_windows() {
        for (loss_every, drop_completion) in [
            (None, false),
            (Some(1000), false),
            (Some(100), false),
            (Some(20), true),
        ] {
            let mut config = Config {
                chunk_bytes: 512 * 1024,
                udp_initial_rto_ms: 10,
                udp_pacing_micros: 1,
                udp_max_retries: 8,
                cluster_secret: Some("loss-test".into()),
                ..Default::default()
            };
            config.udp_window_packets = 64;
            let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let destination = receiver.local_addr().unwrap();
            let receiver_task = tokio::spawn(lossy_receiver(
                receiver,
                config.clone(),
                loss_every,
                drop_completion,
            ));
            let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
            let payload: Vec<_> = (0..config.chunk_bytes)
                .map(|index| (index % 251) as u8)
                .collect();
            let handle = handle(payload.len());
            let stats = send_payload(
                &sender,
                destination,
                &config,
                SendPayload {
                    handle: &handle,
                    transfer_id: Uuid::new_v4(),
                    offset: 0,
                    payload: &payload,
                    read_response: false,
                },
            )
            .await
            .unwrap();
            let (restored, dropped) = receiver_task.await.unwrap();
            assert_eq!(restored, payload);
            assert_eq!(stats.retransmitted_bytes, dropped as u64 * 1200);
            assert!(stats.retransmitted_datagrams >= dropped as u64);
            assert!(
                stats.ack_count > 1,
                "window must have multiple packets in flight"
            );
            assert!(stats.retransmission_timeout_ms >= 10.0);
        }
    }

    #[tokio::test]
    async fn retry_timeout_is_bounded_and_emits_abort() {
        let sink = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let destination = sink.local_addr().unwrap();
        let config = Config {
            chunk_bytes: 4096,
            udp_initial_rto_ms: 10,
            udp_max_retries: 2,
            udp_pacing_micros: 1,
            ..Default::default()
        };
        let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let started = Instant::now();
        let error = send_payload(
            &sender,
            destination,
            &config,
            SendPayload {
                handle: &handle(4096),
                transfer_id: Uuid::new_v4(),
                offset: 0,
                payload: &[7; 4096],
                read_response: false,
            },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("TRAINPOOL_TRANSFER_TIMEOUT"));
        assert!(started.elapsed() < Duration::from_secs(1));
        let mut kinds = Vec::new();
        for _ in 0..4 {
            let mut wire = [0_u8; MAX_DATAGRAM];
            let (size, _) =
                tokio::time::timeout(Duration::from_millis(50), sink.recv_from(&mut wire))
                    .await
                    .unwrap()
                    .unwrap();
            kinds.push(Packet::decode(&wire[..size], None).unwrap().kind);
        }
        assert_eq!(kinds.last(), Some(&Kind::Abort));
    }
}
