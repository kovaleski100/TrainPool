use std::{io, net::SocketAddr};
use tokio::{io::Interest, net::UdpSocket};

use super::udp::MAX_DATAGRAM;

pub const MAX_GSO_SEGMENTS: usize = 64;
const MAX_GSO_BYTES: usize = u16::MAX as usize;
const MAX_GRO_BYTES: usize = MAX_DATAGRAM * MAX_GSO_SEGMENTS;

pub struct ReceivedBatch {
    pub datagrams: Vec<Vec<u8>>,
    pub source: SocketAddr,
}

pub const fn max_gso_segments(segment_size: usize) -> usize {
    let by_bytes = MAX_GSO_BYTES / segment_size;
    if by_bytes < MAX_GSO_SEGMENTS {
        by_bytes
    } else {
        MAX_GSO_SEGMENTS
    }
}

/// Enables Linux UDP receive coalescing when available. Sending with GSO does
/// not require socket setup; unsupported kernels transparently use the regular
/// datagram path.
pub fn enable_gro(socket: &UdpSocket) -> bool {
    #[cfg(target_os = "linux")]
    {
        use std::{mem::size_of_val, os::fd::AsRawFd};

        let enabled: libc::c_int = 1;
        // SAFETY: the file descriptor belongs to a live UDP socket and the
        // option value points to a correctly sized integer for setsockopt.
        unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::IPPROTO_UDP,
                libc::UDP_GRO,
                (&raw const enabled).cast(),
                size_of_val(&enabled) as libc::socklen_t,
            ) == 0
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = socket;
        false
    }
}

/// Sends fixed-size encoded datagrams as one UDP GSO buffer on Linux. The
/// final segment may be shorter. Other platforms and unsupported paths retain
/// the portable one-send-per-datagram behavior.
pub async fn send_segmented(
    socket: &UdpSocket,
    destination: SocketAddr,
    encoded: &[u8],
    segment_size: usize,
    segments: usize,
) -> io::Result<()> {
    debug_assert!(segments > 0 && segments <= MAX_GSO_SEGMENTS);
    debug_assert!(segment_size <= u16::MAX as usize);
    debug_assert!(encoded.len() <= segment_size * segments);
    debug_assert!(encoded.len() > segment_size * (segments - 1));

    if segments > 1 && encoded.len() <= MAX_GSO_BYTES {
        #[cfg(target_os = "linux")]
        if send_gso(socket, destination, encoded, segment_size)
            .await
            .is_ok()
        {
            return Ok(());
        }
    }

    for datagram in encoded.chunks(segment_size) {
        socket.send_to(datagram, destination).await?;
    }
    Ok(())
}

/// Receives either a normal datagram or a GRO-coalesced buffer and restores
/// the original datagram boundaries before handing data to the protocol.
pub async fn receive_batch(socket: &UdpSocket) -> io::Result<ReceivedBatch> {
    loop {
        socket.readable().await?;
        match socket.try_io(Interest::READABLE, || receive_batch_now(socket)) {
            Ok(batch) => return Ok(batch),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            Err(error) => return Err(error),
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn receive_batch_now(socket: &UdpSocket) -> io::Result<ReceivedBatch> {
    let mut buffer = vec![0; MAX_DATAGRAM];
    let (size, source) = socket.try_recv_from(&mut buffer)?;
    buffer.truncate(size);
    Ok(ReceivedBatch {
        datagrams: vec![buffer],
        source,
    })
}

#[cfg(target_os = "linux")]
async fn send_gso(
    socket: &UdpSocket,
    destination: SocketAddr,
    encoded: &[u8],
    segment_size: usize,
) -> io::Result<()> {
    loop {
        socket.writable().await?;
        match socket.try_io(Interest::WRITABLE, || {
            send_gso_now(socket, destination, encoded, segment_size)
        }) {
            Ok(()) => return Ok(()),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => continue,
            Err(error) => return Err(error),
        }
    }
}

#[cfg(target_os = "linux")]
fn send_gso_now(
    socket: &UdpSocket,
    destination: SocketAddr,
    encoded: &[u8],
    segment_size: usize,
) -> io::Result<()> {
    use std::{mem, os::fd::AsRawFd};

    let destination = socket2::SockAddr::from(destination);
    let mut io_vector = libc::iovec {
        iov_base: encoded.as_ptr().cast_mut().cast(),
        iov_len: encoded.len(),
    };
    let control_size = unsafe { libc::CMSG_SPACE(mem::size_of::<u16>() as _) } as usize;
    let mut control = [0_usize; 8];
    debug_assert!(control_size <= mem::size_of_val(&control));
    let mut message: libc::msghdr = unsafe { mem::zeroed() };
    message.msg_name = destination.as_ptr().cast_mut().cast();
    message.msg_namelen = destination.len();
    message.msg_iov = &raw mut io_vector;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = control_size;

    // SAFETY: message owns aligned ancillary storage large enough for one u16
    // UDP_SEGMENT control value, and every pointer remains valid for sendmsg.
    unsafe {
        let header = libc::CMSG_FIRSTHDR(&message);
        if header.is_null() {
            return Err(io::Error::other(
                "failed to allocate UDP GSO control message",
            ));
        }
        (*header).cmsg_level = libc::IPPROTO_UDP;
        (*header).cmsg_type = libc::UDP_SEGMENT;
        (*header).cmsg_len = libc::CMSG_LEN(mem::size_of::<u16>() as _) as usize;
        libc::CMSG_DATA(header)
            .cast::<u16>()
            .write_unaligned(segment_size as u16);
        let sent = libc::sendmsg(socket.as_raw_fd(), &message, libc::MSG_DONTWAIT);
        if sent < 0 {
            return Err(io::Error::last_os_error());
        }
        if sent as usize != encoded.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "partial UDP GSO send",
            ));
        }
    }
    Ok(())
}

#[cfg(target_os = "linux")]
fn receive_batch_now(socket: &UdpSocket) -> io::Result<ReceivedBatch> {
    use std::{mem, os::fd::AsRawFd};

    let mut buffer = vec![0_u8; MAX_GRO_BYTES];
    let mut source = socket2::SockAddrStorage::zeroed();
    let mut io_vector = libc::iovec {
        iov_base: buffer.as_mut_ptr().cast(),
        iov_len: buffer.len(),
    };
    let mut control = [0_usize; 8];
    let mut message: libc::msghdr = unsafe { mem::zeroed() };
    message.msg_name = (&raw mut source).cast();
    message.msg_namelen = source.size_of();
    message.msg_iov = &raw mut io_vector;
    message.msg_iovlen = 1;
    message.msg_control = control.as_mut_ptr().cast();
    message.msg_controllen = mem::size_of_val(&control);

    // SAFETY: all msghdr buffers are initialized, writable, and remain alive
    // for the nonblocking recvmsg call.
    let received = unsafe { libc::recvmsg(socket.as_raw_fd(), &mut message, libc::MSG_DONTWAIT) };
    if received < 0 {
        return Err(io::Error::last_os_error());
    }
    if message.msg_flags & libc::MSG_TRUNC != 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "truncated UDP GRO buffer",
        ));
    }
    let received = received as usize;
    buffer.truncate(received);

    let mut segment_size = received;
    // SAFETY: recvmsg initialized the ancillary-data chain inside `control`;
    // the CMSG traversal macros stay within msg_controllen.
    unsafe {
        let mut header = libc::CMSG_FIRSTHDR(&message);
        while !header.is_null() {
            if (*header).cmsg_level == libc::IPPROTO_UDP
                && (*header).cmsg_type == libc::UDP_GRO
                && (*header).cmsg_len >= libc::CMSG_LEN(mem::size_of::<u16>() as _) as usize
            {
                segment_size = libc::CMSG_DATA(header).cast::<u16>().read_unaligned() as usize;
                break;
            }
            header = libc::CMSG_NXTHDR(&message, header);
        }
    }
    if segment_size == 0 || segment_size > MAX_DATAGRAM {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid UDP GRO segment size",
        ));
    }

    let address = unsafe { socket2::SockAddr::new(source, message.msg_namelen) };
    let source = address
        .as_socket()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "non-IP UDP source"))?;
    Ok(ReceivedBatch {
        datagrams: buffer.chunks(segment_size).map(<[u8]>::to_vec).collect(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn segmented_send_preserves_datagram_boundaries() {
        assert_eq!(max_gso_segments(1_366), 47);
        let receiver = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        enable_gro(&receiver);
        let sender = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let destination = receiver.local_addr().unwrap();
        let segment_size = 1_366;
        let mut encoded = vec![1; segment_size];
        encoded.extend(std::iter::repeat_n(2, segment_size));
        encoded.extend(std::iter::repeat_n(3, 700));

        send_segmented(&sender, destination, &encoded, segment_size, 3)
            .await
            .unwrap();

        let mut datagrams = Vec::new();
        while datagrams.len() < 3 {
            let batch = tokio::time::timeout(Duration::from_secs(1), receive_batch(&receiver))
                .await
                .unwrap()
                .unwrap();
            assert_eq!(batch.source, sender.local_addr().unwrap());
            datagrams.extend(batch.datagrams);
        }
        assert_eq!(datagrams.len(), 3);
        assert_eq!(datagrams[0], vec![1; segment_size]);
        assert_eq!(datagrams[1], vec![2; segment_size]);
        assert_eq!(datagrams[2], vec![3; 700]);
    }
}
