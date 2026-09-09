"""Typed JSON control and bounded binary chunks; connections are local-only."""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import ipaddress
import json
import os
import socket
import struct
import uuid

import blake3

VERSION = 1
MAX_FRAME = 8 * 1024 * 1024


class TrainPoolError(RuntimeError):
    pass


def _receive_into(sock, buffer):
    view = memoryview(buffer).cast("B")
    while view:
        received = sock.recv_into(view)
        if received == 0:
            raise TrainPoolError("TRAINPOOL_TRANSFER_INTERRUPTED: connection closed")
        view = view[received:]


def _read_frame(sock):
    header = bytearray(4)
    _receive_into(sock, header)
    size = struct.unpack("!I", header)[0]
    if size > MAX_FRAME:
        raise TrainPoolError("control frame exceeds limit")
    data = bytearray(size)
    _receive_into(sock, data)
    wire = json.loads(data)
    if wire.get("protocol_version") != VERSION:
        raise TrainPoolError("TRAINPOOL_PROTOCOL_VERSION: incompatible daemon")
    return wire["message"]


def _write_frame(sock, message):
    payload = json.dumps({"protocol_version": VERSION, "message": message}, separators=(",", ":")).encode()
    if len(payload) > MAX_FRAME:
        raise TrainPoolError("control frame exceeds limit")
    sock.sendall(struct.pack("!I", len(payload)))
    sock.sendall(payload)


def _check(response):
    if not response.get("ok"):
        raise TrainPoolError(response.get("error") or "TrainPool protocol error")
    return response.get("data")


class Client:
    def __init__(self, address=None, *, cluster_name=None, cluster_secret=None, timeout=120):
        address = address or os.getenv("TRAINPOOL_ADDRESS", "127.0.0.1:7432")
        host, port = address.rsplit(":", 1)
        host = host.strip("[]")
        addresses = socket.getaddrinfo(host, int(port), type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(item[4][0]).is_loopback for item in addresses):
            raise ValueError("The Python SDK connects only to a local loopback daemon")
        self.address = (host, int(port))
        self.cluster_name = cluster_name or os.getenv("TRAINPOOL_CLUSTER_NAME", "trainpool")
        self.secret = cluster_secret if cluster_secret is not None else os.getenv("TRAINPOOL_CLUSTER_SECRET")
        self.timeout = timeout
        self.status = self.control("status")
        self.local_node = self.status["local_node"]
        self.chunk_bytes = min(
            [self.status["chunk_bytes"]] + [n["network"]["chunk_bytes"] for n in self.status["nodes"]]
        )

    def _signature(self, role, server, client):
        if self.secret is None:
            return None
        text = f"{role}|{server}|{client}|{self.cluster_name}".encode()
        return hmac.new(self.secret.encode(), text, hashlib.sha256).hexdigest()

    @contextlib.contextmanager
    def connect(self):
        with socket.create_connection(self.address, self.timeout) as sock:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            challenge = _read_frame(sock)
            nonce = str(uuid.uuid4())
            _write_frame(
                sock,
                {
                    "cluster_name": self.cluster_name,
                    "nonce": nonce,
                    "mac": self._signature("client", challenge["nonce"], nonce),
                },
            )
            accepted = _read_frame(sock)
            if accepted.get("ok") is False:
                _check(accepted)
            expected = self._signature("server", challenge["nonce"], nonce)
            actual = accepted.get("mac")
            if (expected is None and actual is not None) or (
                expected is not None and not hmac.compare_digest(expected, actual or "")
            ):
                raise TrainPoolError("TRAINPOOL_AUTH_FAILED")
            yield sock

    def control(self, op, **fields):
        with self.connect() as sock:
            _write_frame(sock, {"op": op, **fields})
            return _check(_read_frame(sock))

    @contextlib.contextmanager
    def staging(self, size):
        reservation = self.control("reserve_staging", bytes=size)
        try:
            yield
        finally:
            self.control("release_staging", reservation_id=reservation)

    def put(self, data, job_id, *, preferred_node=None):
        """Upload one bounded block. The caller accounts its staging buffer."""
        view = memoryview(data).cast("B")
        return self.put_chunks(
            len(view),
            (view[i : i + self.chunk_bytes] for i in range(0, len(view), self.chunk_bytes)),
            job_id,
            preferred_node=preferred_node,
        )

    def put_chunks(self, size, chunks, job_id, *, preferred_node=None):
        """Commit one block from bounded, caller-accounted contiguous chunks."""
        handle = self.control(
            "allocate",
            size=size,
            job_id=job_id,
            compute_node=self.local_node,
            preferred_node=preferred_node,
            tensor={"dtype": "uint8", "shape": [size], "layout": "contiguous", "byte_length": size},
        )
        transfer = str(uuid.uuid4())
        try:
            digest = blake3.blake3()
            offset = 0
            for data in chunks:
                chunk = memoryview(data).cast("B")
                if not chunk or len(chunk) > self.chunk_bytes or offset + len(chunk) > size:
                    raise ValueError("invalid streamed upload chunk")
                with self.connect() as sock:
                    _write_frame(
                        sock,
                        {
                            "op": "write_chunk",
                            "handle": handle,
                            "transfer_id": transfer,
                            "offset": offset,
                            "length": len(chunk),
                            "total_size": size,
                            "checksum": blake3.blake3(chunk).hexdigest(),
                            "direct": False,
                        },
                    )
                    _check(_read_frame(sock))
                    sock.sendall(chunk)
                    _check(_read_frame(sock))
                digest.update(chunk)
                offset += len(chunk)
                del chunk, data
            if offset != size:
                raise ValueError("streamed upload size mismatch")
            return self.control("commit", handle=handle, checksum=digest.hexdigest(), direct=False)
        except BaseException:
            try:
                self.control("free", handle=handle, direct=False)
            except TrainPoolError:
                pass
            raise

    def read_into(self, handle, buffer):
        view = memoryview(buffer).cast("B")
        if len(view) != handle["size"]:
            raise ValueError("destination length differs from block size")
        digest = blake3.blake3()
        transfer = str(uuid.uuid4())
        for offset in range(0, len(view), self.chunk_bytes):
            chunk = view[offset : offset + self.chunk_bytes]
            with self.connect() as sock:
                _write_frame(
                    sock,
                    {
                        "op": "read_chunk",
                        "handle": handle,
                        "transfer_id": transfer,
                        "offset": offset,
                        "length": len(chunk),
                        "direct": False,
                    },
                )
                metadata = _check(_read_frame(sock))
                if (
                    metadata["transfer_id"] != transfer
                    or metadata["object_id"] != handle["id"]
                    or metadata["offset"] != offset
                    or metadata["length"] != len(chunk)
                    or metadata["total_size"] != len(view)
                ):
                    raise TrainPoolError("invalid transfer metadata")
                _receive_into(sock, chunk)
                if blake3.blake3(chunk).hexdigest() != metadata["checksum"]:
                    raise TrainPoolError("TRAINPOOL_CHECKSUM_MISMATCH")
            digest.update(chunk)
        if digest.hexdigest() != handle["checksum"]:
            raise TrainPoolError("TRAINPOOL_CHECKSUM_MISMATCH: complete block")

    def read_chunks(self, handle, chunk_bytes=65536):
        """Yield verified payload pieces while holding only one staging reservation.

        The final full-block checksum is verified before this generator completes;
        consumers must exhaust it before exposing the reconstructed tensor.
        """
        digest = blake3.blake3()
        transfer = str(uuid.uuid4())
        chunk_bytes = min(chunk_bytes, self.chunk_bytes)
        for offset in range(0, handle["size"], chunk_bytes):
            length = min(chunk_bytes, handle["size"] - offset)
            with self.staging(length):
                buffer = bytearray(length)
                with self.connect() as sock:
                    _write_frame(
                        sock,
                        {
                            "op": "read_chunk",
                            "handle": handle,
                            "transfer_id": transfer,
                            "offset": offset,
                            "length": length,
                            "direct": False,
                        },
                    )
                    metadata = _check(_read_frame(sock))
                    expected = {
                        "transfer_id": transfer,
                        "object_id": handle["id"],
                        "offset": offset,
                        "length": length,
                        "total_size": handle["size"],
                    }
                    if any(metadata[key] != value for key, value in expected.items()):
                        raise TrainPoolError("invalid transfer metadata")
                    _receive_into(sock, buffer)
                    if blake3.blake3(buffer).hexdigest() != metadata["checksum"]:
                        raise TrainPoolError("TRAINPOOL_CHECKSUM_MISMATCH")
                digest.update(buffer)
                yield buffer
                del buffer
        if digest.hexdigest() != handle["checksum"]:
            raise TrainPoolError("TRAINPOOL_CHECKSUM_MISMATCH: complete block")

    def free(self, handle):
        self.control("free", handle=handle, direct=False)
