#!/usr/bin/env python3
"""Reproducible TrainPool inter-node data-plane benchmark.

Run once with both daemons configured for UDP and once for TCP. The compute
daemon must have an intentional RAM contribution limit so strict local-first
placement chooses the RAM-only peer; otherwise this script fails explicitly.
That limit must still fit one transfer chunk plus relay headroom (32 MB for the
default 16 MB chunk and 64 MB minimum object).
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time

from trainpool_torch import Client

COUNTERS = (
    "network_bytes",
    "network_wait_ms",
    "udp_datagrams_sent",
    "udp_datagrams_received",
    "udp_payload_bytes",
    "udp_retransmitted_datagrams",
    "udp_retransmitted_bytes",
    "ack_count",
    "nack_count",
    "tcp_connections_opened",
    "tcp_connections_reused",
)


def process_cpu_seconds(pid):
    if pid is None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_utime + usage.ru_stime
    try:
        fields = open(f"/proc/{pid}/stat", encoding="utf-8").read().split()  # noqa: SIM115
    except (FileNotFoundError, PermissionError):
        return None
    ticks = os.sysconf("SC_CLK_TCK")
    return (int(fields[13]) + int(fields[14])) / ticks


def metrics(client, job_id):
    return client.control("metrics").get("jobs", {}).get(job_id, {})


def delta(before, after, key):
    return after.get(key, 0) - before.get(key, 0)


def chunks(total, chunk_bytes, byte=0xA5):
    full = bytes([byte]) * chunk_bytes
    remaining = total
    while remaining:
        size = min(remaining, chunk_bytes)
        yield full if size == chunk_bytes else full[:size]
        remaining -= size


def benchmark(client, size, chunk_bytes, label, daemon_pid):
    plan = client.control("plan", stages=[])
    job_id = plan["job_id"]
    before = metrics(client, job_id)
    client_cpu_before = process_cpu_seconds(None)
    daemon_cpu_before = process_cpu_seconds(daemon_pid)
    started = time.perf_counter()
    handle = client.put_chunks(size, chunks(size, chunk_bytes), job_id)
    if handle["owner_node"] == client.local_node:
        client.free(handle)
        raise RuntimeError(
            "benchmark did not cross the machine boundary; lower the compute "
            "daemon --ram-limit-mib and keep adequate capacity on the RAM peer"
        )
    received = 0
    for piece in client.read_chunks(handle, chunk_bytes=chunk_bytes):
        received += len(piece)
    wall = time.perf_counter() - started
    client.free(handle)
    client_cpu = process_cpu_seconds(None) - client_cpu_before
    daemon_cpu_after = process_cpu_seconds(daemon_pid)
    after = metrics(client, job_id)
    result = {
        "label": label,
        "bytes": size,
        "decimal_mb": size / 1_000_000,
        "owner_node": handle["owner_node"],
        "verified_bytes": received,
        "wall_seconds": wall,
        "effective_mbps": size * 2 * 8 / wall / 1_000_000,
        "client_cpu_seconds": client_cpu,
        "client_cpu_percent_of_one_core": client_cpu / wall * 100,
    }
    if daemon_cpu_before is not None and daemon_cpu_after is not None:
        daemon_cpu = daemon_cpu_after - daemon_cpu_before
        result["compute_daemon_cpu_seconds"] = daemon_cpu
        result["compute_daemon_cpu_percent_of_one_core"] = daemon_cpu / wall * 100
    result.update({key: delta(before, after, key) for key in COUNTERS})
    result["estimated_rtt_ms"] = after.get("estimated_rtt_ms")
    result["retransmission_timeout_ms"] = after.get("retransmission_timeout_ms")
    result["packet_loss_estimate"] = after.get("packet_loss_estimate")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default=os.getenv("TRAINPOOL_ADDRESS", "127.0.0.1:7432"))
    parser.add_argument("--label", required=True, choices=("udp", "tcp"))
    parser.add_argument("--sizes-mb", nargs="+", type=int, default=(64, 256, 1024))
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--daemon-pid", type=int)
    args = parser.parse_args()
    client = Client(args.address)
    results = []
    for megabytes in args.sizes_mb:
        try:
            results.append(
                benchmark(
                    client,
                    megabytes * 1_000_000,
                    min(args.chunk_mb * 1_000_000, client.chunk_bytes),
                    args.label,
                    args.daemon_pid,
                )
            )
        except Exception as error:  # Continue larger matrix with an explicit failure row.
            results.append({"label": args.label, "decimal_mb": megabytes, "error": str(error)})
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
