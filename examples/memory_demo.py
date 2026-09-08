"""Bounded RAM-fabric demonstration; no tensor files and no torch requirement."""

import argparse
import json

import blake3
from trainpool_torch import Client


def demonstrate(client, total_mib=12, block_kib=512, require_remote=True):
    plan = client.control("plan", stages=[])
    handles = []
    block_size = min(block_kib * 1024, client.chunk_bytes)
    total = total_mib * 1024 * 1024
    try:
        for offset in range(0, total, block_size):
            size = min(block_size, total - offset)
            with client.staging(size):
                data = bytearray([offset // block_size % 251]) * size
                handles.append(client.put(data, plan["job_id"]))
                del data
        remote_bytes = sum(h["size"] for h in handles if h["owner_node"] != client.local_node)
        if require_remote and remote_bytes == 0:
            raise RuntimeError("No remote RAM was used; lower the local daemon RAM limit or increase --mib")
        for handle in reversed(handles):
            with client.staging(handle["size"]):
                restored = bytearray(handle["size"])
                client.read_into(handle, restored)
                assert blake3.blake3(restored).hexdigest() == handle["checksum"]
                del restored
        metrics = client.control("metrics")["jobs"][plan["job_id"]]
        return {
            "job_id": plan["job_id"],
            "total_bytes": total,
            "remote_backing_bytes": remote_bytes,
            "blocks_verified": len(handles),
            "metrics": metrics,
            "disk_spill": False,
        }
    finally:
        for handle in handles:
            client.free(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mib", type=int, default=12)
    parser.add_argument("--block-kib", type=int, default=512)
    args = parser.parse_args()
    print(json.dumps(demonstrate(Client(), args.mib, args.block_kib), indent=2))
