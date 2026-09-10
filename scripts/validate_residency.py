"""Physical VRAM/local-RAM/remote-RAM residency probe.

This exercises TensorStore ownership without CUDA kernels. It must target a
disposable/current-version local daemon on the physical GPU host.
"""

import argparse
import json

import torch
from trainpool_torch import Client, TensorStore

MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="127.0.0.1:7432")
    parser.add_argument("--mode", choices=("vram", "local", "remote"), required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("physical CUDA is required")
    client = Client(args.address)
    with TensorStore(client, block_bytes=MIB) as store:
        assignment = store.plan["gpu_assignments"][0]
        store.configure_gpu(torch.device("cuda", 0), assignment["usable_bytes"])
        handles = []
        chunk = 64 * MIB
        limit = assignment["usable_bytes"] + sum(store.plan["memory_budgets"].values())
        while True:
            handle = store.offload(torch.zeros(chunk, dtype=torch.uint8), expected_next_use=None)
            handles.append(handle)
            local = sum(
                block["size"]
                for item in handles
                for block in item.blocks
                if block["owner_node"] == client.local_node
            )
            remote = sum(
                block["size"]
                for item in handles
                for block in item.blocks
                if block["owner_node"] != client.local_node
            )
            if args.mode == "vram" or (args.mode == "local" and local >= 64 * MIB) or remote > 0:
                break
            if any(item.blocks for item in handles):
                # Approach RAM tier boundaries without requiring a large remote
                # block; providers reserve 256 KiB of relay headroom.
                chunk = MIB
            if sum(item.size for item in handles) + chunk > limit:
                raise RuntimeError(f"could not reach requested {args.mode} tier within plan capacity")
        store.flush_metrics()
        metrics = client.control("metrics")["jobs"][store.job_id]
        expected = {
            "vram": (True, False, False),
            "local": (True, True, False),
            "remote": (True, True, True),
        }[args.mode]
        observed = (
            metrics["current_vram_resident_bytes"] > 0,
            metrics["current_local_ram_backing_bytes"] > 0,
            metrics["current_remote_ram_backing_bytes"] > 0,
        )
        if observed != expected:
            raise RuntimeError(f"expected tiers {expected}, observed {observed}")
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "job_id": store.job_id,
                    "allocated_bytes": sum(item.size for item in handles),
                    "metrics": metrics,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
