"""Run on the GPU machine. Defaults are small; size arguments scale the workload."""

import argparse
import json

import torch
import trainpool_torch as tp
from torch import nn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=24)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--resident-budget-mib", type=int, default=64)
    parser.add_argument("--block-mib", type=int, default=1)
    parser.add_argument("--prefetch-depth", type=int, choices=[0, 1], default=1)
    parser.add_argument("--memory-node", help="Optional RAM provider UUID")
    parser.add_argument(
        "--test-cpu", action="store_true", help="Explicit simulation backend; never a CUDA fallback"
    )
    args = parser.parse_args()
    device = "cpu" if args.test_cpu else "cuda"
    torch.manual_seed(42)
    # Each nested Sequential is an explicit recomputation/placement boundary.
    original = nn.Sequential(
        *[nn.Sequential(nn.Linear(args.width, args.width), nn.Tanh()) for _ in range(args.layers)]
    )
    parameter_bytes = sum(p.numel() * p.element_size() for p in original.parameters())
    budget = args.resident_budget_mib * 1024 * 1024
    if parameter_bytes <= budget:
        raise ValueError(
            "Increase --layers/--width or lower --resident-budget-mib so parameters exceed the configured residency budget"
        )
    optimizer = torch.optim.AdamW(original.parameters(), lr=1e-3)
    with tp.TensorStore(preferred_node=args.memory_node, block_bytes=args.block_mib * 1024 * 1024) as store:
        model, optimizer = tp.prepare(
            original,
            optimizer,
            store=store,
            device=device,
            _test_cpu=args.test_cpu,
            prefetch_depth=args.prefetch_depth,
            gpu_budget_bytes=budget,
        )
        try:
            for step in range(args.steps):
                optimizer.zero_grad()
                inputs = torch.randn(args.batch, args.width, device=device)
                loss = model(inputs).square().mean()
                loss.backward()
                optimizer.step()
                remote = sum(
                    block["size"]
                    for tensor in store.tensors.values()
                    for block in tensor.blocks
                    if block["owner_node"] != store.client.local_node
                )
                print(
                    json.dumps(
                        {
                            "step": step,
                            "loss": loss.item(),
                            "parameter_bytes": parameter_bytes,
                            "configured_residency_budget": budget,
                            "remote_backing_bytes": remote,
                            "peak_cuda_allocated": torch.cuda.max_memory_allocated()
                            if device == "cuda"
                            else None,
                            "backend": "explicit CPU simulation" if args.test_cpu else "CUDA",
                            "disk_spill": False,
                        }
                    )
                )
            metrics = store.client.control("metrics")["jobs"][store.job_id]
            if metrics["bytes_local_to_remote_ram"] == 0 or metrics["bytes_remote_ram_to_local"] == 0:
                raise RuntimeError(
                    "Remote RAM was not used; lower local --ram-limit-mib or choose --memory-node"
                )
            print(json.dumps({"job_id": store.job_id, "metrics": metrics}, indent=2))
        finally:
            model.prefetch.close()


if __name__ == "__main__":
    main()
