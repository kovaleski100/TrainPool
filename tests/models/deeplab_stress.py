"""Stress a physical TrainPool fabric with real DeepLabV3 training state.

Run this file through the TrainPool launcher, never with plain Python. The
workload keeps multiple independent DeepLabV3/ResNet50 models and AdamW states
alive in one job, then trains them round-robin. This grows real model-backed RAM
residency while repeatedly exercising GPU/RAM transfers.

Example:
    trainpool --address 127.0.0.1:7432 python tests/models/deeplab_stress.py
"""

import argparse
import json
import math
import os
import time

import torch
from torch.nn import functional as F
from torchvision.models.segmentation import deeplabv3_resnet50
from trainpool_torch import Client

MIB = 1024 * 1024


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256, help="Square synthetic image size")
    parser.add_argument("--batch", type=int, default=2, help="Images per optimizer step")
    parser.add_argument("--steps", type=int, default=3, help="Steps per model replica")
    parser.add_argument("--replicas", type=int, default=0, help="0 chooses from remote capacity")
    parser.add_argument("--max-replicas", type=int, default=8, help="Safety cap for auto selection")
    parser.add_argument(
        "--target-remote-fraction",
        type=float,
        default=0.80,
        help="Fraction of currently available remote pool RAM targeted in auto mode",
    )
    parser.add_argument(
        "--remote-reserve-mib",
        type=int,
        default=512,
        help="Remote pool RAM kept unused by the auto-sizing estimate",
    )
    parser.add_argument("--classes", type=int, default=3)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="AdamW")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--report-every", type=int, default=1)
    parser.add_argument(
        "--hold-seconds",
        type=int,
        default=30,
        help="Keep trained state resident so metrics can be inspected",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=int,
        default=0,
        help="Stop before starting another step after this duration; 0 disables the limit",
    )
    parser.add_argument(
        "--aux-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Train the auxiliary DeepLab classifier too",
    )
    parser.add_argument("--seed", type=int, default=321)
    return parser.parse_args()


def validate_args(args):
    for name in ("size", "batch", "steps", "max_replicas", "report_every", "classes"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be at least 1")
    if args.replicas < 0:
        raise ValueError("--replicas cannot be negative")
    if not 0.05 <= args.target_remote_fraction <= 0.90:
        raise ValueError("--target-remote-fraction must be between 0.05 and 0.90")
    if args.remote_reserve_mib < 0 or args.hold_seconds < 0 or args.max_runtime_seconds < 0:
        raise ValueError("reserve and duration arguments cannot be negative")
    if args.batch < 2:
        raise ValueError("DeepLab training needs --batch >= 2 because ASPP uses BatchNorm")


def make_model(args):
    return deeplabv3_resnet50(
        weights=None,
        weights_backbone=None,
        aux_loss=args.aux_loss,
        num_classes=args.classes,
    )


def model_state_estimate(model, optimizer):
    parameter_bytes = sum(value.numel() * value.element_size() for value in model.parameters())
    buffer_bytes = sum(value.numel() * value.element_size() for value in model.buffers())
    # Adam/AdamW retain parameters plus first and second moments after a step.
    # The extra quarter accounts for buffers, scalar state and allocator granularity;
    # gradients are transient and must not inflate the remote residency target.
    multiplier = 3.25 if optimizer in {"Adam", "AdamW"} else 1.25
    return parameter_bytes, buffer_bytes, parameter_bytes * multiplier + buffer_bytes


def remote_nodes(status):
    local_node = status["local_node"]
    return [node for node in status["nodes"] if node["node_id"] != local_node]


def choose_replicas(args, remote_available, estimated_replica_bytes):
    if args.replicas:
        return args.replicas, None
    reserve = args.remote_reserve_mib * MIB
    target = min(int(remote_available * args.target_remote_fraction), max(0, remote_available - reserve))
    if target <= 0:
        raise RuntimeError("Remote pool has no capacity after the configured safety reserve")
    replicas = max(1, math.ceil(target / estimated_replica_bytes))
    return min(replicas, args.max_replicas), target


def snapshot(client, job_id, baseline_remote_used):
    status = client.control("status")
    remotes = remote_nodes(status)
    remote_used = sum(node["memory"]["trainpool_ram_used"] for node in remotes)
    remote_budget = sum(node["memory"]["trainpool_ram_budget"] for node in remotes)
    job = client.control("metrics")["jobs"].get(job_id, {})
    return {
        "current_vram_resident_bytes": job.get("current_vram_resident_bytes", 0),
        "peak_vram_resident_bytes": job.get("peak_vram_resident_bytes", 0),
        "current_local_ram_backing_bytes": job.get("current_local_ram_backing_bytes", 0),
        "peak_local_ram_backing_bytes": job.get("peak_local_ram_backing_bytes", 0),
        "current_remote_ram_backing_bytes": job.get("current_remote_ram_backing_bytes", 0),
        "peak_remote_ram_backing_bytes": job.get("peak_remote_ram_backing_bytes", 0),
        "eviction_count": job.get("eviction_count", 0),
        "prefetch_count": job.get("prefetch_count", 0),
        "remote_ram_used_mib": round(remote_used / MIB, 2),
        "remote_ram_delta_mib": round(max(0, remote_used - baseline_remote_used) / MIB, 2),
        "remote_ram_budget_mib": round(remote_budget / MIB, 2),
        "remote_written_mib": round(job.get("bytes_local_to_remote_ram", 0) / MIB, 2),
        "remote_read_mib": round(job.get("bytes_remote_ram_to_local", 0) / MIB, 2),
        "network_mib": round(job.get("network_bytes", 0) / MIB, 2),
        "network_wait_seconds": round(job.get("network_wait_ms", 0) / 1000, 3),
        "failed_transfers": job.get("failed_transfers", 0),
        "cuda_allocated_mib": round(torch.cuda.memory_allocated() / MIB, 2),
        "cuda_reserved_mib": round(torch.cuda.memory_reserved() / MIB, 2),
        "cuda_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / MIB, 2),
        "cuda_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / MIB, 2),
    }


def emit(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def train_step(model, optimizer, args, replica, step):
    started = time.perf_counter()
    images = torch.randn(args.batch, 3, args.size, args.size, device="cuda")
    masks = torch.randint(args.classes, (args.batch, args.size, args.size), device="cuda")
    optimizer.zero_grad()
    outputs = model(images)
    loss = F.cross_entropy(outputs["out"], masks)
    if args.aux_loss:
        loss = loss + 0.4 * F.cross_entropy(outputs["aux"], masks)
    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite loss in replica {replica}, step {step}")
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    return loss.item(), time.perf_counter() - started


def main():
    args = parse_args()
    validate_args(args)
    if os.getenv("TRAINPOOL_ACTIVE") != "1":
        raise RuntimeError("Run with: trainpool python tests/models/deeplab_stress.py")
    if not torch.cuda.is_available():
        raise RuntimeError("A physical CUDA device is required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.cuda.reset_peak_memory_stats()
    client = Client()
    job_id = os.environ["TRAINPOOL_JOB_ID"]
    initial_status = client.control("status")
    remotes = remote_nodes(initial_status)
    if not remotes:
        raise RuntimeError("No remote TrainPool node is visible; start the local daemon with --seed")
    remote_available = sum(node["memory"]["trainpool_ram_available"] for node in remotes)
    baseline_remote_used = sum(node["memory"]["trainpool_ram_used"] for node in remotes)

    prototype = make_model(args)
    parameter_bytes, buffer_bytes, estimated_replica_bytes = model_state_estimate(prototype, args.optimizer)
    replicas, target = choose_replicas(args, remote_available, estimated_replica_bytes)
    del prototype
    emit(
        "configuration",
        device=torch.cuda.get_device_name(),
        gpu_vram_mib=round(torch.cuda.get_device_properties(0).total_memory / MIB, 2),
        remote_hosts=[node["hostname"] for node in remotes],
        remote_available_mib=round(remote_available / MIB, 2),
        target_remote_mib=round(target / MIB, 2) if target is not None else None,
        replicas=replicas,
        estimated_replica_state_mib=round(estimated_replica_bytes / MIB, 2),
        parameter_mib=round(parameter_bytes / MIB, 2),
        buffer_mib=round(buffer_bytes / MIB, 2),
        image_size=args.size,
        batch=args.batch,
        steps_per_replica=args.steps,
        optimizer=args.optimizer,
    )

    models = []
    optimizers = []
    losses = []
    started = time.monotonic()
    completed_steps = 0
    deadline_reached = False
    try:
        # Create and warm each replica separately. The first optimizer step materializes
        # Adam/AdamW moments, so this phase grows real persistent training state.
        for replica in range(replicas):
            if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                deadline_reached = True
                break
            model = make_model(args).train().to("cuda")
            optimizer = getattr(torch.optim, args.optimizer)(
                model.parameters(), lr=args.learning_rate, foreach=False
            )
            models.append(model)
            optimizers.append(optimizer)
            loss, seconds = train_step(model, optimizer, args, replica, 0)
            losses.append(loss)
            completed_steps += 1
            emit(
                "warmup",
                replica=replica + 1,
                replicas=replicas,
                loss=loss,
                step_seconds=round(seconds, 3),
                **snapshot(client, job_id, baseline_remote_used),
            )

        for step in range(1, args.steps):
            for replica, (model, optimizer) in enumerate(zip(models, optimizers, strict=True)):
                if args.max_runtime_seconds and time.monotonic() - started >= args.max_runtime_seconds:
                    deadline_reached = True
                    break
                loss, seconds = train_step(model, optimizer, args, replica, step)
                losses.append(loss)
                completed_steps += 1
                if completed_steps % args.report_every == 0:
                    emit(
                        "step",
                        replica=replica + 1,
                        step=step + 1,
                        loss=loss,
                        step_seconds=round(seconds, 3),
                        **snapshot(client, job_id, baseline_remote_used),
                    )
            if deadline_reached:
                break

        final = snapshot(client, job_id, baseline_remote_used)
        emit(
            "summary",
            completed_steps=completed_steps,
            elapsed_seconds=round(time.monotonic() - started, 3),
            min_loss=min(losses),
            max_loss=max(losses),
            deadline_reached=deadline_reached,
            **final,
        )
        if final["remote_written_mib"] <= 0 or final["remote_read_mib"] <= 0:
            raise RuntimeError("No remote transfers were measured; lower the local daemon --ram-limit-mib")

        hold_deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < hold_deadline:
            remaining = max(0, math.ceil(hold_deadline - time.monotonic()))
            emit("holding", seconds_remaining=remaining, **snapshot(client, job_id, baseline_remote_used))
            time.sleep(min(5, remaining))
    except torch.OutOfMemoryError:
        emit("cuda_oom", **snapshot(client, job_id, baseline_remote_used))
        raise


if __name__ == "__main__":
    main()
