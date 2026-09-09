"""Record a real CUDA OOM comparison against the identical segmentation script.

Run on the GPU host with both physical-machine daemons already running. No
software GPU residency limits, remote shell, synthetic OOM, or CPU simulation.
Only experiment reports/checkpoints requested by the user are written to disk.
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from trainpool_torch import Client

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "deeplab"], default="unet")
    parser.add_argument("--binary", default=str(ROOT / "target/release/trainpool"))
    parser.add_argument("--address", default="127.0.0.1:7432")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--optimizer", choices=["SGD", "Adam", "AdamW"], default="AdamW")
    parser.add_argument("--timeout", type=int, default=7200)
    parser.add_argument(
        "--local-fabric",
        action="store_true",
        help="Real CUDA with same-host daemons; never passes the physical-machine gate",
    )
    parser.add_argument(
        "--allow-fitting-baseline",
        action="store_true",
        help="CUDA correctness check only; never passes the OOM gate",
    )
    args = parser.parse_args()
    if os.getenv("TRAINPOOL_TEST_CPU") or os.getenv("TRAINPOOL_ACTIVE"):
        raise RuntimeError("Run this harness outside the launcher with CPU simulation disabled")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Physical validation unavailable: CUDA driver/device is not accessible")
    args.output.mkdir(parents=True, exist_ok=False)
    client = Client(args.address)
    status = client.control("status")
    nodes = status["nodes"]
    if not args.local_fabric and len({node["hostname"] for node in nodes}) < 2:
        raise RuntimeError(
            "At least two distinct physical hostnames must be present; verify physical machines manually"
        )
    plan = client.control("plan", stages=[])
    if len(plan["gpu_assignments"]) != 1 or plan["compute_nodes"] != [client.local_node]:
        raise RuntimeError("Run on the selected primary GPU host")
    gpu = plan["gpu_assignments"][0]["gpu_id"]
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    command = [
        sys.executable,
        str(ROOT / "tests/models" / f"{args.model}_train.py"),
        "--device",
        "cuda",
        "--width",
        str(args.width),
        "--size",
        str(args.size),
        "--batch",
        str(args.batch),
        "--steps",
        str(args.steps),
        "--optimizer",
        args.optimizer,
    ]

    def node_metrics():
        records = {}
        for node in nodes:
            proc = subprocess.run(
                [args.binary, "--address", node["network"]["control_address"], "metrics", "--json"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            records[node["node_id"]] = (
                json.loads(proc.stdout) if proc.returncode == 0 else {"error": proc.stderr}
            )
        return records

    report = {
        "kind": "real CUDA / local multiprocess fabric" if args.local_fabric else "physical CUDA experiment",
        "physical_host_verification": (
            "same-host fabric; separate-machine gate not exercised"
            if args.local_fabric
            else "operator must attest separate machines"
        ),
        "status_before": status,
        "plan": plan,
        "command": command,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ROOT / "tests/models").glob("*_train.py")
        },
        "gpu_uuid": gpu,
        "cuda_visible_devices": gpu,
        "torch_version": torch.__version__,
        "topology": client.control("benchmark", bytes=16 * 1024 * 1024),
        "metrics_before": node_metrics(),
        "release_gate_passed": False,
    }
    report["nvidia_smi"] = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True).stdout

    def run(name, argv):
        started = time.monotonic()
        with (
            (args.output / f"{name}.stdout").open("w") as stdout,
            (args.output / f"{name}.stderr").open("w") as stderr,
        ):
            process = subprocess.run(
                argv, env=environment, stdout=stdout, stderr=stderr, timeout=args.timeout
            )
        return {
            "exit_code": process.returncode,
            "seconds": time.monotonic() - started,
            "stdout": (args.output / f"{name}.stdout").read_text(),
            "stderr": (args.output / f"{name}.stderr").read_text(),
        }

    try:
        baseline = report["baseline"] = run("baseline", command)
        oom = baseline["exit_code"] != 0 and "cuda out of memory" in baseline["stderr"].lower()
        report["real_baseline_cuda_oom"] = oom
        if not oom and not args.allow_fitting_baseline:
            raise RuntimeError(
                "Baseline did not produce a real CUDA allocation OOM; this is not an OOM-overcome result"
            )
        # Let the daemon's periodic GPU inventory refresh after the baseline
        # process releases its CUDA context, including an OOM teardown.
        time.sleep(12)
        report["trainpool"] = run("trainpool", [args.binary, "--address", args.address, *command])
        report["metrics_after"] = node_metrics()
        report["status_after"] = client.control("status")
        before = report["metrics_before"].get(client.local_node, {}).get("jobs", {})
        after = report["metrics_after"].get(client.local_node, {}).get("jobs", {})
        new_jobs = {key: value for key, value in after.items() if key not in before}
        remote_read = sum(job["bytes_remote_ram_to_local"] for job in new_jobs.values())
        remote_written = sum(job["bytes_local_to_remote_ram"] for job in new_jobs.values())
        report["remote_bytes_read"] = remote_read
        report["remote_bytes_written"] = remote_written
        training = report["trainpool"]
        measurements = json.loads(training["stdout"].splitlines()[-1]) if training["exit_code"] == 0 else {}
        report["measurements"] = measurements
        finite_steps = len(measurements.get("loss", [])) == args.steps and all(
            math.isfinite(value) for value in measurements.get("loss", [])
        )
        report["cuda_experiment_passed"] = (
            training["exit_code"] == 0 and finite_steps and remote_read > 0 and remote_written > 0
        )
        report["release_gate_passed"] = report["cuda_experiment_passed"] and oom and not args.local_fabric
        if not report["cuda_experiment_passed"]:
            raise RuntimeError("TrainPool did not complete with demonstrated remote reads and writes")
    finally:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(f"CUDA experiment succeeded; evidence: {args.output / 'report.json'}")
    if args.local_fabric:
        print("Separate-machine release gate remains pending: this run used a local fabric.")
    else:
        print("Confirm physical machine identity independently before accepting the hardware gate.")


if __name__ == "__main__":
    main()
