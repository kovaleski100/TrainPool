import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from trainpool_torch import Client

SCRIPT = """\
import argparse
import json
import torch
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument("--device", required=True)
args = parser.parse_args()

torch.manual_seed(123)
model = nn.Sequential(
    nn.Linear(150, 512),
    nn.GELU(),
    nn.Linear(512, 150),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)
model_id = id(model)
optimizer_id = id(optimizer)

model = model.to(args.device)
# dtype-only movement must retain normal PyTorch semantics after attachment too.
model.to(dtype=torch.float64)

last_gradient = None
for _ in range(2):
    optimizer.zero_grad()
    value = torch.randn(5, 150, dtype=torch.float64, requires_grad=True)
    loss = model(value).square().mean()
    loss.backward()
    last_gradient = value.grad.detach().clone()
    optimizer.step()

state = model.state_dict()
print(json.dumps({
    "model_identity": id(model) == model_id,
    "optimizer_identity": id(optimizer) == optimizer_id,
    "param_groups_preserved": len(optimizer.param_groups[0]["params"]) == 4,
    "parameter_sum": sum(tensor.double().sum().item() for tensor in state.values()),
    "gradient_sum": last_gradient.sum().item(),
}))
"""


def _run(argv, *, environment):
    result = subprocess.run(
        argv,
        env=environment,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    return result


def _free_port(kind=socket.SOCK_STREAM):
    with socket.socket(type=kind) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def automatic_cluster(tmp_path):
    root = Path(__file__).resolve().parents[2]
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    ports = [_free_port(), _free_port()]
    discovery_port = _free_port(socket.SOCK_DGRAM)
    processes = []
    try:
        for index, (port, limit) in enumerate(zip(ports, [2, 32], strict=True)):
            processes.append(
                subprocess.Popen(
                    [
                        binary,
                        "--data-dir",
                        str(tmp_path / str(index)),
                        "daemon",
                        "--listen",
                        f"127.0.0.1:{port}",
                        "--advertise-ip",
                        "127.0.0.1",
                        "--multicast-interface",
                        "127.0.0.1",
                        "--discovery-port",
                        str(discovery_port),
                        "--ram-limit-mib",
                        str(limit),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                clients = [Client(f"127.0.0.1:{port}") for port in ports]
                if all(len(client.status["nodes"]) == 2 for client in clients):
                    yield clients
                    return
            except (OSError, RuntimeError, KeyError):
                pass
            time.sleep(0.2)
        raise RuntimeError("automatic-placement cluster did not converge")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def test_zero_import_script_runs_unchanged_with_identity_and_numerical_parity(automatic_cluster, tmp_path):
    clients = automatic_cluster
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "ordinary.py"
    script.write_text(SCRIPT)
    source = script.read_text()
    for forbidden in (
        "trainpool_torch",
        "TensorStore",
        "prepare(",
        "distributed_memory",
        "preferred_node",
    ):
        assert forbidden not in source

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    baseline_environment = environment.copy()
    baseline_environment.pop("TRAINPOOL_ACTIVE", None)
    baseline = _run(
        [sys.executable, str(script), "--device", "cpu"],
        environment=baseline_environment,
    )

    environment["TRAINPOOL_TEST_CPU"] = "1"
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    transparent = _run(
        [
            binary,
            "--address",
            f"{clients[0].address[0]}:{clients[0].address[1]}",
            sys.executable,
            str(script),
            "--device",
            "cuda",
        ],
        environment=environment,
    )

    expected = json.loads(baseline.stdout)
    actual = json.loads(transparent.stdout)
    assert actual["model_identity"]
    assert actual["optimizer_identity"]
    assert actual["param_groups_preserved"]
    assert abs(actual["parameter_sum"] - expected["parameter_sum"]) < 1e-10
    assert abs(actual["gradient_sum"] - expected["gradient_sum"]) < 1e-10
    assert "compatibility=FULL" in transparent.stderr
    metrics = clients[0].control("metrics")["jobs"]
    assert any(job["bytes_local_to_remote_ram"] > 0 for job in metrics.values())
    assert any(job["bytes_remote_ram_to_local"] > 0 for job in metrics.values())


def test_unsupported_graph_fails_before_cuda_materialization(cluster, tmp_path):
    clients, _ = cluster
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "unsupported.py"
    script.write_text(
        """\
import torch
from torch import nn

shared = nn.Linear(4, 4)
model = nn.Sequential(shared, shared)
model.to("cuda")
"""
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    environment["TRAINPOOL_TEST_CPU"] = "1"
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    result = subprocess.run(
        [
            binary,
            "--address",
            f"{clients[0].address[0]}:{clients[0].address[1]}",
            sys.executable,
            str(script),
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "TRAINPOOL_UNSUPPORTED_GRAPH" in result.stderr


def test_cuda_method_and_optimizer_after_model_are_intercepted(cluster, tmp_path):
    clients, _ = cluster
    root = Path(__file__).resolve().parents[2]
    script = tmp_path / "cuda_method.py"
    script.write_text(
        """\
import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 2))
identity = id(model)
model = model.cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer.zero_grad()
loss = model(torch.randn(3, 4)).square().mean()
loss.backward()
optimizer.step()
print(id(model) == identity, all(parameter.device.type == "meta" for parameter in model.parameters()))
"""
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    environment["TRAINPOOL_TEST_CPU"] = "1"
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    result = _run(
        [
            binary,
            "--address",
            f"{clients[0].address[0]}:{clients[0].address[1]}",
            sys.executable,
            str(script),
        ],
        environment=environment,
    )
    assert result.stdout.strip() == "True True"


@pytest.fixture(scope="module")
def segmentation_cluster(tmp_path_factory):
    directory = tmp_path_factory.mktemp("segmentation-fabric")
    root = Path(__file__).resolve().parents[2]
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    ports = [_free_port(), _free_port()]
    discovery = _free_port(socket.SOCK_DGRAM)
    processes = []
    try:
        for index, limit in enumerate((1, 1536)):
            processes.append(
                subprocess.Popen(
                    [
                        binary,
                        "--data-dir",
                        str(directory / str(index)),
                        "daemon",
                        "--listen",
                        f"127.0.0.1:{ports[index]}",
                        "--advertise-ip",
                        "127.0.0.1",
                        "--multicast-interface",
                        "127.0.0.1",
                        "--discovery-port",
                        str(discovery),
                        "--ram-limit-mib",
                        str(limit),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                clients = [Client(f"127.0.0.1:{port}") for port in ports]
                if all(len(client.status["nodes"]) == 2 for client in clients):
                    yield clients
                    return
            except (OSError, RuntimeError):
                pass
            time.sleep(0.2)
        raise RuntimeError("segmentation cluster discovery timeout")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


@pytest.mark.parametrize("script", ["unet_train.py", "deeplab_train.py"])
def test_ordinary_segmentation_scripts_via_launcher(segmentation_cluster, tmp_path, script):
    import torch

    clients = segmentation_cluster
    root = Path(__file__).resolve().parents[2]
    for path in (root / "tests/models").glob("*_train.py"):
        source = path.read_text()
        for forbidden in (
            "trainpool_torch",
            "TensorStore",
            "prepare(",
            "preferred_node",
            "distributed_memory",
        ):
            assert forbidden not in source
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    environment.pop("TRAINPOOL_ACTIVE", None)
    args = [
        sys.executable,
        str(root / "tests/models" / script),
        "--size",
        "32" if script == "deeplab_train.py" else "16",
        "--batch",
        "4" if script == "deeplab_train.py" else "2",
        "--steps",
        "2",
        "--width",
        "32",
    ]
    baseline = subprocess.run(
        args + ["--device", "cpu", "--checkpoint", str(tmp_path / "baseline.pt")],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert baseline.returncode == 0, baseline.stderr
    environment["TRAINPOOL_TEST_CPU"] = "1"
    binary = os.getenv("TRAINPOOL_BINARY", str(root / "target/debug/trainpool"))
    result = subprocess.run(
        [
            binary,
            "--address",
            f"{clients[0].address[0]}:{clients[0].address[1]}",
            *args,
            "--device",
            "cuda",
            "--input-device",
            "cpu",
            "--checkpoint",
            str(tmp_path / "actual.pt"),
        ],
        env=environment,
        capture_output=True,
        text=True,
        # Include checkpoint serialization and fabric cleanup on shared CI CPUs.
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert "compatibility=FULL" in result.stderr
    expected, actual = json.loads(baseline.stdout), json.loads(result.stdout)
    assert len(actual["loss"]) == 2
    # DeepLab uses four 32x32 images: two 16x16 samples leave ASPP BatchNorm
    # extremely sensitive to float32 backend/gradient accumulation order.
    # Strict all-gradient/state parity is separately checked in float64.
    torch.testing.assert_close(
        torch.tensor(actual["loss"]),
        torch.tensor(expected["loss"]),
        rtol=0.002,
        atol=0.002,
        msg=f"{script}: baseline losses {expected['loss']}; TrainPool losses {actual['loss']}",
    )
    if script == "unet_train.py":
        expected_state = torch.load(tmp_path / "baseline.pt", weights_only=True)["model"]
        actual_state = torch.load(tmp_path / "actual.pt", weights_only=True)["model"]
        for name in expected_state:
            torch.testing.assert_close(actual_state[name], expected_state[name], rtol=0.001, atol=0.0001)
    jobs = clients[0].control("metrics")["jobs"]
    assert any(
        job["bytes_local_to_remote_ram"] > 0 and job["bytes_remote_ram_to_local"] > 0 for job in jobs.values()
    )
