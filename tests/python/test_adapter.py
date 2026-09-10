import copy

import pytest
import torch
import trainpool_torch as tp
from torch import nn


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
@pytest.mark.parametrize("prefetch_depth", [0, 1])
def test_sequential_matches_full_pytorch_training(cluster, optimizer_type, prefetch_depth):
    clients, _ = cluster
    torch.manual_seed(17)
    reference = nn.Sequential(nn.Linear(8, 12), nn.Tanh(), nn.Linear(12, 4))
    candidate = copy.deepcopy(reference)
    options = {"lr": 0.01, "weight_decay": 0.02}
    if optimizer_type is torch.optim.SGD:
        options["momentum"] = 0.9
    reference_optimizer = optimizer_type(reference.parameters(), **options)
    candidate_optimizer = optimizer_type(candidate.parameters(), **options)
    with tp.TensorStore(clients[0], preferred_node=clients[1].local_node, block_bytes=4096) as store:
        prepared, optimizer = tp.prepare(
            candidate,
            candidate_optimizer,
            store=store,
            device="cpu",
            _test_cpu=True,
            prefetch_depth=prefetch_depth,
        )
        try:
            for _ in range(3):
                inputs = torch.randn(5, 8, requires_grad=True)
                remote_inputs = inputs.detach().clone().requires_grad_(True)
                target = torch.randn(5, 4)
                reference_optimizer.zero_grad()
                optimizer.zero_grad()
                expected = reference(inputs)
                actual = prepared(remote_inputs)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
                (expected - target).square().mean().backward()
                (actual - target).square().mean().backward()
                torch.testing.assert_close(remote_inputs.grad, inputs.grad, rtol=1e-5, atol=1e-6)
                reference_optimizer.step()
                optimizer.step()
                restored = dict(prepared.named_training_parameters())
                for name, parameter in reference.named_parameters():
                    torch.testing.assert_close(restored[name], parameter, rtol=1e-5, atol=1e-6)
            metrics = clients[0].control("metrics")["jobs"][store.job_id]
            assert metrics["bytes_local_to_remote_ram"] > 0
            assert metrics["bytes_remote_ram_to_local"] > 0
            assert any(s.optimizer_state for s in prepared.stages if s.weights)
        finally:
            prepared.prefetch.close()


def test_saved_activation_hooks_preserve_gradients(cluster):
    clients, _ = cluster
    torch.manual_seed(23)
    x = torch.randn(32, 32, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    (x.sin().square().sum()).backward()
    with tp.TensorStore(clients[0], preferred_node=clients[1].local_node, block_bytes=4096) as store:
        with tp.distributed_memory(store=store, min_bytes=0):
            y.sin().square().sum().backward()
        torch.testing.assert_close(y.grad, x.grad)


def test_tensor_roundtrip_noncontiguous_bfloat16_and_zero_length(cluster):
    clients, _ = cluster
    with tp.TensorStore(clients[0], preferred_node=clients[1].local_node, block_bytes=4096) as store:
        for tensor in [
            torch.arange(12000).reshape(100, 120).T,
            torch.randn(64, 64).bfloat16(),
            torch.empty(0, 4),
            torch.tensor(1.5),
        ]:
            handle = store.offload(tensor)
            torch.testing.assert_close(store.restore(handle), tensor)
            assert handle.layout == "contiguous"
            store.free(handle)


def test_no_disk_tensor_backing(cluster):
    clients, root = cluster
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    with tp.TensorStore(clients[0], preferred_node=clients[1].local_node, block_bytes=4096) as store:
        original = torch.arange(4096, dtype=torch.float32)
        handle = store.offload(original)
        torch.testing.assert_close(store.restore(handle), original)
    after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    assert before == after
    assert all(p.name in {"node_id", "config.toml"} for p in root.rglob("*") if p.is_file())


def test_no_implicit_cpu_compute_and_unsupported_graph_rejected():
    model = nn.Sequential(nn.Linear(2, 2))
    with pytest.raises(tp.TrainPoolError, match="TRAINPOOL_NO_CUDA"):
        tp.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu")
    model = nn.Sequential(nn.BatchNorm1d(2))
    with pytest.raises(ValueError, match="buffers"):
        tp.prepare(model, torch.optim.SGD(model.parameters(), lr=0.1), device="cpu", _test_cpu=True)


def test_python_sdk_rejects_remote_daemon_address():
    with pytest.raises(ValueError, match="local loopback"):
        tp.Client("192.0.2.1:7432")


def test_annotated_random_stage_replays_rng_without_advancing_it(cluster):
    clients, _ = cluster
    torch.manual_seed(8)
    reference = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.3), nn.Linear(4, 2))
    candidate = copy.deepcopy(reference)
    tp.stage(candidate[1])
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.02)
    candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=0.02)
    with tp.TensorStore(clients[0], preferred_node=clients[1].local_node) as store:
        model, optimizer = tp.prepare(
            candidate, candidate_optimizer, store=store, device="cpu", _test_cpu=True
        )
        try:
            inputs = torch.randn(8, 4)
            state = torch.random.get_rng_state()
            expected = reference(inputs)
            torch.random.set_rng_state(state)
            actual = model(inputs)
            torch.testing.assert_close(actual, expected)
            after_forward = torch.random.get_rng_state()
            expected.square().sum().backward()
            actual.square().sum().backward()
            assert torch.equal(torch.random.get_rng_state(), after_forward)
            reference_optimizer.step()
            optimizer.step()
            for name, parameter in model.named_training_parameters():
                torch.testing.assert_close(parameter, reference.state_dict()[name])
        finally:
            model.prefetch.close()


def test_training_example_exceeds_configured_residency_with_remote_state(cluster):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    clients, _ = cluster
    root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "python")
    environment["TRAINPOOL_ADDRESS"] = f"{clients[0].address[0]}:{clients[0].address[1]}"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "examples/train_sequential.py"),
            "--test-cpu",
            "--width",
            "128",
            "--layers",
            "24",
            "--resident-budget-mib",
            "1",
            "--steps",
            "1",
            "--memory-node",
            clients[1].local_node,
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    first = json.loads(result.stdout.splitlines()[0])
    assert first["parameter_bytes"] > first["configured_residency_budget"]
    assert first["remote_backing_bytes"] > 0
    assert first["backend"] == "explicit CPU simulation"
    assert not first["disk_spill"]


def test_remote_allocations_survive_lease_renewal_and_prefetch(cluster):
    import time

    from trainpool_torch.store import SequentialPrefetch, TensorStore

    clients, _ = cluster
    with TensorStore(clients[0], preferred_node=clients[1].local_node) as store:
        value = torch.arange(32, dtype=torch.float64)
        held = store.offload(value)
        initial_expiry = held.blocks[0]["lease_expires_ms"]
        for _ in range(1100):
            handle = store.offload(value)
            torch.testing.assert_close(store.restore(handle), value)
            store.free(handle)
        deadline = time.monotonic() + 12
        while held.blocks[0]["lease_expires_ms"] <= initial_expiry:
            assert time.monotonic() < deadline, "lease renewal did not run"
            time.sleep(0.1)
        remaining = max(0, initial_expiry / 1000 - time.time()) + 0.1
        time.sleep(remaining)
        prefetch = SequentialPrefetch(store)
        try:
            prefetch.schedule("held", lambda: store.restore(held))
            torch.testing.assert_close(prefetch.take("held", lambda: store.restore(held)), value)
            assert store.metrics["prefetch_count"] == 1
        finally:
            prefetch.close()
        assert len(store.tensors) == 1
