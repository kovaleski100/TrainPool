import copy
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from trainpool_torch.graph import capture, prepare_graph_inplace
from trainpool_torch.sequential import CapacityOptimizer
from trainpool_torch.store import TensorHandle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from unet_train import UNet  # noqa: E402


class SimulationStore:
    """RAM tensor transport double; fabric behavior has separate TCP tests."""

    def __init__(self):
        self.values = {}
        self.client = SimpleNamespace(local_node="simulation")
        self.allocations = 0
        self.peak = 0

    def offload(self, value, *, expected_next_use=None):
        handle = TensorHandle(
            str(uuid.uuid4()),
            tuple(value.shape),
            str(value.dtype).removeprefix("torch."),
            value.numel() * value.element_size(),
            str(value.device),
            [],
            expected_next_use=expected_next_use,
        )
        self.values[handle.id] = value.detach().clone()
        self.allocations += 1
        self.peak = max(self.peak, len(self.values))
        return handle

    def restore(self, handle, *, device=None):
        return self.values[handle.id].clone().to(device or handle.device)

    def free(self, handle):
        self.values.pop(handle.id, None)

    def flush_metrics(self):
        pass

    def close(self):
        self.values.clear()


def parity(factory, optimizer_type, *, steps=3, shape=(2, 3, 16, 16), dtype=torch.float64):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    baseline = factory().to(dtype=dtype)
    model = copy.deepcopy(baseline)
    store = SimulationStore()
    runtime = prepare_graph_inplace(model, device=torch.device("cpu"), store=store)
    assert len(runtime.ir.groups) > 1
    baseline_optimizer = optimizer_type(baseline.parameters(), lr=0.001, foreach=False)
    optimizer = CapacityOptimizer(runtime, optimizer_type, {"lr": 0.001, "foreach": False})
    for _ in range(steps):
        optimizer.zero_grad()
        baseline_optimizer.zero_grad()
        x = torch.randn(shape, dtype=dtype, requires_grad=True)
        y = x.detach().clone().requires_grad_(True)
        rng = torch.random.get_rng_state()
        expected = baseline(x)
        if isinstance(expected, dict):
            expected = expected["out"]
        loss = expected.square().mean()
        loss.backward()
        after = torch.random.get_rng_state()
        torch.random.set_rng_state(rng)
        result = model(y)
        if isinstance(result, dict):
            result = result["out"]
        actual_loss = result.square().mean()
        actual_loss.backward()
        torch.testing.assert_close(torch.random.get_rng_state(), after)
        torch.testing.assert_close(actual_loss, loss, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(y.grad, x.grad, rtol=1e-5, atol=1e-6)
        for name, parameter in baseline.named_parameters():
            handle = runtime.records[name].gradients.get(name)
            if parameter.grad is not None:
                assert handle is not None, name
                torch.testing.assert_close(
                    store.restore(handle), parameter.grad, rtol=1e-5, atol=1e-6, msg=name
                )
        optimizer.step()
        baseline_optimizer.step()
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, baseline.state_dict()[name], rtol=2e-4, atol=3e-6, msg=name)
    return model, runtime, store


@pytest.mark.parametrize("optimizer", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_unet_skip_concat_batchnorm_rng_and_optimizer_parity(optimizer):
    model, runtime, store = parity(lambda: UNet(width=2), optimizer)
    assert any(len(value.consumers) > 1 for value in runtime.ir.values.values())
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert store.allocations > 100
    model.close()
    assert not store.values


@pytest.mark.parametrize("optimizer", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_real_torchvision_deeplab_training(optimizer):
    from torchvision.models.segmentation import deeplabv3_resnet50

    model, _, _ = parity(
        lambda: deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False, num_classes=3),
        optimizer,
        steps=2,
        shape=(4, 3, 32, 32),
        dtype=torch.float64,
    )
    model.close()


def test_unknown_graph_is_rejected_before_materialization():
    class Unknown(nn.Module):
        def forward(self, x):
            return torch.sin(x)

    with pytest.raises(RuntimeError, match="TRAINPOOL_UNSUPPORTED_OPERATOR.*sin"):
        capture(Unknown())


@pytest.mark.parametrize("kind", ["unused_buffer", "shared_buffer", "complex", "extra_state"])
def test_unsupported_state_is_rejected_before_model_changes(kind):
    model = nn.Sequential(nn.Linear(4, 4))
    error = "TRAINPOOL_UNSUPPORTED_BUFFER"
    if kind == "unused_buffer":
        model.register_buffer("unused", torch.ones(1))
    elif kind == "shared_buffer":
        buffer = torch.ones(1)
        model.register_buffer("first", buffer)
        model.register_buffer("second", buffer)
    elif kind == "complex":
        model.register_buffer("phase", torch.ones(1, dtype=torch.complex64))
        error = "TRAINPOOL_UNSUPPORTED_GRAPH.*complex"
    else:

        class ExtraState(nn.Sequential):
            def get_extra_state(self):
                return {"version": 1}

        model = ExtraState(nn.Linear(4, 4))
        error = "TRAINPOOL_CHECKPOINT_UNSUPPORTED.*extra state"
    before = {name: value.clone() for name, value in model.named_parameters()}
    with pytest.raises(RuntimeError, match=error):
        prepare_graph_inplace(model, device=torch.device("cpu"), store=SimulationStore())
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu"
        torch.testing.assert_close(parameter, before[name])


class BranchGraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Linear(4, 4)
        self.left = nn.Sequential(*[nn.Linear(4, 4), nn.Tanh()] * 1)
        self.right = nn.Sequential(nn.Linear(4, 4), nn.Sigmoid())
        self.head = nn.Linear(8, 4)

    def forward(self, values, *, bias):
        shared = self.stem(values["images"][0])
        left = self.left(shared)
        right = self.right(shared)
        merged = self.head(torch.cat([left, right], dim=-1)) + shared + bias
        return {"out": merged, "extra": (left, [right]), "label": "segmentation"}


def test_structured_inputs_outputs_fanout_and_unused_output():
    torch.manual_seed(42)
    baseline = BranchGraph().double()
    model = copy.deepcopy(baseline)
    store = SimulationStore()
    runtime = prepare_graph_inplace(model, device=torch.device("cpu"), store=store)
    x = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    bias = torch.randn(2, 4, dtype=torch.float64)
    expected = baseline({"images": [x]}, bias=bias)
    actual = model({"images": [y]}, bias=bias)
    assert actual["label"] == "segmentation"
    (expected["out"].sum() + expected["extra"][0].sum()).backward()
    (actual["out"].sum() + actual["extra"][0].sum()).backward()
    torch.testing.assert_close(x.grad, y.grad)
    for name, parameter in baseline.named_parameters():
        torch.testing.assert_close(store.restore(runtime.records[name].gradients[name]), parameter.grad)
    model.close()


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_checkpoint_roundtrip_continues_training_and_rejects_bad_load(optimizer_type):
    import io

    from trainpool_torch.sequential import attach_optimizer_inplace

    torch.manual_seed(63)
    source = UNet(width=2).double()
    original = source.state_dict()
    store = SimulationStore()
    runtime = prepare_graph_inplace(source, device=torch.device("cpu"), store=store)
    optimizer = optimizer_type(source.parameters(), lr=0.001, foreach=False)
    attach_optimizer_inplace(optimizer, runtime)
    x = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    optimizer.zero_grad()
    source(x).square().mean().backward()
    optimizer.step()
    buffer = io.BytesIO()
    torch.save({"model": source.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)
    destination = UNet(width=2).double()
    destination.load_state_dict(original)
    other_store = SimulationStore()
    other_runtime = prepare_graph_inplace(destination, device=torch.device("cpu"), store=other_store)
    other_optimizer = optimizer_type(destination.parameters(), lr=0.1, foreach=False)
    attach_optimizer_inplace(other_optimizer, other_runtime)
    destination.load_state_dict(checkpoint["model"])
    other_optimizer.load_state_dict(checkpoint["optimizer"])
    rng = torch.random.get_rng_state()
    for model, opt in ((source, optimizer), (destination, other_optimizer)):
        torch.random.set_rng_state(rng)
        opt.zero_grad()
        model(x).square().mean().backward()
        opt.step()
    for name, tensor in source.state_dict().items():
        torch.testing.assert_close(tensor, destination.state_dict()[name])
    before = destination.state_dict()
    bad = dict(before)
    bad["head.weight"] = torch.ones(999)
    with pytest.raises(RuntimeError, match="size mismatch"):
        destination.load_state_dict(bad)
    for name, tensor in destination.state_dict().items():
        torch.testing.assert_close(tensor, before[name])
    source.close()
    destination.close()


def test_admission_splits_groups_and_rejects_impossible_operator():
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 4))
    store = SimulationStore()
    runtime = prepare_graph_inplace(model, device=torch.device("cpu"), store=store, gpu_budget_bytes=10000)
    x = torch.randn(16, 4)
    model(x).sum().backward()
    assert len(runtime.ir.groups) > 1
    assert all(group.estimated_working_set <= 10000 for group in runtime.ir.groups)
    model.close()
    model = nn.Sequential(nn.Linear(4, 4))
    store = SimulationStore()
    runtime = prepare_graph_inplace(model, device=torch.device("cpu"), store=store, gpu_budget_bytes=1)
    with pytest.raises(RuntimeError, match="TRAINPOOL_UNSUPPORTED_WORKING_SET"):
        model(torch.ones(1, 4))
    assert not runtime.pending
    model.close()


def test_admission_does_not_count_region_input_and_output_aliases():
    model = nn.Sequential(nn.Conv2d(6, 64, 3, padding=1))
    store = SimulationStore()
    budget = 1_200_000_000
    runtime = prepare_graph_inplace(model, device=torch.device("cpu"), store=store, gpu_budget_bytes=budget)
    runtime._preflight((torch.empty(8, 6, 256, 256),))
    assert len(runtime.ir.groups) == 1
    assert runtime.ir.groups[0].estimated_working_set <= budget
    model.close()


def test_graph_stress_has_bounded_live_handles():
    model, runtime, store = parity(lambda: UNet(width=2), torch.optim.AdamW, steps=40)
    assert store.allocations > 10000
    parameter_count = sum(len(record.weights) for record in runtime.stages)
    buffer_count = sum(len(record.buffers) for record in runtime.stages)
    # AdamW owns one weight and three state tensors per parameter after step.
    assert len(store.values) == 4 * parameter_count + buffer_count
    model.close()


def test_immutable_buffer_dependency_and_checkpoint():
    class WithBuffer(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(4, 4)
            self.register_buffer("scale", torch.tensor([1.0, 2.0, 3.0, 4.0]))

        def forward(self, x):
            return self.layer(x) * self.scale

    baseline = WithBuffer()
    model = copy.deepcopy(baseline)
    store = SimulationStore()
    prepare_graph_inplace(model, device=torch.device("cpu"), store=store)
    x = torch.ones(2, 4, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    baseline(x).sum().backward()
    model(y).sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    torch.testing.assert_close(model.state_dict()["scale"], baseline.scale)
    model.close()
