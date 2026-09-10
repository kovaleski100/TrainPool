"""Opt-in physical CUDA parity against an already running local fabric."""

import copy
import os
import sys
from pathlib import Path

import pytest
import torch
from trainpool_torch import Client, TensorStore
from trainpool_torch.graph import prepare_graph_inplace
from trainpool_torch.sequential import CapacityOptimizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "models"))
from unet_train import UNet  # noqa: E402

pytestmark = pytest.mark.skipif(
    not os.getenv("TRAINPOOL_CUDA_ADDRESS") or not torch.cuda.is_available(),
    reason="requires physical CUDA and an explicitly supplied running fabric",
)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW])
def test_cuda_unet_buffers_rng_gradients_and_updates(optimizer_type):
    torch.set_num_threads(1)
    torch.manual_seed(91)
    baseline = UNet(width=4).double()
    model = copy.deepcopy(baseline)
    baseline.cuda()
    baseline_optimizer = optimizer_type(baseline.parameters(), lr=0.001, foreach=False)
    client = Client(os.environ["TRAINPOOL_CUDA_ADDRESS"])
    with TensorStore(client) as store:
        runtime = prepare_graph_inplace(
            model,
            device=torch.device("cuda", 0),
            store=store,
            gpu_budget_bytes=store.plan["gpu_assignments"][0]["usable_bytes"],
        )
        optimizer = CapacityOptimizer(runtime, optimizer_type, {"lr": 0.001, "foreach": False})
        for _ in range(2):
            optimizer.zero_grad()
            baseline_optimizer.zero_grad()
            x = torch.randn(4, 3, 32, 32, dtype=torch.float64, device="cuda", requires_grad=True)
            y = x.detach().clone().requires_grad_(True)
            cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            expected = baseline(x).square().mean()
            expected.backward()
            after_cpu, after_cuda = torch.random.get_rng_state(), torch.cuda.get_rng_state()
            torch.random.set_rng_state(cpu_rng)
            torch.cuda.set_rng_state(cuda_rng)
            actual = model(y).square().mean()
            actual.backward()
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(torch.random.get_rng_state(), after_cpu)
            torch.testing.assert_close(torch.cuda.get_rng_state(), after_cuda)
            for name, parameter in baseline.named_parameters():
                torch.testing.assert_close(
                    store.restore(runtime.records[name].gradients[name]), parameter.grad, rtol=1e-5, atol=1e-6
                )
            optimizer.step()
            baseline_optimizer.step()
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, baseline.state_dict()[name].cpu(), rtol=1e-5, atol=1e-6)
        metrics = client.control("metrics")["jobs"][store.job_id]
        assert metrics["peak_vram_resident_bytes"] > 0
        assert metrics["current_vram_resident_bytes"] > 0
        assert metrics["current_local_ram_backing_bytes"] == 0
        assert metrics["current_remote_ram_backing_bytes"] == 0
        assert metrics["gpu_to_remote_bytes"] == 0
        assert metrics["peak_gpu_residency"] > 0
