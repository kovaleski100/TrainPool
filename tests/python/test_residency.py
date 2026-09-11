import threading
from types import SimpleNamespace

from trainpool_torch.store import (
    MemoryTier,
    ResidencyCandidate,
    TensorStore,
    TieredResidencyPolicy,
)


def test_store_close_from_lease_thread_does_not_join_itself(monkeypatch):
    store = object.__new__(TensorStore)
    store._closed = threading.Event()
    store._renewal = threading.current_thread()
    store.client = SimpleNamespace(timeout=0)
    store.tensors = {}
    monkeypatch.setattr(store, "flush_metrics", lambda *, force=False: None)

    store.close()

    assert store._closed.is_set()


def test_cuda_snapshot_does_not_double_count_resident_or_cached_bytes(monkeypatch):
    import torch

    store = object.__new__(TensorStore)
    store._gpu_device = torch.device("cuda", 0)
    store._gpu_budget_bytes = 700
    store._gpu_safety_reserve_bytes = 75
    store._resident_bytes = 200
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (100, 1000))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _device: 400)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _device: 550)

    snapshot = store.cuda_memory_snapshot()
    assert snapshot.driver_used == 900
    assert snapshot.torch_reclaimable == 150
    assert snapshot.physical_allocator_headroom == 250
    assert snapshot.budget_headroom == 300
    assert snapshot.safe_allocatable_now == 250
    assert snapshot.configured_safety_reserve == 75
    # TrainPool residency is already included in torch_allocated; adding it
    # would incorrectly produce 450 bytes of budget headroom.
    assert snapshot.trainpool_resident == 200


def test_case_a_workload_that_fits_stays_in_vram():
    policy = TieredResidencyPolicy(vram_capacity=100, local_ram_capacity=200)
    assert policy.distribution(80) == {
        MemoryTier.GPU: 80,
        MemoryTier.LOCAL_RAM: 0,
        MemoryTier.REMOTE_RAM: 0,
    }


def test_case_b_excess_uses_local_ram_before_remote():
    policy = TieredResidencyPolicy(vram_capacity=100, local_ram_capacity=200)
    assert policy.distribution(250) == {
        MemoryTier.GPU: 100,
        MemoryTier.LOCAL_RAM: 150,
        MemoryTier.REMOTE_RAM: 0,
    }


def test_case_c_remote_ram_is_only_the_third_tier():
    policy = TieredResidencyPolicy(vram_capacity=100, local_ram_capacity=200)
    assert policy.distribution(350) == {
        MemoryTier.GPU: 100,
        MemoryTier.LOCAL_RAM: 200,
        MemoryTier.REMOTE_RAM: 50,
    }


def test_case_e_far_large_tensor_is_evicted_before_near_use_tensor():
    policy = TieredResidencyPolicy(vram_capacity=100, local_ram_capacity=200)
    near = ResidencyCandidate("near", size=30, next_use=11, remaining_consumers=2)
    far = ResidencyCandidate("far", size=60, next_use=50, remaining_consumers=1)
    assert policy.select_evictions([near, far], 40, current_step=10) == [far]


def test_transfer_cost_and_reuse_protect_expensive_hot_values():
    policy = TieredResidencyPolicy(vram_capacity=100, local_ram_capacity=200)
    local_cold = ResidencyCandidate("local", 40, 20, transfer_cost=1, reuse_count=0)
    remote_hot = ResidencyCandidate("remote", 40, 20, transfer_cost=8, reuse_count=4)
    assert policy.select_evictions([remote_hot, local_cold], 40, current_step=10) == [local_cold]


def test_fabric_fills_compute_node_ram_before_remote_ram(cluster):
    import torch

    clients, _ = cluster
    local = clients[0].local_node
    local_budget = clients[0].control("plan", stages=[])["memory_budgets"][local]
    with TensorStore(clients[0], block_bytes=4 * 1024 * 1024) as store:
        value = torch.zeros(local_budget + 8 * 1024 * 1024, dtype=torch.uint8)
        handle = store.offload(value)
        owners = [block["owner_node"] for block in handle.blocks]
        assert owners[0] == local
        assert any(owner != local for owner in owners)
        first_remote = next(index for index, owner in enumerate(owners) if owner != local)
        assert all(owner == local for owner in owners[:first_remote])
        store.flush_metrics()
        metrics = clients[0].control("metrics")["jobs"][store.job_id]
        assert metrics["current_local_ram_backing_bytes"] > 0
        assert metrics["current_remote_ram_backing_bytes"] > 0
        assert metrics["current_vram_resident_bytes"] == 0
