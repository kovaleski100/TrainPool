from trainpool_torch.store import (
    MemoryTier,
    ResidencyCandidate,
    TensorStore,
    TieredResidencyPolicy,
)


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
