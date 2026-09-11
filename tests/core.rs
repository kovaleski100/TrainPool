mod common;
use common::{GIB, node};
use std::{
    sync::{Arc, atomic::Ordering},
    time::{Duration, Instant},
};
use trainpool::{
    cluster::{
        election::elect,
        membership::{Membership, PEER_TIMEOUT},
    },
    config::{Config, node_id},
    memory::{
        allocator::RamAccounting,
        block::{BlockState, Location, MemoryBlockHandle, TensorMetadata},
        local_ram::LocalRam,
    },
    node::{gpu::usable_vram, memory::MemoryCapabilities},
    scheduler::{
        placement::{CapacityPlacement, PlacementPolicy},
        planner::{CapacityPlanner, StageSpec, TrainingPlanner, require_cuda},
        topology::{LinkEstimate, Topology},
    },
    transport::{signature, verify},
};
use uuid::Uuid;

#[test]
fn static_memory_election_and_ties() {
    let a = node(1, 64, None);
    let mut b = node(2, 32, Some(12));
    assert_eq!(elect([&a, &b].into_iter()).leader_id, Some(a.node_id));
    b.memory.os_available_ram = 0;
    assert_eq!(elect([&b, &a].into_iter()).leader_id, Some(a.node_id));
    let c = node(3, 52, Some(12));
    assert_eq!(elect([&a, &c].into_iter()), elect([&c, &a].into_iter()));
    assert_eq!(elect([&a, &c].into_iter()).leader_id, Some(c.node_id));
}
#[test]
fn ram_budget_does_not_recursively_shrink() {
    let before = MemoryCapabilities::calculate(32 * GIB, 16 * GIB, 0, 0.5, None, 0, 0.0);
    let allocated = MemoryCapabilities::calculate(32 * GIB, 8 * GIB, 8 * GIB, 0.5, None, 0, 0.0);
    let pressure = MemoryCapabilities::calculate(32 * GIB, 0, 20 * GIB, 0.5, None, 0, 0.0);
    assert_eq!(before.trainpool_ram_budget, 16 * GIB);
    assert_eq!(allocated.trainpool_ram_budget, 16 * GIB);
    assert_eq!(pressure.trainpool_ram_budget, 16 * GIB);
    assert_eq!(pressure.excess(), 4 * GIB);
    assert_eq!(pressure.trainpool_ram_available, 0);
}

#[test]
fn local_ram_budget_applies_explicit_reserve_to_live_os_headroom() {
    let sample = MemoryCapabilities::calculate(1_000, 800, 100, 0.9, None, 100, 0.1);
    assert_eq!(sample.physical_ram_total, 1_000);
    assert_eq!(sample.os_available_ram, 800);
    assert_eq!(sample.safety_reserve, 100);
    assert_eq!(sample.trainpool_ram_budget, 800);
    assert_eq!(sample.trainpool_ram_used, 100);
    assert_eq!(sample.trainpool_ram_available, 700);
    assert_eq!(sample.safe_local_ram_allocatable_now, 700);

    let pressure = MemoryCapabilities::calculate(1_000, 200, 600, 0.9, None, 100, 0.1);
    assert_eq!(pressure.trainpool_ram_budget, 700);
    assert_eq!(pressure.safe_local_ram_allocatable_now, 100);
}
#[test]
fn cpu_only_leader_and_single_gpu_plan() {
    let nodes = vec![node(1, 32, Some(12)), node(2, 64, None)];
    let plan = CapacityPlanner
        .plan(&nodes, &elect(nodes.iter()), &Topology::default(), &[])
        .unwrap();
    assert_eq!(plan.leader_id, nodes[1].node_id);
    assert_eq!(plan.strategy, "SingleGpuDistributedMemory");
    assert_eq!(plan.compute_nodes, vec![nodes[0].node_id]);
    assert!(plan.memory_nodes.contains(&nodes[1].node_id));
    assert!(require_cuda(&plan, nodes[1].node_id).is_err());
    assert!(require_cuda(&plan, nodes[0].node_id).is_ok());
    assert!(nodes[1].runtime.ram_provider);
    assert!(!nodes[1].runtime.gpu_compute);
    let ranked = CapacityPlacement.rank_ram(&nodes, nodes[0].node_id, GIB, &Topology::default());
    assert_eq!(ranked[0].node_id, nodes[0].node_id);
    assert!(ranked.iter().any(|n| n.node_id == nodes[1].node_id));
}

#[test]
fn local_ram_is_a_strict_tier_and_remote_peers_follow_link_cost() {
    let nodes = vec![node(1, 32, Some(12)), node(2, 64, None), node(3, 64, None)];
    let compute = nodes[0].node_id;
    let mut topology = Topology::default();
    topology.update(LinkEstimate {
        source: compute,
        destination: nodes[1].node_id,
        latency_ms: 20.0,
        bytes_per_second: Some(100_000_000.0),
        sampled_at_ms: 1,
        active_transfers: 0,
    });
    topology.update(LinkEstimate {
        source: compute,
        destination: nodes[2].node_id,
        latency_ms: 1.0,
        bytes_per_second: Some(1_000_000_000.0),
        sampled_at_ms: 1,
        active_transfers: 0,
    });
    let ranked = CapacityPlacement.rank_ram(&nodes, compute, GIB, &topology);
    assert_eq!(ranked[0].node_id, compute);
    assert_eq!(ranked[1].node_id, nodes[2].node_id);
    assert_eq!(ranked[2].node_id, nodes[1].node_id);
}
#[test]
fn unequal_gpu_stages_obey_working_set_capacity() {
    let nodes = vec![
        node(1, 32, Some(12)),
        node(2, 16, Some(4)),
        node(3, 16, Some(8)),
    ];
    let stages: Vec<_> = (0..12)
        .map(|i| StageSpec {
            name: format!("L{i}"),
            working_set_bytes: 2 * GIB,
        })
        .collect();
    let plan = CapacityPlanner
        .plan(&nodes, &elect(nodes.iter()), &Topology::default(), &stages)
        .unwrap();
    assert_eq!(plan.gpu_assignments.len(), 1);
    assert_eq!(plan.gpu_assignments[0].capacity_share, 1.0);
    assert!(plan.stages.iter().all(|s| s.node_id == nodes[0].node_id));
    assert!(plan.execution_supported);
    assert!(
        CapacityPlanner
            .plan(
                &nodes,
                &elect(nodes.iter()),
                &Topology::default(),
                &[StageSpec {
                    name: "too large".into(),
                    working_set_bytes: 13 * GIB
                }]
            )
            .is_err()
    );
}

#[test]
fn automatic_multi_gpu_plan_selects_largest_primary() {
    let nodes = vec![node(1, 32, Some(12)), node(2, 32, Some(4))];
    let plan = CapacityPlanner
        .plan(&nodes, &elect(nodes.iter()), &Topology::default(), &[])
        .unwrap();
    assert_eq!(plan.strategy, "SingleGpuDistributedMemory");
    assert!(plan.execution_supported);
    assert_eq!(plan.gpu_assignments.len(), 1);
    assert_eq!(plan.gpu_assignments[0].node_id, nodes[0].node_id);
    assert_eq!(plan.compute_nodes, vec![nodes[0].node_id]);
}
#[test]
fn leader_times_out_and_is_reelected() {
    let a = node(1, 32, None);
    let b = node(2, 64, None);
    let mut m = Membership::new(a.clone());
    m.update(b.clone());
    assert_eq!(m.leader().leader_id, Some(b.node_id));
    assert_eq!(
        m.expire(Instant::now() + PEER_TIMEOUT + Duration::from_millis(1)),
        vec![b.node_id]
    );
    assert_eq!(m.leader().leader_id, Some(a.node_id));
}
#[test]
fn configuration_and_identity_are_safe() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(node_id(dir.path()).unwrap(), node_id(dir.path()).unwrap());
    let mut config = Config::default();
    assert_eq!(config.ram_fraction, 0.90);
    assert_eq!(config.ram_reserve_bytes, GIB);
    assert_eq!(config.ram_reserve_fraction, 0.10);
    assert_eq!(config.vram_reserve_bytes, 96 * 1024 * 1024);
    assert_eq!(config.vram_reserve_fraction, 0.02);
    assert!(!config.disk.enabled);
    for fraction in [0.09, 0.91, f64::NAN] {
        config.ram_fraction = fraction;
        assert!(config.validate().is_err());
    }
    config.ram_fraction = 0.5;
    config.disk.enabled = true;
    assert!(config.validate().is_err());
    assert_eq!(
        usable_vram(12 * GIB, 500 * 1024 * 1024, 512 * 1024 * 1024, 0.05),
        0
    );
    assert_eq!(
        usable_vram(
            4 * GIB,
            4 * GIB,
            config.vram_reserve_bytes,
            config.vram_reserve_fraction
        ),
        4 * GIB - 96 * 1024 * 1024
    );
    assert_eq!(
        usable_vram(
            32 * GIB,
            32 * GIB,
            config.vram_reserve_bytes,
            config.vram_reserve_fraction
        ),
        32 * GIB - 256 * 1024 * 1024
    );
}
#[test]
fn secret_authentication_detects_wrong_secret_and_tampering() {
    let mac = signature(Some("cluster secret"), b"hello");
    assert!(verify(Some("cluster secret"), b"hello", mac.as_deref()).is_ok());
    assert!(verify(Some("wrong"), b"hello", mac.as_deref()).is_err());
    assert!(verify(Some("cluster secret"), b"tampered", mac.as_deref()).is_err());
    assert!(verify(Some("cluster secret"), b"hello", None).is_err());
}
#[test]
fn tensor_metadata_rejects_overflow_and_invalid_layout() {
    let mut meta = TensorMetadata {
        dtype: "float32".into(),
        shape: vec![2, 3],
        layout: "contiguous".into(),
        byte_length: 24,
    };
    assert!(meta.validate(24).is_ok());
    meta.shape = vec![u64::MAX, 2];
    assert!(meta.validate(24).is_err());
}

#[test]
fn gpu_detection_accepts_missing_optional_driver_fields() {
    let legacy = trainpool::node::gpu::parse_csv(
        "GPU-id, Test NVIDIA, 12288, 11264, 1024, N/A, 550.0, N/A",
        &Config::default(),
    );
    assert_eq!(legacy.len(), 1);
    assert_eq!(legacy[0].vram_total, 12 * GIB);
    assert!(legacy[0].cuda_capability.is_none());
    assert!(legacy[0].temperature.is_none());
    assert!(trainpool::node::gpu::parse_csv("driver unavailable", &Config::default()).is_empty());
}

#[test]
fn inventory_cannot_regress_leases_or_resurrect_freed_blocks() {
    use trainpool::memory::residency::ResidencyTable;
    let mut table = ResidencyTable::default();
    let mut handle = MemoryBlockHandle {
        id: Uuid::new_v4(),
        job_id: Uuid::new_v4(),
        size: 1,
        owner_node: Uuid::nil(),
        owner_incarnation: Uuid::nil(),
        location_type: Location::Ram {
            node_id: Uuid::nil(),
        },
        checksum: Some("hash".into()),
        state: BlockState::Ready,
        lease_token: Uuid::new_v4(),
        lease_expires_ms: 200,
        generation: 0,
        tensor: None,
    };
    table.register(handle.clone());
    handle.lease_expires_ms = 100;
    table.register(handle.clone());
    assert_eq!(table.blocks[&handle.id].lease_expires_ms, 200);
    handle.lease_expires_ms = 0;
    table.register(handle.clone());
    handle.lease_expires_ms = 300;
    table.register(handle.clone());
    assert_eq!(table.blocks[&handle.id].lease_expires_ms, 0);
}

#[tokio::test]
async fn reservations_do_not_overcommit_under_concurrency() {
    let accounting = Arc::new(RamAccounting::default());
    accounting.budget.store(100, Ordering::SeqCst);
    let mut tasks = tokio::task::JoinSet::new();
    for _ in 0..20 {
        let a = accounting.clone();
        tasks.spawn(async move { a.reserve(10) });
    }
    let mut held = vec![];
    while let Some(result) = tasks.join_next().await {
        if let Ok(reservation) = result.unwrap() {
            held.push(reservation);
        }
    }
    assert_eq!(held.len(), 10);
    assert_eq!(accounting.used(), 100);
    drop(held);
    assert_eq!(accounting.used(), 0);
}
#[tokio::test]
async fn ram_allocation_lifetime_and_leases() {
    let store = LocalRam::default();
    store.accounting.budget.store(4096, Ordering::SeqCst);
    let id = Uuid::new_v4();
    let token = Uuid::new_v4();
    let handle = MemoryBlockHandle {
        id,
        job_id: Uuid::new_v4(),
        size: 2048,
        owner_node: Uuid::nil(),
        owner_incarnation: Uuid::nil(),
        location_type: Location::Ram {
            node_id: Uuid::nil(),
        },
        checksum: None,
        state: BlockState::Writing,
        lease_token: token,
        lease_expires_ms: trainpool::now_ms() + 10000,
        generation: 0,
        tensor: None,
    };
    store.allocate(handle).await.unwrap();
    let reader = store.get(id).await.unwrap();
    assert!(store.free(id, Uuid::new_v4()).await.is_err());
    store.free(id, token).await.unwrap();
    assert_eq!(store.accounting.used(), 2048);
    drop(reader);
    assert_eq!(store.accounting.used(), 0);
    let accounting = Arc::new(RamAccounting::default());
    accounting.budget.store(10, Ordering::SeqCst);
    let held = accounting.reserve(8).unwrap();
    assert!(accounting.reserve(3).is_err());
    drop(held);
    assert_eq!(accounting.used(), 0);
}

#[test]
fn inventory_is_not_executable_capacity_and_primary_ties_are_stable() {
    use trainpool::protocol::LogicalTrainingMemory;
    let mut nodes = vec![
        node(1, 32, Some(12)),
        node(2, 64, Some(12)),
        node(3, 16, Some(4)),
    ];
    let first = CapacityPlanner
        .plan(&nodes, &elect(nodes.iter()), &Topology::default(), &[])
        .unwrap();
    let memory = LogicalTrainingMemory::from_nodes(&nodes);
    assert_eq!(memory.cluster_physical_vram, 28 * GIB);
    assert_eq!(memory.primary_gpu_physical_vram, 12 * GIB);
    assert_eq!(
        memory.current_job_backing_capacity,
        memory.primary_gpu_usable_vram + memory.pool_ram_budget
    );
    assert_eq!(
        memory.logical_training_capacity,
        memory.primary_gpu_usable_vram + memory.pool_ram_allocatable
    );
    nodes.reverse();
    let second = CapacityPlanner
        .plan(&nodes, &elect(nodes.iter()), &Topology::default(), &[])
        .unwrap();
    assert_eq!(
        first.gpu_assignments[0].gpu_id,
        second.gpu_assignments[0].gpu_id
    );
    for node in &mut nodes {
        node.runtime.gpu_compute = false;
    }
    assert_eq!(
        LogicalTrainingMemory::from_nodes(&nodes).primary_gpu_usable_vram,
        0
    );
}
