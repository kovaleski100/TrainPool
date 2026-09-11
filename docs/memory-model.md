# Memory model

## Safe RAM contribution

The default contribution ceiling is `f=0.90`, with an explicit reserve of the
larger of 1 GB or 10% of physical RAM:

```
effective = min(physical_ram, os_available_ram + trainpool_owned_ram)
reserve = max(ram_reserve_bytes, physical_ram * ram_reserve_fraction)
safe_capacity = max(0, effective - reserve)
budget = min(safe_capacity, f * physical_ram, optional_absolute_contribution_ceiling)
allocatable = max(0, budget - owned)
excess = max(0, owned - budget)
```

The physical-total clamp handles noisy/inconsistent OS samples. Linux cgroup headroom
also limits observed availability. `owned` includes reserved payload allocations,
active relay buffers and SDK-declared staging buffers. Atomic reservations prevent
concurrent requests from collectively exceeding the last sampled budget. Allocation
and staging requests refresh OS memory information before admission.

Ownership is added back only when deriving effective current capacity, so allocating
TrainPool RAM does not recursively shrink its own budget. The reserve is subtracted
exactly once. The
absolute `--ram-limit-mib` ceiling is useful for protecting test machines and forcing
remote placement without exhausting the host.

This is sampled resource control, not an OS-enforced memory cgroup for the daemon.
Allocator metadata, JSON frames, network/kernel buffers, process stacks and unrelated
application allocations are not included in payload accounting. SDK byte staging is
explicitly reserved; the original user model and explicit CPU-test tensor destinations
are outside the daemon's allocation store. Applications consuming memory between
samples can temporarily put TrainPool over budget. No local application is killed.

## RAM blocks and staging

`LocalRam` owns heap-allocated byte vectors, with one RAII accounting reservation per
block. A committed allocation is immutable. Reads borrow its bytes, writes fill its
next slice, and migration sends borrowed slices. The map contains shared references;
accounting is released only when the final live reference is dropped.

SDK data flows through the local daemon to the owner. The relay allocates 64 KiB and
reserves it in the same budget. New payload allocations leave at least 256 KiB free
for relays. This is minimum deadlock-avoidance headroom, not a promise that any number
of concurrent SDK staging requests will fit. Select a chunk/block size substantially
below the local contribution ceiling.

The SDK first attempts to retain a contiguous tensor in selected-device VRAM. When
the safe VRAM budget is full it evicts low-value residents, copying one bounded slice
to CPU at a time, reserving that host buffer, streaming it, then releasing staging.
The allocator fills compute-node RAM before any remote provider. Restoring a backed
CUDA tensor uses checked host chunks and promotes it to VRAM when useful; successful
promotion releases the RAM blocks, so it is a move rather than replication. The
bounded worker prefetch can overlap the next sequential stage's transfer with compute.

Non-contiguous tensors are normalized with PyTorch `contiguous()` before transfer;
that normalization can itself require a tensor-sized allocation on the original
device. Models used for capacity experiments should use contiguous stage parameters.
The SDK cannot make a single operation whose working set exceeds every GPU executable.

## GPU resource class and authoritative SDK snapshot

```
adaptive = min(0.02 * physical_vram_total, 256 MiB)
reserve = max(96 MiB, adaptive)
usable_vram = max(0, current_free_vram - reserve)
```

Both reserve terms are configurable; an explicitly larger byte floor is not capped.
The Python process that owns the CUDA context is authoritative for execution-time
admission. It samples:

```
reclaimable = max(0, torch_reserved - torch_allocated)
physical_allocator_headroom = driver_free + reclaimable
budget_headroom = max(0, configured_usable_ceiling - torch_allocated)
safe_allocatable_now = min(physical_allocator_headroom, budget_headroom)
```

`driver_free` excludes PyTorch allocator cache, so reclaimable cache is added only to
physical headroom. `trainpool_resident` is already a subset of `torch_allocated`; it
is reported separately and is never added to headroom. This prevents double counting.
The SDK re-samples before each admitted graph group and optimizer parameter, allowing
observed pressure to trigger eviction. Status distinguishes physical, free, usable,
allocated, reserved, reclaimable and TrainPool-resident VRAM. Reports are throttled to
at most one every 250 ms except forced shutdown snapshots. The SDK uses explicit tensor
copies and PyTorch CUDA
allocation; the Rust block store never claims to allocate a distributed CUDA pointer.
GPU observations are refreshed about every ten seconds. SDK GPU allocation reports
come from PyTorch process allocator statistics, so they assume one training job per
compute process and are not a cross-process ownership measurement.

## Placement and pressure

RAM candidates must advertise provider capability, enough allocatable bytes and no
current pressure. Local RAM is a strict tier boundary, followed by remote estimated
transfer time:

```
latency + bytes / estimated_bandwidth
```

The remote heuristic increases cost for active transfer load, prefers available
capacity on equal cost and uses UUIDs for deterministic ties. The leader synchronously
refreshes compute-node RAM before placement and the owner rechecks capacity. The
compute node is always the first RAM candidate. Only if it rejects admission can remote
providers be ranked by topology. `preferred_node` is a hint among remote candidates
and cannot bypass valid local RAM.

Pressure events contain owned bytes, revised budget and excess. The leader serializes
pressure relief, picks committed source blocks deterministically, and migrates until
it has moved at least the excess or exhausted eligible destinations. It may move more
than the excess because a logical block is migrated whole. Source and destination
temporarily both account the complete block. Leases keep interrupted unpublished
copies bounded in lifetime. Copy success, metadata CAS and publication precede release.

If there is no destination with enough capacity, the node retains existing valid data,
reports unresolved pressure and refuses further allocations. It cannot force another
application to release RAM, and it does not automatically use disk. A transient excess
is preferable to destroying the only valid training state.

## Loss, disk and persistence

Disappearing/restarted owners invalidate single-copy blocks. Their RAM no longer
contributes to active capacity. Affected jobs fail explicitly. Leadership metadata and
job state are in-memory; leader failover does not transparently resume training.

There is no file-backed block backend, mmap tensor file, tempfile tensor serialization,
automatic checkpoint, disk spill implementation or TrainPool-managed swap. Explicit
user-requested PyTorch checkpoint files are supported outside the fabric. `Disk` is a
reserved enum variant only. `disk.enabled=true` and `--enable-disk-spill` fail validation.
Only identity and configuration are written by the runtime. Logs contain structured
metadata, never tensor bodies. The test/demo scripts may create temporary **metadata**
directories, which are checked for unexpected files.

The OS may independently page ordinary process memory to its configured swap. TrainPool
does not manage that swap or promise page locking. Experiments requiring a literal
no-storage-I/O guarantee must use hosts with paging disabled or a separately validated
locked-memory deployment. Adding optional mlock accounting is a possible future extension.


## Inventory versus job capacity

`cluster_physical_vram` / `cluster_usable_vram` sum inventory. Only the largest
eligible GPU contributes `primary_gpu_physical_vram` / `primary_gpu_usable_vram`.
`pool_ram_budget` includes `pool_ram_owned`; `pool_ram_allocatable` is the remaining
safe pool contribution. Current status is a cluster snapshot for a new single-GPU
job, not a reservation or a guarantee for an already running job. Job plans pin the
selected GPU; inventory refresh does not migrate computation.

`current_job_backing_capacity = primary_gpu_usable_vram + pool_ram_budget` and
`current_job_remaining_capacity = primary_gpu_usable_vram + pool_ram_allocatable`.
The old `logical_training_capacity` JSON field remains a compatibility alias for
the corrected remaining capacity. Old `physical_vram` / `usable_vram` fields remain
inventory aliases. No other GPU's VRAM enters the executable capacity calculation.

## Residency observability

Per-job metrics expose current and peak TrainPool-owned bytes for VRAM, local RAM
backing and remote RAM backing. Directional counters distinguish GPU/local,
GPU/remote and local/remote traffic. `eviction_count` and `prefetch_count` expose
policy actions; existing allocator and network counters remain for compatibility.
