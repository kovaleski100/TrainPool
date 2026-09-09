# Architecture

TrainPool's useful capacity is the sum of separately managed RAM allocations and
usable GPU memory. An individual CUDA operation still needs its operands and workspace
on its assigned GPU. Remote RAM is backing storage with explicit transfers, not a
CUDA pointer or a distributed allocator intercepted underneath arbitrary PyTorch code.

```mermaid
flowchart LR
    SDK[PyTorch SDK on GPU host] -->|loopback control and chunks| A[Identical trainpool daemon A]
    A -->|allocation / resolve / plan| L[Identical trainpool daemon C: leader]
    L -->|allocation and migration instructions| B[Identical trainpool daemon B: RAM provider]
    B <-->|direct tensor chunks| A
    A <-->|bounded host staging| SDK
    SDK <-->|explicit tensor copies and compute| GPU[Local CUDA GPU]
```

The leader may be A, B or C. When the leader is also an endpoint or the SDK's local
daemon, it naturally handles that endpoint's payloads. It is never inserted as an
intermediate hop solely because it is leader.

## Runtime boundaries

* `cluster`: pluggable async discovery, multicast packets, membership timeout and election.
* `node`: sysinfo CPU/RAM monitoring and optional, time-bounded nvidia-smi probing.
* `transport` / `protocol`: versioned JSON control frames and separate binary data
  connections. The transport trait is independent of policy; its MVP implementation is TCP.
* `memory`: lease-bearing block handles, atomic RAM reservations, local immutable
  committed storage, residency metadata and source-driven migration.
* `scheduler`: directed link estimates, replaceable placement/planning interfaces and
  capacity-aware stage assignments.
* `runtime`: dispatch, membership exchange, pressure handling and services sharing one
  process. No remote execution endpoint exists.
* `python/trainpool_torch`: local SDK, tensor table, saved-tensor hooks, sequential
  recomputation, remote optimizer state and bounded prefetch.

The one executable includes daemon and CLI commands. `run` launches a structured
local program/argument vector with inherited working directory/environment using
`Command`, without a shell, and passes the local daemon configuration and job ID.

## Membership and leadership

The installation UUID is persistent metadata. Each daemon start has a new incarnation,
which prevents a restarted node being mistaken for the surviving owner of RAM bytes.
Announcements arrive approximately every two seconds. Only a successful TCP capability
exchange updates membership liveness; a replayed multicast packet cannot preserve a
dead peer. Expiry uses local monotonic `Instant` values and a seven-second threshold.
The two-second monitor granularity means observed failover is normally seven to nine
seconds plus scheduling delay, not a real-time deadline.

Every peer elects the maximum `(physical_ram + sum(physical_vram), node_uuid)` tuple.
The larger UUID wins ties. CPU-only leaders are ordinary supported peers. Free memory,
budget changes and GPU utilization never enter the score. GPU physical inventory is
fixed for a daemon lifetime; refreshes update observations without changing that score.

`leader_epoch` is a UUID generation token advertised by the current leader. It changes
when that process becomes leader again and on process restart. Followers derive the
same leader and generation once membership exchange converges. Schedule and migration
requests include the generation. Active jobs are invalidated by a leader-generation
change, rather than silently resuming from incomplete job metadata.

This is not consensus. Partitioned views can temporarily elect different leaders.
Epoch checks prevent using known stale leaders but are not quorum fencing. Use one
trusted, connected LAN for experiments. Replicated metadata, durable job recovery,
quorum fencing and partition-tolerant scheduling are future work.

## Discovery and topology

Multicast is bound with address/port reuse so several installations can be simulated on
one host. Interface and advertised address are explicit configurable options. The
default address selection uses a UDP route lookup without sending application data.
Seeds use the same authenticated capability exchange and can replace multicast on VPNs.
Direct peers remain monitored after discovery. Seed-only deployments need a sufficiently
complete set of seeds; general transitive membership gossip is not implemented.

Normal exchanges provide an exponentially weighted control RTT estimate. Explicit
`benchmark` transfers at most 64 MiB per directed pair with bounded 8 KiB scratch,
sequentially, and stores an EWMA of effective bandwidth. Unmeasured links are visibly
unmeasured and use a conservative 10 MB/s, 10 ms planning prior. There is no continuous
background bulk bandwidth test. Reported RTT includes authentication/dispatch overhead.

## Scheduling and heterogeneous GPUs

A `TrainingPlan` records job ID, leader generation, GPU assignments, compute node IDs,
RAM node IDs, placement policy, memory budget snapshot, topology snapshot and explicit
stage assignments. A node with `gpu_compute=false` cannot receive a GPU assignment.
No GPU produces `MemoryOnly`. With no explicit stage descriptions, one or more usable
GPUs produce an executable `SingleGpuDistributedMemory` plan containing only the
largest usable GPU (then stable GPU/node ID). The other GPUs remain visible inventory;
their VRAM is not counted as participating in that job.

The experimental heterogeneous planner accepts `{name, working_set_bytes}` stage
descriptions. It places each complete stage only on a GPU with enough usable capacity
and balances accumulated stage bytes relative to each GPU's usable capacity. Capacity
shares derive from real usable VRAM, not equal `1/N` partitions. The working set must
include the caller's estimates for activations, gradients and operator workspace.

Explicit multi-GPU stage plans are implemented and tested; executing those plans across
GPU hosts is not.
`execution_supported=false` makes this explicit in experimental plans. The Python
sequential adapter deliberately rejects such a plan. Future execution needs a bounded,
typed module/operator protocol, validation of model definitions and cross-stage gradient
transport—not an arbitrary remote Python/shell endpoint.
