# Research notes

The experimental objective is maximizing executable workload size subject to safe
RAM/VRAM capacity and correct training semantics. Throughput is measured as a cost,
not assumed to improve when another memory provider joins.

## Experimental variables

Keep topology, placement and prefetch behind their existing interfaces. Candidate
extensions include measured-cost placement, trace-driven next use, adaptive block size,
look-ahead depth under a residency constraint, tensor replication and graph-stage
planning. Remote VRAM should principally support compute on its owner rather than
being treated as an ordinary network storage device.

The implemented baseline uses local-first RAM, transfer-cost-ranked remote RAM and
no prefetch/one-stage sequential prefetch. Heterogeneous stage placement is planning-only.
The initial training algorithm trades recomputation and network traffic for low live
parameter residency. A single operation's memory requirement remains a hard lower bound.

## Suggested measurements

Compare the same model, data, precision, optimizer and number of steps under:

1. Ordinary one-GPU PyTorch.
2. TrainPool with local RAM only.
3. TrainPool with one GPU node and a CPU-only remote RAM node.
4. The same remote configuration with prefetch disabled/enabled.

Increase model stages/width or batch size until each baseline fails or reaches an
explicit limit. A claimed capacity improvement requires an actual baseline OOM under
the target hardware configuration and a successful, numerically checked TrainPool run.
A configured stage budget or CPU simulation is useful for tests but does not prove a
physical CUDA OOM has been overcome. No such physical-hardware result is claimed here.

Record software versions, GPU UUID/model, physical and free VRAM, RAM budget samples,
OS paging configuration, interfaces, directed bandwidth and latency estimates, stages,
block size, prefetch mode, loss trajectory and elapsed time. Use fixed random seeds and
compare gradients/updates within documented numerical tolerances before scaling up.

Commands:

```sh
trainpool nodes --json
trainpool memory --json
trainpool benchmark --mib 16 --json
trainpool topology --json
trainpool metrics --json
```

Saving **aggregate JSON metrics** to experiment files is allowed; saving tensors is
not part of these demos. Capture metrics on every node with a common job UUID. Physical
memory from two processes on the same machine must not be counted as two real hosts.

## Metric interpretation

| Counter | Meaning |
|---|---|
| `tensor_allocations` | RAM block allocations, including provisional migration destinations |
| `tensor_migrations` | Source-observed, verified and published migrations |
| `bytes_local_to_remote_ram` | SDK relay uploads to remote owners plus source migration bytes |
| `bytes_remote_ram_to_local` | Reads relayed from remote owners to the local SDK |
| `network_bytes` | Raw owner endpoint payload bytes; includes local loopback data paths |
| `network_wait_ms` | Cumulative time in remote SDK relay operations |
| `migration_latency_ms` | Sum of source copy/commit/publication latencies |
| `prefetch_hits` / `prefetch_misses` | Whether a predicted stage was ready on consumption |
| `gpu_wait_for_data_ms` | Foreground CUDA data restore/prefetch wait wall time |
| `peak_gpu_residency` / `current_gpu_residency` | SDK-reported PyTorch process allocator observations |
| `peak_ram_residency` / `current_ram_residency` | Per-job payload RAM ownership on this node, released with the last live allocation reference |
| `ram_pressure_events` | Local pressure samples affecting that job |
| `failed_transfers` | Explicit migration copy failures (connection failures are also logged) |

Remote byte counts are directional and source/local-node scoped. Do not sum endpoint
`network_bytes` and relay direction counters and call the result unique bytes. Estimate
throughput using counter deltas divided by the measurement interval. Topology bandwidth
is an active probe estimate, not the workload's end-to-end effective tensor bandwidth.
CUDA idle time, detailed latency histograms and tracing of individual CUDA kernels are
future profiler integrations; the MVP exposes wall-clock data wait, not those measurements.

## Current robustness limits

No consensus, replicated storage, automatic job recovery, disk checkpoint, resume or
remote GPU process launcher. Metadata is volatile and kept for the daemon lifetime;
completed-job history/tombstone compaction is future work. Large inventories are
exchanged in bounded rotating pages; newly elected leaders can conservatively report
unknown data and fail jobs instead of waiting for full reconciliation. Control buffers
and allocator/process overhead are outside payload accounting. Kernel TCP buffers and
OS swapping remain OS-managed. Long operations can exceed deadlines/leases and fail
explicitly. Changing membership during a job can invalidate its plan.

Before using untrusted networks, replace TCP with authenticated encryption, enforce
per-job authorization and resource quotas, add request rate limits and bounded metadata
retention, and specify partition behavior with quorum fencing. These are engineering
extensions; they are not assumed properties of the current prototype.
