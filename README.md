# TrainPool

**Train larger workloads by moving training state through distributed RAM. Capacity
comes before speed.** TrainPool is an Apache-2.0 research prototype with one Rust
runtime executable, `trainpool`, and a Python SDK. Every daemon is both a network
peer and a RAM provider; a CPU-only peer can also be the elected leader.

TrainPool exposes **logical training memory**, consisting of distinct RAM and VRAM
tiers. It does not combine CUDA address spaces or replicate a full model using DDP.
Tensor files, disk spill and TrainPool-managed swap are absent from the implementation.

## What works

* UDP multicast discovery, authenticated capability exchange, 2-second heartbeats,
  7-second failure detection and deterministic leader election by physical RAM + VRAM.
* Safe RAM budgets, atomic ownership accounting, bounded SDK/relay staging, RAM blocks,
  BLAKE3 checksums, chunked transfers, lease renewal/free and direct node-to-node migration.
* CPU-only RAM providers, a single-GPU training plan, explicit heterogeneous stage
  planning, topology estimates and on-demand directed bandwidth measurements.
* PyTorch tensor offload/restore, saved activation hooks and actual sequential training
  with remote parameters, inputs, gradients and SGD/Adam/AdamW optimizer state.
* One-stage look-ahead prefetch, structured logs, per-job counters and repeatable tests.

**Scope:** actual training execution supports one GPU and explicitly separated pure
sequential stages. The heterogeneous GPU planner generates unequal stage assignments;
cross-host multi-GPU execution is not implemented. CUDA execution requires the Python
script to run on the GPU node. CPU execution is available only through an explicit test
backend. The Rust daemon has no remote shell or Python execution endpoint.

## Build

Use a current stable Rust toolchain (Rust 1.95+ for the locked dependencies) and a C
compiler. The prototype is exercised on Linux; multicast and hardware discovery on
other operating systems need validation.

```sh
cargo build --release
./target/release/trainpool --help
```

The sole runtime binary is `target/release/trainpool`. The Rust library exists to test
the same runtime logic; there are no server/client/agent executable targets.

Install the SDK on the compute machine. Install the appropriate CUDA-enabled PyTorch
for that machine separately; RAM-only machines need neither Python nor CUDA.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e ./python
# Install PyTorch following https://pytorch.org/get-started/locally/.
```

## Run on two machines

Run the **same binary** on both machines on a trusted LAN:

```sh
trainpool daemon
```

Default ports: TCP 7432 for framed control/data connections, UDP multicast
239.255.74.32:7433 for discovery. Permit these on the participating network interfaces.
For hosts with multiple interfaces, set `--advertise-ip` and `--multicast-interface`.

```sh
trainpool nodes
trainpool leader
trainpool memory
trainpool status --json
trainpool benchmark --mib 16
trainpool topology
trainpool metrics --json
```

`metrics` reports the local node, grouped by job UUID; query each node for a complete
experiment. `benchmark` asks the leader to measure all directed peer pairs. Ordinary
heartbeats measure control RTT without sending bandwidth-test payloads.

To exercise remote capacity safely, limit the GPU host's contribution and provide
more RAM on the second host. These are contribution ceilings, not fictitious physical
memory or GPU capacity:

```sh
# Machine A
trainpool daemon --ram-limit-mib 4
# Machine B
trainpool daemon --ram-limit-mib 64
# Machine A, in another terminal (SDK installed)
python examples/memory_demo.py --mib 12
```

The demo creates 12 MiB using 512 KiB blocks, reads and checks every block, prints
remote bytes and metrics, and frees the blocks. It requires actual remote placement
to pass. It exceeds A's configured contribution while keeping local staging bounded.

For CUDA training, choose a larger contribution ceiling on B and use its UUID:

```sh
# Machine A: restart with room for 1 MiB tensor staging.
trainpool daemon --ram-limit-mib 16
# Machine B: restart with a suitable safe ceiling, e.g. 1024 MiB.
trainpool daemon --ram-limit-mib 1024
# Machine A:
trainpool nodes
trainpool run -- python examples/train_sequential.py --memory-node B_UUID
```

The default model has more than 64 MiB of parameters and uses a configured 64 MiB
stage admission budget. AdamW state and saved activations are also backed by RAM.
The script checks that bytes actually traveled to and from remote RAM. Scale `--layers`
until full-model state exceeds physical GPU capacity for a real hardware experiment;
each individual stage, its recomputation and optimizer working set must still fit.
The configured budget is an admission estimate, **not a CUDA allocator hard limit**.
Physical-OOM comparisons require running the baseline and measuring both processes
on the target GPU; the CPU simulation does not establish that result.

Minimal adapter usage:

```python
import torch
from torch import nn
import trainpool_torch as tp

model = nn.Sequential(
    nn.Sequential(nn.Linear(1024, 1024), nn.Tanh()),
    nn.Linear(1024, 1024),
)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
model, optimizer = tp.prepare(model, optimizer, mode="capacity", prefetch_depth=1)
try:
    optimizer.zero_grad()
    loss = model(torch.randn(16, 1024, device="cuda")).square().mean()
    loss.backward()
    optimizer.step()
finally:
    model.close()
```

`prepare` transfers ownership: the original model becomes meta templates and the
original optimizer relinquishes its parameters. Use the returned pair. For saved
activations alone, keep both forward and backward inside `with tp.distributed_memory():`.
See [adapter semantics](docs/pytorch.md) for supported modules and restrictions.

## Reproduce without a GPU

Install CPU PyTorch to run numerical training tests. The standalone RAM demo needs
only the SDK dependencies, not torch:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
pip install -e './python[test]'
pytest -q
ruff check python tests/python examples scripts
python scripts/two_node_demo.py
```

`two_node_demo.py` launches two release binaries with different metadata directories,
uses multicast with no seeds, verifies remote RAM transfers, benchmarks both directions,
kills the leader, verifies re-election/data loss, and checks that metadata directories
contain no tensor files. It terminates all subprocesses. Its 4 MiB/24 MiB contribution
ceilings are intentionally small. Two local processes are a protocol simulation: do
not interpret their summed physical memory announcements as two physical machines.

For a small complete training simulation, start two daemons as above and run:

```sh
python examples/train_sequential.py --test-cpu --width 128 --layers 24 \
  --resident-budget-mib 1 --block-mib 1 --memory-node B_UUID
```

## Configuration and security

Only metadata/configuration is stored in `~/.trainpool/`: a persistent random `node_id`
and optional `config.toml`. `--data-dir` or `TRAINPOOL_HOME` selects another directory.
Never run two live installations with the same directory/node ID.

```sh
trainpool config set ram-fraction 0.50
trainpool config set vram-reserve-bytes 536870912
trainpool config set vram-reserve-fraction 0.05
trainpool config set chunk-bytes 67108864
trainpool config show
```

Restart the daemon to apply configuration. RAM fractions must be 0.10–0.90; the default
is exactly 0.50. Contribution ceilings can only reduce the budget.

Set the same environment on participating daemons and SDK processes:

```sh
export TRAINPOOL_CLUSTER_NAME=my-lab
export TRAINPOOL_CLUSTER_SECRET='your-own-high-entropy-shared-secret'
```

Alternatively persist `cluster-name`/`cluster-secret` using `config set`. `config show`
redacts the secret. The SDK reads `TRAINPOOL_ADDRESS`, `TRAINPOOL_CLUSTER_NAME` and
`TRAINPOOL_CLUSTER_SECRET`; `trainpool run` propagates these automatically. Python
connects only to a loopback address, never directly to remote peers.

This framed TCP MVP authenticates peers using challenge-response HMAC-SHA256 and signs
discovery packets. **It does not encrypt traffic or provide per-user isolation.** Use
an isolated trusted LAN or an encrypted VPN; all holders of the cluster secret are
trusted. Without a secret, membership is intentionally open within the named cluster.
Transport traits allow a later QUIC/TLS implementation.

When multicast is unavailable, supply known endpoints on each side:

```sh
trainpool daemon --no-discovery --seed 100.64.0.2:7432 --advertise-ip 100.64.0.1
```

Peers must advertise mutually reachable addresses. The MVP does not implement general
membership gossip beyond multicast and configured/known direct peers. Routed clusters
should configure enough seeds for a complete membership view.

`trainpool daemon --enable-disk-spill` explicitly returns “not implemented”.

## Documentation and limits

* [Architecture and leadership](docs/architecture.md)
* [Wire protocol and failure semantics](docs/protocol.md)
* [Memory accounting and migration](docs/memory-model.md)
* [PyTorch adapter](docs/pytorch.md)
* [Research notes and experiment design](docs/research-notes.md)
* [Validation record](docs/validation.md)

This is single-copy, in-memory research software. Node loss can lose data and fails
affected jobs; leader changes invalidate existing plans. There is no consensus,
replication, checkpointing, arbitrary graph partitioning or automatic remote execution.
The original model must initially fit in the script's host RAM, and each compute stage
must fit its GPU. OS-managed paging is outside TrainPool's control; TrainPool creates
no payload files or swap. See the linked documents for precise accounting limits.
