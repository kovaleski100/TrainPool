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
* CPU-only RAM providers, deterministic single-GPU compute placement, topology
  estimates and on-demand directed bandwidth measurements.
* Transparent PyTorch activation through `trainpool python ...`, plus tensor
  offload/restore, saved activation hooks and DAG training with RAM-backed
  parameters, live skip tensors, gradients, buffers and SGD/Adam/AdamW optimizer state.
* One-stage look-ahead prefetch, structured logs, per-job counters and repeatable tests.

**Status: progress toward v1, not a v1 release.** Transparent graph execution supports
U-Net and torchvision DeepLabV3/ResNet50, with skip/residual branches, concatenation,
BatchNorm, interpolation and structured outputs. CPU numerical and local TCP tests
exercise these paths. Physical RTX 3050 tests include U-Net numerical parity,
DeepLab execution, and a real U-Net baseline OOM overcome using a local multiprocess
RAM fabric. Separate-machine RAM and longer hardware soak tests remain release
gates; see [validation](docs/validation.md).

Exactly **one GPU** computes: largest eligible `usable_vram`, with GPU UUID then node
UUID breaking ties. Every explicit group assignment uses it too. Other GPUs remain
inventory. Leadership is independent and may belong to a CPU-only RAM provider.
No multi-GPU execution or implicit CPU fallback is implemented.

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

## Quick start

Install the Python adapter in the training interpreter, then run the same TrainPool
binary on participating machines. A RAM-provider machine runs:

```sh
trainpool daemon
```

On the compute machine, change only the command line:

```sh
# Before
python train.py

# With TrainPool (the local runtime starts automatically if needed)
trainpool python train.py
trainpool python train.py --epochs 100
trainpool python -m package.training

trainpool metrics
```

`train.py` must not import TrainPool. The launcher verifies that its exact Python
interpreter contains `trainpool_torch`, obtains a training plan, starts or reuses one
local daemon, injects a process-local Python bootstrap, and cleans it up when the child
exits. Existing environment settings, including virtualenv/Conda selection and
`CUDA_VISIBLE_DEVICES`, are inherited.

The legacy `trainpool run -- python train.py` spelling remains an alias.

A normal `train.py` can use the standard torchvision API:

```python
import torch
from torchvision.models.segmentation import deeplabv3_resnet50

model = deeplabv3_resnet50(weights=None, weights_backbone=None, aux_loss=False)
model = model.to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for _ in range(3):
    images = torch.randn(2, 3, 128, 128, device="cuda")
    masks = torch.randint(21, (2, 128, 128), device="cuda")
    optimizer.zero_grad()
    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs["out"], masks)
    loss.backward()
    optimizer.step()
torch.save(model.state_dict(), "model.pt")  # explicit user checkpoint
```

The runnable [U-Net](tests/models/unet_train.py) and
[DeepLab](tests/models/deeplab_train.py) acceptance scripts have no TrainPool imports.
Run `trainpool python tests/models/unet_train.py` or
`trainpool python tests/models/deeplab_train.py`. The initial model must fit host RAM,
and every individual CUDA operator plus its active working set must fit the primary
GPU. Graphs that cannot be captured or admitted fail explicitly.

### Capacity reporting

`cluster_physical_vram` and `cluster_usable_vram` describe inventory. For a new job,
`current_job_backing_capacity = primary_gpu_usable_vram + pool_ram_budget`;
`current_job_remaining_capacity = primary_gpu_usable_vram + pool_ram_allocatable`.
`pool_ram_owned` is already part of the budget; it is never added a second time.
These are logical **training-state** capacities, not a single CUDA allocation.
Other jobs, staging and changing host pressure can reduce available capacity.

### Cluster inspection

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

### Advanced raw-memory exercise

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

For CUDA training, choose a larger contribution ceiling on B. Automatic allocation
uses the existing local-first/cost/capacity/pressure/load placement policy; no node UUID
belongs in the normal training command:

```sh
# Machine A: restart with room for 1 MiB tensor staging.
trainpool daemon --ram-limit-mib 16
# Machine B: restart with a suitable safe ceiling, e.g. 1024 MiB.
trainpool daemon --ram-limit-mib 1024
# Machine A:
trainpool python train.py
```

For a real capacity experiment, use a segmentation model and the reproducible
[hardware workflow](docs/validation.md#physical-hardware-workflow). It runs the same
source, model, optimizer and GPU in both processes and requires an actual baseline
CUDA allocation failure. A configured residency budget is not physical OOM evidence.

## Advanced Python API

The explicit adapter remains available for research, debugging, placement experiments,
and stage annotation. It is not required by normal training programs:

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
original optimizer relinquishes its parameters. `TensorStore(preferred_node=...)`,
`distributed_memory()` and `stage()` are likewise advanced interfaces. Transparent
stores always pass `preferred_node=None`. See [adapter semantics](docs/pytorch.md) for
supported modules and restrictions.

## Reproduce without a GPU

Install CPU PyTorch to run numerical training tests. The standalone RAM demo needs
only the SDK dependencies, not torch:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --locked --release
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e './python[test]'
pytest -q
ruff check python tests/python tests/models examples scripts
ruff format --check python tests/python tests/models examples scripts
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
`TRAINPOOL_CLUSTER_SECRET`; the launcher propagates these automatically. Python
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
replication, automatic checkpoint recovery, arbitrary Python control-flow capture or
automatic remote execution. Explicit model and optimizer checkpoint save/load are supported.
The original model must initially fit in the script's host RAM, and each compute stage
must fit its GPU. OS-managed paging is outside TrainPool's control; TrainPool creates
no payload files or swap. See the linked documents for precise accounting limits.
