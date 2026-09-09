# Validation record and v1 release gates

This milestone is **not a TrainPool v1 release**. Physical CUDA execution and a real
U-Net baseline OOM overcome have been demonstrated using two daemons on one host.
Distributed RAM across separate physical machines remains unverified; the operator
confirmed that a second machine is not currently available.

## Environment

Development validation on 2026-09-09 used Linux, Python 3.12.3, PyTorch 2.14.0+cpu,
torchvision 0.29.0+cpu and Rust/Cargo 1.98.1. The initial sandbox hid the GPU from
`nvidia-smi`. Verification outside the sandbox found an NVIDIA GeForce RTX 3050
Laptop GPU (4096 MiB), driver 580.95.05 / CUDA 13.0. Physical CUDA validation uses
a separate CUDA-enabled PyTorch environment. A second physical machine has not
been identified; local daemons must not be described as separate physical hosts. Test tools were installed in isolated `/tmp`
directories; these paths are development conveniences, not product dependencies.
Network tests require local TCP/UDP socket access outside the restrictive sandbox.

## What each validation tier establishes

| Tier | Workloads / evidence | Physical capacity claim |
|---|---|---|
| Automated CPU simulation | U-Net and real torchvision DeepLabV3/ResNet50 DAGs; loss, input/parameter gradients, updated parameters, BatchNorm state, RNG and checkpoints | None |
| Local multiprocess fabric | Identical daemons, loopback multicast, leases, remote blocks, streaming/checksums, launcher scripts, pressure/migration and failure semantics | None; processes share host RAM |
| Real physical multi-machine | Reproducible workflow below; not executed here | Pending |
| Real CUDA execution | RTX 3050 Laptop: U-Net numerical parity with SGD/Adam/AdamW; two-step real DeepLab smoke run | Passed for these workloads; DeepLab float32 caveat below |
| Real OOM-overcome experiment | Same U-Net source/model/AdamW/GPU; baseline allocation OOM, TrainPool completion and positive remote bytes | Passed with same-host RAM daemons; separate-machine repetition pending |

A software group budget only tests admission/partitioning. It is never evidence of
physical CUDA capacity. Backend workspace requirements are estimated, not enforced
by a custom CUDA allocator; hardware validation must confirm actual allocated and
reserved peaks before release.

## Automated checks

The required CI commands are:

```sh
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --locked --release
ruff check python tests/python examples scripts
ruff format --check python tests/python examples scripts
pytest -q
python scripts/two_node_demo.py
```

The final CPU suite passed **37 tests**, with 3 opt-in CUDA tests skipped; Rust
passed **22 tests** (5 runtime/CLI,
14 core, 3 fabric). Formatting, Clippy with warnings denied, release build and the
release-binary two-node demo passed. Physical CUDA tests are opt-in and skipped in
CPU CI unless an explicit CUDA fabric endpoint is supplied.

CI also checks Ruff formatting/linting of `tests/models` and installs CPU torchvision
so the real DeepLab path runs rather than being silently skipped.

Rust coverage includes deterministic largest-GPU selection for both automatic and
explicit plans, ties, inventory versus executable capacity, CPU-only leaders,
RAM accounting under pressure/concurrency, leases, generations, direct transfers,
migration success/rejected destination, checksums and interrupted uploads. A partial
upload cannot be committed/read; complete retransmission is verified. Explicit failed
jobs reject status/allocation. The release-binary multiprocess demo verifies 12 MiB
of payload, 9 MiB of remote writes and reads, both directed topology links, leader
loss/re-election, explicit lost-block errors, and zero tensor files.

Python coverage includes:

- Actual encoder/decoder U-Net with skip connections, two concatenations, BatchNorm,
  pooling, ConvTranspose, functional interpolation and Dropout.
- Real torchvision `deeplabv3_resnet50(weights=None, weights_backbone=None,
  aux_loss=False, num_classes=3)`, including residuals, ASPP, dilation, adaptive
  pooling, BatchNorm, interpolation and a mapping output.
- Reduced-resolution float64 comparisons against ordinary PyTorch with SGD, Adam
  and AdamW. All parameter gradients, input gradients, updated parameters and buffers
  are compared, not only loss. Loss tolerance is `rtol=2e-5, atol=2e-6`; gradients
  `rtol=1e-5, atol=1e-6`; state `rtol=2e-4, atol=3e-6`. Batch counters and RNG state
  are checked exactly through integer/byte tensor comparison.
- Two-iteration float32 zero-TrainPool-import U-Net and DeepLab scripts launched
  through `trainpool python ...` against two daemons. Tests require actual remote
  read/write counters. Float32 loss tolerance is `rtol=2e-3, atol=2e-3`, accounting
  for residual/BatchNorm accumulation order. DeepLab uses four 32x32 images;
  U-Net uses two 16x16 images. U-Net checkpoint state is compared too.
- Structured inputs/outputs, unused output branches, immutable buffer dependencies,
  fan-out accumulation, metadata-based group splitting and impossible-group failure.
- Early rejection of shared/uncaptured buffers, complex training state and custom
  checkpoint extra state, without changing the original parameters.
- Ordinary model/optimizer checkpoint save/load and continued training with all
  three optimizers; invalid model shapes leave the previous state unchanged.
- Forty U-Net training iterations and more than 10,000 tensor allocations with a
  bounded live-handle count; 1,100 remote allocation/roundtrips with a retained tensor
  surviving its original lease deadline through renewal and prefetch restoration.
- Existing explicit Sequential/prefetch, saved activation hooks, dtype/scalar/empty
  tensor, no implicit CPU compute, and no disk-backing regressions.

This is useful stress coverage, not certification for unlimited duration or arbitrary
network failures. Single-copy RAM loss explicitly fails the job; there is no automatic
recovery. Longer real-network soak runs remain part of physical validation.

The first hosted CI run exposed unstable float32 loss comparison for DeepLab with
only two 16x16 images: both steps completed, but the second loss differed by 0.00675.
The launcher test now uses the same four-image, 32x32 configuration as the strict
float64 comparisons, while retaining its original loss tolerance. A local diagnostic
found an initial stem-gradient peak of 263.65 with the tiny batch versus 3.84 with
the larger one; first-forward losses matched exactly in both cases. These observations
support sensitivity to rounding amplified through BatchNorm and AdamW, rather than
establishing general float32 parity. The broader numerical limitation remains explicit.

## Physical CUDA checks on a local fabric

Executed on the RTX 3050 outside the sandbox with PyTorch 2.14.0+cu130 and
torchvision 0.29.0+cu130: **3 CUDA parity tests passed**, covering two U-Net steps
for each of SGD, Adam and AdamW. Every gradient, parameter and buffer comparison
used `rtol=1e-5, atol=1e-6`; CPU/CUDA RNG bytes matched exactly. The store was backed
by the second local daemon and remote read/write counters were positive.

The real torchvision DeepLabV3/ResNet50 also completed two AdamW iterations through
the launcher (`--size 32 --batch 2 --steps 2`). Allocated CUDA memory peaked at
681,417,216 bytes in ordinary PyTorch and 272,730,112 bytes in TrainPool. The first
loss matched exactly; second losses were 1.15995765 and 1.15494597 respectively.
This float32 run establishes execution and capacity reduction, **not strict CUDA
DeepLab numerical parity**. Reduced-model float64 gradient/state parity is covered
separately in CPU tests. Investigating and bounding float32 DeepLab differences on
physical CUDA remains part of broader numerical validation.

The actual OOM comparison used the unchanged U-Net source with width 768, image
size 32, batch 2 and one AdamW iteration:

```sh
python scripts/validate_hardware.py --address 127.0.0.1:17432 \
  --local-fabric --output /tmp/trainpool-real-cuda-oom-unet \
  --model unet --width 768 --size 32 --batch 2 --steps 1 --optimizer AdamW
```

| Measurement | Ordinary PyTorch | TrainPool |
|---|---|---|
| Result | Genuine CUDA allocation OOM in `optimizer.step()` | Forward, backward and AdamW step completed |
| Loss before update | 1.1410276889801025 | 1.1410276889801025 |
| Peak CUDA allocated | 3,617,735,168 bytes | 2,038,547,456 bytes |
| Peak CUDA reserved | 3,760,193,536 bytes | 2,069,889,024 bytes |
| Completed step time | None | 121.24 seconds |
| Remote RAM reads / writes | Not applicable | 4,397,810,688 / 5,428,812,800 bytes |

No GPU quota, competing artificial allocation or injected failure was used.
`nvidia-smi` reports 4096 MiB physical VRAM; PyTorch reports 3,951,296,512 bytes
available to CUDA. Both processes were pinned to the same GPU UUID. The compute
daemon contributed at most 4 MiB RAM and the second daemon at most 4096 MiB. The
second daemon's peak job residency was 3,557,283,840 bytes. Both nodes reported zero
owned RAM after completion. This is a **single-step capacity demonstration**, not
a long-run stability result or an updated-parameter comparison with the failed
baseline. Status inventory refreshes periodically, so immediate post-run snapshots
may lag the live metrics.

Raw evidence is retained in [validation-results/2026-09-09](validation-results/2026-09-09/README.md),
including commands, source hashes, GPU inventory, per-node metrics, topology,
allocator diagnostics and CUDA parity test results. The reported physical RAM and
loopback bandwidth must not be interpreted as independent hosts or LAN performance.

A real NVIDIA device can also be checked with two daemons on the same host. This
establishes CUDA execution and TCP transport, not physical multi-machine RAM:

```sh
TRAINPOOL_CUDA_ADDRESS=127.0.0.1:7432 pytest -q tests/python/test_cuda.py
python scripts/validate_hardware.py --address 127.0.0.1:7432 \
  --local-fabric --allow-fitting-baseline --output /tmp/trainpool-cuda-smoke \
  --model deeplab --size 32 --batch 2 --steps 2
```

The parity test compares CPU and CUDA RNG state, all U-Net gradients and updated
parameters/buffers with SGD, Adam and AdamW on the actual device. The harness's
`--local-fabric` mode always keeps the separate-machine release gate false.
`--allow-fitting-baseline` supports a CUDA smoke check; it never turns a fitting
baseline into an OOM result. Both flags are explicitly reflected in report evidence.

## Physical hardware workflow

Use two **separate physical machines**. Verify their hostnames, asset identity and
network endpoints manually; separate daemon UUIDs alone do not prove separate hosts.

Machine B (CPU-only is sufficient):

```sh
trainpool daemon
# In another terminal, collect the host's own inventory/metrics as needed:
trainpool status --json
trainpool metrics --json
```

Machine A (selected NVIDIA GPU, CUDA PyTorch and torchvision installed):

```sh
# A contribution ceiling encourages remote RAM backing; it does not limit GPU VRAM.
trainpool daemon --ram-limit-mib 256
# In another terminal:
python scripts/validate_hardware.py --output /tmp/trainpool-physical-unet \
  --model unet --width 128 --size 512 --batch 2 --steps 3 --optimizer AdamW
```

Choose workload dimensions appropriate for the physical GPU. The initial command may
fit ordinary PyTorch; in that case the harness **fails the OOM gate**. Increase model
width or image size and use a new output directory. Do not occupy the GPU artificially,
use `set_per_process_memory_fraction`, inject an OOM, or pass a software GPU budget to
manufacture the result. Every individual operation must remain feasible on the actual
primary GPU. Scale width and resolution independently to explore these constraints.
For DeepLab use `--model deeplab` and scale resolution/batch; width is not changed.

The harness fixes `CUDA_VISIBLE_DEVICES` to the selected primary GPU UUID for both
processes, runs the same acceptance script/model/optimizer arguments, captures stderr
and exit status, and requires a real baseline CUDA allocation failure. It then runs
the same command under TrainPool and requires successful forward/backward/step and
positive remote read/write counters. It never invokes a remote shell or Python worker.
It rejects CPU simulation and unavailable CUDA. Do not run competing GPU workloads
between baseline and TrainPool; record the physical GPU configuration consistently.

`report.json` and process logs include GPU identity/physical VRAM (`nvidia-smi -q`),
PyTorch version, selected plan, RAM budgets per node, per-node residency/peak metrics,
remote bytes, topology bandwidth/latency, losses, step times and CUDA peak
allocated/reserved memory. The ordinary script emits allocator peaks on successful
completion; an OOM baseline retains its allocator failure diagnostics in stderr.
Inspect the report and independently attest the physical-host identity. An experiment
pass is one release gate, not automatic approval of the entire release.

For soak validation, repeat with `--steps 1000` and an appropriate `--timeout`, record
RSS/owned RAM plateaus, and run separate controlled provider-loss/interrupted-network
experiments on disposable jobs. Require numerical continuation or explicit TrainPool
failure; never classify a silent mismatch as success. Stopping the RAM provider loses
single-copy data and is expected to fail affected jobs.

## Release gates still open

- Broader physical CUDA working-set validation, float32 DeepLab numerical validation
  and soak coverage.
- Distributed RAM across verified separate physical machines.
- Repeat the OOM-overcome experiment with remote RAM on a separate physical machine.
- Longer physical soak/fault runs and final v1 release review.

Do not tag or advertise v1 until these gates and the automated checks are green.
