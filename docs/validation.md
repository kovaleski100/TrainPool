# Validation record

Executed on 2026-09-08 on one Linux x86_64 development machine. This record describes
actual executed checks, not a physical multi-machine or CUDA capacity measurement.

Environment: Linux 6.14.0-33, glibc 2.39, Rust/Cargo 1.98.1, Python 3.12.3,
PyTorch 2.14.0+cpu and NumPy 2.5.3. `torch.cuda.is_available()` returned false.
The locked Rust dependencies require Rust 1.95 or newer.

## Executed checks

| Check | Result |
|---|---|
| `cargo fmt --check` | Passed |
| `cargo clippy --locked --offline --all-targets -- -D warnings` | Passed, no warnings |
| `cargo test --locked --offline` | 14 passed: 12 core and 2 TCP fabric tests |
| `cargo build --locked --offline --release` | Passed; one runtime target named `trainpool` |
| Editable Python SDK install without dependency downloads | Passed |
| `pytest -q` | 13 passed |
| `ruff check python tests/python examples scripts` | Passed |
| `ruff format --check python tests/python examples scripts` | Passed |
| Release binary `--help` | All daemon/CLI commands available |
| `daemon --enable-disk-spill` | Rejected with explicit not-implemented error |
| `config set ram-fraction 0.95` | Rejected as outside the safe range |
| `python scripts/two_node_demo.py` against release binary | Passed |

The initial sandbox denied loopback socket creation. Network integration checks were
run with permission to bind local TCP and UDP multicast sockets; tests were not disabled
or changed to skip networking. Rust and Python dependencies were installed into isolated
directories under `/tmp` because the initial environment had neither Rust nor PyTorch.

## Release binary demonstration

Two identical daemons discovered each other via UDP multicast on loopback with **no
seed nodes**. Their metadata directories and UUIDs were distinct. The first node had a
4 MiB contribution ceiling and the second 24 MiB; both remained subject to safe RAM
accounting. The following were observed:

* 12,582,912 payload bytes allocated in 24 independently placed blocks.
* 9,437,184 bytes (9 MiB) stored on the second node; 3 MiB stored on the first.
* Every block read and verified, including full-object BLAKE3 checksums.
* 9,437,184 remote upload bytes and the same number of remote read bytes recorded by
  the first daemon for the demonstration job.
* Two directed network links measured through explicit 1 MiB probes.
* The leader process killed; survivor elected itself within the demo's failover deadline.
* A block stored exclusively on the killed node rejected with `TRAINPOOL_DATA_LOST`.
* Zero tensor files created; node metadata files were byte-for-byte unchanged by the workload.

This proves the data path, accounting and failure behavior across separate processes.
It does not double the host's actual physical RAM or prove real-network bandwidth.

## Training checks

For SGD with momentum/weight decay, Adam and AdamW, tests compared three training
steps against ordinary PyTorch with prefetch both disabled and enabled. Forward values,
input gradients and updated model parameters matched within `rtol=1e-5, atol=1e-6`.
The training objects were backed by the second daemon's RAM, and actual remote byte
counters were asserted positive.

Additional tests cover saved activation hook gradients, non-contiguous and bfloat16
tensors, scalar and empty tensors, explicit rejection of implicit CPU compute,
stateful-buffer rejection and local-only SDK endpoints. An annotated Dropout stage
matched ordinary PyTorch and preserved global RNG state across recomputation.

The example training script completed with 24 width-128 stages using AdamW on the
explicit CPU test backend. Its parameters exceeded the configured 1 MiB residency
admission budget, while the script observed remote backing and completed forward,
backward and optimizer update. **That admission budget is not a physical GPU limit.**

Rust simulations cover the mandatory 64 GiB CPU-only leader versus 32+12 GiB GPU node,
single-GPU compute assignment with CPU-only remote memory, heterogeneous capacities,
static election/ties, RAM pressure math, concurrent reservations, lease expiry/ownership,
stale inventory handling, driver-field fallback, checksum failure, wrong authentication,
oversized frames and failed-destination migration retaining the source copy.

## Hardware work still required

Run the CUDA example on a real NVIDIA compute node and a separate RAM-only machine.
Compare a baseline physical CUDA OOM against successful TrainPool training, measure
actual GPU residency and network performance, and verify loss/updates numerically.
Cross-host heterogeneous GPU execution, QUIC/TLS, consensus, replication and job
recovery are not implemented or claimed validated.

## Local artifacts

The built Linux runtime is `target/release/trainpool` (approximately 6.2 MiB).
Its SHA-256 for this validation run is:

```
f451be9cc2f6f0443cbc1db93569bb171ebbfeb181071c5f837ef0889e5adedd
```

Temporary development tooling for this workspace can be used with:

```sh
export CARGO_HOME=/tmp/trainpool-cargo
export RUSTUP_HOME=/tmp/trainpool-rustup
export PATH="/tmp/trainpool-cargo/bin:$PATH"
cargo build --release
/tmp/trainpool-venv/bin/python -m pytest -q
```

These `/tmp` locations are session-local conveniences, not runtime dependencies or
portable installation instructions. Standard build/run instructions are in the README.
