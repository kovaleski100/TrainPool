# Contributing to TrainPool

TrainPool is an independently implemented Apache-2.0 research prototype. Contributions
must prioritize correct training semantics and memory accounting before throughput.

Install a current stable Rust toolchain, Python 3.10+ and CPU PyTorch for tests:

```sh
cargo build
python -m venv .venv
. .venv/bin/activate
pip install -e './python[test]'
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
ruff check python tests/python examples scripts
ruff format --check python tests/python examples scripts
pytest -q
python scripts/two_node_demo.py --binary target/debug/trainpool
```

Tests need permission to bind loopback TCP and UDP multicast sockets. They do not
require NVIDIA hardware or contact production clusters. Python tests launch temporary
daemons on random ports with small RAM contribution ceilings and clean them up.

Include tests for changes to the wire protocol, accounting, leases, migration or
training semantics. Protocol incompatibilities need a version bump. Avoid new payload
serialization formats that execute code, shell-based execution endpoints, tensor files,
implicit CPU fallback, or algorithms that replicate the complete model on each GPU.

State exactly which hardware and software were used for measurements. Keep generated
tensor data in RAM; commit only code, configuration, documentation and aggregate results.
Do not describe a simulated GPU capacity limit as a physical CUDA OOM experiment.

Run the relevant checks before submitting a pull request. Describe the concrete
problem, resulting behavior, tests and any unresolved limitations. Contributions are
provided under Apache-2.0. No third-party proprietary runtime source is included.
