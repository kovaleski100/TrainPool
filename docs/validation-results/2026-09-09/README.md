# Physical CUDA evidence — 2026-09-09

These are original local experiment reports from an NVIDIA GeForce RTX 3050 Laptop
GPU. **Both daemons ran on the same physical machine.** Their distinct node UUIDs
and positive remote counters establish cross-process TCP backing, not separate-host
RAM or LAN performance. The harness therefore recorded `release_gate_passed: false`.

- [unet-oom.json](unet-oom.json): the same width-768 U-Net and AdamW configuration
  failed with an actual CUDA allocation OOM in ordinary PyTorch, then completed one
  iteration through `trainpool python`. Includes full baseline allocator diagnostics,
  process outputs, source hashes, GPU identity, RAM metrics and transport counters.
- [deeplab-smoke.json](deeplab-smoke.json): two iterations of real torchvision
  DeepLabV3/ResNet50 in CUDA. Both processes completed; this is not an OOM result.
  The second float32 loss differs by approximately 0.005 and this report is not
  evidence of strict numerical parity.
- [cuda-parity.xml](cuda-parity.xml): three passing physical CUDA U-Net tests, two
  steps each with SGD, Adam and AdamW. Float64 losses, gradients, parameters,
  BatchNorm buffers and RNG replay were compared against ordinary PyTorch.

The reports include periodic status snapshots as well as live per-node metrics.
Immediate post-run status can still show the preceding sample; live metrics report
zero owned RAM after cleanup. Summed physical host RAM in same-host daemon inventory
counts the same machine twice and must not be interpreted as independent capacity.

The archived reports retain the harness wording at execution time. Their
`physical_host_verification` reminder does not attest separate machines. Test tools
and virtual environments lived in `/tmp`; those absolute paths are historical
provenance, not installation requirements. See [the validation record](../../validation.md)
for reproduction commands, tolerances and remaining release gates.
