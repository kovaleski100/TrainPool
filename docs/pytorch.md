# PyTorch adapter semantics

Run ordinary code with `trainpool python train.py`. The process-local bootstrap
intercepts `model.to("cuda")` / `model.cuda()` before full CUDA materialization,
captures the model, transfers state to the existing RAM fabric, and preserves the
root model and optimizer identities. RAM providers execute no Python and require
neither PyTorch nor CUDA. Normal SDK imports do not activate instrumentation.

## Graph execution

`graph.py` uses PyTorch FX as a capture frontend. Its `GraphIR` records producers,
group consumers, boundary inputs/outputs, sizes, remaining consumers and next use.
The original DAG remains intact. Adjacent operations are grouped (up to 12 nodes,
with boundaries at fan-out and parameter-budget limits), retaining every value with
an external consumer. Multiple inputs and outputs are supported. If metadata
execution estimates that a group exceeds the primary GPU budget, the runtime
splits it and repeats admission. A single operator that still exceeds the estimate
fails with `TRAINPOOL_UNSUPPORTED_WORKING_SET` before tensor restoration.

Forward executes one group at a time under no-grad with `torch.func.functional_call`.
Only its parameters, buffers and inputs are restored. All tensor boundaries,
including skips needed by later decoder groups, are offloaded immediately. Local
intermediates remain ordinary PyTorch tensors within the group. Tensor payloads use
independent fabric handles; tuple/list/dict/ordered-mapping metadata stays inside
the process and is not pickled or sent as executable Python objects.

Backward visits the DAG in reverse topological group order. It restores the required
boundary inputs and state, recomputes locally with autograd, and uploads input and
parameter gradients. Contributions at fan-out are **added**, including when separate
outputs contribute to the same input. Gradients are never replaced by the last
branch. Saved boundary values are freed after their final backward use. In inference,
RAM boundaries can be freed after their final forward consumer. GPU copies live only
for the active group and gradient accumulation operation; the input/output tensors
held by the user's script remain the script's responsibility.

## Mutable buffers and randomness

BatchNorm running mean, variance and batch counters are persistent RAM-backed state.
Forward saves the pre-forward buffers, executes once and commits the resulting state.
Backward restores the pre-forward snapshot into temporary tensors and discards any
recomputation mutations. It never commits these mutations to persistent buffers.
Each group's CPU and selected-device CUDA RNG states are recorded. Replay runs inside
`torch.random.fork_rng`, reproducing Dropout while restoring the user's global state.

Module training/evaluation modes are propagated before execution. FX specializes
Python boolean flags: a model using functional Dropout whose captured root training
mode changes is explicitly rejected. Module-based Dropout/BatchNorm mode changes work.
Unknown stateful leaves, hooks, tied/reused parameters and alias-sensitive in-place
operations are rejected. Safe leaf activations are executed without in-place mutation.

## Optimizers and checkpoints

SGD, Adam and AdamW use the existing capacity optimizer, restoring one execution
group at a time. No update begins until DAG backward has completed. Parameters,
gradients and tensor-valued optimizer state return to RAM. Multiple parameter groups,
optimizer closures, populated optimizers at initial interception, capturable, fused
and differentiable optimizers are explicitly unsupported. Scheduler changes to the
supported group's options are read on each step.

`model.state_dict()`, `model.load_state_dict(...)`, `optimizer.state_dict()` and
`optimizer.load_state_dict(...)` work for supported transparent models. Snapshots
contain ordinary CPU tensors in original model/optimizer parameter order, including
BatchNorm buffers. Loading validates key/shape/group compatibility and stages all
uploads before publishing replacements. Model loading supports `strict=False`;
`assign=True` and loading during a pending training step are rejected. A transport
failure during publication makes the job unusable; this is not distributed recovery.
Checkpoint I/O is an explicit application action, not fabric disk spill. Full
snapshots require enough host RAM, just as initial model construction does.

User-visible parameter objects are meta placeholders. Their ordinary `.grad` fields
are not a GPU gradient cache; gradients are owned by the capacity optimizer. Gradient
clipping and arbitrary operations on placeholder parameters are not supported.
`runtime.named_training_parameters(device="cpu")` is an advanced inspection interface.
One forward/backward/step may be in flight; retained graphs, gradient accumulation
across batches, higher-order gradients and AMP/GradScaler are not supported.

## Capacity and transfers

There is exactly one selected GPU, including for explicitly supplied stage plans.
CUDA movement checks both the selected node and GPU UUID. If CUDA is unavailable,
`TRAINPOOL_NO_CUDA` is emitted. There is no CPU fallback. `TRAINPOOL_TEST_CPU=1` is
only the transparent numerical-test backend, explicitly reported as `test-cpu`.

Shape admission runs metadata kernels, accounting conservatively for parameters,
buffers, all group intermediates, gradients and workspace headroom. Available CUDA
memory is checked again before restoring a group. Backend-dependent CUDA workspace
allocation can still fail; such failures are reported as
`TRAINPOOL_UNSUPPORTED_WORKING_SET`, never retried by materializing the full model or
running CPU kernels. **These estimates are not an allocator-enforced CUDA limit.**
Physical peak residency and actual workspace behavior remain hardware release gates.

Graph execution uses deterministic demand restoration. It does not concurrently
prefetch another group's GPU parameters. The explicit Sequential API retains its
optional one-worker look-ahead prefetch, with `prefetch_depth=0/1`.

The default tensor block size is at most 4 MiB, independently of 64 KiB host staging
pieces. Upload/download streaming verifies chunk and complete-block checksums. This
keeps transfer staging bounded even on a node whose RAM contribution is smaller than
one parameter. The daemon's RAM accounting covers SDK staging and relay reservations.
Normal placement is automatic; no node UUID is needed in a training script.

## Supported graph operations

The allowlist includes dense convolutions (including dilation and ConvTranspose2d),
Linear, BatchNorm, LayerNorm, GroupNorm, common activations, Dropout, pooling,
Upsample, functional interpolation/padding, concatenation, residual arithmetic,
shape access, indexing and standard reshape/transpose operations. It is a bounded
initial frontend, not arbitrary PyTorch compatibility. Diagnostics identify the
unsupported graph/node/operator/buffer/output or working set.
Shared buffers, buffers outside the captured graph, complex training state and
custom checkpoint extra state are rejected before replacing the model's tensors.

`FULL` means this runtime owns the model parameters, buffers, boundary activations,
gradients and optimizer state. `PARTIAL` never claims to fix full-model CUDA OOM.
Unsupported capture fails at CUDA model movement, before ordinary PyTorch can place
the whole model there. Models with input-dependent Python control flow, custom
operators, unsupported state mutation, DDP/FSDP and multi-GPU compute remain outside
this milestone.

## Advanced existing APIs

`TensorStore`, `distributed_memory`, `stage`, and `prepare(nn.Sequential, optimizer)`
remain available. The explicit Sequential wrapper retains its original pure-stage
restrictions and ownership-transfer semantics. `distributed_memory` alone offloads
saved activations and does not claim to solve parameter or optimizer capacity.
Close explicit stores/runtimes with `try/finally` or context management. Transparent
stores are closed at process exit. Leases bound retention after unreachable-node
cleanup; warnings identify cleanup deferred to expiry.

## API references

The implementation uses PyTorch's [FX graphs](https://docs.pytorch.org/docs/2.14/fx.html),
[functional calls](https://docs.pytorch.org/docs/2.14/generated/torch.func.functional_call.html)
and [autograd](https://docs.pytorch.org/docs/2.14/autograd.html). Accepted DeepLab
features follow the actual [torchvision implementation](https://docs.pytorch.org/vision/stable/_modules/torchvision/models/segmentation/deeplabv3.html).
