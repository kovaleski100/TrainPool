# PyTorch adapter semantics

The SDK is a library in `python/trainpool_torch`, not a second daemon. It requires a
local loopback TrainPool endpoint and contains no torch.distributed RPC, pickle or
remote Python execution. RAM-only peers need only the Rust binary.

## Three levels

`TensorStore.offload(tensor)` returns a logical `TensorHandle` containing full shape,
dtype, byte length, original device, contiguous layout, expected next use, dirty/clean
state and ordered remote block handles. `restore(handle)` allocates on the recorded
device and fills it from checked chunks. `free` releases RAM blocks. Restoring a CUDA
tensor never falls back to CPU because its backing node lacks a GPU.

`distributed_memory()` uses PyTorch saved-tensors hooks to pack saved activations into
the store and unpack them for backward. The restored values have identical dtype,
shape and content. Tensor ownership objects release blocks when the autograd graph
releases them; the context closes any remaining objects. Finish backward inside the
context. This mechanism alone does not release ordinary model parameters or optimizer
states: use `prepare` to manage those objects at explicit stage boundaries.

`prepare(nn.Sequential, optimizer)` transfers ownership and returns a
`CapacitySequential` and `CapacityOptimizer`. It supports SGD, Adam and AdamW with one
parameter group and initially empty optimizer state. Initial model parameters must
fit in the host process before preparation. Parameters are uploaded one stage at a
time and the original modules become metadata-only PyTorch meta templates. Do not
reuse the original model/optimizer or keep external references expecting them to be
updated in place.

## Training algorithm

Forward uses a custom autograd function:

1. Restore the current stage's weights on the one selected CUDA device.
2. Offload that stage's input and remember CPU/CUDA random-number state.
3. Start look-ahead loading the next stage's weights if enabled.
4. Execute `torch.func.functional_call` with the restored weights under no-grad.
5. Release the current temporary weights. Only the output goes to the next stage.

Backward traverses stages in reverse:

1. Restore that stage's saved input and unchanged weights.
2. Replay its saved RNG inside `torch.random.fork_rng`, recompute with autograd enabled,
   and compute the input/parameter gradients with `torch.autograd.grad`.
3. Pass the input gradient to the preceding stage; offload parameter gradients to RAM.
4. Release input/weight/recomputation objects and free the consumed saved input.

The returned optimizer processes one stage at a time. It restores parameters,
gradients and that stage's tensor-valued momentum/Adam state, executes the ordinary
PyTorch optimizer, uploads new parameters/state, and frees previous blocks only after
uploads succeed. Adam's scalar step tensors retain their original device. No parameter
update happens before all stage backwards complete, preserving ordinary training
semantics. An error marks the prepared model failed; a partial optimizer step is not
silently retried or claimed to be transactional across stages.

## Prefetch

`prefetch_depth=0` selects `NoPrefetch`. Depth one selects `SequentialPrefetch`: one
worker and at most one pending stage. It predicts the next forward stage or previous
backward stage. CUDA copies are explicit and synchronous with respect to their staging
buffer; the background worker overlaps them with the foreground stage. A “hit” means
the future had completed at consumption. A “miss” means it was absent or still pending.
Pending futures are drained/cleared by `model.close()`.

Only one batch can be in flight. Parameters of the active and prefetched stages may
coexist, in addition to the operator's live activations/workspace. Admission uses a
conservative estimate, not an allocator interception. The user must choose stage
boundaries whose **actual** working set fits the GPU. Disable prefetch when the extra
stage would use needed VRAM. Throughput is secondary to capacity.

## Supported scope

Built-in pure stages include Linear, dense convolution, LayerNorm, common non-inplace
activations, Flatten, Identity and nested Sequential. Inputs/outputs are single floating
point tensors. Parameters must not be shared/tied. Mutable buffers such as BatchNorm
running statistics are rejected. Unknown pure single-input/output blocks can be
explicitly annotated with `tp.stage(module)`; the caller promises no buffer mutation,
input mutation or external side effects. RNG replay supports recomputable randomness.

Unsupported: arbitrary graphs, DDP, automatic graph partitioning, multiple GPU execution,
multiple optimizer groups, populated pre-prepare optimizer state, closures, differentiable/
capturable/fused optimizers, automatic mixed precision/GradScaler integration, gradient
accumulation, retained graphs, higher-order gradients, shared weights, sparse/quantized
tensors, and transparent standard `state_dict`/optimizer serialization. Use
`named_training_parameters(device=...)` to inspect one restored parameter at a time.
Collecting that iterator into a dict deliberately materializes a full snapshot in RAM.

Zero-length tensors are metadata-only; scalar, bfloat16 and non-contiguous tensors are
covered by round-trip tests. Non-contiguous normalization may allocate a full contiguous
copy on the original device. Models prepared in evaluation/no-grad mode run stages
without saving backward inputs.

CUDA mode verifies both PyTorch CUDA availability and that the local daemon is the
single planned GPU compute provider. A CPU-only leader cannot execute a CUDA stage.
`_test_cpu=True` plus `device="cpu"` is an explicit numerical test backend; it is never
selected from tensor location or driver failure. `examples/train_sequential.py --test-cpu`
labels its output as simulation.

## Metrics and cleanup

The daemon records actual remote byte counts. The SDK reports CUDA transfer bytes,
prefetch hits/misses, foreground data waiting time and PyTorch allocator residency.
Allocator numbers are process observations; use one job per process when attributing
them to a job. Waiting time is wall time blocked on data, not a CUDA profiler's exact
GPU-idle measurement. There is no fabricated GPU metric for the CPU backend.

Use `try/finally: model.close()` or a `TensorStore` context, and close the prefetcher
before closing a supplied store. A lease renewal thread retains live RAM objects.
If cleanup cannot reach a node, expiry bounds retention; the cleanup warning identifies
this rather than pretending all remote allocations were freed.

## Primary API references

The implementation uses the documented
[saved-tensors hooks and autograd APIs](https://docs.pytorch.org/docs/2.14/autograd.html)
and [functional module calls](https://docs.pytorch.org/docs/stable/generated/torch.func.functional_call.html).
Tests compare outputs, input gradients and updated parameters against ordinary
PyTorch across multiple steps for SGD, Adam and AdamW, both with and without prefetch.
