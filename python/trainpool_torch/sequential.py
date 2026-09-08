"""Explicit single-device sequential training with stage recomputation.

Parameters, gradients and optimizer tensors live in the RAM fabric between uses.
No arbitrary graph interception, remote Python workers, or CPU CUDA fallback.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
from torch import nn

from .client import TrainPoolError
from .store import NoPrefetch, SequentialPrefetch, TensorHandle, TensorStore


def stage(module):
    """Annotate a pure, single-tensor-in/single-tensor-out recomputable block.

    The caller promises no mutable buffers, side effects, tied weights or in-place
    mutation of inputs. RNG is replayed during backward.
    """
    module._trainpool_explicit_stage = True
    return module


@dataclass
class _Stage:
    template: nn.Module
    weights: dict
    trainable: set
    gradients: dict = field(default_factory=dict)
    optimizer_state: dict = field(default_factory=dict)


class _SequentialAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, trigger, runtime):
        ctx.runtime = runtime
        ctx.consumed = False
        ctx.inputs = []
        ctx.rng = []
        ctx.input_requires_grad = value.requires_grad
        try:
            for index, _ in enumerate(runtime.stages):
                weights = runtime.prefetch.take(("forward", index), lambda i=index: runtime.load_weights(i))
                saved = runtime.store.offload(value, expected_next_use=f"backward:{index}")
                ctx.inputs.append(saved)
                ctx.rng.append(
                    (
                        torch.random.get_rng_state(),
                        torch.cuda.get_rng_state(runtime.device) if runtime.device.type == "cuda" else None,
                    )
                )
                runtime.schedule_weights("forward", index + 1)
                value = torch.func.functional_call(runtime.stages[index].template, weights, (value,))
                if not isinstance(value, torch.Tensor):
                    raise TypeError("TrainPool stages must return one tensor")
                del weights
            return value
        except BaseException:
            runtime.pending = False
            for saved in ctx.inputs:
                with contextlib.suppress(Exception):
                    runtime.store.free(saved)
            raise

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        if ctx.consumed:
            raise TrainPoolError("TrainPool supports one backward per forward; retain_graph is unsupported")
        ctx.consumed = True
        runtime = ctx.runtime
        try:
            for index in reversed(range(len(runtime.stages))):
                record = runtime.stages[index]
                weights = runtime.prefetch.take(("backward", index), lambda i=index: runtime.load_weights(i))
                value = (
                    runtime.store.restore(ctx.inputs[index], device=runtime.device)
                    .detach()
                    .requires_grad_(True)
                )
                runtime.schedule_weights("backward", index - 1)
                names = [name for name in weights if name in record.trainable]
                for name in names:
                    weights[name].requires_grad_(True)
                devices = [runtime.device.index] if runtime.device.type == "cuda" else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    torch.random.set_rng_state(ctx.rng[index][0])
                    if runtime.device.type == "cuda":
                        torch.cuda.set_rng_state(ctx.rng[index][1], runtime.device)
                    output = torch.func.functional_call(record.template, weights, (value,))
                    gradients = torch.autograd.grad(
                        output,
                        [value] + [weights[name] for name in names],
                        grad_outputs=grad_output,
                        allow_unused=True,
                    )
                grad_output = gradients[0]
                if grad_output is None:
                    grad_output = torch.zeros_like(value)
                for name, gradient in zip(names, gradients[1:], strict=True):
                    if gradient is not None:
                        record.gradients[name] = runtime.store.offload(
                            gradient, expected_next_use="optimizer.step"
                        )
                runtime.store.free(ctx.inputs[index])
                del weights, output, gradients, value
            runtime.backward_complete = True
            runtime.store.flush_metrics()
            return grad_output if ctx.input_requires_grad else None, None, None
        except BaseException:
            runtime.failed = True
            raise
        finally:
            for saved in ctx.inputs:
                with contextlib.suppress(Exception):
                    runtime.store.free(saved)


class CapacitySequential(nn.Module):
    def __init__(self, stages, store, device, prefetch_depth, gpu_budget_bytes):
        super().__init__()
        self.templates = nn.ModuleList([s.template for s in stages])
        self.stages = stages
        self.store = store
        self.device = device
        self.prefetch = SequentialPrefetch(store) if prefetch_depth else NoPrefetch()
        self.gpu_budget_bytes = gpu_budget_bytes
        self.pending = False
        self.backward_complete = False
        self.failed = False

    def load_weights(self, index):
        return {
            name: self.store.restore(handle, device=self.device)
            for name, handle in self.stages[index].weights.items()
        }

    def schedule_weights(self, phase, index):
        if 0 <= index < len(self.stages):
            self.prefetch.schedule((phase, index), lambda: self.load_weights(index))

    def forward(self, value):
        if value.device != self.device:
            raise TrainPoolError(f"TRAINPOOL_NO_CUDA_FALLBACK: input must be on {self.device}")
        if self.failed:
            raise TrainPoolError("Training failed; create a new prepared job")
        if self.pending:
            raise TrainPoolError("Complete backward and optimizer.step/zero_grad before the next forward")
        if not value.is_floating_point():
            raise TypeError("Sequential MVP accepts floating-point inputs")
        if self.gpu_budget_bytes:
            largest = max((sum(h.size for h in s.weights.values()) for s in self.stages), default=0)
            # Conservative admission estimate; actual operator workspaces still must fit.
            required = 6 * largest + 3 * value.numel() * value.element_size()
            if required > self.gpu_budget_bytes:
                raise TrainPoolError(
                    f"TRAINPOOL_STAGE_TOO_LARGE: estimate {required} exceeds stage budget {self.gpu_budget_bytes}"
                )
        if not self.training or not torch.is_grad_enabled():
            with torch.no_grad():
                for index, record in enumerate(self.stages):
                    weights = self.load_weights(index)
                    value = torch.func.functional_call(record.template, weights, (value,))
                    del weights
            return value
        self.pending = True
        self.backward_complete = False
        trigger = torch.empty((), device=self.device, requires_grad=True)
        return _SequentialAutograd.apply(value, trigger, self)

    def named_training_parameters(self, *, device="cpu"):
        """Yield one parameter at a time; caller controls snapshot RAM consumption."""
        for index, record in enumerate(self.stages):
            for name, handle in record.weights.items():
                yield f"{index}.{name}", self.store.restore(handle, device=device)

    def close(self):
        self.prefetch.close()
        self.store.close()


class CapacityOptimizer:
    def __init__(self, model, optimizer_type, options):
        self.model = model
        self.optimizer_type = optimizer_type
        self.options = options

    def zero_grad(self, set_to_none=True):
        if not set_to_none:
            raise ValueError("Capacity optimizer supports zero_grad(set_to_none=True)")
        if self.model.pending and not self.model.backward_complete:
            raise TrainPoolError("Cannot zero gradients while a forward is awaiting backward")
        for record in self.model.stages:
            for handle in record.gradients.values():
                self.model.store.free(handle)
            record.gradients.clear()
        self.model.pending = False
        self.model.backward_complete = False

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Optimizer closures are unsupported")
        model = self.model
        if model.failed or not model.backward_complete:
            raise TrainPoolError("Complete a successful backward before optimizer.step")
        try:
            for record in model.stages:
                if not record.trainable:
                    continue
                parameters = {
                    name: nn.Parameter(model.store.restore(record.weights[name], device=model.device))
                    for name in record.weights
                    if name in record.trainable
                }
                optimizer = self.optimizer_type(list(parameters.values()), **self.options)
                for name, parameter in parameters.items():
                    if name in record.gradients:
                        parameter.grad = model.store.restore(record.gradients[name], device=model.device)
                    state = record.optimizer_state.get(name, {})
                    optimizer.state[parameter] = {
                        key: model.store.restore(value) if isinstance(value, TensorHandle) else value
                        for key, value in state.items()
                    }
                optimizer.step()
                # Publish new state only after successful uploads. Old state remains valid on failure.
                new_weights = {
                    name: model.store.offload(parameter, expected_next_use="next forward")
                    for name, parameter in parameters.items()
                }
                new_states = {
                    name: {
                        key: model.store.offload(value, expected_next_use="next optimizer.step")
                        if isinstance(value, torch.Tensor)
                        else value
                        for key, value in optimizer.state[parameter].items()
                    }
                    for name, parameter in parameters.items()
                }
                for name, handle in new_weights.items():
                    model.store.free(record.weights[name])
                    record.weights[name] = handle
                for state in record.optimizer_state.values():
                    for value in state.values():
                        if isinstance(value, TensorHandle):
                            model.store.free(value)
                record.optimizer_state = new_states
                for handle in record.gradients.values():
                    model.store.free(handle)
                record.gradients.clear()
                del parameters, optimizer
            model.pending = False
            model.backward_complete = False
            model.store.flush_metrics()
        except BaseException:
            model.failed = True
            raise


def prepare(
    model,
    optimizer,
    *,
    mode="capacity",
    device="cuda",
    store=None,
    preferred_node=None,
    prefetch_depth=1,
    gpu_budget_bytes=None,
    _test_cpu=False,
):
    """Transfer ownership of an explicit Sequential and fresh SGD/Adam/AdamW.

    Do not reuse the input model or optimizer after preparation. The returned
    model owns meta templates and remote parameters. CPU execution is confined to
    the explicitly selected test backend, never chosen from memory placement.
    """
    if mode != "capacity" or not isinstance(model, nn.Sequential) or not len(model):
        raise ValueError("prepare currently supports nonempty nn.Sequential in capacity mode")
    if type(optimizer) not in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
        raise ValueError("Only SGD, Adam and AdamW optimizers are supported")
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise ValueError("The MVP requires one parameter group and a fresh optimizer")
    if prefetch_depth not in (0, 1):
        raise ValueError("prefetch_depth must be 0 or 1")
    target = torch.device(device)
    if target.type == "cpu" and not _test_cpu:
        raise TrainPoolError("TRAINPOOL_NO_CUDA: CPU execution requires the explicit test backend")
    if target.type not in ("cuda", "cpu"):
        raise ValueError("Unsupported compute device")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainPoolError("TRAINPOOL_NO_CUDA: run this script on the GPU node with CUDA PyTorch")
        target = torch.device(
            "cuda", target.index if target.index is not None else torch.cuda.current_device()
        )
    options = {k: v for k, v in optimizer.param_groups[0].items() if k != "params"}
    options.pop("initial_lr", None)
    if type(optimizer) is torch.optim.AdamW:
        # AdamW stores this inherited Adam flag but does not accept it in __init__.
        options.pop("decoupled_weight_decay", None)
    if options.get("differentiable") or options.get("capturable") or options.get("fused"):
        raise ValueError("Differentiable, capturable and fused optimizers are unsupported")
    options["foreach"] = False
    all_parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(p) for _, p in all_parameters}) != len(all_parameters):
        raise ValueError("Tied/shared parameters are unsupported")
    if {id(p) for _, p in all_parameters if p.requires_grad} != {
        id(p) for p in optimizer.param_groups[0]["params"] if p.requires_grad
    }:
        raise ValueError("Optimizer parameters must match the sequential model")
    allowed = (
        nn.Linear,
        nn.ReLU,
        nn.GELU,
        nn.Tanh,
        nn.Sigmoid,
        nn.SiLU,
        nn.Flatten,
        nn.Identity,
        nn.LeakyReLU,
        nn.ELU,
        nn.Softplus,
        nn.Sequential,
        nn.LayerNorm,
        nn.Conv1d,
        nn.Conv2d,
    )
    for block in model:
        if list(block.buffers()):
            raise ValueError("Mutable/stateful buffers are unsupported; use pure stages")
        if not getattr(block, "_trainpool_explicit_stage", False):
            if any(type(m) not in allowed or getattr(m, "inplace", False) for m in block.modules()):
                raise ValueError("Unknown stage: annotate a pure block with trainpool_torch.stage()")
    own_store = store is None
    store = store or TensorStore(preferred_node=preferred_node)
    if target.type == "cuda":
        if (
            store.plan["strategy"] != "SingleGpuDistributedMemory"
            or store.client.local_node not in store.plan["compute_nodes"]
        ):
            if own_store:
                store.close()
            raise TrainPoolError(
                "Single-GPU SDK must run on the elected GPU compute node; heterogeneous execution is planning-only"
            )
        if gpu_budget_bytes is None:
            gpu_budget_bytes = store.plan["gpu_assignments"][0]["usable_bytes"]
    records = []
    try:
        # Clear optimizer ownership before progressively releasing the original modules.
        optimizer.param_groups[0]["params"] = []
        for block in model:
            named = dict(block.named_parameters())
            weights = {
                name: store.offload(parameter, expected_next_use="forward")
                for name, parameter in named.items()
            }
            trainable = {name for name, parameter in named.items() if parameter.requires_grad}
            block.to("meta")
            records.append(_Stage(block, weights, trainable))
        prepared = CapacitySequential(records, store, target, prefetch_depth, gpu_budget_bytes)
        prepared.train(model.training)
        return prepared, CapacityOptimizer(prepared, type(optimizer), options)
    except BaseException:
        if own_store:
            store.close()
        raise
