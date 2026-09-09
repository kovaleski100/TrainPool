"""Explicit single-device sequential training with stage recomputation.

Parameters, gradients and optimizer tensors live in the RAM fabric between uses.
No arbitrary graph interception, remote Python workers, or CPU CUDA fallback.
"""

from __future__ import annotations

import contextlib
import copy
import types
from collections import OrderedDict
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


class SequentialRuntime:
    """Capacity state machine independent of the user's ``nn.Module`` identity."""

    def __init__(self, stages, store, device, prefetch_depth, gpu_budget_bytes):
        self.stages = stages
        self.store = store
        self.device = device
        self.prefetch = SequentialPrefetch(store) if prefetch_depth else NoPrefetch()
        self.gpu_budget_bytes = gpu_budget_bytes
        self.pending = False
        self.backward_complete = False
        self.failed = False
        self.closed = False

    def load_weights(self, index):
        return {
            name: self.store.restore(handle, device=self.device)
            for name, handle in self.stages[index].weights.items()
        }

    def schedule_weights(self, phase, index):
        if 0 <= index < len(self.stages):
            self.prefetch.schedule((phase, index), lambda: self.load_weights(index))

    def forward(self, value, *, training=True):
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
        if not training or not torch.is_grad_enabled():
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

    def convert_dtype(self, dtype):
        """Apply a dtype-only module conversion to remote weights stage by stage."""
        for record in self.stages:
            replacements = {
                name: self.store.offload(
                    self.store.restore(handle, device="cpu").to(dtype=dtype),
                    expected_next_use="forward",
                )
                for name, handle in record.weights.items()
            }
            for name, replacement in replacements.items():
                self.store.free(record.weights[name])
                record.weights[name] = replacement
            record.template.to(dtype=dtype)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.prefetch.close()
        self.store.close()


class CapacitySequential(nn.Module):
    """Backward-compatible explicit wrapper around :class:`SequentialRuntime`."""

    def __init__(self, runtime):
        super().__init__()
        self.templates = nn.ModuleList([s.template for s in runtime.stages])
        object.__setattr__(self, "runtime", runtime)

    @property
    def stages(self):
        return self.runtime.stages

    @property
    def store(self):
        return self.runtime.store

    @property
    def device(self):
        return self.runtime.device

    @property
    def prefetch(self):
        return self.runtime.prefetch

    @property
    def pending(self):
        return self.runtime.pending

    @pending.setter
    def pending(self, value):
        self.runtime.pending = value

    @property
    def backward_complete(self):
        return self.runtime.backward_complete

    @backward_complete.setter
    def backward_complete(self, value):
        self.runtime.backward_complete = value

    @property
    def failed(self):
        return self.runtime.failed

    @failed.setter
    def failed(self, value):
        self.runtime.failed = value

    def forward(self, value):
        return self.runtime.forward(value, training=self.training)

    def named_training_parameters(self, *, device="cpu"):
        yield from self.runtime.named_training_parameters(device=device)

    def close(self):
        self.runtime.close()


class CapacityOptimizer:
    def __init__(self, model, optimizer_type, options):
        self.model = model.runtime if isinstance(model, CapacitySequential) else model
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
                guard = getattr(model, "instrumentation_guard", contextlib.nullcontext)
                with guard():
                    optimizer = self.optimizer_type(list(parameters.values()), **self.options)
                for name, parameter in parameters.items():
                    if name in record.gradients:
                        parameter.grad = model.store.restore(record.gradients[name], device=model.device)
                    state = record.optimizer_state.get(name, {})
                    optimizer.state[parameter] = {
                        key: model.store.restore(value, device="cpu" if key == "step" else model.device)
                        if isinstance(value, TensorHandle)
                        else value
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
        except torch.OutOfMemoryError as error:
            model.failed = True
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_WORKING_SET: CUDA allocation during optimizer.step"
            ) from error
        except BaseException:
            model.failed = True
            raise


SUPPORTED_OPTIMIZERS = (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW)


def _validate_model(model):
    if not isinstance(model, nn.Sequential) or not len(model):
        raise ValueError("prepare currently supports nonempty nn.Sequential in capacity mode")
    all_parameters = list(model.named_parameters(remove_duplicate=False))
    if len({id(parameter) for _, parameter in all_parameters}) != len(all_parameters):
        raise ValueError("Tied/shared parameters are unsupported")
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
            if any(
                type(module) not in allowed or getattr(module, "inplace", False) for module in block.modules()
            ):
                raise ValueError("Unknown stage: annotate a pure block with trainpool_torch.stage()")
    return all_parameters


def _target_device(device, *, test_cpu):
    target = torch.device(device)
    if target.type == "cpu" and not test_cpu:
        raise TrainPoolError("TRAINPOOL_NO_CUDA: CPU execution requires the explicit test backend")
    if target.type not in ("cuda", "cpu"):
        raise ValueError("Unsupported compute device")
    if target.type == "cuda":
        if not torch.cuda.is_available():
            raise TrainPoolError("TRAINPOOL_NO_CUDA: run this script on the GPU node with CUDA PyTorch")
        target = torch.device(
            "cuda", target.index if target.index is not None else torch.cuda.current_device()
        )
    return target


def _optimizer_options(optimizer):
    if type(optimizer) not in SUPPORTED_OPTIMIZERS:
        raise ValueError("Only SGD, Adam and AdamW optimizers are supported")
    if len(optimizer.param_groups) != 1 or optimizer.state:
        raise ValueError("The MVP requires one parameter group and a fresh optimizer")
    options = {key: value for key, value in optimizer.param_groups[0].items() if key != "params"}
    options.pop("initial_lr", None)
    if type(optimizer) is torch.optim.AdamW:
        options.pop("decoupled_weight_decay", None)
    if options.get("differentiable") or options.get("capturable") or options.get("fused"):
        raise ValueError("Differentiable, capturable and fused optimizers are unsupported")
    options["foreach"] = False
    return options


def build_sequential_runtime(
    model,
    *,
    device,
    store,
    prefetch_depth=1,
    gpu_budget_bytes=None,
):
    """Move a validated Sequential's state into the fabric and return its runtime."""
    records = []
    for block in model:
        named = dict(block.named_parameters())
        weights = {
            name: store.offload(parameter, expected_next_use="forward") for name, parameter in named.items()
        }
        trainable = {name for name, parameter in named.items() if parameter.requires_grad}
        block.to("meta")
        records.append(_Stage(block, weights, trainable))
    return SequentialRuntime(records, store, device, prefetch_depth, gpu_budget_bytes)


def prepare_model_inplace(
    model,
    *,
    device,
    store,
    prefetch_depth=1,
    gpu_budget_bytes=None,
):
    """Attach capacity execution to the same model object."""
    _validate_model(model)
    runtime = build_sequential_runtime(
        model,
        device=device,
        store=store,
        prefetch_depth=prefetch_depth,
        gpu_budget_bytes=gpu_budget_bytes,
    )

    def transparent_forward(owner, value, *args, **kwargs):
        if args or kwargs:
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: FULL sequential mode accepts one tensor input")
        return runtime.forward(value, training=owner.training)

    def transparent_state_dict(owner, destination=None, prefix="", keep_vars=False):
        del owner
        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()
        for index, record in enumerate(runtime.stages):
            for name, handle in record.weights.items():
                tensor = runtime.store.restore(handle, device="cpu")
                destination[f"{prefix}{index}.{name}"] = tensor if keep_vars else tensor.detach()
        return destination

    def unsupported_load_state_dict(owner, *args, **kwargs):
        del owner, args, kwargs
        raise TrainPoolError(
            "TRAINPOOL_CHECKPOINT_UNSUPPORTED: load_state_dict is not yet safe for a prepared model"
        )

    def transparent_close(owner):
        del owner
        runtime.close()

    object.__setattr__(model, "_trainpool_runtime", runtime)
    object.__setattr__(model, "forward", types.MethodType(transparent_forward, model))
    object.__setattr__(model, "state_dict", types.MethodType(transparent_state_dict, model))
    object.__setattr__(model, "load_state_dict", types.MethodType(unsupported_load_state_dict, model))
    object.__setattr__(model, "close", types.MethodType(transparent_close, model))
    return runtime


def attach_optimizer_inplace(optimizer, runtime):
    """Delegate a supported optimizer without replacing or emptying it."""
    options = _optimizer_options(optimizer)
    delegate = CapacityOptimizer(runtime, type(optimizer), options)

    def zero_grad(owner, set_to_none=True):
        del owner
        return delegate.zero_grad(set_to_none=set_to_none)

    def step(owner, closure=None):
        delegate.options = _optimizer_options(owner)
        return delegate.step(closure=closure)

    def state_dict(owner):
        groups = []
        next_index = 0
        for group in owner.param_groups:
            encoded = {key: value for key, value in group.items() if key != "params"}
            encoded["params"] = []
            for _parameter in group["params"]:
                encoded["params"].append(next_index)
                next_index += 1
            groups.append(encoded)
        state = {}
        for index, (record, name) in enumerate(parameter_entries()):
            values = record.optimizer_state.get(name)
            if values:
                state[index] = {
                    key: runtime.store.restore(value, device="cpu")
                    if isinstance(value, TensorHandle)
                    else value
                    for key, value in values.items()
                }
        return {"state": state, "param_groups": groups}

    def parameter_entries():
        if hasattr(runtime, "parameter_order"):
            return [(runtime.records[name], name) for name in runtime.parameter_order]
        return [(record, name) for record in runtime.stages for name in record.weights]

    def load_state_dict(owner, checkpoint):
        if runtime.pending:
            raise TrainPoolError("TRAINPOOL_CHECKPOINT_UNSUPPORTED: pending training step")
        entries = parameter_entries()
        groups = checkpoint.get("param_groups", [])
        if len(groups) != 1 or len(groups[0].get("params", [])) != len(entries):
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_OPTIMIZER: checkpoint parameter groups")
        indices = groups[0]["params"]
        if len(set(indices)) != len(indices) or set(checkpoint["state"]) - set(indices):
            raise TrainPoolError("TRAINPOOL_CHECKPOINT_UNSUPPORTED: invalid parameter indices")
        options = {key: copy.deepcopy(value) for key, value in groups[0].items() if key != "params"}
        guard = getattr(runtime, "instrumentation_guard", contextlib.nullcontext)
        with guard():
            check = type(owner)(
                [nn.Parameter(torch.empty((), device="cpu"))],
                **{
                    key: value
                    for key, value in options.items()
                    if key not in ("initial_lr", "decoupled_weight_decay")
                },
            )
        _optimizer_options(check)
        staged = []
        try:
            for index, (record, name) in zip(indices, entries, strict=True):
                values = checkpoint["state"].get(index, {})
                allowed = (
                    {"momentum_buffer"}
                    if type(owner) is torch.optim.SGD
                    else {"step", "exp_avg", "exp_avg_sq", "max_exp_avg_sq"}
                )
                required = set() if type(owner) is torch.optim.SGD else {"step", "exp_avg", "exp_avg_sq"}
                if values and (set(values) - allowed or required - set(values)):
                    raise TrainPoolError(f"TRAINPOOL_CHECKPOINT_UNSUPPORTED: optimizer state for {name}")
                uploaded = {}
                staged.append((record, name, uploaded))
                for key, value in values.items():
                    if not isinstance(value, torch.Tensor):
                        if key != "step" or type(value) not in (int, float):
                            raise TrainPoolError(f"TRAINPOOL_CHECKPOINT_UNSUPPORTED: {name}.{key}")
                        value = torch.tensor(float(value))
                    if (key == "step" and value.numel() != 1) or (
                        key != "step" and tuple(value.shape) != record.weights[name].shape
                    ):
                        raise TrainPoolError(f"TRAINPOOL_CHECKPOINT_UNSUPPORTED: shape of {name}.{key}")
                    value = value.to(
                        device="cpu",
                        dtype=value.dtype if key == "step" else getattr(torch, record.weights[name].dtype),
                    )
                    uploaded[key] = runtime.store.offload(value, expected_next_use="optimizer.step")
        except BaseException:
            for _, _, state in staged:
                for value in state.values():
                    runtime.store.free(value)
            raise
        try:
            for record, name, state in staged:
                for value in record.optimizer_state.get(name, {}).values():
                    if isinstance(value, TensorHandle):
                        runtime.store.free(value)
                record.optimizer_state[name] = state
            owner.param_groups[0].update(options)
            delegate.options = _optimizer_options(owner)
        except BaseException:
            runtime.failed = True
            raise

    object.__setattr__(optimizer, "_trainpool_delegate", delegate)
    object.__setattr__(optimizer, "zero_grad", types.MethodType(zero_grad, optimizer))
    object.__setattr__(optimizer, "step", types.MethodType(step, optimizer))
    object.__setattr__(optimizer, "state_dict", types.MethodType(state_dict, optimizer))
    object.__setattr__(
        optimizer,
        "load_state_dict",
        types.MethodType(load_state_dict, optimizer),
    )
    return optimizer


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
    """Transfer ownership to the backward-compatible explicit wrapper API."""
    if mode != "capacity" or not isinstance(model, nn.Sequential) or not len(model):
        raise ValueError("prepare currently supports nonempty nn.Sequential in capacity mode")
    if prefetch_depth not in (0, 1):
        raise ValueError("prefetch_depth must be 0 or 1")
    all_parameters = _validate_model(model)
    options = _optimizer_options(optimizer)
    if {id(parameter) for _, parameter in all_parameters if parameter.requires_grad} != {
        id(parameter) for parameter in optimizer.param_groups[0]["params"] if parameter.requires_grad
    }:
        raise ValueError("Optimizer parameters must match the sequential model")
    target = _target_device(device, test_cpu=_test_cpu)
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
                "TRAINPOOL_NO_CUDA: single-GPU SDK must run on the selected primary GPU node"
            )
        try:
            store._require_cuda(target)
        except BaseException:
            if own_store:
                store.close()
            raise
        if gpu_budget_bytes is None:
            gpu_budget_bytes = store.plan["gpu_assignments"][0]["usable_bytes"]
    try:
        # Clear optimizer ownership before progressively releasing the original modules.
        optimizer.param_groups[0]["params"] = []
        runtime = build_sequential_runtime(
            model,
            device=target,
            store=store,
            prefetch_depth=prefetch_depth,
            gpu_budget_bytes=gpu_budget_bytes,
        )
        prepared = CapacitySequential(runtime)
        prepared.train(model.training)
        return prepared, CapacityOptimizer(runtime, type(optimizer), options)
    except BaseException:
        if own_store:
            store.close()
        raise
