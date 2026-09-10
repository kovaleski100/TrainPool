"""Single-device DAG execution with RAM-backed boundaries and local autograd.

FX is the capture frontend. GraphIR owns group dependencies and value lifetimes;
no graph boundary tensors are retained on the GPU between groups. Only tensor
payloads enter TensorStore; containers remain validated, process-local metadata.
"""

from __future__ import annotations

import contextlib
import copy
import inspect
import operator
import types
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum

import torch
from torch import fx, nn
from torch.nn import functional as F
from torch.utils._pytree import tree_flatten, tree_unflatten

from .client import TrainPoolError
from .sequential import SequentialRuntime, _Stage
from .store import TensorHandle


class Residency(str, Enum):
    GPU_RESIDENT = "GPU_RESIDENT"
    LOCAL_RAM = "LOCAL_RAM"
    REMOTE_RAM = "REMOTE_RAM"
    PREFETCHING = "PREFETCHING"


def flatten(value):
    def normalize(item):
        if type(item) in (dict, OrderedDict, fx.immutable_collections.immutable_dict):
            kind = OrderedDict if type(item) is OrderedDict else dict
            return kind((key, normalize(child)) for key, child in item.items())
        if type(item) in (list, fx.immutable_collections.immutable_list):
            return [normalize(child) for child in item]
        if type(item) is tuple:
            return tuple(normalize(child) for child in item)
        return item

    value = normalize(value)

    def validate(item):
        if isinstance(item, torch.Tensor) or item is None or type(item) in (bool, int, float, str, slice):
            return
        if type(item) in (tuple, list, dict, OrderedDict, torch.Size):
            if isinstance(item, dict):
                if any(type(key) not in (str, int) for key in item):
                    raise TrainPoolError(
                        "TRAINPOOL_UNSUPPORTED_OUTPUT: mapping keys must be strings or integers"
                    )
                item = item.values()
            for child in item:
                validate(child)
            return
        raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OUTPUT: {type(item).__name__}")

    validate(value)
    return tree_flatten(value)


def map_tree(function, value):
    leaves, spec = flatten(value)
    return tree_unflatten([function(x) if isinstance(x, torch.Tensor) else x for x in leaves], spec)


MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose2d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.ReLU,
    nn.LeakyReLU,
    nn.GELU,
    nn.SiLU,
    nn.ELU,
    nn.Tanh,
    nn.Sigmoid,
    nn.Softplus,
    nn.Identity,
    nn.Flatten,
    nn.Dropout,
    nn.Dropout2d,
    nn.MaxPool2d,
    nn.AvgPool2d,
    nn.AdaptiveAvgPool2d,
    nn.Upsample,
)
FUNCTIONS = {
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.getitem,
    getattr,
    torch.cat,
    torch.concat,
    torch.add,
    torch.flatten,
    torch.relu,
    torch.sigmoid,
    torch.tanh,
    F.relu,
    F.leaky_relu,
    F.gelu,
    F.silu,
    F.interpolate,
    F.pad,
    F.max_pool2d,
    F.avg_pool2d,
    F.adaptive_avg_pool2d,
    F.dropout,
    F.dropout2d,
}
METHODS = {"size", "dim", "reshape", "view", "flatten", "contiguous", "permute", "transpose"}


@dataclass
class GraphValue:
    producer: str
    consumers: tuple
    remaining_consumers: int = 0
    size: int = 0
    residency: Residency = Residency.LOCAL_RAM
    next_use: int | None = None


@dataclass
class Group:
    module: fx.GraphModule
    inputs: tuple
    outputs: tuple
    nodes: tuple
    estimated_working_set: int = 0
    forward_peak: int = 0
    backward_peak: int = 0
    optimizer_peak: int = 0
    workspace_margin: int = 0


@dataclass
class GraphIR:
    frontend: fx.GraphModule
    groups: list
    values: dict
    placeholders: tuple
    output: object


def capture(model):
    parameters = list(model.named_parameters(remove_duplicate=False))
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if len({id(p) for _, p in parameters}) != len(parameters):
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: tied/shared parameters")
    if len({id(b) for b in buffers.values()}) != len(buffers):
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_BUFFER: tied/shared buffers")
    if any(p.is_complex() for _, p in parameters) or any(b.is_complex() for b in buffers.values()):
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: complex training state")
    extra_state = set(model.state_dict()) - {name for name, _ in parameters} - buffers.keys()
    if extra_state:
        raise TrainPoolError(f"TRAINPOOL_CHECKPOINT_UNSUPPORTED: extra state {sorted(extra_state)}")
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks for m in model.modules()):
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: module hooks are not replay-safe")
    try:
        graph = fx.symbolic_trace(model)
    except Exception as error:
        raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_GRAPH: {type(model).__name__}: {error}") from error
    used_parameters = set()
    used_buffers = set()
    for node in graph.graph.nodes:
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            if type(module) not in MODULES:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPERATOR: {node.name}: {type(module).__name__}")
            names = {f"{node.target}.{name}" for name, _ in module.named_parameters()}
            if names & used_parameters:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_GRAPH: reused parameter at {node.name}")
            used_parameters.update(names)
            buffer_names = {f"{node.target}.{name}" for name, _ in module.named_buffers()}
            if buffer_names & used_buffers:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_BUFFER: reused state at {node.name}")
            used_buffers.update(buffer_names)
            if getattr(module, "inplace", False) and any(len(n.users) > 1 for n in node.all_input_nodes):
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_GRAPH: aliased in-place input at {node.name}")
        elif node.op == "get_attr":
            if node.target not in buffers or buffers[node.target].layout != torch.strided:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_BUFFER: {node.name}: {node.target}")
            used_buffers.add(node.target)
        elif node.op == "call_function":
            if node.target not in FUNCTIONS:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPERATOR: {node.name}: {node.target}")
            if node.kwargs.get("inplace", False):
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPERATOR: in-place function {node.name}")
            if node.target is getattr and node.args[1] not in ("shape", "ndim"):
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPERATOR: attribute {node.args[1]}")
        elif node.op == "call_method":
            if node.target not in METHODS:
                raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPERATOR: {node.name}: {node.target}")
        elif node.op not in ("placeholder", "output"):
            raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_GRAPH: {node.name}: {node.op}")
    if used_parameters != {name for name, _ in parameters}:
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: parameters outside captured module calls")
    if missing := buffers.keys() - used_buffers:
        raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_BUFFER: outside captured graph: {sorted(missing)}")
    return graph


def partition(graph, budget=None, max_nodes=12):
    """Fuse adjacent nodes, preserving every edge crossing a group boundary.

    Parameter admission bounds grouping before shapes are known. At invocation,
    metadata execution estimates all intermediate sizes and splits oversized
    groups before any CUDA execution.
    """
    executable = [n for n in graph.graph.nodes if n.op.startswith("call_") or n.op == "get_attr"]
    chunks, chunk, size = [], [], 0
    for node in executable:
        node_size = 0
        if node.op == "call_module":
            module = graph.get_submodule(node.target)
            node_size = sum(p.numel() * p.element_size() for p in module.parameters())
        if chunk and (len(chunk) >= max_nodes or (budget and 3 * (size + node_size) > budget)):
            chunks.append(chunk)
            chunk, size = [], 0
        chunk.append(node)
        size += node_size
        if len(node.users) > 1:
            chunks.append(chunk)
            chunk, size = [], 0
    if chunk:
        chunks.append(chunk)
    return make_ir(graph, chunks)


def make_ir(graph, chunks):
    groups = []
    for chunk in chunks:
        members = set(chunk)
        inputs = tuple(dict.fromkeys(n for node in chunk for n in node.all_input_nodes if n not in members))
        outputs = tuple(n for n in chunk if any(user not in members for user in n.users))
        region = fx.Graph()
        env = {node: region.placeholder(node.name) for node in inputs}
        for node in chunk:
            env[node] = region.node_copy(node, lambda n, env=env: env[n])
        region.output(tuple(env[node] for node in outputs))
        groups.append(
            Group(
                fx.GraphModule(graph, region),
                tuple(n.name for n in inputs),
                tuple(n.name for n in outputs),
                tuple(n.name for n in chunk),
            )
        )
    consumers = {}
    for i, group in enumerate(groups):
        for name in group.inputs:
            consumers.setdefault(name, []).append(i)
    output = next(n for n in graph.graph.nodes if n.op == "output").args[0]

    def output_consumer(node):
        consumers.setdefault(node.name, []).append(len(groups))
        return node

    fx.node.map_arg(output, output_consumer)
    values = {
        n.name: GraphValue(n.name, tuple(dict.fromkeys(consumers.get(n.name, []))))
        for n in graph.graph.nodes
        if n.op != "output"
    }
    return GraphIR(
        graph, groups, values, tuple(n for n in graph.graph.nodes if n.op == "placeholder"), output
    )


@dataclass
class SavedTree:
    leaves: list
    spec: object

    @classmethod
    def save(cls, value, store, next_use):
        leaves, spec = flatten(value)
        saved = []
        try:
            for leaf in leaves:
                saved.append(
                    store.offload(leaf, expected_next_use=next_use)
                    if isinstance(leaf, torch.Tensor)
                    else leaf
                )
        except BaseException:
            for leaf in saved:
                if isinstance(leaf, TensorHandle):
                    store.free(leaf)
            raise
        return cls(saved, spec)

    def restore(self, store, device):
        return tree_unflatten(
            [store.restore(x, device=device) if isinstance(x, TensorHandle) else x for x in self.leaves],
            self.spec,
        )

    def free(self, store):
        for leaf in self.leaves:
            if isinstance(leaf, TensorHandle):
                store.free(leaf)


class _GraphAutograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, runtime, spec, trigger, *leaves):
        ctx.runtime = runtime
        ctx.consumed = False
        ctx.requires = [isinstance(x, torch.Tensor) and x.requires_grad for x in leaves]
        values = tree_unflatten(list(leaves), spec)
        result, ctx.tape = runtime.execute(values, save=True)
        result_leaves, runtime.output_spec = flatten(result)
        result_leaves = [x for x in result_leaves if isinstance(x, torch.Tensor)]
        # Only metadata is retained on ctx; retaining output tensors would keep VRAM alive.
        ctx.output_leaves = len(result_leaves)
        ctx.mark_non_differentiable(*[x for x in result_leaves if not x.is_floating_point()])
        ctx.set_materialize_grads(False)
        return tuple(result_leaves)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_outputs):
        if ctx.consumed:
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_GRAPH: one backward per forward; retain_graph unsupported"
            )
        ctx.consumed = True
        runtime = ctx.runtime
        try:
            inputs = runtime.reverse(ctx.tape, grad_outputs)
            leaves, _ = flatten(inputs)
            runtime.backward_complete = True
            return (
                None,
                None,
                None,
                *[g if needed else None for g, needed in zip(leaves, ctx.requires, strict=True)],
            )
        except torch.OutOfMemoryError as error:
            runtime.failed = True
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_WORKING_SET: CUDA allocation during graph backward"
            ) from error
        except BaseException:
            runtime.failed = True
            raise
        finally:
            runtime.release_tape(ctx.tape)


class GraphRuntime(SequentialRuntime):
    def __init__(self, model, graph, store, device, budget):
        super().__init__([], store, device, 0, budget)
        self.ir = partition(graph, budget)
        self.signature = inspect.signature(model.forward)
        self.parameter_order = tuple(name for name, _ in model.named_parameters())
        self.state_metadata = copy.deepcopy(getattr(model.state_dict(), "_metadata", {}))
        self.records = {}
        self.state_names = tuple(model.state_dict())
        self.owner = model  # Replaced by a weak reference after installation.
        if self.device.type == "cuda":
            self.store.configure_gpu(self.device, budget)
        for index, group in enumerate(self.ir.groups):
            named = dict(group.module.named_parameters())
            buffers = dict(group.module.named_buffers())
            record = _Stage(
                group.module,
                {name: store.offload(p, expected_next_use=index) for name, p in named.items()},
                {name for name, p in named.items() if p.requires_grad},
            )
            record.buffers = {name: store.offload(b, expected_next_use=index) for name, b in buffers.items()}
            self.stages.append(record)
            for name in named:
                self.records[name] = record
            # Release original leaf-module storage progressively, just as the
            # explicit adapter does; no full CPU state duplicate is retained here.
            group.module.to("meta")
        model.to("meta")
        graph.to("meta")
        for group in self.ir.groups:
            # GraphModule shares the original leaf modules. Never mutate user in-place settings.
            group.module.to("meta")
            group.module = copy.deepcopy(group.module)
            for module in group.module.modules():
                if hasattr(module, "inplace"):
                    module.inplace = False
        for group, record in zip(self.ir.groups, self.stages, strict=True):
            record.template = group.module
        self.output_spec = None
        self.output_metadata = None
        self.last_liveness = {}
        self.captured_training = model.training
        self.functional_training = any(
            n.op == "call_function" and n.target in (F.dropout, F.dropout2d) for n in graph.graph.nodes
        )

    def named_training_parameters(self, *, device="cpu"):
        for name in self.parameter_order:
            yield name, self.store.restore(self.records[name].weights[name], device=device)

    def convert_dtype(self, dtype):
        if self.pending:
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: dtype change during a pending step")
        for record in self.stages:
            for table in (record.weights, record.buffers):
                for name, handle in list(table.items()):
                    value = self.store.restore(handle, device="cpu")
                    if value.is_floating_point():
                        replacement = self.store.offload(value.to(dtype=dtype))
                        self.store.free(handle)
                        table[name] = replacement
            record.template.to(dtype=dtype)
        self.ir.frontend.to(dtype=dtype)

    def _state(self, record, buffers=None):
        return {
            **{name: self.store.restore(h, device=self.device) for name, h in record.weights.items()},
            **{
                name: self.store.restore(h, device=self.device)
                for name, h in (record.buffers if buffers is None else {**record.buffers, **buffers}).items()
            },
        }

    def _sync_modes(self):
        owner = self.owner()
        if self.functional_training and owner.training != self.captured_training:
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_GRAPH: functional dropout training mode changed after capture"
            )
        modes = {name: module.training for name, module in owner.named_modules()}
        for record in self.stages:
            for name, module in record.template.named_modules():
                if name in modes:
                    module.training = modes[name]

    def _admit(self, group, record, args):
        class Sizes(fx.Interpreter):
            def __init__(self, module):
                super().__init__(module)
                self.total = 0
                self.sizes = {}

            def run_node(self, node):
                result = super().run_node(node)
                leaves, _ = flatten(result)
                self.sizes[node.name] = sum(
                    x.numel() * x.element_size() for x in leaves if isinstance(x, torch.Tensor)
                )
                # Placeholders and the region output alias tensors that are already
                # represented by operator results. Counting them again makes a
                # single-node region appear to need roughly three times its real
                # activation storage and can reject otherwise admissible operators.
                if node.op.startswith("call_") or node.op == "get_attr":
                    self.total += self.sizes[node.name]
                return result

        meta = map_tree(lambda x: torch.empty_like(x, device="meta"), args)
        interpreter = Sizes(copy.deepcopy(record.template))
        try:
            meta_output = interpreter.run(*meta)
        except Exception as error:
            raise TrainPoolError(
                f"TRAINPOOL_UNSUPPORTED_WORKING_SET: metadata at {group.nodes}: {error}"
            ) from error
        nodes = list(interpreter.module.graph.nodes)
        positions = {node: index for index, node in enumerate(nodes)}
        last_use = {
            node.name: max((positions[user] for user in node.users), default=positions[node])
            for node in nodes
        }
        live_sizes = {}
        activation_peak = 0
        for index, node in enumerate(nodes):
            if node.op != "output":
                live_sizes[node.name] = interpreter.sizes.get(node.name, 0)
            activation_peak = max(activation_peak, sum(live_sizes.values()))
            for name in tuple(live_sizes):
                if last_use[name] <= index:
                    live_sizes.pop(name)

        parameter_bytes = sum(h.size for h in record.weights.values())
        buffer_bytes = sum(h.size for h in record.buffers.values())
        largest_parameter = max((h.size for h in record.weights.values()), default=0)
        largest_value = max(interpreter.sizes.values(), default=0)
        # Each phase is modeled independently because TrainPool restores and
        # publishes state one group at a time. Backward includes recomputation
        # plus gradients; optimizer state is processed parameter by parameter.
        group.forward_peak = parameter_bytes + buffer_bytes + activation_peak
        group.backward_peak = 2 * parameter_bytes + buffer_bytes + 2 * activation_peak
        group.optimizer_peak = 5 * largest_parameter
        group.workspace_margin = max(
            largest_value // 2,
            (parameter_bytes + buffer_bytes + activation_peak) // 10,
        )
        required = max(group.forward_peak, group.backward_peak, group.optimizer_peak)
        required += group.workspace_margin
        group.estimated_working_set = required
        for name, size in interpreter.sizes.items():
            if name in self.ir.values:
                self.ir.values[name].size = size
        return meta_output

    def _check_budget(self, group):
        required = group.estimated_working_set
        available = self.gpu_budget_bytes
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            if required > free:
                # Cached inactive allocator blocks can be returned without evicting
                # any live graph value. Recheck physical free bytes afterwards.
                torch.cuda.empty_cache()
                free, _ = torch.cuda.mem_get_info(self.device)
            evictable = getattr(self.store, "_resident_bytes", 0)
            available = min(available or free + evictable, free + evictable)
        if available and required > available:
            raise TrainPoolError(
                f"TRAINPOOL_UNSUPPORTED_WORKING_SET: {group.nodes}: phase peak {required} > {available}"
            )
        if self.device.type == "cuda":
            self.store.ensure_cuda_capacity(required)

    def _repartition(self, chunks):
        weights, buffers, gradients, states, trainable = {}, {}, {}, {}, set()
        for record in self.stages:
            weights.update(record.weights)
            buffers.update(record.buffers)
            gradients.update(record.gradients)
            states.update(record.optimizer_state)
            trainable.update(record.trainable)
        self.ir = make_ir(self.ir.frontend, chunks)
        self.stages, self.records = [], {}
        for group in self.ir.groups:
            group.module.to("meta")
            group.module = copy.deepcopy(group.module)
            for module in group.module.modules():
                if hasattr(module, "inplace"):
                    module.inplace = False
            names = dict(group.module.named_parameters())
            record = _Stage(
                group.module,
                {name: weights[name] for name in names},
                set(names) & trainable,
                {name: gradients[name] for name in names if name in gradients},
                {name: states[name] for name in names if name in states},
            )
            record.buffers = {name: buffers[name] for name, _ in group.module.named_buffers()}
            self.stages.append(record)
            for name in names:
                self.records[name] = record
        self._sync_modes()

    def _preflight(self, values):
        # Pure metadata only: complete admission before loading even the first CUDA input.
        while True:
            env = {
                node.name: map_tree(lambda x: torch.empty_like(x, device="meta"), value)
                for node, value in zip(self.ir.placeholders, values, strict=True)
            }
            retry = False
            for index, (group, record) in enumerate(zip(self.ir.groups, self.stages, strict=True)):
                result = self._admit(group, record, tuple(env[name] for name in group.inputs))
                if self.gpu_budget_bytes and group.estimated_working_set > self.gpu_budget_bytes:
                    if len(group.nodes) == 1:
                        self._check_budget(group)
                    nodes = {node.name: node for node in self.ir.frontend.graph.nodes}
                    chunks = [[nodes[name] for name in g.nodes] for g in self.ir.groups]
                    cut = len(chunks[index]) // 2
                    chunks[index : index + 1] = [chunks[index][:cut], chunks[index][cut:]]
                    self._repartition(chunks)
                    retry = True
                    break
                for name, value in zip(group.outputs, result, strict=True):
                    env[name] = value
            if not retry:
                flatten(fx.node.map_arg(self.ir.output, lambda node, env=env: env[node.name]))
                return

    def forward(self, *args, training=True, **kwargs):
        if self.failed or self.closed:
            raise TrainPoolError("TRAINPOOL_JOB_FAILED: create a new prepared job")
        if self.pending:
            raise TrainPoolError("Complete backward and optimizer.step/zero_grad before the next forward")
        if torch.is_autocast_enabled() or torch.is_autocast_enabled("cpu"):
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_GRAPH: autocast is not supported")
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = tuple(bound.arguments[str(n.target)] for n in self.ir.placeholders)
        leaves, spec = flatten(values)
        for tensor in leaves:
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                raise TrainPoolError(f"TRAINPOOL_NO_CUDA_FALLBACK: input must be on {self.device}")
        self._sync_modes()
        self._preflight(values)
        if not training or not torch.is_grad_enabled():
            with torch.no_grad():
                result, tape = self.execute(values, save=False)
            self.release_tape(tape)
            return result
        self.pending, self.backward_complete = True, False
        try:
            outputs = _GraphAutograd.apply(
                self, spec, torch.empty((), device=self.device, requires_grad=True), *leaves
            )
            iterator = iter(outputs)
            return tree_unflatten(
                [next(iterator) if tensor else value for tensor, value in self.output_metadata],
                self.output_spec,
            )
        except torch.OutOfMemoryError as error:
            self.pending, self.failed = False, True
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_WORKING_SET: CUDA allocation during graph forward"
            ) from error
        except BaseException:
            self.pending, self.failed = False, True
            raise

    def execute(self, inputs, save):
        tape = {"values": {}, "buffers": [], "rng": [], "output": None}
        live = copy.deepcopy(self.ir.values)
        for value in live.values():
            value.remaining_consumers = len(value.consumers)
            value.next_use = value.consumers[0] if value.consumers else None
        try:
            for node, value in zip(self.ir.placeholders, inputs, strict=True):
                tape["values"][node.name] = SavedTree.save(value, self.store, "forward")
            for i, (group, record) in enumerate(zip(self.ir.groups, self.stages, strict=True)):
                access = i
                if hasattr(self.store, "set_access_index"):
                    access = self.store.set_access_index(i)
                if hasattr(self.store, "update_priority"):
                    for handle in (*record.weights.values(), *record.buffers.values()):
                        self.store.update_priority(handle, next_use=access, remaining_consumers=1)
                self._check_budget(group)
                if hasattr(self.store, "update_priority"):
                    for name in group.inputs:
                        for handle in tape["values"][name].leaves:
                            if isinstance(handle, TensorHandle):
                                self.store.update_priority(
                                    handle,
                                    next_use=(
                                        None
                                        if live[name].next_use is None
                                        else access + max(1, live[name].next_use - i)
                                    ),
                                    remaining_consumers=live[name].remaining_consumers,
                                )
                arguments = tuple(
                    tape["values"][name].restore(self.store, self.device) for name in group.inputs
                )
                snapshot = {}
                tape["buffers"].append(snapshot)
                mutable = {
                    f"{path}.{name}" if path else name
                    for path, module in record.template.named_modules()
                    if isinstance(module, nn.modules.batchnorm._BatchNorm)
                    for name, _ in module.named_buffers(recurse=False)
                }
                if save:
                    for name, handle in record.buffers.items():
                        if name not in mutable:
                            continue
                        snapshot[name] = self.store.offload(self.store.restore(handle, device=self.device))
                state = self._state(record)
                tape["rng"].append(
                    (
                        torch.random.get_rng_state(),
                        torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                    )
                )
                try:
                    result = torch.func.functional_call(record.template, state, arguments)
                except torch.OutOfMemoryError as error:
                    raise TrainPoolError(
                        f"TRAINPOOL_UNSUPPORTED_WORKING_SET: CUDA workspace at {group.nodes}"
                    ) from error
                for name in mutable:
                    replacement = self.store.offload(state[name], expected_next_use="next forward")
                    self.store.free(record.buffers[name])
                    record.buffers[name] = replacement
                if hasattr(self.store, "update_priority"):
                    for handle in (*record.weights.values(), *record.buffers.values()):
                        self.store.update_priority(
                            handle,
                            next_use=access + len(self.ir.groups),
                            remaining_consumers=1,
                        )
                for name, value in zip(group.outputs, result, strict=True):
                    next_use = (
                        None if live[name].next_use is None else access + max(1, live[name].next_use - i)
                    )
                    saved = SavedTree.save(value, self.store, next_use)
                    tape["values"][name] = saved
                    live[name].size = sum(h.size for h in saved.leaves if isinstance(h, TensorHandle))
                    handles = [h for h in saved.leaves if isinstance(h, TensorHandle)]
                    if any(h.resident is not None for h in handles):
                        live[name].residency = Residency.GPU_RESIDENT
                    elif any(
                        block["owner_node"] != self.store.client.local_node
                        for handle in handles
                        for block in handle.blocks
                    ):
                        live[name].residency = Residency.REMOTE_RAM
                    else:
                        live[name].residency = Residency.LOCAL_RAM
                for name in group.inputs:
                    live[name].remaining_consumers -= 1
                    live[name].next_use = next((j for j in live[name].consumers if j > i), None)
                    if not save and live[name].remaining_consumers == 0:
                        tape["values"].pop(name).free(self.store)
                del arguments, state, result
                if group.outputs:
                    del value
            result = fx.node.map_arg(
                self.ir.output, lambda n: tape["values"][n.name].restore(self.store, self.device)
            )
            leaves, self.output_spec = flatten(result)
            self.output_metadata = [
                (isinstance(x, torch.Tensor), None if isinstance(x, torch.Tensor) else x) for x in leaves
            ]
            tape["output"] = fx.node.map_arg(self.ir.output, lambda n: n.name)
            self.last_liveness = live
            return result, tape
        except BaseException:
            self.release_tape(tape)
            raise

    def _reverse_group(self, index, tape, gradients, accumulate):
        # A separate frame bounds tensor references (including autograd edges) to
        # this group. No previous group's local variables survive the next load.
        group, record = self.ir.groups[index], self.stages[index]
        access = len(self.ir.groups) + (len(self.ir.groups) - index)
        if hasattr(self.store, "set_access_index"):
            access = self.store.set_access_index(access)
        if hasattr(self.store, "update_priority"):
            for handle in (*record.weights.values(), *record.buffers.values()):
                self.store.update_priority(
                    handle,
                    next_use=access,
                    remaining_consumers=1,
                )
        args = tuple(tape["values"][name].restore(self.store, self.device) for name in group.inputs)
        args = map_tree(lambda x: x.detach().requires_grad_(x.is_floating_point()), args)
        state = self._state(record, tape["buffers"][index])
        names = [name for name in record.weights if name in record.trainable]
        for name in names:
            state[name].requires_grad_(True)
        arg_leaves, arg_spec = flatten(args)
        tensors = [x for x in arg_leaves if isinstance(x, torch.Tensor) and x.requires_grad]
        devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.random.set_rng_state(tape["rng"][index][0])
            if devices:
                torch.cuda.set_rng_state(tape["rng"][index][1], self.device)
            result = torch.func.functional_call(record.template, state, args)
            outputs, upstream = [], []
            for name, value in zip(group.outputs, result, strict=True):
                saved = gradients.pop(name, None)
                if saved is None:
                    continue
                grad = saved.restore(self.store, self.device)
                saved.free(self.store)
                out_leaves, _ = flatten(value)
                grad_leaves, _ = flatten(grad)
                for output, g in zip(out_leaves, grad_leaves, strict=True):
                    if isinstance(output, torch.Tensor) and output.requires_grad and g is not None:
                        outputs.append(output)
                        upstream.append(g)
            targets = tensors + [state[name] for name in names]
            derivatives = (
                torch.autograd.grad(outputs, targets, upstream, allow_unused=True)
                if outputs and targets
                else [None] * len(targets)
            )
        it = iter(derivatives[: len(tensors)])
        input_grads = tree_unflatten(
            [next(it) if isinstance(x, torch.Tensor) and x.requires_grad else None for x in arg_leaves],
            arg_spec,
        )
        for name, grad in zip(group.inputs, input_grads, strict=True):
            accumulate(name, grad)
        for name, grad in zip(names, derivatives[len(tensors) :], strict=True):
            if grad is not None:
                record.gradients[name] = self.store.offload(grad, expected_next_use="optimizer.step")
        # Backward has consumed these group boundary snapshots permanently.
        for name in group.outputs:
            tape["values"].pop(name).free(self.store)

    def reverse(self, tape, grad_outputs):
        gradients = {}

        def accumulate(name, values):
            previous = gradients.pop(name, None)
            if previous:
                old = previous.restore(self.store, self.device)
                a, spec = flatten(old)
                b, _ = flatten(values)
                values = tree_unflatten(
                    [y if x is None else x if y is None else x + y for x, y in zip(a, b, strict=True)], spec
                )
                previous.free(self.store)
            gradients[name] = SavedTree.save(values, self.store, "backward")

        # Traverse graph output structure with corresponding public output gradients.
        iterator = iter(grad_outputs)

        def seed(node):
            saved = tape["values"][node.name]
            values = tree_unflatten(
                [next(iterator) if isinstance(x, TensorHandle) else None for x in saved.leaves], saved.spec
            )
            accumulate(node.name, values)
            return node

        fx.node.map_arg(self.ir.output, seed)
        try:
            for index in reversed(range(len(self.ir.groups))):
                group = self.ir.groups[index]
                if not any(name in gradients for name in group.outputs):
                    continue
                self._check_budget(group)
                self._reverse_group(index, tape, gradients, accumulate)
            results = tuple(
                gradients[n.name].restore(self.store, self.device)
                if n.name in gradients
                else tree_unflatten([None] * len(tape["values"][n.name].leaves), tape["values"][n.name].spec)
                for n in self.ir.placeholders
            )
            self.store.flush_metrics()
            return results
        finally:
            for saved in gradients.values():
                saved.free(self.store)

    def release_tape(self, tape):
        for saved in tape["values"].values():
            with contextlib.suppress(Exception):
                saved.free(self.store)
        for buffers in tape["buffers"]:
            for handle in buffers.values():
                with contextlib.suppress(Exception):
                    self.store.free(handle)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = OrderedDict()
            destination._metadata = copy.deepcopy(self.state_metadata)
        state = {}
        for record in self.stages:
            state.update(record.weights)
            state.update(record.buffers)
        for name in self.state_names:
            value = self.store.restore(state[name], device="cpu")
            destination[prefix + name] = value if keep_vars else value.detach()
        return destination

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.pending or assign:
            raise TrainPoolError("TRAINPOOL_CHECKPOINT_UNSUPPORTED: pending step or assign=True")
        expected = set(self.state_names)
        missing, unexpected = sorted(expected - state_dict.keys()), sorted(state_dict.keys() - expected)
        if strict and (missing or unexpected):
            raise RuntimeError(f"state_dict missing keys: {missing}; unexpected keys: {unexpected}")
        replacements = []
        try:
            for record in self.stages:
                for table in (record.weights, record.buffers):
                    for name, handle in table.items():
                        if name not in state_dict:
                            continue
                        value = state_dict[name]
                        if not isinstance(value, torch.Tensor) or tuple(value.shape) != handle.shape:
                            raise RuntimeError(f"state_dict size mismatch: {name}")
                        replacement = self.store.offload(
                            value.to(device="cpu", dtype=getattr(torch, handle.dtype))
                        )
                        replacements.append((table, name, replacement))
        except BaseException:
            for _, _, handle in replacements:
                self.store.free(handle)
            raise
        try:
            for table, name, handle in replacements:
                old = table[name]
                table[name] = handle
                self.store.free(old)
        except BaseException:
            self.failed = True
            raise
        return nn.modules.module._IncompatibleKeys(missing, unexpected)


def prepare_graph_inplace(model, *, device, store, prefetch_depth=0, gpu_budget_bytes=None, graph=None):
    import weakref

    runtime = GraphRuntime(
        model, graph if graph is not None else capture(model), store, device, gpu_budget_bytes
    )
    runtime.owner = weakref.ref(model)

    def forward(owner, *args, **kwargs):
        return runtime.forward(*args, training=owner.training, **kwargs)

    def state_dict(owner, *args, **kwargs):
        return runtime.state_dict(*args, **kwargs)

    def load_state_dict(owner, *args, **kwargs):
        return runtime.load_state_dict(*args, **kwargs)

    object.__setattr__(model, "_trainpool_runtime", runtime)
    object.__setattr__(model, "forward", types.MethodType(forward, model))
    object.__setattr__(model, "state_dict", types.MethodType(state_dict, model))
    object.__setattr__(model, "load_state_dict", types.MethodType(load_state_dict, model))
    object.__setattr__(model, "close", runtime.close)
    return runtime
