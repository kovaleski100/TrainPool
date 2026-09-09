"""Process-local transparent PyTorch activation used by TrainPool's launcher.

Importing this module is lightweight. PyTorch is imported only by ``autoinstall``
after the launcher has explicitly set ``TRAINPOOL_ACTIVE=1``.
"""

from __future__ import annotations

import atexit
import contextlib
import os
import sys
import threading
import weakref
from dataclasses import dataclass
from enum import Enum

from .client import Client, TrainPoolError


class Compatibility(str, Enum):
    FULL = "FULL"
    PARTIAL = "PARTIAL"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass
class _ModelRecord:
    runtime: object
    original_parameter_ids: frozenset
    original_parameter_order: tuple
    current_parameter_ids: frozenset
    placeholder_parameters: tuple
    requested_device: object
    compatibility: Compatibility = Compatibility.FULL


_installed = False
_lock = threading.RLock()
_movement = threading.local()
_models = weakref.WeakKeyDictionary()
_optimizers = weakref.WeakKeyDictionary()
_runtimes = weakref.WeakSet()


@contextlib.contextmanager
def _internal_movement():
    previous = getattr(_movement, "active", False)
    _movement.active = True
    try:
        yield
    finally:
        _movement.active = previous


def _optimizer_parameter_ids(optimizer):
    return frozenset(id(parameter) for group in optimizer.param_groups for parameter in group["params"])


def _matching_record(parameter_ids):
    for model, record in list(_models.items()):
        if parameter_ids and parameter_ids in (
            record.original_parameter_ids,
            record.current_parameter_ids,
        ):
            return model, record
    return None, None


def _overlapping_record(parameter_ids):
    for model, record in list(_models.items()):
        if parameter_ids & (record.original_parameter_ids | record.current_parameter_ids):
            return model, record
    return None, None


def _attach_optimizer(optimizer):
    from .sequential import SUPPORTED_OPTIMIZERS, attach_optimizer_inplace

    parameter_ids = _optimizer_parameter_ids(optimizer)
    _optimizers[optimizer] = parameter_ids
    _, record = _matching_record(parameter_ids)
    if record is None:
        _, overlap = _overlapping_record(parameter_ids)
        if overlap is not None:
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_OPTIMIZER: parameters must match the complete model")
        return
    if type(optimizer) not in SUPPORTED_OPTIMIZERS:
        raise TrainPoolError(
            f"TRAINPOOL_UNSUPPORTED_OPTIMIZER: {type(optimizer).__module__}.{type(optimizer).__name__}; "
            "FULL mode supports torch.optim.SGD, Adam and AdamW"
        )
    if len(optimizer.param_groups) != 1:
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_OPTIMIZER: multiple parameter groups")
    if not hasattr(optimizer, "_trainpool_delegate"):
        order = tuple(id(parameter) for parameter in optimizer.param_groups[0]["params"])
        if order not in (
            record.original_parameter_order,
            tuple(id(p) for p in record.placeholder_parameters),
        ):
            raise TrainPoolError(
                "TRAINPOOL_UNSUPPORTED_OPTIMIZER: parameter order must match model.parameters()"
            )
        if len(optimizer.param_groups[0]["params"]) != len(record.placeholder_parameters):
            raise TrainPoolError("TRAINPOOL_UNSUPPORTED_OPTIMIZER: parameters must match the complete model")
        optimizer.param_groups[0]["params"][:] = record.placeholder_parameters
        _optimizers[optimizer] = record.current_parameter_ids
        try:
            attach_optimizer_inplace(optimizer, record.runtime)
        except ValueError as error:
            raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPTIMIZER: {error}") from error


def _requested_cuda(torch, args, kwargs):
    device = kwargs.get("device")
    if device is None and args:
        candidate = args[0]
        if isinstance(candidate, torch.Tensor):
            device = candidate.device
        elif isinstance(candidate, (str, torch.device)):
            device = candidate
    if device is None:
        return None
    try:
        parsed = torch.device(device)
    except (TypeError, RuntimeError):
        return None
    return parsed if parsed.type == "cuda" else None


def _requested_dtype(torch, args, kwargs):
    dtype = kwargs.get("dtype")
    if dtype is None and args:
        candidate = args[0]
        if isinstance(candidate, torch.Tensor):
            dtype = candidate.dtype
        elif isinstance(candidate, torch.dtype):
            dtype = candidate
    return dtype


def _activate_model(model, requested_device):
    import torch

    from .graph import capture, prepare_graph_inplace
    from .sequential import SUPPORTED_OPTIMIZERS
    from .store import TensorStore

    if model in _models:
        return model
    original_parameter_order = tuple(id(parameter) for parameter in model.parameters())
    original_parameter_ids = frozenset(original_parameter_order)
    related = [
        optimizer
        for optimizer, parameter_ids in list(_optimizers.items())
        if parameter_ids & original_parameter_ids
    ]
    if any(_optimizers[optimizer] != original_parameter_ids for optimizer in related):
        raise TrainPoolError("TRAINPOOL_UNSUPPORTED_OPTIMIZER: parameters must match the complete model")
    matching = related
    for optimizer in matching:
        if type(optimizer) not in SUPPORTED_OPTIMIZERS:
            raise TrainPoolError(
                f"TRAINPOOL_UNSUPPORTED_OPTIMIZER: {type(optimizer).__module__}.{type(optimizer).__name__}"
            )
    try:
        graph = capture(model)
    except (TypeError, ValueError) as error:
        framework = " framework=ultralytics" if "ultralytics" in type(model).__module__ else ""
        raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_GRAPH:{framework} {error}") from error

    test_cpu = os.getenv("TRAINPOOL_TEST_CPU") == "1"
    actual_device = torch.device("cpu") if test_cpu else requested_device
    if not test_cpu and not torch.cuda.is_available():
        raise TrainPoolError("TRAINPOOL_NO_CUDA: CUDA was requested but is unavailable")
    store = TensorStore(preferred_node=None)
    try:
        if not test_cpu:
            if (
                store.plan["strategy"] != "SingleGpuDistributedMemory"
                or store.client.local_node not in store.plan["compute_nodes"]
            ):
                raise TrainPoolError(
                    "TRAINPOOL_NO_CUDA: transparent FULL mode must run on the selected primary GPU node"
                )
        if not test_cpu:
            assignment = store.plan["gpu_assignments"][0]
            eligible = [
                i
                for i in range(torch.cuda.device_count())
                if str(torch.cuda.get_device_properties(i).uuid).removeprefix("GPU-").lower()
                == assignment["gpu_id"].removeprefix("GPU-").lower()
            ]
            if not eligible:
                raise TrainPoolError("TRAINPOOL_NO_CUDA: selected primary GPU is not visible to PyTorch")
            index = eligible[0]
            if requested_device.index is not None and requested_device.index != index:
                raise TrainPoolError("TRAINPOOL_NO_CUDA: requested device differs from selected primary GPU")
            torch.cuda.set_device(index)
            actual_device = torch.device("cuda", index)
        budget = None if test_cpu else store.plan["gpu_assignments"][0]["usable_bytes"]
        with _internal_movement():
            runtime = prepare_graph_inplace(
                model,
                graph=graph,
                device=actual_device,
                store=store,
                prefetch_depth=1,
                gpu_budget_bytes=budget,
            )
    except BaseException:
        store.close()
        raise
    runtime.requested_device = requested_device
    runtime.compatibility = Compatibility.FULL
    runtime.instrumentation_guard = _internal_movement
    placeholder_parameters = tuple(model.parameters())
    current_parameter_ids = frozenset(id(parameter) for parameter in placeholder_parameters)
    record = _ModelRecord(
        runtime,
        original_parameter_ids,
        original_parameter_order,
        current_parameter_ids,
        placeholder_parameters,
        requested_device,
    )
    _models[model] = record
    _runtimes.add(runtime)
    weakref.finalize(model, runtime.close)
    for optimizer in matching:
        _attach_optimizer(optimizer)
    print(
        f"[TrainPool] compatibility=FULL backend={'test-cpu' if test_cpu else requested_device} "
        "parameters=remote activations=recomputed gradients=remote optimizer_state=remote",
        file=sys.stderr,
    )
    return model


def _cleanup():
    for runtime in list(_runtimes):
        with contextlib.suppress(Exception):
            runtime.close()
    _runtimes.clear()


def autoinstall():
    """Install transparent instrumentation once for a launcher-activated process."""
    global _installed
    if os.getenv("TRAINPOOL_ACTIVE") != "1":
        return False
    with _lock:
        if _installed:
            return True

        job_id = os.getenv("TRAINPOOL_JOB_ID")
        if not job_id:
            raise TrainPoolError("TRAINPOOL_BOOTSTRAP_ERROR: TRAINPOOL_JOB_ID is missing")
        client = Client()
        client.control("job_status", job_id=job_id)

        import torch

        original_to = torch.nn.Module.to
        original_cuda = torch.nn.Module.cuda
        original_optimizer_init = torch.optim.Optimizer.__init__

        def module_to(module, *args, **kwargs):
            if getattr(_movement, "active", False):
                return original_to(module, *args, **kwargs)
            requested = _requested_cuda(torch, args, kwargs)
            if requested is None:
                record = _models.get(module)
                dtype = _requested_dtype(torch, args, kwargs)
                if record is not None and dtype is not None:
                    with _internal_movement():
                        record.runtime.convert_dtype(dtype)
                    return module
                return original_to(module, *args, **kwargs)
            return _activate_model(module, requested)

        def module_cuda(module, device=None):
            if getattr(_movement, "active", False):
                return original_cuda(module, device=device)
            if device is None:
                requested = torch.device("cuda")
            elif isinstance(device, (str, torch.device)):
                requested = torch.device(device)
            else:
                requested = torch.device(f"cuda:{device}")
            return _activate_model(module, requested)

        def optimizer_init(optimizer, params, defaults):
            original_optimizer_init(optimizer, params, defaults)
            if getattr(_movement, "active", False):
                return
            _optimizers[optimizer] = _optimizer_parameter_ids(optimizer)
            _, record = _matching_record(_optimizers[optimizer])
            if record is None:
                _, overlap = _overlapping_record(_optimizers[optimizer])
                if overlap is not None:
                    raise TrainPoolError(
                        "TRAINPOOL_UNSUPPORTED_OPTIMIZER: parameters must match the complete model"
                    )
            if record is not None:
                from .sequential import SUPPORTED_OPTIMIZERS

                if type(optimizer) not in SUPPORTED_OPTIMIZERS:
                    raise TrainPoolError(f"TRAINPOOL_UNSUPPORTED_OPTIMIZER: {type(optimizer).__name__}")

        torch.nn.Module.to = module_to
        torch.nn.Module.cuda = module_cuda
        torch.optim.Optimizer.__init__ = optimizer_init

        for optimizer_type in (torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW):
            original_init = optimizer_type.__init__

            def supported_init(
                optimizer,
                *args,
                __original=original_init,
                __type=optimizer_type,
                **kwargs,
            ):
                __original(optimizer, *args, **kwargs)
                if type(optimizer) is __type and not getattr(_movement, "active", False):
                    _attach_optimizer(optimizer)

            optimizer_type.__init__ = supported_init

        atexit.register(_cleanup)
        _installed = True
        return True


__all__ = ["Compatibility", "autoinstall"]
