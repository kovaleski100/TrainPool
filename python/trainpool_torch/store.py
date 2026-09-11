"""Tensor table and a single-worker, bounded look-ahead prefetch policy."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import enum
import math
import os
import threading
import time
import uuid
import warnings

from .client import Client, TrainPoolError


class MemoryTier(str, enum.Enum):
    GPU = "gpu"
    LOCAL_RAM = "local_ram"
    REMOTE_RAM = "remote_ram"


@dataclasses.dataclass(frozen=True)
class ResidencyCandidate:
    id: str
    size: int
    next_use: int | None
    remaining_consumers: int = 1
    transfer_cost: float = 1.0
    reuse_count: int = 0


@dataclasses.dataclass(frozen=True)
class CudaMemorySnapshot:
    physical_total: int
    driver_free: int
    driver_used: int
    torch_allocated: int
    torch_reserved: int
    torch_reclaimable: int
    trainpool_resident: int
    configured_usable_ceiling: int
    configured_safety_reserve: int
    physical_allocator_headroom: int
    budget_headroom: int
    safe_allocatable_now: int

    def diagnostic(self):
        return (
            f"driver_free={self.driver_free} torch_allocated={self.torch_allocated} "
            f"torch_reserved={self.torch_reserved} reclaimable_cache={self.torch_reclaimable} "
            f"trainpool_resident={self.trainpool_resident} "
            f"safe_allocatable={self.safe_allocatable_now}"
        )


class TieredResidencyPolicy:
    """Deterministic policy shared by the real store and simulation tests.

    Capacity is filled strictly in GPU, compute-node RAM, then remote RAM order.
    A larger eviction key means that the value is cheaper to remove from VRAM.
    """

    def __init__(self, vram_capacity: int, local_ram_capacity: int):
        self.vram_capacity = max(0, vram_capacity)
        self.local_ram_capacity = max(0, local_ram_capacity)

    def distribution(self, size: int):
        gpu = min(size, self.vram_capacity)
        local = min(size - gpu, self.local_ram_capacity)
        return {
            MemoryTier.GPU: gpu,
            MemoryTier.LOCAL_RAM: local,
            MemoryTier.REMOTE_RAM: size - gpu - local,
        }

    @staticmethod
    def eviction_key(candidate: ResidencyCandidate, current_step: int):
        distance = math.inf if candidate.next_use is None else max(0, candidate.next_use - current_step)
        # Expensive transfers and frequently reused values are more valuable in
        # VRAM. Size and next-use distance make a large, cold value preferable.
        value = max(1.0, candidate.transfer_cost) * (1 + candidate.reuse_count)
        return (
            distance,
            candidate.size / value,
            -candidate.remaining_consumers,
            candidate.id,
        )

    def select_evictions(self, candidates, bytes_needed: int, current_step: int):
        selected = []
        released = 0
        for candidate in sorted(
            candidates,
            key=lambda item: self.eviction_key(item, current_step),
            reverse=True,
        ):
            selected.append(candidate)
            released += candidate.size
            if released >= bytes_needed:
                break
        return selected if released >= bytes_needed else []


@dataclasses.dataclass
class TensorHandle:
    id: str
    shape: tuple
    dtype: str
    size: int
    device: str
    blocks: list
    layout: str = "contiguous"
    expected_next_use: str | int | None = None
    dirty: bool = False
    resident: object | None = dataclasses.field(default=None, repr=False, compare=False)
    remaining_consumers: int = 1
    reuse_count: int = 0
    access_index: int = 0
    transfer_cost: float = 1.0

    @property
    def current_location(self):
        if self.resident is not None:
            return [MemoryTier.GPU.value]
        return [b["location_type"] for b in self.blocks]


class TensorStore:
    def __init__(self, client=None, *, preferred_node=None, job_id=None, block_bytes=None):
        self.client = client or Client()
        self.preferred_node = preferred_node
        job_id = job_id or os.getenv("TRAINPOOL_JOB_ID")
        self.plan = (
            self.client.control("plan", stages=[])
            if job_id is None
            else self.client.control("job_status", job_id=job_id)["plan"]
        )
        self.job_id = self.plan["job_id"]
        # Blocks and staging are independent: large blocks avoid metadata churn,
        # while at most 64 KiB of host payload is staged for each transfer.
        self.block_bytes = min(block_bytes or 4 * 1024 * 1024, self.client.chunk_bytes)
        if self.block_bytes < 4096:
            raise ValueError("block_bytes must be at least 4096")
        self.tensors = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._lease_error = None
        self._gpu_device = None
        self._gpu_budget_bytes = 0
        self._gpu_safety_reserve_bytes = 0
        self._resident_bytes = 0
        self._access_index = 0
        self._last_metrics_flush = 0.0
        self._metrics_interval_seconds = 0.250
        local_budget = self.plan.get("memory_budgets", {}).get(self.client.local_node, 0)
        self.policy = TieredResidencyPolicy(0, local_budget)
        self.metrics = {
            "bytes_local_ram_to_gpu": 0,
            "bytes_gpu_to_local_ram": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "gpu_wait_for_data_ms": 0.0,
            "peak_gpu_residency": 0,
            "peak_vram_resident_bytes": 0,
            "gpu_to_local_bytes": 0,
            "local_to_gpu_bytes": 0,
            "gpu_to_remote_bytes": 0,
            "remote_to_gpu_bytes": 0,
            "eviction_count": 0,
            "prefetch_count": 0,
            "peak_local_ram_backing_bytes": 0,
            "peak_remote_ram_backing_bytes": 0,
        }
        self._renewal = threading.Thread(target=self._renew_loop, name="trainpool-leases", daemon=True)
        self._renewal.start()

    def _require_cuda(self, device):
        if device.type == "cuda":
            if self.client.local_node not in self.plan["compute_nodes"]:
                raise TrainPoolError("TRAINPOOL_NO_CUDA: local node is not a CUDA compute provider")
            import torch

            index = device.index if device.index is not None else torch.cuda.current_device()
            selected = self.plan["gpu_assignments"][0]["gpu_id"].removeprefix("GPU-").lower()
            actual = str(torch.cuda.get_device_properties(index).uuid).removeprefix("GPU-").lower()
            if actual != selected:
                raise TrainPoolError("TRAINPOOL_NO_CUDA: tensor device is not the selected primary GPU")
            self._gpu_device = torch.device("cuda", index)

    def configure_gpu(self, device, budget_bytes):
        """Enable the VRAM residency tier before initial model ownership moves."""
        import torch

        target = torch.device(device)
        self._require_cuda(target)
        self._gpu_budget_bytes = max(0, int(budget_bytes or 0))
        assignment = next(
            assignment
            for assignment in self.plan["gpu_assignments"]
            if assignment["node_id"] == self.client.local_node
        )
        self._gpu_safety_reserve_bytes = int(assignment.get("safety_reserve_bytes", 0))
        self.policy = TieredResidencyPolicy(
            self._gpu_budget_bytes,
            self.policy.local_ram_capacity,
        )

    def set_access_index(self, index):
        index = int(index)
        self._access_index = self._access_index + 1 if index <= self._access_index else index
        return self._access_index

    def update_priority(self, handle, *, next_use=None, remaining_consumers=None):
        with self._lock:
            if next_use is not None or handle.expected_next_use is not None:
                handle.expected_next_use = next_use
            if remaining_consumers is not None:
                handle.remaining_consumers = remaining_consumers

    def _next_use_index(self, value):
        if isinstance(value, int):
            return value
        if value is None:
            return None
        # String hints used by the explicit Sequential API are ordered phases,
        # not exact GraphIR positions. Keep them near unless explicitly "next".
        return self._access_index + (2 if "next" in value else 1)

    def _candidate(self, handle):
        return ResidencyCandidate(
            handle.id,
            handle.size,
            self._next_use_index(handle.expected_next_use),
            handle.remaining_consumers,
            handle.transfer_cost,
            handle.reuse_count,
        )

    def _effective_vram_capacity(self):
        if self._gpu_device is None or not self._gpu_budget_bytes:
            return 0
        snapshot = self.cuda_memory_snapshot()
        # Existing TrainPool residents are a subset of torch_allocated. Adding
        # only safe *new* allocator headroom yields a maximum resident ceiling
        # without counting cached or resident bytes twice.
        return min(
            self._gpu_budget_bytes,
            self._resident_bytes + snapshot.safe_allocatable_now,
        )

    def cuda_memory_snapshot(self):
        """Return the authoritative, process-local CUDA allocator snapshot.

        Driver free bytes exclude PyTorch's reserved cache. Reclaimable cache is
        therefore added only to physical headroom, while budget headroom is
        derived from total PyTorch allocation. TrainPool resident handles are a
        subset of torch_allocated and are never added to either side.
        """
        if self._gpu_device is None:
            return None
        import torch

        driver_free, physical_total = torch.cuda.mem_get_info(self._gpu_device)
        allocated = torch.cuda.memory_allocated(self._gpu_device)
        reserved = torch.cuda.memory_reserved(self._gpu_device)
        reclaimable = max(0, reserved - allocated)
        physical_headroom = driver_free + reclaimable
        budget_headroom = max(0, self._gpu_budget_bytes - allocated)
        return CudaMemorySnapshot(
            physical_total=physical_total,
            driver_free=driver_free,
            driver_used=physical_total - driver_free,
            torch_allocated=allocated,
            torch_reserved=reserved,
            torch_reclaimable=reclaimable,
            trainpool_resident=self._resident_bytes,
            configured_usable_ceiling=self._gpu_budget_bytes,
            configured_safety_reserve=self._gpu_safety_reserve_bytes,
            physical_allocator_headroom=physical_headroom,
            budget_headroom=budget_headroom,
            safe_allocatable_now=min(physical_headroom, budget_headroom),
        )

    def _resident_candidates(self, *, exclude=()):
        excluded = {handle.id for handle in exclude}
        return [
            self._candidate(handle)
            for handle in self.tensors.values()
            if handle.resident is not None and handle.id not in excluded and not handle.dirty
        ]

    def _update_residency_peak(self):
        self.metrics["peak_vram_resident_bytes"] = max(
            self.metrics["peak_vram_resident_bytes"], self._resident_bytes
        )

    def ensure_cuda_capacity(self, bytes_needed, *, protected=(), context=None):
        """Evict cold values until a transient CUDA working set can fit."""
        if self._gpu_device is None:
            return
        protected_resident = sum(
            handle.size
            for handle in {handle.id: handle for handle in protected}.values()
            if handle.resident is not None
        )
        additional = max(0, int(bytes_needed) - protected_resident)
        while True:
            snapshot = self.cuda_memory_snapshot()
            if snapshot.safe_allocatable_now >= additional:
                return snapshot
            shortage = additional - snapshot.safe_allocatable_now
            selected = self.policy.select_evictions(
                self._resident_candidates(exclude=protected), shortage, self._access_index
            )
            if not selected:
                detail = f" {context}" if context else ""
                raise TrainPoolError(
                    f"TRAINPOOL_UNSUPPORTED_WORKING_SET:{detail} required={bytes_needed} "
                    f"additional={additional} {snapshot.diagnostic()}"
                )
            for candidate in selected:
                self._spill(self.tensors[candidate.id])

    def prepare_transient(self, handles):
        """Make optimizer inputs durable and nonresident before transient restore.

        A normal CUDA restore may promote a handle and retain the restored value.
        Optimizer inputs must instead have exactly one live CUDA copy, while the
        old transactional version remains recoverable in RAM until publication.
        """
        with self._lock:
            for handle in handles:
                if handle.resident is not None:
                    self._spill(handle)

    def restore_transient(self, handle, *, device=None):
        return self.restore(handle, device=device, retain=False)

    def _try_retain(self, value, handle):
        import torch

        if self._gpu_device is None or handle.size > self._gpu_budget_bytes:
            return False
        capacity = self._effective_vram_capacity()
        needed = max(0, self._resident_bytes + handle.size - capacity)
        if needed:
            incoming = self._candidate(handle)
            selected = self.policy.select_evictions(self._resident_candidates(), needed, self._access_index)
            # Do not evict a more valuable near-use tensor for a colder arrival.
            if not selected or any(
                self.policy.eviction_key(item, self._access_index)
                <= self.policy.eviction_key(incoming, self._access_index)
                for item in selected
            ):
                return False
            for candidate in selected:
                self._spill(self.tensors[candidate.id])
        # Avoid probing the allocator all the way to its last pages. PyTorch can
        # request a slightly larger internal block than the tensor itself.
        free, _total = torch.cuda.mem_get_info(self._gpu_device)
        cached = torch.cuda.memory_reserved(self._gpu_device) - torch.cuda.memory_allocated(self._gpu_device)
        if handle.size and free + cached < handle.size + 16 * 1024 * 1024:
            return False
        try:
            try:
                handle.resident = value.detach().to(self._gpu_device, copy=True).contiguous()
            except TypeError:
                handle.resident = value.detach().clone().to(self._gpu_device).contiguous()
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return False
        self._resident_bytes += handle.size
        self._update_residency_peak()
        return True

    def _check(self):
        if self._closed.is_set():
            raise TrainPoolError("TensorStore is closed")
        if self._lease_error:
            raise TrainPoolError(f"TRAINPOOL_LEASE_RENEWAL_FAILED: {self._lease_error}")

    def _renew_loop(self):
        period = max(1, self.client.status["lease_seconds"] / 3)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=8, thread_name_prefix="trainpool-renew"
        ) as executor:
            while not self._closed.wait(period):
                try:
                    # Hold the table lock so free cannot race renewal. Calls are
                    # independent and bounded, so renewing in parallel prevents a
                    # large DeepLab handle table from exceeding short test leases.
                    with self._lock:
                        tensors = [tensor for tensor in self.tensors.values() if tensor.blocks]
                        blocks = [block for tensor in tensors for block in tensor.blocks]
                        renewed = list(
                            executor.map(
                                lambda block: self.client.control("renew", handle=block, direct=False),
                                blocks,
                            )
                        )
                        iterator = iter(renewed)
                        for tensor in tensors:
                            tensor.blocks = [next(iterator) for _ in tensor.blocks]
                except Exception as error:
                    self._lease_error = error
                    return

    def _backing_totals(self):
        local = remote = 0
        for tensor in self.tensors.values():
            for block in tensor.blocks:
                if block["owner_node"] == self.client.local_node:
                    local += block["size"]
                else:
                    remote += block["size"]
        return local, remote

    def _update_backing_peaks(self):
        local, remote = self._backing_totals()
        self.metrics["peak_local_ram_backing_bytes"] = max(
            self.metrics["peak_local_ram_backing_bytes"], local
        )
        self.metrics["peak_remote_ram_backing_bytes"] = max(
            self.metrics["peak_remote_ram_backing_bytes"], remote
        )

    def _write_backing(self, value, handle):
        import torch

        raw = value.reshape(-1).view(torch.uint8)
        new_blocks = []
        try:
            for offset in range(0, raw.numel(), self.block_bytes):
                size = min(self.block_bytes, raw.numel() - offset)

                def chunks(offset=offset, size=size):
                    for start in range(offset, offset + size, 65536):
                        length = min(65536, offset + size - start)
                        with self.client.staging(length):
                            chunk = raw[start : start + length].to("cpu")
                            yield memoryview(chunk.numpy())
                            del chunk

                block = self.client.put_chunks(
                    size, chunks(), self.job_id, preferred_node=self.preferred_node
                )
                new_blocks.append(block)
                with self._lock:
                    handle.blocks.append(block)
                    self._update_backing_peaks()
            return new_blocks
        except BaseException:
            for block in new_blocks:
                with contextlib.suppress(Exception):
                    self.client.free(block)
            with self._lock:
                handle.blocks = [block for block in handle.blocks if block not in new_blocks]
            raise

    def _drop_blocks(self, handle):
        blocks, handle.blocks = handle.blocks, []
        for block in blocks:
            self.client.free(block)

    def _spill(self, handle):
        if handle.resident is None:
            return
        resident = handle.resident
        handle.dirty = True
        try:
            blocks = self._write_backing(resident, handle)
        except BaseException:
            handle.dirty = False
            raise
        local = sum(block["size"] for block in blocks if block["owner_node"] == self.client.local_node)
        remote = handle.size - local
        handle.transfer_cost = 4.0 if remote else 1.0
        self.metrics["bytes_gpu_to_local_ram"] += local
        self.metrics["gpu_to_local_bytes"] += local
        self.metrics["gpu_to_remote_bytes"] += remote
        handle.resident = None
        self._resident_bytes -= handle.size
        handle.dirty = False
        self.metrics["eviction_count"] += 1
        del resident

    def offload(
        self,
        tensor,
        *,
        expected_next_use=None,
        remaining_consumers=1,
        allow_vram=True,
    ):
        import torch

        self._check()
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Only dense strided, non-quantized tensors are supported")
        self._require_cuda(tensor.device)
        value = tensor.detach().resolve_conj().resolve_neg().contiguous()
        handle = TensorHandle(
            str(uuid.uuid4()),
            tuple(tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
            tensor.numel() * tensor.element_size(),
            str(tensor.device),
            [],
            expected_next_use=expected_next_use,
            dirty=True,
            remaining_consumers=remaining_consumers,
            access_index=self._access_index,
        )
        # Renew completed chunks even when a multi-gigabyte upload lasts several leases.
        with self._lock:
            self.tensors[handle.id] = handle
        try:
            if allow_vram and self._try_retain(value, handle):
                if tensor.device.type == "cpu":
                    self.metrics["bytes_local_ram_to_gpu"] += handle.size
                    self.metrics["local_to_gpu_bytes"] += handle.size
                handle.dirty = False
                return handle
            blocks = self._write_backing(value, handle)
            with self._lock:
                handle.dirty = False
                if tensor.device.type == "cuda":
                    local = sum(
                        block["size"] for block in blocks if block["owner_node"] == self.client.local_node
                    )
                    remote = handle.size - local
                    handle.transfer_cost = 4.0 if remote else 1.0
                    self.metrics["bytes_gpu_to_local_ram"] += local
                    self.metrics["gpu_to_local_bytes"] += local
                    self.metrics["gpu_to_remote_bytes"] += remote
            return handle
        except BaseException:
            with contextlib.suppress(Exception):
                self.free(handle)
            raise

    def restore(self, handle, *, device=None, retain=True):
        import torch

        self._check()
        if handle.dirty:
            raise TrainPoolError("Tensor upload is not complete")
        self.client.control("job_status", job_id=self.job_id)
        target = torch.device(device or handle.device)
        self._require_cuda(target)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("Only explicit CPU tensors and CUDA tensors are supported")
        with self._lock:
            handle.reuse_count += 1
            handle.access_index = self._access_index
            resident = handle.resident
        if resident is not None:
            return resident.detach().to(target, copy=True).contiguous()
        if target.type == "cuda":
            self.ensure_cuda_capacity(handle.size, protected=(handle,))
        start = time.perf_counter()
        result = torch.empty(handle.shape, dtype=getattr(torch, handle.dtype), device=target)
        raw = result.reshape(-1).view(torch.uint8)
        offset = 0
        for block in handle.blocks:
            for buffer in self.client.read_chunks(block):
                source = torch.frombuffer(buffer, dtype=torch.uint8)
                raw[offset : offset + len(buffer)].copy_(source)
                offset += len(buffer)
                # Synchronous copy keeps the bounded host buffer alive until done.
                del source, buffer
        with self._lock:
            if target.type == "cuda":
                local = sum(
                    block["size"] for block in handle.blocks if block["owner_node"] == self.client.local_node
                )
                remote = handle.size - local
                handle.transfer_cost = 4.0 if remote else 1.0
                self.metrics["bytes_local_ram_to_gpu"] += local
                self.metrics["local_to_gpu_bytes"] += local
                self.metrics["remote_to_gpu_bytes"] += remote
                self.metrics["peak_gpu_residency"] = max(
                    self.metrics["peak_gpu_residency"], torch.cuda.memory_allocated(target)
                )
            if target.type == "cuda" and not threading.current_thread().name.startswith("trainpool-prefetch"):
                self.metrics["gpu_wait_for_data_ms"] += (time.perf_counter() - start) * 1000
            # Promotion is a move: after a successful RAM -> VRAM restore, the
            # backing allocation is released instead of becoming a replica.
            if (
                retain
                and target.type == "cuda"
                and self._resident_bytes + handle.size <= self._effective_vram_capacity()
            ):
                handle.resident = result
                self._resident_bytes += handle.size
                self._update_residency_peak()
                self._drop_blocks(handle)
                return result.detach().clone()
        return result

    def transactional_offload(self, tensor, *, expected_next_use=None):
        """Publish replacement state without requiring a second VRAM copy.

        Optimizer transactions retain the old handle until all replacements are
        durable. The replacement is therefore staged in RAM and can be promoted
        on its next use after the old ownership has been released.
        """
        return self.offload(
            tensor,
            expected_next_use=expected_next_use,
            allow_vram=False,
        )

    def free(self, handle):
        with self._lock:
            if handle.id not in self.tensors:
                return
            self._drop_blocks(handle)
            if handle.resident is not None:
                handle.resident = None
                self._resident_bytes -= handle.size
            self.tensors.pop(handle.id)

    def flush_metrics(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_metrics_flush < self._metrics_interval_seconds:
            return False
        with self._lock:
            metrics = dict(self.metrics)
            local, remote = self._backing_totals()
            metrics["current_vram_resident_bytes"] = self._resident_bytes
            metrics["current_local_ram_backing_bytes"] = local
            metrics["current_remote_ram_backing_bytes"] = remote
            if self._gpu_device is not None:
                import torch

                metrics["current_gpu_residency"] = torch.cuda.memory_allocated(self._gpu_device)
                metrics["peak_gpu_residency"] = max(
                    metrics["peak_gpu_residency"], torch.cuda.max_memory_allocated(self._gpu_device)
                )
                metrics["gpu_id"] = next(
                    g["gpu_id"]
                    for g in self.plan["gpu_assignments"]
                    if g["node_id"] == self.client.local_node
                )
                snapshot = self.cuda_memory_snapshot()
                metrics.update(
                    {
                        "physical_vram_bytes": snapshot.physical_total,
                        "driver_free_vram_bytes": snapshot.driver_free,
                        "driver_used_vram_bytes": snapshot.driver_used,
                        "torch_allocated_bytes": snapshot.torch_allocated,
                        "torch_reserved_bytes": snapshot.torch_reserved,
                        "torch_reclaimable_bytes": snapshot.torch_reclaimable,
                        "safe_vram_allocatable_bytes": snapshot.safe_allocatable_now,
                        "configured_usable_vram_ceiling_bytes": snapshot.configured_usable_ceiling,
                        "configured_vram_safety_reserve_bytes": snapshot.configured_safety_reserve,
                    }
                )
            metrics["sdk_metrics_timestamp_ms"] = int(time.time() * 1000)
            # Rust uses explicit fields; unspecified counters are filled with zeros.
            defaults = {
                "tensor_allocations": 0,
                "tensor_migrations": 0,
                "bytes_remote_ram_to_local": 0,
                "bytes_local_to_remote_ram": 0,
                "migration_latency_ms": 0.0,
                "ram_pressure_events": 0,
                "network_bytes": 0,
                "network_wait_ms": 0.0,
                "peak_ram_residency": 0,
                "failed_transfers": 0,
                "remote_to_local_bytes": 0,
                "local_to_remote_bytes": 0,
            }
            self.client.control("report_metrics", job_id=self.job_id, metrics={**defaults, **metrics})
            self._last_metrics_flush = time.monotonic()
            for key in self.metrics:
                self.metrics[key] = 0
        return True

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        # A weakref finalizer may run on the lease thread when that thread
        # releases the last reference to a model/runtime. Joining the current
        # thread raises RuntimeError and would skip durable-handle cleanup.
        if threading.current_thread() is not self._renewal:
            self._renewal.join(timeout=self.client.timeout + 1)
        try:
            self.flush_metrics(force=True)
        finally:
            for handle in list(self.tensors.values()):
                try:
                    self.free(handle)
                except Exception as error:
                    warnings.warn(
                        f"TrainPool cleanup deferred to lease expiry: {error}", RuntimeWarning, stacklevel=2
                    )
            # Send zero current-tier gauges after releasing ownership. Counter
            # deltas were already reported by the first flush above.
            with contextlib.suppress(Exception):
                self.flush_metrics(force=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class NoPrefetch:
    def schedule(self, key, function):
        pass

    def take(self, key, function):
        return function()

    def close(self):
        pass


class SequentialPrefetch:
    """One future stage at a time; never an unbounded tensor cache."""

    def __init__(self, store):
        self.store = store
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="trainpool-prefetch"
        )
        self.pending = {}

    def schedule(self, key, function):
        if not self.pending:
            self.store.metrics["prefetch_count"] += 1
            self.pending[key] = self.executor.submit(function)

    def take(self, key, function):
        future = self.pending.pop(key, None)
        if future is None:
            self.store.metrics["prefetch_misses"] += 1
            return function()
        self.store.metrics["prefetch_hits" if future.done() else "prefetch_misses"] += 1
        started = time.perf_counter()
        result = future.result()
        if self.store._gpu_device is not None:
            self.store.metrics["gpu_wait_for_data_ms"] += (time.perf_counter() - started) * 1000
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()
