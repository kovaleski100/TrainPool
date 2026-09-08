"""Tensor table and a single-worker, bounded look-ahead prefetch policy."""

from __future__ import annotations

import concurrent.futures
import contextlib
import dataclasses
import os
import threading
import time
import uuid
import warnings

from .client import Client, TrainPoolError


@dataclasses.dataclass
class TensorHandle:
    id: str
    shape: tuple
    dtype: str
    size: int
    device: str
    blocks: list
    layout: str = "contiguous"
    expected_next_use: str | None = None
    dirty: bool = False

    @property
    def current_location(self):
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
        self.block_bytes = min(block_bytes or self.client.chunk_bytes, self.client.chunk_bytes)
        if self.block_bytes < 4096:
            raise ValueError("block_bytes must be at least 4096")
        self.tensors = {}
        self._lock = threading.RLock()
        self._closed = threading.Event()
        self._lease_error = None
        self._gpu_device = None
        self.metrics = {
            "bytes_local_ram_to_gpu": 0,
            "bytes_gpu_to_local_ram": 0,
            "prefetch_hits": 0,
            "prefetch_misses": 0,
            "gpu_wait_for_data_ms": 0.0,
            "peak_gpu_residency": 0,
        }
        self._renewal = threading.Thread(target=self._renew_loop, name="trainpool-leases", daemon=True)
        self._renewal.start()

    def _require_cuda(self, device):
        if device.type == "cuda":
            if self.client.local_node not in self.plan["compute_nodes"]:
                raise TrainPoolError("TRAINPOOL_NO_CUDA: local node is not a CUDA compute provider")
            self._gpu_device = device

    def _check(self):
        if self._closed.is_set():
            raise TrainPoolError("TensorStore is closed")
        if self._lease_error:
            raise TrainPoolError(f"TRAINPOOL_LEASE_RENEWAL_FAILED: {self._lease_error}")

    def _renew_loop(self):
        period = max(1, self.client.status["lease_seconds"] / 3)
        while not self._closed.wait(period):
            try:
                # Hold the table lock so free cannot race lease renewal.
                with self._lock:
                    for tensor in self.tensors.values():
                        tensor.blocks = [
                            self.client.control("renew", handle=b, direct=False) for b in tensor.blocks
                        ]
            except Exception as error:
                self._lease_error = error
                return

    def offload(self, tensor, *, expected_next_use=None):
        import torch

        self._check()
        if tensor.layout != torch.strided or tensor.is_quantized:
            raise ValueError("Only dense strided, non-quantized tensors are supported")
        self._require_cuda(tensor.device)
        value = tensor.detach().resolve_conj().resolve_neg().contiguous()
        raw = value.reshape(-1).view(torch.uint8)
        handle = TensorHandle(
            str(uuid.uuid4()),
            tuple(tensor.shape),
            str(tensor.dtype).removeprefix("torch."),
            tensor.numel() * tensor.element_size(),
            str(tensor.device),
            [],
            expected_next_use=expected_next_use,
            dirty=True,
        )
        # Renew completed chunks even when a multi-gigabyte upload lasts several leases.
        with self._lock:
            self.tensors[handle.id] = handle
        try:
            for offset in range(0, raw.numel(), self.block_bytes):
                size = min(self.block_bytes, raw.numel() - offset)
                with self.client.staging(size):
                    chunk = raw[offset : offset + size].to("cpu")
                    block = self.client.put(
                        memoryview(chunk.numpy()), self.job_id, preferred_node=self.preferred_node
                    )
                    with self._lock:
                        handle.blocks.append(block)
                    del chunk
            with self._lock:
                handle.dirty = False
                if tensor.device.type == "cuda":
                    self.metrics["bytes_gpu_to_local_ram"] += handle.size
            return handle
        except BaseException:
            with contextlib.suppress(Exception):
                self.free(handle)
            raise

    def restore(self, handle, *, device=None):
        import torch

        self._check()
        if handle.dirty:
            raise TrainPoolError("Tensor upload is not complete")
        self.client.control("job_status", job_id=self.job_id)
        target = torch.device(device or handle.device)
        self._require_cuda(target)
        if target.type not in {"cpu", "cuda"}:
            raise ValueError("Only explicit CPU tensors and CUDA tensors are supported")
        start = time.perf_counter()
        result = torch.empty(handle.shape, dtype=getattr(torch, handle.dtype), device=target)
        raw = result.reshape(-1).view(torch.uint8)
        offset = 0
        for block in handle.blocks:
            size = block["size"]
            with self.client.staging(size):
                buffer = bytearray(size)
                self.client.read_into(block, buffer)
                source = torch.frombuffer(buffer, dtype=torch.uint8)
                raw[offset : offset + size].copy_(source)
                # copy_ is synchronous here; host memory remains alive until GPU copy completes.
                del source, buffer
            offset += size
        with self._lock:
            if target.type == "cuda":
                self.metrics["bytes_local_ram_to_gpu"] += handle.size
                self.metrics["peak_gpu_residency"] = max(
                    self.metrics["peak_gpu_residency"], torch.cuda.memory_allocated(target)
                )
            if target.type == "cuda" and not threading.current_thread().name.startswith("trainpool-prefetch"):
                self.metrics["gpu_wait_for_data_ms"] += (time.perf_counter() - start) * 1000
        return result

    def free(self, handle):
        with self._lock:
            if handle.id not in self.tensors:
                return
            for block in handle.blocks:
                self.client.free(block)
            self.tensors.pop(handle.id)

    def flush_metrics(self):
        with self._lock:
            metrics = dict(self.metrics)
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
            }
            self.client.control("report_metrics", job_id=self.job_id, metrics={**defaults, **metrics})
            for key in self.metrics:
                self.metrics[key] = 0

    def close(self):
        if self._closed.is_set():
            return
        self._closed.set()
        self._renewal.join(timeout=self.client.timeout + 1)
        try:
            self.flush_metrics()
        finally:
            for handle in list(self.tensors.values()):
                try:
                    self.free(handle)
                except Exception as error:
                    warnings.warn(
                        f"TrainPool cleanup deferred to lease expiry: {error}", RuntimeWarning, stacklevel=2
                    )

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
