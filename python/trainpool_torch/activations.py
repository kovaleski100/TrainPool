from __future__ import annotations

import contextlib

from .store import TensorStore


class _SavedTensor:
    def __init__(self, store, tensor):
        self.store = store
        self.handle = store.offload(tensor)

    def __del__(self):
        with contextlib.suppress(Exception):
            self.store.free(self.handle)


@contextlib.contextmanager
def distributed_memory(*, store=None, preferred_node=None, min_bytes=4096):
    """Offload saved activations. Forward AND backward must finish inside the context.

    Model parameters and optimizer state are managed by prepare(), not these hooks.
    """
    import torch

    own_store = store is None
    store = store or TensorStore(preferred_node=preferred_node)

    def pack(tensor):
        if tensor.numel() * tensor.element_size() < min_bytes:
            return tensor.detach()
        return _SavedTensor(store, tensor)

    def unpack(saved):
        return store.restore(saved.handle) if isinstance(saved, _SavedTensor) else saved

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield store
    finally:
        if own_store:
            store.close()
