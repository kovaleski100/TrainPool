"""TrainPool SDK. Importing this package does not require CUDA or even PyTorch."""

from .activations import distributed_memory
from .client import Client, TrainPoolError
from .store import NoPrefetch, SequentialPrefetch, TensorHandle, TensorStore


def prepare(model, optimizer, **kwargs):
    from .sequential import prepare as implementation

    return implementation(model, optimizer, **kwargs)


def stage(module):
    from .sequential import stage as implementation

    return implementation(module)


__all__ = [
    "Client",
    "TrainPoolError",
    "TensorHandle",
    "TensorStore",
    "NoPrefetch",
    "SequentialPrefetch",
    "distributed_memory",
    "prepare",
    "stage",
]
