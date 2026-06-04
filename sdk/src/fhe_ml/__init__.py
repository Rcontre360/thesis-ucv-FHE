"""Public API of the fhe-ml SDK.

Single entry point for everything a user needs to build, train, encrypt and
run an FHE-friendly neural network on top of CKKS. Re-exports the surface
that lives across `api.*` and `core.*`; do not import from those directly
in user code.
"""
from api import (
    EncryptedVector,
    FHEContext,
    FHESDKError,
    Input,
    LayerConfigError,
    PlaintextTensor,
    ShapeError,
)
from api.sequential import Sequential
from core import PlaintextVector, SecurityLevel

__all__ = [
    "FHEContext",
    "Sequential",
    "Input",
    "PlaintextTensor",
    "PlaintextVector",
    "EncryptedVector",
    "SecurityLevel",
    "FHESDKError",
    "ShapeError",
    "LayerConfigError",
]
