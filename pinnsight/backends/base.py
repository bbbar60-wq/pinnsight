"""
Backend protocol — the only layer that touches PyTorch or JAX directly.
All metrics are written against this protocol, never against torch.* or jax.* directly.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


# Type aliases — kept simple intentionally
Params = Any       # torch nn.Module parameters or JAX pytree
Array = Any        # torch.Tensor or jax.Array
FlatVector = Any   # 1-D flattened parameter vector


@runtime_checkable
class Backend(Protocol):
    """
    Minimal differentiable-compute protocol over PyTorch and JAX.

    Implementations: TorchBackend, JaxBackend.
    Auto-detected from model/params type via detect_backend().
    """

    def grad(self, scalar: Array, params: Params) -> FlatVector:
        """
        Compute gradient of a scalar w.r.t. params.
        Returns a flat 1-D vector of the same dtype as params.
        """
        ...

    def jacobian(self, fn: Any, x: Array, params: Params) -> Array:
        """
        Compute per-output parameter Jacobian of fn(params, x).
        Shape: [len(x), num_params] — used for empirical NTK.
        """
        ...

    def hvp(self, scalar: Array, params: Params, v: FlatVector) -> FlatVector:
        """
        Hessian-vector product: H(scalar, params) @ v.
        Used for Hessian diagnostics in V2.
        """
        ...

    def flatten(self, params: Params) -> FlatVector:
        """Flatten params into a 1-D numpy-compatible vector."""
        ...

    def to_numpy(self, x: Array) -> np.ndarray:
        """Detach and convert to numpy, regardless of backend."""
        ...


def detect_backend(model: Any) -> str:
    """
    Auto-detect whether the model/params are PyTorch or JAX.
    Returns 'torch' or 'jax'.
    Raises ValueError if neither is detectable.
    """
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return "torch"
    except ImportError:
        pass

    try:
        import jax
        # JAX models are typically (apply_fn, params) tuples or pytrees
        if isinstance(model, tuple) and callable(model[0]):
            return "jax"
    except ImportError:
        pass

    raise ValueError(
        "Cannot detect backend from model type. "
        "Pass backend='torch' or backend='jax' explicitly to Scope()."
    )