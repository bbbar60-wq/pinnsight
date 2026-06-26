"""
pinnsight — A diagnostics and instrumentation layer for PINNs.

You keep your framework; it tells you why your training is failing.

Basic usage (PyTorch):
    import pinnsight as ps

    scope = ps.Scope(
        model=net,
        residuals={"pde": pde_residual, "bc": bc_residual},
        eval_points={"pde": x_eval, "bc": x_bc},
        diagnose_every=500,
    )

    with scope:
        for step in range(n_steps):
            losses = {"pde": loss_pde, "bc": loss_bc}
            scope.step(losses, step)

    scope.report("diagnosis.html")
"""

__version__ = "0.0.1"

# Core orchestration — imported as they are built
# from pinnsight.scope import Scope          # M1 end
# from pinnsight.history import RunHistory   # M1 end
# from pinnsight.diagnosis import Diagnosis  # M3

# Backend detection utility — available now
from pinnsight.backends.base import Backend, detect_backend

__all__ = [
    "__version__",
    "Backend",
    "detect_backend",
]