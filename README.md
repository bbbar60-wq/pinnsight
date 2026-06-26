# pinnsight

> Diagnose why your physics-informed neural network won't train.

[![CI](https://github.com/bbbar60-wq/pinnsight/actions/workflows/ci.yml/badge.svg)](https://github.com/bbbar60-wq/pinnsight/actions)
[![PyPI](https://img.shields.io/pypi/v/pinnsight)](https://pypi.org/project/pinnsight/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`pinnsight` is a diagnostics and instrumentation layer for physics-informed neural networks (PINNs). You keep your framework — it tells you **why your training is failing**.

## The problem

Your PINN loss looks converged but the solution is physically wrong. You have no idea which of the known failure modes — loss imbalance, gradient conflict, spectral bias, causality violation — is responsible. You're debugging with print statements and intuition.

`pinnsight` fixes that.

## Status

🚧 **Active development — v0.0.1 (pre-release)**

The library is being built in public. Follow along or contribute.

## Installation

```bash
pip install pinnsight          # core (numpy + matplotlib only)
pip install pinnsight[torch]   # with PyTorch backend
pip install pinnsight[jax]     # with JAX backend
pip install pinnsight[viz]     # with interactive visualizations
```

## Roadmap

- **M1–M4 (MVP):** gradient norm diagnostics, residual heatmaps, rule-based `suggest()`, killer demo
- **M5–M10 (V1.0):** NTK module, causality profiler, spectral-bias analyzer, JAX backend, workshop paper
- **M11–M20 (V2.0):** PI-RNN diagnostics, Hessian analysis, live monitor, JOSS paper

## Citation

If you use `pinnsight` in your research, please cite:

```bibtex
@software{pinnsight,
  author = {Reza},
  title  = {pinnsight: A diagnostics and instrumentation layer for PINNs},
  url    = {https://github.com/bbbar60-wq/pinnsight},
  year   = {2026},
}
```

## License

Apache 2.0