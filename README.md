# pinnsight

> Diagnose why your physics-informed neural network won't train.

[![CI](https://github.com/bbbar60-wq/pinnsight/actions/workflows/ci.yml/badge.svg)](https://github.com/bbbar60-wq/pinnsight/actions)
[![Version](https://img.shields.io/badge/version-0.0.1-blue)](https://github.com/bbbar60-wq/pinnsight)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

`pinnsight` is a diagnostics and instrumentation layer for physics-informed neural networks (PINNs). You keep your framework — it tells you **why your training is failing**.

## The problem

Your PINN loss looks converged but the solution is physically wrong. You have no idea which of the known failure modes — loss imbalance, gradient conflict, spectral bias, causality violation — is responsible. You're debugging with print statements and intuition.

`pinnsight` fixes that.

## Status

**Active development — v0.0.1 (pre-release)**

The library is being built in public. Follow along or contribute.

## Installation

```bash
pip install pinnsight          # core (numpy + matplotlib only)
pip install pinnsight[torch]   # with PyTorch backend
pip install pinnsight[jax]     # with JAX backend
pip install pinnsight[viz]     # with interactive visualizations
```

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