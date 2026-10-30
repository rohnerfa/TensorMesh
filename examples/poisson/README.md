# Poisson Equation

Elliptic PDE examples covering basic 2D / 3D solves and h-adaptivity.

For batch generation across many source terms (ML workflows), see
[`examples/dataset/poisson/`](../dataset/poisson/).

## Scripts

| Script | Description |
|--------|-------------|
| `poisson.py` | 2D Poisson on a rectangle with multi-frequency source term |
| `poisson_3d.py` | 3D Poisson on a unit cube with tetrahedral mesh |
| `poisson_h_adaptivity.py` | H-adaptive refinement on an L-shaped domain with gradient singularity |
| `compare_hierarchical.py` | Matched nodal/hierarchical triangle solves, analytical errors and change-of-basis checks |
| `benchmark_hierarchical.py` | Separate reference setup, tabulation, integration, projection, solve and accuracy diagnostics |

## Problem Setup

- **PDE:** $-\Delta u = f$ in $\Omega$, $u = 0$ on $\partial\Omega$
- **Source Term:** Multi-frequency Fourier series via `PoissonMultiFrequency`
- **Domains:** Rectangle, cube, L-shaped

## Usage

```bash
python poisson.py                            # basic 2D example
python poisson_3d.py                         # 3D example
python poisson_h_adaptivity.py               # adaptive refinement (requires gmsh >= 4.8)
```

Run the basis comparison and benchmark from the repository root:

```bash
python examples/poisson/compare_hierarchical.py --degree 3 --length 0.2 --modes 4
python examples/poisson/benchmark_hierarchical.py
# Optional CUDA run; requires CUDA-enabled PyTorch and an available GPU:
python examples/poisson/benchmark_hierarchical.py --device cuda
```

Both comparisons use identical geometry, quadrature and forcing for the two
families. JSON reports include the numerical accuracy separately from timing.
The benchmark defaults to CPU float64 and a small mesh; GPU memory is reported
only when CUDA measurement actually runs. CUDA solve timing uses CG, while CPU
uses sparse LU, so compare basis families within a backend, not CPU/GPU solver
timings as if the algorithms were identical. The high-degree Lagrange basis is
diagnosed rather than assumed accurate. Generated outputs live under `output/`
and are not source files.

## Output

- `poisson.png`: 2D solution contour
- `poisson_3d.vtu`: 3D solution (open with ParaView)
- `poisson_h_adaptivity.png`: convergence plot (adaptive vs uniform) and final mesh
