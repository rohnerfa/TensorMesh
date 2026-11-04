# Allen-Cahn Equation

Nonlinear phase-field evolution solved with Newton's method at each time
step. `ac.py` and `ac_torch_sla.py` solve the *same* fully-implicit
backward-Euler step and produce identical phase fields to round-off — they
differ only in how the per-step Newton solve is driven. `ac_convex_concave.py`
solves a *different* time discretisation — Eyre's convex-concave splitting —
which is unconditionally energy-stable and so stays well-behaved at far
larger time steps.

## Scripts

| Script | Description |
|--------|-------------|
| `ac.py` | The Newton loop written out by hand: assemble the tangent `K` and residual `R`, `K.solve(R)`, update, repeat until `‖R‖` is small. |
| `ac_torch_sla.py` | The same step handed to `torch_sla.nonlinear_solve` — one call per time step. Newton / Picard / Anderson with an Armijo line search and an adjoint backward pass, all packaged. |
| `ac_convex_concave.py` | Eyre's convex splitting: the convex `c^3` reaction stays implicit, the concave `-c` term is taken explicitly. The Newton tangent is SPD and the scheme is unconditionally energy-stable, so `--dt` can be raised far beyond the fully-implicit limit. |

`ac_torch_sla.py` passes the FEM consistent tangent explicitly as
`jacobian_fn` (rather than letting `nonlinear_solve` build it via
autograd). Note the sign: `KAssembler` assembles the *negative* tangent
`K = -∂R/∂c`, so `jacobian_fn` returns `-K.values` to hand back the true
Jacobian `J`, which `nonlinear_solve` then steps with `J du = -R`.

## Problem Setup

- **PDE:** $u_t = \Delta u + \varepsilon^2 u(1 - u^2)$
- **Geometry:** Unit square $[0,1]^2$ (triangular mesh)
- **Boundary Conditions:** Natural (no-flux) boundary conditions
- **Initial Condition:** Multi-frequency Fourier series via `PoissonMultiFrequency`
- **Nonlinear Solver:** Newton iteration (max 50 iterations per step, tolerance $10^{-10}$)
- **Time Integration:** Implicit Euler, $\Delta t = 10^{-6}$, 200 steps

## Usage

```bash
python ac.py                # hand-written Newton loop
python ac_torch_sla.py      # same step via torch_sla.nonlinear_solve
python ac_convex_concave.py # Eyre convex splitting (unconditionally stable)
```

## Output

- `Allen-Cahn.mp4`: animation of the phase-field evolution (`ac.py`)
- `Allen-Cahn-torch-sla.mp4`: the identical evolution driven by `nonlinear_solve` (`ac_torch_sla.py`)
- `Allen-Cahn-convex-concave-dt<dt>.mp4`: the convex-splitting evolution (`ac_convex_concave.py`); the file name encodes the `--dt` used

## High-order runs and four-way benchmark

`ac_convex_concave_p2.py`, `ac_convex_concave_p3.py`, and
`ac_convex_concave_p4.py` retain the convex-splitting example's physical
settings, using degree-8/12/16 quadrature respectively. Each writes its own
video and reports simulation and rendering times separately.

From the repository root, compare upstream **nodal** / current **hierarchical**
and Element/Mixed on P3:

```sh
python examples/diffusion/allen-cahn/benchmark_four_way.py \
    --upstream /path/to/clean/upstream-checkout \
    --current /path/to/hierarchical-triangle-element-checkout \
    --output dist/allen-cahn-four-way-hierarchical-p3
```

The upstream checkout must match the pinned commit printed in the source
(`2d544e8...`, overridable with `--upstream-commit`). Both
upstream cases use **nodal Lagrange**; both current cases require
**hierarchical** fields, with no nodal fallback. All use float64 CPU,
the same saved P3 mesh and initial physical field, 10 steps, `dt=5e-5`,
`epsilon=220`, and five CPU threads.
`--order 2`, `--order 4`, and `--order 5` select the other orders. There are three trials
per case in fresh processes, alternating forward/reverse case order.
`--allow-dirty-current` explicitly permits an uncommitted implementation;
source changes and hashes are recorded, and workers fail if sources change
while they run. The harness does not modify either library checkout.

Upstream cannot generate the high-degree triangle rules itself, so the
harness overrides **only `Triangle.get_quadrature` in both checkouts** with
identical external Gaussian data (degree `4*p`). Assembly and solver code
are not patched. This is an assembly/API comparison with matched integration,
not an unmodified-upstream feature-support comparison.

Jacobian assembly uses `batch_size=-1`; residual assembly uses `batch_size=1`
for every case, matching the original Element/Node example rather than
silently using Mixed's different residual default. No compilation is enabled.
Upstream Element uses separate Element/Node assemblers; upstream Mixed uses
one assembler with `forward`/`forward_vector`.

**Current Element uses hierarchical Element + hierarchical NodeAssembler**
for the Jacobian and residual, respectively, without a Mixed fallback. The
two assemblers use separate factory calls, as in the upstream Element/Node
case. Their nonlinear `point_data` API requires geometry-node values, so modal
states are converted with `field_layout().prolong` inside both timed assembly
calls. Current Mixed instead consumes modal states through `field_data`.
An earlier report in `allen-cahn-four-way-hierarchical-p3` used a Mixed
residual fallback before hierarchical NodeAssembler was available; it should
not be confused with the newer `allen-cahn-four-way-node-*` reports.

Modal initial coefficients represent the same nodal interpolant as upstream.
For validation, a sparse `T` maps modal coefficients to physical mesh-node
values. Jacobians/loads are compared as `K_H = T.T @ K_L @ T` and
`R_H = T.T @ R_L`; raw modal/nodal arrays are not compared. Every Newton loop
uses the same **nodal-dual** residual norm, converting modal residuals with
`T.T`'s inverse. That conversion is included in `update_and_norm`. The
diagnostic sparse map/LU and initial conversion setup are reported separately;
inclusive setup-and-stepping times are also provided.

`report.json` records source commits/hashes, import paths, inputs, setup times,
per-Newton-iteration assembly and solve timings, full stepping wall times,
and all trial samples. Timings exclude input generation, process startup,
artifact output and video rendering. Stepping wall time includes a first-matrix
snapshot used for validation; operation timings exclude that snapshot. Each
trial starts cold; later iterations can benefit from the library's own caches.
The report checks the initial Jacobian/residual under change of basis and
every saved physical time-step field against the first upstream Element run
before marking results complete. Earlier results in `allen-cahn-four-way-p3`
were an **all-nodal baseline**, not a hierarchical performance comparison.

### Direct versus iterative linear solves

`--linear-solver lu` (default) explicitly uses the same SciPy sparse LU
backend selected by the original `matrix.solve()` auto dispatch, via
`matrix.solve(backend="scipy", method="lu")` in torch-sla.

`--linear-solver cg-jacobi` uses SciPy conjugate gradient with diagonal
(Jacobi) preconditioning. The convex-split Jacobian is SPD: it is the sum
of stiffness, positive mass/`dt`, and a nonnegative reaction-weighted mass.
Defaults are `--linear-rtol 1e-10 --linear-atol 1e-14 --linear-maxiter 5000`.
Every call starts from zero and builds a new Jacobi preconditioner; there
is no reuse, mixed precision or silent direct-solver fallback.

Linear solve timings include matrix conversion, preconditioner setup for
CG, and explicit true-residual verification. CG iteration counts and
true residuals are recorded for every Newton system. The strict `1e-10`
physical Newton stopping criterion remains unchanged across solver choices.

```sh
python examples/diffusion/allen-cahn/benchmark_four_way.py \
    --upstream /path/to/clean/upstream-checkout \
    --current /path/to/hierarchical-triangle-element-checkout \
    --linear-solver cg-jacobi --output dist/allen-cahn-four-way-node-cg-p3
```

### Euler CUDA run

See [`cluster/README.md`](../../../cluster/README.md) for the existing
GPU-backed JupyterLab-terminal workflow and pull/run commands.
`--device cuda --linear-solver torch-cg-jacobi` selects the documented
torch-sla PyTorch CG/Jacobi backend; no solver is reimplemented. Timers
synchronize CUDA, and CPU diagnostic transfers remain explicit. CG iteration
counts are unavailable from this API and recorded as `null`.
`--prepare-only` saves inputs locally; `--input FILE.npz` loads them on Euler
without running Gmsh. Start with P4, then P5/P6; P8 is an optional
high-order accuracy/stress case and must pass the same numerical checks.
