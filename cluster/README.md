# Allen–Cahn four-way CUDA test on Euler

This uses the existing benchmark and **torch-sla's documented PyTorch CG +
Jacobi backend** (`spsolve(..., backend="pytorch", method="cg",
preconditioner="jacobi")`). No new solver, library changes, cuDSS, compiled
extensions, or scheduler submission scripts are needed.

The current source is the refreshed `perf/hierarchical-tetrahedron-assembly`
branch. Upstream is pinned to `2d544e8bc7ee8de412ffbc768b313875c3b12895`.
Both current cases use hierarchical fields; Element uses hierarchical Node
for the residual. Both upstream cases use nodal Lagrange.

## Euler environment

Use the JupyterLab-terminal workflow: start a **GPU-backed** session at
<https://jupyter.euler.hpc.ethz.ch/>, then
open a terminal. JupyterHub already owns the allocation: do **not** run
`sbatch`, `srun`, or request another allocation from that terminal. Do not
benchmark on a login node. Stop the server through Hub Control Panel when
finished; closing the browser does not release it.

First inspect, rather than replacing, the supplied environment:

```sh
module list
nvidia-smi
python -c 'import sys, torch; print(sys.executable, torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

Keep working CUDA-enabled PyTorch. If unavailable, the existing Euler guide
lists `stack/2024-06` + `python_cuda/3.11.6`, with `eth_proxy` for downloads.
Verify availability/compatibility in the session; do not mix unrelated stacks.
Use Python 3.11+ and a CUDA-enabled PyTorch. A100 is preferable for float64
if available; the script records the actual GPU model. CUDA is not validated
by the local macOS checks.

## Checkouts and installation (in the GPU session)

Use a **fresh directory**, not an old extracted bundle. Example:

```sh
mkdir -p /cluster/home/rohnerfa/tensormesh-allen-cahn-gpu
cd /cluster/home/rohnerfa/tensormesh-allen-cahn-gpu
git clone --branch perf/hierarchical-tetrahedron-assembly \
  https://github.com/rohnerfa/TensorMesh.git current
git -C current worktree add --detach ../baseline 2d544e8bc7ee8de412ffbc768b313875c3b12895
cd current
python -m venv --system-site-packages .venv
source .venv/bin/activate
module load eth_proxy  # if package downloads require it in this session
python -m pip install -e '.[test]'
python -m pip check
```

Keep the same checked CUDA/Python stack active. This does not change `main`.
The harness, runner, and prepared P4/P5/P6/P8 inputs are all committed on the
performance branch: **no rsync or local mesh generation is needed**.

If you already cloned the performance branch on Euler, update it from its
root instead of cloning again:

```sh
git switch perf/hierarchical-tetrahedron-assembly
git pull --ff-only origin perf/hierarchical-tetrahedron-assembly
git worktree list
# Only if ../baseline does not already exist:
git worktree add --detach ../baseline 2d544e8bc7ee8de412ffbc768b313875c3b12895
source .venv/bin/activate
```

If Git reports local changes, preserve/investigate them rather than resetting
or overwriting them. The runner expects a sibling `../baseline` checkout;
pass `--upstream /actual/path` if yours is elsewhere.

## Included inputs

The compressed `cluster/inputs/p{4,5,6,8}/input.npz` files contain the high-order
mesh (`h=0.05`), initial physical field, and common degree-`4*p` quadrature.
They avoid running Gmsh or requiring system `libGLU` on Euler.

To regenerate a different mesh locally (optional, not part of the Euler setup):

```sh
python examples/diffusion/allen-cahn/benchmark_four_way.py \
  --upstream ../baseline --order 4 --chara-length 0.05 \
  --prepare-only --output dist/new-input-p4
```

## Smoke, then P4–P6

Inside the GPU-backed terminal, from `current` with `.venv` active:

```sh
# Small four-way CUDA smoke: P4, h=0.05, two steps, one trial.
bash cluster/run_allen_cahn_gpu.sh

# Same physics as the local comparison: 10 steps, three trials per case.
for p in 4 5 6; do
  bash cluster/run_allen_cahn_gpu.sh \
    --order "$p" --input "cluster/inputs/p$p/input.npz" \
    --steps 10 --repeats 3 --output "dist/euler-allen-cahn-cuda-p$p" \
    > "allen-cahn-cuda-p$p.log" 2>&1 || break
done
```

Start with P4; increase only after its `report.json` says `completed`.
P8 is an **optional stress/accuracy test**, not a promised valid nodal reference:

```sh
bash cluster/run_allen_cahn_gpu.sh \
  --order 8 --input cluster/inputs/p8/input.npz --steps 2 --repeats 1 \
  --output dist/euler-allen-cahn-cuda-p8
```

The existing high-order nodal basis may lose accuracy. Failed change-of-basis
or field checks must not be presented as successful performance comparisons.
If memory is tight, stop and reduce mesh size (increase `h`, then regenerate
inputs); do not silently change batching for only one API or basis.

## What is measured

- Four fresh-process cases/trial: upstream/current × Element/Mixed.
- Float64, common mesh/source/quadrature, identical physics and batching.
- Existing torch-sla PyTorch CG/Jacobi on the requested device, with no CPU
  solve fallback. True linear residuals are independently checked.
- CUDA synchronization before/after each operation timer; separate setup and
  device transfer times; GPU allocator peak and resident memory recorded.
- The physical Newton stopping norm and field/matrix validation remain CPU
  diagnostics. Their transfers are included in total stepping time, not in
  Jacobian/residual/solve timings. This is not a transfer-free GPU application.
- The installed torch-sla `spsolve` does not expose CG iteration counts; these
  are `null`, **not zero iterations**. Backend, version and residuals are saved.
- `report.json` must pass the matrix/load change-of-basis and physical-field
  checks. Keep the report, per-worker logs, and `nvidia-smi` output.

To sanity-check the same backend on CPU, rerun a prepared input with
`--device cpu --linear-solver torch-cg-jacobi` and a different output directory.
Do not compare these GPU totals directly with the earlier SciPy CG totals
without noting the different backend and synchronization/transfer overhead.

Solver documentation: <https://github.com/walkerchi/torch-sla#cuda-solve>.
