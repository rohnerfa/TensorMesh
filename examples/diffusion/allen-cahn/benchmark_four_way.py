"""Process-isolated Allen-Cahn: upstream nodal/current hierarchical x Element/Mixed.

Example (from the current checkout):
    python examples/diffusion/allen-cahn/benchmark_four_way.py \
        --upstream ../baseline --current ../triangle-element --allow-dirty-current

All workers receive identical mesh, initial values and external quadrature.
Only Triangle.get_quadrature is adapted, on BOTH checkouts; no assembly or
solver implementation is patched. Both current cases REQUIRE hierarchical
fields. Element/Node and Mixed are compared without cross-API residual
fallbacks. CPU sparse LU/SciPy CG and torch-sla PyTorch CG (CPU/CUDA) are available.
"""
import argparse
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
from scipy import sparse
from scipy.special import roots_jacobi


ROOT = Path(__file__).resolve().parents[3]
PINNED_UPSTREAM = "2d544e8bc7ee8de412ffbc768b313875c3b12895"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def source_info(root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((root / "tensormesh").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"root": str(root), "commit": git(root, "rev-parse", "HEAD"),
            "library_changes": git(root, "status", "--porcelain", "--", "tensormesh"),
            "python_source_sha256": digest.hexdigest()}


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def triangle_rule(degree):
    n = (degree + 2) // 2
    xu, wu = roots_jacobi(n, 1, 0)
    xv, wv = roots_jacobi(n, 0, 0)
    u, v = np.meshgrid((xu + 1) / 2, (xv + 1) / 2, indexing="ij")
    return (np.outer(wu / 4, wv / 2).reshape(-1),
            np.stack((u, (1 - u) * v), -1).reshape(-1, 2))


def install_quadrature(data, degree):
    import torch
    from tensormesh.element import Triangle

    @lru_cache(None)
    def external_rule(cls, order=1, dtype=torch.float32, device=torch.device("cpu")):
        if order != degree:
            raise ValueError(f"unexpected quadrature request: {order}, expected {degree}")
        return (torch.tensor(data["weights"], dtype=dtype, device=device),
                torch.tensor(data["quadrature_points"], dtype=dtype, device=device))

    Triangle.get_quadrature = classmethod(external_rule)


def check_imports(root):
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tensormesh" or name.startswith("tensormesh."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root / "tensormesh"):
                    raise RuntimeError(f"contaminated import: {name}: {path}")
                paths[name] = str(path)
    if "tensormesh" not in paths:
        raise RuntimeError("TensorMesh was not imported")
    return paths


def timed(fn, device=None):
    if device is not None and device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)
    start = perf_counter()
    result = fn()
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
    return result, perf_counter() - start


def solve_increment(matrix, rhs, options):
    """Solve a Newton system; include format conversion and preconditioner setup.

    LU uses the original torch-sla entry point, explicitly pinned to SciPy.
    CG uses SciPy directly to expose callback iteration counts and tolerance
    semantics without relying on torch-sla's legacy SparseTensor.solve API.
    """
    import torch

    if options["method"] == "torch-cg-jacobi":
        # Existing documented torch-sla PyTorch backend, not a benchmark CG implementation.
        # Supply the combined requested tolerance as atol and verify it ourselves;
        # some torch-sla releases do not expose a configurable backend rtol.
        from torch_sla import spsolve
        rhs_norm = float(torch.linalg.vector_norm(rhs))
        limit = max(options["atol"], options["rtol"] * rhs_norm)
        x = spsolve(matrix.values, matrix.row_indices, matrix.col_indices, tuple(matrix.shape),
                    rhs, backend="pytorch", method="cg", preconditioner="jacobi",
                    atol=limit, maxiter=options["maxiter"])
        if x.device != rhs.device:
            raise RuntimeError("solver silently changed device")
        norm = float(torch.linalg.vector_norm(matrix @ x - rhs))
        if not np.isfinite(norm) or norm > 1.1 * limit + 1e-15:
            raise RuntimeError(f"torch-sla CG true residual {norm} exceeds target {limit}")
        return x, {"method": "cg", "backend": "torch-sla/pytorch", "preconditioner": "jacobi",
                   "iterations": None, "iteration_count_note": "not exposed by spsolve",
                   "converged": True, "residual_norm": norm, "rhs_norm": rhs_norm}
    if rhs.device.type != "cpu":
        raise ValueError("SciPy solvers are CPU-only; use --linear-solver torch-cg-jacobi for CUDA")
    if options["method"] == "lu":
        x = matrix.solve(rhs, backend="scipy", method="lu")
        defect = (matrix @ x - rhs).numpy()
        return x, {"method": "lu", "backend": "scipy via torch-sla", "iterations": 0,
                   "converged": True, "residual_norm": float(np.linalg.norm(defect)),
                   "rhs_norm": float(np.linalg.norm(rhs.numpy()))}
    if options["method"] != "cg-jacobi":
        raise ValueError(f"unknown solver: {options['method']}")
    cpu_matrix = sparse.coo_matrix(
        (matrix.values.numpy(), (matrix.row_indices.numpy(), matrix.col_indices.numpy())),
        shape=tuple(matrix.shape)).tocsr()
    diagonal = cpu_matrix.diagonal()
    if not np.isfinite(diagonal).all() or (diagonal <= 0).any():
        raise RuntimeError("CG/Jacobi requires finite, positive diagonal entries")
    preconditioner = sparse.linalg.LinearOperator(cpu_matrix.shape, dtype=np.float64,
                                                matvec=lambda v: v / diagonal)
    iterations = 0

    def callback(x):
        nonlocal iterations
        iterations += 1

    x, status = sparse.linalg.cg(
        cpu_matrix, rhs.numpy(), M=preconditioner, rtol=options["rtol"],
        atol=options["atol"], maxiter=options["maxiter"], callback=callback)
    norm = float(np.linalg.norm(cpu_matrix @ x - rhs.numpy()))
    rhs_norm = float(np.linalg.norm(rhs.numpy()))
    limit = max(options["atol"], options["rtol"] * rhs_norm)
    # Check the true (not just recursively updated) algebraic residual.
    if status != 0 or not np.isfinite(norm) or norm > 1.1 * limit + 1e-15:
        raise RuntimeError(f"CG failed: status={status}, iterations={iterations}, "
                           f"true residual={norm}, target={limit}")
    return torch.tensor(x), {"method": "cg", "backend": "scipy", "preconditioner": "jacobi",
                             "iterations": iterations, "converged": True,
                             "residual_norm": norm, "rhs_norm": rhs_norm}


def modal_to_nodal(assembler, kind):
    """Sparse T: nodal physical samples = T @ global modal coefficients.

    Used for validation and basis-invariant Newton stopping, not to assemble
    matrices or residuals. Shared nodal samples must have consistent traces.
    """
    import torch
    from tensormesh.element import Triangle

    if hasattr(assembler, "field_conn"):
        order = assembler.layout.field("u").order
        conn = assembler.field_conn[f"u__{kind}"].numpy()
        signs = assembler.field_sign[f"u__{kind}"].numpy()
        n_dofs = assembler.n_dofs
    else:
        order = assembler._hierarchical.order
        conn = assembler._hierarchical.blocks[kind].conn.numpy()
        signs = assembler._hierarchical.blocks[kind].sign.numpy()
        n_dofs = assembler._hierarchical.n_dofs
    points = Triangle.get_basis(order, torch.float64)
    values = Triangle.get_hierarchical_basis_fns(order, torch.float64).map(points).numpy()
    cells = assembler.elements[kind].numpy()
    local = values[None] * signs[:, None, :]
    rows = np.broadcast_to(cells[:, :, None], local.shape).ravel()
    cols = np.broadcast_to(conn[:, None, :], local.shape).ravel()
    counts = np.bincount(cells.ravel(), minlength=assembler.n_points)
    if (counts == 0).any():
        raise RuntimeError("mesh has unused points")
    shape = (assembler.n_points, n_dofs)
    transform = sparse.coo_matrix((local.ravel() / counts[rows], (rows, cols)), shape=shape).tocsr()
    square_sum = sparse.coo_matrix((local.ravel() ** 2, (rows, cols)), shape=shape).tocsr()
    variance = square_sum - sparse.diags(counts) @ transform.multiply(transform)
    if np.max(np.abs(variance.data), initial=0) > 2e-11:
        raise RuntimeError("inconsistent hierarchical traces at shared nodes")
    transform.eliminate_zeros()
    return transform


def run_worker(job):
    root = Path(job["source"]["root"])
    sys.path.insert(0, str(root))
    import meshio
    import torch
    import tensormesh as tm

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(job["threads"])
    device = torch.device(job.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run inside a GPU-backed Euler Jupyter session")
    if source_info(root) != job["source"]:
        raise RuntimeError("library checkout changed after inputs were prepared")
    params = job["parameters"]
    hierarchical = job["basis"] == "hierarchical"
    if hierarchical != job["label"].startswith("current-"):
        raise RuntimeError("current must be hierarchical and upstream must be nodal")
    with np.load(job["input"]) as archive:
        data = dict(archive)
    degree = 4 * params["order"]
    install_quadrature(data, degree)
    start = perf_counter()
    kind = f"triangle{data['cells'].shape[1]}"
    mesh = tm.Mesh(meshio.Mesh(data["points"], [(kind, data["cells"])]), reorder=False).double()
    mesh_seconds = perf_counter() - start

    def initialize(self, dt, epsilon):
        self.dt, self.epsilon = dt, epsilon

    def jacobian(self, c, u, v, gradu, gradv):
        return (u * v / self.dt + (gradu @ gradv)
                + 3.0 * self.epsilon ** 2 * c ** 2 * u * v)

    def residual(self, v, gradv, c, gradc, cold):
        return ((c - cold) / self.dt * v + gradc @ gradv
                + self.epsilon ** 2 * c ** 3 * v - self.epsilon ** 2 * cold * v)

    start = perf_counter()
    kwargs = dict(dt=params["dt"], epsilon=params["epsilon"], quadrature_order=degree)
    mixed_asm = None
    if job["api"] == "mixed":
        field = tm.Field("u", "v", params["order"],
                         **({"basis": "hierarchical"} if hierarchical else {}))
        mixed_cls = type("AllenCahnMixed", (tm.MixedElementAssembler,), {
            "fields": [field], "__post_init__": initialize,
            "forward": jacobian, "forward_vector": residual})
        mixed_asm = mixed_cls.from_mesh(mesh, **kwargs)
        if hierarchical:
            if mixed_asm.field_kind["u"] != "hierarchical":
                raise RuntimeError("Mixed assembler silently selected a nonhierarchical field")
        elif not torch.equal(mixed_asm.layout.node_ids("u"), torch.arange(mesh.n_points)):
            raise RuntimeError("mixed coefficients are not in mesh-node order")

    def field_data(point_data):
        return {key: ("u", point_data[key]) for key in ("c", "cold")}

    if job["api"] == "element":
        matrix_cls = type("AllenCahnJacobian", (tm.ElementAssembler,),
                          {"__post_init__": initialize, "forward": jacobian})
        matrix_asm = matrix_cls.from_mesh(
            mesh, **kwargs, **({"basis": "hierarchical", "field_order": params["order"]}
                              if hierarchical else {}))
        residual_cls = type("AllenCahnResidual", (tm.NodeAssembler,),
                            {"__post_init__": initialize, "forward": residual})
        residual_asm = residual_cls.from_mesh(
            mesh, **kwargs, **({"basis": "hierarchical", "field_order": params["order"]}
                              if hierarchical else {}))
        if hierarchical:
            if matrix_asm._hierarchical is None or residual_asm._hierarchical is None:
                raise RuntimeError("Element/Node assembler silently selected a nonhierarchical field")
            block = matrix_asm._hierarchical.blocks[kind]
            node_block = residual_asm._hierarchical.blocks[kind]
            if (not torch.equal(block.conn, node_block.conn)
                    or not torch.equal(block.sign, node_block.sign)):
                raise RuntimeError("Element and Node modal coefficient layouts differ")
            layout = matrix_asm.field_layout()
            # Element's point_data API expects geometry-node values, not modal
            # coefficients. Include this production conversion in matrix timing.
            matrix_fn = lambda pd: matrix_asm(
                mesh.points, point_data={"c": layout.prolong("u", pd["c"])}, batch_size=-1)
            residual_fn = lambda pd: residual_asm(
                mesh.points, point_data={key: layout.prolong("u", pd[key])
                                         for key in ("c", "cold")}, batch_size=1)
        else:
            matrix_fn = lambda pd: matrix_asm(mesh.points, point_data=pd, batch_size=-1)
            residual_fn = lambda pd: residual_asm(mesh.points, point_data=pd, batch_size=1)
    else:
        matrix_asm = mixed_asm
        residual_asm = matrix_asm
        layout = mixed_asm.layout
        if hierarchical:
            matrix_fn = lambda pd: matrix_asm(mesh.points, field_data=field_data(pd), batch_size=-1)
            residual_fn = lambda pd: matrix_asm.assemble_vector(
                mesh.points, field_data=field_data(pd), batch_size=1)
        else:
            matrix_fn = lambda pd: matrix_asm(mesh.points, point_data=pd, batch_size=-1)
            residual_fn = lambda pd: matrix_asm.assemble_vector(
                mesh.points, point_data=pd, batch_size=1)
    setup_seconds = perf_counter() - start
    for asm in (matrix_asm, residual_asm):
        w, q = asm.transformation[kind].quadrature
        if array_hash(w.numpy(), q.numpy()) != array_hash(data["weights"], data["quadrature_points"]):
            raise RuntimeError("installed quadrature does not match shared input")
    transform = factor = None
    start = perf_counter()
    if hierarchical:
        transform = modal_to_nodal(matrix_asm, kind)
        factor = sparse.linalg.splu(transform.tocsc())
        cold = torch.tensor(factor.solve(data["initial"]))
        np.testing.assert_allclose(layout.prolong("u", cold).numpy(), data["initial"],
                                   atol=2e-12, rtol=2e-12)
    else:
        cold = torch.tensor(data["initial"])
    basis_setup_seconds = perf_counter() - start
    start = perf_counter()
    mesh = mesh.to(device=device)
    matrix_asm.to(device)
    if residual_asm is not matrix_asm:
        residual_asm.to(device)
    cold = cold.to(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        resident_bytes = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        resident_bytes = 0
    transfer_seconds = perf_counter() - start
    history = [data["initial"].copy()]
    iteration_log = []
    steps = []
    initial_matrix = initial_residual = None
    totals = {key: 0.0 for key in ("matrix_assembly", "residual_assembly", "linear_solve", "update_and_norm")}
    with torch.no_grad():
        start = perf_counter()
        for step in range(params["steps"]):
            step_start = perf_counter()
            c = cold
            for iteration in range(params["max_iter"]):
                point_data = {"c": c, "cold": cold}
                matrix, matrix_time = timed(lambda: matrix_fn(point_data), device)
                rhs, residual_time = timed(lambda: residual_fn(point_data), device)
                (increment, solve_info), solve_time = timed(
                    lambda: solve_increment(matrix, -rhs, job["solver"]), device)
                update_start = perf_counter()
                c = c + increment
                # R_H = T.T @ R_L. Use the SAME nodal-dual residual norm
                # in every case, rather than basis-dependent coefficient norms.
                norm_rhs = torch.tensor(factor.solve(rhs.detach().cpu().numpy(), trans="T")) if hierarchical else rhs
                rnorm = float(torch.linalg.norm(norm_rhs))
                update_time = perf_counter() - update_start
                values = (matrix_time, residual_time, solve_time, update_time)
                for key, value in zip(totals, values):
                    totals[key] += value
                iteration_log.append({"step": step + 1, "iteration": iteration + 1,
                                      **dict(zip(totals, values)), "residual_norm": rnorm,
                                      "linear_solver": solve_info})
                if step == iteration == 0:
                    initial_matrix = sparse.coo_matrix(
                        (matrix.values.detach().cpu().numpy().copy(),
                         (matrix.row.detach().cpu().numpy(), matrix.col.detach().cpu().numpy())),
                        shape=tuple(matrix.shape)).tocsr()
                    initial_residual = rhs.detach().cpu().numpy().copy()
                if not np.isfinite(rnorm):
                    raise RuntimeError("nonfinite residual")
                if rnorm < 1e-10:
                    break
            else:
                raise RuntimeError(f"Newton did not converge at step {step + 1}")
            cold = c
            cpu_c = c.detach().cpu().numpy()
            history.append(transform @ cpu_c if hierarchical else cpu_c.copy())
            steps.append({"step": step + 1, "iterations": iteration + 1,
                          "residual_norm": rnorm, "wall_seconds": perf_counter() - step_start})
        stepping_seconds = perf_counter() - start
    prefix = job["artifact"]
    if source_info(root) != job["source"]:
        raise RuntimeError("library checkout changed while the benchmark was running")
    sparse.save_npz(prefix + "-initial-matrix.npz", initial_matrix)
    np.savez(prefix + "-fields.npz", history=np.array(history), initial_residual=initial_residual)
    if hierarchical:
        sparse.save_npz(prefix + "-modal-to-nodal.npz", transform)
    return {"label": job["label"], "api": job["api"], "trial": job["trial"], "basis": job["basis"],
            "residual_api": "MixedElementAssembler" if job["api"] == "mixed" else "NodeAssembler",
            "solver": job["solver"],
            "jacobian_coefficient_conversion_in_timing": hierarchical and job["api"] == "element",
            "residual_coefficient_conversion_in_timing": hierarchical and job["api"] == "element",
            "source": job["source"], "imported_modules": check_imports(root),
            "environment": {"torch": torch.__version__, "python": platform.python_version(),
                            "torch_sla": version("torch-sla"), "threads": torch.get_num_threads(),
                             "dtype": "float64", "device": str(device), "cuda_runtime": torch.version.cuda,
                             "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None},
            "cuda_memory": ({"resident_bytes": resident_bytes,
                             "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                             "incremental_peak_bytes": torch.cuda.max_memory_allocated(device) - resident_bytes,
                             "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
                            if device.type == "cuda" else None),
            "input_sha256": array_hash(data["points"], data["cells"], data["initial"]),
            "quadrature_sha256": array_hash(data["weights"], data["quadrature_points"]),
            "dofs": transform.shape[1] if hierarchical else mesh.n_points, "triangles": mesh.n_elements,
            "setup": {"mesh_seconds": mesh_seconds, "assembler_seconds": setup_seconds,
                      "device_transfer_seconds": transfer_seconds,
                      "diagnostic_basis_map_and_initial_conversion_seconds": basis_setup_seconds},
            "timings": {**totals, "stepping_wall_seconds": stepping_seconds,
                        "setup_and_stepping_seconds": mesh_seconds + setup_seconds + transfer_seconds + stepping_seconds,
                        "including_basis_setup_seconds": mesh_seconds + setup_seconds + basis_setup_seconds
                                                         + transfer_seconds + stepping_seconds},
            "steps": steps, "iterations": iteration_log, "artifact": prefix}


def relative_error(actual, expected):
    if sparse.issparse(actual):
        return float(sparse.linalg.norm(actual - expected) / max(sparse.linalg.norm(expected), 1e-30))
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def compare_cases(cases):
    reference = cases[0]
    reference_matrix = sparse.load_npz(reference["artifact"] + "-initial-matrix.npz")
    with np.load(reference["artifact"] + "-fields.npz") as archive:
        reference_fields = dict(archive)
    checks = []
    for case in cases:
        matrix = sparse.load_npz(case["artifact"] + "-initial-matrix.npz")
        with np.load(case["artifact"] + "-fields.npz") as archive:
            fields = dict(archive)
        expected_matrix = reference_matrix
        expected_residual = reference_fields["initial_residual"]
        if case.get("basis") == "hierarchical":
            transform = sparse.load_npz(case["artifact"] + "-modal-to-nodal.npz")
            expected_matrix = transform.T @ reference_matrix @ transform
            expected_residual = transform.T @ expected_residual
        if fields["history"].shape != reference_fields["history"].shape:
            raise RuntimeError("physical-field histories have different shapes")
        errors = {"initial_jacobian_relative_error": relative_error(matrix, expected_matrix),
                  "initial_residual_relative_error": relative_error(
                      fields["initial_residual"], expected_residual),
                  "max_step_relative_field_error": max(relative_error(a, b) for a, b in zip(
                      fields["history"], reference_fields["history"]))}
        same_inputs = all(case[key] == reference[key] for key in
                          ("input_sha256", "quadrature_sha256", "dofs", "triangles"))
        checks.append({"label": case["label"], "trial": case["trial"], **errors,
                       "same_inputs": same_inputs,
                       "passed": same_inputs and all(np.isfinite(v) and v < 1e-8 for v in errors.values())})
    return checks


def check_basis_selection(cases):
    for case in cases:
        expected = "hierarchical" if case["label"].startswith("current-") else "lagrange"
        if case.get("basis") != expected:
            raise RuntimeError(f"{case['label']} must use {expected}, not {case.get('basis')}")


def coordinate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sources = {"upstream": source_info(args.upstream), "current": source_info(args.current)}
    if sources["upstream"]["library_changes"]:
        raise RuntimeError("upstream library source tree must be clean")
    if sources["current"]["library_changes"] and not args.allow_dirty_current:
        raise RuntimeError("current library is modified; explicitly pass --allow-dirty-current")
    if sources["upstream"]["commit"] != args.upstream_commit:
        raise RuntimeError(f"upstream checkout must match pinned commit {args.upstream_commit}")
    sys.path.insert(0, str(ROOT))
    import torch
    from tensormesh import Mesh
    from tensormesh.dataset import PoissonMultiFrequency

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not args.prepare_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; CPU fallback is not allowed")
    if args.device == "cuda" and args.linear_solver != "torch-cg-jacobi":
        raise ValueError("CUDA requires --linear-solver torch-cg-jacobi")
    if args.input:
        input_path = args.input.resolve()
        with np.load(input_path) as archive:
            if int(archive["order"]) != args.order or not np.isclose(
                    float(archive["chara_length"]), args.chara_length):
                raise ValueError("prepared mesh order/length does not match CLI")
            w, q = archive["weights"], archive["quadrature_points"]
    else:
        mesh = Mesh.gen_rectangle(chara_length=args.chara_length, order=args.order, element_type="tri").double()
        generator = torch.Generator().manual_seed(0)
        a = torch.empty(24, 24).uniform_(-1, 1, generator=generator)
        initial = PoissonMultiFrequency(a=a, r=1).source_term(mesh.points)
        w, q = triangle_rule(4 * args.order)
        input_path = output / "input.npz"
        np.savez(input_path, points=mesh.points.numpy(), cells=mesh.elements().numpy(),
                 initial=initial.numpy(), weights=w, quadrature_points=q,
                 order=args.order, chara_length=args.chara_length)
    parameters = {"order": args.order, "chara_length": args.chara_length,
                  "steps": args.steps, "max_iter": 50, "dt": 5e-5, "epsilon": 220,
                  "seed": 0, "n_freq": 24}
    solver = {"method": args.linear_solver, "rtol": args.linear_rtol,
              "atol": args.linear_atol, "maxiter": args.linear_maxiter}
    report = {"status": "running", "sources": sources, "parameters": parameters,
              "solver": solver,
              "protocol": {"basis": {"upstream": "lagrange", "current": "hierarchical"},
                           "current_element_residual": "hierarchical NodeAssembler",
                           "current_element_point_data": "modal-to-nodal conversion inside Jacobian and residual timing",
                           "linear_solve_timing": "includes input conversion, per-call Jacobi setup (CG) and true residual verification; zero initial guess and no reuse",
                           "newton_norm": "nodal-dual residual norm in every case; modal dual conversion in update_and_norm",
                           "basis_setup": "diagnostic sparse change of basis and LU excluded from setup_and_stepping_seconds; also reported inclusively",
                           "threads": args.threads, "device": args.device,
                           "cuda_timing": "synchronize before/after each timed operation",
                           "host_diagnostics": "nodal-dual stopping norm and field validation stay on CPU; transfers included in stepping wall time, not assembly/solve",
                           "matrix_quadrature_batch_size": -1, "residual_quadrature_batch_size": 1,
                           "quadrature_degree": 4 * args.order, "quadrature_points": len(w),
                           "adapter": "identical external Triangle.get_quadrature in both checkouts",
                           "warmup": "none; each trial uses a fresh process and assembler caches",
                           "repeats": args.repeats, "video_and_process_startup_in_timing": False,
                           "initial_matrix_snapshot_in_stepping_wall_time": True}, "cases": []}
    report_path = output / "report.json"
    if args.prepare_only:
        report.update(status="prepared", input=str(input_path))
        write_json(report_path, report)
        print(f"Prepared input: {input_path}")
        return
    write_json(report_path, report)
    combinations = [(source, api) for source in sources for api in ("element", "mixed")]
    for trial in range(args.repeats):
        # Alternate forward/reverse execution to reduce systematic order bias.
        sequence = combinations if trial % 2 == 0 else list(reversed(combinations))
        for source, api in sequence:
            label = f"{source}-{api}"
            prefix = output / f"{label}-trial{trial + 1}"
            job = {"source": sources[source], "api": api, "label": label, "trial": trial + 1,
                   "solver": solver, "device": args.device,
                   "basis": "hierarchical" if source == "current" else "lagrange",
                   "parameters": parameters, "input": str(input_path), "threads": args.threads,
                   "artifact": str(prefix)}
            job_path = prefix.with_suffix(".job.json")
            result_path = prefix.with_suffix(".result.json")
            write_json(job_path, job)
            print(f"Running {label}, trial {trial + 1}/{args.repeats}", flush=True)
            env = dict(os.environ, OMP_NUM_THREADS=str(args.threads),
                       OPENBLAS_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
            with prefix.with_suffix(".log").open("w") as log:
                completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                            "--worker", str(job_path), "--result", str(result_path)],
                                           cwd=sources[source]["root"], env=env, stdout=log, stderr=log)
            if completed.returncode:
                report["status"] = "failed"
                report["failed_worker_log"] = str(prefix.with_suffix(".log"))
                write_json(report_path, report)
                raise RuntimeError(prefix.with_suffix(".log").read_text())
            case = json.loads(result_path.read_text())
            report["cases"].append(case)
            write_json(report_path, report)
            print(f"  stepping: {case['timings']['stepping_wall_seconds']:.3f} s; "
                  f"Newton iterations: {sum(s['iterations'] for s in case['steps'])}", flush=True)
    check_basis_selection(report["cases"])
    report["validation"] = compare_cases(report["cases"])
    report["summary"] = {}
    for source, api in combinations:
        label = f"{source}-{api}"
        cases = [case for case in report["cases"] if case["label"] == label]
        report["summary"][label] = {
            key: {"median_seconds": statistics.median(case["timings"][key] for case in cases),
                  "samples_seconds": [case["timings"][key] for case in cases]}
            for key in cases[0]["timings"]}
    report["status"] = "completed" if all(c["passed"] for c in report["validation"]) else "validation_failed"
    write_json(report_path, report)
    print(json.dumps({"status": report["status"], "summary": report["summary"]}, indent=2))
    if report["status"] != "completed":
        raise RuntimeError("four-way numerical comparison failed")
    print(f"Report: {report_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--upstream", type=Path)
    parser.add_argument("--upstream-commit", default=PINNED_UPSTREAM)
    parser.add_argument("--current", type=Path, default=ROOT,
                        help="current hierarchical implementation worktree (default: this checkout)")
    parser.add_argument("--allow-dirty-current", action="store_true",
                        help="record and benchmark uncommitted current library sources without modifying them")
    parser.add_argument("--order", type=int, choices=range(2, 9), default=3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--input", type=Path, help="prepared NPZ input; avoids Gmsh on the cluster")
    parser.add_argument("--prepare-only", action="store_true", help="save inputs without running workers")
    parser.add_argument("--chara-length", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--linear-solver", choices=("lu", "cg-jacobi", "torch-cg-jacobi"), default="lu")
    parser.add_argument("--linear-rtol", type=float, default=1e-10)
    parser.add_argument("--linear-atol", type=float, default=1e-14)
    parser.add_argument("--linear-maxiter", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("dist/allen-cahn-four-way-hierarchical"))
    args = parser.parse_args()
    if args.worker:
        write_json(args.result, run_worker(json.loads(args.worker.read_text())))
        return
    if args.upstream is None:
        parser.error("--upstream is required")
    if min(args.steps, args.threads, args.repeats, args.chara_length) <= 0:
        parser.error("steps, threads, repeats and chara-length must be positive")
    if args.linear_rtol <= 0 or args.linear_atol < 0 or args.linear_maxiter < 1:
        parser.error("invalid linear solver tolerances or iteration limit")
    coordinate(args)


if __name__ == "__main__":
    main()
