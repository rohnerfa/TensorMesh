"""Separated triangle-basis timings and numerical acceptance diagnostics.

CPU float64 is the default. CUDA timing/memory is optional; no MPS float64
or GPU memory measurements are fabricated when CUDA is unavailable.
This deliberately uses private assembler stages to isolate integration
from orientation and ReduceProjector; reconstruction is checked against
the public assembler before any timings are reported.
"""

import argparse
import ast
import hashlib
import inspect
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import textwrap
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import meshio
import scipy
import scipy.sparse.linalg
import torch

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.element import Triangle, HierarchicalTriangleBasisFns


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(fn, device, repeats, warmup):
    """Warm medians/spread; CUDA allocation high-water from a separate call."""
    for _ in range(warmup):
        result = fn()
        synchronize(device)
        del result
    samples = []
    for _ in range(repeats):
        synchronize(device)
        start = perf_counter()
        result = fn()
        synchronize(device)
        samples.append(perf_counter() - start)
        del result
    memory = {"status": "not_measured", "reason": "CPU run; peak GPU memory requires CUDA"}
    if device.type == "cuda":
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = fn()
        synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        memory = {"status": "measured", "allocator": "torch.cuda allocated bytes (not driver/RSS)",
                  "baseline_bytes": baseline, "peak_allocated_bytes": peak,
                  "incremental_peak_bytes": peak - baseline}
        del result
    return {"median_seconds": statistics.median(samples), "min_seconds": min(samples),
            "max_seconds": max(samples), "samples_seconds": samples, "gpu_memory": memory}


def cold_basis(p, family, device):
    if family == "hierarchical":
        basis = HierarchicalTriangleBasisFns(p, torch.float64)
    else:
        # Bypass the basis-function lru_cache, not reusable node/exponent
        # templates. In particular every call recomputes the inverse/pinv.
        factory = Triangle.get_basis_fns.__func__.__wrapped__
        basis = factory(Triangle, p, torch.float64, torch.device("cpu"))
    gradient = basis.grad()
    # Use identical CPU construction + transfer policy for both families.
    return basis.to(device), gradient.to(device)


def make_assembler(mesh, p, family, degree):
    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", p, basis=family)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            return 2 * math.pi ** 2 * torch.sin(math.pi * x[0]) * torch.sin(math.pi * x[1]) * v

    assembler = Poisson.from_mesh(mesh, quadrature_order=degree)
    # Materialize reusable geometry VALUE tables on CPU as well as the
    # field tables. Otherwise the first GPU load pass can invoke upstream's
    # nodal-basis constructor on CUDA instead of reusing transferred data.
    for transformation in assembler.transformation.values():
        _ = transformation.shape_val
    return assembler


def integration_stage(asm, batch_size):
    """Production _batches/_run_pass/_integrate_pair, excluding projection."""
    field = asm.fields[0]
    params = asm._classify_params(asm.forward, {"x": asm._points}, {}, {}, {})
    zeros, _, needs_val, needs_grad = asm._pass_context(params, {}, asm.dtype, asm.device)
    local = {}
    checked = False
    for kind, jxw, values, gradients, data in asm._batches(
            params, {"x": asm._points}, {}, {}, {}, needs_val, needs_grad, batch_size):
        if not checked:
            asm._check_bilinear(asm.forward, params, data, asm.dtype, asm.device)
            checked = True
        out, has_e = asm._run_pass(asm.forward, params, field, field, values, gradients,
                                   data, zeros, {"trial": None, "test": None})
        contribution = asm._integrate_pair(out, jxw, has_e)
        local[kind] = contribution if kind not in local else local[kind] + contribution
    return local


def orientation_stage(asm, local):
    field = asm.fields[0]
    return {kind: asm._orient_contribution(values, kind, field, field)
            for kind, values in local.items()}


def projection_stage(asm, local):
    total = None
    for kind, values in local.items():
        projected = asm.pair_projector[f"u__u__{kind}"](values)
        total = projected if total is None else total + projected
    return total


def cell_loop_iterators(source):
    tree = ast.parse(textwrap.dedent(source))
    suspicious = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.comprehension)):
            names = {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}
            attributes = {n.attr for n in ast.walk(node.iter) if isinstance(n, ast.Attribute)}
            if (names & {"elements", "cells", "conn", "connectivity", "n_element", "n_elements", "n_cell", "n_cells"}
                    or attributes & {"elements", "cells", "field_conn"}):
                suspicious.append(ast.unparse(node.iter))
    return suspicious


def audit_hotpath(asm, batch_size):
    methods = [MixedElementAssembler._batches, MixedElementAssembler._run_pass,
               MixedElementAssembler.__call__, MixedElementAssembler.assemble_vector]
    direct_cell_loops = []
    for method in methods:
        direct_cell_loops.extend(f"{method.__name__}: {iterator}"
                                 for iterator in cell_loop_iterators(inspect.getsource(method)))
    with patch.object(HierarchicalTriangleBasisFns, "map",
                      side_effect=AssertionError("reference recurrence evaluated in hot path")):
        asm(batch_size=batch_size)
        asm.assemble_vector(batch_size=batch_size)
    reference_only = all(asm.ref_val[f"u__{kind}"].ndim == 2
                         and asm.ref_grad[f"u__{kind}"].ndim == 3 for kind in asm.element_types)
    oriented = all(asm.field_sign[f"u__{kind}"].shape == asm.field_conn[f"u__{kind}"].shape
                   and not asm.field_sign[f"u__{kind}"].requires_grad
                   and ((asm.field_sign[f"u__{kind}"] == 1) | (asm.field_sign[f"u__{kind}"] == -1)).all()
                   for kind in asm.element_types)
    return {"direct_cell_loops_detected": direct_cell_loops,
            "hierarchical_recurrence_blocked_during_matrix_and_load_assembly": True,
            "reference_tables_have_no_element_axis": reference_only,
            "orientation_precomputed": bool(oriented)}


def basis_diagnostics(p, family, basis):
    points = torch.rand(max(256, 4 * basis.shape[0]), 2,
                        generator=torch.Generator().manual_seed(301), dtype=torch.float64)
    points[:, 1] *= 1 - points[:, 0]
    device = next(basis.buffers()).device
    values = basis.map(points.to(device)).detach().cpu()
    n = basis.shape[0]
    coefficients = torch.zeros(n, dtype=torch.float64)
    if family == "hierarchical":
        coefficients[:3] = 1
        linear_coefficients = torch.zeros_like(coefficients)
        linear_coefficients[1] = 1
        nodal_defect = None
    else:
        coefficients[:] = 1
        nodes = Triangle.get_basis(p, torch.float64)
        linear_coefficients = nodes[:, 0]
        at_nodes = basis.map(nodes.to(device)).detach().cpu()
        nodal_defect = float((at_nodes - torch.eye(n, dtype=torch.float64)).abs().max())
    constant_error = float((values @ coefficients - 1).abs().max())
    linear_error = float((values @ linear_coefficients - points[:, 0]).abs().max())
    exponents = [(i, total - i) for total in range(p + 1) for i in range(total + 1)]
    monomials = lambda x: torch.stack([x[:, 0] ** i * x[:, 1] ** j for i, j in exponents], -1)
    fitted = torch.linalg.lstsq(values, monomials(points), driver="gelsd").solution
    check = torch.rand(113, 2, generator=torch.Generator().manual_seed(991), dtype=torch.float64)
    check[:, 1] *= 1 - check[:, 0]
    reproduction_error = float((basis.map(check.to(device)).detach().cpu() @ fitted - monomials(check)).abs().max())
    rank = int(torch.linalg.matrix_rank(values))
    valid = (constant_error < 1e-9 and linear_error < 1e-9 and rank == n
             and reproduction_error < 1e-9 and (nodal_defect is None or nodal_defect < 1e-8))
    return {"basis_count": n, "sampled_rank": rank, "constant_max_error": constant_error,
            "linear_max_error": linear_error, "Pp_reproduction_max_error": reproduction_error,
            "nodal_interpolation_defect": nodal_defect, "passes_accuracy_screen": valid}


def conditioning(matrix):
    a = matrix.to_dense().detach().cpu()
    norm = torch.linalg.vector_norm(a).clamp_min(1e-30)
    symmetry = float(torch.linalg.vector_norm(a - a.T) / norm)
    symmetric = (a + a.T) / 2
    eigenvalues = torch.linalg.eigvalsh(symmetric)
    lo, hi = float(eigenvalues[0]), float(eigenvalues[-1])
    resolved_spd = lo > hi * torch.finfo(a.dtype).eps * len(a)
    condition = hi / lo if lo > 0 else None
    scaled_condition = None
    if (a.diagonal() > 0).all():
        scale = a.diagonal().rsqrt()
        scaled_eigenvalues = torch.linalg.eigvalsh(symmetric * scale[:, None] * scale[None, :])
        if scaled_eigenvalues[0] > 0:
            scaled_condition = float(scaled_eigenvalues[-1] / scaled_eigenvalues[0])
    return {"relative_symmetry_defect": symmetry, "lambda_min": lo, "lambda_max": hi,
            "condition_2_symmetric_part": condition, "SPD_resolved_at_float64_precision": resolved_spd,
            "diagonal_scaled_condition_2": scaled_condition}


def numerical_solve(asm, matrix, load, timing, device, diagnostic_asm=None):
    condenser = Condenser(asm.layout.boundary_mask("u"))
    reduced, rhs = condenser(matrix, load)
    result = {"free_dofs": len(rhs), "conditioning": conditioning(reduced)}
    if device.type == "cpu":
        solver = lambda: reduced.solve(rhs, backend="scipy", method="lu")
        result["solver"] = "torch-sla scipy sparse LU; fixed for both families"
    else:
        solver = lambda: reduced.solve(rhs, backend="pytorch", method="cg", atol=1e-12, maxiter=10000)
        result["solver"] = "torch-sla pytorch CG atol=1e-12 maxiter=10000; fixed for both families"
    try:
        result["solve_timing"] = timing(solver)
        free = solver()
        if not torch.isfinite(free).all():
            raise RuntimeError("solver returned nonfinite coefficients")
        result["relative_linear_system_residual"] = float(
            torch.linalg.vector_norm(reduced @ free - rhs) / torch.linalg.vector_norm(rhs).clamp_min(1e-30))
        # Numerical diagnostics use CPU float64 for either timed backend,
        # just like the eigenspectrum above; these transfers are NOT timed.
        solution = condenser.recover(free).cpu()
        evaluation_asm = diagnostic_asm or asm
        weights, reference = Triangle.get_quadrature(40, torch.float64)
        geometry_values = Triangle.get_basis_fns(1, torch.float64).map(reference)
        geometry_grad = Triangle.get_basis_grad_fns(1, torch.float64).map(reference)
        error_squared = norm_squared = 0.0
        for kind in evaluation_asm.element_types:
            coords = evaluation_asm._points[evaluation_asm.elements[kind]]
            x = torch.einsum("qb,ebd->eqd", geometry_values, coords)
            jac = torch.einsum("qib,ebj->eqij", geometry_grad, coords)
            jxw = torch.linalg.det(jac).abs() * weights
            exact = torch.sin(math.pi * x[..., 0]) * torch.sin(math.pi * x[..., 1])
            actual = evaluation_asm.layout.evaluate("u", solution, reference, element_type=kind)
            error_squared += float(((actual - exact).square() * jxw).sum())
            norm_squared += float((exact.square() * jxw).sum())
        result["relative_L2_error"] = math.sqrt(error_squared / norm_squared)
        result["status"] = "solved"
        if device.type == "cpu":
            # Explicit fresh-factorization timing, excluding tensor/CSC conversion.
            # Dense diagnostic matrices are small; use their nonzeros for CSC
            # conversion outside the timers, independent of sparse wrapper API.
            csc = scipy.sparse.csc_matrix(reduced.to_dense().detach().numpy())
            rhs_np = rhs.detach().numpy()
            result["fresh_scipy_LU_factorization_timing"] = timing(lambda: scipy.sparse.linalg.splu(csc))
            factor = scipy.sparse.linalg.splu(csc)
            result["factorized_triangular_solve_timing"] = timing(lambda: factor.solve(rhs_np))
    except (RuntimeError, ValueError) as exc:
        result["status"] = "solve_failed"
        result["error"] = str(exc)
    return result


def benchmark_case(mesh, p, family, repeats, warmup, batch_size, extra, device=None):
    device = torch.device(device or mesh.points.device)
    timing = lambda fn: measure(fn, device, repeats, warmup)
    degree = 2 * p + extra
    _, q = Triangle.get_quadrature(degree, torch.float64, device)
    setup_timing = timing(lambda: cold_basis(p, family, device))
    basis, gradient = cold_basis(p, family, device)
    tabulation_timing = timing(lambda: (basis.map(q), gradient.map(q)))
    start = perf_counter()
    # Topology setup is a CPU operation. Move the complete assembler after
    # construction rather than feeding GPU connectivity to CPU topology code.
    asm = make_assembler(mesh, p, family, degree).to(device)
    synchronize(device)
    construction_seconds = perf_counter() - start
    # Warm both public paths (including mesh-data reference tables) before
    # checking that no modal recurrence can execute during assembly.
    matrix = asm(batch_size=batch_size)
    load = asm.assemble_vector(batch_size=batch_size)
    audit = audit_hotpath(asm, batch_size) if family == "hierarchical" else None
    local = integration_stage(asm, batch_size)
    oriented = orientation_stage(asm, local)
    projected = projection_stage(asm, oriented)
    rows, cols = asm._coo_pattern(("u__u",), device)
    reconstructed = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=torch.float64, device=device)
    reconstructed.index_put_((rows, cols), projected.reshape(-1), accumulate=True)
    torch.testing.assert_close(reconstructed, matrix.to_dense(), atol=1e-12, rtol=1e-12)
    stages = {
        "basis_and_gradient_construction_uncached": setup_timing,
        "reference_values_and_gradients_tabulation": tabulation_timing,
        "element_integration_excluding_orientation_projection": timing(lambda: integration_stage(asm, batch_size)),
        "orientation_only": timing(lambda: orientation_stage(asm, local)),
        "ReduceProjector_only": timing(lambda: projection_stage(asm, oriented)),
        "full_matrix_assembly": timing(lambda: asm(batch_size=batch_size)),
        "full_load_assembly": timing(lambda: asm.assemble_vector(batch_size=batch_size)),
    }
    diagnostic_asm = make_assembler(mesh, p, family, degree) if device.type == "cuda" else None
    numerical = numerical_solve(asm, matrix, load, timing, device, diagnostic_asm)
    n_element = sum(len(value) for value in asm.elements.values())
    return {"degree": p, "family": family, "quadrature_degree": degree, "quadrature_points": len(q),
            "elements": n_element, "scalar_dofs": asm.n_dofs, "local_basis_count": basis.shape[0],
            "assembler_construction_warm_cache_seconds_diagnostic": construction_seconds,
            "timings": stages, "full_assembly_elements_per_second": n_element / stages["full_matrix_assembly"]["median_seconds"],
            "basis_accuracy": basis_diagnostics(p, family, basis), "numerical_solution": numerical,
            "hotpath_audit": audit}


def markdown_report(report):
    config = report["configuration"]
    lines = ["# Hierarchical triangle benchmark", "",
             f"Device: {config['device']}; dtype: float64; threads: {config['threads']}; "
             f"triangles: {report['mesh']['triangles']}; geometry: P1.", "",
             "## Timing medians (milliseconds)", "",
             "| p | Family | Basis setup | Tabulation | Integration | Orientation | Projection | Full matrix | Sparse solve |",
             "|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    names = ["basis_and_gradient_construction_uncached", "reference_values_and_gradients_tabulation",
             "element_integration_excluding_orientation_projection", "orientation_only", "ReduceProjector_only", "full_matrix_assembly"]
    for row in report["cases"]:
        times = [f"{1000 * row['timings'][name]['median_seconds']:.3f}" for name in names]
        solve = row["numerical_solution"].get("solve_timing")
        times.append(f"{1000 * solve['median_seconds']:.3f}" if solve else "failed")
        lines.append(f"| {row['degree']} | {row['family']} | " + " | ".join(times) + " |")
    lines += ["", "## Numerical accuracy and conditioning (separate from timing)", "",
              "| p | Family | Basis screen | Rank | Relative L2 error | Raw stiffness condition | Diagonal-scaled condition |",
              "|---:|:---|:---|:---|---:|---:|---:|"]
    for row in report["cases"]:
        basis = row["basis_accuracy"]
        numerical = row["numerical_solution"]
        cond = numerical["conditioning"]
        fmt = lambda x: f"{x:.3e}" if x is not None else "unresolved/failed"
        lines.append(f"| {row['degree']} | {row['family']} | {'PASS' if basis['passes_accuracy_screen'] else 'FAIL'} | "
                     f"{basis['sampled_rank']}/{basis['basis_count']} | {fmt(numerical.get('relative_L2_error'))} | "
                     f"{fmt(cond['condition_2_symmetric_part'])} | {fmt(cond['diagonal_scaled_condition_2'])} |")
    lines += ["", "## Acceptance", ""]
    for key, value in report["acceptance"].items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## Method and limitations", ""] + [f"- {note}" for note in report["methodology"]]
    return "\n".join(lines) + "\n"


def run_benchmark(degrees=(1, 2, 4, 6, 8, 10), length=0.5, repeats=7, warmup=2,
                  batch_size=16, extra=8, device="cpu", threads=1, mesh_json=None):
    if repeats < 1 or warmup < 0 or length <= 0 or batch_size < 1 or extra < 0:
        raise ValueError("invalid benchmark sizes/repetition counts")
    if not degrees or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in degrees):
        raise ValueError("degrees must contain positive integers")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("benchmark supports CPU or CUDA float64; MPS does not support float64")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no peak GPU memory measurement can be made")
    torch.set_num_threads(threads)
    if mesh_json is None:
        mesh = Mesh.gen_rectangle(chara_length=length, order=1, element_type="tri").double().cpu()
    else:
        # Portable pre-generated mesh: no Gmsh or system libGLU is needed.
        data = json.loads(Path(mesh_json).read_text())
        points = np.asarray(data["points"], dtype=np.float64)
        triangles = np.asarray(data["triangles"], dtype=np.int64)
        mesh = Mesh(meshio.Mesh(points, [("triangle", triangles)])).double().cpu()
    cases = []
    with torch.no_grad():
        for p in degrees:
            for family in ("lagrange", "hierarchical"):
                print(f"Benchmark p={p}, {family}", file=sys.stderr, flush=True)
                cases.append(benchmark_case(mesh, p, family, repeats, warmup, batch_size, extra, device))
    mesh_hash = hashlib.sha256(mesh.points.cpu().numpy().tobytes() + mesh.elements().cpu().numpy().tobytes()).hexdigest()
    root = Path(__file__).resolve().parents[2]
    source_files = ["examples/poisson/benchmark_hierarchical.py", "tensormesh/assemble/mixed_assembler.py",
                    "tensormesh/assemble/topology.py", "tensormesh/element/hierarchical_triangle.py",
                    "tensormesh/element/element.py", "tensormesh/element/quadrature.py",
                    "tensormesh/assemble/projector.py"]
    hierarchical = sorted((row for row in cases if row["family"] == "hierarchical"), key=lambda row: row["degree"])
    accuracy_passed = all(row["basis_accuracy"]["passes_accuracy_screen"] for row in hierarchical)
    solves_passed = all(row["numerical_solution"]["status"] == "solved"
                        and row["numerical_solution"]["conditioning"]["SPD_resolved_at_float64_precision"]
                        and row["numerical_solution"].get("relative_linear_system_residual", float("inf")) < 1e-9
                        for row in hierarchical)
    audits_passed = all(not row["hotpath_audit"]["direct_cell_loops_detected"]
                        and row["hotpath_audit"]["reference_tables_have_no_element_axis"]
                        and row["hotpath_audit"]["hierarchical_recurrence_blocked_during_matrix_and_load_assembly"]
                        and row["hotpath_audit"]["orientation_precomputed"] for row in hierarchical)
    errors = [row["numerical_solution"].get("relative_L2_error", float("inf")) for row in hierarchical]
    convergence = all(fine < coarse for coarse, fine in zip(errors, errors[1:])) if len(errors) > 1 else None
    return {"configuration": {"degrees": list(degrees), "length": length, "repeats": repeats,
                              "warmup": warmup, "quadrature_batch_size": batch_size, "quadrature_extra": extra,
                              "device": str(device), "dtype": "float64", "threads": threads,
                              "mesh_input": str(mesh_json) if mesh_json is not None else "generated by Gmsh"},
            "environment": {"platform": platform.platform(), "machine": platform.machine(),
                            "python": platform.python_version(), "torch": torch.__version__,
                            "scipy": scipy.__version__, "numpy": np.__version__,
                            "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available()},
            "mesh": {"triangles": len(mesh.elements()), "vertices": mesh.n_points, "sha256": mesh_hash},
            "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in source_files},
            "acceptance": {"required_degrees_covered": {1, 2, 4, 6, 8, 10}.issubset(degrees),
                           "hierarchical_basis_accuracy_passed": accuracy_passed,
                           "hierarchical_constrained_solve_and_SPD_passed": solves_passed,
                           "hierarchical_hotpath_audits_passed": audits_passed,
                           "hierarchical_L2_error_decreases_with_p": convergence,
                           "requested_cpu_numerical_acceptance_passed": device.type == "cpu" and accuracy_passed and solves_passed and audits_passed and convergence is not False,
                           "gpu_peak_memory_measured": device.type == "cuda",
                           "lagrange_basis_screen_failures": [row["degree"] for row in cases if row["family"] == "lagrange" and not row["basis_accuracy"]["passes_accuracy_screen"]]},
            "cases": cases,
            "methodology": [
                "Identical mesh, P1 affine geometry, quadrature degree 2p+extra, float64 and sine forcing for both families.",
                "Both families use MixedElementAssembler; this isolates basis/topology rather than comparing different assemblers.",
                "Basis coefficient/gradient construction bypasses the basis cache; reusable nodal/exponent templates may be warm. Both families construct on CPU, then transfer if CUDA is selected.",
                "Other timings are warmed medians; orientation and pure ReduceProjector are timed on precomputed local matrices.",
                "Integration invokes the production batched private stages; its projected result is checked against public assembly.",
                "Stage medians are independent measurements and need not sum to the end-to-end median.",
                "Solver timing excludes constraints; CPU uses fixed SciPy LU. Fresh LU and prefactorized solves are also measured separately.",
                "Accuracy/conditioning diagnostics and dense conversions are on CPU float64 outside assembly timers; condition numbers use the symmetric part.",
                "Diagonal-scaled conditioning is diagnostic only; no new solver/preconditioner is introduced.",
                "High-order nodal basis failures are reported, never treated as an accuracy oracle or hidden behind throughput claims.",
                "CUDA memory, if selected, is a separate allocator high-water call with baseline and delta, not total process/device memory.",
                "CPU runs do not measure GPU peak memory. MPS is not used because it lacks float64 support.",
                "Small-mesh timings are exploratory, not a scalable CPU/GPU performance claim.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--length", type=float, default=0.5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--quadrature-extra", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--mesh-json", type=Path, help="pre-generated P1 unit-square mesh; avoids Gmsh/libGLU")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output") / "basis_benchmark")
    args = parser.parse_args()
    report = run_benchmark(args.degrees, args.length, args.repeats, args.warmup,
                           args.batch_size, args.quadrature_extra, args.device, args.threads, args.mesh_json)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    text = markdown_report(report)
    (args.output_dir / "benchmark.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
