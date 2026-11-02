"""Compare high-order tetrahedral assembly with a pinned upstream checkout.

The coordinator writes identical mesh/quadrature inputs for isolated workers.
Run with --help for the protocol or --inspect REPORT.json for timing ratios.
"""

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
import traceback

import numpy as np
import scipy
from scipy import sparse

try:
    from ._tetrahedron_comparison import (
        PINNED_UPSTREAM, array_hash, expected_integrals, high_order_cube,
        install_input_adapters, modal_probes, polynomial_probes, quadrature,
        scaled_error, transfer_matrix,
    )
except ImportError:  # Direct script execution.
    from _tetrahedron_comparison import (
        PINNED_UPSTREAM, array_hash, expected_integrals, high_order_cube,
        install_input_adapters, modal_probes, polynomial_probes, quadrature,
        scaled_error, transfer_matrix,
    )


ROOT = Path(__file__).resolve().parents[2]


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def source_info(root):
    root = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted((root / "tensormesh").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"root": str(root), "commit": git(root, "rev-parse", "HEAD"),
            "tracked_changes": git(root, "status", "--porcelain", "--untracked-files=no"),
            "python_source_sha256": digest.hexdigest(),
            "remotes": git(root, "remote", "-v")}


def json_safe(value):
    """Keep failed numerical diagnostics readable as strict JSON (null for NaN/Inf)."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def timed(fn, device):
    synchronize(device)
    start = perf_counter()
    result = fn()
    synchronize(device)
    return result, perf_counter() - start


def measure(fn, device, repeats, warmup, first):
    import torch

    for _ in range(warmup):
        result, _ = timed(fn, device)
        del result
    samples = []
    for _ in range(repeats):
        result, seconds = timed(fn, device)
        samples.append(seconds)
        del result
    memory = {"status": "unavailable"}
    if device.type == "cuda":
        synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        result = fn()
        synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
        memory = {"status": "measured", "baseline_allocated_bytes": baseline,
                  "peak_allocated_bytes": peak,
                  "incremental_peak_allocated_bytes": peak - baseline,
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
        del result
    return {"first_call_seconds": first, "samples_seconds": samples,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples), "max_seconds": max(samples), "gpu_memory": memory}


def assert_import_root(root):
    paths = {}
    for name, module in tuple(sys.modules.items()):
        if name == "tensormesh" or name.startswith("tensormesh."):
            filename = getattr(module, "__file__", None)
            if filename:
                path = Path(filename).resolve()
                if not path.is_relative_to(root / "tensormesh"):
                    raise RuntimeError(f"contaminated import: {name} from {path}")
                paths[name] = str(path)
    if "tensormesh" not in paths:
        raise RuntimeError("TensorMesh was not imported")
    return paths


def run_worker(job, result):
    """Executed only in a fresh interpreter; import the selected checkout first."""
    root = Path(job["source"]["root"])
    sys.path.insert(0, str(root))
    import torch
    import meshio
    import tensormesh as tm
    from tensormesh.element import Tetrahedron

    result["imported_modules"] = assert_import_root(root)
    observed = source_info(root)
    if any(observed[k] != job["source"][k] for k in ("commit", "python_source_sha256")):
        raise RuntimeError("source checkout changed after coordinator recorded provenance")
    torch.set_num_threads(job["threads"])
    dtype = getattr(torch, job["dtype"])
    device = torch.device(job["device"])
    result["imported_package"] = str(Path(tm.__file__).resolve())
    result["environment"] = {"python": platform.python_version(), "torch": torch.__version__,
                             "numpy": np.__version__, "scipy": scipy.__version__,
                             "torch_sla": version("torch-sla"), "cuda_runtime": torch.version.cuda,
                             "device": str(device), "threads": torch.get_num_threads(),
                             "device_model": (torch.cuda.get_device_name(device)
                                              if device.type == "cuda" and torch.cuda.is_available()
                                              else platform.processor())}
    if device.type == "cuda" and not torch.cuda.is_available():
        result.update(status="unavailable", reason="CUDA unavailable")
        return
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result["environment"].update(
            device_total_memory_bytes=properties.total_memory,
            device_uuid=str(getattr(properties, "uuid", "unavailable")),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with np.load(job["input"]) as archive:
        data = dict(archive)
    p, form = job["p"], job["form"]
    calls = install_input_adapters(data)
    result["adapters"] = ["external_conical_gauss_jacobi_quadrature",
                          "integer_side_length_in_tet_linspace"]
    result["mesh_sha256"] = array_hash(data["points"], data["cells"])
    result["input_quadrature_sha256"] = array_hash(
        data["quadrature_weights"], data["quadrature_points"])
    result["autograd_enabled"] = False
    kind = "tetra" if p == 1 else f"tetra{data['cells'].shape[1]}"

    def mass(self, u, v):
        return u * v

    def poisson(self, gradu, gradv):
        return (gradu * gradv).sum()

    def unit_load(self, v):
        return v

    start = perf_counter()
    mesh = tm.Mesh(meshio.Mesh(data["points"], [(kind, data["cells"])]), reorder=False)
    mesh = mesh.to(dtype=dtype)
    result["setup"] = {"mesh_seconds": perf_counter() - start}
    start = perf_counter()
    hierarchical = job["family"] == "hierarchical"
    if hierarchical or job.get("upstream_assembler", "element") == "mixed":
        # Upstream Field has no basis keyword; its default is nodal Lagrange.
        field = tm.Field("u", "v", p, **({"basis": "hierarchical"} if hierarchical else {}))
        cls = type("ComparisonMixed", (tm.MixedElementAssembler,), {
            "fields": [field],
            "forward": mass if form == "mass" else poisson, "forward_vector": unit_load})
        asm = cls.from_mesh(mesh, quadrature_order=2*p)
        load_asm = None
        result["assembly_api"] = ("MixedElementAssembler (grouped CSR)" if hierarchical else
                                  "MixedElementAssembler (default reduce projection)")
    else:
        cls = type("ComparisonLagrange", (tm.ElementAssembler,), {
            "forward": mass if form == "mass" else poisson})
        load_cls = type("ComparisonLoad", (tm.NodeAssembler,), {"forward": unit_load})
        asm = cls.from_mesh(mesh, quadrature_order=2*p)
        load_asm = load_cls.from_mesh(mesh, quadrature_order=2*p)
        result["assembly_api"] = "ElementAssembler / NodeAssembler (default reduce projection)"
    result["setup"]["assembler_cpu_seconds"] = perf_counter() - start
    _, seconds = timed(lambda: asm.to(device), device)
    if load_asm is not None:
        _, extra = timed(lambda: load_asm.to(device), device)
        seconds += extra
    result["setup"]["device_transfer_seconds"] = seconds
    result["setup"]["total_seconds"] = sum(result["setup"].values())

    with torch.no_grad():
        matrix_fn = lambda: asm(batch_size=job["batch_size"])
        load_fn = (lambda: asm.assemble_vector(batch_size=job["batch_size"])) if load_asm is None \
            else (lambda: load_asm(batch_size=job["batch_size"]))
        matrix, first_matrix = timed(matrix_fn, device)
        cpu_matrix = sparse.coo_matrix((matrix.values.cpu().numpy().copy(),
                                       (matrix.row.cpu().numpy(), matrix.col.cpu().numpy())),
                                      shape=tuple(matrix.shape)).tocsr()
        del matrix
        load, first_load = timed(load_fn, device)
        cpu_load = load.cpu().numpy().copy()
        del load
        result["dofs"] = cpu_matrix.shape[0]
        # Only production caches exist while measuring. CPU copies of the first
        # outputs are kept for checks; diagnostic GPU tabulation happens later.
        if not job["validation"]:
            result["timings"] = {
                "full_matrix_assembly": measure(matrix_fn, device, job["repeats"],
                                                job["warmup"], first_matrix),
                "full_load_assembly": measure(load_fn, device, job["repeats"],
                                              job["warmup"], first_load)}

        # Verify actual installed nodes/rule, including dtype conversion and dispatch.
        actual_nodes = Tetrahedron.get_basis(p, dtype, device)
        reference = torch.tensor(data["reference_nodes"], dtype=dtype, device=device)
        if not torch.allclose(actual_nodes, reference,
                              atol=2e-7 if dtype == torch.float32 else 1e-14, rtol=0):
            raise ValueError("upstream reference-node ordering differs from input mesh")
        w, q = asm.transformation[kind].quadrature
        result["quadrature"] = {"sha256": array_hash(w.cpu().numpy(), q.cpu().numpy()),
                                "points": len(w), "exactness_degree": 2*p+1,
                                "assembler_requests": sorted(calls)}
        if load_asm is not None:
            lw, lq = load_asm.transformation[kind].quadrature
            if array_hash(lw.cpu().numpy(), lq.cpu().numpy()) != result["quadrature"]["sha256"]:
                raise ValueError("matrix and load quadrature differ")
        nodal_identity = Tetrahedron.get_basis_fns(p, dtype, device).map(actual_nodes)
        identity_error = float((nodal_identity - torch.eye(len(reference), dtype=dtype,
                                                          device=device)).abs().max())
        del nodal_identity
        exponents, probes = polynomial_probes(data, p)
        if job["family"] == "hierarchical":
            probes = modal_probes(asm, data, p, probes)
        gram, moments = expected_integrals(exponents, form)
        numerical = {"nodal_identity_error": identity_error,
                     "polynomial_gram_error": scaled_error(probes.T @ cpu_matrix @ probes, gram),
                     "polynomial_load_error": scaled_error(probes.T @ cpu_load, moments),
                     "symmetry_error": scaled_error(cpu_matrix, cpu_matrix.T)}
        tolerance = 3e-4 if dtype == torch.float32 else 2e-8
        numerical["tolerance"] = tolerance
        numerical["passed"] = all(np.isfinite(v) and v <= tolerance
                                  for k, v in numerical.items() if k.endswith("error"))
        result["numerical"] = numerical
        if job["validation"]:
            prefix = job["artifact"]
            sparse.save_npz(prefix + "-matrix.npz", cpu_matrix)
            np.save(prefix + "-load.npy", cpu_load)
            if job["family"] == "hierarchical":
                transfer, error = transfer_matrix(asm, data, p)
                sparse.save_npz(prefix + "-transfer.npz", transfer)
                numerical["trace_variance_error"] = error
        del probes, cpu_matrix, cpu_load
        if not numerical["passed"]:
            result.update(status="numerical_failure", reason="physical polynomial or basis check failed")
        else:
            result["status"] = "passed"
    result["imported_modules"] = assert_import_root(root)


def worker_main(path):
    job = json.loads(Path(path).read_text())
    result = dict(job)
    result["status"] = "error"
    try:
        with redirect_stdout(sys.stderr):
            run_worker(job, result)
    except Exception as exc:
        # A killed worker is handled by the coordinator, separately from CUDA OOM.
        result["status"] = "out_of_memory" if "out of memory" in str(exc).lower() else "error"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc(file=sys.stderr)
    write_json(job["result"], result)
    return 0 if result["status"] == "passed" else 1


def launch(job, directory, index):
    prefix = directory / f"{index:04d}-{job['family']}-p{job['p']}-{job['form']}"
    job.update(artifact=str(prefix), result=str(prefix) + ".json")
    job_path = str(prefix) + "-job.json"
    write_json(job_path, job)
    env = dict(os.environ, PYTHONPATH=job["source"]["root"],
               OMP_NUM_THREADS=str(job["threads"]), OPENBLAS_NUM_THREADS=str(job["threads"]),
               MKL_NUM_THREADS=str(job["threads"]), VECLIB_MAXIMUM_THREADS=str(job["threads"]))
    print(f"{'check' if job['validation'] else 'time'} {job['family']} P{job['p']} "
          f"{job['form']} n={job['subdivisions']} {job['orientation']} "
          f"(upstream={job.get('upstream_assembler', 'element')})", file=sys.stderr, flush=True)
    with open(str(prefix) + ".log", "w") as log:
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--worker", job_path],
                                   cwd=job["source"]["root"], env=env, stdout=log, stderr=log)
    result_path = Path(job["result"])
    result = json.loads(result_path.read_text()) if result_path.exists() else {
        **job, "status": "worker_failed", "reason": f"worker exit code {completed.returncode}"}
    if completed.returncode != 0 and result["status"] == "passed":
        result.update(status="worker_failed", reason=f"worker exit code {completed.returncode}")
    result["log"] = str(prefix) + ".log"
    result["returncode"] = completed.returncode
    return result


def compare_operators(nodal, modal, tolerance):
    """Full assembled matrix and load equivalence on all 24 orientations."""
    if nodal["status"] != "passed" or modal["status"] != "passed":
        return {"passed": False, "reason": "a validation worker failed"}
    for key in ("mesh_sha256", "input_quadrature_sha256", "dofs"):
        if nodal[key] != modal[key]:
            return {"passed": False, "reason": f"mismatched {key}"}
    if nodal["quadrature"]["sha256"] != modal["quadrature"]["sha256"]:
        return {"passed": False, "reason": "mismatched actual quadrature"}
    lp, hp = nodal["artifact"], modal["artifact"]
    lmat = sparse.load_npz(lp + "-matrix.npz")
    hmat = sparse.load_npz(hp + "-matrix.npz")
    transfer = sparse.load_npz(hp + "-transfer.npz")
    errors = {"matrix_change_of_basis_error": scaled_error(transfer.T @ lmat @ transfer, hmat),
              "load_change_of_basis_error": scaled_error(
                  transfer.T @ np.load(lp + "-load.npy"), np.load(hp + "-load.npy"))}
    return {**errors, "tolerance": tolerance,
            "passed": all(np.isfinite(e) and e <= tolerance for e in errors.values())}


def timing_pair(nodal, modal, validation):
    result = {k: modal[k] for k in ("p", "form", "subdivisions", "orientation")}
    result["upstream_assembler"] = nodal.get("upstream_assembler", "element")
    result["cells"] = 6 * result["subdivisions"]**3
    result["status"] = "not_comparable"
    result["worker_status"] = {"lagrange": nodal["status"], "hierarchical": modal["status"]}
    if not validation["passed"] or any(c["status"] != "passed" for c in (nodal, modal)):
        return result
    for key in ("mesh_sha256", "input_quadrature_sha256", "dofs", "batch_size", "dtype"):
        if nodal[key] != modal[key]:
            result["reason"] = f"mismatched {key}"
            return result
    if nodal["quadrature"]["sha256"] != modal["quadrature"]["sha256"] or \
            nodal["environment"] != modal["environment"]:
        result["reason"] = "quadrature or execution environment differs"
        return result
    result.update(status="passed", dofs=modal["dofs"], operations={})
    for operation in ("full_matrix_assembly", "full_load_assembly"):
        l = nodal["timings"][operation]["median_seconds"]
        h = modal["timings"][operation]["median_seconds"]
        result["operations"][operation] = {"lagrange_ms": l * 1000, "hierarchical_ms": h * 1000,
                                           "lagrange_over_hierarchical": l / h,
                                           "hierarchical_time_change_percent": 100 * (h/l - 1)}
    return result


def run(args):
    upstream = source_info(args.upstream)
    if upstream["commit"] != args.upstream_commit or upstream["tracked_changes"]:
        raise ValueError(f"upstream must be a clean checkout at {args.upstream_commit}")
    if git(args.upstream, "status", "--porcelain", "--untracked-files=all", "--", "tensormesh"):
        raise ValueError("upstream tensormesh/ contains modified or untracked source files")
    feature = source_info(ROOT)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=output.stem + "-workers-", dir=output.parent))
    report = {"schema_version": 1, "status": "running", "upstream": upstream, "feature": feature,
              "driver_sha256": array_hash(np.frombuffer(Path(__file__).read_bytes(), dtype=np.uint8),
                                          np.frombuffer(Path(__file__).with_name(
                                              "_tetrahedron_comparison.py").read_bytes(), dtype=np.uint8)),
              "artifacts": str(directory), "protocol": {
                  "upstream_assembler": args.upstream_assembler,
                  "forms": "scalar mass / Poisson and unit load", "geometry": "same affine Pp nodal mesh",
                  "quadrature": "same external conical Gauss-Jacobi rule; exact through 2*p+1",
                  "batch_size": args.batch_size, "element_batch_size": None,
                  "validation": "full A_H=T.T@A_L@T and b_H=T.T@b_L on 48 cells / 24 orientations",
                  "ratio": "L/H > 1 favors hierarchical; positive H time change means slower",
                  "adapters": "external quadrature + integer lattice-size fix on both checkouts",
                  "isolation": "fresh process per implementation, degree, mesh and form",
                  "timing": "forward-only warmed public assembly, synchronized wall clock",
                  "memory": "matrix/load assembly state resident; GPU outputs released before samples"},
              "validation": [], "cases": [], "comparisons": [], "gpu_performance_measured": False}
    index = 0
    for p in args.degrees:
        for form in args.forms:
            check = None
            for validation, n, orientation in [(True, 2, "balanced")] + [
                    (False, n, o) for n in args.subdivisions for o in args.orientations]:
                if not validation and not check["passed"]:
                    report["comparisons"].append({"p": p, "form": form, "subdivisions": n,
                                                  "orientation": orientation, "status": "validation_failed",
                                                  "upstream_assembler": args.upstream_assembler})
                    continue
                data = {**high_order_cube(n, p, orientation), **quadrature(p)}
                input_path = directory / f"input-{p}-{n}-{orientation}.npz"
                np.savez(input_path, **data)
                del data
                pair = {}
                # Alternate run order across pairs; never run GPU workers concurrently.
                families = ("lagrange", "hierarchical") if index % 4 == 0 else ("hierarchical", "lagrange")
                for family in families:
                    job = {"source": upstream if family == "lagrange" else feature,
                           "upstream_assembler": args.upstream_assembler,
                           "family": family, "p": p, "form": form, "subdivisions": n,
                           "orientation": orientation, "input": str(input_path), "validation": validation,
                           "device": args.device, "dtype": args.dtype, "threads": args.threads,
                           "batch_size": args.batch_size, "element_batch_size": None,
                           "repeats": args.repeats, "warmup": args.warmup}
                    pair[family] = launch(job, directory, index)
                    index += 1
                    report["cases"].append(pair[family])
                    write_json(output, report)
                if validation:
                    check = compare_operators(pair["lagrange"], pair["hierarchical"],
                                              3e-4 if args.dtype == "float32" else 2e-8)
                    report["validation"].append({"p": p, "form": form,
                                                  "upstream_assembler": args.upstream_assembler, **check})
                else:
                    report["comparisons"].append(timing_pair(pair["lagrange"], pair["hierarchical"], check))
                write_json(output, report)
    report["status"] = "completed" if all(c["status"] == "passed" for c in report["comparisons"]) else "incomplete"
    report["gpu_performance_measured"] = args.device.startswith("cuda") and any(
        c["status"] == "passed" for c in report["comparisons"])
    report = json_safe(report)
    write_json(output, report)
    return report


def inspect_reports(paths):
    for path in paths:
        report = json.loads(Path(path).read_text())
        print(f"\n{path}: {report['status']}; upstream {report['upstream']['commit'][:12]}")
        environment = next((c["environment"] for c in report["cases"] if "environment" in c), {})
        print(f"Device: {environment.get('device', 'unknown')} "
              f"{environment.get('device_model', '')}; batch size: {report['protocol']['batch_size']}")
        upstream_api = ("MixedElementAssembler" if report["protocol"].get(
            "upstream_assembler", "element") == "mixed" else "ElementAssembler / NodeAssembler")
        print(f"L: upstream {upstream_api}; H: feature grouped MixedElementAssembler")
        print("Matrix times; L/H > 1 favors hierarchical; H change > 0 means slower.")
        print(f"{'p':>2} {'Cells':>8} {'Form':8} {'Orientation':11} {'L ms':>10} {'H ms':>10} "
              f"{'L/H':>7} {'H change':>10} {'L GiB':>8} {'H GiB':>8}")
        cases = {(c['p'], c['form'], c['subdivisions'], c['orientation'], c['family']): c
                 for c in report['cases'] if not c['validation']}
        for c in report["comparisons"]:
            if c["status"] != "passed":
                print(f"P{c['p']} n={c['subdivisions']} {c['form']} {c['orientation']}: "
                      f"{c['status']} {c.get('worker_status', '')}")
                continue
            t = c["operations"]["full_matrix_assembly"]
            memory = []
            for family in ("lagrange", "hierarchical"):
                case = cases[c['p'], c['form'], c['subdivisions'], c['orientation'], family]
                peak = case["timings"]["full_matrix_assembly"]["gpu_memory"].get("peak_allocated_bytes")
                memory.append(f"{peak / 2**30:.3f}" if peak is not None else "n/a")
            print(f"{c['p']:2} {c['cells']:8} {c['form']:8} {c['orientation']:11} "
                  f"{t['lagrange_ms']:10.3f} {t['hierarchical_ms']:10.3f} "
                  f"{t['lagrange_over_hierarchical']:7.3f} "
                  f"{t['hierarchical_time_change_percent']:+9.1f}% {memory[0]:>8} {memory[1]:>8}")
        for c in report["cases"]:
            if c["status"] != "passed":
                print(f"  {c['family']} P{c['p']} {c['status']}: {c.get('reason', '')}; log: {c['log']}")
        for c in report["validation"]:
            if not c["passed"]:
                print("  Failed operator equivalence:", c)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, help="separate clean upstream checkout")
    parser.add_argument("--upstream-commit", default=PINNED_UPSTREAM)
    parser.add_argument("--upstream-assembler", choices=("element", "mixed"), default="element",
                        help="upstream Lagrange API: element (original baseline) or mixed")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--degrees", nargs="+", type=int, default=[3, 4, 6])
    parser.add_argument("--forms", nargs="+", choices=("mass", "poisson"), default=["mass", "poisson"])
    parser.add_argument("--subdivisions", nargs="+", type=int, default=[2, 4])
    parser.add_argument("--orientations", nargs="+", choices=("balanced", "single"), default=["balanced"])
    parser.add_argument("--batch-size", type=int, default=-1, help="quadrature points per batch on BOTH paths")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--output", default="dist/tetra-upstream.json")
    parser.add_argument("--inspect", nargs="+", metavar="REPORT.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker_main(args.worker)
    if args.inspect:
        inspect_reports(args.inspect)
        return 0
    if args.upstream is None:
        parser.error("--upstream is required")
    if any(p < 3 or p > 10 for p in args.degrees):
        parser.error("comparison degrees must be between 3 and 10")
    if min(args.subdivisions) < 1 or min(args.threads, args.repeats, args.warmup) < 1:
        parser.error("subdivisions, threads, repeats and warmup must be positive")
    if args.batch_size != -1 and args.batch_size < 1:
        parser.error("batch size must be -1 or positive")
    report = run(args)
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
