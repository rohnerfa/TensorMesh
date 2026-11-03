"""Reproducible, forward-only grouped tetrahedral assembly benchmark.

Run from the repository root. JSON is written to stdout (progress to stderr),
and optionally to --output. Stage reconstruction uses the production tensor
kernels, but independently schedules packing and reduction projection. It streams
quadrature batches rather than retaining a mesh-sized collection of gradients.
"""

import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import itertools
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import meshio
import numpy as np
import scipy
import scipy.sparse.linalg
import torch

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.assemble import mixed_assembler
from tensormesh.assemble._tetrahedron_groups import GroupedTetrahedra
from tensormesh.element import Tetrahedron
from tensormesh.sparse import SparseMatrix

try:
    from ._tetrahedron_reference_report import reference_report
except ImportError:
    from _tetrahedron_reference_report import reference_report


FORMS = ("mass", "poisson", "mixed", "field_data")
ORIENTATIONS = tuple(itertools.permutations(range(4)))


def cube_mesh(subdivisions=1, orientation="balanced", dtype=torch.float64):
    """Create a conforming Freudenthal triangulation of the unit cube.

    Parameters
    ----------
    subdivisions : int
        Cubes per coordinate direction; the mesh has 6 * subdivisions**3 cells.
    orientation : str
        ``balanced``, ``single`` or ``skewed`` local vertex-order distribution.
        Only local slot order changes, never the physical tetrahedra.
    dtype : torch.dtype
        Floating-point coordinate precision.
    """
    if subdivisions < 1 or orientation not in ("balanced", "single", "skewed"):
        raise ValueError("invalid cube size or orientation distribution")
    n = subdivisions
    grid = np.arange(n + 1, dtype=np.float64) / n
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    origins = np.stack(np.meshgrid(*([np.arange(n)] * 3), indexing="ij"), -1)
    origins = origins.reshape(-1, 1, 3)
    paths = []
    for axes in itertools.permutations(range(3)):
        paths.append(np.concatenate([np.zeros((1, 3), dtype=np.int64),
                                     np.eye(3, dtype=np.int64)[list(axes)].cumsum(0)]))
    corners = origins[:, None] + np.asarray(paths)[None]
    cells = ((corners[..., 0] * (n + 1) + corners[..., 1]) * (n + 1)
             + corners[..., 2]).reshape(-1, 4)
    cells.sort(axis=1)
    count = len(cells)
    classes = np.zeros(count, dtype=np.int64)
    if orientation == "balanced":
        classes = np.arange(count) % 24
    elif orientation == "skewed":
        minority = max(1, count // 10)
        classes[-minority:] = 1 + np.arange(minority) % 23
    # The class is argsort(local global IDs), so invert the desired permutation.
    slots = np.argsort(np.asarray(ORIENTATIONS), axis=1)[classes]
    cells = np.take_along_axis(cells, slots, axis=1)
    mesh = Mesh(meshio.Mesh(points, [("tetra", cells)]))
    return (mesh.double() if dtype == torch.float64 else mesh.float()).cpu()


def assembler_class(p, form, family="hierarchical"):
    """Declare a scalar or vector/mixed benchmark form."""
    fields = [Field("u", "v", p, components=3 if form == "mixed" else 1, basis=family)]
    if form == "mixed":
        fields.append(Field("w", "z", max(1, p - 1), basis=family))

    def mass(self, u, v):
        return u * v

    def poisson(self, gradu, gradv):
        return (gradu * gradv).sum()

    def mixed(self, u, gradu, w, v, gradv, z):
        return ((gradu * gradv).sum() + (u * v).sum() + 2 * w * z
                + 0.25 * (u.sum() * z + w * v.sum()))

    def field_data(self, gradu, gradv, rho):
        return (1 + rho.square()) * (gradu * gradv).sum()

    def unit_load(self, v):
        return v

    def mixed_load(self, v, z):
        return v.sum() + z

    def weighted_load(self, v, rho):
        return (1 + rho.square()) * v

    forwards = {"mass": mass, "poisson": poisson, "mixed": mixed, "field_data": field_data}
    loads = {"mixed": mixed_load, "field_data": weighted_load}
    return type("TetrahedronBenchmark", (MixedElementAssembler,), {
        "fields": fields, "forward": forwards[form],
        "forward_vector": loads.get(form, unit_load),
    })


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device):
    synchronize(device)
    start = perf_counter()
    result = fn()
    synchronize(device)
    return result, perf_counter() - start


def summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "min_seconds": min(samples), "max_seconds": max(samples)}


def memory_probe(fn, device):
    """Separate CUDA allocator high-water call, with a warmed allocator."""
    if device.type != "cuda":
        return {"status": "unavailable", "reason": "CPU run; CUDA memory was not measured"}
    synchronize(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    result = fn()
    synchronize(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    del result
    return {"status": "measured", "baseline_allocated_bytes": allocated,
            "baseline_reserved_bytes": reserved, "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "incremental_peak_allocated_bytes": peak_allocated - allocated,
            "incremental_peak_reserved_bytes": peak_reserved - reserved}


def measure(fn, device, repeats, warmup, first_seconds=None):
    for _ in range(warmup):
        result, _ = timed_call(fn, device)
        del result
    samples = []
    for _ in range(repeats):
        result, elapsed = timed_call(fn, device)
        samples.append(elapsed)
        del result
    report = summary(samples)
    if first_seconds is not None:
        report["first_call_seconds"] = first_seconds
    report["gpu_memory"] = memory_probe(fn, device)
    return report


class SetupRecorder:
    """Exclusive CPU setup times, including nested tabulation in group setup."""

    def __init__(self):
        self.seconds = defaultdict(float)
        self.stack = []

    @contextmanager
    def section(self, name):
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            self.seconds[name] += elapsed - frame[1]
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, name, fn):
        def measured(*args, **kwargs):
            with self.section(name):
                return fn(*args, **kwargs)
        return measured


class TabulationProbe:
    """Setup-only adapter; the production assembler retains only its tensors."""

    def __init__(self, basis, recorder):
        self.basis, self.recorder = basis, recorder

    def map(self, points):
        with self.recorder.section("oriented_reference_tabulation"):
            return self.basis.map(points)

    def grad(self):
        with self.recorder.section("oriented_reference_tabulation"):
            return TabulationProbe(self.basis.grad(), self.recorder)


def construct(mesh, p, form, family, request, device):
    """Instrument one real CPU construction, then transfer the complete module.

    No second mesh-sized assembler is retained for setup profiling. Timings
    include Python instrumentation overhead; process-level caches are not reset.
    """
    recorder = SetupRecorder()
    factory = Tetrahedron.get_hierarchical_basis_fns

    def basis_factory(*args, **kwargs):
        with recorder.section("oriented_reference_tabulation"):
            return TabulationProbe(factory(*args, **kwargs), recorder)

    with ExitStack() as patches:
        for name in ("_tetrahedron_topology", "hierarchical_tetrahedron_dofmap",
                     "lagrange_dofmap"):
            patches.enter_context(patch.object(
                mixed_assembler, name, recorder.wrap("entity_topology_and_dof_maps",
                                                    getattr(mixed_assembler, name))))
        patches.enter_context(patch.object(
            mixed_assembler, "build_edges",
            recorder.wrap("projection_and_sparse_pattern", mixed_assembler.build_edges)))
        for name, category in (("__init__", "group_layout"),
                               ("add_pair", "projection_and_sparse_pattern"),
                               ("add_vectors", "projection_and_sparse_pattern")):
            patches.enter_context(patch.object(
                GroupedTetrahedra, name,
                recorder.wrap(category, getattr(GroupedTetrahedra, name))))
        patches.enter_context(patch.object(Tetrahedron, "get_hierarchical_basis_fns",
                                          side_effect=basis_factory))
        asm, cpu_seconds = timed_call(
            lambda: assembler_class(p, form, family).from_mesh(mesh, quadrature_order=request),
            torch.device("cpu"))
    asm, transfer_seconds = timed_call(lambda: asm.to(device), device)
    stages = dict(recorder.seconds)
    stages["other_geometry_nodal_tables_and_module_setup"] = cpu_seconds - sum(stages.values())
    return asm, {"first_construction_cpu_seconds": cpu_seconds,
                 "transfer_seconds": transfer_seconds,
                 "total_seconds": cpu_seconds + transfer_seconds,
                 "exclusive_cpu_stages_seconds": stages,
                 "cache_policy": "first assembler for this case; process caches may be warm"}


def affine_coefficients(asm, name, constant=0.0, axis=None):
    """Represent an affine function using the vertex-mode contract, without fitting."""
    field = asm.layout.field(name)
    values = asm._points.new_zeros(asm.layout.n_nodes(name))
    if field.basis == "hierarchical":
        for kind in asm.element_types:
            geometry = asm._points[asm.elements[kind][:, :4]]
            nodal = geometry[..., axis] if axis is not None else geometry[..., 0] * 0
            values[asm.field_conn[f"{name}__{kind}"][:, :4]] = constant + nodal
    else:
        points = asm.layout.points(name)
        values[:] = constant + (points[:, axis] if axis is not None else 0)
    return values if field.components == 1 else values[:, None].expand(-1, field.components)


def data_for(asm, form, shift=0.0):
    if form != "field_data":
        return {}
    return {"rho": ("u", affine_coefficients(asm, "u", constant=shift, axis=0))}


def reconstruct(asm, matrix=True, batch_size=16, element_batch_size=None, field_data=None,
                reference_mode="auto"):
    """Stream geometry, kernels, packing and reduction projection independently.

    Returns
    -------
    result : SparseMatrix or torch.Tensor
        Reconstructed public matrix/load, with no dense global conversion.
    seconds : dict
        Per-stage accumulated synchronized wall times for this pass. The extra
        synchronization deliberately perturbs execution; use public timings for
        throughput and allocator peaks.
    """
    if not asm.grouped_tetrahedra:
        raise ValueError("stage reconstruction requires grouped tetrahedra")
    field_data = field_data or {}
    fn = asm.forward if matrix else asm.forward_vector
    points = {"x": asm._points}
    params = asm._classify_params(fn, points, {}, {}, field_data)
    zeros, eyes, needs_val, needs_grad = asm._pass_context(
        params, field_data, asm.dtype, asm.device)
    if matrix:
        executed = asm._executed_pairs(asm.fields, params)
    else:
        active = {f.trial for _, role, f in params if role in ("test_val", "test_grad")}
        executed = [(None, f) for f in asm.fields if f.trial in active]
    grouping = asm.grouping
    projectors = grouping.pair_projector if matrix else grouping.vector_projector
    slices = grouping.pair_slices if matrix else grouping.vector_slices
    batches = iter(grouping.batches(asm, params, points, {}, {}, field_data,
                                   needs_val, needs_grad, batch_size, element_batch_size,
                                   reference_mode))
    seconds = defaultdict(float)
    sources, accumulated = {}, {}
    checked = False
    while True:
        batch, elapsed = timed_call(lambda: next(batches, None), asm.device)
        seconds["geometry_data_and_physical_gradients"] += elapsed
        if batch is None:
            break
        kind, start, stop, last_q, jxw, val, grad, data, plan = batch
        if not checked:
            _, elapsed = timed_call(lambda: asm._check_bilinear(
                fn, params, data, asm.dtype, asm.device,
                form="bilinear" if matrix else "linear"), asm.device)
            seconds["form_validation"] += elapsed
            checked = True
        for alpha, beta in executed:
            key = f"{beta.trial}__{alpha.trial}" if matrix else beta.trial

            def evaluate():
                out, has_e = asm._run_pass(
                    fn, params, alpha, beta, val, grad, data, zeros,
                    {"trial": eyes.get(alpha.trial) if matrix else None,
                     "test": eyes.get(beta.trial)})
                return out, has_e

            (out, has_e), elapsed = timed_call(evaluate, asm.device)
            seconds["reference_form_evaluation" if plan is not None else
                    "cell_form_evaluation"] += elapsed

            def integrate():
                contribution = grouping.contract(asm, out, jxw, has_e, plan)
                return (accumulated[key] + contribution if key in accumulated
                        else contribution)

            accumulated[key], elapsed = timed_call(integrate, asm.device)
            seconds["reference_contraction" if plan is not None else "cell_quadrature"] += elapsed
            del out
            if not last_q:
                continue

            def pack():
                local = accumulated.pop(key)
                trailing = (beta.components, alpha.components) if matrix else (beta.components,)
                if key not in sources:
                    sources[key] = local.new_empty(projectors[key].from_shape[0], *trailing)
                width = local.numel() // ((stop - start) * np.prod(trailing))
                offset = slices[key][kind][0]
                sources[key][offset + start * width:offset + stop * width] = \
                    local.reshape(-1, *trailing)

            _, elapsed = timed_call(pack, asm.device)
            seconds["local_buffer_packing"] += elapsed
    projected = {}
    for key, source in sources.items():
        projected[key], elapsed = timed_call(lambda: projectors[key](source), asm.device)
        seconds["global_projection"] += elapsed

    def finish():
        if matrix:
            keys = tuple(f"{beta.trial}__{alpha.trial}" for alpha, beta in executed)
            rows, cols = asm._coo_pattern(keys, asm.device)
            return SparseMatrix(torch.cat([projected[key].reshape(-1) for key in keys]),
                                rows, cols, (asm.n_dofs, asm.n_dofs))
        return torch.cat([projected[f.trial].reshape(-1) if f.trial in projected else
                          asm._points.new_zeros(asm.layout.n_nodes(f.trial) * f.components)
                          for f in asm.fields])

    result, elapsed = timed_call(finish, asm.device)
    seconds["global_output_layout"] += elapsed
    return result, dict(seconds)


def measure_stages(fn, repeats, warmup, first):
    for _ in range(warmup):
        result, _ = fn()
        del result
    samples = defaultdict(list)
    for _ in range(repeats):
        result, stages = fn()
        del result
        for key, elapsed in stages.items():
            samples[key].append(elapsed)
    return {key: {**summary(values), "first_reconstruction_seconds": first[key]}
            for key, values in samples.items()}


def numerical_diagnostics(asm, matrix, load, form, shift=0.0):
    """Analytic unit-cube identities; independent of the stage reconstruction."""
    constant = asm.layout.cat({f.trial: affine_coefficients(asm, f.trial, 1)
                               for f in asm.fields})
    linear = asm.layout.cat({f.trial: affine_coefficients(asm, f.trial, axis=0)
                             for f in asm.fields})
    tolerance = 3e-4 if asm.dtype == torch.float32 else 2e-10
    weighted_volume = 1 + shift * shift + shift + 1 / 3
    expected_load = 4.0 if form == "mixed" else weighted_volume if form == "field_data" else 1.0
    expected_constant = {"mass": 1.0, "poisson": 0.0, "mixed": 6.5, "field_data": 0.0}[form]
    expected_linear = {"mass": 1 / 3, "poisson": 1.0, "mixed": 31 / 6,
                       "field_data": weighted_volume}[form]
    action = matrix @ constant
    actual = {"constant_energy": float(constant @ action),
              "linear_coordinate_energy": float(linear @ (matrix @ linear)),
              "constant_load_integral": float(constant @ load)}
    expected = {"constant_energy": expected_constant,
                "linear_coordinate_energy": expected_linear,
                "constant_load_integral": expected_load}
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], atol=tolerance, rtol=tolerance)
    if form in ("poisson", "field_data"):
        nullspace = float(action.abs().max())
        torch.testing.assert_close(action, torch.zeros_like(action), atol=tolerance, rtol=0)
        actual["constant_nullspace_max_error"] = nullspace
    if form == "mass":
        torch.testing.assert_close(action, load, atol=tolerance, rtol=tolerance)
        actual["mass_times_constant_minus_load_max_error"] = float((action - load).abs().max())
    if not bool(torch.isfinite(matrix.values).all() and torch.isfinite(load).all()):
        raise AssertionError("nonfinite matrix/load")
    return {"passed": True, "actual": actual, "expected": expected,
            "absolute_and_relative_tolerance": tolerance}


def storage_report(asm, batch_size, element_batch_size):
    """Persistent table/index storage and explicit scratch/source size estimates."""
    if not asm.grouped_tetrahedra:
        return {"status": "not_grouped"}
    grouping = asm.grouping
    size = asm._points.element_size()
    tables = list(grouping.ref_val.values()) + list(grouping.ref_grad.values())
    # BufferDict normally registers each space once; count shared storage once too.
    storage = {tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
               for tensor in tables}
    projection_bytes = sum(projector.indices.numel() * projector.indices.element_size()
                           for projector in itertools.chain(grouping.pair_projector.values(),
                                                            grouping.vector_projector.values()))
    local_bytes = sum(grouping.pair_projector[f"{beta.trial}__{alpha.trial}"].from_shape[0]
                      * beta.components * alpha.components * size
                      for beta in asm.fields for alpha in asm.fields)
    load_bytes = sum(grouping.vector_projector[f.trial].from_shape[0] * f.components * size
                     for f in asm.fields)
    global_bytes = sum(grouping.pair_projector[f"{beta.trial}__{alpha.trial}"].to_shape[0]
                       * beta.components * alpha.components * size
                       for beta in asm.fields for alpha in asm.fields)
    connectivity_bytes = sum(tensor.numel() * tensor.element_size()
                             for buffers in (grouping.cells, grouping.elements,
                                              grouping.field_conn, grouping.orientation,
                                              grouping.table_index)
                             for tensor in buffers.values())
    largest_gradients = 0
    largest_reference_gather = 0
    cell_batches = {}
    for kind in grouping.element_types:
        n_cells = len(grouping.elements[kind])
        cells = min(n_cells, element_batch_size or n_cells)
        cell_batches[kind] = (n_cells + cells - 1) // cells
        nq = asm.transformation[kind].n_quadrature
        nq = nq if batch_size in (-1, None) else min(nq, batch_size)
        width = sum(grouping.field_conn[f"{f.trial}__{kind}"].shape[1]
                    for f in asm.fields)
        largest_gradients = max(largest_gradients, 3 * cells * nq * width * size)
        modal_width = sum(grouping.field_conn[f"{f.trial}__{kind}"].shape[1]
                          for f in asm.fields if f.basis == "hierarchical")
        largest_reference_gather = max(largest_reference_gather,
                                       4 * cells * nq * modal_width * size)
    return {"oriented_reference_storage_bytes": sum(storage.values()),
            "projection_backend": "ReduceProjector",
            "all_declared_matrix_and_load_projection_bytes": projection_bytes,
            "all_declared_local_matrix_source_bytes": local_bytes,
            "all_declared_local_load_source_bytes": load_bytes,
            "all_declared_global_matrix_values_bytes": global_bytes,
            "global_load_vector_bytes": asm.n_dofs * size,
            "grouped_connectivity_and_orientation_bytes": connectivity_bytes,
            "physical_gradient_scratch_upper_estimate_bytes": largest_gradients,
            "reference_gather_scratch_upper_estimate_bytes": largest_reference_gather,
            "reference_contraction_index_bytes": sum(t.numel() * t.element_size()
                                                     for t in grouping.reference_plans.buffers()),
            "cell_batches_per_quadrature_sweep": cell_batches,
            "reference_tables_have_no_element_axis": all(t.ndim == 3 for t in grouping.ref_val.values())
            and all(t.ndim == 4 for t in grouping.ref_grad.values())}


def optional_solve(asm, matrix, load, device, repeats, warmup):
    """Homogeneous Dirichlet Poisson solve, excluding assembly/constraint setup."""
    condenser = Condenser(asm.layout.boundary_mask("u"))
    reduced, rhs = condenser(matrix, load)
    if rhs.numel() == 0:
        return {"status": "no_free_dofs", "free_dofs": 0}
    if device.type == "cpu":
        csc, data = reduced.to_scipy_coo().tocsc(), rhs.detach().numpy()

        def solve():
            return torch.from_numpy(scipy.sparse.linalg.splu(csc).solve(data))

        solver = "SciPy fresh sparse LU factorization + solve; CSC conversion excluded"
    else:
        tolerance = 1e-6 if asm.dtype == torch.float32 else 1e-12

        def solve():
            return reduced.solve(rhs, backend="pytorch", method="cg",
                                 atol=tolerance, maxiter=10000)

        solver = "torch-sla PyTorch CG; maxiter=10000"
    solution, first = timed_call(solve, device)
    residual = float(torch.linalg.vector_norm(reduced @ solution - rhs)
                     / torch.linalg.vector_norm(rhs).clamp_min(torch.finfo(rhs.dtype).tiny))
    if not np.isfinite(residual) or residual > (1e-4 if asm.dtype == torch.float32 else 1e-8):
        raise AssertionError(f"solve relative residual {residual} is too large")
    del solution
    return {"status": "solved", "solver": solver, "free_dofs": rhs.numel(),
            "relative_residual": residual,
            "timing": measure(solve, device, repeats, warmup, first)}


def benchmark_case(mesh, p, form, family, device, repeats, warmup, batch_size,
                   element_batch_size, quadrature_order, quadrature_extra, solve,
                   reference_mode="auto"):
    degree = quadrature_order or max(2 * p, 3 if form == "field_data" else 1) + quadrature_extra
    # Corrected quadrature dispatch honors the requested degree for both bases.
    asm, setup = construct(mesh, p, form, family, degree, device)
    chunks = {"batch_size": batch_size}
    if asm.grouped_tetrahedra:
        chunks["element_batch_size"] = element_batch_size
        chunks["reference_mode"] = reference_mode
    fields = data_for(asm, form)
    matrix_fn = lambda: asm(field_data=fields, **chunks)
    load_fn = lambda: asm.assemble_vector(field_data=fields, **chunks)
    matrix, matrix_first = timed_call(matrix_fn, device)
    load, load_first = timed_call(load_fn, device)
    numerical = numerical_diagnostics(asm, matrix, load, form)
    reconstruction = {"status": "not_grouped"}
    first_stages = {}
    if asm.grouped_tetrahedra:
        tolerance = 3e-5 if asm.dtype == torch.float32 else 2e-12
        for name, is_matrix, expected in (("matrix", True, matrix), ("load", False, load)):
            result, first_stages[name] = reconstruct(asm, is_matrix, field_data=fields, **chunks)
            if is_matrix:
                torch.testing.assert_close(result.row, expected.row)
                torch.testing.assert_close(result.col, expected.col)
                result, expected = result.values, expected.values
            torch.testing.assert_close(result, expected, atol=tolerance, rtol=tolerance)
            del result
        del expected
        reconstruction = {"status": "passed", "matrix_and_load_match_public_assembly": True}
    solved = (optional_solve(asm, matrix, load, device, repeats, warmup)
              if solve and form == "poisson" else {"status": "not_requested_for_this_form"})
    del matrix, load
    timings = {"full_matrix_assembly": measure(matrix_fn, device, repeats, warmup, matrix_first),
               "full_load_assembly": measure(load_fn, device, repeats, warmup, load_first)}
    if asm.grouped_tetrahedra:
        for name, is_matrix in (("matrix", True), ("load", False)):
            timings[f"{name}_stages"] = measure_stages(
                lambda: reconstruct(asm, is_matrix, field_data=fields, **chunks),
                repeats, warmup, first_stages[name])
    changing = None
    if form == "field_data":
        states = [data_for(asm, form, shift) for shift in (0.0, 0.5)]
        diagnostics = []
        for shift, state in zip((0.0, 0.5), states):
            matrix = asm(field_data=state, **chunks)
            load = asm.assemble_vector(field_data=state, **chunks)
            diagnostics.append(numerical_diagnostics(asm, matrix, load, form, shift))
            del matrix, load
        counter = itertools.count()

        def reassemble():
            state = states[next(counter) % len(states)]
            return asm(field_data=state, **chunks), asm.assemble_vector(field_data=state, **chunks)

        changing = {"coefficient": "rho = x + shift; diffusivity = 1 + rho**2",
                    "shifts": [0.0, 0.5], "diagnostics": diagnostics,
                    "matrix_and_load_timing": measure(reassemble, device, repeats, warmup)}
    quadrature = {}
    for kind, transformation in asm.transformation.items():
        weights, points = transformation.quadrature
        digest = hashlib.sha256(weights.detach().cpu().numpy().tobytes()
                                + points.detach().cpu().numpy().tobytes()).hexdigest()
        quadrature[kind] = {"requested_exactness_degree": degree,
                            "assembler_request": degree,
                            "actual_rule_index": transformation.quadrature_order,
                            "points": transformation.n_quadrature, "sha256": digest}
    histogram = ({str(g.class_id): g.n_cells for g in asm.grouping.groups}
                 if asm.grouped_tetrahedra else None)
    params = asm._classify_params(asm.forward, {"x": asm._points}, {}, {}, fields)
    return {"status": "passed", "p": p, "form": form, "family": family,
            "device": str(device), "dtype": str(asm.dtype).removeprefix("torch."),
            "n_dofs": asm.n_dofs, "fields": [vars(f) for f in asm.fields],
            "batch_size": batch_size, "element_batch_size": chunks.get("element_batch_size"),
            "cell_batching": ("original_order_across_orientations" if asm.grouped_tetrahedra
                              else "all_cells_per_element_type"),
            "reference_integration": {
                name: reference_report(asm, fn, matrix, batch_size, element_batch_size,
                                       reference_mode, fields)
                for name, fn, matrix in (("matrix", asm.forward, True),
                                        ("load", asm.forward_vector, False))},
            "quadrature": quadrature, "orientation_group_histogram": histogram,
            "executed_blocks": [f"{b.trial}__{a.trial}"
                                for a, b in asm._executed_pairs(asm.fields, params)],
            "autograd_enabled": torch.is_grad_enabled(), "setup": setup,
            "storage": storage_report(asm, batch_size, element_batch_size),
            "timings": timings, "reconstruction": reconstruction,
            "numerical": numerical, "changing_field_data": changing, "solve": solved}


def run_benchmark(degrees=(1, 2, 3, 4, 6), subdivisions=(1,), forms=FORMS,
                  orientations=("balanced",), device="cpu", dtype="float64", repeats=3,
                  warmup=1, batch_size=16, element_batch_size=32, quadrature_order=None,
                  quadrature_extra=0, threads=1, compare_lagrange=False, solve=False,
                  reference_mode="auto"):
    """Return a JSON-compatible report; unsupported hardware is explicit."""
    if (not degrees or any(isinstance(p, bool) or not isinstance(p, int) or p < 1 for p in degrees)
            or not subdivisions or any(n < 1 for n in subdivisions)
            or not forms or any(form not in FORMS for form in forms)
            or not orientations or any(o not in ("balanced", "single", "skewed") for o in orientations)
            or repeats < 1 or warmup < 0 or threads < 1 or quadrature_extra < 0
            or (quadrature_order is not None and quadrature_order < 1)):
        raise ValueError("invalid benchmark configuration")
    GroupedTetrahedra.validate_batches(batch_size, element_batch_size)
    GroupedTetrahedra.shares_reference([], reference_mode)
    if dtype not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("benchmark supports CPU and CUDA")
    root = Path(__file__).resolve().parents[2]
    sources = ["examples/poisson/benchmark_hierarchical_tetrahedra.py",
               "examples/poisson/_tetrahedron_reference_report.py",
               "tensormesh/assemble/_tetrahedron_groups.py", "tensormesh/assemble/mixed_assembler.py",
               "tensormesh/assemble/topology.py", "tensormesh/assemble/projector.py",
               "tensormesh/element/hierarchical_tetrahedron.py", "tensormesh/element/quadrature.py"]
    report = {"schema_version": 1, "status": "completed", "gpu_performance_measured": False,
        "configuration": {
        "degrees": list(degrees), "subdivisions": list(subdivisions), "forms": list(forms),
        "orientations": list(orientations), "device": str(device), "dtype": dtype,
        "repeats": repeats, "warmup": warmup, "threads": threads, "batch_size": batch_size,
        "element_batch_size": element_batch_size, "quadrature_order": quadrature_order,
        "reference_mode": reference_mode,
        "quadrature_extra": quadrature_extra, "compare_lagrange": compare_lagrange,
        "solve": solve, "autograd_enabled": False}, "environment": {
            "platform": platform.platform(), "processor": platform.processor(),
             "python": platform.python_version(), "torch": torch.__version__,
             "torch_sla": version("torch-sla"),
            "numpy": np.__version__, "scipy": scipy.__version__, "meshio": meshio.__version__,
            "cuda_runtime": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "device_model": torch.cuda.get_device_name(device) if device.type == "cuda"
            and torch.cuda.is_available() else platform.machine()},
        "source_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in sources}, "meshes": [], "cases": [],
        "methodology": [
            "CPU construction then complete module transfer; one case is retained at a time.",
            "First-call and warmed assembly times are separate; process caches are not cleared.",
            "CUDA wall timers synchronize before and after every timed call.",
            "Streamed stage timers synchronize each stage; their sums are not throughput estimates.",
            "Setup timers instrument actual construction, including eager geometry value tables.",
            "GPU peaks are separate warm-allocator calls, not driver memory or CPU RSS.",
            "Analytic energies/load integrals and reconstruction run outside public assembly timings.",
            "Lagrange comparisons are limited to supported P1/P2 topology and identical rules.",
            "No timing ratios or hardware performance acceptance thresholds are asserted."]}
    if device.type == "cuda" and not torch.cuda.is_available():
        report.update(status="unavailable", reason="CUDA unavailable; no GPU results measured")
        return report
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.no_grad():
            for n, orientation in itertools.product(subdivisions, orientations):
                mesh = cube_mesh(n, orientation, getattr(torch, dtype))
                mesh_id = f"cube_{n}_{orientation}"
                report["meshes"].append({"id": mesh_id, "subdivisions": n,
                    "orientation_distribution": orientation, "vertices": mesh.n_points,
                    "tetrahedra": len(mesh.elements()), "geometry_order": 1,
                    "sha256": hashlib.sha256(mesh.points.numpy().tobytes()
                                             + mesh.elements().numpy().tobytes()).hexdigest()})
                families = ("hierarchical", "lagrange") if compare_lagrange else ("hierarchical",)
                for p, form, family in itertools.product(degrees, forms, families):
                    identity = {"mesh_id": mesh_id, "p": p, "form": form, "family": family}
                    if family == "lagrange" and p > 2:
                        report["cases"].append({**identity, "status": "unsupported",
                            "reason": "generalized tetrahedral Lagrange face topology supports only P1/P2"})
                        continue
                    print(f"{mesh_id}: p={p}, {form}, {family}", file=sys.stderr, flush=True)
                    try:
                        case = benchmark_case(mesh, p, form, family, device, repeats, warmup,
                                              batch_size, element_batch_size, quadrature_order,
                                              quadrature_extra, solve, reference_mode)
                    except (torch.cuda.OutOfMemoryError, MemoryError) as exc:
                        case = {"status": "out_of_memory", "reason": str(exc)}
                    except (AssertionError, RuntimeError) as exc:
                        case = {"status": "failed", "reason": str(exc)}
                    report["cases"].append({**identity, **case})
                    if case["status"] in ("failed", "out_of_memory"):
                        report["status"] = "incomplete"
                    # BlockLayout keeps an assembler back-reference. Release that
                    # cycle before constructing another large projection layout.
                    gc.collect()
                    if case["status"] == "out_of_memory" and device.type == "cuda":
                        torch.cuda.empty_cache()
                del mesh
    finally:
        torch.set_num_threads(previous_threads)
    report["gpu_performance_measured"] = device.type == "cuda" and any(
        case["status"] == "passed" for case in report["cases"])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degrees", type=int, nargs="+", default=[1, 2, 3, 4, 6])
    parser.add_argument("--extended", action="store_true", help="also run p=8,10")
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[1])
    parser.add_argument("--forms", choices=FORMS, nargs="+", default=list(FORMS))
    parser.add_argument("--orientations", choices=["balanced", "single", "skewed"],
                        nargs="+", default=["balanced"])
    parser.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16, help="quadrature points; -1 = all")
    parser.add_argument("--element-batch-size", type=int, default=32,
                        help="cells across orientations; 0 = all cells of each element type")
    parser.add_argument("--reference-mode", choices=("auto", "cell"), default="auto",
                        help="auto shares eligible integrands by orientation; cell disables sharing")
    parser.add_argument("--quadrature-order", type=int)
    parser.add_argument("--quadrature-extra", type=int, default=0)
    parser.add_argument("--compare-lagrange", action="store_true")
    parser.add_argument("--solve", action="store_true", help="separate homogeneous Poisson solve")
    parser.add_argument("--output", type=Path)
    args = vars(parser.parse_args(argv))
    output = args.pop("output")
    if args.pop("extended"):
        args["degrees"] = list(dict.fromkeys([*args["degrees"], 8, 10]))
    if args["element_batch_size"] == 0:
        args["element_batch_size"] = None
    report = run_benchmark(**args)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text, end="")
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
