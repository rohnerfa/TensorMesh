"""Opt-in diagnostics around actual MixedElementAssembler calls.

Import TensorMesh only inside functions, after the isolated worker has selected
its checkout. Instrumentation never changes production source or tensor kernels.
"""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
import statistics
from time import perf_counter
from unittest.mock import patch


def _synchronize(device):
    if device.type == "cuda":
        import torch
        torch.cuda.synchronize(device)


def _summary(samples):
    return {"samples_seconds": samples, "median_seconds": statistics.median(samples),
            "min_seconds": min(samples), "max_seconds": max(samples)}


class _Stages:
    def __init__(self, device, trace=False):
        self.device, self.trace = device, trace
        self.seconds, self.calls = defaultdict(float), defaultdict(int)
        self.stack = []

    @contextmanager
    def section(self, name):
        if self.trace:
            from torch.profiler import record_function
            with record_function("assembly::" + name):
                yield
            return
        _synchronize(self.device)
        frame = [perf_counter(), 0.0]
        self.stack.append(frame)
        try:
            yield
        finally:
            _synchronize(self.device)
            elapsed = perf_counter() - frame[0]
            self.stack.pop()
            self.seconds[name] += elapsed - frame[1]
            self.calls[name] += 1
            if self.stack:
                self.stack[-1][1] += elapsed

    def wrap(self, name, fn):
        def measured(*args, **kwargs):
            with self.section(name):
                return fn(*args, **kwargs)
        return measured

    def generator(self, name, fn):
        def measured(*args, **kwargs):
            iterator = iter(fn(*args, **kwargs))
            while True:
                with self.section(name):
                    try:
                        value = next(iterator)
                    except StopIteration:
                        return
                # Consumer work must not be charged to generator preparation.
                yield value
        return measured


def _projectors(asm):
    return (asm.grouping.pair_projector if getattr(asm, "grouped_tetrahedra", False)
            else asm.pair_projector)


@contextmanager
def _projector_calls(asm, callback):
    """Special methods are resolved on the class; intercept only these instances."""
    projectors = _projectors(asm)
    keys = {id(projector): key for key, projector in projectors.items()}
    with ExitStack() as stack:
        for cls in {type(projector) for projector in projectors.values()}:
            original = cls.__call__

            def call(projector, value, original=original):
                key = keys.get(id(projector))
                if key is None:
                    return original(projector, value)
                return callback(key, original, projector, value)

            stack.enter_context(patch.object(cls, "__call__", call))
        yield


@contextmanager
def _instrument(asm, stages):
    with ExitStack() as stack:
        for method, name in (
                ("_normalize_inputs", "input_validation"),
                ("type", "module_dtype_conversion"),
                ("to", "module_device_conversion"),
                ("_classify_params", "signature_and_context"),
                ("_executed_pairs", "signature_and_context"),
                ("_pass_context", "signature_and_context"),
                ("_check_bilinear", "form_validation"),
                ("_run_pass", "weak_form_evaluation"),
                ("_coo_pattern", "sparse_pattern")):
            stack.enter_context(patch.object(asm, method, stages.wrap(name, getattr(asm, method))))
        grouped = getattr(asm, "grouped_tetrahedra", False)
        owner, method = (asm.grouping, "batches") if grouped else (asm, "_batches")
        stack.enter_context(patch.object(owner, method, stages.generator(
            "geometry_and_reference_tables", getattr(owner, method))))
        owner, method = (asm.grouping, "contract") if grouped else (asm, "_integrate_pair")
        stack.enter_context(patch.object(owner, method, stages.wrap(
            "quadrature_contraction", getattr(owner, method))))

        def project(key, original, projector, value):
            with stages.section("global_projection"):
                return original(projector, value)

        stack.enter_context(_projector_calls(asm, project))
        yield


def _check_matrix(matrix, reference):
    import torch
    for actual, expected in zip((matrix.row, matrix.col, matrix.values), reference):
        torch.testing.assert_close(actual.detach().cpu(), expected)


def _operator_profile(asm, fn, device, reference, trace_path):
    import torch

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    stages = _Stages(device, trace=True)
    _synchronize(device)
    with _instrument(asm, stages), torch.profiler.profile(
            activities=activities, record_shapes=True) as profiler:
        with stages.section("other_assembly"):
            result = fn()
        _synchronize(device)
    _check_matrix(result, reference)
    del result
    trace_path = Path(trace_path).resolve()
    profiler.export_chrome_trace(str(trace_path))
    operators = [{"name": item.key, "calls": item.count,
                  "self_cpu_us": item.self_cpu_time_total,
                  "self_device_us": item.self_device_time_total,
                  "input_shapes": item.input_shapes}
                 for item in profiler.key_averages(group_by_input_shape=True)
                 if not item.key.startswith("assembly::")]
    return {"trace_path": str(trace_path), "profiled_calls": 1,
            "top_self_cpu": sorted(operators, key=lambda x: x["self_cpu_us"], reverse=True)[:20],
            "top_self_device": sorted(operators, key=lambda x: x["self_device_us"], reverse=True)[:20]
            if device.type == "cuda" else []}


def _projection_probe(asm, fn, device, repeats, warmup):
    """Compare CSR/reduction with identical contributions and destination indices."""
    import torch
    from tensormesh.assemble.projector import ReduceProjector, SparseProjector

    inputs = {}

    def capture(key, original, projector, value):
        inputs[key] = value.detach()
        return original(projector, value)

    with _projector_calls(asm, capture):
        result = fn()
    del result
    reports = {}
    for key, source in inputs.items():
        actual = _projectors(asm)[key]
        count = 1
        for size in actual.from_shape:
            count *= size
        if isinstance(actual, SparseProjector):
            csr = actual
            projection = csr.projection
            columns = projection.col_indices()
            torch.testing.assert_close(projection.values(), torch.ones_like(projection.values()))
            torch.testing.assert_close(torch.bincount(columns, minlength=count),
                                       torch.ones(count, dtype=torch.long, device=device))
            rows = torch.repeat_interleave(
                torch.arange(projection.shape[0], device=device), projection.crow_indices().diff())
            indices = torch.empty(count, dtype=torch.long, device=device)
            indices[columns] = rows
            reduce = ReduceProjector(indices, actual.from_shape, actual.to_shape)
        elif isinstance(actual, ReduceProjector):
            reduce = actual
            csr = SparseProjector(torch.arange(count, device=device), actual.indices,
                                  actual.from_shape, actual.to_shape, dtype=source.dtype).to(device)
        else:
            raise TypeError(f"unsupported projector {type(actual).__name__}")
        torch.testing.assert_close(csr(source), reduce(source))
        projectors = {"csr": csr, "reduce": reduce}
        for _ in range(warmup):
            for projector in projectors.values():
                result = projector(source)
                del result
        samples = {name: [] for name in projectors}
        for iteration in range(repeats):
            order = ("csr", "reduce") if iteration % 2 == 0 else ("reduce", "csr")
            for name in order:
                _synchronize(device)
                start = perf_counter()
                result = projectors[name](source)
                _synchronize(device)
                samples[name].append(perf_counter() - start)
                del result
        reports[key] = {"production_projector": type(actual).__name__,
                        "input_shape": list(source.shape), "source_entries": count,
                        "output_shape": list(actual.to_shape), "equivalence": "passed",
                        "timings": {name: _summary(values) for name, values in samples.items()}}
    return reports


def profile_matrix(asm, fn, device, repeats, warmup, trace_path):
    """Profile warmed public calls after ordinary throughput/memory measurements.

    Stage wall times synchronize at boundaries, perturbing launch/overlap. A
    separate annotated profiler trace has no intermediate synchronizations.
    Neither diagnostic replaces the uninstrumented whole-assembly timings.
    """
    with_matrix = fn()
    reference = tuple(t.detach().cpu().clone()
                      for t in (with_matrix.row, with_matrix.col, with_matrix.values))
    del with_matrix
    samples, counts, totals = defaultdict(list), {}, []
    for iteration in range(warmup + repeats):
        stages = _Stages(device)
        with _instrument(asm, stages), stages.section("other_assembly"):
            result = fn()
        _check_matrix(result, reference)
        del result
        if iteration >= warmup:
            for name, elapsed in stages.seconds.items():
                samples[name].append(elapsed)
                counts[name] = stages.calls[name]
            totals.append(sum(stages.seconds.values()))
    operators = _operator_profile(asm, fn, device, reference, trace_path)
    projection = _projection_probe(asm, fn, device, repeats, warmup)
    return {"status": "passed", "output_equivalence": "passed",
            "stage_clock": "exclusive synchronized wall time; instrumentation perturbs execution",
            "other_assembly": "unattributed dispatch, accumulation, packing, sparse output and wrapper overhead",
            "stages": {name: {**_summary(values), "calls_per_assembly": counts[name]}
                       for name, values in samples.items()},
            "instrumented_total": _summary(totals), "operators": operators,
            "projection_probe_scope": "isolated projectors on identical local tensors; not whole-path speedup",
            "projection_probe": projection}
