"""Check benchmark decomposition/diagnostics, never assert timing speedups."""

import json

import pytest
import torch

from examples.poisson.benchmark_hierarchical import (
    cell_loop_iterators, cold_basis, basis_diagnostics, markdown_report, run_benchmark,
)


def test_small_benchmark_runs_and_reconstructs_public_assembly():
    report = run_benchmark(degrees=(1, 2, 4), length=0.6, repeats=2, warmup=1,
                           batch_size=8, extra=4)
    assert len(report["cases"]) == 6
    assert report["configuration"]["device"] == "cpu"
    assert report["source_sha256"]
    assert report["acceptance"]["requested_cpu_numerical_acceptance_passed"]
    assert not report["acceptance"]["gpu_peak_memory_measured"]
    for row in report["cases"]:
        assert row["basis_accuracy"]["passes_accuracy_screen"]
        assert row["numerical_solution"]["status"] == "solved"
        assert row["numerical_solution"]["conditioning"]["SPD_resolved_at_float64_precision"]
        assert row["numerical_solution"]["relative_linear_system_residual"] < 1e-10
        for values in row["timings"].values():
            assert len(values["samples_seconds"]) == 2
            assert values["median_seconds"] > 0
            assert values["gpu_memory"]["status"] == "not_measured"
        if row["family"] == "hierarchical":
            audit = row["hotpath_audit"]
            assert not audit["direct_cell_loops_detected"]
            assert audit["hierarchical_recurrence_blocked_during_matrix_and_load_assembly"]
            assert audit["reference_tables_have_no_element_axis"]
            assert audit["orientation_precomputed"]
    assert "Numerical accuracy and conditioning" in markdown_report(report)
    json.dumps(report, allow_nan=False)


def test_high_order_modal_basis_accuracy_diagnostics():
    basis, _ = cold_basis(10, "hierarchical", torch.device("cpu"))
    result = basis_diagnostics(10, "hierarchical", basis)
    assert result["sampled_rank"] == result["basis_count"] == 66
    assert result["passes_accuracy_screen"]
    assert result["Pp_reproduction_max_error"] < 1e-11


def test_loop_audit_distinguishes_cells_from_types_and_batches():
    assert not cell_loop_iterators("for element_type in self.element_types:\n    pass")
    assert not cell_loop_iterators("for i in range(n_batch):\n    pass")
    assert cell_loop_iterators("for cell in self.elements[kind]:\n    pass")
    assert cell_loop_iterators("for e in range(n_element):\n    pass")


def test_no_fabricated_cuda_or_mps_float64_results():
    with pytest.raises(ValueError, match="MPS"):
        run_benchmark(device="mps")
    if not torch.cuda.is_available():
        with pytest.raises(ValueError, match="CUDA unavailable"):
            run_benchmark(device="cuda")
