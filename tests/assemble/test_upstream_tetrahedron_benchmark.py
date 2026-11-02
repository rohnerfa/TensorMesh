"""Independent mesh/quadrature checks and genuine process-isolated comparisons."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from examples.poisson._tetrahedron_comparison import (
    PINNED_UPSTREAM, high_order_cube, quadrature, reference_lattice,
)
from examples.poisson.benchmark_upstream_tetrahedra import (
    ROOT, inspect_reports, timing_pair, write_json,
)


@pytest.mark.parametrize("p", [3, 4, 5, 6])
def test_high_order_cube_has_shared_face_nodes_and_affine_internal_order(p):
    data = high_order_cube(2, p)
    cells, points = data["cells"], data["points"]
    bary = reference_lattice(p)
    assert len(points) == (2 * p + 1)**3
    assert len(np.unique(cells[:, :4].argsort(1), axis=0)) == 24
    mapped = np.einsum("bv,evd->ebd", bary / p, points[cells[:, :4]])
    np.testing.assert_allclose(points[cells], mapped, atol=2e-16)
    # Two neighbours must share the entire face-node set, not just its corners.
    faces = {}
    for conn in cells:
        for opposite in range(4):
            nodes = tuple(sorted(conn[bary[:, opposite] == 0]))
            key = tuple(sorted(np.delete(conn[:4], opposite)))
            faces.setdefault(key, []).append(nodes)
    interior = 0
    for occurrences in faces.values():
        assert len(occurrences) in (1, 2)
        if len(occurrences) == 2:
            assert occurrences[0] == occurrences[1]
            interior += 1
    assert interior > 0


@pytest.mark.parametrize("p", [3, 4, 6])
def test_comparison_quadrature_integrates_every_required_monomial(p):
    rule = quadrature(p)
    q, w = rule["quadrature_points"], rule["quadrature_weights"]
    assert (w > 0).all()
    for a in range(2*p + 1):
        for b in range(2*p + 1 - a):
            for c in range(2*p + 1 - a - b):
                exact = math.prod(math.factorial(k) for k in (a, b, c)) / math.factorial(a+b+c+3)
                actual = w @ np.prod(q**(a, b, c), axis=1)
                assert actual == pytest.approx(exact, rel=2e-13, abs=2e-17)


def upstream_checkout():
    path = os.environ.get("TENSORMESH_UPSTREAM")
    if not path:
        pytest.skip("set TENSORMESH_UPSTREAM to the pinned, clean upstream checkout")
    return Path(path).resolve()


def run_isolated(tmp_path, extra, expected_returncode=0):
    output = tmp_path / "comparison.json"
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    completed = subprocess.run([
        sys.executable, str(ROOT / "examples/poisson/benchmark_upstream_tetrahedra.py"),
        "--upstream", str(upstream_checkout()), "--subdivisions", "1", "--repeats", "1",
        "--warmup", "1", "--output", str(output), *extra],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert output.exists(), completed.stderr
    report = json.loads(output.read_text())
    assert completed.returncode == expected_returncode, (report, completed.stderr)
    return report


@pytest.mark.parametrize("upstream_assembler", ["element", "mixed"])
def test_real_upstream_p3_p4_p6_operators_and_import_isolation(tmp_path, capsys, upstream_assembler):
    report = run_isolated(tmp_path, ["--degrees", "3", "4", "6", "--forms", "mass", "poisson",
                                     "--upstream-assembler", upstream_assembler])
    assert report["upstream"]["commit"] == PINNED_UPSTREAM
    assert report["protocol"]["upstream_assembler"] == upstream_assembler
    assert report["protocol"]["hierarchical_element_batch_size"] is None
    assert report["protocol"]["hierarchical_cell_batching"] == "original_order_across_orientations"
    assert report["protocol"]["hierarchical_reference_mode"] == "auto"
    assert report["wall_seconds"] > 0
    assert report["status"] == "completed"
    assert len(report["validation"]) == len(report["comparisons"]) == 6
    assert not report["gpu_performance_measured"]
    expected = "MixedElementAssembler" if upstream_assembler == "mixed" else "ElementAssembler"
    for check in report["validation"]:
        assert check["upstream_assembler"] == upstream_assembler
        assert check["passed"]
        assert check["matrix_change_of_basis_error"] < 2e-8
        assert check["load_change_of_basis_error"] < 2e-8
    for case in report["cases"]:
        assert case["numerical"]["passed"]
        assert case["element_batch_size"] is None
        assert case["worker_wall_seconds"] > 0
        if case["family"] == "lagrange":
            assert case["assembly_api"].startswith(expected)
        else:
            references = case["reference_integration"]
            assert references["matrix"]["strategy"] == (
                "orientation_shared" if case["form"] == "mass" else "cellwise")
            assert references["load"]["strategy"] == "orientation_shared"
        root = Path(case["source"]["root"]) / "tensormesh"
        assert all(Path(path).is_relative_to(root) for path in case["imported_modules"].values())
        assert case["source"]["root"] == report[
            "upstream" if case["family"] == "lagrange" else "feature"]["root"]
    inspect_reports([tmp_path / "comparison.json"])
    printed = capsys.readouterr().out
    assert "H change" in printed
    assert f"L: upstream {expected}" in printed
    assert "Cell chunks: L all cells; H all cells, spanning orientations" in printed
    assert "H reference mode: auto" in printed
    assert "median of 1 warmed matrix calls" in printed
    assert all(c["upstream_assembler"] == upstream_assembler for c in report["comparisons"])
    pair = {c["family"]: c for c in report["cases"] if not c["validation"]
            and c["p"] == 3 and c["form"] == "mass"}
    # Numerically passing runs must still be rejected if the supplied rules differ.
    nodal = {**pair["lagrange"], "quadrature": {"sha256": "different_rule"}}
    comparison = timing_pair(nodal, pair["hierarchical"], {"passed": True})
    assert comparison["status"] == "not_comparable"
    assert "operations" not in comparison


@pytest.mark.parametrize("upstream_assembler", ["element", "mixed"])
def test_real_upstream_p5_chunked_quadrature_and_single_orientation(tmp_path, upstream_assembler):
    report = run_isolated(tmp_path, ["--degrees", "5", "--forms", "mass", "poisson", "--batch-size", "17",
                                     "--element-batch-size", "5",
                                     "--orientations", "balanced", "single",
                                     "--upstream-assembler", upstream_assembler])
    assert all(check["passed"] for check in report["validation"])
    assert len(report["comparisons"]) == 4
    for case in report["cases"]:
        assert case["batch_size"] == 17
        assert case["element_batch_size"] == (5 if case["family"] == "hierarchical" else None)
        assert case["numerical"]["passed"]


@pytest.mark.parametrize("reference_mode", ["auto", "cell"])
def test_real_upstream_mass_reports_shared_and_cellwise_integrand_storage(tmp_path, reference_mode):
    report = run_isolated(tmp_path, ["--upstream-assembler", "mixed", "--degrees", "4",
                                     "--forms", "mass", "--subdivisions", "2",
                                     "--batch-size", "17", "--element-batch-size", "37",
                                     "--reference-mode", reference_mode])
    assert report["protocol"]["hierarchical_reference_mode"] == reference_mode
    assert report["comparisons"][0]["hierarchical_reference_mode"] == reference_mode
    assert report["validation"][0]["passed"]
    for case in report["cases"]:
        if case["family"] != "hierarchical":
            continue
        reference = case["reference_integration"]["matrix"]
        assert reference["eligible"]
        sizes = reference["element_types"]["tetra35"]
        assert sizes["max_integrand_bytes"] == (24 if reference_mode == "auto" else 37) * 17 * 35**2 * 8
        assert sizes["cell_mode_max_integrand_bytes"] == 37 * 17 * 35**2 * 8


def test_known_p6_oom_case_is_skipped_without_starting_any_workers(tmp_path, capsys):
    report = run_isolated(tmp_path, ["--degrees", "6", "--forms", "poisson", "--subdivisions", "4"])
    assert report["status"] == "completed"
    assert report["cases"] == report["validation"] == []
    assert len(report["comparisons"]) == 1
    assert report["comparisons"][0]["status"] == "skipped"
    assert "operations" not in report["comparisons"][0]
    assert not list(Path(report["artifacts"]).iterdir())
    assert not report["gpu_performance_measured"]
    inspect_reports([tmp_path / "comparison.json"])
    printed = capsys.readouterr().out
    assert "P6 n=4 poisson balanced: skipped" in printed
    assert "0 isolated workers" in printed


def test_real_upstream_matrix_profiles_use_both_selected_checkouts(tmp_path, capsys):
    report = run_isolated(tmp_path, ["--upstream-assembler", "mixed", "--degrees", "3",
                                     "--forms", "mass", "--profile-matrix"])
    assert report["status"] == "completed"
    assert report["protocol"]["profile_matrix"]
    for case in report["cases"]:
        if case["validation"]:
            assert "matrix_profile" not in case
            continue
        profile = case["matrix_profile"]
        assert profile["output_equivalence"] == "passed"
        assert profile["stages"]["weak_form_evaluation"]["calls_per_assembly"] == 1
        expected = "SparseProjector" if case["family"] == "hierarchical" else "ReduceProjector"
        assert all(probe["production_projector"] == expected
                   for probe in profile["projection_probe"].values())
        assert Path(profile["operators"]["trace_path"]).is_file()
        root = Path(case["source"]["root"]) / "tensormesh"
        assert all(Path(path).is_relative_to(root) for path in case["imported_modules"].values())
    inspect_reports([tmp_path / "comparison.json"])
    printed = capsys.readouterr().out
    assert "Matrix stage diagnostics" in printed
    assert "same-input projection" in printed


def test_real_upstream_inaccurate_float32_p6_suppresses_ratios(tmp_path):
    report = run_isolated(tmp_path, ["--degrees", "6", "--forms", "poisson", "--dtype", "float32"],
                          expected_returncode=1)
    assert not report["validation"][0]["passed"]
    assert any(case["status"] == "numerical_failure" for case in report["cases"])
    assert all("operations" not in comparison for comparison in report["comparisons"])


def test_nonfinite_numerical_failure_remains_a_readable_report(tmp_path):
    path = tmp_path / "failed.json"
    write_json(path, {"status": "numerical_failure", "numerical": {"error": float("nan"),
                                                               "passed": False}})
    report = json.loads(path.read_text())
    assert report["status"] == "numerical_failure"
    assert report["numerical"] == {"error": None, "passed": False}


def test_failed_validation_never_produces_a_timing_ratio():
    case = {"p": 6, "form": "poisson", "subdivisions": 2,
            "orientation": "balanced", "status": "passed"}
    comparison = timing_pair(case, case, {"passed": False})
    assert comparison["status"] == "not_comparable"
    assert "operations" not in comparison


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("upstream_assembler", ["element", "mixed"])
def test_real_upstream_cuda_operator_equivalence_and_memory(tmp_path, upstream_assembler):
    report = run_isolated(tmp_path, ["--device", "cuda", "--degrees", "3", "--forms", "poisson",
                                     "--upstream-assembler", upstream_assembler])
    assert report["gpu_performance_measured"]
    for case in report["cases"]:
        if not case["validation"]:
            memory = case["timings"]["full_matrix_assembly"]["gpu_memory"]
            assert memory["peak_allocated_bytes"] > 0
