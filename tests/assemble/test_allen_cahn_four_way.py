"""Independent quadrature and numerical-gating checks for Allen-Cahn timings."""
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse


SCRIPT = (Path(__file__).resolve().parents[2] / "examples/diffusion/allen-cahn/benchmark_four_way.py")
spec = importlib.util.spec_from_file_location("allen_cahn_four_way", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("order", [4, 5, 6, 8])
def test_committed_euler_inputs_are_self_contained(order):
    root = Path(__file__).resolve().parents[2]
    with np.load(root / f"cluster/inputs/p{order}/input.npz") as data:
        assert int(data["order"]) == order
        assert float(data["chara_length"]) == 0.05
        assert data["cells"].shape == (936, (order + 1) * (order + 2) // 2)
        assert data["points"].shape[1] == 2
        assert data["initial"].shape == (len(data["points"]),)
        assert data["cells"].min() >= 0 and data["cells"].max() < len(data["points"])
        assert np.isfinite(data["points"]).all() and np.isfinite(data["initial"]).all()
        weights, points = benchmark.triangle_rule(4 * order)
        np.testing.assert_allclose(data["weights"], weights, atol=1e-15, rtol=1e-13)
        np.testing.assert_allclose(data["quadrature_points"], points, atol=1e-15, rtol=1e-13)


@pytest.mark.parametrize("degree", [8, 12, 16, 20, 24, 32])
def test_rule_integrates_all_nonlinear_terms(degree):
    weights, points = benchmark.triangle_rule(degree)
    assert len(weights) == ((degree + 2) // 2) ** 2
    assert (weights > 0).all()
    assert (points > 0).all() and (points.sum(1) < 1).all()
    for total in range(degree + 1):
        for a in range(total + 1):
            b = total - a
            exact = math.factorial(a) * math.factorial(b) / math.factorial(total + 2)
            actual = weights @ (points[:, 0] ** a * points[:, 1] ** b)
            assert actual == pytest.approx(exact, rel=3e-13, abs=1e-17)


def make_case(tmp_path, label, offset=0.0):
    prefix = str(tmp_path / label)
    sparse.save_npz(prefix + "-initial-matrix.npz", sparse.eye(2, format="csr"))
    np.savez(prefix + "-fields.npz", history=np.array([[1., 2.], [2., 3.]]) + offset,
             initial_residual=np.array([3., 4.]))
    return {"label": label, "trial": 1, "artifact": prefix, "input_sha256": "mesh-and-source",
            "quadrature_sha256": "degree-12", "dofs": 2, "triangles": 1}


def test_four_way_comparison_rejects_different_inputs_and_fields(tmp_path):
    reference = make_case(tmp_path, "upstream-element")
    same = make_case(tmp_path, "current-mixed")
    assert all(check["passed"] for check in benchmark.compare_cases([reference, same]))
    assert not benchmark.compare_cases([
        reference, {**same, "quadrature_sha256": "degree-2"}])[1]["passed"]
    changed = make_case(tmp_path, "different-field", offset=0.01)
    check = benchmark.compare_cases([reference, changed])[1]
    assert not check["passed"]
    assert check["max_step_relative_field_error"] > 1e-4


def test_current_must_be_hierarchical_without_silent_nodal_fallback():
    cases = [{"label": label, "basis": basis} for label, basis in (
        ("upstream-element", "lagrange"), ("upstream-mixed", "lagrange"),
        ("current-element", "hierarchical"), ("current-mixed", "hierarchical"))]
    benchmark.check_basis_selection(cases)
    with pytest.raises(RuntimeError, match="current-element must use hierarchical"):
        benchmark.check_basis_selection([{**cases[2], "basis": "lagrange"}])
    with pytest.raises(RuntimeError, match="upstream-mixed must use lagrange"):
        benchmark.check_basis_selection([{**cases[1], "basis": "hierarchical"}])


def test_comparison_checks_modal_matrix_and_load_under_change_of_basis(tmp_path):
    reference = make_case(tmp_path, "upstream-element")
    modal = {**make_case(tmp_path, "current-element"), "basis": "hierarchical"}
    transform = sparse.csr_matrix([[1., 0.], [1., 2.]])
    sparse.save_npz(modal["artifact"] + "-modal-to-nodal.npz", transform)
    sparse.save_npz(modal["artifact"] + "-initial-matrix.npz", transform.T @ transform)
    np.savez(modal["artifact"] + "-fields.npz", history=np.array([[1., 2.], [2., 3.]]),
             initial_residual=transform.T @ np.array([3., 4.]))
    assert all(check["passed"] for check in benchmark.compare_cases([reference, modal]))
    # Raw matrices are deliberately different: direct coefficient comparison
    # would incorrectly reject two equivalent physical discretizations.
    assert not np.allclose((transform.T @ transform).toarray(), np.eye(2))


def test_comparison_rejects_missing_time_steps(tmp_path):
    reference = make_case(tmp_path, "upstream-element")
    truncated = make_case(tmp_path, "truncated")
    np.savez(truncated["artifact"] + "-fields.npz", history=np.array([[1., 2.]]),
             initial_residual=np.array([3., 4.]))
    with pytest.raises(RuntimeError, match="different shapes"):
        benchmark.compare_cases([reference, truncated])


@pytest.mark.parametrize("method", ["lu", "cg-jacobi", "torch-cg-jacobi"])
def test_linear_solver_reports_true_residual(method):
    import torch
    from torch_sla import SparseTensor

    matrix = SparseTensor(torch.tensor([2., 3., 4.], dtype=torch.float64),
                          torch.arange(3), torch.arange(3), (3, 3))
    rhs = torch.tensor([2., 6., 12.], dtype=torch.float64)
    solution, info = benchmark.solve_increment(
        matrix, rhs, {"method": method, "rtol": 1e-10, "atol": 1e-14, "maxiter": 50})
    torch.testing.assert_close(solution, torch.tensor([1., 2., 3.], dtype=torch.float64))
    assert info["converged"]
    assert info["residual_norm"] < 1e-10
    expected = {"lu": 0, "cg-jacobi": 1, "torch-cg-jacobi": None}
    assert info["iterations"] == expected[method]


def test_cg_does_not_silently_accept_nonconvergence():
    import torch
    from torch_sla import SparseTensor

    matrix = SparseTensor(torch.tensor([2., -1., -1., 3.], dtype=torch.float64),
                          torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 0, 1]), (2, 2))
    with pytest.raises(RuntimeError, match="CG failed"):
        benchmark.solve_increment(matrix, torch.ones(2, dtype=torch.float64),
                                  {"method": "cg-jacobi", "rtol": 1e-14,
                                   "atol": 0., "maxiter": 1})


def test_cuda_timing_synchronizes_before_and_after(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    result, seconds = benchmark.timed(lambda: calls.append("operation"), torch.device("cuda"))
    assert result is None and seconds >= 0
    assert calls == ["sync", "operation", "sync"]


def test_scipy_solver_refuses_non_cpu_tensors():
    import torch

    with pytest.raises(ValueError, match="CPU-only"):
        benchmark.solve_increment(None, torch.ones(2, device="meta"), {"method": "lu"})


def test_existing_native_cg_on_cuda_when_available():
    import torch
    from torch_sla import SparseTensor

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    matrix = SparseTensor(torch.tensor([2., 3.], dtype=torch.float64, device="cuda"),
                          torch.arange(2, device="cuda"), torch.arange(2, device="cuda"), (2, 2))
    rhs = torch.tensor([2., 6.], dtype=torch.float64, device="cuda")
    solution, info = benchmark.solve_increment(
        matrix, rhs, {"method": "torch-cg-jacobi", "rtol": 1e-10, "atol": 1e-14, "maxiter": 50})
    assert solution.is_cuda and info["converged"]
    torch.testing.assert_close(solution.cpu(), torch.tensor([1., 2.], dtype=torch.float64))
