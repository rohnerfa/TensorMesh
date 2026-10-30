"""Regression of the original example pipeline against the modal pipeline."""

import pytest

from examples.poisson.compare_hierarchical import run_comparison


@pytest.mark.parametrize("degree", [1, 3])
def test_original_poisson_example_agrees_with_hierarchical(degree):
    report, _ = run_comparison(degree=degree, length=0.3, modes=4)
    agreement = report["agreement"]
    assert agreement["relative_L2_field_difference"] < 1e-11
    assert agreement["relative_matrix_change_of_basis_error"] < 1e-11
    assert agreement["relative_load_change_of_basis_error"] < 1e-11
    assert agreement["boundary_dof_counts_equal"]
    assert report["analytic_PDE_relative_residual_at_check_points"] < 1e-12
    assert report["lagrange"]["relative_free_system_residual"] < 1e-12
    assert report["hierarchical"]["relative_free_system_residual"] < 1e-12
