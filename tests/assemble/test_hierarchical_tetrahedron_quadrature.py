"""High-p tetrahedral integration with analytic moments and refinement checks."""

import math

import pytest
import torch

from tensormesh import Field, MixedElementAssembler
from tensormesh.element import Tetrahedron
from tests.assemble._tetrahedron_fixtures import (
    DTYPE, cell_tables, independent_quadrature, orientation_mesh, patch_mesh,
    poisson, scalar_reference,
)


def _moment(a, b=0, c=0):
    return math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)


@pytest.mark.parametrize("p", [3, 4, 5, 6])
def test_default_rule_integrates_degree_p_polynomial_mass_stiffness_and_load(p):
    # Class 17 makes physical x differ from the first reference coordinate.
    # P3 specifically audits the grouped promotion of defective legacy rules.
    mesh = orientation_mesh([17])
    asm = poisson(p).from_mesh(mesh)
    refined = poisson(p).from_mesh(mesh, quadrature_order=2 * p + 4)
    if p >= 4:
        assert asm.transformation["tetra"].quadrature_order == 2 * p
        assert asm.transformation["tetra"].n_quadrature == (p + 1) ** 3
    mass = asm(func=lambda u, v: u * v, batch_size=37).to_dense()
    stiffness = asm(batch_size=37).to_dense()
    load = asm.assemble_vector(batch_size=37)
    expected_mass, expected_stiffness, expected_load = scalar_reference(asm)
    torch.testing.assert_close(mass, expected_mass, atol=3e-14, rtol=4e-11)
    torch.testing.assert_close(stiffness, expected_stiffness, atol=4e-13, rtol=4e-11)
    torch.testing.assert_close(load, expected_load, atol=3e-14, rtol=4e-11)
    torch.testing.assert_close(refined(func=lambda u, v: u * v, batch_size=37).to_dense(),
                               mass, atol=3e-14, rtol=4e-11)
    torch.testing.assert_close(refined(batch_size=37).to_dense(), stiffness,
                               atol=4e-13, rtol=4e-11)
    assert torch.linalg.eigvalsh(mass).min() > 0

    # Fit x^p on an independent integer barycentric lattice. No high-order
    # Lagrange factory, topology, or field-evaluation helper is needed.
    reference = torch.tensor([(i, j, k) for i in range(p + 1)
                              for j in range(p + 1 - i)
                              for k in range(p + 1 - i - j)], dtype=DTYPE) / p
    assert len(reference) == math.comb(p + 3, 3)
    weights = torch.ones(len(reference), dtype=DTYPE)
    physical, _, tables, _ = cell_tables(asm, 0, reference, weights)
    local = torch.linalg.solve(tables["u"][0], physical[:, 0] ** p)
    coefficients = torch.zeros(asm.n_dofs, dtype=DTYPE)
    coefficients[asm.field_conn["u__tetra"][0]] = local
    _, independent = independent_quadrature(2 * p + 2)
    x, _, independent_tables, _ = cell_tables(
        asm, 0, independent, torch.ones(len(independent), dtype=DTYPE))
    torch.testing.assert_close(independent_tables["u"][0] @ local, x[:, 0] ** p,
                               atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(coefficients @ mass @ coefficients,
                               torch.tensor(_moment(2 * p), dtype=DTYPE),
                               atol=2e-13, rtol=3e-10)
    torch.testing.assert_close(coefficients @ stiffness @ coefficients,
                               torch.tensor(p * p * _moment(2 * p - 2), dtype=DTYPE),
                               atol=2e-12, rtol=3e-10)
    exact_load = _moment(p) + _moment(p + 1) + _moment(p, 2) + 0.5 * _moment(p, 0, 1)
    torch.testing.assert_close(coefficients @ load, torch.tensor(exact_load, dtype=DTYPE),
                               atol=2e-13, rtol=3e-10)


@pytest.mark.parametrize("p", [4, 5, 6])
def test_minimal_affine_stiffness_degree_and_uneven_quadrature_chunks(p):
    mesh = patch_mesh()
    minimal = poisson(p).from_mesh(mesh, quadrature_order=2 * p - 2)
    exact = poisson(p).from_mesh(mesh)
    expected = exact(batch_size=43).to_dense()
    torch.testing.assert_close(minimal(batch_size=11, element_batch_size=1).to_dense(), expected,
                               atol=4e-13, rtol=4e-11)


def test_requested_degree_seven_exactness_despite_legacy_table_defect():
    mesh = patch_mesh()
    requested = poisson(3).from_mesh(mesh, quadrature_order=7)
    refined = poisson(3).from_mesh(mesh, quadrature_order=12)
    # P3 * P3 * x is degree seven. The legacy rule numbered 7 is only
    # degree-six exact, so the grouped path must select a sufficient rule.
    form = lambda u, v, x: (1 + x[0]) * u * v
    load = lambda v, x: x[0] ** 4 * v
    torch.testing.assert_close(requested(func=form).to_dense(), refined(func=form).to_dense(),
                               atol=3e-14, rtol=4e-11)
    torch.testing.assert_close(requested.assemble_vector(func=load),
                               refined.assemble_vector(func=load), atol=3e-14, rtol=4e-11)


@pytest.mark.parametrize("p", [4, 5, 6])
def test_quartic_bubble_manufactured_solve_and_exact_energy(p):
    class Bubble(MixedElementAssembler):
        fields = [Field("u", "v", p, basis="hierarchical")]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            return 2 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2]) * v

    asm = Bubble.from_mesh(orientation_mesh([23]))
    stiffness = asm(batch_size=37).to_dense()
    load = asm.assemble_vector(batch_size=37)
    conn = asm.field_conn["u__tetra"][0]
    # A single tetrahedron has only private volume modes free. This is an
    # assembly diagnostic independent of the field-helper boundary machinery.
    n_interior = math.comb(p - 1, 3)
    interior = conn[-n_interior:]
    reduced = stiffness[interior[:, None], interior[None, :]]
    assert torch.linalg.eigvalsh(reduced).min() > 0
    coefficients = torch.zeros(asm.n_dofs, dtype=DTYPE)
    coefficients[interior] = torch.linalg.solve(reduced, load[interior])
    _, reference = independent_quadrature(16)
    x, _, tables, _ = cell_tables(asm, 0, reference, torch.ones(len(reference), dtype=DTYPE))
    exact = (1 - x.sum(-1)) * x.prod(-1)
    torch.testing.assert_close(tables["u"][0] @ coefficients[conn], exact,
                               atol=3e-13, rtol=3e-11)
    torch.testing.assert_close((stiffness @ coefficients - load)[interior],
                               torch.zeros(n_interior, dtype=DTYPE), atol=3e-14, rtol=0)
    # Integral |grad(lambda0*lambda1*lambda2*lambda3)|^2 = 1/15120.
    torch.testing.assert_close(coefficients @ stiffness @ coefficients,
                               torch.tensor(1 / 15120, dtype=DTYPE), atol=2e-15, rtol=3e-11)


def test_quadratic_fixture_preserves_internal_nodes_and_valid_curved_map():
    mesh = patch_mesh(2, curved=True)
    affine = patch_mesh(2)
    _, reference = independent_quadrature(10)
    geom = Tetrahedron.get_basis_fns(2, DTYPE)
    values, gradients = geom.map(reference), geom.grad().map(reference)
    conn = mesh.elements()
    old = torch.einsum("qb,ebd->eqd", values, affine.points[conn])
    x, y, z = old.unbind(-1)
    expected = torch.stack((x + 0.12 * y * z, y + 0.10 * x * z, z + 0.08 * x * y), -1)
    actual = torch.einsum("qb,ebd->eqd", values, mesh.points[conn])
    torch.testing.assert_close(actual, expected, atol=3e-15, rtol=3e-14)
    jacobian = torch.einsum("ebi,qrb->eqir", mesh.points[conn], gradients)
    determinant = torch.linalg.det(jacobian)
    assert determinant.abs().min() > 0.5
    assert (determinant * determinant[:, :1] > 0).all()


def test_curved_variable_coefficient_matrix_and_load_quadrature_refinement():
    class Weighted(MixedElementAssembler):
        fields = [Field("u", "v", 4, basis="hierarchical")]

        def forward(self, u, v, gradu, gradv, x):
            return torch.exp(3 * x[0] + 2 * x[1] + x[2]) * ((gradu * gradv).sum() + u * v)

        def forward_vector(self, v, x):
            return torch.exp(3 * x[0] + 2 * x[1] + x[2]) * v

    matrices, loads = [], []
    for degree in [8, 12, 18, 24]:
        asm = Weighted.from_mesh(patch_mesh(2, curved=True), quadrature_order=degree)
        matrices.append(asm(batch_size=73, element_batch_size=1).to_dense())
        loads.append(asm.assemble_vector(batch_size=73, element_batch_size=1))
    for approximations in (matrices, loads):
        reference = approximations[-1]
        errors = [float(torch.linalg.vector_norm(value - reference)
                        / torch.linalg.vector_norm(reference)) for value in approximations[:-1]]
        assert errors[0] > 10 * errors[1], errors
        assert errors[1] < 1e-6, errors
        assert errors[2] < 1e-10, errors
    # Independent positive product rule, geometry inverse, and cell scatter at
    # the refined end guard against agreement between two wrong group schedules.
    weights, reference = independent_quadrature(26)
    expected_matrix = torch.zeros_like(matrices[-1])
    expected_load = torch.zeros_like(loads[-1])
    for cell in range(2):
        x, w, tables, _ = cell_tables(asm, cell, reference, weights)
        value, grad = tables["u"]
        weight = w * torch.exp(3 * x[:, 0] + 2 * x[:, 1] + x[:, 2])
        local = (torch.einsum("qi,qj,q->ij", value, value, weight)
                 + torch.einsum("qid,qjd,q->ij", grad, grad, weight))
        conn = asm.field_conn["u__tetra10"][cell]
        expected_matrix[conn[:, None], conn[None, :]] += local
        expected_load[conn] += value.T @ weight
    torch.testing.assert_close(matrices[-1], expected_matrix, atol=3e-12, rtol=3e-10)
    torch.testing.assert_close(loads[-1], expected_load, atol=3e-13, rtol=3e-10)
