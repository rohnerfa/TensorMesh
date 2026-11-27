"""High-order triangle assembly with generated Gaussian quadrature."""

import math

import pytest
import torch

from tensormesh import Condenser, Field, MixedElementAssembler
from tensormesh.element import Triangle
from tests.assemble._triangle_fixtures import patch_mesh, poisson, physical_points


@pytest.mark.parametrize("p", [4, 10])
def test_default_quadrature_and_degree_p_manufactured_poisson(p):
    m = p - 3
    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", p, basis="hierarchical")]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            a = x[0] ** m * (1 - x[0])
            a_second = -m * (m + 1) * x[0] ** (m - 1)
            if m > 1:
                a_second = a_second + m * (m - 1) * x[0] ** (m - 2)
            return (2 * a - a_second * x[1] * (1 - x[1])) * v

    asm = Poisson.from_mesh(patch_mesh())
    assert asm.transformation["triangle"].quadrature_order == 2 * p
    lay = asm.layout
    condenser = Condenser(lay.boundary_mask("u"))
    k, b = condenser(asm(), asm.assemble_vector())
    assert torch.linalg.eigvalsh(k.to_dense()).min() > 0
    coefficients = condenser.recover(torch.linalg.solve(k.to_dense(), b))
    _, ref = Triangle.get_quadrature(32, torch.float64)
    xy = physical_points(asm, ref)
    expected = xy[..., 0] ** m * (1 - xy[..., 0]) * xy[..., 1] * (1 - xy[..., 1])
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref), expected, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("p", [6, 10])
def test_exact_mass_stiffness_and_batching_under_quadrature_refinement(p):
    mesh = patch_mesh()
    exact = poisson(p, "hierarchical").from_mesh(mesh)
    refined = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=2 * p + 8)
    mass = lambda u, v: u * v
    for form in [None, mass]:
        reference = refined(func=form).to_dense()
        assembled = exact(func=form).to_dense()
        torch.testing.assert_close(assembled, reference, atol=3e-13, rtol=2e-10)
        torch.testing.assert_close(exact(func=form, batch_size=13).to_dense(), assembled, atol=2e-14, rtol=2e-12)
    assert torch.linalg.eigvalsh(exact(func=mass).to_dense()).min() > 0
    torch.testing.assert_close(exact.assemble_vector(batch_size=7), refined.assemble_vector(), atol=1e-14, rtol=1e-10)
    # The minimal affine stiffness degree is independently selectable.
    minimal = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=2 * p - 2)
    torch.testing.assert_close(minimal().to_dense(), exact().to_dense(), atol=3e-13, rtol=2e-10)


def test_curved_variable_coefficient_quadrature_convergence():
    class Weighted(MixedElementAssembler):
        fields = [Field("u", "v", 6, basis="hierarchical")]

        def forward(self, u, v, gradu, gradv, x):
            return torch.exp(5 * x[0] + 3 * x[1]) * ((gradu * gradv).sum() + u * v)

        def forward_vector(self, v, x):
            return torch.exp(5 * x[0] + 3 * x[1]) * v

    mesh = patch_mesh(2, curved=True)
    matrices, loads = [], []
    for degree in [12, 20, 28, 40]:
        asm = Weighted.from_mesh(mesh, quadrature_order=degree)
        matrices.append(asm().to_dense())
        loads.append(asm.assemble_vector())
    for approximations in [matrices, loads]:
        reference = approximations[-1]
        errors = [float(torch.linalg.vector_norm(a - reference) / torch.linalg.vector_norm(reference))
                  for a in approximations[:-1]]
        assert errors[0] > 100 * errors[1]
        assert errors[1] < 1e-6
        assert errors[2] < 1e-10


def test_high_order_poisson_p_convergence_for_smooth_solution():
    errors = []
    for p in [6, 8, 10]:
        class Poisson(MixedElementAssembler):
            fields = [Field("u", "v", p, basis="hierarchical")]

            def forward(self, gradu, gradv):
                return (gradu * gradv).sum()

            def forward_vector(self, v, x):
                return 2 * math.pi ** 2 * torch.sin(math.pi * x[0]) * torch.sin(math.pi * x[1]) * v

        asm = Poisson.from_mesh(patch_mesh(), quadrature_order=2 * p + 8)
        lay = asm.layout
        condenser = Condenser(lay.boundary_mask("u"))
        k, b = condenser(asm(), asm.assemble_vector())
        coefficients = condenser.recover(torch.linalg.solve(k.to_dense(), b))
        weights, ref = Triangle.get_quadrature(40, torch.float64)
        xy = physical_points(asm, ref)
        expected = torch.sin(math.pi * xy[..., 0]) * torch.sin(math.pi * xy[..., 1])
        residual = lay.evaluate("u", coefficients, ref) - expected
        errors.append(float(torch.sqrt((residual.square() * weights).sum())))
    assert all(fine < 0.2 * coarse for coarse, fine in zip(errors, errors[1:])), errors
    assert errors[-1] < 1e-6, errors
