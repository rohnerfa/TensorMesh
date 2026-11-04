"""Public modal assembly/field helpers after transferring a CPU-built space."""

import pytest
import torch

from tensormesh.element import Triangle
from tests.assemble._triangle_fixtures import patch_mesh, poisson


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_modal_public_helpers_follow_assembler_to_cuda():
    asm = poisson(4, "hierarchical").from_mesh(patch_mesh(2, curved=True))
    function = lambda x: x[:, 0] ** 2 + 2 * x[:, 1]
    cpu_coefficients = asm.layout.interpolate("u", function)
    cpu_matrix, cpu_load = asm().to_dense(), asm.assemble_vector()
    cpu_lifting = asm.layout.dirichlet_values("u", function)
    cpu_boundary = asm.layout.boundary_mask("u")
    asm.cuda()
    coefficients = asm.layout.interpolate("u", function)
    assert coefficients.device.type == "cuda"
    torch.testing.assert_close(coefficients.cpu(), cpu_coefficients, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(asm().to_dense().cpu(), cpu_matrix, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(asm.assemble_vector().cpu(), cpu_load, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(asm.layout.dirichlet_values("u", function).cpu(),
                               cpu_lifting, atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(asm.layout.boundary_mask("u").cpu(), cpu_boundary)
    reference = torch.tensor([[0.2, 0.3], [0.1, 0.6]], dtype=torch.float64, device="cuda")
    kind = asm.element_types[0]
    geometry = Triangle.get_basis_fns(2, torch.float64, torch.device("cuda")).map(reference)
    physical = torch.einsum("qb,ebd->eqd", geometry, asm._points[asm.elements[kind]])
    exact = function(physical.reshape(-1, 2)).reshape(physical.shape[:2])
    values = asm.layout.evaluate("u", coefficients, physical, physical=True)
    torch.testing.assert_close(values, exact, atol=2e-12, rtol=2e-11)
    gradient = asm.layout.evaluate("u", coefficients, reference, gradient=True)
    exact_gradient = torch.stack((2 * physical[..., 0], torch.full_like(physical[..., 0], 2)), -1)
    torch.testing.assert_close(gradient, exact_gradient, atol=5e-12, rtol=2e-11)
    torch.testing.assert_close(asm.layout.prolong("u", coefficients), function(asm._points),
                               atol=2e-12, rtol=2e-11)
