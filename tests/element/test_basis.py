import sys 
import numpy as np
import pytest
import torch
import meshio
from typing import Type
sys.path.append("../..")

from tensormesh import ElementAssembler, NodeAssembler,  Mesh
from tensormesh.element.basis import tet_linspace
from tensormesh.element.quadrature import lin_quadrature,\
                                        tri_quadrature,\
                                        tet_quadrature
from tensormesh.element import   Element,\
                                Triangle,\
                                Quadrilateral,\
                                Tetrahedron,\
                                Hexahedron,\
                                Prism,\
                                Pyramid
import skfem

def _basis(element:Type[Element], start:int=1, end:int=9):
    for order in range(1, 4):
        if element is Prism and order == 3:
            continue # TODO: the inverse solve will explode for this condition
        basis = element.get_basis(order, dtype=torch.float64) # [n_basis, n_dim]
        assert basis.dim() == 2
        assert basis.shape[1] == element.dim
        shift_basis = []
        for i in range(element.get_n_basis(order)):
            shift_basis.append(basis.roll(i, 0))
        basis = torch.stack(shift_basis, 0)  # [n_basis, n_basis, n_dim]
        basis_fns = element.get_basis_fns(order, dtype=torch.float64) # Polynomials [n_basis] n_vars=dim n_exp=n_basis
        result    = basis_fns(basis)             # [n_basis, n_basis]
        assert result[0].allclose(torch.ones(result.shape[1], dtype=basis.dtype), atol=1e-3, rtol=1e-3), f"order: {order}"
        assert result[1:].allclose(torch.zeros(result.shape[0]-1, result.shape[1], dtype=basis.dtype), atol=1e-5, rtol=1e-5)

def test_tri():
    _basis(Triangle)

def test_quad():
    _basis(Quadrilateral)

def test_tet():
    _basis(Tetrahedron)

@pytest.mark.parametrize("include_boundary", [True, False])
def test_p5_tet_linspace(include_boundary):
    nodes = tet_linspace(Tetrahedron.points.double(), 5, include_boundary)
    lo = 0 if include_boundary else 1
    # The fourth barycentric numerator, 5 - i - j - k, must also be >= lo.
    lattice = [(i, j, k) for k in range(lo, 6) for j in range(lo, 6) for i in range(lo, 6)
               if 5 - i - j - k >= lo]
    torch.testing.assert_close(5 * nodes, nodes.new_tensor(lattice), rtol=0, atol=1e-14)

def test_hex():
    _basis(Hexahedron)

def test_pyr():
    _basis(Pyramid)

def test_pri():
    _basis(Prism)
