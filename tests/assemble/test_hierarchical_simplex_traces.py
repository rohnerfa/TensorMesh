"""Physical traces with the default evaluator and a direct coefficient gather."""

from itertools import permutations

import pytest
import torch

from tensormesh.assemble._simplex_frame import _canonical_simplex
from tensormesh.assemble.topology import hierarchical_dofmap
from tensormesh.element import Tetrahedron, Triangle


@pytest.mark.parametrize("element,kind", [(Triangle, "triangle"), (Tetrahedron, "tetra")])
@pytest.mark.parametrize("p", [3, 4, 6])
def test_values_and_tangential_gradients_across_permuted_cells(element, kind, p):
    dim = element.dim
    dtype = torch.float64
    # Two affine cells glued along a non-axis-aligned edge/face. Point IDs
    # have a different order from their coordinate rows.
    points = torch.tensor([[0., 0., 0.], [1., .1, 0.], [.2, 1., .1],
                           [.1, .2, 1.], [1.2, 1.1, 1.3]], dtype=dtype)
    if dim == 2:
        points = torch.tensor([[0., 0.], [1., .1], [.2, 1.], [1.2, 1.1]], dtype=dtype)
    ids = torch.tensor([19, 2, 41, 7, 29])[:dim + 2]
    cells = torch.stack((torch.arange(dim + 1), torch.arange(1, dim + 2)))
    face = points[1:dim + 1]
    tangents = (face[1:] - face[0]).T
    weights = torch.tensor([[.13, .24, .63], [.41, .33, .26], [1., 0., 0.]], dtype=dtype)[:, :dim]
    samples = (weights / weights.sum(1, keepdim=True)) @ face
    basis = element.get_hierarchical_basis_fns(p, dtype)
    # Each side independently visits every possible input ordering.
    for side in (0, 1):
        for ordering in permutations(cells[side].tolist()):
            original = cells.clone()
            original[side] = torch.tensor(ordering)
            canonical, perm = _canonical_simplex(kind, ids[original], 42)
            m = hierarchical_dofmap({kind: canonical}, 42, p)
            tables, gradients = [], []
            for vertices, conn in zip(points[original.gather(1, perm)], m.conn[kind]):
                J = (vertices[1:] - vertices[0]).T
                ref = torch.linalg.solve(J, (samples - vertices[0]).T).T
                values = basis.map(ref)
                dt = torch.einsum("qdb,dt->qtb", basis.grad().map(ref), torch.linalg.solve(J, tangents))
                tables.append(values.new_zeros(len(ref), m.n_dofs).index_copy_(1, conn, values))
                gradients.append(dt.new_zeros(len(ref), dim - 1, m.n_dofs).index_copy_(2, conn, dt))
            torch.testing.assert_close(tables[0], tables[1], atol=3e-14, rtol=4e-12)
            torch.testing.assert_close(gradients[0], gradients[1], atol=5e-13, rtol=8e-12)
