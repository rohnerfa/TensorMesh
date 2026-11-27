"""Geometry-frame checks independent of the modal evaluator and numberer."""

from itertools import permutations
from math import comb

import pytest
import torch

from tensormesh.assemble._simplex_frame import _canonical_simplex
from tensormesh.element import Tetrahedron, Triangle


def _cells(element, order):
    """Disjoint cells with all corner permutations and scrambled noncorner IDs."""
    nv = element.n_vertex
    width = comb(order + element.dim, element.dim)
    perm = torch.tensor(list(permutations(range(nv))))
    ids = torch.arange(width).repeat(len(perm), 1)
    ids[:, :nv] = perm.argsort(1)
    rng = torch.Generator().manual_seed(81)
    ids[:, nv:] = nv + torch.rand(len(perm), width - nv, generator=rng).argsort(1)
    # Decreasing block offsets make a sort of the cell rows observable.
    ids += width * torch.arange(len(perm) - 1, -1, -1)[:, None]
    prefix = "triangle" if element is Triangle else "tetra"
    return prefix if order == 1 else f"{prefix}{width}", ids, perm


@pytest.mark.parametrize("element", [Triangle, Tetrahedron])
@pytest.mark.parametrize("order", range(1, 11))
def test_every_node_moves_with_its_reference_simplex(element, order):
    kind, cells, expected_perm = _cells(element, order)
    original = cells.clone()
    canonical, perm = _canonical_simplex(kind, cells, cells.numel())
    assert torch.equal(cells, original)
    assert canonical.data_ptr() != cells.data_ptr()
    assert torch.equal(perm, expected_perm)
    assert torch.equal(canonical.sort(1).values, original.sort(1).values)

    # Assign each old node its original reference coordinates. In the new
    # order these must be the affine images of *all* reference nodes under
    # the corner map. This includes every face and interior slot at high p.
    ref = element.get_basis(order, torch.float64)
    xyz = ref.new_empty(cells.numel(), element.dim)
    xyz[cells] = ref
    bary = torch.cat((1 - ref.sum(1, keepdim=True), ref), dim=1)
    expected = bary @ element.points.double()[expected_perm]
    torch.testing.assert_close(xyz[canonical], expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("element", [Triangle, Tetrahedron])
@pytest.mark.parametrize("order", [1, 2])
def test_physical_geometry_map_in_the_canonical_frame(element, order):
    kind, cells, expected_perm = _cells(element, order)
    ref = element.get_basis(order, torch.float64)
    vertices = element.points.double()
    rng = torch.Generator().manual_seed(193)
    points = ref.new_empty(cells.numel(), element.dim)
    points[cells] = ref + 0.2 * ref.sum(1, keepdim=True)
    if order == 2:
        # Move each edge midpoint independently so corner-only reindexing
        # cannot represent the same physical map.
        mid = cells[:, element.n_vertex:]
        points[mid] += 0.03 * torch.randn(points[mid].shape, dtype=ref.dtype, generator=rng)

    canonical, _ = _canonical_simplex(kind, cells, len(points))
    bary = torch.rand(7, element.n_vertex, dtype=ref.dtype, generator=rng)
    bary /= bary.sum(1, keepdim=True)
    old_query = bary @ vertices[expected_perm]
    basis = element.get_basis_fns(order, torch.float64)
    old_values = basis.map(old_query.reshape(-1, element.dim)).reshape(len(cells), len(bary), -1)
    old_physical = old_values @ points[cells]
    new_physical = basis.map(bary[:, 1:]) @ points[canonical]
    torch.testing.assert_close(new_physical, old_physical, rtol=0, atol=2e-14)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_empty_and_noncontiguous_blocks_keep_layout(device):
    kind, cells, expected_perm = _cells(Tetrahedron, 2)
    cells = cells.to(device=device, dtype=torch.int32)
    for block in (cells[::2], cells[:0]):
        canonical, perm = _canonical_simplex(kind, block, cells.numel())
        assert canonical.shape == block.shape and canonical.dtype == block.dtype
        assert canonical.device == perm.device == block.device
        assert torch.equal(perm.cpu(), expected_perm[::2][:len(block)])
        assert torch.equal(canonical[:, :4], block[:, :4].sort(1).values)


@pytest.mark.parametrize("kind,cells,n_points,error,message", [
    ("quad", [[0, 1, 2, 3]], 4, NotImplementedError, "unsupported simplex"),
    ("triangle7", [list(range(7))], 7, NotImplementedError, "unsupported simplex"),
    ("tetra10", [[0, 1, 2, 3]], 10, ValueError, "shape"),
    ("triangle", [[1, 1, 2]], 3, ValueError, "distinct"),
    ("triangle6", [[0, 1, 2, 3, 4, -1]], 6, ValueError, "point IDs"),
    ("triangle6", [[0, 1, 2, 3, 4, 6]], 6, ValueError, "point IDs"),
    ("tetra", [[0., 1., 2., 3.]], 4, ValueError, "integer"),
])
def test_rejects_invalid_geometry(kind, cells, n_points, error, message):
    with pytest.raises(error, match=message):
        _canonical_simplex(kind, torch.tensor(cells), n_points)
