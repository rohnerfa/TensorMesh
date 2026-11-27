"""Signed modal topology, independently checked on physical shared edges."""

from itertools import permutations

import pytest
import torch

from tensormesh.element import Triangle
from tensormesh.assemble.topology import (
    hierarchical_boundary_mask, hierarchical_dofmap, lagrange_dofmap,
)


CELLS = torch.tensor([[0, 1, 2], [1, 3, 2]])
POINTS = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)


@pytest.mark.parametrize("p", [1, 2, 3, 6, 10])
def test_numbering_and_orientation(p):
    mapping = hierarchical_dofmap({"triangle": CELLS}, 4, p)
    conn, sign = mapping.conn["triangle"], mapping.sign["triangle"]
    ni = (p - 1) * (p - 2) // 2
    assert mapping.n_dofs == 4 + 5 * (p - 1) + 2 * ni
    assert conn.shape == sign.shape == (2, (p + 1) * (p + 2) // 2)
    assert conn.dtype == torch.long and sign.dtype == torch.int8
    assert conn.device.type == sign.device.type == "cpu"
    torch.testing.assert_close(torch.unique(conn), torch.arange(mapping.n_dofs))
    torch.testing.assert_close(mapping.vertex_ids, torch.arange(4))
    torch.testing.assert_close(conn[:, :3], CELLS)
    assert (sign[:, :3] == 1).all() and (sign[:, 3 * p:] == 1).all()
    # The shared edge (1,2) is local edge 0 of cell 0 and edge 1 of cell 1.
    torch.testing.assert_close(conn[0, 3:p + 2], conn[1, p + 2:2 * p + 1])
    # Edge (3,2) of cell 1 is reversed: odd degrees change sign, not index.
    expected = torch.tensor([(-1) ** n for n in range(2, p + 1)], dtype=torch.int8)
    torch.testing.assert_close(sign[1, 3:p + 2], expected)
    assert (sign[0] == 1).all()
    if ni:
        assert not set(conn[0, 3 * p:].tolist()) & set(conn[1, 3 * p:].tolist())


def trace_tables(cells, ids, p, reverse_cells=False):
    """Evaluate globally signed functions at common *physical* points."""
    if reverse_cells:
        cells = cells.flip(0)
    mesh_cells = ids[cells]
    mapping = hierarchical_dofmap({"triangle": mesh_cells}, int(ids.max()) + 1, p)
    # Physical diagonal shared by the two triangles, from physical vertex 1 to 2.
    t = torch.linspace(0, 1, 17, dtype=torch.float64)
    physical = (1 - t[:, None]) * POINTS[1] + t[:, None] * POINTS[2]
    tangent = POINTS[2] - POINTS[1]
    basis = Triangle.get_hierarchical_basis_fns(p, torch.float64)
    tables, derivatives, unsigned = [], [], []
    for k, cell in enumerate(cells):
        vertices = POINTS[cell]
        jac = (vertices[1:] - vertices[0]).T
        reference = torch.linalg.solve(jac, (physical - vertices[0]).T).T
        ref_tangent = torch.linalg.solve(jac, tangent)
        values = basis.map(reference)
        dt = torch.einsum("qdb,d->qb", basis.grad().map(reference), ref_tangent)
        conn, sign = mapping.conn["triangle"][k], mapping.sign["triangle"][k]
        indices = conn.expand(len(t), -1)
        def scatter(local):
            return torch.zeros(len(t), mapping.n_dofs, dtype=torch.float64).scatter_add(1, indices, local)
        tables.append(scatter(values * sign))
        derivatives.append(scatter(dt * sign))
        unsigned.append(scatter(values))
    return tables, derivatives, unsigned, mapping


@pytest.mark.parametrize("p", [1, 2, 3, 6, 10])
def test_opposite_shared_edge_traversal(p):
    # Here shared edge (1,2) is local edge 0 with opposite directions.
    cells = torch.tensor([[0, 1, 2], [3, 2, 1]])
    values, dt, unsigned, mapping = trace_tables(cells, torch.arange(4), p)
    torch.testing.assert_close(values[0], values[1], atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(dt[0], dt[1], atol=2e-13, rtol=2e-12)
    # Arbitrary global coefficients, not just individual trace functions.
    u = torch.randn(mapping.n_dofs, dtype=torch.float64, generator=torch.Generator().manual_seed(11))
    torch.testing.assert_close(values[0] @ u, values[1] @ u, atol=2e-14, rtol=2e-12)
    if p >= 3:
        assert (unsigned[0] - unsigned[1]).abs().max() > 0.05


def test_all_local_vertex_permutations():
    ids = torch.arange(4)
    for left in permutations([0, 1, 2]):
        for right in permutations([1, 3, 2]):
            cells = torch.tensor([left, right])
            values, dt, _, _ = trace_tables(cells, ids, 6)
            torch.testing.assert_close(values[0], values[1], atol=2e-14, rtol=2e-12)
            torch.testing.assert_close(dt[0], dt[1], atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("cells, ids", [
    (CELLS, torch.arange(4)),
    (torch.tensor([[0, 1, 2], [3, 2, 1]]), torch.tensor([9, 2, 14, 6])),
    (torch.tensor([[2, 0, 1], [2, 3, 1]]), torch.tensor([5, 11, 0, 8])),
])
def test_vertex_renumbering_and_cell_reordering(cells, ids):
    values, dt, _, mapping = trace_tables(cells, ids, 6)
    torch.testing.assert_close(values[0], values[1], atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(dt[0], dt[1], atol=2e-13, rtol=2e-12)
    reordered = hierarchical_dofmap({"triangle": ids[cells.flip(0)]}, int(ids.max()) + 1, 6)
    # Edge/vertex global IDs are independent of cell ordering.
    torch.testing.assert_close(mapping.conn["triangle"][:, :18],
                               reordered.conn["triangle"].flip(0)[:, :18])
    torch.testing.assert_close(mapping.sign["triangle"], reordered.sign["triangle"].flip(0))
    reordered_values, reordered_dt, _, _ = trace_tables(cells, ids, 6, reverse_cells=True)
    for table in reordered_values:
        torch.testing.assert_close(table, values[0], atol=2e-14, rtol=2e-12)
    torch.testing.assert_close(reordered_dt[0], reordered_dt[1], atol=2e-13, rtol=2e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 6])
def test_hierarchical_boundary_mask_excludes_shared_edge_and_bubbles(p):
    cells = torch.tensor([[0, 1, 2], [3, 2, 1]])
    mapping = hierarchical_dofmap({"triangle": cells}, 4, p)
    mask = hierarchical_boundary_mask(mapping.conn, p, mapping.n_dofs)
    local_mask = mask[mapping.conn["triangle"]]
    assert mask.sum() == 4 * p
    assert local_mask[:, :3].all()
    assert not local_mask[:, 3:p + 2].any()
    assert local_mask[:, p + 2:3 * p].all()
    assert not local_mask[:, 3 * p:].any()


def test_hierarchical_boundary_mask_excludes_interior_vertex():
    cells = torch.tensor([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    mapping = hierarchical_dofmap({"triangle": cells}, 5, 3)
    mask = hierarchical_boundary_mask(mapping.conn, 3, mapping.n_dofs)
    torch.testing.assert_close(mask[:5], torch.tensor([True, True, True, True, False]))


def test_high_order_geometry_and_multiple_triangle_types():
    # Extra geometry-node IDs are deliberately unrelated to the field DOFs.
    cells6 = torch.cat((CELLS, torch.tensor([[4, 5, 6], [7, 8, 9]])), dim=1)
    linear = hierarchical_dofmap({"triangle": CELLS}, 10, 4)
    curved = hierarchical_dofmap({"triangle6": cells6}, 10, 4)
    torch.testing.assert_close(linear.conn["triangle"], curved.conn["triangle6"])
    torch.testing.assert_close(linear.sign["triangle"], curved.sign["triangle6"])
    mixed = hierarchical_dofmap({"triangle": CELLS[:1], "triangle6": cells6[1:]}, 10, 4)
    assert mixed.n_dofs == linear.n_dofs
    torch.testing.assert_close(mixed.vertex_ids, linear.vertex_ids)
    torch.testing.assert_close(mixed.conn["triangle"], linear.conn["triangle"][:1])
    torch.testing.assert_close(mixed.conn["triangle6"], linear.conn["triangle"][1:])


def test_lagrange_connectivity_regression():
    # Exact pre-refactor P4 numbering, including reversed nodal edge slots.
    mapping = lagrange_dofmap({"triangle": CELLS}, 4, 4)
    expected = torch.tensor([
        [0, 1, 2, 10, 11, 12, 7, 8, 9, 4, 5, 6, 19, 20, 21],
        [1, 3, 2, 18, 17, 16, 10, 11, 12, 13, 14, 15, 22, 23, 24],
    ])
    torch.testing.assert_close(mapping.conn["triangle"], expected)
    assert mapping.n_dofs == 25


@pytest.mark.parametrize("p", [1, 2, 4])
def test_empty_triangle_array(p):
    mapping = hierarchical_dofmap({"triangle": torch.empty(0, 3, dtype=torch.long)}, 0, p)
    assert mapping.n_dofs == 0
    assert mapping.conn["triangle"].shape == (0, (p + 1) * (p + 2) // 2)


@pytest.mark.parametrize("order", [0, -1, True, 2.5])
def test_hierarchical_dofmap_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="integer >= 1"):
        hierarchical_dofmap({"triangle": CELLS}, 4, order)


@pytest.mark.parametrize("kind, cells, n_points, error", [
    ("quad", torch.tensor([[0, 1, 2, 3]]), 4, NotImplementedError),
    ("triangle", torch.tensor([[0, 0, 1]]), 4, ValueError),
    ("triangle", torch.tensor([[0, 1, 4]]), 4, ValueError),
    ("triangle", torch.tensor([[0, -1, 2]]), 4, ValueError),
    ("triangle", CELLS.float(), 4, ValueError),
    ("triangle", torch.tensor([[0, 1]]), 4, ValueError),
])
def test_invalid_connectivity(kind, cells, n_points, error):
    with pytest.raises(error):
        hierarchical_dofmap({kind: cells}, n_points, 3)


def test_empty_mapping_rejected():
    with pytest.raises(ValueError, match="at least one triangle"):
        hierarchical_dofmap({}, 0, 3)
