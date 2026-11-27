"""Tetrahedral entities, stable groups and physical shared-face conformity."""

from itertools import combinations, permutations

import pytest
import torch

from tensormesh.assemble import topology as topology_module
from tensormesh.assemble.topology import (
    _tetrahedron_topology, hierarchical_tetrahedron_boundary_mask,
    hierarchical_tetrahedron_dofmap,
)
from tensormesh.element import Tetrahedron, TETRAHEDRON_ORIENTATIONS


CELLS = torch.tensor([[0, 1, 2, 3], [4, 3, 1, 2]])
# Shared physical face is (1,2,3), with non-axis-aligned tangents.
POINTS = torch.tensor([[0., 0., 0.], [1., 0.1, 0.], [0.2, 1., 0.1],
                       [0.1, 0.2, 1.], [1.2, 1.1, 1.3]], dtype=torch.float64)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 8, 10])
def test_tetrahedral_global_dimension_entity_numbering_and_private_modes(p):
    ids = torch.tensor([19, 2, 41, 7, 29])
    elements = {"tetra": ids[CELLS]}
    result = hierarchical_tetrahedron_dofmap(elements, 50, p)
    conn, topo = result.conn["tetra"], result.topology
    ne = p - 1
    nf = (p - 1) * (p - 2) // 2 if p >= 3 else 0
    nc = (p - 1) * (p - 2) * (p - 3) // 6 if p >= 4 else 0
    assert result.n_dofs == 5 + 9 * ne + 7 * nf + 2 * nc
    assert conn.shape == (2, (p + 1) * (p + 2) * (p + 3) // 6)
    assert conn.dtype == torch.long and conn.device.type == "cpu"
    torch.testing.assert_close(torch.unique(conn), torch.arange(result.n_dofs))
    torch.testing.assert_close(result.vertex_ids, ids.sort().values)
    torch.testing.assert_close(result.vertex_ids[conn[:, :4]], ids[CELLS])
    expected_edges = sorted({tuple(sorted(edge)) for cell in ids[CELLS].tolist()
                             for edge in combinations(cell, 2)})
    expected_faces = sorted({tuple(sorted(face)) for cell in ids[CELLS].tolist()
                             for face in combinations(cell, 3)})
    torch.testing.assert_close(topo.edges, torch.tensor(expected_edges))
    torch.testing.assert_close(topo.faces, torch.tensor(expected_faces))
    torch.testing.assert_close(topo.edges[topo.edge_ids["tetra"]],
                               ids[CELLS][:, Tetrahedron.edge].sort(-1).values)
    torch.testing.assert_close(topo.faces[topo.face_ids["tetra"]],
                               ids[CELLS][:, torch.tensor(Tetrahedron.face)].sort(-1).values)
    assert (topo.face_counts == 1).sum() == 6
    assert (topo.face_counts == 2).sum() == 1
    assert topo.boundary_faces.sum() == 6
    for f, (type_id, cell, local_face) in enumerate(topo.face_representatives.tolist()):
        assert type_id == 0
        assert topo.face_ids["tetra"][cell, local_face] == f
        first = torch.nonzero(topo.face_ids["tetra"] == f)[0]
        torch.testing.assert_close(first, torch.tensor([cell, local_face]))
    for e in range(6):
        torch.testing.assert_close(conn[:, 4 + e * ne:4 + (e + 1) * ne],
            5 + topo.edge_ids["tetra"][:, e, None] * ne + torch.arange(ne))
    for f in range(4):
        torch.testing.assert_close(conn[:, 4 + 6 * ne + f * nf:4 + 6 * ne + (f + 1) * nf],
            5 + 9 * ne + topo.face_ids["tetra"][:, f, None] * nf + torch.arange(nf))
    cell_base = 5 + 9 * ne + 7 * nf
    torch.testing.assert_close(conn[:, 4 + 6 * ne + 4 * nf:],
                               torch.arange(cell_base, result.n_dofs).reshape(2, nc))


def test_tetrahedral_orientation_classes_and_stable_groups_are_order_independent(monkeypatch):
    # Every permutation, with unequal group sizes and scrambled original rows.
    cells = torch.tensor(list(permutations([8, 2, 17, 5])))
    cells = torch.cat((cells.flip(0), cells[::3], cells[[5, 5, 19]]))
    elements = {"tetra": cells}
    topology = _tetrahedron_topology(elements, 20)
    expected = torch.tensor([TETRAHEDRON_ORIENTATIONS.index(
        tuple(sorted(range(4), key=cell.__getitem__))) for cell in cells.tolist()])
    torch.testing.assert_close(topology.orientation["tetra"], expected)
    assert list(topology.groups["tetra"]) == list(range(24))
    for c, group in topology.groups["tetra"].items():
        torch.testing.assert_close(group, torch.nonzero(expected == c).flatten())
        assert group.dtype == torch.long and group.device.type == "cpu"
    assert topology.edges.shape == (6, 2) and topology.faces.shape == (4, 3)

    def forbid_rebuild(*args, **kwargs):
        raise AssertionError("reused topology must not rebuild entities or groups")

    monkeypatch.setattr(topology_module, "_tetrahedron_topology", forbid_rebuild)
    for p in [1, 2, 4, 6, 10]:
        mapping = hierarchical_tetrahedron_dofmap(elements, 20, p, topology=topology)
        assert mapping.topology is topology
        assert mapping.vertex_ids is topology.vertex_ids
        assert mapping.topology.groups is topology.groups


def physical_face_tables(cells, ids, p):
    """Oriented global trace columns at common physical points and tangents."""
    mapping = hierarchical_tetrahedron_dofmap({"tetra": ids[cells]}, int(ids.max()) + 1, p)
    bary = torch.tensor([[0.13, 0.24, 0.63], [0.41, 0.33, 0.26], [0.21, 0.59, 0.20],
                          [0., 0.3, 0.7], [1., 0., 0.]], dtype=torch.float64)
    physical = bary @ POINTS[[1, 2, 3]]
    tangents = (POINTS[[2, 3]] - POINTS[1]).T
    tables, derivatives = [], []
    for cell_id, cell in enumerate(cells):
        vertices = POINTS[cell]
        jac = (vertices[1:] - vertices[0]).T
        reference = torch.linalg.solve(jac, (physical - vertices[0]).T).T
        ref_tangents = torch.linalg.solve(jac, tangents)
        orientation = TETRAHEDRON_ORIENTATIONS[mapping.topology.orientation["tetra"][cell_id]]
        basis = Tetrahedron.get_hierarchical_basis_fns(p, torch.float64, orientation=orientation)
        values = basis.map(reference)
        dt = torch.einsum("qdb,dt->qtb", basis.grad().map(reference), ref_tangents)
        conn = mapping.conn["tetra"][cell_id]
        tables.append(torch.zeros(len(bary), mapping.n_dofs, dtype=torch.float64)
                      .scatter_add(1, conn.expand(len(bary), -1), values))
        derivatives.append(torch.zeros(len(bary), 2, mapping.n_dofs, dtype=torch.float64)
                           .scatter_add(2, conn.expand(len(bary), 2, -1), dt))
    return tables, derivatives, mapping


@pytest.mark.parametrize("p", [4, 6])
@pytest.mark.parametrize("ids", [torch.arange(5), torch.tensor([19, 3, 31, 7, 12])])
def test_tetrahedral_all_face_orderings_slots_and_physical_tangential_traces(p, ids):
    # Each cell visits all 24 permutations: all six face orders in all four
    # local face slots. Comparing each side with the common fixed trace also
    # covers all 24x24 pairings by transitivity, without repeated tabulation.
    generator = torch.Generator().manual_seed(29)
    for side in range(2):
        for ordering in permutations(CELLS[side].tolist()):
            cells = CELLS.clone()
            cells[side] = torch.tensor(ordering)
            values, dt, mapping = physical_face_tables(cells, ids, p)
            torch.testing.assert_close(values[0], values[1], atol=3e-14, rtol=4e-12)
            torch.testing.assert_close(dt[0], dt[1], atol=5e-13, rtol=8e-12)
            coefficients = torch.randn(mapping.n_dofs, 3, dtype=torch.float64, generator=generator)
            torch.testing.assert_close(values[0] @ coefficients, values[1] @ coefficients,
                                       atol=4e-14, rtol=4e-12)
            torch.testing.assert_close(dt[0] @ coefficients, dt[1] @ coefficients,
                                       atol=8e-13, rtol=8e-12)


def test_tetrahedral_high_order_corners_and_original_cell_order():
    cells10 = torch.cat((CELLS, torch.arange(5, 17).reshape(2, 6)), dim=1)
    linear = hierarchical_tetrahedron_dofmap({"tetra": CELLS}, 40, 6)
    quadratic = hierarchical_tetrahedron_dofmap({"tetra10": cells10}, 40, 6)
    mixed = hierarchical_tetrahedron_dofmap({"tetra": CELLS[:1], "tetra10": cells10[1:]}, 40, 6)
    assert linear.n_dofs == quadratic.n_dofs == mixed.n_dofs
    torch.testing.assert_close(linear.conn["tetra"], quadratic.conn["tetra10"])
    torch.testing.assert_close(mixed.conn["tetra"], linear.conn["tetra"][:1])
    torch.testing.assert_close(mixed.conn["tetra10"], linear.conn["tetra"][1:])
    torch.testing.assert_close(linear.topology.orientation["tetra"],
                               quadratic.topology.orientation["tetra10"])
    # Changing high-order geometry nodes leaves reusable field topology valid.
    cells10[:, 4:] += 20
    reused = hierarchical_tetrahedron_dofmap({"tetra10": cells10}, 40, 6,
                                            topology=quadratic.topology)
    torch.testing.assert_close(reused.conn["tetra10"], quadratic.conn["tetra10"])
    reverse = hierarchical_tetrahedron_dofmap({"tetra": CELLS.flip(0)}, 40, 6)
    nc = 10
    torch.testing.assert_close(linear.conn["tetra"][:, :-nc], reverse.conn["tetra"].flip(0)[:, :-nc])
    torch.testing.assert_close(linear.conn["tetra"][:, -nc:], reverse.conn["tetra"][:, -nc:])
    torch.testing.assert_close(linear.topology.orientation["tetra"],
                               reverse.topology.orientation["tetra"].flip(0))


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 10])
def test_tetrahedral_boundary_mask_excludes_internal_vertices_edges_faces_and_cells(p):
    # Split an outer tetrahedron at vertex 4. Edges meeting 4 are interior;
    # boundary faces consist entirely of the original four vertices.
    cells = torch.tensor([[4, *face] for face in Tetrahedron.face])
    mapping = hierarchical_tetrahedron_dofmap({"tetra": cells}, 5, p)
    topo = mapping.topology
    mask = hierarchical_tetrahedron_boundary_mask(mapping.conn, p, mapping.n_dofs)
    ne, nf = p - 1, max(0, (p - 1) * (p - 2) // 2)
    assert mask.dtype == torch.bool and mask.device.type == "cpu"
    assert mask.sum() == 4 + 6 * ne + 4 * nf
    expected = torch.zeros_like(mask)
    expected[:4] = True
    for e, edge in enumerate(topo.edges.tolist()):
        expected[5 + e * ne:5 + (e + 1) * ne] = 4 not in edge
    face_base = 5 + len(topo.edges) * ne
    for f, face in enumerate(topo.faces.tolist()):
        expected[face_base + f * nf:face_base + (f + 1) * nf] = 4 not in face
    torch.testing.assert_close(mask, expected)
    # Face incidence is shared across actual geometry types too.
    split = hierarchical_tetrahedron_dofmap({"tetra": cells[:2], "tetra10": cells[2:]}, 5, p)
    torch.testing.assert_close(hierarchical_tetrahedron_boundary_mask(split.conn, p, split.n_dofs),
                               expected)


@pytest.mark.parametrize("p", [1, 4, 10])
def test_tetrahedral_empty_blocks_preserve_shapes(p):
    empty = torch.empty(0, 4, dtype=torch.long)
    mapping = hierarchical_tetrahedron_dofmap({"tetra": empty}, 0, p)
    assert mapping.n_dofs == 0
    assert mapping.conn["tetra"].shape == (0, (p + 1) * (p + 2) * (p + 3) // 6)
    assert mapping.topology.edges.shape == (0, 2)
    assert mapping.topology.faces.shape == (0, 3)
    assert mapping.topology.face_representatives.shape == (0, 3)
    assert mapping.topology.groups == {"tetra": {}}
    assert hierarchical_tetrahedron_boundary_mask(mapping.conn, p, 0).shape == (0,)
    nonempty = hierarchical_tetrahedron_dofmap({"tetra": empty, "tetra10": CELLS}, 5, p)
    assert nonempty.topology.groups["tetra"] == {}
    assert (nonempty.topology.face_representatives[:, 0] == 1).all()


@pytest.mark.parametrize("p", [0, -1, True, 2.5])
def test_tetrahedral_topology_and_boundary_reject_invalid_order(p):
    with pytest.raises(ValueError, match="integer >= 1"):
        hierarchical_tetrahedron_dofmap({"tetra": CELLS}, 5, p)
    with pytest.raises(ValueError, match="integer >= 1"):
        hierarchical_tetrahedron_boundary_mask({"tetra": CELLS}, p, 5)


@pytest.mark.parametrize("kind, cells, count, error", [
    ("triangle", CELLS, 5, NotImplementedError),
    ("hexahedron", CELLS, 5, NotImplementedError),
    ("tetra_unknown", CELLS, 5, ValueError),
    ("tetra", torch.tensor([[0, 1, 1, 3]]), 5, ValueError),
    ("tetra", torch.tensor([[0, 1, -1, 3]]), 5, ValueError),
    ("tetra", CELLS, 4, ValueError),
    ("tetra", CELLS.float(), 5, ValueError),
    ("tetra", CELLS.bool(), 5, ValueError),
    ("tetra", CELLS[:, :3], 5, ValueError),
    ("tetra", CELLS[0], 5, ValueError),
    ("tetra", CELLS, -1, ValueError),
    ("tetra", CELLS, True, ValueError),
    ("tetra", CELLS, 5., ValueError),
])
def test_tetrahedral_invalid_mesh_connectivity(kind, cells, count, error):
    with pytest.raises(error):
        hierarchical_tetrahedron_dofmap({kind: cells}, count, 4)


def test_tetrahedral_reused_topology_rejects_stale_connectivity_and_is_a_snapshot():
    cells = CELLS.clone()
    topo = _tetrahedron_topology({"tetra": cells}, 5)
    cells[0] = cells[0].flip(0)
    torch.testing.assert_close(topo.corners["tetra"], CELLS)
    for elements, n_points in [({"tetra": cells}, 5), ({"tetra": CELLS}, 6),
                                ({"tetra10": CELLS}, 5)]:
        with pytest.raises(ValueError, match="topology must match"):
            hierarchical_tetrahedron_dofmap(elements, n_points, 4, topology=topo)
    with pytest.raises(ValueError, match="topology must match"):
        hierarchical_tetrahedron_dofmap({"tetra": CELLS}, 5, 4, topology=object())
    with pytest.raises(ValueError, match="at least one tetrahedron"):
        hierarchical_tetrahedron_dofmap({}, 0, 1)


def test_tetrahedral_boundary_validation_includes_all_modal_ids():
    mapping = hierarchical_tetrahedron_dofmap({"tetra": CELLS}, 5, 4)
    invalid = mapping.conn["tetra"].clone()
    invalid[0, -1] = mapping.n_dofs
    with pytest.raises(ValueError, match="coefficient IDs"):
        hierarchical_tetrahedron_boundary_mask({"tetra": invalid}, 4, mapping.n_dofs)
    with pytest.raises(ValueError, match="width"):
        hierarchical_tetrahedron_boundary_mask(mapping.conn, 3, mapping.n_dofs)
    with pytest.raises(ValueError, match="n_dofs"):
        hierarchical_tetrahedron_boundary_mask(mapping.conn, 4, -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tetrahedral_cuda_connectivity_cpu_setup_and_device_boundary_mask():
    elements = {"tetra": CELLS.cuda()}
    mapping = hierarchical_tetrahedron_dofmap(elements, 5, 6)
    assert mapping.conn["tetra"].device.type == "cpu"
    cpu_mask = hierarchical_tetrahedron_boundary_mask(mapping.conn, 6, mapping.n_dofs)
    cuda_mask = hierarchical_tetrahedron_boundary_mask(
        {k: v.cuda() for k, v in mapping.conn.items()}, 6, mapping.n_dofs)
    torch.testing.assert_close(cuda_mask.cpu(), cpu_mask)
