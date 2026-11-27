"""Entity IDs, boundary closure and topology reuse in canonical simplex frames."""

from itertools import combinations
from math import comb

import pytest
import torch

from tensormesh.assemble._hierarchical_topology import _EntityBlock, _Layout, _build_topology, _number_dofs
from tensormesh.assemble._simplex_frame import _canonical_simplex
from tensormesh.assemble.topology import hierarchical_boundary_mask, hierarchical_dofmap, lagrange_dofmap
from tensormesh.element import Prism


@pytest.mark.parametrize("kind,nv", [("triangle", 3), ("tetra", 4)])
@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 8, 10])
def test_global_ids_and_exterior_closure(kind, nv, p):
    # Split an outer simplex at an interior vertex. Its sparse point ID is
    # interleaved with the boundary IDs. Put one cell in a quadratic block.
    outside = [19, 2, 41, 7][:nv]
    center = 11
    cells = torch.tensor([[center, *facet] for facet in combinations(outside, nv - 1)])
    width = comb(nv + 1, 2)
    high = f"{kind}{width}"
    extra = torch.arange(50, 50 + width - nv)[None, :]
    elements = {kind: cells[:-1], high: torch.cat((cells[-1:], extra), dim=1)}
    elements = {k: _canonical_simplex(k, v, 100)[0] for k, v in elements.items()}
    result = hierarchical_dofmap(elements, 100, p)
    conn = torch.cat(list(result.conn.values()))
    assert conn.dtype == torch.long and conn.device.type == "cpu"
    assert conn.shape == (nv, comb(p + nv - 1, nv - 1))

    # Independent enumeration from unordered corner sets. Check not just the
    # global range but which ID each reference-evaluator column receives.
    vertices = sorted(outside + [center])
    assert result.vertex_ids.tolist() == vertices
    rows = [sorted(c) for c in cells.tolist()]
    expected = [[vertices.index(v) for v in row] for row in rows]
    boundary = [v != center for v in vertices]
    for arity in range(2, nv):
        keys = sorted({key for row in rows for key in combinations(row, arity)})
        modes = comb(p - 1, arity - 1)
        # Both simplex evaluators list edges in reverse lexicographic order
        # of the local pair (and tetra faces in reverse order of the triple).
        for row, local in zip(rows, expected):
            for entity in reversed(list(combinations(row, arity))):
                start = len(boundary) + keys.index(entity) * modes
                local.extend(range(start, start + modes))
        boundary.extend(center not in entity for entity in keys for _ in range(modes))
    for local in expected:
        n_cell = comb(p - 1, nv - 1)
        local.extend(range(len(boundary), len(boundary) + n_cell))
        boundary.extend([False] * n_cell)
    assert conn.tolist() == expected
    assert result.n_dofs == len(boundary)
    assert hierarchical_boundary_mask(result).tolist() == boundary


def test_degree_independent_snapshot_and_reuse(monkeypatch):
    elements = {"triangle6": torch.tensor([[1, 4, 9, 10, 11, 12]]),
                "triangle": torch.tensor([[2, 4, 9]])}
    first = hierarchical_dofmap(elements, 14, 1)
    topo = first.topology
    assert topo.entities[1, 2].tolist() == [[1, 4], [1, 9], [2, 4], [2, 9], [4, 9]]
    assert topo.counts[1, 2].tolist() == [1, 1, 1, 1, 2]
    elements["triangle6"][0, -1] = 13   # geometry-only node

    def forbid_unique(*args, **kwargs):
        pytest.fail("topology reuse and boundary extraction must not enumerate entities again")

    monkeypatch.setattr(torch, "unique", forbid_unique)
    higher = hierarchical_dofmap(elements, 14, 5, topology=topo)
    assert higher.topology is topo and higher.vertex_ids is first.vertex_ids
    assert hierarchical_boundary_mask(higher).sum() == 4 * 5
    for other, count in [(elements, 15), (dict(reversed(elements.items())), 14)]:
        with pytest.raises(ValueError, match="topology must match"):
            hierarchical_dofmap(other, count, 3, topology=topo)
    elements["triangle"][0, 0] = 3
    assert topo.corners["triangle"].tolist() == [[2, 4, 9]]
    with pytest.raises(ValueError, match="topology must match"):
        hierarchical_dofmap(elements, 14, 3, topology=topo)


@pytest.mark.parametrize("kind,nv", [("triangle", 3), ("tetra", 4)])
@pytest.mark.parametrize("p", [1, 4, 10])
def test_empty_blocks(kind, nv, p):
    empty = torch.empty((0, nv), dtype=torch.long)
    elements = {kind: empty}
    m = hierarchical_dofmap(elements, 0, p)
    assert m.conn[kind].shape == (0, comb(p + nv - 1, nv - 1))
    assert m.n_dofs == hierarchical_boundary_mask(m).numel() == 0
    width = comb(nv + 1, 2)
    k2 = f"{kind}{width}"
    elements[k2] = torch.arange(width)[None]
    mixed = hierarchical_dofmap(elements, width, p)
    alone = hierarchical_dofmap({k2: elements[k2]}, width, p)
    assert torch.equal(mixed.conn[k2], alone.conn[k2])
    assert torch.equal(hierarchical_boundary_mask(mixed), hierarchical_boundary_mask(alone))


def test_prism_descriptor_preserves_interleaved_evaluator_order():
    # Synthetic prism: triangular face, three square faces, triangular face.
    # Reversal of a square's corner order must not create a new entity ID.
    faces = Prism.face
    tri = lambda p: comb(p - 1, 2)
    layout = _Layout(6, 3, (
        _EntityBlock(1, tuple(map(tuple, Prism.edge.tolist())), lambda p: p - 1),
        _EntityBlock(2, faces[:1], tri),
        _EntityBlock(2, faces[1:4], lambda p: (p - 1) ** 2),
        _EntityBlock(2, faces[4:], tri)), lambda p: (p - 1) * tri(p))
    cells = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 0, 6, 4, 3, 7], [3, 4, 5, 8, 9, 10]])
    topo = _build_topology(layout, {"fixture": cells}, 11)
    m = _number_dofs(topo, 3)
    conn = m.conn["fixture"]
    assert conn.shape == (3, 40)
    assert torch.equal(conn[0, 25:29], conn[1, 25:29])  # shared square
    assert conn[0, 37] == conn[2, 24]                   # shared triangle
    assert topo.entities[2, 3].shape == (5, 3) and topo.entities[2, 4].shape == (8, 4)
    mask = hierarchical_boundary_mask(m)[conn]
    assert not mask[0, 25:29].any() and not mask[0, 37]
    assert mask[0, 24] and mask[0, 29:37].all()
    assert not mask[:, 38:].any()
    assert torch.equal(_number_dofs(_build_topology(layout, {"fixture": cells}, 11), 3).conn["fixture"], conn)


@pytest.mark.parametrize("elements,count,error,message", [
    ({}, 0, ValueError, "at least one"),
    ({"triangle": [[0, 2, 1]]}, 3, ValueError, "canonical"),
    ({"tetra": [[0, 1, 1, 3]]}, 4, ValueError, "distinct"),
    ({"triangle": [[-1, 1, 2]]}, 3, ValueError, "point IDs"),
    ({"tetra": [[0, 1, 2, 4]]}, 4, ValueError, "point IDs"),
    ({"triangle": [[0., 1., 2.]]}, 3, ValueError, "integer"),
    ({"triangle6": [[0, 1, 2]]}, 3, ValueError, "shape"),
    ({"tetra": [[0, 1, 2, 3]]}, True, ValueError, "n_points"),
    ({"tetra": [[0, 1, 2, 3]], "triangle": [[0, 1, 2]]}, 4, ValueError, "one shape family"),
])
def test_invalid_topology(elements, count, error, message):
    with pytest.raises(error, match=message):
        hierarchical_dofmap({k: torch.tensor(v) for k, v in elements.items()}, count, 3)


@pytest.mark.parametrize("kind", ["quad9", "hexahedron", "wedge", "pyramid", "line", "polygon"])
def test_other_shapes_are_explicitly_unsupported(kind):
    with pytest.raises(NotImplementedError, match=kind):
        hierarchical_dofmap({kind: torch.empty(0, 0, dtype=torch.long)}, 0, 1)


@pytest.mark.parametrize("order", [0, -1, True, 2.5])
def test_invalid_order(order):
    with pytest.raises(ValueError, match="integer >= 1"):
        hierarchical_dofmap({"triangle": torch.tensor([[0, 1, 2]])}, 3, order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_connectivity_cpu_numbering():
    cells = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], device="cuda")
    m = hierarchical_dofmap({"tetra": cells}, 5, 6)
    assert m.conn["tetra"].device.type == hierarchical_boundary_mask(m).device.type == "cpu"


def test_lagrange_connectivity_regression():
    cells = torch.tensor([[0, 1, 2], [1, 3, 2]])
    m = lagrange_dofmap({"triangle": cells}, 4, 4)
    assert m.conn["triangle"].tolist() == [
        [0, 1, 2, 10, 11, 12, 7, 8, 9, 4, 5, 6, 19, 20, 21],
        [1, 3, 2, 18, 17, 16, 10, 11, 12, 13, 14, 15, 22, 23, 24],
    ]
    assert m.n_dofs == 25
