"""Single-process checks of the integer maps, without a global matrix or process group."""

from itertools import combinations

import meshio
import pytest
import torch

from tensormesh import Mesh
from tensormesh.assemble.topology import hierarchical_boundary_mask
from tensormesh.distributed import _hierarchical_partition as partition
from tensormesh.element import Triangle, Tetrahedron
from tensormesh.mesh.partition import partition_mesh


def _star(dim, mixed_geometry=False):
    """Split a simplex about its center, permute corners, and add shared edge nodes."""
    element, linear = (Triangle, "triangle") if dim == 2 else (Tetrahedron, "tetra")
    coords = element.points.double().tolist()
    center = len(coords)
    coords.append(element.points.double().mean(0).tolist())
    edges, cells = {}, []
    for i, facet in enumerate(combinations(range(center), dim)):
        corners = [center, *facet[::-1]]
        corners = corners[i:] + corners[:i]
        row = corners.copy()
        for a, b in element.edge.tolist():
            edge = tuple(sorted((corners[a], corners[b])))
            if edge not in edges:
                edges[edge] = len(coords)
                shift = 0 if mixed_geometry else .003 * (1 + sum(edge))
                coords.append([(x + y) / 2 + shift for x, y in zip(coords[edge[0]], coords[edge[1]])])
            row.append(edges[edge])
        cells.append(row)
    # Noncompact corner IDs: reverse all point IDs, with one unused point.
    coords.append([-2.] * dim)
    cells = len(coords) - 1 - torch.tensor(cells)
    high = f"{linear}{cells.shape[1]}"
    blocks = [(high, cells.numpy())] if not mixed_geometry else [
        (linear, cells[:1, :center].numpy()), (high, cells[1:].numpy())]
    # Lower-dimensional geometry and a named side set live on the parent.
    blocks.append(("line", cells[:1, :2].numpy()))
    mesh = Mesh(meshio.Mesh(torch.tensor(coords[::-1], dtype=torch.float64).numpy(), blocks))
    mesh.register_side_set("tagged", {high: torch.tensor([[0, 0]])})
    return mesh, len(coords) - 1 - center


def _independent_rows(mesh, rank_maps, nv):
    """Recover parent rows by corner identity, without using labels or parent_rows."""
    rows = {}
    for k in rank_maps.geometry:
        parents = {tuple(sorted(row[:nv])): i for i, row in enumerate(mesh.cells[k].tolist())}
        local = rank_maps.orig_nid[rank_maps.geometry[k][:, :nv]]
        rows[k] = torch.tensor([parents[tuple(row)] for row in local.tolist()], dtype=torch.long)
    return rows


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("mixed_geometry", [False, True])
def test_rank_maps_and_weighted_data_in_parent_order(dim, mixed_geometry):
    mesh, _ = _star(dim, mixed_geometry)
    fields = {"velocity": (4, dim), "pressure": (2, 1), "temperature": (4, 1)}
    plan = partition._HierarchicalPartition(mesh, 2, fields)
    packed = torch.arange(plan.owner.numel()) + 9
    seen = {k: [] for k in plan.labels}
    has_cut = False
    for rank in range(2):
        r = plan.for_rank(rank)
        assert r.submesh is not None and "line" not in r.submesh.cells
        assert r.scalar_to_serial["velocity"] is r.scalar_to_serial["temperature"]
        assert r.dofs[4].topology is r.dofs[2].topology
        assert (r.orig_nid.diff() > 0).all()
        assert torch.equal(r.submesh.points, mesh.points[r.orig_nid])
        ids = _independent_rows(mesh, r, dim + 1)
        local_offset = 0
        for name, (p, c) in fields.items():
            size = r.dofs[p].n_dofs * c
            local_field = packed[r.flat_to_serial[local_offset:local_offset + size]].reshape(-1, c)
            start = plan.field_offsets[name]
            serial_field = packed[start:start + plan.dofs[p].n_dofs * c].reshape(-1, c)
            assert torch.equal(local_field, serial_field[r.scalar_to_serial[name]])
            for k, parent_rows in ids.items():
                assert torch.equal(r.parent_rows[k], parent_rows)
                geom = r.geometry[k]
                assert torch.equal(r.orig_nid[geom], plan.geometry[k][parent_rows])
                assert torch.equal(r.corner_perm[k], plan.corner_perm[k][parent_rows])
                # Same cell signal with point-, cell-, quadrature- and field-indexed data.
                # Distinct q weights make a silent change of sample order visible.
                q = torch.arange(7 * len(mesh.cells[k])).reshape(-1, 7)
                weights = torch.arange(1, 8)
                conn = r.dofs[p].conn[k]
                assert torch.equal(r.scalar_to_serial[name][conn], plan.dofs[p].conn[k][parent_rows])
                local_data = (r.submesh.points[geom].sum((1, 2)) + local_field[conn].sum((1, 2))
                              + (q[r.parent_rows[k]] * weights).sum(1) + r.parent_rows[k] * 11)
                global_data = (mesh.points[plan.geometry[k]].sum((1, 2))
                               + serial_field[plan.dofs[p].conn[k]].sum((1, 2))
                               + (q * weights).sum(1) + torch.arange(len(q)) * 11)
                assert torch.equal(local_data, global_data[parent_rows])
            local_mask = hierarchical_boundary_mask(r.dofs[p])
            physical = r.boundary[local_offset:local_offset + size].reshape(-1, c)
            expected = hierarchical_boundary_mask(plan.dofs[p])[r.scalar_to_serial[name]]
            assert torch.equal(physical, expected[:, None].expand_as(physical))
            assert (physical[:, 0] <= local_mask).all()
            has_cut |= bool((local_mask & ~physical[:, 0]).any())
            local_offset += size
        assert r.flat_to_serial.unique().numel() == local_offset
        for k, values in ids.items():
            seen[k].extend(values.tolist())
    assert has_cut
    assert all(sorted(v) == list(range(len(mesh.cells[k]))) for k, v in seen.items())


@pytest.mark.parametrize("dim", [2, 3])
def test_incident_ownership_solver_roundtrip_and_empty_rank(dim):
    mesh, center = _star(dim)
    k, = mesh.dim2eletyp[dim]
    # Deliberately permuted ranks. A vertex of a shared edge/face is owned by
    # rank zero although rank zero has no cell incident to that edge/face.
    ranks = torch.arange(len(mesh.cells[k])).roll(1)
    world = len(ranks) + 1
    plan = partition._HierarchicalPartition(mesh, world, {"u": (5, 2)}, labels={k: ranks})
    m = plan.dofs[5]
    incident = [[] for _ in range(m.n_dofs)]
    for row, rank in zip(m.conn[k].tolist(), ranks.tolist()):
        for i in row:
            incident[i].append(rank)
    assert plan.owner.tolist() == [min(rs) for rs in incident for _ in range(2)]
    topo = m.topology
    vertex_owner = dict(zip(m.vertex_ids.tolist(), plan.owner[0:2 * len(m.vertex_ids):2].tolist()))
    assert vertex_owner[center] == 0
    for arity in range(2, dim + 1):
        # Check a shared carrier that endpoint ownership would get wrong.
        keys = topo.entities[arity - 1, arity].tolist()
        raw = mesh.cells[k][:, :dim + 1].tolist()
        incident_ranks = [[r for cell, r in zip(raw, ranks.tolist()) if set(key) <= set(cell)]
                          for key in keys]
        assert any(center in key and len(rs) > 1 and min(rs) > 0 for key, rs in zip(keys, incident_ranks))
    all_ids = torch.arange(len(plan.owner))
    assert torch.equal(plan.solver_to_serial[plan.serial_to_solver], all_ids)
    assert torch.equal(plan.serial_to_solver[plan.solver_to_serial], all_ids)
    assert not torch.equal(plan.solver_to_serial, all_ids)
    for rank in range(world):
        r = plan.for_rank(rank)
        assert torch.equal(r.owned_serial, all_ids[plan.owner == rank])
        assert torch.equal(r.owned_boundary, plan.boundary[r.owned_serial])
        assert torch.isin(r.owned_serial, r.flat_to_serial).all()
    empty = plan.for_rank(world - 1)
    assert empty.submesh is None and empty.dofs[5].n_dofs == 0
    assert empty.flat_to_serial.numel() == empty.owned_serial.numel() == 0
    assert plan.rank_offsets[-1] == plan.rank_offsets[-2] == len(all_ids)


def test_label_and_frame_validation(monkeypatch):
    mesh, _ = _star(2)
    fields = {"u": (3, 1)}
    for labels in [{}, {"triangle6": torch.tensor([0])}, {"triangle6": torch.tensor([0, 1, 2])},
                   {"triangle6": torch.tensor([0., 0., 1.])}]:
        with pytest.raises(ValueError, match="labels"):
            partition._HierarchicalPartition(mesh, 2, fields, labels=labels)
    plan = partition._HierarchicalPartition(mesh, 2, fields)
    original = partition._extract_partition
    for corrupt in ("orig_nid", "geometry"):
        def bad(*args):
            sub, rows = original(*args)
            if corrupt == "orig_nid":
                sub.point_data["orig_nid"] = sub.point_data["orig_nid"].flip(0)
            else:
                sub.cells["triangle6"][0, [3, 4]] = sub.cells["triangle6"][0, [4, 3]]
            return sub, rows
        monkeypatch.setattr(partition, "_extract_partition", bad)
        with pytest.raises(ValueError, match="orig_nid|does not match parent"):
            plan.for_rank(0)


def test_local_lookup_rejects_missing_conflicting_and_aliased_assignments():
    mesh, _ = _star(2)
    plan = partition._HierarchicalPartition(mesh, 1, {"u": (4, 1)})
    r = plan.for_rank(0)
    local, parent = r.dofs[4], plan.dofs[4]
    with pytest.raises(ValueError, match="incomplete"):
        partition._scalar_lookup(local._replace(n_dofs=local.n_dofs + 1), parent, r.parent_rows)
    k, = local.conn
    ids = parent.conn[k].clone()
    assert (local.conn[k] == local.conn[k][0, 0]).sum() > 1
    ids[0, 0] = ids[0, -1]
    with pytest.raises(ValueError, match="inconsistent"):
        partition._scalar_lookup(local, parent._replace(conn={k: ids}), r.parent_rows)
    ids = parent.conn[k].clone()
    ids[ids == 1] = 0
    with pytest.raises(ValueError, match="alias"):
        partition._scalar_lookup(local, parent._replace(conn={k: ids}), r.parent_rows)


def test_legacy_partition_contract():
    mesh, _ = _star(3, mixed_geometry=True)
    subs = partition_mesh(mesh, 5)
    assert len(subs) == 5 and any(s is None for s in subs)
    for sub in filter(lambda s: s is not None, subs):
        assert list(sub.point_data.keys()) == ["orig_nid"]
        assert not sub.side_sets and "line" not in sub.cells
        ids = sub.point_data["orig_nid"]
        assert torch.equal(sub.points, mesh.points[ids])
        for k, cells in sub.cells.items():
            parent = mesh.cells[k].tolist()
            indices = [parent.index(row) for row in ids[cells].tolist()]
            assert indices == sorted(indices)
