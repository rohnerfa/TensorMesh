"""Entity-major H¹ numbering. Geometry preparation and basis evaluation live elsewhere."""

from math import comb
from typing import Callable, Mapping, NamedTuple

import torch

from ..element import Triangle, Tetrahedron, element_type2element, element_type2order


class _EntityBlock(NamedTuple):
    dim: int
    corners: tuple[tuple[int, ...], ...]  # local entities in evaluator order, one arity
    n_modes: Callable[[int], int]

    @property
    def key(self):
        return self.dim, len(self.corners[0])


class _Layout(NamedTuple):
    n_vertices: int
    dim: int
    blocks: tuple[_EntityBlock, ...]
    n_cell: Callable[[int], int]

    def indexed_blocks(self):
        """Slices of each (dim, arity) table, visited in local basis order."""
        stops = {}
        for block in self.blocks:
            start = stops.get(block.key, 0)
            stops[block.key] = start + len(block.corners)
            yield block, slice(start, stops[block.key])


def _edge_modes(p):
    return p - 1


def _triangle_modes(p):
    return comb(p - 1, 2)


def _tetrahedron_modes(p):
    return comb(p - 1, 3)


def _edges(element):
    return _EntityBlock(1, tuple(map(tuple, element.edge.tolist())), _edge_modes)


# For non-simplices, determine each edge/face orientation from ordered corners at setup.
# Assembly uses the basis transform for that orientation; sorted keys below only identify DOFs.
_LAYOUTS = {
    Triangle: _Layout(3, 2, (_edges(Triangle),), _triangle_modes),
    Tetrahedron: _Layout(4, 3, (_edges(Tetrahedron),
                                _EntityBlock(2, Tetrahedron.face, _triangle_modes)), _tetrahedron_modes),
}

_Key = tuple[int, int]


class _Topology(NamedTuple):
    layout: _Layout
    n_points: int
    corners: dict[str, torch.Tensor]
    vertex_ids: torch.Tensor
    vertex_conn: dict[str, torch.Tensor]
    entities: dict[_Key, torch.Tensor]  # rows of original point IDs
    counts: dict[_Key, torch.Tensor]
    entity_ids: dict[str, dict[_Key, torch.Tensor]]


class HierarchicalDofMap(NamedTuple):
    """Scalar modal connectivity in the canonical frame, on CPU.

    ``conn[kind][cell, slot]`` indexes a global coefficient directly.
    ``vertex_ids[i]`` is the original mesh point ID of vertex coefficient i.
    ``order`` is the field degree; ``topology`` is a reusable, degree-independent
    snapshot. Treat its tensors and the map as read-only.
    """
    conn: dict[str, torch.Tensor]
    n_dofs: int
    vertex_ids: torch.Tensor
    topology: _Topology
    order: int


def _canonical_corners(elements, n_points):
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    if not elements:
        raise ValueError("elements must contain at least one simplex block")
    layout = None
    corners = {}
    for kind, cells in elements.items():
        if kind not in element_type2order:
            raise ValueError(f"unknown element type: {kind!r}")
        try:
            current = _LAYOUTS[element_type2element(kind)]
        except KeyError:
            raise NotImplementedError(
                f"no hierarchical layout for {kind!r}; only triangles and tetrahedra are supported"
            ) from None
        if layout is not None and current is not layout:
            raise ValueError("hierarchical numbering requires one shape family per mesh")
        layout = current
        width = comb(element_type2order[kind] + layout.dim, layout.dim)
        if not isinstance(cells, torch.Tensor) or cells.ndim != 2 or cells.shape[1] != width:
            raise ValueError(f"{kind} geometry connectivity must have shape [n_cells, {width}]")
        if cells.dtype not in (torch.int32, torch.int64):
            raise ValueError("connectivity must contain integer point IDs")
        ids = cells[:, :layout.n_vertices].to(device="cpu", dtype=torch.long)
        if ((ids < 0) | (ids >= n_points)).any():
            raise ValueError("corner point IDs must lie in [0, n_points)")
        if (ids[:, 1:] <= ids[:, :-1]).any():
            raise ValueError("canonical simplex corners must be distinct and strictly increasing")
        corners[kind] = ids
    return layout, corners


def _build_topology(layout, corners, n_points):
    corners = {kind: ids.clone() for kind, ids in corners.items()}
    vertices = torch.unique(torch.cat([ids.flatten() for ids in corners.values()]))
    vertex_conn = {kind: torch.searchsorted(vertices, ids.contiguous()) for kind, ids in corners.items()}
    tables = {}
    for block in layout.blocks:
        tables.setdefault(block.key, []).extend(block.corners)
    entities, counts = {}, {}
    entity_ids = {kind: {} for kind in corners}
    for key, table in tables.items():
        local = torch.tensor(table, dtype=torch.long, device="cpu")
        chunks = [ids[:, local].sort(dim=-1).values.reshape(-1, key[1]) for ids in corners.values()]
        entities[key], inverse, counts[key] = torch.unique(
            torch.cat(chunks), dim=0, sorted=True, return_inverse=True, return_counts=True)
        for (kind, ids), flat in zip(corners.items(), inverse.split([len(c) for c in chunks])):
            entity_ids[kind][key] = flat.reshape(len(ids), len(table))
    return _Topology(layout, n_points, corners, vertices, vertex_conn, entities, counts, entity_ids)


def _number_dofs(topology, order):
    layout = topology.layout
    base = len(topology.vertex_ids)
    parts = {kind: [vertices] for kind, vertices in topology.vertex_conn.items()}
    offsets = {}
    for block, slots in layout.indexed_blocks():
        key, width = block.key, block.n_modes(order)
        if key not in offsets:
            offsets[key] = base
            base += len(topology.entities[key]) * width
        modes = torch.arange(width, device="cpu")
        for kind, columns in parts.items():
            ids = topology.entity_ids[kind][key][:, slots]
            columns.append((offsets[key] + ids[..., None] * width + modes).flatten(1))
    conn = {}
    nc = layout.n_cell(order)
    for kind, columns in parts.items():
        n = len(columns[0])
        columns.append(base + torch.arange(n * nc, device="cpu").reshape(n, nc))
        conn[kind] = torch.cat(columns, dim=1)
        base += n * nc
    return HierarchicalDofMap(conn, base, topology.vertex_ids, topology, order)


def hierarchical_dofmap(elements: Mapping[str, torch.Tensor], n_points: int, order: int,
                        *, topology=None) -> HierarchicalDofMap:
    """Number a uniform-order H¹ field on canonical triangles or tetrahedra.

    Supply complete geometry rows in TensorMesh node order, with strictly
    increasing corner IDs. Reframing a high-order cell requires permuting
    *all* geometry nodes. Only the corners participate in numbering.

    Shared entities have lexicographic global IDs, with consecutive modes
    per entity. Cell interiors follow in input block and row order. Local
    columns follow the reference evaluator. For another field order on the
    same cells pass ``topology=previous_map.topology`` to reuse incidence.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("field order must be an integer >= 1")
    layout, corners = _canonical_corners(elements, n_points)
    if topology is None:
        topology = _build_topology(layout, corners, n_points)
    elif (not isinstance(topology, _Topology) or topology.layout is not layout
          or topology.n_points != n_points or tuple(topology.corners) != tuple(corners)
          or any(not torch.equal(ids, topology.corners[kind]) for kind, ids in corners.items())):
        raise ValueError("topology must match the mesh point count and ordered cell corners")
    return _number_dofs(topology, order)


def hierarchical_boundary_mask(dofs: HierarchicalDofMap) -> torch.Tensor:
    """Return the CPU boolean mask of modes on the closure of one-incidence facets.

    Incidence is taken over all cell blocks of this map. No geometry tests or
    mesh tags are used. Modes private to the cell are never boundary modes.
    """
    topo = dofs.topology
    layout = topo.layout
    supports = [({i}, range(i, i + 1)) for i in range(layout.n_vertices)]
    facets = []
    start = layout.n_vertices
    for block, slots in layout.indexed_blocks():
        width = block.n_modes(dofs.order)
        for slot, corners in enumerate(block.corners, slots.start):
            support = set(corners)
            supports.append((support, range(start, start + width)))
            start += width
            if block.dim == layout.dim - 1:
                facets.append((block.key, slot, support))
    mask = torch.zeros(dofs.n_dofs, dtype=torch.bool, device="cpu")
    for key, slot, facet in facets:
        closure = [i for support, indices in supports if support <= facet for i in indices]
        for kind, conn in dofs.conn.items():
            ids = topo.entity_ids[kind][key][:, slot]
            rows = torch.where(topo.counts[key][ids] == 1)[0]
            mask[conn[rows[:, None], closure]] = True
    return mask
