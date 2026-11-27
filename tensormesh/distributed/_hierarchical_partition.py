"""CPU topology oracle for replicated-parent, rank-local hierarchical volume assembly.

Fields here are modal ``name: (degree, components)`` pairs in serial block order.
No process group, quadrature, sparse pattern, or global assembler is needed.
Use ``orig_nid`` for point data, ``parent_rows`` for cell and quadrature data
(the original sample axis stays intact), and ``scalar_to_serial`` for field
coefficients. ``flat_to_serial`` maps component-expanded local indices.
Permute both matrix axes and the RHS with ``serial_to_solver`` for the solver.
"""

from typing import NamedTuple

import torch

from ..assemble._simplex_frame import _canonical_simplex
from ..assemble.topology import HierarchicalDofMap, hierarchical_boundary_mask, hierarchical_dofmap
from ..mesh import Mesh
from ..mesh.partition import _extract_partition, _partition_labels


def _maps_by_order(geometry, n_points, fields):
    maps, topology = {}, None
    for order, _ in fields.values():
        if order not in maps:
            maps[order] = hierarchical_dofmap(geometry, n_points, order, topology=topology)
            topology = maps[order].topology
    return maps


def _incident_owners(incidence, size, labels, n_parts):
    """Lowest rank with a cell referencing each carrier, across all cell types."""
    owner = torch.full((size,), n_parts, dtype=torch.long, device="cpu")
    for kind, ids in incidence.items():
        ranks = labels[kind][:, None].expand_as(ids)
        owner.scatter_reduce_(0, ids.flatten(), ranks.reshape(-1), reduce="amin")
    if (owner == n_parts).any():
        raise ValueError("carrier has no incident owned cell")
    return owner


def _scalar_lookup(local, parent, rows):
    """Pair canonical basis slots, checking repeated incidences, coverage and injection."""
    lookup = local.vertex_ids.new_full((local.n_dofs,), -1)
    targets = {k: parent.conn[k][rows[k]] for k in local.conn}
    for k, ids in local.conn.items():
        lookup.scatter_reduce_(0, ids.flatten(), targets[k].flatten(), reduce="amax")
    if (lookup < 0).any() or any(not torch.equal(lookup[ids], targets[k]) for k, ids in local.conn.items()):
        raise ValueError("incomplete or inconsistent local-to-parent DOF incidences")
    if lookup.unique().numel() != local.n_dofs:
        raise ValueError("distinct local DOFs alias a parent DOF")
    return lookup


class _RankMaps(NamedTuple):
    submesh: Mesh | None
    orig_nid: torch.Tensor
    parent_rows: dict[str, torch.Tensor]
    geometry: dict[str, torch.Tensor]
    corner_perm: dict[str, torch.Tensor]
    dofs: dict[int, HierarchicalDofMap]
    scalar_to_serial: dict[str, torch.Tensor]   # field-local scalar IDs -> parent field IDs
    flat_to_serial: torch.Tensor                # local system IDs -> parent system IDs
    boundary: torch.Tensor                      # physical boundary in local system order
    owned_serial: torch.Tensor                  # serial IDs in owned solver order
    owned_boundary: torch.Tensor


class _HierarchicalPartition:
    """Replicated maps for fixed mesh connectivity, field layout and cell labels.

    With no supplied labels, use coordinate bisection. Supplied labels follow
    parent cell order (in a distributed caller these would be broadcast once).
    ``for_rank`` extracts one part only; an empty part still has valid maps.
    """

    @torch.no_grad()
    def __init__(self, mesh: Mesh, n_parts: int, fields: dict[str, tuple[int, int]], *, labels=None):
        if mesh.device.type != "cpu":
            raise ValueError("hierarchical partition setup requires a CPU parent mesh")
        if type(n_parts) is not int or n_parts < 1:
            raise ValueError("n_parts must be a positive integer")
        if not fields or any(len(spec) != 2 or any(type(x) is not int or x < 1 for x in spec)
                             for spec in fields.values()):
            raise ValueError("modal fields require positive integer degrees and component counts")
        self.mesh, self.n_parts, self.fields = mesh, n_parts, dict(fields)
        kinds = mesh.dim2eletyp[mesh.dim]
        labels = _partition_labels(mesh, n_parts) if labels is None else labels
        if set(labels) != set(kinds):
            raise ValueError("partition labels must cover exactly the volume cell blocks")
        self.labels, self.geometry, self.corner_perm = {}, {}, {}
        for k in kinds:
            r = labels[k]
            if (not isinstance(r, torch.Tensor) or r.device.type != "cpu"
                    or r.dtype not in (torch.int32, torch.int64) or r.shape != (len(mesh.cells[k]),)
                    or ((r < 0) | (r >= n_parts)).any()):
                raise ValueError(f"invalid partition labels for {k!r}")
            self.labels[k] = r.to(dtype=torch.long, copy=True)
            self.geometry[k], self.corner_perm[k] = _canonical_simplex(k, mesh.cells[k], mesh.n_points)

        self.dofs = _maps_by_order(self.geometry, mesh.n_points, self.fields)
        topo = next(iter(self.dofs.values())).topology
        vertices = _incident_owners(topo.vertex_conn, len(topo.vertex_ids), self.labels, n_parts)
        shared = {
            key: _incident_owners({k: ids[key] for k, ids in topo.entity_ids.items()},
                                  len(entities), self.labels, n_parts)
            for key, entities in topo.entities.items()
        }
        # Same global carrier order as hierarchical_dofmap. Do incidence
        # reductions once on entities, independent of field degree/components.
        cell_owner = torch.cat(list(self.labels.values()))
        widths = {b.key: b.n_modes for b in topo.layout.blocks}
        owners, boundaries = {}, {}
        for p, dofs in self.dofs.items():
            blocks = [v.repeat_interleave(widths[key](p)) for key, v in shared.items()]
            owners[p] = torch.cat([vertices, *blocks, cell_owner.repeat_interleave(topo.layout.n_cell(p))])
            boundaries[p] = hierarchical_boundary_mask(dofs)

        self.field_offsets = {}
        offset = 0
        for name, (p, components) in self.fields.items():
            self.field_offsets[name] = offset
            offset += self.dofs[p].n_dofs * components
        self.owner = torch.cat([owners[p].repeat_interleave(c) for p, c in self.fields.values()])
        self.boundary = torch.cat([boundaries[p].repeat_interleave(c) for p, c in self.fields.values()])
        self.solver_to_serial = torch.argsort(self.owner, stable=True)
        self.serial_to_solver = torch.empty_like(self.solver_to_serial)
        self.serial_to_solver[self.solver_to_serial] = torch.arange(offset, device="cpu")
        self.rank_offsets = torch.cat([self.owner.new_zeros(1),
                                       self.owner.bincount(minlength=n_parts).cumsum(0)])

    @torch.no_grad()
    def for_rank(self, rank: int) -> _RankMaps:
        if type(rank) is not int or not 0 <= rank < self.n_parts:
            raise ValueError("rank must be an integer in [0, n_parts)")
        sub, rows = _extract_partition(self.mesh, self.labels, rank)
        orig = sub.point_data["orig_nid"] if sub is not None else self.owner.new_empty(0)
        if (orig[1:] <= orig[:-1]).any():
            raise ValueError("orig_nid must be strictly increasing to preserve simplex frames")
        geometry, perm = {}, {}
        for k, selected in rows.items():
            cells = sub.cells[k] if selected.numel() else self.mesh.cells[k][:0]
            geometry[k], perm[k] = _canonical_simplex(k, cells, len(orig))
            if (not torch.equal(orig[geometry[k]], self.geometry[k][selected])
                    or not torch.equal(perm[k], self.corner_perm[k][selected])):
                raise ValueError(f"rank-local {k} geometry or frame does not match parent rows")
        dofs = _maps_by_order(geometry, len(orig), self.fields)
        by_order = {p: _scalar_lookup(m, self.dofs[p], rows) for p, m in dofs.items()}
        scalar = {name: by_order[p] for name, (p, _) in self.fields.items()}
        flat = torch.cat([
            (self.field_offsets[name] + scalar[name][:, None] * c + torch.arange(c, device="cpu")).flatten()
            for name, (_, c) in self.fields.items()
        ])
        start, stop = self.rank_offsets[rank:rank + 2].tolist()
        owned = self.solver_to_serial[start:stop]
        return _RankMaps(sub, orig, rows, geometry, perm, dofs, scalar, flat,
                         self.boundary[flat], owned, self.boundary[owned])
