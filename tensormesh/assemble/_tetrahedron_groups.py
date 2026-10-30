"""Shared oriented reference tables and CSR projection for tetrahedral fields.

Groups are an internal integration schedule. Mesh connectivity, user cell data,
and public field connectivity retain their original element ordering.
"""

from typing import NamedTuple, Tuple

import torch
from torch import nn

from ..element import Tetrahedron
from ..element.element import small_inverse
from ..element.hierarchical_tetrahedron import TETRAHEDRON_ORIENTATIONS
from ..nn import BufferDict
from .projector import SparseProjector


class AssemblyGroup(NamedTuple):
    """Host-only scheduling metadata; tensor indices live in module buffers."""

    key: str
    element_type: str
    class_id: int
    orientation: Tuple[int, int, int, int]
    n_cells: int


class GroupedTetrahedra(nn.Module):
    """Cache occupied orientation classes and their local-to-global projections."""

    def __init__(self, topology, elements, fields, field_conn, transformations):
        super().__init__()
        groups, cells, geometry, connectivity = [], {}, {}, {}
        values, gradients, table_keys = {}, {}, {}
        for kind in elements:
            _, q = transformations[kind].quadrature
            for class_id, ids in sorted(topology.groups[kind].items()):
                key = f"{kind}__o{class_id}"
                orientation = TETRAHEDRON_ORIENTATIONS[class_id]
                groups.append(AssemblyGroup(key, kind, class_id, orientation, len(ids)))
                cells[key] = ids
                geometry[key] = elements[kind].detach().cpu()[ids]
                for field in fields:
                    fkey = f"{field.trial}__{key}"
                    connectivity[fkey] = field_conn[f"{field.trial}__{kind}"].cpu()[ids]
                    if field.basis != "hierarchical":
                        continue
                    table_key = f"{kind}__p{field.order}__o{class_id}"
                    table_keys[fkey] = table_key
                    if table_key not in values:
                        basis = Tetrahedron.get_hierarchical_basis_fns(
                            field.order, q.dtype, q.device, orientation=orientation)
                        values[table_key] = basis.map(q)
                        gradients[table_key] = basis.grad().map(q)
        if not groups:
            raise ValueError("hierarchical tetrahedral assembly requires at least one cell")
        self.groups = tuple(groups)
        self.cells = BufferDict(cells)
        self.elements = BufferDict(geometry)
        self.field_conn = BufferDict(connectivity)
        self.orientation = BufferDict(dict(topology.orientation))
        self.ref_val = BufferDict(values)
        self.ref_grad = BufferDict(gradients)
        self.table_keys = table_keys
        self.pair_projector = nn.ModuleDict()
        self.vector_projector = nn.ModuleDict()
        self.pair_slices = {}
        self.vector_slices = {}

    def add_pair(self, beta, alpha, eids, n_entries, dtype):
        """Build one CSR map from group-ordered basis pairs to global entries."""
        key = f"{beta.trial}__{alpha.trial}"
        chunks, offsets, offset = [], {}, 0
        for group in self.groups:
            nb = self.field_conn[f"{beta.trial}__{group.key}"].shape[1]
            na = self.field_conn[f"{alpha.trial}__{group.key}"].shape[1]
            slots = eids[group.element_type].reshape(-1, nb, na)
            chunks.append(slots[self.cells[group.key]].reshape(-1))
            size = group.n_cells * nb * na
            offsets[group.key] = (offset, offset + size)
            offset += size
        self.pair_projector[key] = SparseProjector(
            torch.arange(offset), torch.cat(chunks), (offset,), (n_entries,), dtype=dtype)
        self.pair_slices[key] = offsets

    def add_vectors(self, fields, field_n_nodes, dtype):
        """Build the corresponding CSR maps for linear forms and vector loads."""
        for field in fields:
            chunks, offsets, offset = [], {}, 0
            for group in self.groups:
                conn = self.field_conn[f"{field.trial}__{group.key}"]
                chunks.append(conn.reshape(-1))
                offsets[group.key] = (offset, offset + conn.numel())
                offset += conn.numel()
            self.vector_projector[field.trial] = SparseProjector(
                torch.arange(offset), torch.cat(chunks), (offset,),
                (field_n_nodes[field.trial],), dtype=dtype)
            self.vector_slices[field.trial] = offsets

    def get_extra_state(self):
        """Include host-only table and projection layout in checkpoints."""
        return {"groups": [tuple(group) for group in self.groups],
                "table_keys": self.table_keys,
                "pair_slices": self.pair_slices,
                "vector_slices": self.vector_slices}

    def set_extra_state(self, state):
        # Buffers have topology-dependent shapes. Loading into a different layout
        # must fail rather than silently assigning a table to the wrong class.
        if state != self.get_extra_state():
            raise RuntimeError("grouped tetrahedral checkpoint has a different assembly layout")

    @staticmethod
    def validate_batches(batch_size, element_batch_size):
        if batch_size is not None and (
                isinstance(batch_size, bool) or not isinstance(batch_size, int)
                or batch_size == 0 or batch_size < -1):
            raise ValueError("batch_size must be -1, None, or a positive quadrature count")
        if element_batch_size is not None and (
                isinstance(element_batch_size, bool)
                or not isinstance(element_batch_size, int) or element_batch_size < 1):
            raise ValueError("element_batch_size must be None or a positive cell count")

    def batches(self, asm, params, point_data, element_data, scalar_data, field_data,
                needs_val, needs_grad, batch_size, element_batch_size):
        """Yield shared tables and active-cell data for each quadrature chunk.

        No polynomial evaluation, topology discovery, or device scalar reads
        occur here. Only changing geometry/coefficient values are gathered.
        """
        self.validate_batches(batch_size, element_batch_size)
        point_keys = [k for k, kind, _ in params if kind == "point"]
        gradpoint_keys = [k for k, kind, _ in params if kind == "gradpoint"]
        field_keys = [k for k, kind, _ in params if kind == "fielddata"]
        gradfield_keys = [k for k, kind, _ in params if kind == "gradfielddata"]
        for group in self.groups:
            kind, gkey = group.element_type, group.key
            trans = asm.transformation[kind]
            nq = trans.n_quadrature
            qstep = nq if batch_size in (-1, None) else batch_size
            estep = group.n_cells if element_batch_size is None else element_batch_size
            for start in range(0, group.n_cells, estep):
                stop = min(start + estep, group.n_cells)
                cells = self.cells[gkey][start:stop]
                geom_conn = self.elements[gkey][start:stop]
                coords = trans.points[geom_conn]
                nodal_data = {k: v[geom_conn] for k, v in point_data.items()}
                modal_data = {
                    k: data[self.field_conn[f"{name}__{gkey}"][start:stop]]
                    for k, (name, data) in field_data.items()
                }
                for qs in range(0, nq, qstep):
                    qe = min(qs + qstep, nq)
                    w, _ = trans.batch_quadrature(qs, qstep)
                    geom_grad = asm.geom_ref_grad[kind][qs:qe]
                    jac = torch.einsum("ebj,qib->eqij", coords, geom_grad)
                    inv_jac = small_inverse(jac)
                    jxw = w[None] * torch.linalg.det(jac).abs()
                    tables_val, tables_grad = {}, {}
                    for field in asm.fields:
                        name = field.trial
                        if field.basis == "hierarchical":
                            table_key = self.table_keys[f"{name}__{gkey}"]
                            val, grad = self.ref_val[table_key], self.ref_grad[table_key]
                        else:
                            table_key = f"{name}__{kind}"
                            val, grad = asm.ref_val[table_key], asm.ref_grad[table_key]
                        if name in needs_val:
                            tables_val[name] = val[qs:qe]
                        if name in needs_grad:
                            tables_grad[name] = torch.einsum(
                                "qib,eqji->eqbj", grad[qs:qe], inv_jac)
                    data_args = {}
                    if point_keys or gradpoint_keys:
                        mesh_val = trans.batch_shape_val(qs, qstep)
                        for key in point_keys:
                            data_args[key] = torch.einsum(
                                "eb...,qb->eq...", nodal_data[key], mesh_val)
                        if gradpoint_keys:
                            mesh_grad = torch.einsum("qib,eqji->eqbj", geom_grad, inv_jac)
                            for key in gradpoint_keys:
                                data_args[key] = torch.einsum(
                                    "eb...,eqbd->eq...d", nodal_data[key[4:]], mesh_grad)
                    for key in field_keys:
                        name = field_data[key][0]
                        data_args[key] = torch.einsum(
                            "eb...,qb->eq...", modal_data[key], tables_val[name])
                    for key in gradfield_keys:
                        name = field_data[key[4:]][0]
                        data_args[key] = torch.einsum(
                            "eb...,eqbd->eq...d", modal_data[key[4:]], tables_grad[name])
                    for key, data_kind, _ in params:
                        if data_kind == "element":
                            data_args[key] = element_data[key][kind][cells]
                        elif data_kind == "scalar":
                            data_args[key] = scalar_data[key]
                    yield (group, start, stop, qe == nq, jxw,
                           tables_val, tables_grad, data_args)

    def integrate(self, asm, fn, params, executed, zeros, eyes_by_field,
                  point_data, element_data, scalar_data, field_data,
                  needs_val, needs_grad, batch_size, element_batch_size, *, matrix):
        """Integrate into disjoint source slices, then project once per block."""
        self.validate_batches(batch_size, element_batch_size)
        sources, acc = {}, {}
        checked = False
        for group, start, stop, last_q, jxw, values, grads, data in self.batches(
                asm, params, point_data, element_data, scalar_data, field_data,
                needs_val, needs_grad, batch_size, element_batch_size):
            if not checked:
                asm._check_bilinear(fn, params, data, asm.dtype, asm.device,
                                    form="bilinear" if matrix else "linear")
                checked = True
            for alpha, beta in executed:
                key = f"{beta.trial}__{alpha.trial}" if matrix else beta.trial
                eyes = {"trial": eyes_by_field.get(alpha.trial) if matrix else None,
                        "test": eyes_by_field.get(beta.trial)}
                out, has_e = asm._run_pass(
                    fn, params, alpha, beta, values, grads, data, zeros, eyes)
                local = asm._integrate_pair(out, jxw, has_e)
                acc[key] = local if key not in acc else acc[key] + local
                if not last_q:
                    continue
                local = acc.pop(key)
                projector = self.pair_projector[key] if matrix else self.vector_projector[key]
                slices = self.pair_slices[key] if matrix else self.vector_slices[key]
                trailing = (beta.components, alpha.components) if matrix else (beta.components,)
                if key not in sources:
                    sources[key] = local.new_empty(projector.from_shape[0], *trailing)
                width = local.shape[1] * (local.shape[2] if matrix else 1)
                offset = slices[group.key][0]
                sources[key][offset + start * width:offset + stop * width] = \
                    local.reshape(-1, *trailing)
        projected = {}
        for alpha, beta in executed:
            key = f"{beta.trial}__{alpha.trial}" if matrix else beta.trial
            projector = self.pair_projector[key] if matrix else self.vector_projector[key]
            projector.type(sources[key].dtype)
            projected[(alpha.trial, beta.trial) if matrix else beta.trial] = \
                projector(sources[key])
        return projected
