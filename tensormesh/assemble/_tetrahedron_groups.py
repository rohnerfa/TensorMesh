"""Orientation table banks and cell-batched CSR assembly for tetrahedral fields.

Cell chunks span orientations in original mesh order. Orientation groups describe
the shared reference tables, not the integration schedule.
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
    """Host-only orientation metadata; tensor indices live in module buffers."""

    key: str
    element_type: str
    class_id: int
    orientation: Tuple[int, int, int, int]
    n_cells: int


class GroupedTetrahedra(nn.Module):
    """Cache orientation banks and project contributions in original cell order."""

    def __init__(self, topology, elements, fields, field_conn, transformations):
        super().__init__()
        groups, cells, indices = [], {}, {}
        values, gradients, table_keys = {}, {}, {}
        self.element_types = tuple(kind for kind in elements if len(elements[kind]))
        for kind in self.element_types:
            _, q = transformations[kind].quadrature
            occupied = sorted(topology.groups[kind].items())
            indices[kind] = torch.empty(len(elements[kind]), dtype=torch.long)
            for slot, (class_id, ids) in enumerate(occupied):
                key = f"{kind}__o{class_id}"
                orientation = TETRAHEDRON_ORIENTATIONS[class_id]
                groups.append(AssemblyGroup(key, kind, class_id, orientation, len(ids)))
                cells[key] = ids
                indices[kind][ids] = slot
            for field in fields:
                if field.basis != "hierarchical":
                    continue
                table_key = f"{kind}__p{field.order}"
                table_keys[f"{field.trial}__{kind}"] = table_key
                if table_key not in values:
                    val, grad = [], []
                    for class_id, _ in occupied:
                        basis = Tetrahedron.get_hierarchical_basis_fns(
                            field.order, q.dtype, q.device,
                            orientation=TETRAHEDRON_ORIENTATIONS[class_id])
                        val.append(basis.map(q))
                        grad.append(basis.grad().map(q))
                    values[table_key] = torch.stack(val)       # [O, Q, B]
                    gradients[table_key] = torch.stack(grad)  # [O, Q, D, B]
        if not groups:
            raise ValueError("hierarchical tetrahedral assembly requires at least one cell")
        self.groups = tuple(groups)
        self.cells = BufferDict(cells)
        self.elements = BufferDict({k: v.detach().cpu() for k, v in elements.items()})
        self.field_conn = BufferDict({k: v.cpu() for k, v in field_conn.items()})
        self.orientation = BufferDict(dict(topology.orientation))
        self.table_index = BufferDict(indices)
        self.ref_val = BufferDict(values)
        self.ref_grad = BufferDict(gradients)
        self.table_keys = table_keys
        self.pair_projector = nn.ModuleDict()
        self.vector_projector = nn.ModuleDict()
        self.pair_slices = {}
        self.vector_slices = {}

    def add_pair(self, beta, alpha, eids, n_entries, dtype):
        """Build one CSR map from original-cell-order pairs to global entries."""
        key = f"{beta.trial}__{alpha.trial}"
        chunks, offsets, offset = [], {}, 0
        for kind in self.element_types:
            slots = eids[kind].reshape(-1)
            chunks.append(slots)
            size = slots.numel()
            offsets[kind] = (offset, offset + size)
            offset += size
        self.pair_projector[key] = SparseProjector(
            torch.arange(offset), torch.cat(chunks), (offset,), (n_entries,), dtype=dtype)
        self.pair_slices[key] = offsets

    def add_vectors(self, fields, field_n_nodes, dtype):
        """Build the corresponding CSR maps for linear forms and vector loads."""
        for field in fields:
            chunks, offsets, offset = [], {}, 0
            for kind in self.element_types:
                conn = self.field_conn[f"{field.trial}__{kind}"]
                chunks.append(conn.reshape(-1))
                offsets[kind] = (offset, offset + conn.numel())
                offset += conn.numel()
            self.vector_projector[field.trial] = SparseProjector(
                torch.arange(offset), torch.cat(chunks), (offset,),
                (field_n_nodes[field.trial],), dtype=dtype)
            self.vector_slices[field.trial] = offsets

    def get_extra_state(self):
        """Include host-only table and projection layout in checkpoints."""
        return {"layout": "original_cell_order_v2", "element_types": self.element_types,
                "groups": [tuple(group) for group in self.groups],
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
        """Yield original-order cell/quadrature chunks spanning all orientations.

        No polynomial evaluation, topology discovery, or device scalar reads
        occur here. Per-cell reference gathers are limited to the active chunk;
        persistent tables are shared by orientation and polynomial degree.
        """
        self.validate_batches(batch_size, element_batch_size)
        point_keys = [k for k, kind, _ in params if kind == "point"]
        gradpoint_keys = [k for k, kind, _ in params if kind == "gradpoint"]
        field_keys = [k for k, kind, _ in params if kind == "fielddata"]
        gradfield_keys = [k for k, kind, _ in params if kind == "gradfielddata"]
        for kind in self.element_types:
            trans = asm.transformation[kind]
            nq = trans.n_quadrature
            qstep = nq if batch_size in (-1, None) else batch_size
            n_cells = len(self.elements[kind])
            estep = n_cells if element_batch_size is None else element_batch_size
            for start in range(0, n_cells, estep):
                stop = min(start + estep, n_cells)
                table_ids = self.table_index[kind][start:stop]
                geom_conn = self.elements[kind][start:stop]
                coords = trans.points[geom_conn]
                nodal_data = {k: v[geom_conn] for k, v in point_data.items()}
                modal_data = {
                    k: data[self.field_conn[f"{name}__{kind}"][start:stop]]
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
                    gathered = {}
                    for field in asm.fields:
                        name = field.trial
                        if field.basis == "hierarchical":
                            table_key = self.table_keys[f"{name}__{kind}"]
                            val, grad = self.ref_val[table_key], self.ref_grad[table_key]
                            # A single occupied orientation needs no per-cell copy.
                            # Otherwise gather only this cell/quadrature chunk, and
                            # share the gathered tensors across equal-degree fields.
                            if name in needs_val:
                                key = (table_key, "value")
                                if key not in gathered:
                                    gathered[key] = (val[0, qs:qe] if val.shape[0] == 1 else
                                                     val[:, qs:qe][table_ids])
                                tables_val[name] = gathered[key]
                            if name in needs_grad:
                                key = (table_key, "gradient")
                                if key not in gathered:
                                    if grad.shape[0] == 1:
                                        physical = torch.einsum(
                                            "qib,eqji->eqbj", grad[0, qs:qe], inv_jac)
                                    else:
                                        physical = torch.einsum(
                                            "eqib,eqji->eqbj", grad[:, qs:qe][table_ids], inv_jac)
                                    gathered[key] = physical
                                tables_grad[name] = gathered[key]
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
                        val = tables_val[name]
                        data_args[key] = torch.einsum(
                            "eb...,eqb->eq..." if val.ndim == 3 else "eb...,qb->eq...",
                            modal_data[key], val)
                    for key in gradfield_keys:
                        name = field_data[key[4:]][0]
                        data_args[key] = torch.einsum(
                            "eb...,eqbd->eq...d", modal_data[key[4:]], tables_grad[name])
                    for key, data_kind, _ in params:
                        if data_kind == "element":
                            data_args[key] = element_data[key][kind][start:stop]
                        elif data_kind == "scalar":
                            data_args[key] = scalar_data[key]
                    yield (kind, start, stop, qe == nq, jxw,
                           tables_val, tables_grad, data_args)

    def integrate(self, asm, fn, params, executed, zeros, eyes_by_field,
                  point_data, element_data, scalar_data, field_data,
                  needs_val, needs_grad, batch_size, element_batch_size, *, matrix):
        """Integrate into disjoint source slices, then project once per block."""
        self.validate_batches(batch_size, element_batch_size)
        sources, acc = {}, {}
        checked = False
        for kind, start, stop, last_q, jxw, values, grads, data in self.batches(
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
                del out
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
                offset = slices[kind][0]
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
