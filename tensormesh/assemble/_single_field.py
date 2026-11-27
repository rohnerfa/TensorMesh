"""Single-field volume spaces tabulated independently of nodal geometry."""

import torch
from torch import nn

from ..element.element import small_inverse
from ..element.transformation import _version_counter
from ._simplex_frame import _canonical_simplex, _canonical_reference
from ._hierarchical_topology import hierarchical_dofmap

# Budget for scalar entries in a modal weak-form batch, before component axes.
_MODAL_BATCH_ENTRIES = 1 << 20


def _physical_tables(coords, geom_grad, ref_grad, weights, affine):
    """Pull field gradients into physical coordinates; a P1 simplex has one Jacobian."""
    if affine:
        jac = torch.einsum("ebj,ib->eij", coords, geom_grad[0])
        inv = small_inverse(jac)
        grad = torch.einsum("qib,eji->eqbj", ref_grad, inv)
        return grad, jac.det().abs()[:, None] * weights, inv[:, None].expand(-1, len(weights), -1, -1)
    jac = torch.einsum("ebj,qib->eqij", coords, geom_grad)
    inv = small_inverse(jac)
    return torch.einsum("qib,eqji->eqbj", ref_grad, inv), jac.det().abs() * weights, inv


class _ReferenceTables(nn.Module):
    def __init__(self, conn, basis, trans):
        super().__init__()
        self.register_buffer("conn", conn)
        _, q = trans.quadrature
        _ = trans.shape_val
        self.register_buffer("ref_val", basis.map(q))              # [Q,B]
        self.register_buffer("ref_grad", basis.grad().map(q))      # [Q,D,B]
        self.register_buffer("geom_grad", trans.element.get_basis_grad_fns(
            trans.basis_order, q.dtype, q.device).map(q))
        self._geometry_key = None

    def _apply(self, fn, recurse=True):
        # Complex coefficients do not turn geometry or basis tables complex.
        def real_only(t):
            out = fn(t)
            return out.real if out.is_complex() else out
        return super()._apply(real_only, recurse=recurse)

    def batch(self, trans, start, size):
        trans.update_points(trans.points)
        points = trans.points
        stop = min(start + size, trans.n_quadrature)
        version = _version_counter(points)
        key = (id(points), version, points.dtype, points.device,
               id(trans.elements), _version_counter(trans.elements), _version_counter(self.ref_grad),
               _version_counter(self.geom_grad))
        cacheable = version is not None and not (
            torch.is_grad_enabled() and points.requires_grad)
        if key != self._geometry_key or not cacheable:
            for name in ("_grad", "_jxw", "_inverse"):
                self._buffers.pop(name, None)
            self._geometry_key = key if cacheable else None
        values = self.ref_val[start:stop]
        if "_grad" in self._buffers:
            return (values, self._grad[:, start:stop], self._jxw[:, start:stop],
                    self._inverse[:, start:stop])
        w, _ = trans.quadrature
        gradient, jxw, inverse = _physical_tables(
            trans.element_coords, self.geom_grad[start:stop], self.ref_grad[start:stop],
            w[start:stop], trans.basis_order == 1 and hasattr(self, "frame"))
        if cacheable and start == 0 and stop == trans.n_quadrature:
            self.register_buffer("_grad", gradient, persistent=False)
            self.register_buffer("_jxw", jxw, persistent=False)
            self.register_buffer("_inverse", inverse, persistent=False)
        return values, gradient, jxw, inverse


def canonical_field_map(elements, n_points, order):
    geometry, frames = {}, {}
    for kind, cells in elements.items():
        geometry[kind], frames[kind] = _canonical_simplex(kind, cells, n_points)
    return geometry, hierarchical_dofmap(geometry, n_points, order), frames


class _SingleField(nn.Module):
    def __init__(self, mapping, transformations, order, basis, frames=None):
        super().__init__()
        self.order = order
        self.basis = basis
        self.n_dofs = mapping.n_dofs
        blocks = {}
        for kind, conn in mapping.conn.items():
            trans = transformations[kind]
            _, q = trans.quadrature
            factory = trans.element.get_hierarchical_basis_fns
            block = _ReferenceTables(conn, factory(order, q.dtype, q.device), trans)
            block.register_buffer("frame", frames[kind])
            blocks[kind] = block
        self.blocks = nn.ModuleDict(blocks)

    def layout(self, assembler, components):
        from .mixed_assembler import BlockLayout, Field
        field = Field("u", "v", self.order, components, basis=self.basis)
        return BlockLayout(_SingleFieldView(assembler, field))

    def get_extra_state(self):
        state = {"order": self.order, "n_dofs": self.n_dofs, "element_types": list(self.blocks)}
        if self.basis == "hierarchical":
            state["frame"] = "canonical-simplex-v1"
        return state

    def set_extra_state(self, state):
        if self.basis == "hierarchical" and state.get("frame") != "canonical-simplex-v1":
            raise RuntimeError("incompatible pre-canonical hierarchical checkpoint; reinterpolate coefficients")
        if state != self.get_extra_state():
            raise RuntimeError(
                f"{self.basis} assembler checkpoint has a different field layout"
            )


class _SingleFieldView:
    """Live buffer view for reusing BlockLayout, without a mixed assembler."""

    def __init__(self, assembler, field):
        self._assembler = assembler
        self.fields = [field]
        self._offsets = {"u": 0}
        self.field_kind = {"u": "hierarchical"}
        self.canonical_modal = True
        self.field_n_nodes = {"u": assembler._field_space.n_dofs}
        self.n_dofs = self.field_n_nodes["u"] * field.components
        self.n_points = assembler.n_points
        self.dimension = assembler.dimension
        self.element_types = assembler.element_types
        self.mesh_order = next(iter(assembler.transformation.values())).basis_order

    def _field(self, name):
        if name != "u":
            raise KeyError(f"unknown field {name!r}; the single field is 'u'")
        return self.fields[0]

    @property
    def _points(self):
        return next(iter(self._assembler.transformation.values())).points

    @property
    def elements(self):
        return {k: t.elements for k, t in self._assembler.transformation.items()}

    def canonical_reference(self, kind, cells, ref):
        frame = self._assembler._hierarchical.blocks[kind].frame
        return _canonical_reference(ref, frame[cells])

    @property
    def field_conn(self):
        return {f"u__{kind}": block.conn
                for kind, block in self._assembler._field_space.blocks.items()}

    field_sign = {}


def volume_batches(trans, field, size, parameters, point_data):
    """Yield field and mesh tables in one quadrature frame for either volume form."""
    starts = range(0, trans.n_quadrature, size)
    if field is None:
        for start in starts:
            val = trans.batch_shape_val(start, size)
            grad, jxw = trans.batch_shape_grad_jxw(quadrature_start=start, quadrature_batch=size)
            yield start, val, grad, jxw, val, grad
    else:
        need_grad = any(k.startswith("grad") and k[4:] in point_data for k in parameters)
        for start in starts:
            val, grad, jxw, inverse = field.batch(trans, start, size)
            mesh_val = trans.batch_shape_val(start, size)
            mesh_grad = (torch.einsum("qib,eqji->eqbj", field.geom_grad[start:start+size], inverse)
                         if need_grad else None)
            yield start, val, grad, jxw, mesh_val, mesh_grad


def _chunk_batches(coords, tables, weights, size, need_grad, affine):
    val, ref_grad, mesh_val, geom_grad = tables
    for start in range(0, len(weights), size):
        q = slice(start, start + size)
        grad, jxw, inv = _physical_tables(coords, geom_grad[q], ref_grad[q], weights[q], affine)
        mesh_grad = torch.einsum("qib,eqji->eqbj", geom_grad[q], inv) if need_grad else None
        yield start, val[q], grad, jxw, mesh_val[q], mesh_grad


def _modal_groups(trans, field, order, size, sampled, cell_limit, need_grad):
    """Bound the active cell batch. Only externally indexed samples need other frames."""
    weights, q = trans.quadrature
    if sampled:
        perms, ids, counts = field.frame.unique(dim=0, return_inverse=True, return_counts=True)
        groups = zip(perms, ids.argsort(stable=True).split(counts.tolist()))
        modal = trans.element.get_hierarchical_basis_fns(order, q.dtype, q.device)
        geom = trans.element.get_basis_fns(trans.basis_order, q.dtype, q.device)
    else:
        groups = [(None, None)]
    for perm, group in groups:
        if perm is None:
            tables = field.ref_val, field.ref_grad, trans.shape_val, field.geom_grad
            chunks = (slice(i, i + cell_limit) for i in range(0, trans.n_elements, cell_limit))
        else:
            # No permutation of q indices exists for a general conical rule.
            ref = _canonical_reference(q, perm)
            tables = modal.map(ref), modal.grad().map(ref), geom.map(ref), geom.grad().map(ref)
            chunks = group.split(cell_limit)
        for rows in chunks:
            coords = trans.points[trans.elements[rows]]
            yield rows, _chunk_batches(coords, tables, weights, size, need_grad, trans.basis_order == 1)


def integrate_volume(trans, space, kind, size, parameters, point_data, element_data,
                     quadrature_data, scalar_data, parallel_fn, integrate, parallel, *, arity):
    """Contract a matrix or load in local field slots; return rows in input cell order."""
    field = None if space is None else space.blocks[kind]
    nb = trans.n_basis if field is None else field.conn.shape[1]
    grouped = False
    if space is not None and space.basis == "hierarchical":
        # Bound modal cell x quadrature x basis-pair temporaries (before components).
        limit = max(1, _MODAL_BATCH_ENTRIES // (min(size, trans.n_quadrature) * nb**arity))
        grouped = trans.n_elements > 0 and (bool(quadrature_data) or trans.n_elements > limit)
    if grouped:
        need_grad = any(k.startswith("grad") and k[4:] in point_data for k in parameters)
        groups = _modal_groups(trans, field, space.order, size, bool(quadrature_data), limit, need_grad)
    else:
        groups = [(None, volume_batches(trans, field, size, parameters, point_data))]

    row_order, chunks = [], []
    for rows, batches in groups:
        cells = trans.elements if rows is None else trans.elements[rows]
        nodal = {k: v[cells] for k, v in point_data.items()}
        local = None
        for start, val, grad, jxw, mesh_val, mesh_grad in batches:
            args = []
            for k in parameters:
                if k == "v" or (arity == 2 and k == "u"):
                    arg = val
                elif k == "gradv" or (arity == 2 and k == "gradu"):
                    arg = grad
                elif k in nodal:
                    coeff = nodal[k]
                    table = mesh_val.to(coeff.dtype) if coeff.is_complex() else mesh_val
                    arg = torch.einsum("eb...,qb->eq...", coeff, table)
                elif k.startswith("grad") and k[4:] in nodal:
                    coeff = nodal[k[4:]]
                    table = mesh_grad.to(coeff.dtype) if coeff.is_complex() else mesh_grad
                    arg = torch.einsum("eb...,eqbd->eq...d", coeff, table)
                elif k in element_data:
                    data = element_data[k][kind]
                    arg = data if rows is None else data[rows]
                elif k in quadrature_data:
                    data = quadrature_data[k][kind]
                    arg = data[:, start:start+size] if rows is None else data[rows, start:start+size]
                else:
                    arg = scalar_data[k]
                args.append(arg)
            part = integrate(parallel_fn(*args), jxw, len(cells), nb, parallel)
            local = part if local is None else local + part
        if rows is None:
            return local
        chunks.append(local)
        if quadrature_data:
            row_order.append(rows)
    local = torch.cat(chunks)
    return local[torch.cat(row_order).argsort()] if quadrature_data else local


class _SingleFieldMixin:
    def _set_field_space(self, field):
        # The default nodal path adds no buffers.
        self._hierarchical = field

    @property
    def _field_space(self):
        return self._hierarchical

    def field_layout(self, components: int = 1):
        """DOF layout of the hierarchical field named ``u``."""
        if self._field_space is None:
            raise NotImplementedError("field_layout() requires an independent field")
        return self._field_space.layout(self, components)
