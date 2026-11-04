"""Private triangle/tetrahedron field tables; geometry remains nodal."""

import torch
from torch import nn

from ..element import Triangle, Tetrahedron, TETRAHEDRON_ORIENTATIONS
from ..element.element import small_inverse
from ..element.transformation import _version_counter


class _TriangleTables(nn.Module):
    element_dependent = False

    def __init__(self, conn, sign, trans, order):
        super().__init__()
        self.register_buffer("conn", conn)
        self.register_buffer("sign", sign)
        _, q = trans.quadrature
        _ = trans.shape_val
        basis = Triangle.get_hierarchical_basis_fns(order, q.dtype, q.device)
        self.register_buffer("ref_val", basis.map(q))              # [Q,B]
        self.register_buffer("ref_grad", basis.grad().map(q))      # [Q,D,B]
        self.register_buffer("geom_grad", Triangle.get_basis_grad_fns(
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
               _version_counter(trans.elements), _version_counter(self.ref_grad),
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
        jac = torch.einsum("ebj,qib->eqij", trans.element_coords, self.geom_grad[start:stop])
        inverse = small_inverse(jac)
        jxw = torch.linalg.det(jac).abs() * w[None, start:stop]
        gradient = torch.einsum("qib,eqji->eqbj", self.ref_grad[start:stop], inverse)
        if cacheable and start == 0 and stop == trans.n_quadrature:
            self.register_buffer("_grad", gradient, persistent=False)
            self.register_buffer("_jxw", jxw, persistent=False)
            self.register_buffer("_inverse", inverse, persistent=False)
        return values, gradient, jxw, inverse

    def orient(self, local):
        trailing = (1,) * (local.ndim - 3)
        return (local * self.sign.reshape(*self.sign.shape, 1, *trailing)
                * self.sign.reshape(self.sign.shape[0], 1, self.sign.shape[1], *trailing))


class _TetrahedronTables(nn.Module):
    element_dependent = True

    def __init__(self, conn, orientation, trans, order):
        super().__init__()
        self.register_buffer("conn", conn)
        classes, table_ids = torch.unique(orientation, sorted=True, return_inverse=True)
        self.register_buffer("table_ids", table_ids)
        _, q = trans.quadrature
        _ = trans.shape_val
        values, gradients = [], []
        for class_id in classes.tolist() or [0]:
            basis = Tetrahedron.get_hierarchical_basis_fns(
                order, q.dtype, q.device, orientation=TETRAHEDRON_ORIENTATIONS[class_id])
            values.append(basis.map(q))
            gradients.append(basis.grad().map(q))
        self.register_buffer("ref_val", torch.stack(values))       # [O,Q,B]
        self.register_buffer("ref_grad", torch.stack(gradients))   # [O,Q,D,B]
        self.register_buffer("geom_grad", Tetrahedron.get_basis_grad_fns(
            trans.basis_order, q.dtype, q.device).map(q))
        self._values_key = None
        self._geometry_key = None

    def _apply(self, fn, recurse=True):
        def real_only(t):
            out = fn(t)
            return out.real if out.is_complex() else out
        return super()._apply(real_only, recurse=recurse)

    def values(self, start, stop):
        key = (id(self.ref_val), _version_counter(self.ref_val),
               id(self.table_ids), _version_counter(self.table_ids))
        if key != self._values_key:
            self._buffers.pop("_values", None)
            self._values_key = key
        if "_values" in self._buffers:
            return self._values[:, start:stop]
        values = self.ref_val[:, start:stop].index_select(0, self.table_ids)
        if start == 0 and stop == self.ref_val.shape[1]:
            self.register_buffer("_values", values, persistent=False)
        return values

    def batch(self, trans, start, size):
        trans.update_points(trans.points)
        points = trans.points
        stop = min(start + size, trans.n_quadrature)
        version = _version_counter(points)
        key = (id(points), version, points.dtype, points.device,
               _version_counter(trans.elements), _version_counter(self.ref_grad),
               _version_counter(self.geom_grad), _version_counter(self.table_ids))
        cacheable = version is not None and not (
            torch.is_grad_enabled() and points.requires_grad)
        if key != self._geometry_key or not cacheable:
            for name in ("_grad", "_jxw", "_inverse"):
                self._buffers.pop(name, None)
            self._geometry_key = key if cacheable else None
        values = self.values(start, stop)
        if "_grad" in self._buffers:
            return (values, self._grad[:, start:stop], self._jxw[:, start:stop],
                    self._inverse[:, start:stop])
        w, _ = trans.quadrature
        jac = torch.einsum("ebj,qib->eqij", trans.element_coords, self.geom_grad[start:stop])
        inverse = small_inverse(jac)
        jxw = torch.linalg.det(jac).abs() * w[None, start:stop]
        reference = self.ref_grad[:, start:stop].index_select(0, self.table_ids)
        gradient = torch.einsum("eqib,eqji->eqbj", reference, inverse)
        if cacheable and start == 0 and stop == trans.n_quadrature:
            self.register_buffer("_grad", gradient, persistent=False)
            self.register_buffer("_jxw", jxw, persistent=False)
            self.register_buffer("_inverse", inverse, persistent=False)
        return values, gradient, jxw, inverse

    def orient(self, local):
        # Tetrahedral reference tables already incorporate orientation.
        return local


class _HierarchicalElement(nn.Module):
    def __init__(self, mapping, transformations, order):
        super().__init__()
        self.order = order
        self.n_dofs = mapping.n_dofs
        blocks = {}
        for kind, conn in mapping.conn.items():
            trans = transformations[kind]
            if trans.element is Triangle:
                blocks[kind] = _TriangleTables(conn, mapping.sign[kind], trans, order)
            else:
                blocks[kind] = _TetrahedronTables(
                    conn, mapping.topology.orientation[kind], trans, order)
        self.blocks = nn.ModuleDict(blocks)
        self.element_dependent = next(iter(blocks.values())).element_dependent

    def layout(self, assembler, components):
        if self.element_dependent:
            raise NotImplementedError("single-field layout helpers currently support triangles only")
        from .mixed_assembler import BlockLayout, Field
        field = Field("u", "v", self.order, components, basis="hierarchical")
        return BlockLayout(_SingleFieldView(assembler, field))

    def get_extra_state(self):
        return {"order": self.order, "n_dofs": self.n_dofs,
                "element_types": list(self.blocks)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError(
                "hierarchical assembler checkpoint has a different field layout"
            )


class _SingleFieldView:
    """Live buffer view for reusing BlockLayout, without a mixed assembler."""

    def __init__(self, assembler, field):
        self._assembler = assembler
        self.fields = [field]
        self._offsets = {"u": 0}
        self.field_kind = {"u": "hierarchical"}
        self.grouped_tetrahedra = False
        self.field_n_nodes = {"u": assembler._hierarchical.n_dofs}
        self.n_dofs = self.field_n_nodes["u"] * field.components
        self.n_points = assembler.n_points
        self.dimension = assembler.dimension
        self.element_types = assembler.element_types
        self.elements = assembler.elements

    def _field(self, name):
        if name != "u":
            raise KeyError(f"unknown field {name!r}; the single field is 'u'")
        return self.fields[0]

    @property
    def _points(self):
        return next(iter(self._assembler.transformation.values())).points

    @property
    def field_conn(self):
        return {f"u__{kind}": block.conn
                for kind, block in self._assembler._hierarchical.blocks.items()}

    @property
    def field_sign(self):
        return {f"u__{kind}": block.sign
                for kind, block in self._assembler._hierarchical.blocks.items()}
