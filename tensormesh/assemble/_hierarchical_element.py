"""Private triangle field tables; geometry remains nodal in Transformation."""

import torch
from torch import nn

from ..element import Triangle
from ..element.element import small_inverse
from ..element.transformation import _version_counter


class _TriangleTables(nn.Module):
    def __init__(self, conn, sign, trans, order):
        super().__init__()
        self.register_buffer("conn", conn)
        self.register_buffer("sign", sign)
        _, q = trans.quadrature
        # Keep persistent keys identical before and after the first assembly.
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
        # Apply signs once, after quadrature, including scalar/vector blocks.
        trailing = (1,) * (local.ndim - 3)
        return (local * self.sign.reshape(*self.sign.shape, 1, *trailing)
                * self.sign.reshape(self.sign.shape[0], 1, self.sign.shape[1], *trailing))


class _HierarchicalElement(nn.Module):
    def __init__(self, mapping, transformations, order):
        super().__init__()
        self.order = order
        self.n_dofs = mapping.n_dofs
        self.blocks = nn.ModuleDict({
            kind: _TriangleTables(conn, mapping.sign[kind], transformations[kind], order)
            for kind, conn in mapping.conn.items()
        })

    def layout(self, assembler, components):
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
