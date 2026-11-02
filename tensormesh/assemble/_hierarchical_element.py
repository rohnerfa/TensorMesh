"""Private field tables for ElementAssembler's hierarchical tetrahedral path.

Geometry stays in Transformation. Only field connectivity and reference tables
are persistent here; physical tables are disposable, graph-safe caches.
"""

import torch
from torch import nn

from ..element import Tetrahedron, TETRAHEDRON_ORIENTATIONS
from ..element.element import small_inverse
from ..element.transformation import _version_counter


class _TetrahedronTables(nn.Module):
    def __init__(self, conn, orientation, trans, order):
        super().__init__()
        self.register_buffer("conn", conn)
        classes, table_ids = torch.unique(orientation, sorted=True, return_inverse=True)
        self.register_buffer("table_ids", table_ids)
        _, q = trans.quadrature
        # Warm and cold hierarchical assemblers have the same persistent keys.
        _ = trans.shape_val
        values, gradients = [], []
        # An empty element-type block still has well-defined reference tables.
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
        # As with Transformation, basis/geometry stay real under complex forms.
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
        # Also notice in-place coordinate changes when the caller omits points.
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


class _HierarchicalElement(nn.Module):
    def __init__(self, mapping, transformations, order):
        super().__init__()
        self.order = order
        self.n_dofs = mapping.n_dofs
        self.blocks = nn.ModuleDict({
            kind: _TetrahedronTables(conn, mapping.topology.orientation[kind],
                                    transformations[kind], order)
            for kind, conn in mapping.conn.items()
        })

    def get_extra_state(self):
        return {"order": self.order, "n_dofs": self.n_dofs,
                "element_types": list(self.blocks)}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise RuntimeError(
                "hierarchical ElementAssembler checkpoint has a different field layout"
            )
