"""Field interpolation, modal traces and cell-wise evaluation for BlockLayout.

These are postprocessing/setup operations, not part of the assembly hot path.
All cell operations are batched; Python loops run over element types, edges,
or Newton iterations only. No monomial expansion is constructed.
"""

import math

import torch

from ..element import Triangle, element_type2element, element_type2order
from .topology import boundary_facet_mask


def _basis(layout, name, kind):
    field = layout.field(name)
    elem = element_type2element(kind)
    points = layout._asm._points
    factory = (elem.get_hierarchical_basis_fns if field.basis == "hierarchical"
               else elem.get_basis_fns)
    return factory(field.order, points.dtype, points.device)


def _geometry(layout, kind, cells, reference, gradients=False):
    """Map [E,Q,D] to physical points, optionally dX/d(reference)."""
    asm = layout._asm
    elem = element_type2element(kind)
    basis = elem.get_basis_fns(element_type2order[kind], reference.dtype, reference.device)
    coords = asm._points[asm.elements[kind][cells]]
    e, q, d = reference.shape
    flat = reference.reshape(-1, d)
    values = basis.map(flat).reshape(e, q, coords.shape[1])
    physical = torch.einsum("eqb,ebd->eqd", values, coords)
    if not gradients:
        return physical
    grad = basis.grad().map(flat).reshape(e, q, d, coords.shape[1])
    jacobian = torch.einsum("eqrb,ebd->eqdr", grad, coords)
    return physical, jacobian


def _sample(fn, points, components):
    """Call user data on flat [N,D] coordinates and validate/broadcast output."""
    flat = points.reshape(-1, points.shape[-1])
    values = torch.as_tensor(fn(flat), dtype=points.dtype, device=points.device)
    target = (len(flat),) if components == 1 else (len(flat), components)
    try:
        values = torch.broadcast_to(values, target)
    except RuntimeError as exc:
        raise ValueError(f"function must return values broadcastable to {target}") from exc
    return values.reshape(*points.shape[:-1], *target[1:])


def _scatter_average(out, counts, conn, local):
    out.index_add_(0, conn.reshape(-1), local.reshape(-1, *out.shape[1:]))
    counts.index_add_(0, conn.reshape(-1), torch.ones(conn.numel(), dtype=counts.dtype,
                                                   device=counts.device))


def _finish_average(out, counts):
    return out / counts.clamp_min(1).to(out.dtype).reshape(-1, *((1,) * (out.ndim - 1)))


def _modal_interpolate(layout, name, sample):
    """Interpolate sampled values by solving a reference modal collocation system.

    Uniform barycentric interpolation nodes are used, but never a nodal
    high-order basis or a monomial Vandermonde. Shared coefficients are
    averaged after orientation to remove floating-point discrepancies.
    """
    asm, field = layout._asm, layout.field(name)
    pts = asm._points
    ref = Triangle.get_basis(field.order, pts.dtype, pts.device)
    out = None
    counts = torch.zeros(layout.n_nodes(name), dtype=pts.dtype, device=pts.device)
    for kind in asm.element_types:
        key = f"{name}__{kind}"
        conn, sign = asm.field_conn[key], asm.field_sign[key]
        cells = torch.arange(len(conn), device=pts.device)
        reference = ref[None].expand(len(conn), -1, -1)
        samples = sample(kind, cells, reference)
        if out is None:
            out = torch.zeros((layout.n_nodes(name), *samples.shape[2:]),
                              dtype=samples.dtype, device=samples.device)
        table = _basis(layout, name, kind).map(ref).to(samples.dtype)
        # Broadcast one reference solve across all cells and trailing components.
        rhs = samples.reshape(len(conn), len(ref), math.prod(samples.shape[2:]))
        local = torch.linalg.solve(table, rhs).reshape_as(samples)
        local = local * sign.to(local.dtype).reshape(*sign.shape, *((1,) * (local.ndim - 2)))
        _scatter_average(out, counts, conn, local)
    return _finish_average(out, counts)


def interpolate(layout, name, fn):
    """Interpolate a callable physical field into nodal/modal coefficients."""
    field = layout.field(name)
    if field.basis != "hierarchical":
        return _sample(fn, layout.points(name), field.components)
    return _modal_interpolate(layout, name, lambda kind, cells, ref:
                              _sample(fn, _geometry(layout, kind, cells, ref), field.components))


def restrict_modal(layout, name, point_data):
    """Interpolate the mesh's nodal FE field into hierarchical coefficients."""
    asm = layout._asm
    if point_data.shape[0] != asm.n_points:
        raise ValueError(f"point_data must have leading dimension {asm.n_points}")
    if point_data.device != asm._points.device:
        raise ValueError("point_data and mesh must reside on the same device")
    if not (point_data.is_floating_point() or point_data.is_complex()):
        raise ValueError("modal interpolation requires floating or complex point_data")
    def sample(kind, cells, reference):
        elem = element_type2element(kind)
        e, q, d = reference.shape
        n_basis = asm.elements[kind].shape[1]
        table = elem.get_basis_fns(element_type2order[kind], reference.dtype, reference.device).map(
            reference.reshape(-1, d)).reshape(e, q, n_basis).to(point_data.dtype)
        return torch.einsum("eqb,eb...->eq...", table, point_data[asm.elements[kind][cells]])
    return _modal_interpolate(layout, name, sample)


def _reference_coordinates(layout, kind, cells, physical, max_iterations, tolerance):
    if element_type2element(kind) is not Triangle:
        raise NotImplementedError("physical evaluation currently supports triangles only")
    if (isinstance(max_iterations, bool) or not isinstance(max_iterations, int)
            or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer")
    if tolerance is not None and (not math.isfinite(tolerance) or tolerance <= 0):
        raise ValueError("tolerance must be finite and positive")

    def solve(jacobian, rhs):
        try:
            return torch.linalg.solve(jacobian, rhs)
        except torch.linalg.LinAlgError as exc:
            raise ValueError("physical-to-reference mapping failed: singular geometry Jacobian") from exc

    asm = layout._asm
    coords = asm._points[asm.elements[kind][cells, :3]]
    jac = (coords[:, 1:] - coords[:, :1]).transpose(-1, -2)
    reference = solve(jac[:, None], (physical - coords[:, :1])[..., None]).squeeze(-1)
    if physical.numel() == 0:
        return reference
    eps = torch.finfo(physical.dtype).eps
    scale = max(1.0, float(coords.detach().abs().max()))
    tol = 128 * eps * scale if tolerance is None else tolerance
    for _ in range(max_iterations):
        mapped, jac = _geometry(layout, kind, cells, reference, gradients=True)
        residual = mapped - physical
        if float(residual.detach().abs().max()) <= tol:
            break
        reference = reference - solve(jac, residual[..., None]).squeeze(-1)
    residual = _geometry(layout, kind, cells, reference) - physical
    if not torch.isfinite(reference).all() or float(residual.detach().abs().max()) > tol:
        raise ValueError("physical-to-reference mapping did not converge")
    domain_tol = max(1024 * eps, tol / scale)
    if (reference < -domain_tol).any() or (reference.sum(-1) > 1 + domain_tol).any():
        raise ValueError("physical point lies outside its selected triangle")
    return reference


def evaluate(layout, name, coefficients, points, element_type=None, cell_indices=None,
             physical=False, gradient=False, max_iterations=30, tolerance=None):
    """Cell-wise values/physical gradients, at shared or per-cell points."""
    asm = layout._asm
    if coefficients.ndim < 1 or coefficients.shape[0] != layout.n_nodes(name):
        raise ValueError(f"coefficients must have leading dimension {layout.n_nodes(name)}")
    if not (coefficients.is_floating_point() or coefficients.is_complex()):
        raise ValueError("coefficients must have floating or complex dtype")
    if coefficients.device != asm._points.device:
        raise ValueError("coefficients and mesh must reside on the same device")
    if element_type is None:
        if len(asm.element_types) != 1:
            raise ValueError("element_type is required for a mixed-element mesh")
        element_type = asm.element_types[0]
    if element_type not in asm.element_types:
        raise ValueError(f"unknown element_type {element_type!r}")
    conn = asm.field_conn[f"{name}__{element_type}"]
    if cell_indices is None:
        cells = torch.arange(len(conn), device=conn.device)
    else:
        cells = torch.as_tensor(cell_indices, device=conn.device)
        if cells.dtype not in (torch.int32, torch.int64) or cells.ndim > 1:
            raise ValueError("cell_indices must be an integer scalar or 1D array")
        cells = cells.long().reshape(-1)
        if ((cells < 0) | (cells >= len(conn))).any():
            raise ValueError("cell_indices are out of range")
    reference = torch.as_tensor(points, dtype=asm._points.dtype, device=asm._points.device)
    d = asm.dimension
    if reference.ndim == 2 and reference.shape[-1] == d:
        reference = reference[None].expand(len(cells), -1, -1)
    elif reference.ndim != 3 or reference.shape[0] != len(cells) or reference.shape[-1] != d:
        raise ValueError("points must have shape [Q,D] or [n_selected_cells,Q,D]")
    if not torch.isfinite(reference).all():
        raise ValueError("evaluation points must be finite")
    if physical:
        reference = _reference_coordinates(layout, element_type, cells, reference,
                                           max_iterations, tolerance)
    basis = _basis(layout, name, element_type)
    local = coefficients[conn[cells]]
    key = f"{name}__{element_type}"
    if key in asm.field_sign:
        sign = asm.field_sign[key][cells]
        local = local * sign.to(local.dtype).reshape(*sign.shape, *((1,) * (local.ndim - 2)))
    e, q, _ = reference.shape
    flat = reference.reshape(-1, d)
    if gradient:
        grad = basis.grad().map(flat).reshape(e, q, d, conn.shape[1])
        _, jac = _geometry(layout, element_type, cells, reference, gradients=True)
        # J^T grad_phys = grad_reference for every basis function.
        grad = torch.linalg.solve(jac.transpose(-1, -2), grad).transpose(-1, -2)
        return torch.einsum("eqbd,eb...->eq...d", grad.to(local.dtype), local)
    values = basis.map(flat).reshape(e, q, conn.shape[1])
    return torch.einsum("eqb,eb...->eq...", values.to(local.dtype), local)


def prolong_modal(layout, name, coefficients):
    """Evaluate a modal field at the geometry's nodes (for output/plotting)."""
    asm = layout._asm
    out = torch.zeros((asm.n_points, *coefficients.shape[1:]),
                      dtype=coefficients.dtype, device=coefficients.device)
    counts = torch.zeros(asm.n_points, dtype=asm._points.dtype, device=asm._points.device)
    for kind in asm.element_types:
        ref = element_type2element(kind).get_basis(
            element_type2order[kind], asm._points.dtype, asm._points.device)
        values = evaluate(layout, name, coefficients, ref, element_type=kind)
        _scatter_average(out, counts, asm.elements[kind], values)
    return _finish_average(out, counts)


def dirichlet_values(layout, name, fn, component=None):
    """Full block-vector lifting from boundary trace interpolation only."""
    asm, field = layout._asm, layout.field(name)
    if component is not None and not 0 <= component < field.components:
        raise ValueError(f"component must lie in [0, {field.components})")
    c = field.components if component is None else 1
    points = asm._points
    shape = (layout.n_nodes(name),) if c == 1 else (layout.n_nodes(name), c)
    coefficients = torch.zeros(shape, dtype=points.dtype, device=points.device)
    if field.basis != "hierarchical":
        mask = layout.boundary_mask(name, component=0)
        off = asm._offsets[name]
        mask = mask[off:off + layout.n_nodes(name) * field.components].reshape(-1, field.components)[:, 0]
        coefficients[mask] = _sample(fn, layout.points(name)[mask], c)
    else:
        counts = torch.zeros(layout.n_nodes(name), dtype=points.dtype, device=points.device)
        boundary = boundary_facet_mask(asm.elements)
        p = field.order
        # Chebyshev-Lobatto nodes reduce equispaced high-order trace
        # interpolation instability. They need no quadrature extension.
        t = (1 - torch.cos(torch.arange(p + 1, dtype=points.dtype, device=points.device) * math.pi / p)) / 2
        verts = Triangle.points.to(points)
        for kind in asm.element_types:
            key = f"{name}__{kind}"
            basis = _basis(layout, name, kind)
            for edge, (a, b) in enumerate(Triangle.edge.tolist()):
                cells = boundary[kind][0][:, edge].nonzero().flatten()
                if not len(cells):
                    continue
                ref = (1 - t[:, None]) * verts[a] + t[:, None] * verts[b]
                slots = torch.cat((torch.tensor([a, b], device=points.device),
                                   torch.arange(3 + edge * (p - 1), 3 + (edge + 1) * (p - 1), device=points.device)))
                table = basis.map(ref)[:, slots]
                physical = _geometry(layout, kind, cells, ref[None].expand(len(cells), -1, -1))
                samples = _sample(fn, physical, c)
                rhs = samples.reshape(len(cells), p + 1, c)
                local = torch.linalg.solve(table, rhs).reshape_as(samples)
                conn = asm.field_conn[key][cells][:, slots]
                sign = asm.field_sign[key][cells][:, slots]
                local = local * sign.reshape(*sign.shape, *((1,) * (local.ndim - 2)))
                _scatter_average(coefficients, counts, conn, local)
        coefficients = _finish_average(coefficients, counts)
    block = torch.zeros(layout.n_nodes(name), field.components, dtype=points.dtype, device=points.device)
    if component is None:
        block[:] = coefficients.reshape(-1, field.components)
    else:
        block[:, component] = coefficients
    lifting = torch.zeros(layout.n_dofs, dtype=points.dtype, device=points.device)
    off = asm._offsets[name]
    lifting[off:off + block.numel()] = block.reshape(-1)
    return lifting
