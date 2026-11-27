"""Entity interpolation and oriented evaluation of hierarchical tetrahedra.

These are setup/postprocessing operations. Temporary tables follow the current
mesh dtype/device and are not cached across calls or coordinate updates.
Persistent connectivity and orientation are read from assembler buffers.
"""

from itertools import product
import math

import torch

from ..element import Tetrahedron
from ..element.hierarchical_tetrahedron import TETRAHEDRON_ORIENTATIONS


def _interior_samples(order, dimension, points):
    """Positive barycentric integer lattice, divided by p, in lexicographic order.

    Faces use triples and cells quadruples with every integer >= 1 and sum p.
    Only entity-interior residuals are fitted here; edges use Lobatto samples.
    """
    rows = [(*head, order - sum(head))
            for head in product(range(1, order), repeat=dimension)
            if sum(head) < order]
    return points.new_tensor(rows).reshape(-1, dimension + 1) / order


def _entity_stages(order, points):
    """Yield (local corner slots, modal slots, barycentric samples) by dimension."""
    yield [(tuple([v]), [v], points.new_ones(1, 1)) for v in range(4)]
    if order >= 2:
        t = (1 - torch.cos(torch.arange(1, order, dtype=points.dtype,
                                       device=points.device) * math.pi / order)) / 2
        weights = torch.stack((1 - t, t), -1)
        yield [(tuple(edge), list(range(4 + e * (order - 1),
                                       4 + (e + 1) * (order - 1))), weights)
               for e, edge in enumerate(Tetrahedron.edge.tolist())]
    n_face = (order - 1) * (order - 2) // 2
    face_start = 4 + 6 * (order - 1)
    if n_face:
        weights = _interior_samples(order, 2, points)
        yield [(face, list(range(face_start + f * n_face,
                                face_start + (f + 1) * n_face)), weights)
               for f, face in enumerate(Tetrahedron.face)]
    if order >= 4:
        start = face_start + 4 * n_face
        size = (order - 1) * (order - 2) * (order - 3) // 6
        yield [((0, 1, 2, 3), list(range(start, start + size)),
                _interior_samples(order, 3, points))]


def _representatives(cells, first_dof):
    """Choose one incident cell per entity, batched and in stable cell order."""
    unique, inverse = torch.unique(first_dof, return_inverse=True)
    first = torch.full_like(unique, len(cells))
    first.scatter_reduce_(0, inverse, torch.arange(len(cells), device=cells.device),
                          reduce="amin", include_self=True)
    return cells[first]


def interpolate_entities(layout, name, sample, boundary_only=False, components=None):
    """Fit vertices, edges, face residuals, then private cell residuals.

    Parameters
    ----------
    layout : BlockLayout
        Layout backed by a tetrahedral mixed assembler.
    name : str
        Field trial name.
    sample : callable
        Receives (element type, cell IDs, reference points [E,Q,3]) and
        returns [E,Q,...] samples, possibly with arbitrary trailing data axes.
    boundary_only : bool, optional
        Fit only entities in the closure of boundary faces. No other physical
        points are passed to ``sample`` and all other coefficients stay zero.
    components : int, optional
        Output components for an empty boundary; normally inferred from samples.

    Returns
    -------
    torch.Tensor
        Global modal coefficients [n_scalar_dofs,...]. Every shared entity is
        expanded once using one representative incident cell, without averaging.
    """
    asm, field = layout._asm, layout.field(name)
    points = asm._points
    n = layout.n_nodes(name)
    seen = torch.zeros(n, dtype=torch.bool, device=points.device)
    allowed = torch.ones_like(seen)
    if boundary_only:
        mask = layout.boundary_mask(name)
        offset = asm._offsets[name]
        allowed = mask[offset:offset + n * field.components].reshape(n, field.components)[:, 0]
    out = None
    vertices = Tetrahedron.points.to(points)
    # The complete lower-dimensional skeleton must be fitted before residuals
    # on the next dimension, including across different element types/classes.
    for entities in _entity_stages(field.order, points):
        for kind in asm.element_types:
            conn = asm.field_conn[f"{name}__{kind}"]
            orientation_ids = asm.cell_orientation[kind]
            for orientation_id, orientation in enumerate(TETRAHEDRON_ORIENTATIONS):
                group = (orientation_ids == orientation_id).nonzero().flatten()
                if not len(group):
                    continue
                basis = Tetrahedron.get_hierarchical_basis_fns(
                    field.order, points.dtype, points.device, orientation=orientation)
                rank = {slot: i for i, slot in enumerate(orientation)}
                for corners, modes, weights in entities:
                    slots = torch.tensor(modes, dtype=torch.long, device=points.device)
                    first = conn[group, modes[0]]
                    cells = group[allowed[first] & ~seen[first]]
                    if not len(cells):
                        continue
                    cells = _representatives(cells, conn[cells, modes[0]])
                    # Cell bubbles have a fixed local convention; shared traces
                    # use increasing global vertex IDs inside each entity.
                    ordered = sorted(corners, key=rank.__getitem__) if len(corners) < 4 else corners
                    ref = weights @ vertices[list(ordered)]
                    values = sample(kind, cells, ref[None].expand(len(cells), -1, -1))
                    if out is None:
                        out = values.new_zeros((n, *values.shape[2:]))
                    table = basis.map(ref).to(values.dtype)
                    residual = values - torch.einsum("qb,eb...->eq...", table, out[conn[cells]])
                    rhs = residual.reshape(len(cells), len(ref), math.prod(values.shape[2:]))
                    fitted = torch.linalg.solve(table[:, slots], rhs).reshape_as(values)
                    target = conn[cells][:, slots]
                    out[target] = fitted
                    seen[target.reshape(-1)] = True
    if out is None:
        c = field.components if components is None else components
        return points.new_zeros((n,) if c == 1 else (n, c))
    return out


def evaluate_oriented(layout, name, coefficients, kind, cells, reference, gradient=False):
    """Evaluate orientation-homogeneous batches and restore the requested order."""
    from .field_evaluation import _geometry

    asm = layout._asm
    e, q, d = reference.shape
    shape = (e, q, *coefficients.shape[1:], *((d,) if gradient else ()))
    out = coefficients.new_zeros(shape)
    if not e or not q:
        return out
    conn = asm.field_conn[f"{name}__{kind}"]
    orientations = asm.cell_orientation[kind][cells]
    shared_points = reference.stride(0) == 0
    for orientation_id, orientation in enumerate(TETRAHEDRON_ORIENTATIONS):
        selected = (orientations == orientation_id).nonzero().flatten()
        if not len(selected):
            continue
        basis = Tetrahedron.get_hierarchical_basis_fns(
            layout.field(name).order, reference.dtype, reference.device,
            orientation=orientation)
        ref = reference[selected]
        flat = reference[0] if shared_points else ref.reshape(-1, d)
        local = coefficients[conn[cells[selected]]]
        if gradient:
            grad = basis.grad().map(flat)
            grad = grad[None] if shared_points else grad.reshape(len(selected), q, d, -1)
            _, jac = _geometry(layout, kind, cells[selected], ref, gradients=True)
            grad = torch.linalg.solve(jac.transpose(-1, -2), grad).transpose(-1, -2)
            values = torch.einsum("eqbd,eb...->eq...d", grad.to(local.dtype), local)
        else:
            table = basis.map(flat).to(local.dtype)
            if shared_points:
                values = torch.einsum("qb,eb...->eq...", table, local)
            else:
                values = torch.einsum("eqb,eb...->eq...", table.reshape(len(selected), q, -1), local)
        out = out.index_copy(0, selected, values)
    return out
