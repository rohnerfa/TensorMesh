"""Deterministic tetrahedra and cellwise integration oracles for assembly tests."""

import itertools
import math

import meshio
import numpy as np
import torch

from tensormesh import Field, Mesh, MixedElementAssembler
from tensormesh.element import Tetrahedron


ORIENTATIONS = tuple(itertools.permutations(range(4)))
DTYPE = torch.float64


def _mesh(vertices, cells, order=1, curved=False):
    """Build P1/P2 geometry in internal ordering, sharing edge midpoints."""
    if order not in (1, 2):
        raise ValueError("The tetrahedron fixture supports geometry orders 1 and 2")
    points = vertices.clone()
    conn = cells.clone()
    if order == 2:
        edges = sorted({tuple(sorted((int(cell[a]), int(cell[b]))))
                        for cell in cells for a, b in Tetrahedron.edge.tolist()})
        edge_ids = {edge: len(vertices) + i for i, edge in enumerate(edges)}
        midpoints = torch.stack([(vertices[a] + vertices[b]) / 2 for a, b in edges])
        points = torch.cat((vertices, midpoints))
        mids = torch.tensor([[edge_ids[tuple(sorted((int(cell[a]), int(cell[b]))))]
                              for a, b in Tetrahedron.edge.tolist()] for cell in cells])
        conn = torch.cat((cells, mids), dim=1)
    if curved:
        if order != 2:
            raise ValueError("Curved geometry needs quadratic nodes")
        x, y, z = points.unbind(-1)
        points = torch.stack((x + 0.12 * y * z,
                              y + 0.10 * x * z,
                              z + 0.08 * x * y), dim=-1)
    kind = "tetra" if order == 1 else "tetra10"
    # The nodes above already follow Tetrahedron.edge. In particular, applying
    # get_gmsh_permutation(10) here would swap two midside nodes incorrectly.
    mesh = Mesh(meshio.Mesh(points.numpy(), [(kind, conn.numpy())]), reorder=False).double()
    torch.testing.assert_close(mesh.elements(), conn)
    return mesh


def patch_mesh(order=1, curved=False):
    """Two tetrahedra sharing the face x+y+z=1, with scrambled corner IDs."""
    vertices = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.],
                             [0., 0., 1.], [1., 1., 1.]], dtype=DTYPE)
    cells = torch.tensor([[2, 0, 3, 1], [4, 3, 1, 2]])
    return _mesh(vertices, cells, order, curved)


def orientation_mesh(class_ids=None, order=1, curved=False):
    """Nonoverlapping translated tetrahedra with prescribed lexicographic classes.

    ``class_ids`` is in original cell order and may repeat, giving uneven groups.
    All 24 classes occur once by default. The one-cell class-0 mesh is the unit
    reference tetrahedron; the other cells have mildly sheared affine geometry.
    """
    if class_ids is None:
        class_ids = list(range(23, -1, -1))
    vertices, cells = [], []
    reference = Tetrahedron.points.to(DTYPE)
    for cell, class_id in enumerate(class_ids):
        shear = torch.tensor([[1 + 0.02 * cell, 0.01 * cell, 0.],
                              [0., 1., 0.015 * cell], [0., 0., 1.]], dtype=DTYPE)
        offset = torch.tensor([3. * cell, 0., 0.], dtype=DTYPE)
        vertices.append(reference @ shear.T + offset)
        # pi lists local slots in increasing global-ID order, so conn=pi^-1.
        ranks = torch.argsort(torch.tensor(ORIENTATIONS[class_id]))
        cells.append(4 * cell + ranks)
    return _mesh(torch.cat(vertices), torch.stack(cells), order, curved)


def poisson(p, family="hierarchical", components=1):
    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", p, components=components, basis=family)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            return (1 + x[0] + x[1] ** 2 + 0.5 * x[2]) * v.sum()

    return Poisson


def independent_quadrature(degree):
    """Conical Gauss-Legendre rule, independent of TensorMesh's rule generator.

    The Jacobian is explicit; n integrates the transformed degree d+2 in u.
    """
    n = math.ceil((degree + 3) / 2)
    nodes, weights = np.polynomial.legendre.leggauss(n)
    nodes = torch.from_numpy((nodes + 1) / 2)
    weights = torch.from_numpy(weights / 2)
    u, v, w = torch.meshgrid(nodes, nodes, nodes, indexing="ij")
    wu, wv, ww = torch.meshgrid(weights, weights, weights, indexing="ij")
    points = torch.stack((u, (1 - u) * v, (1 - u) * (1 - v) * w), -1)
    return (wu * wv * ww * (1 - u) ** 2 * (1 - v)).reshape(-1), points.reshape(-1, 3)


def cell_tables(asm, cell, reference, weights, points=None, element_type=None):
    """Evaluate a cell without consulting grouped tables or transformations.

    Returns physical points, integration weights, field value/gradient tables,
    and nodal geometry value/physical-gradient tables.
    """
    kind = element_type or asm.element_types[0]
    points = asm._points if points is None else points
    conn = asm.elements[kind][cell]
    coords = points[conn]
    geom = Tetrahedron.get_basis_fns(asm.mesh_order, reference.dtype, reference.device)
    geom_val = geom.map(reference)
    geom_grad = geom.grad().map(reference)
    # J has physical rows and reference columns; use torch.linalg independently
    # of the production small_inverse/Transformation implementation.
    jacobian = torch.einsum("bi,qrb->qir", coords, geom_grad)
    inverse = torch.linalg.inv(jacobian)
    jxw = weights * torch.linalg.det(jacobian).abs()
    physical = geom_val @ coords
    geometry_grad = torch.einsum("qrb,qrd->qbd", geom_grad, inverse)
    orientation = tuple(torch.argsort(conn[:4]).tolist())
    tables = {}
    for field in asm.fields:
        if field.basis == "hierarchical":
            basis = Tetrahedron.get_hierarchical_basis_fns(
                field.order, reference.dtype, reference.device, orientation=orientation)
        else:
            basis = Tetrahedron.get_basis_fns(field.order, reference.dtype, reference.device)
        tables[field.trial] = (basis.map(reference),
                               torch.einsum("qrb,qrd->qbd", basis.grad().map(reference), inverse))
    return physical, jxw, tables, (geom_val, geometry_grad)


def field_indices(asm, name, cell, element_type=None):
    """Global component-expanded field indices for one original-order cell."""
    kind = element_type or asm.element_types[0]
    field = asm.layout.field(name)
    conn = asm.field_conn[f"{name}__{kind}"][cell]
    return (asm.layout.offsets[name] + conn[:, None] * field.components
            + torch.arange(field.components, device=conn.device)).reshape(-1)


def scatter_matrix(matrix, rows, cols, local):
    """Dense cellwise scatter; local field maps have no repeated indices."""
    matrix[rows[:, None], cols[None, :]] += local


def scalar_reference(asm, degree=None, points=None):
    """Independent scalar mass, stiffness, and polynomial load integration."""
    if degree is None:
        degree = 2 * max(field.order for field in asm.fields) + 2
    weights, reference = independent_quadrature(degree)
    mass = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=DTYPE)
    stiffness = torch.zeros_like(mass)
    load = torch.zeros(asm.n_dofs, dtype=DTYPE)
    kind = asm.element_types[0]
    for cell in range(len(asm.elements[kind])):
        x, jxw, tables, _ = cell_tables(asm, cell, reference, weights, points)
        value, grad = tables["u"]
        indices = field_indices(asm, "u", cell)
        scatter_matrix(mass, indices, indices, torch.einsum("qi,qj,q->ij", value, value, jxw))
        scatter_matrix(stiffness, indices, indices, torch.einsum("qid,qjd,q->ij", grad, grad, jxw))
        forcing = 1 + x[:, 0] + x[:, 1] ** 2 + 0.5 * x[:, 2]
        load[indices] += torch.einsum("qi,q,q->i", value, forcing, jxw)
    return mass, stiffness, load


def physical_points(asm, reference):
    kind = asm.element_types[0]
    table = Tetrahedron.get_basis_fns(asm.mesh_order, reference.dtype, reference.device).map(reference)
    return torch.einsum("qb,ebd->eqd", table, asm._points[asm.elements[kind]])


def change_of_basis(nodal, modal, name="u"):
    """Global H->L map from independent nodal sampling, for supported P1/P2."""
    field = modal.layout.field(name)
    assert field.order in (1, 2)
    transform = torch.zeros(nodal.layout.n_nodes(name), modal.layout.n_nodes(name), dtype=DTYPE)
    seen = torch.zeros(transform.shape[0], dtype=torch.bool)
    reference = Tetrahedron.get_basis(field.order, DTYPE)
    for kind in modal.element_types:
        key = f"{name}__{kind}"
        for cell, (cl, ch) in enumerate(zip(nodal.field_conn[key], modal.field_conn[key])):
            orientation = tuple(torch.argsort(modal.elements[kind][cell, :4]).tolist())
            values = Tetrahedron.get_hierarchical_basis_fns(
                field.order, DTYPE, orientation=orientation).map(reference)
            rows = torch.zeros(len(cl), transform.shape[1], dtype=DTYPE)
            rows[:, ch] = values
            torch.testing.assert_close(transform[cl[seen[cl]]], rows[seen[cl]],
                                       atol=2e-13, rtol=2e-12)
            transform[cl] = rows
            seen[cl] = True
    assert seen.all()
    return transform
