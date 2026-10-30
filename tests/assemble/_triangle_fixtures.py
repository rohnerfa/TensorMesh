"""Small, deterministic triangle spaces shared by modal integration tests."""

import meshio
import torch

from tensormesh import Field, Mesh, MixedElementAssembler
from tensormesh.element import Triangle
from tensormesh.assemble.topology import lagrange_dofmap


def patch_mesh(order=1, curved=False):
    vertices = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    cells = torch.tensor([[0, 1, 2], [3, 2, 1]])
    if order == 1:
        points, conn, kind = vertices, cells, "triangle"
    else:
        kind = {2: "triangle6", 3: "triangle10"}[order]
        dofs = lagrange_dofmap({"triangle": cells}, 4, order)
        conn, ref = dofs.conn["triangle"], dofs.ref_nodes["triangle"]
        points = torch.zeros(dofs.n_dofs, 2, dtype=torch.float64)
        for k, cell in enumerate(cells):
            v = vertices[cell]
            points[conn[k]] = v[0] + ref[:, :1] * (v[1] - v[0]) + ref[:, 1:] * (v[2] - v[0])
        if curved:
            x, y = points.unbind(-1)
            points = torch.stack((x + 0.12 * x * y, y + 0.08 * x * (1 - x)), -1)
    return Mesh(meshio.Mesh(points.numpy(), [(kind, conn.numpy())])).double()


def poisson(p, family):
    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", order=p, basis=family)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            return (1 + x[0] + x[1] ** 2) * v

    return Poisson


def physical_points(asm, reference):
    kind = asm.element_types[0]
    table = Triangle.get_basis_fns(asm.mesh_order, torch.float64).map(reference)
    return torch.einsum("qb,ebd->eqd", table, asm._points[asm.elements[kind]])


def change_of_basis(nodal, modal, name="u"):
    """Global H->L map by sampling; check shared values from incident cells."""
    field = modal.layout.field(name)
    transform = torch.zeros(nodal.layout.n_nodes(name), modal.layout.n_nodes(name), dtype=torch.float64)
    seen = torch.zeros(transform.shape[0], dtype=torch.bool)
    values = Triangle.get_hierarchical_basis_fns(field.order, torch.float64).map(
        Triangle.get_basis(field.order, torch.float64))
    for kind in modal.element_types:
        key = f"{name}__{kind}"
        for cl, ch, sign in zip(nodal.field_conn[key], modal.field_conn[key], modal.field_sign[key]):
            rows = torch.zeros(len(cl), transform.shape[1], dtype=torch.float64)
            rows[:, ch] = values * sign
            torch.testing.assert_close(transform[cl[seen[cl]]], rows[seen[cl]], atol=2e-13, rtol=2e-12)
            transform[cl] = rows
            seen[cl] = True
    assert seen.all()
    return transform
