"""Single-field frame contracts for both simplex shapes."""
from itertools import permutations

import meshio
import pytest
import torch

from tensormesh import Mesh, NodeAssembler
from tensormesh.assemble import MassElementAssembler
from tensormesh.assemble import _single_field
from tensormesh.element import Tetrahedron, Triangle
from tests.assemble._tetrahedron_fixtures import patch_mesh as tet_patch
from tests.assemble._tetrahedron_fixtures import orientation_mesh
from tests.assemble._triangle_fixtures import patch_mesh as tri_patch


@pytest.mark.parametrize("elem,p,kind", [(Triangle, 4, "triangle"), (Tetrahedron, 5, "tetra")])
def test_all_corner_permutations_give_same_local_operators(elem, p, kind):
    perms = torch.tensor(list(permutations(range(elem.n_vertex))))
    cells = perms + torch.arange(len(perms))[:, None]*elem.n_vertex
    points = elem.points.double().repeat(len(perms), 1)
    mesh = Mesh(meshio.Mesh(points.numpy(), [(kind, cells.numpy())]), reorder=False).double()
    a = MassElementAssembler.from_mesh(mesh, basis="hierarchical", field_order=p, quadrature_order=2*p)
    b = NodeAssembler.from_assembler(a)
    block = a._hierarchical.blocks[kind]
    assert block.ref_val.ndim == 2 and block.ref_grad.ndim == 3
    ids = block.conn
    def form(u, v, gradu, gradv, x):
        return (1+x.sum())*u*v + gradu @ gradv
    K = a(func=form).to_dense()[ids[:, :, None], ids[:, None, :]]
    f = b(func=lambda v, x: (1+x.sum())*v, batch_size=-1)[ids]
    torch.testing.assert_close(K, K[:1].expand_as(K))
    torch.testing.assert_close(f, f[:1].expand_as(f))


@pytest.mark.parametrize("patch,elem", [(tri_patch, Triangle), (tet_patch, Tetrahedron)])
def test_curved_sampled_forms_integrate_at_original_physical_points(patch, elem):
    """Compare weighted matrix and load actions to direct original-cell integration."""
    mesh = patch(2, curved=True)
    a = MassElementAssembler.from_mesh(mesh, basis="hierarchical", field_order=4, quadrature_order=8)
    b = NodeAssembler.from_assembler(a)
    layout = b.field_layout()
    trans = next(iter(a.transformation.values()))
    w, q = trans.quadrature
    geom = elem.get_basis_fns(2, mesh.dtype)
    coords = mesh.points[mesh.elements()]
    x = geom.map(q) @ coords
    jac = torch.einsum("qrb,ebd->eqrd", geom.grad().map(q), coords)
    jw = jac.det().abs() * w
    data = torch.arange(jw.numel(), dtype=w.dtype).reshape_as(jw).sin().requires_grad_()
    kappa = mesh.points[:, -1]**2
    kap = torch.einsum("qb,eb->eq", geom.map(q), kappa[mesh.elements()])
    dk = torch.linalg.solve(jac, torch.einsum("qrb,eb->eqr", geom.grad().map(q),
                                              kappa[mesh.elements()]).unsqueeze(-1)).squeeze(-1)
    # Both test functions are represented exactly on quadratic geometry.
    u = layout.interpolate("u", lambda x: 1+x[:, 0]**2)
    v = layout.interpolate("u", lambda x: 2-x[:, -1])
    expected_f = ((2-x[..., -1])*(data*x[..., 0]+kap+dk[..., 0])*jw).sum()
    tag = torch.tensor([.3, 1.1], dtype=w.dtype)
    expected_k = ((1+x[..., 0]**2)*(data+tag[:, None])*jw).sum() * -1
    for batch in (5, -1):
        f = b(func=lambda v, x, h, k, gradk: v*(h*x[0]+k+gradk[0]),
              point_data={"k": kappa}, quadrature_data={"h": data}, batch_size=batch)
        K = a(func=lambda u, gradv, h, tag: u*gradv[-1]*(h+tag),
              element_data={"tag": tag}, quadrature_data={"h": data}, batch_size=batch)
        actual = u @ K.to_dense() @ v
        torch.testing.assert_close(v @ f, expected_f)
        torch.testing.assert_close(actual, expected_k)
        torch.testing.assert_close(torch.autograd.grad(actual, data)[0], -(1+x[..., 0]**2)*jw)


@pytest.mark.parametrize("patch,elem", [(tri_patch, Triangle), (tet_patch, Tetrahedron)])
def test_reference_query_gradients_prolongation_and_boundary(patch, elem):
    mesh = patch(2, curved=True)
    a = MassElementAssembler.from_mesh(mesh, basis="hierarchical", field_order=4)
    layout = a.field_layout()
    def fn(x):
        return 1+x[..., 0]**2+x[..., 1]
    c = layout.interpolate("u", fn)
    ref = torch.full((2, mesh.points.shape[1]), .16, dtype=mesh.dtype, requires_grad=True)
    coords = mesh.points[mesh.elements()]
    x = elem.get_basis_fns(2, mesh.dtype).map(ref) @ coords
    z = layout.evaluate("u", c, ref)
    torch.testing.assert_close(z, fn(x))
    torch.testing.assert_close(torch.autograd.grad(z.sum(), ref)[0],
                               torch.autograd.grad(fn(x).sum(), ref)[0])
    x = x.detach().requires_grad_()
    value = layout.evaluate("u", c, x, physical=True)
    phys_grad = torch.autograd.grad(value.sum(), x)[0]
    torch.testing.assert_close(phys_grad, torch.autograd.grad(fn(x).sum(), x)[0])
    torch.testing.assert_close(layout.evaluate("u", c, ref, gradient=True), phys_grad)
    torch.testing.assert_close(layout.prolong("u", c), fn(mesh.points))
    torch.testing.assert_close(layout.prolong("u", layout.restrict("u", mesh.points)), mesh.points)
    mask = layout.boundary_mask("u")
    # Shared edge/face interior modes plus both sets of private cell bubbles.
    assert (~mask).sum() == (9 if elem is Triangle else 5)
    torch.testing.assert_close(layout.dirichlet_values("u", fn), c*mask)
    # A random field, not an interpolated smooth function, must have a single trace.
    shared = torch.tensor([0., .27, .73, 0.] if elem is Triangle else [0., .19, .34, .47, 0.],
                          dtype=mesh.dtype)
    bary = shared[mesh.elements()[:, :elem.n_vertex]]
    coefs = torch.randn(c.shape, dtype=c.dtype, generator=torch.Generator().manual_seed(9))
    trace = layout.evaluate("u", coefs, bary[:, None, 1:])
    torch.testing.assert_close(trace[0], trace[1])


@pytest.mark.parametrize("patch", [tri_patch, tet_patch])
def test_precanonical_modal_checkpoints_are_rejected_even_with_non_strict_load(patch):
    a = MassElementAssembler.from_mesh(patch(), basis="hierarchical", field_order=4)
    state = a.state_dict()
    del state["_hierarchical._extra_state"]["frame"]
    with pytest.raises(RuntimeError, match="pre-canonical"):
        a.load_state_dict(state, strict=False)


def test_cell_chunking_restores_original_rows_for_vector_loads(monkeypatch):
    mesh = orientation_mesh([6, 2, 6, 2, 6], order=2, curved=True)
    a = MassElementAssembler.from_mesh(mesh, basis="hierarchical", field_order=4, quadrature_order=8)
    b = NodeAssembler.from_assembler(a)
    trans = a.transformation["tetra10"]
    history = torch.arange(5 * trans.n_quadrature, dtype=mesh.dtype).reshape(5, -1).cos()
    def outputs():
        return (b(func=lambda v, x: v*x, batch_size=9),
                b(func=lambda v, x, h: h*v*x, quadrature_data={"h": history}, batch_size=9))
    expected = outputs()
    monkeypatch.setattr(_single_field, "_MODAL_BATCH_ENTRIES", 1)
    actual = outputs()
    for f, reference in zip(actual, expected):
        torch.testing.assert_close(f, reference)
        assert f.numel() == a.field_layout(components=3).n_dofs
