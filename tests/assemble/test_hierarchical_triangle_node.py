"""Triangle load vectors and shared single-field interpolation/BC helpers."""

import copy

import pytest
import torch

from tensormesh import Condenser, Field, MixedElementAssembler, NodeAssembler
from tensormesh.assemble import LaplaceElementAssembler, MassElementAssembler
from tests.assemble._triangle_fixtures import patch_mesh, poisson, change_of_basis


class Load(NodeAssembler):
    def forward(self, v, x):
        return (1+x[0]+x[1]**2)*v


def make_node(mesh, p=3, **kwargs):
    return Load.from_mesh(mesh, quadrature_order=2*p, basis="hierarchical",
                          field_order=p, **kwargs)


@pytest.mark.parametrize("p", [1, 4])
def test_triangle_node_load_and_signed_nodal_change_of_basis(p):
    mesh = patch_mesh(2, curved=True)
    modal = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=2*p)
    nodal = poisson(p, "lagrange").from_mesh(mesh, quadrature_order=2*p)
    transform = change_of_basis(nodal, modal)
    asm = make_node(mesh, p)
    expected = modal.assemble_vector()
    torch.testing.assert_close(asm(), expected)
    torch.testing.assert_close(asm(batch_size=-1), expected)
    torch.testing.assert_close(expected, transform.T @ nodal.assemble_vector())


@pytest.mark.parametrize("project", ["reduce", "sparse"])
def test_triangle_node_signed_construction_and_assembler_reuse(project):
    mesh = patch_mesh(2, curved=True)
    p = 3
    expected = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=2*p).assemble_vector()
    torch.testing.assert_close(make_node(mesh, p, project=project)(), expected)
    matrix = MassElementAssembler.from_mesh(
        mesh, quadrature_order=2*p, basis="hierarchical", field_order=p, project=project)
    reused = Load.from_assembler(matrix)
    assert reused._hierarchical is matrix._hierarchical
    assert reused.transformation is matrix.transformation
    torch.testing.assert_close(reused(batch_size=3), expected)
    assert reused.n_points == len(mesh.points)
    torch.testing.assert_close(Load.from_assembler(reused)(), expected)
    raw = Load.from_elements(mesh.points, {"triangle6": mesh.elements()},
                             quadrature_order=2*p, basis="hierarchical", field_order=p,
                             dtype=mesh.dtype, project=project)
    torch.testing.assert_close(raw(), expected)


def test_triangle_node_vector_load_and_layout_component_order():
    class Vector(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=2, basis="hierarchical")]

        def forward_vector(self, v, x):
            return ((1+x)*v).sum()

    mesh = patch_mesh(2, curved=True)
    asm = make_node(mesh)
    actual = asm(func=lambda v, x: v*(1+x), batch_size=3)
    expected = Vector.from_mesh(mesh, quadrature_order=6).assemble_vector()
    torch.testing.assert_close(actual, expected)
    layout = asm.field_layout(components=2)
    assert actual.shape == (layout.n_dofs,)
    torch.testing.assert_close(layout.cat(layout.split(actual)), actual)


def test_triangle_node_coefficients_autograd_and_quadrature_batches():
    class Weighted(NodeAssembler):
        def forward(self, v, gradv, kappa, gradkappa, alpha):
            return alpha*(kappa*v + (gradkappa*gradv).sum())

    mesh = patch_mesh(2, curved=True)
    asm = Weighted.from_mesh(mesh, basis="hierarchical", field_order=3, quadrature_order=6)
    mixed = poisson(3, "hierarchical").from_mesh(mesh, quadrature_order=6)
    points = mesh.points.detach().clone().requires_grad_()
    kappa = torch.linspace(.7, 1.3, len(points), dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    for _ in range(2):
        kwargs = dict(point_data={"kappa": kappa}, scalar_data={"alpha": alpha})
        actual = asm(points, **kwargs)
        grad = torch.autograd.grad(actual.square().sum(), (points, kappa, alpha))
        expected = mixed.assemble_vector(points, point_data={"kappa": kappa},
            func=lambda v, gradv, kappa, gradkappa:
            alpha*(kappa*v + (gradkappa*gradv).sum()))
        torch.testing.assert_close(actual, expected)
        target = torch.autograd.grad(expected.square().sum(), (points, kappa, alpha))
        for a, b in zip(grad, target):
            torch.testing.assert_close(a, b)
    # The optimized nodal broadcast path must not be used for modal fields.
    torch.testing.assert_close(asm.flat_mode()(points, batch_size=3, **kwargs), actual)
    trans = asm.transformation["triangle6"]
    physical = torch.einsum("qb,ebd->eqd", trans.shape_val, points[mesh.elements()])
    history = 1+physical[..., 0]
    actual = asm(points, func=lambda v, history: history*v,
                 quadrature_data={"history": history}, batch_size=3)
    expected = mixed.assemble_vector(points, func=lambda v, x: (1+x[0])*v)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("components", [1, 2])
def test_triangle_single_field_helpers_match_mixed_and_follow_buffers(components):
    class Reference(MixedElementAssembler):
        fields = [Field("u", "v", 4, components=components, basis="hierarchical")]

    mesh = patch_mesh(2, curved=True)
    matrix = MassElementAssembler.from_mesh(
        mesh, basis="hierarchical", field_order=4, quadrature_order=8)
    node = Load.from_assembler(matrix)
    reference = Reference.from_mesh(mesh, quadrature_order=8)
    layouts = [matrix.field_layout(components), node.field_layout(components)]

    def fn(x):
        a, b = 1+x[:, 0]**2+x[:, 1], 2+x[:, 0]*x[:, 1]
        return a if components == 1 else torch.stack((a, b), -1)

    ref = torch.tensor([[.1, .2], [.5, .3]], dtype=torch.float64)
    expected = reference.layout.interpolate("u", fn)
    for layout in layouts:
        coefficients = layout.interpolate("u", fn)
        torch.testing.assert_close(coefficients, expected)
        for physical, gradient in ((False, False), (False, True), (True, False)):
            queries = ref
            if physical:
                from tensormesh.element import Triangle
                table = Triangle.get_basis_fns(2, torch.float64).map(ref)
                queries = torch.einsum("qb,ebd->eqd", table, mesh.points[mesh.elements()])
            actual = layout.evaluate("u", coefficients, queries, physical=physical, gradient=gradient)
            target = reference.layout.evaluate("u", expected, queries, physical=physical, gradient=gradient)
            torch.testing.assert_close(actual, target)
        nodal = fn(mesh.points)
        torch.testing.assert_close(layout.prolong("u", layout.restrict("u", nodal)), nodal)
        mask = layout.boundary_mask("u")
        torch.testing.assert_close(mask, reference.layout.boundary_mask("u"))
        torch.testing.assert_close(layout.dirichlet_values("u", fn), reference.layout.dirichlet_values("u", fn))
        if components == 2:
            torch.testing.assert_close(layout.dirichlet_values("u", lambda x: fn(x)[:, 1], component=1),
                reference.layout.dirichlet_values("u", lambda x: fn(x)[:, 1], component=1))
    # Keep a layout across conversions: it must not hold stale buffer copies.
    matrix.float()
    assert layouts[0].interpolate("u", fn).dtype == torch.float32
    assert layouts[1].interpolate("u", fn).dtype == torch.float32
    torch.testing.assert_close(layouts[0].boundary_mask("u"), mask)


def test_triangle_element_node_poisson_with_nonzero_modal_dirichlet_data():
    class Forcing(NodeAssembler):
        def forward(self, v, x):
            return 2*(x[0]*(1-x[0])+x[1]*(1-x[1]))*v

    mesh = patch_mesh(2)
    asm = LaplaceElementAssembler.from_mesh(
        mesh, basis="hierarchical", field_order=4, quadrature_order=8)
    rhs = Forcing.from_assembler(asm)
    layout = asm.field_layout()
    exact = lambda x: 1+x[:, 0]+x[:, 0]*(1-x[:, 0])*x[:, 1]*(1-x[:, 1])
    condenser = Condenser(layout.boundary_mask("u"), layout.dirichlet_values("u", exact))
    matrix, load = condenser(asm(), rhs())
    coefficients = condenser.recover(torch.linalg.solve(matrix.to_dense(), load))
    torch.testing.assert_close(coefficients, layout.interpolate("u", exact), atol=2e-12, rtol=2e-11)
    torch.testing.assert_close(layout.prolong("u", coefficients), exact(mesh.points))


def test_triangle_node_checkpoint_geometry_changes_and_nodal_defaults():
    mesh = patch_mesh()
    asm = make_node(mesh)
    layout = asm.field_layout()
    expected = asm(batch_size=-1)
    clone = make_node(mesh)
    clone.load_state_dict(copy.deepcopy(asm.state_dict()))
    torch.testing.assert_close(clone(batch_size=-1), expected)
    moved = mesh.points*1.2
    asm(moved)
    coefficients = layout.interpolate("u", lambda x: x[:, 0])
    torch.testing.assert_close(layout.prolong("u", coefficients), moved[:, 0])
    inferred = Load.from_mesh(patch_mesh(2), basis="hierarchical")
    assert inferred._hierarchical.order == 2
    assert inferred.transformation["triangle6"].quadrature_order == 2
    default = Load.from_mesh(mesh)
    explicit = Load.from_mesh(mesh, basis="lagrange", field_order=1)
    torch.testing.assert_close(default(), explicit())
    assert default.state_dict().keys() == explicit.state_dict().keys()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triangle_node_and_single_field_helpers_cuda():
    asm = make_node(patch_mesh(2, curved=True))
    expected = asm()
    layout = asm.field_layout()
    asm.cuda()
    torch.testing.assert_close(asm().cpu(), expected)
    coefficients = layout.interpolate("u", lambda x: x[:, 0])
    assert coefficients.device.type == "cuda"
    assert layout.boundary_mask("u").device.type == "cuda"
    torch.testing.assert_close(layout.prolong("u", coefficients), asm.transformation["triangle6"].points[:, 0])
