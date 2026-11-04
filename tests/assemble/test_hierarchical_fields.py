"""Boundary trace lifting, interpolation, output and physical evaluation."""

import meshio
import pytest
import torch

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.element import Triangle
from tests.assemble._triangle_fixtures import patch_mesh, poisson, physical_points


def sample_reference():
    return torch.tensor([[0.1, 0.2], [0.6, 0.1], [0.2, 0.6], [0., 0.], [0., 0.4]], dtype=torch.float64)


def test_layout_boundary_mask_and_dtype_movement():
    p = 4
    asm = poisson(p, "hierarchical").from_mesh(patch_mesh(2), quadrature_order=7)
    mask = asm.layout.boundary_mask("u")
    assert mask.sum() == 4 * p
    # Cached topology must follow later dtype/device movement.
    asm.float()
    torch.testing.assert_close(asm.layout.boundary_mask("u"), mask)


def test_p1_lifting_solves_with_interior_vertex():
    mesh = Mesh(meshio.Mesh(
        [[0., 0.], [1., 0.], [0., 1.], [1., 1.], [0.5, 0.5]],
        [("triangle", [[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])])).double()
    asm = poisson(1, "hierarchical").from_mesh(mesh)
    lay = asm.layout
    mask = lay.boundary_mask("u")
    exact = lambda x: 2 + x[:, 0] - x[:, 1]
    lifting = lay.dirichlet_values("u", exact)
    assert lifting[4] == 0
    condenser = Condenser(mask, lifting)
    k, b = condenser(asm(), torch.zeros(asm.n_dofs, dtype=torch.float64))
    coefficients = condenser.recover(torch.linalg.solve(k.to_dense(), b))
    torch.testing.assert_close(lay.prolong("u", coefficients), exact(mesh.points), atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("family,p", [("lagrange", 1), ("lagrange", 3),
                                     ("hierarchical", 1), ("hierarchical", 3),
                                     ("hierarchical", 6), ("hierarchical", 10)])
def test_constant_and_polynomial_interpolation(family, p):
    asm = poisson(p, family).from_mesh(patch_mesh(), quadrature_order=7)
    lay = asm.layout
    constant = lay.interpolate("u", lambda x: 2.5)
    if family == "hierarchical":
        torch.testing.assert_close(constant[:4], torch.full((4,), 2.5, dtype=torch.float64), atol=2e-12, rtol=0)
        torch.testing.assert_close(constant[4:], torch.zeros_like(constant[4:]), atol=2e-10, rtol=0)
    fn = lambda x: 1 + x[:, 0] ** p + x[:, 1]
    coefficients = lay.interpolate("u", fn)
    ref = sample_reference()
    physical = physical_points(asm, ref)
    expected = fn(physical.reshape(-1, 2)).reshape(physical.shape[:2])
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref), expected, atol=5e-11, rtol=2e-11)
    torch.testing.assert_close(lay.prolong("u", coefficients), fn(asm._points), atol=5e-11, rtol=2e-11)


@pytest.mark.parametrize("geometry", [1, 2, 3])
def test_restrict_and_prolong_are_modal_aware(geometry):
    class Vector(MixedElementAssembler):
        fields = [Field("u", "v", geometry, components=2, basis="hierarchical")]
    asm = Vector.from_mesh(patch_mesh(geometry), quadrature_order=7)
    fn = lambda x: torch.stack((1 + x[:, 0], x[:, 1] ** geometry), -1)
    data = fn(asm._points)
    coefficients = asm.layout.restrict("u", data)
    torch.testing.assert_close(asm.layout.prolong("u", coefficients), data, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(coefficients, asm.layout.interpolate("u", fn), atol=2e-12, rtol=2e-12)
    if geometry >= 2:
        assert not torch.allclose(coefficients, data)


def test_restrict_interpolates_mesh_data_not_unsampled_analytic_function():
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh())
    lay = asm.layout
    # x**2 and x have identical samples at this P1 mesh's vertices.
    restricted = lay.restrict("u", asm._points[:, 0] ** 2)
    linear = lay.interpolate("u", lambda x: x[:, 0])
    quadratic = lay.interpolate("u", lambda x: x[:, 0] ** 2)
    torch.testing.assert_close(restricted, linear, atol=1e-13, rtol=1e-12)
    assert not torch.allclose(restricted, quadratic)


@pytest.mark.parametrize("geometry", [1, 2, 3])
@pytest.mark.parametrize("family", ["lagrange", "hierarchical"])
def test_reference_and_physical_evaluation_with_gradients(geometry, family):
    asm = poisson(4, family).from_mesh(patch_mesh(geometry, curved=geometry > 1), quadrature_order=7)
    lay = asm.layout
    fn = lambda x: x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 2 * x[:, 1] ** 2
    coefficients = lay.interpolate("u", fn)
    ref = sample_reference()
    physical = physical_points(asm, ref)
    expected = fn(physical.reshape(-1, 2)).reshape(physical.shape[:2])
    gradient = torch.stack((2 * physical[..., 0] + physical[..., 1],
                            physical[..., 0] + 4 * physical[..., 1]), -1)
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref), expected, atol=3e-11, rtol=2e-11)
    torch.testing.assert_close(lay.evaluate("u", coefficients, physical, physical=True), expected, atol=3e-11, rtol=2e-11)
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref, gradient=True), gradient, atol=5e-11, rtol=2e-11)
    torch.testing.assert_close(lay.evaluate("u", coefficients, physical, physical=True, gradient=True),
                               gradient, atol=5e-11, rtol=2e-11)
    # Explicit/reordered cell selection and common physical points for one cell.
    selected = lay.evaluate("u", coefficients, physical.flip(0), cell_indices=[1, 0], physical=True)
    torch.testing.assert_close(selected, expected.flip(0), atol=3e-11, rtol=2e-11)
    selected = lay.evaluate("u", coefficients, physical[1], cell_indices=1, physical=True)
    torch.testing.assert_close(selected, expected[1:2], atol=3e-11, rtol=2e-11)


@pytest.mark.parametrize("homogeneous", [True, False])
@pytest.mark.parametrize("family", ["hierarchical", "lagrange"])
def test_manufactured_dirichlet_poisson(homogeneous, family):
    p = 4 if homogeneous else 3
    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", p, basis=family)]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            forcing = 2 * (x[0] * (1 - x[0]) + x[1] * (1 - x[1])) if homogeneous else x[0] * 0
            return forcing * v

    asm = Poisson.from_mesh(patch_mesh(3), quadrature_order=7)
    lay = asm.layout
    if homogeneous:
        exact = lambda x: x[:, 0] * (1 - x[:, 0]) * x[:, 1] * (1 - x[:, 1])
    else:
        exact = lambda x: 1 + 2 * x[:, 0] + x[:, 1] + x[:, 0] ** 3 - 3 * x[:, 0] * x[:, 1] ** 2
    mask = lay.boundary_mask("u")
    lifting = lay.dirichlet_values("u", exact)
    assert (lifting[~mask] == 0).all()
    if homogeneous:
        torch.testing.assert_close(lifting, torch.zeros_like(lifting), atol=2e-13, rtol=0)
    elif family == "hierarchical":
        assert lifting[4:].abs().max() > 0.1  # actual nonzero trace modes
    condenser = Condenser(mask, lifting)
    matrix, rhs = condenser(asm(), asm.assemble_vector())
    solution = condenser.recover(torch.linalg.solve(matrix.to_dense(), rhs))
    ref = sample_reference()
    physical = physical_points(asm, ref)
    expected = exact(physical.reshape(-1, 2)).reshape(physical.shape[:2])
    torch.testing.assert_close(lay.evaluate("u", solution, ref), expected, atol=2e-11, rtol=2e-11)
    torch.testing.assert_close(lay.prolong("u", solution), exact(asm._points), atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("p,tol", [(3, 2e-3), (6, 2e-7), (10, 2e-12)])
def test_nonpolynomial_boundary_trace_accuracy_and_boundary_only_sampling(p, tol):
    asm = poisson(p, "hierarchical").from_mesh(patch_mesh(), quadrature_order=7)
    def trace(x):
        on_boundary = ((x[:, 0].abs() < 1e-12) | ((x[:, 0] - 1).abs() < 1e-12)
                       | (x[:, 1].abs() < 1e-12) | ((x[:, 1] - 1).abs() < 1e-12))
        assert on_boundary.all()
        return torch.exp(x[:, 0] + 0.2 * x[:, 1])
    lifting = asm.layout.dirichlet_values("u", trace)
    assert (lifting[~asm.layout.boundary_mask("u")] == 0).all()
    t = torch.linspace(0, 1, 47, dtype=torch.float64)
    verts = Triangle.points.to(torch.float64)
    # Local boundary edges 1 and 2; includes opposite canonical directions.
    for edge in [1, 2]:
        a, b = Triangle.edge[edge]
        ref = (1 - t[:, None]) * verts[a] + t[:, None] * verts[b]
        physical = physical_points(asm, ref)
        exact = torch.exp(physical[..., 0] + 0.2 * physical[..., 1])
        assert (asm.layout.evaluate("u", lifting, ref) - exact).abs().max() < tol


@pytest.mark.parametrize("family", ["lagrange", "hierarchical"])
def test_curved_boundary_lifting_and_vector_components(family):
    class TwoFields(MixedElementAssembler):
        fields = [Field("u", "v", 4, components=2, basis=family), Field("p", "q", 1)]
    asm = TwoFields.from_mesh(patch_mesh(2, curved=True), quadrature_order=7)
    lay = asm.layout
    fn = lambda x: torch.stack((x[:, 0] ** 2 + x[:, 1], 2 + x[:, 0] * x[:, 1]), -1)
    full = lay.dirichlet_values("u", fn)
    mask = lay.boundary_mask("u")
    first = lay.dirichlet_values("u", lambda x: fn(x)[:, 0], component=0)
    second = lay.dirichlet_values("u", lambda x: fn(x)[:, 1], component=1)
    torch.testing.assert_close(full, first + second, atol=1e-13, rtol=1e-12)
    assert (full[~mask] == 0).all()
    assert not (lay.boundary_mask("u", 0) & lay.boundary_mask("u", 1)).any()
    assert (lay.split(full)["p"] == 0).all()
    t = torch.linspace(0, 1, 23, dtype=torch.float64)
    for edge in [1, 2]:
        a, b = Triangle.edge[edge]
        verts = Triangle.points.double()
        ref = (1 - t[:, None]) * verts[a] + t[:, None] * verts[b]
        physical = physical_points(asm, ref)
        expected = fn(physical.reshape(-1, 2)).reshape(2, len(t), 2)
        torch.testing.assert_close(lay.evaluate("u", lay.split(full)["u"], ref), expected, atol=1e-12, rtol=2e-12)


def test_evaluation_and_interpolation_autograd():
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh())
    amplitude = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    coefficients = asm.layout.interpolate("u", lambda x: amplitude * (x[:, 0] ** 2 + x[:, 1]))
    ref = sample_reference()
    values = asm.layout.evaluate("u", coefficients, ref)
    physical = physical_points(asm, ref)
    expected = physical[..., 0] ** 2 + physical[..., 1]
    derivative = torch.autograd.grad(values.sum(), amplitude)[0]
    torch.testing.assert_close(derivative, expected.sum(), atol=1e-12, rtol=2e-12)
    trace = asm.layout.dirichlet_values("u", lambda x: amplitude * x[:, 0])
    assert torch.autograd.grad(trace.square().sum(), amplitude)[0] > 0


def test_curved_physical_query_autograd_at_converged_affine_seed():
    ref = Triangle.get_basis(2, torch.float64, torch.device("cpu"))
    r, s = ref.unbind(-1)
    points = torch.stack((r, s + 0.1 * (r * (1 - r) - s * (1 - s))), -1)
    mesh = Mesh(meshio.Mesh(points.numpy(), [("triangle6", [list(range(6))])])).double()
    asm = poisson(3, "hierarchical").from_mesh(mesh)
    lay = asm.layout
    coefficients = lay.interpolate("u", lambda x: x[:, 1])
    query = torch.tensor([[0.2, 0.2]], dtype=torch.float64, requires_grad=True)
    values = lay.evaluate("u", coefficients, query, physical=True)
    explicit = lay.evaluate("u", coefficients, query, physical=True, gradient=True)
    derivative = torch.autograd.grad(values.sum(), query)[0]
    torch.testing.assert_close(values, query[:, 1][None], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(explicit[0], torch.tensor([[0., 1.]], dtype=torch.float64), atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(derivative, explicit[0], atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("family", ["lagrange", "hierarchical"])
def test_complex_callable_interpolation_and_dirichlet_lifting(family):
    class TwoFields(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=2, basis=family), Field("p", "q", 1)]
    asm = TwoFields.from_mesh(patch_mesh(), quadrature_order=7)
    lay = asm.layout
    amplitude = torch.tensor(1 + 2j, dtype=torch.complex128, requires_grad=True)
    fn = lambda x: amplitude * torch.stack((1 + x[:, 0] ** 2, 2 - x[:, 1]), -1)
    coefficients = lay.interpolate("u", fn)
    assert coefficients.dtype == torch.complex128
    ref = sample_reference()
    physical = physical_points(asm, ref)
    expected = fn(physical.reshape(-1, 2)).reshape(*physical.shape[:2], 2)
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref), expected, atol=1e-12, rtol=1e-12)

    def trace(x):
        on_boundary = ((x[:, 0].abs() < 1e-12) | ((x[:, 0] - 1).abs() < 1e-12)
                       | (x[:, 1].abs() < 1e-12) | ((x[:, 1] - 1).abs() < 1e-12))
        assert on_boundary.all()
        return fn(x)

    lifting = lay.dirichlet_values("u", trace)
    assert lifting.dtype == torch.complex128
    mask = lay.boundary_mask("u")
    expected = lay.cat(u=coefficients, p=0.)
    torch.testing.assert_close(lifting[mask], expected[mask], atol=1e-12, rtol=1e-12)
    assert (lifting[~mask] == 0).all()
    first = lay.dirichlet_values("u", lambda x: trace(x)[:, 0], component=0)
    second = lay.dirichlet_values("u", lambda x: trace(x)[:, 1], component=1)
    torch.testing.assert_close(lifting, first + second, atol=1e-12, rtol=1e-12)
    derivative = torch.autograd.grad(lifting.imag.sum(), amplitude)[0]
    torch.testing.assert_close(derivative, 1j * (lifting.imag.sum() / amplitude.imag), atol=1e-12, rtol=1e-12)


def test_float32_layout_and_empty_queries():
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh()).float()
    coefficients = asm.layout.interpolate("u", lambda x: x[:, 0] ** 2)
    ref = sample_reference().float()
    assert asm.layout.evaluate("u", coefficients, ref).dtype == torch.float32
    assert asm.layout.evaluate("u", coefficients, ref[:0]).shape == (2, 0)
    assert asm.layout.evaluate("u", coefficients, ref[:0], gradient=True).shape == (2, 0, 2)
    assert asm.layout.evaluate("u", coefficients, ref, cell_indices=torch.empty(0, dtype=torch.long)).shape == (0, len(ref))


def test_float32_curved_physical_evaluation():
    asm = poisson(4, "hierarchical").from_mesh(patch_mesh(2, curved=True), quadrature_order=7).float()
    fn = lambda x: x[:, 0] ** 2 + 2 * x[:, 1]
    coefficients = asm.layout.interpolate("u", fn)
    ref = sample_reference().float()
    kind = asm.element_types[0]
    table = Triangle.get_basis_fns(2, torch.float32).map(ref)
    physical = torch.einsum("qb,ebd->eqd", table, asm._points[asm.elements[kind]])
    actual = asm.layout.evaluate("u", coefficients, physical, physical=True)
    expected = fn(physical.reshape(-1, 2)).reshape(physical.shape[:2])
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex128])
def test_coefficient_dtype_preserved_independently_of_geometry(dtype):
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh(3))
    data = (asm._points[:, 0] ** 2).to(dtype)
    if dtype.is_complex:
        data = data * (1 + 2j)
    coefficients = asm.layout.restrict("u", data)
    assert coefficients.dtype == dtype
    result = asm.layout.prolong("u", coefficients)
    assert result.dtype == dtype
    torch.testing.assert_close(result, data, atol=2e-6 if dtype == torch.float32 else 2e-12, rtol=2e-6)
    assert asm.layout.evaluate("u", coefficients, sample_reference()).dtype == dtype
    assert asm.layout.evaluate("u", coefficients, sample_reference(), gradient=True).dtype == dtype


def test_shared_edge_evaluation_of_arbitrary_coefficients():
    asm = poisson(6, "hierarchical").from_mesh(patch_mesh(2, curved=True), quadrature_order=7)
    coefficients = torch.randn(asm.n_dofs, dtype=torch.float64, generator=torch.Generator().manual_seed(27))
    t = torch.linspace(0, 1, 31, dtype=torch.float64)
    ref = torch.stack((t, 1 - t), -1)
    # Both cells share local edge 0, with opposite endpoint traversal.
    reference = torch.stack((ref, ref.flip(0)))
    values = asm.layout.evaluate("u", coefficients, reference)
    torch.testing.assert_close(values[0], values[1], atol=2e-13, rtol=2e-12)
    physical = physical_points(asm, ref)
    physical[1] = physical[1].flip(0)
    via_physical = asm.layout.evaluate("u", coefficients, physical, physical=True)
    torch.testing.assert_close(via_physical, values, atol=1e-11, rtol=2e-11)


def test_invalid_queries():
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh(2, curved=True))
    lay = asm.layout
    coefficients = lay.interpolate("u", lambda x: x[:, 0])
    with pytest.raises(ValueError, match="outside"):
        lay.evaluate("u", coefficients, torch.tensor([[2., 2.]]), cell_indices=0, physical=True)
    with pytest.raises(ValueError, match="did not converge"):
        lay.evaluate("u", coefficients, torch.tensor([[0.2, 0.2]]), cell_indices=0, physical=True,
                     max_iterations=1, tolerance=1e-30)
    with pytest.raises(ValueError, match="out of range"):
        lay.evaluate("u", coefficients, sample_reference(), cell_indices=[8])
    with pytest.raises(ValueError, match="shape"):
        lay.evaluate("u", coefficients, torch.zeros(2))
    with pytest.raises(ValueError, match="leading dimension"):
        lay.evaluate("u", coefficients[:2], sample_reference())
    with pytest.raises(ValueError, match="component"):
        lay.dirichlet_values("u", lambda x: 0, component=1)
    with pytest.raises(ValueError, match="finite and positive"):
        lay.evaluate("u", coefficients, sample_reference(), cell_indices=0,
                     physical=True, tolerance=float("nan"))
    with pytest.raises(NotImplementedError, match="modal coefficients"):
        lay.points("u")
    with pytest.raises(NotImplementedError, match="modal coefficients"):
        lay.dof_mask("u", where=lambda x: x[:, 0] == 0)
