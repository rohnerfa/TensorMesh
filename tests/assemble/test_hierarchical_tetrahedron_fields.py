"""Analytical field, trace and manufactured-Poisson checks for modal tetrahedra."""

from itertools import permutations, product
import math

import meshio
import pytest
import torch

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.assemble.topology import lagrange_dofmap
from tensormesh.element import Tetrahedron, Triangle


def _warp(x):
    a, b, c = x.unbind(-1)
    return torch.stack((a + 0.1 * b * c, b, c + 0.08 * b * (1 - b)), -1)


def _unwarp(x):
    a, b, c = x.unbind(-1)
    z = c - 0.08 * b * (1 - b)
    return torch.stack((a - 0.1 * b * z, b, z), -1)


def _mesh(vertices, cells, geometry=1, curved=False):
    vertices = torch.as_tensor(vertices, dtype=torch.float64)
    cells = torch.as_tensor(cells, dtype=torch.long)
    if geometry == 1:
        points, conn, kind = vertices, cells, "tetra"
    else:
        dofs = lagrange_dofmap({"tetra": cells}, len(vertices), 2)
        conn, ref = dofs.conn["tetra"], dofs.ref_nodes["tetra"]
        v = vertices[cells]
        local = v[:, :1] + torch.einsum("qd,edk->eqk", ref, v[:, 1:] - v[:, :1])
        points = torch.empty(dofs.n_dofs, 3, dtype=torch.float64)
        points[conn] = local
        kind = "tetra10"
    if curved:
        points = _warp(points)
    return Mesh(meshio.Mesh(points.numpy(), [(kind, conn.numpy())])).double()


def _patch(geometry=1, curved=False):
    return _mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]],
                 [[3, 0, 2, 1], [2, 4, 1, 0]], geometry, curved)


def _star(geometry=1, curved=False):
    # Four connected cells with an interior vertex, edges and faces.
    return _mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.25, .25, .25]],
                 [[4, 1, 3, 2], [2, 4, 3, 0], [1, 3, 0, 4], [0, 2, 4, 1]],
                 geometry, curved)


def _space(p, components=1, other=False):
    class Space(MixedElementAssembler):
        fields = [Field("u", "v", p, components=components, basis="hierarchical")]
        if other:
            fields = [Field("pressure", "q", 1)] + fields

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

    return Space


def _reference(dtype=torch.float64, device="cpu"):
    return torch.tensor([[.11, .17, .23], [.53, .13, .07], [.04, .52, .21],
                         [0, 0, 0], [1, 0, 0], [0, .31, .22]], dtype=dtype, device=device)


def _physical(asm, ref, cells=None):
    kind = asm.element_types[0]
    conn = asm.elements[kind] if cells is None else asm.elements[kind][cells]
    basis = Tetrahedron.get_basis_fns(asm.mesh_order, ref.dtype, ref.device)
    if ref.ndim == 2:
        return torch.einsum("qb,ebd->eqd", basis.map(ref), asm._points[conn])
    table = basis.map(ref.reshape(-1, 3)).reshape(*ref.shape[:2], conn.shape[1])
    return torch.einsum("eqb,ebd->eqd", table, asm._points[conn])


def _polynomial(x, p):
    return 1 + .4 * x[..., 0] ** p + .3 * x[..., 1] ** p - .2 * x[..., 2] ** p + .1 * x.sum(-1) ** p


def _polynomial_gradient(x, p):
    weights = x.new_tensor([.4, .3, -.2])
    return p * (weights * x ** (p - 1) + .1 * x.sum(-1, keepdim=True) ** (p - 1))


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 8, 10])
def test_tetrahedral_entity_interpolation_reproduces_polynomial_and_gradient(p):
    asm = _space(p).from_mesh(_patch(), quadrature_order=3)
    lay = asm.layout
    coefficients = lay.interpolate("u", lambda x: _polynomial(x, p))
    ref = _reference()
    physical = _physical(asm, ref)
    # At p=10 the residual interpolation systems are more ill-conditioned;
    # these absolute tolerances still resolve 8+ accurate digits in derivatives.
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref),
                               _polynomial(physical, p), atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(lay.evaluate("u", coefficients, ref, gradient=True),
                               _polynomial_gradient(physical, p), atol=2e-9, rtol=2e-9)
    constant = lay.interpolate("u", lambda x: 2.5)
    torch.testing.assert_close(constant[:5], constant.new_full((5,), 2.5), atol=1e-13, rtol=0)
    torch.testing.assert_close(constant[5:], torch.zeros_like(constant[5:]), atol=2e-10, rtol=0)


@pytest.mark.parametrize("p", [2, 4, 6])
def test_tetrahedral_shared_entities_are_sampled_once(p):
    asm = _space(p).from_mesh(_star(), quadrature_order=3)
    sampled = []

    def fn(x):
        sampled.append(x.detach())
        return torch.exp(x[:, 0] + .3 * x[:, 1] - .2 * x[:, 2])

    asm.layout.interpolate("u", fn)
    assert sum(len(x) for x in sampled) == asm.layout.n_nodes("u")
    sampled.clear()
    lifting = asm.layout.dirichlet_values("u", fn)
    mask = asm.layout.boundary_mask("u")
    assert sum(len(x) for x in sampled) == int(mask.sum())
    samples = torch.cat(sampled)
    on_boundary = (samples.abs() < 1e-13).any(-1) | ((samples.sum(-1) - 1).abs() < 1e-13)
    assert on_boundary.all()
    assert (lifting[~mask] == 0).all()
    assert mask.sum() == 4 + 6 * (p - 1) + 4 * (p - 1) * (p - 2) // 2
    assert not mask[4]  # interior vertex


def test_tetrahedral_all_orientation_classes_and_repeated_cell_selection():
    vertices, cells = [], []
    for i, permutation in enumerate(permutations(range(4))):
        vertices.extend((Tetrahedron.points.double() + torch.tensor([2 * i, 0, 0])).tolist())
        cells.append([4 * i + v for v in permutation])
    asm = _space(4).from_mesh(_mesh(vertices, cells), quadrature_order=3)
    fn = lambda x: 1 + .02 * x[..., 0] + x[..., 1] * x[..., 2] + x[..., 1] ** 4
    coefficients = asm.layout.interpolate("u", fn)
    ref = _reference()
    physical = _physical(asm, ref)
    torch.testing.assert_close(asm.layout.evaluate("u", coefficients, ref), fn(physical),
                               atol=3e-12, rtol=3e-12)
    selected = torch.tensor([23, 0, 7, 7, 2])
    queries = physical[selected]
    gradient = torch.stack((torch.full_like(queries[..., 0], .02),
                            queries[..., 2] + 4 * queries[..., 1] ** 3, queries[..., 1]), -1)
    torch.testing.assert_close(asm.layout.evaluate("u", coefficients, queries,
                                                   cell_indices=selected, physical=True),
                               fn(queries), atol=3e-12, rtol=3e-12)
    torch.testing.assert_close(asm.layout.evaluate("u", coefficients, queries,
                                                   cell_indices=selected, physical=True,
                                                   gradient=True),
                               gradient, atol=8e-12, rtol=3e-12)


@pytest.mark.parametrize("curved", [False, True])
def test_tetrahedral_curved_batched_values_physical_inversion_and_gradients(curved):
    asm = _space(4, components=2).from_mesh(_patch(2, curved), quadrature_order=3)
    fn = lambda x: torch.stack((x[..., 0] ** 2 + x[..., 1] * x[..., 2],
                                2 + x[..., 0] * x[..., 2]), -1)
    coefficients = asm.layout.interpolate("u", fn)
    reference = torch.stack((_reference(), _reference().roll(2, 0)))
    physical = _physical(asm, reference)
    expected = fn(physical)
    x, y, z = physical.unbind(-1)
    gradient = torch.stack((torch.stack((2 * x, z, y), -1),
                            torch.stack((z, torch.zeros_like(x), x), -1)), -2)
    for queries, is_physical in [(reference, False), (physical, True)]:
        torch.testing.assert_close(asm.layout.evaluate("u", coefficients, queries,
                                                       physical=is_physical),
                                   expected, atol=5e-12, rtol=5e-12)
        torch.testing.assert_close(asm.layout.evaluate("u", coefficients, queries,
                                                       physical=is_physical, gradient=True),
                                   gradient, atol=8e-12, rtol=8e-12)
    one = asm.layout.evaluate("u", coefficients, physical[1], cell_indices=1, physical=True)
    torch.testing.assert_close(one, expected[1:2], atol=5e-12, rtol=5e-12)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.complex128])
def test_tetrahedral_restriction_of_mesh_fe_data_preserves_trailing_axes(dtype):
    asm = _space(4).from_mesh(_patch(2, curved=True), quadrature_order=3)
    data = torch.randn(asm.n_points, 2, 3, dtype=torch.float64,
                       generator=torch.Generator().manual_seed(31)).to(dtype)
    if dtype.is_complex:
        data = data * (1 + 2j)
    coefficients = asm.layout.restrict("u", data)
    assert coefficients.dtype == dtype
    tolerance = 4e-6 if dtype == torch.float32 else 3e-12
    torch.testing.assert_close(asm.layout.prolong("u", coefficients), data,
                               atol=tolerance, rtol=tolerance)
    ref = _reference()
    table = Tetrahedron.get_basis_fns(2, ref.dtype).map(ref).to(dtype)
    conn = asm.elements[asm.element_types[0]]
    expected = torch.einsum("qb,ebij->eqij", table, data[conn])
    torch.testing.assert_close(asm.layout.evaluate("u", coefficients, ref), expected,
                               atol=tolerance, rtol=tolerance)
    assert asm.layout.evaluate("u", coefficients, ref, gradient=True).shape == (2, 6, 2, 3, 3)


def test_tetrahedral_restriction_uses_nodal_fe_field_not_imagined_modal_points():
    asm = _space(4).from_mesh(_patch(), quadrature_order=3)
    restricted = asm.layout.restrict("u", asm._points[:, 0] ** 2)
    linear = asm.layout.interpolate("u", lambda x: x[:, 0])
    quadratic = asm.layout.interpolate("u", lambda x: x[:, 0] ** 2)
    torch.testing.assert_close(restricted, linear, atol=1e-13, rtol=1e-12)
    assert not torch.allclose(restricted, quadratic)


@pytest.mark.parametrize("curved", [False, True])
def test_tetrahedral_boundary_only_component_lifting_with_interior_entities(curved):
    asm = _space(4, components=2, other=True).from_mesh(_star(2, curved), quadrature_order=3)
    lay = asm.layout

    def fn(x):
        original = _unwarp(x) if curved else x
        boundary = (original.abs() < 2e-13).any(-1) | ((original.sum(-1) - 1).abs() < 2e-13)
        assert boundary.all(), "Dirichlet data were sampled away from the boundary"
        return torch.stack((1 + x[:, 0] ** 2 + x[:, 1] * x[:, 2],
                            2 + x[:, 0] * x[:, 2]), -1)

    full = lay.dirichlet_values("u", fn)
    first = lay.dirichlet_values("u", lambda x: fn(x)[:, 0], component=0)
    second = lay.dirichlet_values("u", lambda x: fn(x)[:, 1], component=1)
    torch.testing.assert_close(full, first + second, atol=2e-12, rtol=2e-12)
    assert (full[~lay.boundary_mask("u")] == 0).all()
    assert (first[~lay.boundary_mask("u", component=0)] == 0).all()
    assert (second[~lay.boundary_mask("u", component=1)] == 0).all()
    assert (lay.split(full)["pressure"] == 0).all()
    _, tri = Triangle.get_quadrature(8, torch.float64)
    bary = torch.cat((1 - tri.sum(-1, keepdim=True), tri), -1)
    mesh_cells = asm.elements[asm.element_types[0]]
    for cell in range(len(mesh_cells)):
        face_slots = [slot for slot, vertex in enumerate(mesh_cells[cell, :4]) if vertex != 4]
        ref = bary @ Tetrahedron.points.double()[face_slots]
        physical = _physical(asm, ref, [cell])
        expected = fn(physical.reshape(-1, 3))[None]
        actual = lay.evaluate("u", lay.split(full)["u"], ref, cell_indices=cell)
        torch.testing.assert_close(actual, expected, atol=5e-12, rtol=5e-12)


def test_tetrahedral_arbitrary_modal_shared_face_values_and_tangential_gradients():
    asm = _space(6).from_mesh(_patch(2, curved=True), quadrature_order=3)
    coefficients = torch.randn(asm.n_dofs, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(29))
    uv = torch.tensor([[.12, .23], [.43, .17], [.11, .76]], dtype=torch.float64)
    original = torch.cat((uv, torch.zeros(len(uv), 1, dtype=torch.float64)), -1)
    physical = _warp(original)
    values = asm.layout.evaluate("u", coefficients, physical, physical=True)
    gradient = asm.layout.evaluate("u", coefficients, physical, physical=True, gradient=True)
    torch.testing.assert_close(values[0], values[1], atol=2e-12, rtol=2e-12)
    tangent_x = physical.new_tensor([1, 0, 0]).expand_as(physical)
    tangent_y = torch.stack((torch.zeros_like(uv[:, 1]), torch.ones_like(uv[:, 1]),
                             .08 * (1 - 2 * uv[:, 1])), -1)
    for tangent in (tangent_x, tangent_y):
        derivative = (gradient * tangent).sum(-1)
        torch.testing.assert_close(derivative[0], derivative[1], atol=8e-12, rtol=8e-12)


@pytest.mark.parametrize("p", [1, 3, 4, 6])
def test_tetrahedral_manufactured_nonzero_dirichlet_poisson(p):
    class Poisson(_space(p)):
        def forward_vector(self, v, x):
            if p == 1:
                return 0 * v
            forcing = -p * (p - 1) * (.4 * x[0] ** (p - 2) + .3 * x[1] ** (p - 2)
                                      - .2 * x[2] ** (p - 2) + .3 * x.sum() ** (p - 2))
            return forcing * v

    asm = Poisson.from_mesh(_star(), quadrature_order=max(8, 2 * p))
    lay = asm.layout
    lifting = lay.dirichlet_values("u", lambda x: _polynomial(x, p))
    condenser = Condenser(lay.boundary_mask("u"), lifting)
    matrix, load = condenser(asm(), asm.assemble_vector())
    coefficients = condenser.recover(torch.linalg.solve(matrix.to_dense(), load))
    reference = _reference()
    physical = _physical(asm, reference)
    torch.testing.assert_close(lay.evaluate("u", coefficients, reference),
                               _polynomial(physical, p), atol=8e-11, rtol=8e-11)
    torch.testing.assert_close(lay.evaluate("u", coefficients, reference, gradient=True),
                               _polynomial_gradient(physical, p), atol=3e-10, rtol=3e-10)


def test_tetrahedral_quartic_volume_bubble_poisson():
    class Bubble(_space(4)):
        def forward_vector(self, v, x):
            return 2 * (x[0] * x[1] + x[0] * x[2] + x[1] * x[2]) * v

    asm = Bubble.from_mesh(_mesh(Tetrahedron.points, [[2, 0, 3, 1]]), quadrature_order=8)
    exact = lambda x: (1 - x.sum(-1)) * x.prod(-1)
    lifting = asm.layout.dirichlet_values("u", exact)
    torch.testing.assert_close(lifting, torch.zeros_like(lifting), atol=2e-15, rtol=0)
    condenser = Condenser(asm.layout.boundary_mask("u"), lifting)
    matrix, load = condenser(asm(), asm.assemble_vector())
    coefficients = condenser.recover(torch.linalg.solve(matrix.to_dense(), load))
    assert int((~asm.layout.boundary_mask("u")).sum()) == 1
    assert coefficients[-1].abs() > .5
    ref = _reference()
    torch.testing.assert_close(asm.layout.evaluate("u", coefficients, ref),
                               exact(_physical(asm, ref)), atol=2e-13, rtol=2e-12)


def _cube():
    vertices = list(product(range(2), repeat=3))
    cells = []
    for axes in permutations(range(3)):
        corner = [0, 0, 0]
        cell = [0]
        for axis in axes:
            corner[axis] = 1
            cell.append(vertices.index(tuple(corner)))
        cells.append(cell)
    return _mesh(vertices, cells)


def test_tetrahedral_sine_poisson_p_convergence_on_connected_cube():
    errors = []
    for p in (2, 4, 6):
        class Poisson(_space(p)):
            def forward_vector(self, v, x):
                return 3 * math.pi ** 2 * torch.sin(math.pi * x).prod() * v

        asm = Poisson.from_mesh(_cube(), quadrature_order=18)
        condenser = Condenser(asm.layout.boundary_mask("u"))
        matrix, load = condenser(asm(), asm.assemble_vector())
        coefficients = condenser.recover(torch.linalg.solve(matrix.to_dense(), load))
        weights, ref = Tetrahedron.get_quadrature(22, torch.float64)
        exact = torch.sin(math.pi * _physical(asm, ref)).prod(-1)
        error = asm.layout.evaluate("u", coefficients, ref) - exact
        # All six affine cube tetrahedra have |det J|=1.
        errors.append(float(torch.sqrt((weights * error.square()).sum())))
    assert errors[1] < .3 * errors[0]
    assert errors[2] < .3 * errors[1]


def test_tetrahedral_field_autograd_and_coordinate_updates():
    asm = _space(4).from_mesh(_patch(2, curved=True), quadrature_order=3)
    amplitude = torch.tensor(1.7, dtype=torch.float64, requires_grad=True)
    fn = lambda x: x[..., 0] ** 2 + x[..., 1] * x[..., 2]
    ref = _reference()
    coefficients = asm.layout.interpolate("u", lambda x: amplitude * fn(x))
    values = asm.layout.evaluate("u", coefficients, ref)
    derivative = torch.autograd.grad(values.sum(), amplitude)[0]
    torch.testing.assert_close(derivative, fn(_physical(asm, ref)).sum(), atol=2e-12, rtol=2e-12)
    lifting = asm.layout.dirichlet_values("u", lambda x: amplitude * fn(x))
    assert torch.autograd.grad(lifting.square().sum(), amplitude)[0] > 0
    points = (asm._points + .02).detach().requires_grad_()
    for transformation in asm.transformation.values():
        transformation.update_points(points)
    for _ in range(2):
        coefficients = asm.layout.interpolate("u", fn)
        total = asm.layout.evaluate("u", coefficients, ref).sum()
        actual = torch.autograd.grad(total, asm._points)[0]
        expected = torch.autograd.grad(fn(_physical(asm, ref)).sum(), asm._points)[0]
        torch.testing.assert_close(actual, expected, atol=8e-12, rtol=8e-12)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tetrahedral_field_helpers_follow_dtype_device_and_state_dict(dtype, device):
    asm = _space(4).from_mesh(_patch(2, curved=True), quadrature_order=3)
    asm.to(device=device, dtype=dtype)
    clone = _space(4).from_mesh(_patch(2, curved=True), quadrature_order=3).to(device=device, dtype=dtype)
    clone.load_state_dict(asm.state_dict())
    fn = lambda x: x[..., 0] ** 2 + 2 * x[..., 1] + x[..., 2]
    coefficients = clone.layout.interpolate("u", fn)
    ref = _reference(dtype, device)
    physical = _physical(clone, ref)
    tolerance = 1e-4 if dtype == torch.float32 else 8e-12
    torch.testing.assert_close(clone.layout.evaluate("u", coefficients, physical, physical=True),
                               fn(physical), atol=tolerance, rtol=tolerance)
    gradient = torch.stack((2 * physical[..., 0], torch.full_like(physical[..., 0], 2),
                            torch.ones_like(physical[..., 0])), -1)
    torch.testing.assert_close(clone.layout.evaluate("u", coefficients, ref, gradient=True),
                               gradient, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(clone.layout.prolong("u", coefficients), fn(clone._points),
                               atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(clone.layout.dirichlet_values("u", fn),
                               asm.layout.dirichlet_values("u", fn), atol=tolerance, rtol=tolerance)
    assert clone.layout.evaluate("u", coefficients, ref[:0]).shape == (2, 0)
    assert clone.layout.evaluate("u", coefficients, ref[:0], gradient=True).shape == (2, 0, 3)
    empty = torch.empty(0, dtype=torch.long, device=device)
    assert clone.layout.evaluate("u", coefficients, ref, cell_indices=empty).shape == (0, len(ref))


def test_tetrahedral_physical_queries_validate_domain_and_newton_convergence():
    asm = _space(4).from_mesh(_patch(2, curved=True), quadrature_order=3)
    coefficients = asm.layout.interpolate("u", lambda x: x[:, 0])
    with pytest.raises(ValueError, match="outside.*tetrahedron"):
        asm.layout.evaluate("u", coefficients, [[2., 2., 2.]], cell_indices=0, physical=True)
    with pytest.raises(ValueError, match="did not converge"):
        asm.layout.evaluate("u", coefficients, [[.2, .2, .2]], cell_indices=0,
                            physical=True, max_iterations=1, tolerance=1e-30)
    with pytest.raises(ValueError, match="finite and positive"):
        asm.layout.evaluate("u", coefficients, _reference(), physical=True, tolerance=float("nan"))
    with pytest.raises(ValueError, match="component"):
        asm.layout.dirichlet_values("u", lambda x: 0, component=1)
    with pytest.raises(NotImplementedError, match="modal coefficients"):
        asm.layout.points("u")


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6, 8, 10])
def test_tetrahedral_entity_collocation_conditioning(p):
    from tensormesh.assemble.tetrahedron_fields import _entity_stages

    points = Tetrahedron.points.double()
    basis = Tetrahedron.get_hierarchical_basis_fns(p, torch.float64)
    for entities in _entity_stages(p, points):
        for corners, slots, weights in entities:
            table = basis.map(weights @ points[list(corners)])[:, slots]
            condition = torch.linalg.cond(table)
            assert torch.isfinite(condition)
            # The largest measured condition is 319.3 at p=10 (cell block).
            # Bound the actual recurrence systems, not a monomial Vandermonde.
            assert condition < 1e3
