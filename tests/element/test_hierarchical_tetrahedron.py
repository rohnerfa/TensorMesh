"""Independent polynomial, Jacobi, trace and derivative tests for tetrahedra."""

from itertools import permutations

import numpy as np
import pytest
import torch
from scipy.special import eval_jacobi

from tensormesh.element import (HierarchicalTetrahedronBasisFns, Tetrahedron,
                               TETRAHEDRON_ORIENTATIONS)


def sample(n, seed=23):
    """Uniform tetrahedral samples, independent of the recurrence coordinates."""
    cuts = torch.rand(n, 3, generator=torch.Generator().manual_seed(seed),
                      dtype=torch.float64).sort(dim=1).values
    return torch.cat((cuts[:, :1], cuts[:, 1:] - cuts[:, :-1]), dim=1)


def scipy_tables(points, p, orientation):
    """Independent Jacobi evaluation and chain rule, at interior points only."""
    xyz = points.numpy()
    values = np.concatenate((1 - xyz.sum(-1, keepdims=True), xyz), axis=-1)
    derivatives = np.concatenate((-np.ones((1, 3)), np.eye(3)))
    lam = [(values[:, i], np.broadcast_to(derivatives[i], (len(points), 3)))
           for i in range(4)]
    one = (np.ones(len(points)), np.zeros((len(points), 3)))

    def add(a, b, scale=1):
        return a[0] + scale * b[0], a[1] + scale * b[1]

    def mul(*factors):
        v, g = one
        for b, db in factors:
            v, g = v * b, g * b[:, None] + v[:, None] * db
        return v, g

    def jacobi(n, alpha, s, t):
        z = s[0] / t[0]
        poly = eval_jacobi(n, alpha, 1, z)
        if n == 0:
            return one
        derivative = 0.5 * (n + alpha + 2) * eval_jacobi(n - 1, alpha + 1, 2, z)
        ds = t[0] ** (n - 1) * derivative
        dt = t[0] ** (n - 1) * (n * poly - z * derivative)
        return t[0] ** n * poly, ds[:, None] * s[1] + dt[:, None] * t[1]

    rank = {slot: i for i, slot in enumerate(orientation)}
    result = list(lam)
    for edge in Tetrahedron.edge.tolist():
        a, b = sorted(edge, key=rank.__getitem__)
        for n in range(2, p + 1):
            v, g = mul(lam[a], lam[b], jacobi(n - 2, 1,
                       add(lam[b], lam[a], -1), add(lam[a], lam[b])))
            result.append((-2 * v / (n - 1), -2 * g / (n - 1)))
    for face in Tetrahedron.face:
        a, b, c = sorted(face, key=rank.__getitem__)
        s, sf = add(lam[a], lam[b]), add(add(lam[a], lam[b]), lam[c])
        for total in range(p - 2):
            for i in range(total + 1):
                j = total - i
                result.append(mul(lam[a], lam[b], lam[c],
                    jacobi(i, 1, add(lam[b], lam[a], -1), s),
                    jacobi(j, 2 * i + 3, add(add(lam[c], lam[c]), sf, -1), sf)))
    s, sf = add(lam[0], lam[1]), add(add(lam[0], lam[1]), lam[2])
    for total in range(p - 3):
        for i in range(total + 1):
            for j in range(total - i + 1):
                k = total - i - j
                result.append(mul(*lam,
                    jacobi(i, 1, add(lam[1], lam[0], -1), s),
                    jacobi(j, 2 * i + 3, add(add(lam[2], lam[2]), sf, -1), sf),
                    jacobi(k, 2 * (i + j) + 5, add(add(lam[3], lam[3]), one, -1), one)))
    return (torch.from_numpy(np.stack([r[0] for r in result], -1)),
            torch.from_numpy(np.stack([r[1] for r in result], -1)))


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6, 8, 10])
def test_tetrahedron_dimension_rank_and_all_monomial_reproduction(p):
    basis = Tetrahedron.get_hierarchical_basis_fns(p, torch.float64,
                                                  orientation=(2, 0, 3, 1))
    n = (p + 1) * (p + 2) * (p + 3) // 6
    points, check = sample(4 * n), sample(2 * n, seed=163)
    powers = [(i, j, total - i - j) for total in range(p + 1)
              for i in range(total + 1) for j in range(total - i + 1)]

    def exact(q):
        return torch.stack([q[:, 0] ** i * q[:, 1] ** j * q[:, 2] ** k
                            for i, j, k in powers], -1)

    def exact_grad(q):
        columns = []
        for power in powers:
            grad = []
            for d in range(3):
                exponents = tuple(v - (a == d) for a, v in enumerate(power))
                grad.append(power[d] * torch.prod(q ** q.new_tensor(exponents), -1)
                            if power[d] else torch.zeros(len(q), dtype=q.dtype))
            columns.append(torch.stack(grad, -1))
        return torch.stack(columns, -1)

    values = basis.map(points)
    fit = torch.linalg.lstsq(values, exact(points), driver="gelsd")
    assert basis.shape == (n,) and values.shape == (4 * n, n)
    assert fit.rank.item() == n
    # Independent points, all monomials (not just a random polynomial).
    torch.testing.assert_close(basis.map(check) @ fit.solution, exact(check),
                               atol=3e-11, rtol=2e-9)
    torch.testing.assert_close(basis.grad().map(check) @ fit.solution, exact_grad(check),
                               atol=4e-10, rtol=2e-8)


@pytest.mark.parametrize("orientation", TETRAHEDRON_ORIENTATIONS)
def test_tetrahedron_all_orientation_classes_against_scipy_values_and_gradients(orientation):
    basis = HierarchicalTetrahedronBasisFns(6, torch.float64, orientation=orientation)
    points = sample(19)
    expected, expected_grad = scipy_tables(points, 6, orientation)
    torch.testing.assert_close(basis.map(points), expected, atol=2e-14, rtol=3e-12)
    torch.testing.assert_close(basis.grad().map(points), expected_grad, atol=2e-13, rtol=3e-12)


def test_tetrahedron_extended_jacobi_formula_and_lexicographic_classes():
    assert TETRAHEDRON_ORIENTATIONS == tuple(permutations(range(4)))
    basis = HierarchicalTetrahedronBasisFns(10, torch.float64, orientation=(3, 1, 2, 0))
    expected, expected_grad = scipy_tables(sample(31), 10, basis.orientation)
    torch.testing.assert_close(basis.map(sample(31)), expected, atol=3e-14, rtol=5e-12)
    torch.testing.assert_close(basis.grad().map(sample(31)), expected_grad,
                               atol=4e-13, rtol=1e-11)
    assert basis.edges == tuple(map(tuple, Tetrahedron.edge.tolist()))
    assert basis.faces == Tetrahedron.face


@pytest.mark.parametrize("p", [1, 2, 4, 6, 10])
def test_tetrahedron_analytic_gradients_at_all_boundary_entities(p):
    vertices = Tetrahedron.points.double()
    edges = vertices[Tetrahedron.edge].mean(1)
    faces = vertices[torch.tensor(Tetrahedron.face)].mean(1)
    points = torch.cat((sample(3), vertices, edges, faces))
    basis = HierarchicalTetrahedronBasisFns(p, torch.float64, orientation=(3, 2, 0, 1))
    actual = basis.grad().map(points)
    expected = torch.func.vmap(torch.func.jacrev(basis.map))(points).transpose(-1, -2)
    assert torch.isfinite(actual).all() and torch.isfinite(basis.map(points)).all()
    torch.testing.assert_close(actual, expected, atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(basis.map(vertices)[:, :4], torch.eye(4, dtype=torch.float64))
    assert torch.count_nonzero(basis.map(vertices)[:, 4:]) == 0


@pytest.mark.parametrize("p", [2, 4, 6, 10])
def test_tetrahedron_entity_trace_support_and_canonical_face_formula(p):
    basis = HierarchicalTetrahedronBasisFns(p, torch.float64, orientation=(2, 3, 1, 0))
    vertices = Tetrahedron.points.double()
    rank = {v: i for i, v in enumerate(basis.orientation)}
    nf = max(0, (p - 1) * (p - 2) // 2)
    face_base = 4 + 6 * (p - 1)
    bary = torch.tensor([[0.19, 0.27, 0.54], [0.61, 0.16, 0.23], [0., 0.4, 0.6]],
                        dtype=torch.float64)
    for f, face in enumerate(basis.faces):
        canonical = sorted(face, key=rank.__getitem__)
        points = bary @ vertices[canonical]
        values = basis.map(points)
        # Every mode belonging to an entity outside this face has zero trace.
        absent = [v for v in range(4) if v not in face]
        for e, edge in enumerate(basis.edges):
            if not set(edge) <= set(face):
                absent.extend(range(4 + e * (p - 1), 4 + (e + 1) * (p - 1)))
        for other in range(4):
            if other != f:
                absent.extend(range(face_base + other * nf, face_base + (other + 1) * nf))
        absent.extend(range(face_base + 4 * nf, basis.shape[0]))
        torch.testing.assert_close(values[:, absent], torch.zeros_like(values[:, absent]),
                                   atol=2e-15, rtol=0)
        # The canonical triangular interior formula, not the local face slot order.
        a, b, c = bary.numpy().T
        canonical_face = [torch.from_numpy(a * b * c * (a + b) ** i
            * eval_jacobi(i, 1, 1, (b - a) / (a + b))
            * eval_jacobi(total - i, 2 * i + 3, 1, 2 * c - 1))
            for total in range(p - 2) for i in range(total + 1)]
        if nf:
            torch.testing.assert_close(values[:, face_base + f * nf:face_base + (f + 1) * nf],
                                       torch.stack(canonical_face, -1), atol=2e-15, rtol=3e-12)
    for e, edge in enumerate(basis.edges):
        t = torch.linspace(0, 1, 13, dtype=torch.float64)
        values = basis.map((1 - t[:, None]) * vertices[edge[0]] + t[:, None] * vertices[edge[1]])
        absent = list(range(face_base, basis.shape[0]))
        for other in range(6):
            if other != e:
                absent.extend(range(4 + other * (p - 1), 4 + (other + 1) * (p - 1)))
        torch.testing.assert_close(values[:, absent], torch.zeros_like(values[:, absent]),
                                   atol=2e-15, rtol=0)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6, 9])
def test_tetrahedron_exact_entity_mode_nesting(p):
    low = HierarchicalTetrahedronBasisFns(p, torch.float64, orientation=(3, 0, 2, 1))
    high = HierarchicalTetrahedronBasisFns(p + 1, torch.float64, orientation=low.orientation)
    ne, nf = p - 1, max(0, (p - 1) * (p - 2) // 2)
    nc = (p - 1) * (p - 2) * (p - 3) // 6 if p >= 4 else 0
    high_nf = p * (p - 1) // 2
    slots = list(range(4))
    for edge in range(6):
        slots.extend(range(4 + edge * p, 4 + edge * p + ne))
    for face in range(4):
        slots.extend(range(4 + 6 * p + face * high_nf, 4 + 6 * p + face * high_nf + nf))
    slots.extend(range(4 + 6 * p + 4 * high_nf, 4 + 6 * p + 4 * high_nf + nc))
    torch.testing.assert_close(low.map(sample(9)), high.map(sample(9))[:, slots], atol=0, rtol=0)
    torch.testing.assert_close(low.grad().map(sample(9)), high.grad().map(sample(9))[:, :, slots],
                               atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tetrahedron_batch_shapes_and_module_dtype_movement(dtype):
    basis = HierarchicalTetrahedronBasisFns(4, orientation=(1, 3, 2, 0)).to(dtype=dtype)
    points = sample(6).reshape(2, 3, 3)
    assert basis.map(points).shape == (2, 3, 35)
    assert basis.grad().map(points).shape == (2, 3, 3, 35)
    assert basis.map(points).dtype == dtype
    assert basis.map(points[:0]).shape == (0, 3, 35)
    assert basis.grad().map(points[:0]).shape == (0, 3, 3, 35)
    assert basis.map(points[0, 0]).shape == (35,)
    clone = HierarchicalTetrahedronBasisFns(4, dtype, orientation=basis.orientation)
    clone.load_state_dict(basis.state_dict())
    torch.testing.assert_close(clone.map(points), basis.map(points), atol=0, rtol=0)
    torch.testing.assert_close(basis.double().map(points), clone.double().map(points), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tetrahedron_cuda_values_gradients_and_autograd(dtype):
    cpu = HierarchicalTetrahedronBasisFns(6, dtype, orientation=(3, 1, 0, 2))
    gpu = HierarchicalTetrahedronBasisFns(6, dtype, orientation=cpu.orientation).cuda()
    points = sample(13).to(dtype)
    gpu_points = points.cuda().requires_grad_()
    tolerance = dict(atol=3e-6, rtol=3e-5) if dtype == torch.float32 else dict(atol=3e-13, rtol=3e-12)
    torch.testing.assert_close(gpu.map(gpu_points).cpu(), cpu.map(points), **tolerance)
    torch.testing.assert_close(gpu.grad().map(gpu_points).cpu(), cpu.grad().map(points), **tolerance)
    derivative = torch.autograd.grad(gpu.map(gpu_points).sum(), gpu_points)[0]
    torch.testing.assert_close(derivative, gpu.grad().map(gpu_points).sum(-1), **tolerance)


@pytest.mark.parametrize("p", [0, -1, True, 2.5])
def test_tetrahedron_invalid_polynomial_order(p):
    with pytest.raises(ValueError, match="integer >= 1"):
        HierarchicalTetrahedronBasisFns(p)


@pytest.mark.parametrize("orientation", [(0, 1, 2), (0, 1, 2, 2), (0, 1, 2, 4),
                                          (False, 1, 2, 3), (0., 1, 2, 3), [0, 1, 2, 3]])
def test_tetrahedron_invalid_orientation_descriptor(orientation):
    with pytest.raises(ValueError, match="tuple permuting"):
        Tetrahedron.get_hierarchical_basis_fns(orientation=orientation)


def test_tetrahedron_invalid_point_shape_and_second_derivatives():
    basis = HierarchicalTetrahedronBasisFns()
    for shape in [(), (2,), (3, 2)]:
        with pytest.raises(ValueError, match="reference points"):
            basis.map(torch.empty(shape))
    with pytest.raises(NotImplementedError, match="second derivatives"):
        basis.grad().grad()
    nodes = Tetrahedron.get_basis(2, torch.float64)
    torch.testing.assert_close(Tetrahedron.get_basis_fns(2, torch.float64).map(nodes),
                               torch.eye(10, dtype=torch.float64), atol=1e-14, rtol=0)
