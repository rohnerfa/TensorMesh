"""Independent polynomial, trace, hierarchy and derivative checks."""

import pytest
import torch
from scipy.special import eval_jacobi, eval_legendre

from tensormesh.element import Triangle, HierarchicalTriangleBasisFns


def sample(n, seed=17):
    uv = torch.rand(n, 2, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
    return torch.stack((uv[:, 0], (1 - uv[:, 0]) * uv[:, 1]), -1)


@pytest.mark.parametrize("p", [1, 2, 3, 6, 10])
def test_count_independence_and_polynomial_reproduction(p):
    basis = Triangle.get_hierarchical_basis_fns(p, torch.float64)
    n = (p + 1) * (p + 2) // 2
    points, check = sample(4 * n), sample(2 * n, seed=123)
    values = basis.map(points)
    assert basis.shape == (n,)
    assert values.shape == (4 * n, n)
    assert torch.linalg.matrix_rank(values).item() == n
    # Reproduce every monomial through degree p, at independent points.
    powers = [(i, total - i) for total in range(p + 1) for i in range(total + 1)]
    exact = lambda q: torch.stack([q[:, 0] ** i * q[:, 1] ** j for i, j in powers], -1)
    coefficients = torch.linalg.lstsq(values, exact(points)).solution
    torch.testing.assert_close(basis.map(check) @ coefficients, exact(check), atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("p", [2, 3, 6, 10])
def test_traces_and_edge_parity(p):
    basis = Triangle.get_hierarchical_basis_fns(p, torch.float64)
    assert tuple(map(tuple, Triangle.edge.tolist())) == basis.edges
    vertices = Triangle.points.to(torch.float64)
    t = torch.linspace(0, 1, 31, dtype=torch.float64)
    for edge, (a, b) in enumerate(basis.edges):
        points = (1 - t[:, None]) * vertices[a] + t[:, None] * vertices[b]
        values = basis.map(points)
        for other in range(3):
            block = values[:, 3 + other * (p - 1):3 + (other + 1) * (p - 1)]
            if other != edge:
                torch.testing.assert_close(block, torch.zeros_like(block), atol=1e-15, rtol=0)
            else:
                for k in range(p - 1):
                    n = k + 2
                    z = (2 * t - 1).numpy()
                    # Independently evaluate the integrated Legendre identity.
                    expected = torch.from_numpy((eval_legendre(n, z) - eval_legendre(n - 2, z)) / (2 * n - 1))
                    torch.testing.assert_close(block[:, k], expected, atol=2e-15, rtol=2e-13)
                    torch.testing.assert_close(block.flip(0)[:, k], (-1) ** n * block[:, k], atol=2e-15, rtol=2e-13)
        interior = values[:, 3 + 3 * (p - 1):]
        torch.testing.assert_close(interior, torch.zeros_like(interior), atol=1e-15, rtol=0)
    at_vertices = basis.map(vertices)
    torch.testing.assert_close(at_vertices[:, :3], torch.eye(3, dtype=torch.float64))
    torch.testing.assert_close(at_vertices[:, 3:], torch.zeros_like(at_vertices[:, 3:]))


@pytest.mark.parametrize("p", [1, 2, 4, 10])
def test_analytic_gradient_against_autograd(p):
    basis = Triangle.get_hierarchical_basis_fns(p, torch.float64)
    points = torch.cat((sample(5), Triangle.points.to(torch.float64),
                        torch.tensor([[0.5, 0.5], [0., 0.5]], dtype=torch.float64)))
    expected = torch.func.vmap(torch.func.jacrev(basis.map))(points).transpose(-1, -2)
    actual = basis.grad().map(points)
    assert actual.shape == (len(points), 2, basis.shape[0])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-13, rtol=2e-12)


def test_bubble_formula_against_scipy():
    p = 10
    points = sample(40)
    x, y = points.numpy().T
    s, d = 1 - y, 2 * x + y - 1
    expected = []
    for total in range(p - 2):
        for i in range(total + 1):
            j = total - i
            expected.append(torch.from_numpy((1 - x - y) * x * y * s ** i
                            * eval_jacobi(i, 1, 1, d / s)
                            * eval_jacobi(j, 2 * i + 3, 1, 2 * y - 1)))
    actual = Triangle.get_hierarchical_basis_fns(p, torch.float64).map(points)
    torch.testing.assert_close(actual[:, 3 * p:], torch.stack(expected, -1), atol=1e-14, rtol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 6, 9])
def test_nesting(p):
    points = sample(13)
    low = Triangle.get_hierarchical_basis_fns(p, torch.float64)
    high = Triangle.get_hierarchical_basis_fns(p + 1, torch.float64)
    indices = list(range(3))
    for edge in range(3):
        indices.extend(range(3 + edge * p, 3 + edge * p + p - 1))
    indices.extend(range(3 + 3 * p, 3 + 3 * p + (p - 1) * (p - 2) // 2))
    torch.testing.assert_close(low.map(points), high.map(points)[:, indices], atol=0, rtol=0)
    torch.testing.assert_close(low.grad().map(points), high.grad().map(points)[:, :, indices], atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_layout_dtype_and_module_conversion(dtype):
    basis = HierarchicalTriangleBasisFns(4).to(dtype=dtype)
    points = sample(6).reshape(2, 3, 2).to(dtype)
    assert basis.map(points).shape == (2, 3, 15)
    assert basis.grad().map(points).shape == (2, 3, 2, 15)
    assert basis.map(points).dtype == dtype
    assert basis.map(points).device == points.device
    assert basis.map(points[:0]).shape == (0, 3, 15)


def test_factory_module_conversion_isolation():
    device = torch.device("cpu")
    points = sample(6)
    basis = Triangle.get_hierarchical_basis_fns(5, torch.float64, device)
    expected = basis.map(points)
    basis.float()
    assert basis.map(points).dtype == torch.float32
    actual = Triangle.get_hierarchical_basis_fns(5, torch.float64, device).map(points)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    basis = HierarchicalTriangleBasisFns(6).to(device)
    points = sample(12).float()
    torch.testing.assert_close(basis.map(points.to(device)).cpu(),
                               HierarchicalTriangleBasisFns(6).map(points), atol=2e-6, rtol=2e-5)
    assert basis.grad().map(points.to(device)).device.type == device


@pytest.mark.parametrize("order", [0, -1, 2.5, True])
def test_hierarchical_basis_rejects_invalid_order(order):
    with pytest.raises(ValueError, match="integer >= 1"):
        HierarchicalTriangleBasisFns(order)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triangle_nodal_geometry_basis_constructs_on_cuda():
    # Modal fields continue to use nodal geometry on the selected device.
    nodes = Triangle.get_basis(2, torch.float64, torch.device("cuda"))
    values = Triangle.get_basis_fns(2, torch.float64, nodes.device).map(nodes)
    torch.testing.assert_close(values, torch.eye(6, dtype=nodes.dtype, device=nodes.device),
                               atol=1e-14, rtol=0)
