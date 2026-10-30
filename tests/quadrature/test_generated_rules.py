"""Gaussian rules checked against analytic moments, not another FEM backend."""

import math

import pytest
import torch

from tensormesh.element import Line, Triangle, Tetrahedron, Quadrilateral
from tensormesh.element import quadrature


@pytest.mark.parametrize("n", [1, 2, 7, 8, 11, 21, 32, 64])
def test_line_gaussian_exactness(n):
    weights, points = Line.get_quadrature(n, torch.float64)
    assert weights.shape == (n,) and points.shape == (n, 1)
    assert (weights > 0).all()
    assert ((points > 0) & (points < 1)).all()
    powers = torch.arange(2 * n, dtype=torch.float64)
    actual = (weights[:, None] * points ** powers).sum(0)
    exact = 1 / (powers + 1)
    torch.testing.assert_close(actual / exact, torch.ones_like(exact), atol=3e-12, rtol=0)


@pytest.mark.parametrize("degree", [*range(1, 11), 12, 18, 20, 21, 30, 40, 64])
def test_triangle_all_monomials_through_requested_degree(degree):
    weights, points = Triangle.get_quadrature(degree, torch.float64)
    powers = [(a, total - a) for total in range(degree + 1) for a in range(total + 1)]
    exponents = torch.tensor(powers)
    monomials = points[:, :1] ** exponents[:, 0] * points[:, 1:] ** exponents[:, 1]
    exact = torch.tensor([math.exp(math.lgamma(a + 1) + math.lgamma(b + 1)
                                   - math.lgamma(a + b + 3)) for a, b in powers], dtype=torch.float64)
    actual = torch.einsum("q,qi->i", weights, monomials)
    # Relative normalization is essential: tiny high-degree moments must
    # not pass merely because of a generous absolute tolerance.
    torch.testing.assert_close(actual / exact, torch.ones_like(exact), atol=4e-12, rtol=0)
    if degree > 7:
        assert len(weights) == ((degree + 2) // 2) ** 2
        assert (weights > 0).all()
        assert (points > 0).all() and (points.sum(-1) < 1).all()
    torch.testing.assert_close(weights.sum(), torch.tensor(0.5, dtype=torch.float64), atol=3e-15, rtol=0)


def test_original_low_degree_triangle_rules_retained():
    for degree, count in enumerate([1, 3, 4, 6, 7, 12, 13], start=1):
        weights, points = Triangle.get_quadrature(degree, torch.float64)
        assert len(weights) == count
        if degree in (3, 7):
            assert weights[0] < 0  # preserve the original nonpositive rules
            torch.testing.assert_close(points[0], torch.full((2,), 1 / 3, dtype=torch.float64), atol=1e-15, rtol=0)
    weights, points = Triangle.get_quadrature(2, torch.float64)
    torch.testing.assert_close(weights, torch.full((3,), 1 / 6, dtype=torch.float64), atol=0, rtol=0)
    torch.testing.assert_close(points, torch.tensor([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]], dtype=torch.float64), atol=0, rtol=0)


def test_reference_generation_cache_and_copy_safety(monkeypatch):
    quadrature._gaussian_interval_rule.cache_clear()
    quadrature._gaussian_triangle_rule.cache_clear()
    calls = []
    original = quadrature.roots_jacobi
    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(quadrature, "roots_jacobi", counted)
    w, q = quadrature.tri_quadrature(20, torch.float64)
    assert len(calls) == 2
    w.zero_()
    q.zero_()
    # Degrees 20 and 21 share the same 11x11 canonical rule.
    weights, points = quadrature.tri_quadrature(21, torch.float32)
    assert len(calls) == 2
    assert weights.sum() > 0.49 and points.sum() > 0
    weights, points = quadrature.tri_quadrature(20, torch.float64)
    assert weights.sum() > 0.49 and points.sum() > 0
    quadrature.lin_quadrature(11, torch.float64)
    assert len(calls) == 2


def test_canonical_rule_cache_uses_cpu_independently_of_default_device():
    quadrature._gaussian_interval_rule.cache_clear()
    quadrature._gaussian_triangle_rule.cache_clear()
    with torch.device("meta"):
        weights, points = quadrature._gaussian_triangle_rule(9)
    assert weights.device.type == points.device.type == "cpu"
    assert weights.dtype == points.dtype == torch.float64
    torch.testing.assert_close(weights.sum(), torch.tensor(0.5, dtype=torch.float64))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_rule_layouts_dtype_and_facets(dtype):
    for element, order, count, dim in [(Line, 12, 12, 1), (Triangle, 20, 121, 2),
                                      (Quadrilateral, 8, 64, 2)]:
        weights, points = element.get_quadrature(order, dtype)
        assert weights.dtype == points.dtype == dtype
        assert weights.shape == (count,) and points.shape == (count, dim)
        assert weights.device.type == points.device.type == "cpu"
    weights, points = Triangle.get_facet_quadrature(12, dtype=dtype)
    assert weights.shape == (3, 12) and points.shape == (3, 12, 2)
    torch.testing.assert_close(weights.sum(-1), torch.ones(3, dtype=dtype))


@pytest.mark.parametrize("device", ["mps", "cuda"])
def test_generated_accelerator_rules(device):
    available = torch.backends.mps.is_available() if device == "mps" else torch.cuda.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    weights, points = quadrature.tri_quadrature(20, torch.float32, torch.device(device))
    assert weights.device.type == points.device.type == device
    reference = quadrature.tri_quadrature(20, torch.float32)
    torch.testing.assert_close(weights.cpu(), reference[0])
    torch.testing.assert_close(points.cpu(), reference[1])


@pytest.mark.parametrize("order", [0, -1, True, 1.5])
@pytest.mark.parametrize("rule", [quadrature.lin_quadrature, quadrature.tri_quadrature])
def test_invalid_orders(rule, order):
    with pytest.raises(ValueError, match="integer >= 1"):
        rule(order)


def test_tetrahedron_limit_still_shape_specific():
    with pytest.raises(AssertionError, match="tet quadrature"):
        Tetrahedron.get_quadrature(8)
