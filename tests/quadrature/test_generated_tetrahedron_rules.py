"""Tetrahedral Gaussian exactness, rule lifecycle, and tabulated-rule audit."""

import hashlib
import math
import struct

import pytest
import torch

from tensormesh.element import Tetrahedron
from tensormesh.element import quadrature


def _assert_tetrahedron_moments(weights, points, degree, rtol=5e-13):
    # Factorial moments are independent of the Gaussian generator and its map.
    # Normalize relatively so tiny high-degree moments cannot pass on atol alone.
    powers = [
        (a, b, total - a - b)
        for total in range(degree + 1)
        for a in range(total + 1)
        for b in range(total - a + 1)
    ]
    for start in range(0, len(powers), 128):
        batch = powers[start:start + 128]
        exponents = torch.tensor(batch, device=points.device)
        monomials = (
            points[:, :1] ** exponents[:, 0]
            * points[:, 1:2] ** exponents[:, 1]
            * points[:, 2:] ** exponents[:, 2]
        )
        exact = torch.tensor(
            [math.factorial(a) * math.factorial(b) * math.factorial(c)
             / math.factorial(a + b + c + 3) for a, b, c in batch],
            dtype=points.dtype, device=points.device,
        )
        actual = torch.einsum("q,qi->i", weights, monomials)
        torch.testing.assert_close(actual / exact, torch.ones_like(exact), rtol=rtol, atol=0)


@pytest.mark.parametrize("degree", [7, 8, 9, 10, 11, 12, 16, 20, 21, 30])
def test_generated_tetrahedron_all_analytic_moments(degree):
    weights, points = Tetrahedron.get_quadrature(degree, dtype=torch.float64)
    n = math.ceil((degree + 1) / 2)
    assert weights.shape == (n ** 3,)
    assert points.shape == (n ** 3, 3)
    assert weights.dtype == points.dtype == torch.float64
    assert weights.device.type == points.device.type == "cpu"
    assert torch.isfinite(weights).all() and torch.isfinite(points).all()
    assert (weights > 0).all()
    assert (points > 0).all() and (points.sum(-1) < 1).all()
    torch.testing.assert_close(
        weights.sum(), torch.tensor(1 / 6, dtype=torch.float64), rtol=0, atol=3e-15
    )
    _assert_tetrahedron_moments(weights, points, degree)


# Exact little-endian float64 snapshots from prerequisite 93d96117c06e39e1.
# These protect every table entry and its ordering, not just volume/exactness.
# The former degree-6/7 tables are now correctly labelled degree 5/6.
_TABULATED_RULES = [
    (1, "e16c38822d28f113093ca57981727fc41af26fd1d00dc325796f7eced47d43d5"),
    (4, "0098da04a31975dc836c7a4d5afbe0d9d12a0aac9411f20066e9ed5ea104e91c"),
    (5, "9d1790148861e6ada3ccf72149b06216e4f9a1a8292dcd792621bc5048256465"),
    (11, "43e54045c4b33eca25191027233ee02f6a792a5181fa2a0e964f0ef89d6046e1"),
    (15, "902bcc9504af32858fdd22b45fe7f853ccdd17ac67ab0388094276fb99ceecfa"),
    (24, "0bd7ba144ff32b59f1d4e8158eda911c7f35dc62332036dc9d40ff0136d48cd4"),
]


@pytest.mark.parametrize("order", range(1, 7))
def test_tabulated_tetrahedron_rules_and_point_order(order):
    weights, points = Tetrahedron.get_quadrature(order, dtype=torch.float64)
    count, fingerprint = _TABULATED_RULES[order - 1]
    assert weights.shape == (count,) and points.shape == (count, 3)
    values = torch.cat((weights, points.flatten())).tolist()
    assert hashlib.sha256(struct.pack(f"<{len(values)}d", *values)).hexdigest() == fingerprint
    if order in (3, 4):
        assert weights[0] < 0
    if order == 5:
        assert (points == 0).any()  # the 15-point rule includes face points
    _assert_tetrahedron_moments(weights, points, order)


@pytest.mark.parametrize("order", range(1, 8))
def test_tetrahedron_advertised_exactness(order):
    weights, points = Tetrahedron.get_quadrature(order, dtype=torch.float64)
    _assert_tetrahedron_moments(weights, points, order)


def test_generated_tetrahedron_cache_reuse_and_mutation_isolation(monkeypatch):
    quadrature._gaussian_interval_rule.cache_clear()
    quadrature._gaussian_triangle_rule.cache_clear()
    quadrature._gaussian_tetrahedron_rule.cache_clear()
    calls = []
    original = quadrature.roots_jacobi

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(quadrature, "roots_jacobi", counted)
    weights, points = quadrature.tet_quadrature(20, dtype=torch.float64)
    reference = (weights.clone(), points.clone())
    assert len(calls) == 3
    weights.zero_()
    points.fill_(float("nan"))

    # Even/odd degrees share n=11; conversions must not expose cached storage.
    converted = quadrature.tet_quadrature(21, dtype=torch.float32)
    for actual, expected in zip(converted, reference):
        torch.testing.assert_close(actual, expected.float(), rtol=0, atol=0)
        actual.zero_()
    fresh = quadrature.tet_quadrature(20, dtype=torch.float64)
    for actual, expected in zip(fresh, reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert quadrature._gaussian_tetrahedron_rule.cache_info().hits == 2

    # Shared 1D rules also remain valid when used by other element shapes.
    tri_weights, _ = quadrature.tri_quadrature(20, dtype=torch.float64)
    line_weights, _ = quadrature.lin_quadrature(11, dtype=torch.float64)
    torch.testing.assert_close(tri_weights.sum(), torch.tensor(0.5, dtype=torch.float64))
    torch.testing.assert_close(line_weights.sum(), torch.tensor(1.0, dtype=torch.float64))
    assert len(calls) == 3


def test_generated_tetrahedron_canonical_cache_ignores_default_device_and_dtype():
    quadrature._gaussian_interval_rule.cache_clear()
    quadrature._gaussian_tetrahedron_rule.cache_clear()
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("meta"):
            canonical = quadrature._gaussian_tetrahedron_rule(6)
            returned = quadrature.tet_quadrature(10, dtype=torch.float64, device="cpu")
    finally:
        torch.set_default_dtype(original_dtype)
    for cached, copied in zip(canonical, returned):
        assert cached.device.type == copied.device.type == "cpu"
        assert cached.dtype == copied.dtype == torch.float64
        assert cached.data_ptr() != copied.data_ptr()
        torch.testing.assert_close(copied, cached, rtol=0, atol=0)
    _assert_tetrahedron_moments(*returned, degree=10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", [5, 6, 7, 10])
def test_tetrahedron_requested_cpu_dtype(dtype, degree):
    weights, points = Tetrahedron.get_quadrature(degree, dtype=dtype, device=torch.device("cpu"))
    assert weights.dtype == points.dtype == dtype
    assert weights.device.type == points.device.type == "cpu"
    _assert_tetrahedron_moments(
        weights, points, degree=degree, rtol=2e-6 if dtype == torch.float32 else 5e-13
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("degree", [5, 6, 7, 10])
def test_tetrahedron_cuda_conversion_and_moments(dtype, degree):
    reference = quadrature.tet_quadrature(degree, dtype=dtype)
    weights, points = quadrature.tet_quadrature(degree, dtype=dtype, device="cuda")
    assert weights.dtype == points.dtype == dtype
    assert weights.device.type == points.device.type == "cuda"
    for actual, expected in zip((weights, points), reference):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    _assert_tetrahedron_moments(
        weights, points, degree=degree, rtol=2e-6 if dtype == torch.float32 else 5e-13
    )
    weights.zero_()
    points.zero_()
    for actual, expected in zip(quadrature.tet_quadrature(degree, dtype=dtype), reference):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("order", [0, -1, True, False, 1.0, 8.0, 8.5, "8", None])
def test_tetrahedron_rejects_invalid_quadrature_degree(order):
    with pytest.raises(ValueError, match="tetrahedron quadrature order must be an integer >= 1"):
        quadrature.tet_quadrature(order)
