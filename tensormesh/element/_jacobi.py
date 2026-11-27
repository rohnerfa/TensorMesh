"""Homogenized Jacobi recurrences on value/reference-gradient pairs.

The gradient can have any spatial dimension. No division by the homogenizing
coordinate is used, so collapsed simplex coordinates are valid inputs.
"""

import torch


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _scale(c, a):
    return c * a[0], c * a[1]


def _mul(a, b):
    return a[0] * b[0], a[1] * b[0][..., None] + a[0][..., None] * b[1]


def _jacobi(degree, alpha, beta, x, t):
    """Return value/gradient pairs of t**n P_n^(alpha,beta)(x/t)."""
    result = [(torch.ones_like(x[0]), torch.zeros_like(x[1]))]
    if degree == 0:
        return result
    result.append(_scale(0.5, _add(_scale(alpha - beta, t),
                                   _scale(alpha + beta + 2, x))))
    t2 = _mul(t, t)
    for n in range(2, degree + 1):
        s = 2 * n + alpha + beta
        linear = _add(_scale(s * (s - 2), x),
                      _scale(alpha * alpha - beta * beta, t))
        first = _scale(s - 1, _mul(linear, result[-1]))
        second = _scale(-2 * (n + alpha - 1) * (n + beta - 1) * s,
                         _mul(t2, result[-2]))
        result.append(_scale(1 / (2 * n * (n + alpha + beta) * (s - 2)),
                             _add(first, second)))
    return result
