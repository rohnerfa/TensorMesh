"""Recurrence-based, non-nodal hierarchical H1 triangle functions.

No collapsed-coordinate divisions are used, including at vertices.
"""

import torch
from torch import nn


def _add(a, b):
    return a[0] + b[0], a[1] + b[1]


def _scale(c, a):
    return c * a[0], c * a[1]


def _mul(a, b):
    return a[0] * b[0], a[1] * b[0][..., None] + a[0][..., None] * b[1]


def _jacobi(degree, alpha, beta, x, t):
    """Value/gradient pairs of t**n P_n^(alpha,beta)(x/t).

    Homogenizing the three-term Jacobi recurrence avoids division by t.
    """
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


class HierarchicalTriangleBasisFns(nn.Module):
    r"""Hierarchical basis of P_p on (0,0), (1,0), (0,1), p >= 1.

    Ordering is vertices lambda_0, lambda_1, lambda_2; edge blocks in
    TensorMesh's Triangle.edge order (1,2), (0,2), (0,1); then bubbles.
    Each edge (a,b) has modes n=2,...,p given by

    .. math::

        -\frac{2\lambda_a\lambda_b}{n-1}
        (\lambda_a+\lambda_b)^{n-2}
        P_{n-2}^{(1,1)}\left(\frac{\lambda_b-\lambda_a}
                                      {\lambda_a+\lambda_b}\right).

    These are scaled integrated Legendre functions: on the edge their
    derivative in z=lambda_b-lambda_a is P_{n-1}(z). Reversing endpoints
    multiplies mode n by (-1)**n, WITHOUT reversing mode indices.

    Bubbles are lambda_0 lambda_1 lambda_2 times
    scaled P_i^(1,1)(lambda_1-lambda_0, lambda_0+lambda_1) times
    P_j^(2i+3,1)(2 lambda_2-1), ordered by i+j=0,...,p-3,
    then i=0,...,i+j. Existing functions are unchanged as p increases,
    though edge blocks moving means nesting is not a prefix relation.

    map(points) returns [..., N_b]; grad().map(points) returns
    [..., 2, N_b], matching Polynomials. Coefficients are NOT nodal
    values. This evaluator does not change geometry or assembly defaults.

    Parameters
    ----------
    order : int, optional
        Polynomial degree, at least 1. Default 1.
    dtype : torch.dtype, optional
        Tabulation dtype. Default torch.float32.
    device : torch.device, optional
        Tabulation device. Default CPU.
    is_grad : bool, optional
        Internal gradient-evaluator flag. Prefer :meth:`grad` to obtain
        reference gradients. Default False.
    """

    edges = ((1, 2), (0, 2), (0, 1))
    n_vars = 2

    def __init__(self, order=1, dtype=torch.float32, device=torch.device("cpu"),
                 is_grad=False):
        super().__init__()
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("triangle order must be an integer >= 1")
        self.order = order
        self.is_grad = is_grad
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype, device=device))

    @property
    def shape(self):
        n = (self.order + 1) * (self.order + 2) // 2
        return (2, n) if self.is_grad else (n,)

    def grad(self):
        """Return the reference-gradient evaluator.

        Returns
        -------
        HierarchicalTriangleBasisFns
            Evaluator whose ``map`` returns ``[..., 2, N_b]``.

        Notes
        -----
        Second derivatives are not implemented.
        """
        if self.is_grad:
            raise NotImplementedError("second derivatives are not implemented")
        return type(self)(self.order, self._anchor.dtype, self._anchor.device, True)

    def map(self, points):
        """Tabulate all functions or reference gradients.

        Parameters
        ----------
        points : torch.Tensor
            Reference coordinates of shape ``[..., 2]``. Converted to the
            evaluator's dtype/device before tabulation.

        Returns
        -------
        torch.Tensor
            Values ``[..., N_b]`` or gradients ``[..., 2, N_b]`` for a
            gradient evaluator.
        """
        if points.ndim < 1 or points.shape[-1] != 2:
            raise ValueError("reference points must have shape [..., 2]")
        points = points.to(self._anchor)
        x, y = points.unbind(-1)
        zero, one = torch.zeros_like(x), torch.ones_like(x)
        lam = [(1 - x - y, torch.stack((-one, -one), -1)),
               (x, torch.stack((one, zero), -1)),
               (y, torch.stack((zero, one), -1))]
        constant = (one, torch.zeros_like(lam[0][1]))
        functions = list(lam)
        if self.order >= 2:
            for a, b in self.edges:
                s = _add(lam[a], lam[b])
                d = _add(lam[b], _scale(-1, lam[a]))
                modes = _jacobi(self.order - 2, 1, 1, d, s)
                bubble = _mul(lam[a], lam[b])
                functions.extend(_scale(-2 / (k + 1), _mul(bubble, mode))
                                 for k, mode in enumerate(modes))
        if self.order >= 3:
            s = _add(lam[0], lam[1])
            d = _add(lam[1], _scale(-1, lam[0]))
            first = _jacobi(self.order - 3, 1, 1, d, s)
            z = _add(_scale(2, lam[2]), _scale(-1, constant))
            second = [_jacobi(self.order - 3 - i, 2 * i + 3, 1, z, constant)
                      for i in range(self.order - 2)]
            bubble = _mul(_mul(lam[0], lam[1]), lam[2])
            for total in range(self.order - 2):
                for i in range(total + 1):
                    functions.append(_mul(bubble, _mul(first[i], second[i][total - i])))
        if self.is_grad:
            return torch.stack([f[1] for f in functions], -1)
        return torch.stack([f[0] for f in functions], -1)
