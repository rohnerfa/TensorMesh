"""Vertex-oriented hierarchical H1 tetrahedra, using polynomial recurrences."""

from itertools import permutations

import torch
from torch import nn

from ._jacobi import _add, _jacobi, _mul, _scale


TETRAHEDRON_ORIENTATIONS = tuple(permutations(range(4)))
"""Lexicographic class ID -> local slots in ascending global vertex-ID order."""


class HierarchicalTetrahedronBasisFns(nn.Module):
    r"""Hierarchical basis of P_p on the reference tetrahedron, p >= 1.

    Parameters
    ----------
    order : int, optional
        Polynomial degree, at least 1. Default 1.
    dtype : torch.dtype, optional
        Tabulation dtype. Default torch.float32.
    device : torch.device, optional
        Tabulation device. Default CPU.
    is_grad : bool, optional
        Internal gradient-evaluator flag. Prefer :meth:`grad`. Default False.
    orientation : tuple of int, optional
        Keyword-only permutation of local slots ``(0, 1, 2, 3)`` that sorts
        the cell's global corner IDs. Default is ascending local slot order.
        Class IDs index :data:`TETRAHEDRON_ORIENTATIONS` lexicographically.

    Notes
    -----
    Barycentric coordinates are ``(1-x-y-z, x, y, z)``. Local blocks are
    vertices, edges ``(2,3), (1,3), (1,2), (0,3), (0,2), (0,1)``, faces
    ``(1,2,3), (0,2,3), (0,1,3), (0,1,2)``, then private cell bubbles.
    Orientation sorts coordinate arguments within each edge/face block;
    it never reorders blocks, vertices, cell bubbles, or reference points.
    Gather global coefficients directly: no additional orientation signs.

    With :math:`H_n^{a,b}(s,t)=t^nP_n^{(a,b)}(s/t)`, edge modes are
    :math:`-2\lambda_a\lambda_b H_{n-2}^{1,1}/(n-1)`, n=2,...,p.
    Face modes are :math:`\lambda_a\lambda_b\lambda_c
    H_i^{1,1}(\lambda_b-\lambda_a,\lambda_a+\lambda_b)
    H_j^{2i+3,1}(2\lambda_c-s_f,s_f)`, with
    :math:`s_f=\lambda_a+\lambda_b+\lambda_c`.
    They run by total i+j=0,...,p-3, then ascending i (j=total-i).
    Cell modes use the fixed local order 0,1,2,3, the same first two
    Jacobi factors, and :math:`H_k^{2(i+j)+5,1}(2\lambda_3-1,1)` times
    all four barycentric coordinates. They run by i+j+k=0,...,p-4,
    then ascending i, then ascending j (k=total-i-j).

    Homogenized recurrences give finite values and analytic gradients even
    when a collapsed coordinate vanishes. Mode identity is nested with p;
    the full basis array is not prefix-nested. See the user guide for the
    complete formulas and the canonical triangular-face convention.
    """

    edges = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))
    faces = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))
    n_vars = 3

    def __init__(self, order=1, dtype=torch.float32, device=torch.device("cpu"),
                 is_grad=False, *, orientation=(0, 1, 2, 3)):
        super().__init__()
        if isinstance(order, bool) or not isinstance(order, int) or order < 1:
            raise ValueError("tetrahedron order must be an integer >= 1")
        if (not isinstance(orientation, tuple) or len(orientation) != 4
                or any(isinstance(i, bool) or not isinstance(i, int) for i in orientation)
                or sorted(orientation) != [0, 1, 2, 3]):
            raise ValueError("orientation must be a tuple permuting (0, 1, 2, 3)")
        self.order = order
        self.is_grad = is_grad
        self.orientation = orientation
        rank = {slot: i for i, slot in enumerate(orientation)}
        self._edges = tuple(tuple(sorted(edge, key=rank.__getitem__)) for edge in self.edges)
        self._faces = tuple(tuple(sorted(face, key=rank.__getitem__)) for face in self.faces)
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype, device=device))

    @property
    def shape(self):
        """tuple of int: Output axes excluding the input point batch axes."""
        n = (self.order + 1) * (self.order + 2) * (self.order + 3) // 6
        return (3, n) if self.is_grad else (n,)

    def grad(self):
        """Return the analytic reference-gradient evaluator.

        Returns
        -------
        HierarchicalTetrahedronBasisFns
            Same orientation/dtype/device, with ``map`` shape ``[..., 3, B]``.

        Notes
        -----
        Second derivatives are not implemented.
        """
        if self.is_grad:
            raise NotImplementedError("second derivatives are not implemented")
        return type(self)(self.order, self._anchor.dtype, self._anchor.device, True,
                          orientation=self.orientation)

    def map(self, points: torch.Tensor) -> torch.Tensor:
        """Tabulate oriented values or reference gradients at arbitrary points.

        Parameters
        ----------
        points : torch.Tensor
            Reference coordinates ``[..., 3]``, converted to the evaluator's
            dtype/device. Boundary points and empty batches are supported.

        Returns
        -------
        torch.Tensor
            Values ``[..., B]`` or gradients ``[..., 3, B]``, where
            ``B = (order+1)*(order+2)*(order+3)//6``.
        """
        if points.ndim < 1 or points.shape[-1] != 3:
            raise ValueError("reference points must have shape [..., 3]")
        points = points.to(self._anchor)
        x, y, z = points.unbind(-1)
        zero, one = torch.zeros_like(x), torch.ones_like(x)
        lam = [(1 - x - y - z, torch.stack((-one, -one, -one), -1)),
               (x, torch.stack((one, zero, zero), -1)),
               (y, torch.stack((zero, one, zero), -1)),
               (z, torch.stack((zero, zero, one), -1))]
        constant = (one, torch.zeros_like(lam[0][1]))
        functions = list(lam)
        if self.order >= 2:
            for a, b in self._edges:
                s = _add(lam[a], lam[b])
                d = _add(lam[b], _scale(-1, lam[a]))
                modes = _jacobi(self.order - 2, 1, 1, d, s)
                bubble = _mul(lam[a], lam[b])
                functions.extend(_scale(-2 / (k + 1), _mul(bubble, mode))
                                 for k, mode in enumerate(modes))
        if self.order >= 3:
            for a, b, c in self._faces:
                s = _add(lam[a], lam[b])
                sf = _add(s, lam[c])
                d = _add(lam[b], _scale(-1, lam[a]))
                first = _jacobi(self.order - 3, 1, 1, d, s)
                zf = _add(_scale(2, lam[c]), _scale(-1, sf))
                second = [_jacobi(self.order - 3 - i, 2 * i + 3, 1, zf, sf)
                          for i in range(self.order - 2)]
                bubble = _mul(_mul(lam[a], lam[b]), lam[c])
                for total in range(self.order - 2):
                    for i in range(total + 1):
                        functions.append(_mul(bubble, _mul(first[i], second[i][total - i])))
        if self.order >= 4:
            s = _add(lam[0], lam[1])
            sf = _add(s, lam[2])
            first = _jacobi(self.order - 4, 1, 1,
                            _add(lam[1], _scale(-1, lam[0])), s)
            zf = _add(_scale(2, lam[2]), _scale(-1, sf))
            second = [_jacobi(self.order - 4 - i, 2 * i + 3, 1, zf, sf)
                      for i in range(self.order - 3)]
            zc = _add(_scale(2, lam[3]), _scale(-1, constant))
            third = [_jacobi(self.order - 4 - total, 2 * total + 5, 1, zc, constant)
                     for total in range(self.order - 3)]
            bubble = _mul(_mul(lam[0], lam[1]), _mul(lam[2], lam[3]))
            for total in range(self.order - 3):
                for i in range(total + 1):
                    for j in range(total - i + 1):
                        mode = _mul(_mul(first[i], second[i][j]), third[i + j][total - i - j])
                        functions.append(_mul(bubble, mode))
        return torch.stack([f[1 if self.is_grad else 0] for f in functions], -1)
