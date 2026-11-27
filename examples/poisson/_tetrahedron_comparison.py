"""Shared inputs and numerical oracles for the isolated upstream benchmark.

No TensorMesh imports here: workers must select their checkout first.
"""

import hashlib
import itertools
import math
from functools import lru_cache

import numpy as np
from scipy import sparse
from scipy.special import roots_jacobi


PINNED_UPSTREAM = "2d544e8bc7ee8de412ffbc768b313875c3b12895"
ORIENTATIONS = tuple(itertools.permutations(range(4)))
EDGES = ((2, 3), (1, 3), (1, 2), (0, 3), (0, 2), (0, 1))
FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))


def reference_lattice(p):
    """Integer barycentric nodes in TensorMesh's vertex/edge/face/cell order."""
    nodes = list(p * np.eye(4, dtype=np.int64))
    for a, b in EDGES:
        for i in range(1, p):
            w = np.zeros(4, dtype=np.int64)
            w[a], w[b] = p - i, i
            nodes.append(w)
    for a, b, c in FACES:
        for j in range(1, p - 1):
            for i in range(1, p - j):
                w = np.zeros(4, dtype=np.int64)
                w[a], w[b], w[c] = p - i - j, i, j
                nodes.append(w)
    for k in range(1, p - 2):
        for j in range(1, p - k - 1):
            for i in range(1, p - j - k):
                nodes.append(np.array([p - i - j - k, i, j, k]))
    result = np.asarray(nodes)
    assert result.shape == (math.comb(p + 3, 3), 4)
    return result


def high_order_cube(n, p, orientation="balanced"):
    """Conforming straight-sided mesh; deduplicate exact integer coordinates."""
    origins = np.stack(np.meshgrid(*([np.arange(n)] * 3), indexing="ij"), -1)
    paths = [np.concatenate([np.zeros((1, 3), dtype=np.int64),
                             np.eye(3, dtype=np.int64)[list(axes)].cumsum(0)])
             for axes in itertools.permutations(range(3))]
    corners = (origins.reshape(-1, 1, 1, 3) + np.asarray(paths)[None]).reshape(-1, 4, 3)
    ids = (corners[..., 0] * (n + 1) + corners[..., 1]) * (n + 1) + corners[..., 2]
    corners = np.take_along_axis(corners, ids.argsort(1)[..., None], axis=1)
    classes = np.arange(len(corners)) % 24 if orientation == "balanced" else np.zeros(
        len(corners), dtype=np.int64)
    slots = np.argsort(np.asarray(ORIENTATIONS), axis=1)[classes]
    corners = np.take_along_axis(corners, slots[..., None], axis=1)
    lattice = reference_lattice(p)
    local = np.einsum("bv,evd->ebd", lattice, corners)
    points, inverse = np.unique(local.reshape(-1, 3), axis=0, return_inverse=True)
    return {"points": points.astype(np.float64) / (n * p),
            "cells": inverse.reshape(len(corners), len(lattice)),
            "reference_nodes": lattice[:, 1:].astype(np.float64) / p}


def quadrature(p):
    """Positive conical Gauss-Jacobi rule, exact through degree 2*p+1."""
    rules = [roots_jacobi(p + 1, a, 0) for a in (2, 1, 0)]
    u, v, w = np.meshgrid(*[(x + 1) / 2 for x, _ in rules], indexing="ij")
    weights = [w / 2**(a + 1) for (_, w), a in zip(rules, (2, 1, 0))]
    return {"quadrature_points": np.stack([u, (1-u)*v, (1-u)*(1-v)*w], -1).reshape(-1, 3),
            "quadrature_weights": (weights[0][:, None, None] * weights[1][None, :, None]
                                   * weights[2][None, None, :]).reshape(-1)}


def array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def install_input_adapters(data):
    """Adapt quadrature and the lattice-size bug, leaving assembly kernels intact.

    Call once, in a fresh worker before any element caches are populated.
    The tetrahedral lattice fix uses the known integer side length instead of
    floor(cube_root(number_of_points)); all other node and basis code is upstream.
    """
    import torch
    from tensormesh.element import Tetrahedron
    from tensormesh.element import basis

    def integer_tet_linspace(tet, order=1, include_boundary=True):
        x = basis.hex_linspace(tet, order, include_boundary)
        n = order + 1 if include_boundary else order - 1
        k, j, i = torch.meshgrid(*([torch.arange(n, device=tet.device)] * 3), indexing="ij")
        mask = i + j + k < (n if include_boundary else n - 2)
        return x[..., mask.flatten(), :]

    basis.tet_linspace = integer_tet_linspace
    calls = set()

    @lru_cache(None)
    def external_quadrature(cls, order=1, dtype=torch.float32, device=torch.device("cpu")):
        calls.add(order)
        return (torch.tensor(data["quadrature_weights"], dtype=dtype, device=device),
                torch.tensor(data["quadrature_points"], dtype=dtype, device=device))

    Tetrahedron.get_quadrature = classmethod(external_quadrature)
    return calls


def transfer_matrix(asm, data, p):
    """Sparse T mapping global hierarchical coefficients to global nodal values.

    Shared rows are averaged and their trace agreement is explicitly checked.
    This is a verification operator; it is never used by timed assembly.
    """
    import torch
    from tensormesh.element import Tetrahedron

    kind = next(iter(asm.elements))
    hconn = asm.field_conn[f"u__{kind}"].detach().cpu().numpy()
    cells = data["cells"]
    q = torch.tensor(data["reference_nodes"], dtype=torch.float64)
    rows, cols, values = [], [], []
    for orientation in np.unique(cells[:, :4].argsort(1), axis=0):
        selected = np.flatnonzero((cells[:, :4].argsort(1) == orientation).all(1))
        h = Tetrahedron.get_hierarchical_basis_fns(
            p, torch.float64, orientation=tuple(map(int, orientation))).map(q).numpy()
        h[np.abs(h) < 1e-14] = 0
        a, b = np.nonzero(h)
        rows.append(cells[selected[:, None], a].ravel())
        cols.append(hconn[selected[:, None], b].ravel())
        values.append(np.broadcast_to(h[a, b], (len(selected), len(a))).ravel())
    rows, cols, values = map(np.concatenate, (rows, cols, values))
    counts = np.bincount(cells.ravel(), minlength=len(data["points"]))
    shape = (len(counts), asm.n_dofs)
    transfer = sparse.coo_matrix((values / counts[rows], (rows, cols)), shape=shape).tocsr()
    # Duplicate rows represent the same physical trace, including zeros.
    square_sum = sparse.coo_matrix((values**2, (rows, cols)), shape=shape).tocsr()
    variance = square_sum - sparse.diags(counts) @ transfer.multiply(transfer)
    error = float(np.max(np.abs(variance.data), initial=0))
    if not np.isfinite(error) or error > 2e-11:
        raise ValueError(f"nonconforming hierarchical traces: {error}")
    return transfer, error


def polynomial_probes(data, p):
    """High-degree physical monomials plus constants and linear coordinates."""
    exponents = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
                 (p, 0, 0), (0, p, 0), (0, 0, p), (p-2, 1, 1)]
    return exponents, np.stack([np.prod(data["points"]**e, axis=1) for e in exponents], 1)


def modal_probes(asm, data, p, nodal):
    """Interpolate verification polynomials cellwise, checking shared coefficients."""
    import torch
    from tensormesh.element import Tetrahedron

    kind = next(iter(asm.elements))
    conn = asm.field_conn[f"u__{kind}"].detach().cpu().numpy()
    cells = data["cells"]
    result = np.zeros((asm.n_dofs, nodal.shape[1]))
    q = torch.tensor(data["reference_nodes"], dtype=torch.float64)
    local = np.empty((len(cells), conn.shape[1], nodal.shape[1]))
    classes = cells[:, :4].argsort(1)
    for orientation in np.unique(classes, axis=0):
        selected = np.flatnonzero((classes == orientation).all(1))
        h = Tetrahedron.get_hierarchical_basis_fns(
            p, torch.float64, orientation=tuple(map(int, orientation))).map(q).numpy()
        local[selected] = np.linalg.solve(h, nodal[cells[selected]].transpose(1, 0, 2).reshape(
            len(h), -1)).reshape(len(h), len(selected), -1).transpose(1, 0, 2)
    result[conn] = local
    error = float(np.max(np.abs(result[conn] - local)))
    if not np.isfinite(error) or error > 2e-9:
        raise ValueError(f"inconsistent polynomial coefficients across cells: {error}")
    return result


def expected_integrals(exponents, form):
    def moment(e):
        return float(np.prod(1 / (np.asarray(e) + 1)))

    gram = np.zeros((len(exponents), len(exponents)))
    for i, a in enumerate(exponents):
        for j, b in enumerate(exponents):
            e = np.asarray(a) + b
            if form == "mass":
                gram[i, j] = moment(e)
            else:
                for d in range(3):
                    if a[d] and b[d]:
                        shift = np.eye(3, dtype=int)[d] * 2
                        gram[i, j] += a[d] * b[d] * moment(e - shift)
    return gram, np.array([moment(e) for e in exponents])


def scaled_error(actual, expected):
    """Maximum absolute error scaled by max(1, max absolute reference value)."""
    if sparse.issparse(expected):
        delta = actual - expected
        error = np.max(np.abs(delta.data), initial=0)
        scale = np.max(np.abs(expected.data), initial=0)
    else:
        error = np.max(np.abs(actual - expected), initial=0)
        scale = np.max(np.abs(expected), initial=0)
    return float(error / max(1., scale))
