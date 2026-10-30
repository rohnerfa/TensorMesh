"""Manufactured 3D Poisson problem with hierarchical fields and nonzero traces.

Run from the repository: python examples/poisson/hierarchical_tetrahedron.py --degree 4
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import meshio
import torch

from tensormesh import Condenser, Field, Mesh, MixedElementAssembler
from tensormesh.element import Tetrahedron


def run(degree=4, device="cpu"):
    """Solve on a four-cell tetrahedral patch and return numerical diagnostics.

    Parameters
    ----------
    degree : int, optional
        Uniform hierarchical field degree, at least one.
    device : str, optional
        PyTorch device; defaults to CPU. Computations use float64.

    Returns
    -------
    dict
        Problem dimensions, free-system residual and independent value/gradient
        errors at degree-``2*degree+4`` quadrature points.
    """
    if isinstance(degree, bool) or not isinstance(degree, int) or degree < 1:
        raise ValueError("degree must be an integer >= 1")
    mesh = Mesh(meshio.Mesh(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [.25, .25, .25]],
        [("tetra", [[4, 1, 3, 2], [2, 4, 3, 0], [1, 3, 0, 4], [0, 2, 4, 1]])]
    )).double().to(device)

    def exact(x):
        return (1 + .4 * x[..., 0] ** degree + .3 * x[..., 1] ** degree
                - .2 * x[..., 2] ** degree + .1 * x.sum(-1) ** degree)

    class Poisson(MixedElementAssembler):
        fields = [Field("u", "v", degree, basis="hierarchical")]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, x):
            if degree == 1:
                return v * 0
            # Analytic -Laplacian of exact(), including all three derivatives
            # of the (x+y+z)**degree term.
            forcing = -degree * (degree - 1) * (
                .4 * x[0] ** (degree - 2) + .3 * x[1] ** (degree - 2)
                - .2 * x[2] ** (degree - 2) + .3 * x.sum() ** (degree - 2))
            return forcing * v

    quadrature_degree = max(8, 2 * degree)
    asm = Poisson.from_mesh(mesh, quadrature_order=quadrature_degree)
    layout = asm.layout
    boundary = layout.boundary_mask("u")
    lifting = layout.dirichlet_values("u", exact)
    condenser = Condenser(boundary, lifting)
    matrix, rhs = condenser(asm(), asm.assemble_vector())
    # A dense solve keeps this deliberately small example backend-independent.
    free = torch.linalg.solve(matrix.to_dense(), rhs)
    solution = condenser.recover(free)
    _, reference = Tetrahedron.get_quadrature(max(10, 2 * degree + 4),
                                              torch.float64, torch.device(device))
    geometry = Tetrahedron.get_basis_fns(1, torch.float64, torch.device(device)).map(reference)
    physical = torch.einsum("qb,ebd->eqd", geometry, asm._points[asm.elements["tetra"]])
    values = layout.evaluate("u", solution, reference)
    gradient = layout.evaluate("u", solution, physical, physical=True, gradient=True)
    expected_gradient = degree * (
        physical.new_tensor([.4, .3, -.2]) * physical ** (degree - 1)
        + .1 * physical.sum(-1, keepdim=True) ** (degree - 1))
    return {
        "degree": degree, "geometry_order": 1, "cells": 4,
        "device": str(device), "dtype": "float64",
        "quadrature_degree": quadrature_degree,
        "scalar_dofs": layout.n_dofs, "boundary_dofs": int(boundary.sum()),
        "max_value_error": float((values - exact(physical)).abs().max()),
        "max_gradient_error": float((gradient - expected_gradient).abs().max()),
        "relative_free_residual": float(torch.linalg.vector_norm(matrix @ free - rhs)
                                        / torch.linalg.vector_norm(rhs).clamp_min(1e-30)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.degree, args.device), indent=2))


if __name__ == "__main__":
    main()
