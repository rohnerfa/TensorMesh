"""Small, reproducible Lagrange/hierarchical version of poisson.py.

Run from the TensorMesh repository, for example:
    python examples/poisson/compare_hierarchical.py --degree 3 --length 0.2

The upstream LaplaceElementAssembler + NodeAssembler path is retained as
the reference solve. The modal solve uses MixedElementAssembler. Both use
the SAME mesh-order-interpolated source, as in the original example, not
different approximations of the analytic forcing. Timings are single-run
diagnostics, not performance benchmarks.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tensormesh import (Condenser, Field, LaplaceElementAssembler, Mesh,
                        MixedElementAssembler, NodeAssembler)
from tensormesh.dataset import PoissonMultiFrequency
from tensormesh.element import Triangle


def relative_norm(difference, reference):
    return float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference).clamp_min(1e-30))


def solve(matrix, rhs, boundary):
    condenser = Condenser(boundary)
    reduced, load = condenser(matrix, rhs)
    start = perf_counter()
    free = reduced.solve(load)
    seconds = perf_counter() - start
    residual = relative_norm(reduced @ free - load, load)
    return condenser.recover(free), residual, seconds


def run_comparison(degree=3, length=0.2, modes=16, seed=17, quadrature_degree=None):
    if degree not in (1, 2, 3, 4):
        raise ValueError("use degree 1..4 for a trustworthy low-degree nodal comparison")
    if length <= 0 or modes < 1:
        raise ValueError("length and modes must be positive")
    qdegree = 2 * degree if quadrature_degree is None else quadrature_degree
    start = perf_counter()
    mesh = Mesh.gen_rectangle(chara_length=length, order=degree, element_type="tri").double().cpu()
    mesh_seconds = perf_counter() - start
    kind = mesh.default_element_type
    cells = mesh.cells[kind]
    a = torch.empty(modes, modes, dtype=torch.float64).uniform_(
        -1, 1, generator=torch.Generator().manual_seed(seed))
    equation = PoissonMultiFrequency(a=a)
    source = equation.source_term(mesh.points, domain="rectangle")
    checks = torch.tensor([[0.23, 0.37], [0.59, 0.41], [0.72, 0.68]], dtype=torch.float64)
    hessians = torch.func.vmap(torch.func.hessian(
        lambda point: equation.solution(point[None])[0]))(checks)
    analytic_forcing = -hessians.diagonal(dim1=-2, dim2=-1).sum(-1)
    analytic_consistency = relative_norm(analytic_forcing - equation.source_term(checks),
                                        equation.source_term(checks))

    class Load(NodeAssembler):
        def forward(self, v, f):
            return v * f

    # This is the original example's assembly and source-interpolation path.
    start = perf_counter()
    upstream = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=qdegree)
    upstream_load = Load.from_mesh(mesh, quadrature_order=qdegree)
    upstream_setup = perf_counter() - start
    start = perf_counter()
    kl = upstream(mesh.points)
    bl = upstream_load(mesh.points, point_data={"f": source})
    upstream_assembly = perf_counter() - start
    ul, residual_l, solve_l = solve(kl, bl, mesh.boundary_mask)

    class HierarchicalPoisson(MixedElementAssembler):
        fields = [Field("u", "v", degree, basis="hierarchical")]

        def forward(self, gradu, gradv):
            return (gradu * gradv).sum()

        def forward_vector(self, v, f):
            return v * f

    start = perf_counter()
    modal = HierarchicalPoisson.from_mesh(mesh, quadrature_order=qdegree)
    modal_setup = perf_counter() - start
    start = perf_counter()
    kh = modal()
    bh = modal.assemble_vector(point_data={"f": source})
    modal_assembly = perf_counter() - start
    uh, residual_h, solve_h = solve(kh, bh, modal.layout.boundary_mask("u"))

    # Explicit global coefficient change of basis, solely for verification.
    # Averaging repeated nodal rows handles shared vertices/edges.
    ref_nodes = Triangle.get_basis(degree, torch.float64)
    values = Triangle.get_hierarchical_basis_fns(degree, torch.float64).map(ref_nodes)
    conn = modal.field_conn[f"u__{kind}"]
    signs = modal.field_sign[f"u__{kind}"]
    transform = torch.zeros(mesh.n_points, modal.n_dofs, dtype=torch.float64)
    rows = cells[:, :, None].expand(-1, -1, conn.shape[1])
    cols = conn[:, None, :].expand(-1, cells.shape[1], -1)
    local = values[None] * signs[:, None, :]
    transform.index_put_((rows.flatten(), cols.flatten()), local.flatten(), accumulate=True)
    incidence = torch.bincount(cells.flatten(), minlength=mesh.n_points)
    transform = transform / incidence.clamp_min(1)[:, None]
    expected_kh = transform.T @ kl.to_dense() @ transform
    expected_bh = transform.T @ bl

    # Independent, higher-degree integration of physical fields and errors.
    w, ref = Triangle.get_quadrature(max(40, qdegree + 8), torch.float64)
    geom_values = Triangle.get_basis_fns(degree, torch.float64).map(ref)
    geom_grad = Triangle.get_basis_grad_fns(degree, torch.float64).map(ref)
    coords = mesh.points[cells]
    physical = torch.einsum("qb,ebd->eqd", geom_values, coords)
    jac = torch.einsum("qib,ebj->eqij", geom_grad, coords)
    jxw = torch.linalg.det(jac).abs() * w
    field_l = torch.einsum("qb,eb->eq", geom_values, ul[cells])
    field_h = modal.layout.evaluate("u", uh, ref)
    analytical = equation.solution(physical.reshape(-1, 2)).reshape_as(field_l)
    def l2(values):
        return float(torch.sqrt((jxw * values.square()).sum()))
    norm_exact, norm_l = l2(analytical), l2(field_l)
    hash_input = mesh.points.numpy().tobytes() + cells.numpy().tobytes()
    report = {
        "example": "examples/poisson/poisson.py",
        "configuration": {"degree": degree, "geometry_order": degree, "characteristic_length": length,
                          "modes": modes, "seed": seed, "quadrature_degree": qdegree,
                          "source": "same nodal mesh-order interpolation in both paths",
                          "dtype": "float64", "device": "cpu", "torch_threads": torch.get_num_threads(),
                          "torch_version": torch.__version__, "coefficients": a.tolist()},
        "mesh": {"triangles": len(cells), "points": mesh.n_points, "scalar_dofs_per_basis": modal.n_dofs,
                 "sha256": hashlib.sha256(hash_input).hexdigest()},
        "agreement": {
            "relative_matrix_change_of_basis_error": relative_norm(kh.to_dense() - expected_kh, expected_kh),
            "relative_load_change_of_basis_error": relative_norm(bh - expected_bh, expected_bh),
            "relative_L2_field_difference": l2(field_h - field_l) / max(norm_l, 1e-30),
            "max_mesh_point_difference": float((modal.layout.prolong("u", uh) - ul).abs().max()),
            "boundary_dof_counts_equal": int(mesh.boundary_mask.sum()) == int(modal.layout.boundary_mask("u").sum()),
        },
        "analytic_PDE_relative_residual_at_check_points": analytic_consistency,
        "lagrange": {"relative_L2_error_vs_analytic": l2(field_l - analytical) / norm_exact,
                     "relative_free_system_residual": residual_l},
        "hierarchical": {"relative_L2_error_vs_analytic": l2(field_h - analytical) / norm_exact,
                         "relative_free_system_residual": residual_h},
        "single_run_seconds_not_benchmarks": {
            "mesh": mesh_seconds,
            "lagrange": {"setup": upstream_setup, "matrix_and_load_assembly": upstream_assembly, "solve": solve_l},
            "hierarchical": {"setup": modal_setup, "matrix_and_load_assembly": modal_assembly, "solve": solve_h}},
    }
    if not all(report["agreement"][key] < 1e-9 for key in (
            "relative_matrix_change_of_basis_error", "relative_load_change_of_basis_error",
            "relative_L2_field_difference")) or not report["agreement"]["boundary_dof_counts_equal"] \
            or analytic_consistency > 1e-10:
        raise RuntimeError(f"basis comparison failed: {report['agreement']}")
    return report, (mesh, modal, ul, uh, equation)


def plot_comparison(data, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    mesh, modal, ul, uh, equation = data
    ref = Triangle.get_basis(12, torch.float64)
    local_triangles = torch.as_tensor(mtri.Triangulation(ref[:, 0], ref[:, 1]).triangles.copy())
    kind = mesh.default_element_type
    cells = mesh.cells[kind]
    p = modal.layout.field("u").order
    basis = Triangle.get_basis_fns(p, torch.float64).map(ref)
    xy = torch.einsum("qb,ebd->eqd", basis, mesh.points[cells]).reshape(-1, 2)
    triangles = (local_triangles[None] + torch.arange(len(cells))[:, None, None] * len(ref)).reshape(-1, 3)
    triangulation = mtri.Triangulation(xy[:, 0], xy[:, 1], triangles)
    values_l = torch.einsum("qb,eb->eq", basis, ul[cells]).reshape(-1)
    values_h = modal.layout.evaluate("u", uh, ref).reshape(-1)
    exact = equation.solution(xy)
    lo, hi = float(torch.stack((values_l, values_h, exact)).min()), float(torch.stack((values_l, values_h, exact)).max())
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.7), constrained_layout=True)
    for ax, title, values in zip(axes, ["Analytic", "Original Lagrange", "Hierarchical", "Hierarchical − Lagrange"],
                                 [exact, values_l, values_h, values_h - values_l]):
        kw = {"vmin": lo, "vmax": hi} if ax is not axes[-1] else {}
        artist = ax.tripcolor(triangulation, values.numpy(), shading="gouraud", **kw)
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(artist, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--degree", type=int, default=3)
    parser.add_argument("--length", type=float, default=0.2)
    parser.add_argument("--modes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--quadrature-degree", type=int)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    report, data = run_comparison(args.degree, args.length, args.modes, args.seed, args.quadrature_degree)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    (args.output_dir / "comparison.json").write_text(text + "\n")
    plot_comparison(data, args.output_dir / "comparison.png")
    # Keep reproducibility data in JSON without flooding terminal output.
    summary = dict(report)
    summary["configuration"] = {k: v for k, v in report["configuration"].items()
                                if k != "coefficients"}
    print(json.dumps(summary, indent=2))
    print(f"Results and plot: {args.output_dir}")


if __name__ == "__main__":
    main()
