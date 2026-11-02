"""Shared reference products: independent integration, ragged packing and AD."""

import pytest
import torch

from tensormesh import Field, MixedElementAssembler
from tests.assemble._tetrahedron_fixtures import (
    DTYPE, cell_tables, field_indices, independent_quadrature, orientation_mesh,
    patch_mesh, poisson, scalar_reference, scatter_matrix,
)


@pytest.mark.parametrize("classes", [list(range(24)) * 3,
                                    [17] * 19 + [0] * 3 + [9] * 2 + [5], [7] * 5])
def test_shared_mass_and_unit_load_with_uneven_populations_and_cell_chunks(classes):
    asm = poisson(4).from_mesh(orientation_mesh(classes))
    mass, _, _ = scalar_reference(asm)
    constant = torch.zeros(asm.n_dofs, dtype=DTYPE)
    constant[asm.field_conn["u__tetra"][:, :4]] = 1
    for qchunk, echunk in [(-1, None), (17, 11), (31, 1)]:
        kwargs = {"batch_size": qchunk, "element_batch_size": echunk}
        torch.testing.assert_close(asm(func=lambda u, v: u * v, **kwargs).to_dense(), mass,
                                   atol=5e-13, rtol=3e-11)
        torch.testing.assert_close(asm.assemble_vector(func=lambda v: v, **kwargs),
                                   mass @ constant, atol=5e-13, rtol=3e-11)
        plans = asm.grouping.reference_chunks("tetra", echunk)
        assert all(p.n_cells <= p.padded_cells < 2 * p.n_cells for p in plans)


def test_products_are_evaluated_per_orientation_and_contracted_together(monkeypatch):
    asm = poisson(4).from_mesh(orientation_mesh(list(range(24)) * 3))
    original_pass, original_bmm = asm._run_pass, torch.bmm
    integrands, contractions = [], []

    def counted_pass(*args):
        result, has_batch = original_pass(*args)
        integrands.append(result.shape)
        return result, has_batch

    def counted_bmm(weights, products, *args, **kwargs):
        contractions.append((weights.shape, products.shape))
        return original_bmm(weights, products, *args, **kwargs)

    monkeypatch.setattr(asm, "_run_pass", counted_pass)
    monkeypatch.setattr(torch, "bmm", counted_bmm)
    shared = asm(func=lambda u, v: u * v)
    nq = asm.transformation["tetra"].n_quadrature
    # 72 cells, 24 reference integrands, ONE batched contraction (not 24 calls).
    assert integrands == [torch.Size([24, nq, 35, 35, 1, 1])]
    assert contractions == [(torch.Size([24, 4, nq]), torch.Size([24, nq, 1225]))]
    integrands.clear()
    cellwise = asm(func=lambda u, v: u * v, reference_mode="cell")
    assert integrands == [torch.Size([72, nq, 35, 35, 1, 1])]
    torch.testing.assert_close(shared.values, cellwise.values, atol=5e-13, rtol=3e-11)


def test_shared_rectangular_vector_nodal_blocks_on_curved_geometry():
    class Mass(MixedElementAssembler):
        fields = [Field("u", "v", 4, components=2, basis="hierarchical"), Field("p", "q", 2)]

        def forward(self, u, v, p, q, scale):
            return scale * ((u * v).sum() + 2 * p * q + 3 * p * v[0] + 5 * u[1] * q)

        def forward_vector(self, v, q, scale):
            return scale * (v.sum() + 2 * q)

    asm = Mass.from_mesh(patch_mesh(2, curved=True), quadrature_order=12)
    matrix = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=DTYPE)
    load = torch.zeros(asm.n_dofs, dtype=DTYPE)
    weights, reference = independent_quadrature(12)
    for cell in range(2):
        _, w, tables, _ = cell_tables(asm, cell, reference, weights)
        u, p = tables["u"][0], tables["p"][0]
        iu, ip = field_indices(asm, "u", cell), field_indices(asm, "p", cell)
        uu = torch.einsum("qi,qj,q->ij", u, u, w)
        pp = 2 * torch.einsum("qi,qj,q->ij", p, p, w)
        up = torch.zeros(len(iu), len(ip), dtype=DTYPE)
        up[::2] = 3 * torch.einsum("qi,qj,q->ij", u, p, w)
        pu = torch.zeros(len(ip), len(iu), dtype=DTYPE)
        pu[:, 1::2] = 5 * torch.einsum("qi,qj,q->ij", p, u, w)
        scatter_matrix(matrix, iu, iu, torch.kron(uu.contiguous(), torch.eye(2, dtype=DTYPE)))
        scatter_matrix(matrix, ip, ip, pp)
        scatter_matrix(matrix, iu, ip, up)
        scatter_matrix(matrix, ip, iu, pu)
        load[iu] += (u.T @ w)[:, None].expand(-1, 2).reshape(-1)
        load[ip] += 2 * p.T @ w
    for scale in (1.0, 2.0):
        inputs = {"scalar_data": {"scale": scale}, "batch_size": 23}
        torch.testing.assert_close(asm(**inputs).to_dense(), scale * matrix,
                                   atol=5e-13, rtol=3e-11)
        torch.testing.assert_close(asm.assemble_vector(**inputs), scale * load,
                                   atol=5e-13, rtol=3e-11)


def test_shared_geometry_and_parameter_gradients_and_reassembly():
    class Mass(MixedElementAssembler):
        fields = [Field("u", "v", 4, basis="hierarchical")]

        def __post_init__(self):
            self.scale = torch.nn.Parameter(torch.tensor(1.25, dtype=DTYPE))

        def forward(self, u, v):
            return self.scale * u * v

        def forward_vector(self, v):
            return self.scale * v

    asm = Mass.from_mesh(patch_mesh(2, curved=True), quadrature_order=12)
    points = asm._points.detach().clone().requires_grad_()
    direction = torch.randn(points.shape, dtype=DTYPE, generator=torch.Generator().manual_seed(89))

    def objective(x, mode="auto"):
        return (asm(points=x, batch_size=19, reference_mode=mode).values.square().sum()
                + asm.assemble_vector(points=x, batch_size=19, reference_mode=mode).square().sum())

    for _ in range(2):
        value = objective(points)
        dx, ds = torch.autograd.grad(value, (points, asm.scale))
        ref_dx, ref_ds = torch.autograd.grad(objective(points, "cell"), (points, asm.scale))
        torch.testing.assert_close(dx, ref_dx, atol=2e-13, rtol=3e-11)
        torch.testing.assert_close(ds, ref_ds, atol=2e-13, rtol=3e-11)
        with torch.no_grad():
            h = 2e-6
            difference = (objective(points + h * direction) - objective(points - h * direction)) / (2*h)
            torch.testing.assert_close((dx * direction).sum(), difference, atol=3e-8, rtol=3e-6)
            torch.testing.assert_close(ds, 2 * value / asm.scale)
            asm.scale.mul_(1.5)


def test_reference_plans_are_rebuilt_after_checkpoint_and_move_with_module():
    mesh = orientation_mesh([17, 0, 17, 0, 9])
    asm = poisson(3).from_mesh(mesh)
    expected = asm(func=lambda u, v: u * v, element_batch_size=3)
    state = asm.state_dict()
    assert not any("reference_plans" in key for key in state)
    other = poisson(3).from_mesh(mesh)
    other.load_state_dict(state)
    assert len(other.grouping.reference_plans) == 0
    actual = other(func=lambda u, v: u * v, element_batch_size=3)
    torch.testing.assert_close(actual.values, expected.values)
    other.float()
    assert all(t.dtype == torch.int64 for t in other.grouping.reference_plans.buffers())
    with pytest.raises(ValueError, match="reference_mode"):
        other(reference_mode="unknown")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_shared_reference_mass_reduces_peak_scratch():
    from examples.poisson.benchmark_hierarchical_tetrahedra import assembler_class, cube_mesh

    asm = assembler_class(4, "mass").from_mesh(cube_mesh(3)).cuda()
    with torch.no_grad():
        shared = asm().values.cpu()
        cell = asm(reference_mode="cell").values.cpu()
        torch.testing.assert_close(shared, cell, atol=3e-13, rtol=3e-11)
        peaks = {}
        for mode in ("auto", "cell"):
            torch.cuda.synchronize()
            baseline = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            result = asm(reference_mode=mode)
            torch.cuda.synchronize()
            peaks[mode] = torch.cuda.max_memory_allocated() - baseline
            del result
    assert peaks["auto"] < 0.5 * peaks["cell"]
