"""Independent-order tetrahedral Lagrange fields, in the original mesh frame."""
from itertools import permutations
from math import comb

import pytest
import torch

from tensormesh import ElementAssembler, Field, MixedElementAssembler, NodeAssembler
from tensormesh.assemble.topology import lagrange_dofmap
from tensormesh.distributed import DistributedMesh, distributed
from tensormesh.element import Tetrahedron
from _tetrahedron_fixtures import patch_mesh


@pytest.mark.parametrize("p", [3, 4, 5])
def test_nodal_ids_under_all_tetrahedral_permutations(p):
    # Disjoint two-tet patches; cycle the common face and try every corner order.
    cells = torch.tensor([[a + 5 * i for a in row]
                          for i, perm in enumerate(permutations(range(4)))
                          for row in [perm, (2, 3, 1, 4)]])
    dofs = lagrange_dofmap({"tetra": cells}, 120, p)
    xi = Tetrahedron.get_basis(p, torch.float64)
    weights = (p * torch.cat((1 - xi.sum(1, keepdim=True), xi), dim=1)).round().long()
    # Independent oracle: a node is the sum of weights on its actual vertices.
    # Compare exact keys; merely counting shared IDs misses face-slot reversals.
    key_to_id = {}
    for row, ids in zip(cells.tolist(), dofs.conn["tetra"].tolist()):
        for bary, gid in zip(weights.tolist(), ids):
            key = tuple(sorted((v, w) for v, w in zip(row, bary) if w))
            assert key_to_id.setdefault(key, gid) == gid
    assert len(key_to_id) == len(set(key_to_id.values())) == dofs.n_dofs
    assert dofs.n_dofs == 24 * (2 * comb(p + 3, 3) - comb(p + 2, 2))


class Stiffness(ElementAssembler):
    def forward(self, gradu, gradv, kappa):
        return kappa * (gradu @ gradv)


class Load(NodeAssembler):
    def forward(self, v, gradkappa, x):
        return v * (1 + gradkappa.sum() + x.sum())


@pytest.mark.parametrize("p,geometry", [(3, 1), (4, 2), (5, 1)])
@pytest.mark.parametrize("project", ["reduce", "sparse"])
def test_matrix_load_and_layout_match_mixed(p, geometry, project):
    class Mixed(MixedElementAssembler):
        fields = [Field("u", "v", p)]

        def forward(self, gradu, gradv, kappa):
            return kappa * (gradu @ gradv)

    mesh = patch_mesh(geometry, curved=geometry == 2)
    args = dict(field_order=p, quadrature_order=2*p, project=project)
    asm = Stiffness.from_mesh(mesh, **args)
    load = Load.from_assembler(asm)
    separate = Load.from_mesh(mesh, **args)
    ref = Mixed.from_mesh(mesh, quadrature_order=2*p)
    assert asm._field_space is load._field_space
    assert asm._field_space is Stiffness.from_assembler(asm)._field_space
    layout = asm.field_layout()
    torch.testing.assert_close(layout.points("u"), ref.layout.points("u"))
    boundary = layout.boundary_mask("u")
    # Inner face nodes and private cell nodes only.
    assert (~boundary).sum() == comb(p-1, 2) + 2 * comb(p-1, 3)
    assert torch.equal(boundary, load.field_layout().boundary_mask("u"))
    data = {"kappa": .9 + mesh.points.square().sum(1)}
    kw = dict(point_data=data, batch_size=7)
    torch.testing.assert_close(asm(**kw).to_dense(), ref(**kw).to_dense())
    expected = ref.assemble_vector(func=load.forward, **kw)
    torch.testing.assert_close(load(**kw), expected)
    torch.testing.assert_close(separate.compile(mode="disable")(**kw), expected)

    # Nonconstant trace at an off-lattice point on the shared physical face.
    cells = mesh.elements()[:, :4]
    bary = torch.tensor([0., .19, .34, .47, 0.], dtype=mesh.points.dtype)[cells]
    coeff = torch.randn(layout.n_dofs, dtype=mesh.points.dtype,
                        generator=torch.Generator().manual_seed(14))
    trace = layout.evaluate("u", coeff, bary[:, None, 1:])
    torch.testing.assert_close(trace[0], trace[1])


def test_vector_and_moving_geometry():
    class Vec(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=3)]

        def forward(self, gradu, gradv):
            return gradu.trace() * gradv.trace()

    mesh = patch_mesh(2, curved=True)
    a = Stiffness.from_mesh(mesh, field_order=3, quadrature_order=6)
    b = Load.from_assembler(a)
    ref = Vec.from_mesh(mesh, quadrature_order=6)
    x0 = mesh.points
    for factor in (1., 1.03):
        x = (x0 * factor).detach().requires_grad_()
        # ElementAssembler's block axes follow its (u, v) basis-pair axes.
        K = a(x, func=lambda gradu, gradv: torch.outer(gradu, gradv)).to_dense()
        torch.testing.assert_close(K, ref(x).to_dense())
        f = b(x, func=lambda v, x: v * x)
        expected = ref.assemble_vector(x, func=lambda v, x: v @ x)
        torch.testing.assert_close(f, expected)
        torch.testing.assert_close(torch.autograd.grad(f.square().sum(), x)[0],
                                   torch.autograd.grad(expected.square().sum(), x)[0])


def test_p3_on_p1_dirichlet_solve():
    # Harmonic cubic: an exact solution even with the small patch's few free DOFs.
    mesh = patch_mesh()
    a = Stiffness.from_mesh(mesh, field_order=3, quadrature_order=6)
    K = a(func=lambda gradu, gradv: gradu @ gradv).to_dense()
    layout = a.field_layout()
    x, y, _ = layout.points("u").unbind(1)
    exact = x**3 - 3*x*y**2
    bd = layout.boundary_mask("u")
    free = ~bd
    assert free.any()
    u = torch.linalg.solve(K[free][:, free], -K[free][:, bd] @ exact[bd])
    torch.testing.assert_close(u, exact[free], atol=1e-12, rtol=1e-12)


def test_samples_keep_original_curved_geometry_frame():
    mesh = patch_mesh(2, curved=True)
    a = Stiffness.from_mesh(mesh, field_order=4, quadrature_order=8)
    b = Load.from_assembler(a)
    trans = a.transformation["tetra10"]
    history = torch.arange(2 * trans.n_quadrature, dtype=mesh.dtype).reshape(2, -1)
    # Partition of unity: sum of every load entry must integrate history
    # in the original two rows and q order, even for a nonsymmetric rule.
    expected = (history * trans.JxW).sum()
    for batch_size in (-1, 5):
        f = b(func=lambda v, history: history*v, quadrature_data={"history": history},
              batch_size=batch_size)
        torch.testing.assert_close(f.sum(), expected)
        k = a(func=lambda u, v, history: history*u*v, quadrature_data={"history": history},
              batch_size=batch_size)
        torch.testing.assert_close(k.values.sum(), expected)


def test_point_id_distributed_scatter_rejects_independent_field():
    dmesh = DistributedMesh(patch_mesh(), num_partitions=2, devices=["cpu", "cpu"])
    for form in (Stiffness, Load):
        with pytest.raises(NotImplementedError, match="independent Lagrange"):
            distributed(form).from_mesh(dmesh, field_order=3)
