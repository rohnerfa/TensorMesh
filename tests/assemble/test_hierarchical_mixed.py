"""Modal assembly verified against nodal change-of-basis identities."""

import meshio
import pytest
import torch

from tensormesh import Field, Mesh, MixedElementAssembler
from tensormesh.element import Triangle
from tests.assemble._triangle_fixtures import patch_mesh, poisson, change_of_basis


@pytest.mark.parametrize("p", [1, 2, 3, 4])
@pytest.mark.parametrize("geometry", [1, 2, 3])
def test_poisson_change_of_basis_and_load(p, geometry):
    mesh = patch_mesh(geometry, curved=geometry > 1)
    # P4 stiffness is integrated exactly by degree 7 on affine geometry.
    # Curved cases compare the identical quadrature discretizations.
    modal = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=7)
    nodal = poisson(p, "lagrange").from_mesh(mesh, quadrature_order=7)
    assert modal.field_kind["u"] == "hierarchical"
    t = change_of_basis(nodal, modal)
    kh, kl = modal().to_dense(), nodal().to_dense()
    torch.testing.assert_close(kh, t.T @ kl @ t, atol=5e-12, rtol=2e-11)
    torch.testing.assert_close(modal.assemble_vector(), t.T @ nodal.assemble_vector(), atol=2e-13, rtol=2e-11)
    torch.testing.assert_close(kh, kh.T, atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("p", [2, 3, 4])
def test_constrained_poisson_spd(p):
    modal = poisson(p, "hierarchical").from_mesh(patch_mesh(), quadrature_order=7)
    k = modal().to_dense()
    conn = modal.field_conn["u__triangle"]
    # Both triangles have boundary edges 1 and 2. Their edge 0 is internal.
    boundary = torch.zeros(modal.n_dofs, dtype=torch.bool)
    boundary[conn[:, :3]] = True
    boundary[conn[:, p + 2:3 * p]] = True
    free = ~boundary
    assert free.sum() == p - 1 + (p - 1) * (p - 2)
    assert torch.linalg.eigvalsh(k[free][:, free]).min() > 0


def mixed_form(family, pressure_family="lagrange", pressure_order=2):
    class Mixed(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=2, basis=family),
                  Field("p", "q", pressure_order, basis=pressure_family)]

        def forward(self, u, v, gradu, gradv, p, q, gradp, gradq, x):
            return ((gradu * gradv).sum() + (u * v).sum()
                    + p * gradv.diagonal().sum() + q * gradu.diagonal().sum()
                    + (gradp * gradq).sum() + p * q
                    + x[0] * (q * u[0] + p * v[0]))

        def forward_vector(self, v, gradv, q, gradq, x):
            return (x * v).sum() + gradv[0, 1] + x[0] * q + gradq[1]

    return Mixed


@pytest.mark.parametrize("batch_size", [-1, 1, 4])
@pytest.mark.parametrize("pressure_family,pressure_order", [
    ("lagrange", 2), ("hierarchical", 2), ("hierarchical", 3)])
def test_mixed_vector_scalar_blocks_and_loads(batch_size, pressure_family, pressure_order):
    mesh = patch_mesh(3, curved=True)
    modal = mixed_form("hierarchical", pressure_family, pressure_order).from_mesh(mesh)
    nodal = mixed_form("lagrange", pressure_order=pressure_order).from_mesh(mesh)
    tu = change_of_basis(nodal, modal)
    tp = (change_of_basis(nodal, modal, "p") if pressure_family == "hierarchical"
          else torch.eye(nodal.layout.n_nodes("p"), dtype=torch.float64))
    t = torch.block_diag(torch.kron(tu, torch.eye(2, dtype=torch.float64)), tp)
    torch.testing.assert_close(modal(batch_size=batch_size).to_dense(),
                               t.T @ nodal().to_dense() @ t, atol=5e-12, rtol=2e-11)
    torch.testing.assert_close(modal.assemble_vector(batch_size=batch_size),
                               t.T @ nodal.assemble_vector(), atol=2e-13, rtol=2e-11)


@pytest.mark.parametrize("components", [1, 2])
def test_modal_field_data_values_gradients_and_autograd(components):
    class DataForm(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=components, basis="hierarchical")]

        def forward(self, u, v, w, gradw):
            return (u * v).sum() * (1 + w.sum() + gradw.sum())

        def forward_vector(self, v, gradv, w, gradw):
            return (w * v).sum() + (gradw * gradv).sum()

    mesh = patch_mesh()
    modal = DataForm.from_mesh(mesh)
    nodal_type = type("NodalDataForm", (DataForm,), {
        "fields": [Field("u", "v", 3, components=components)]})
    nodal = nodal_type.from_mesh(mesh)
    t_scalar = change_of_basis(nodal, modal)
    t = torch.kron(t_scalar, torch.eye(components, dtype=torch.float64))
    generator = torch.Generator().manual_seed(218)
    shape = (modal.layout.n_nodes("u"),) if components == 1 else (modal.layout.n_nodes("u"), components)
    wh = torch.randn(shape, dtype=torch.float64, generator=generator, requires_grad=True)
    wl = (t @ wh.reshape(-1)).reshape(-1) if components == 1 else (t @ wh.reshape(-1)).reshape(-1, components)
    kh = modal(field_data={"w": ("u", wh)}).to_dense()
    kl = nodal(field_data={"w": ("u", wl)}).to_dense()
    torch.testing.assert_close(kh, t.T @ kl @ t, atol=5e-12, rtol=2e-11)
    bh = modal.assemble_vector(field_data={"w": ("u", wh)}, batch_size=2)
    bl = nodal.assemble_vector(field_data={"w": ("u", wl)})
    torch.testing.assert_close(bh, t.T @ bl, atol=2e-13, rtol=2e-11)
    gh = torch.autograd.grad(kh.square().sum() + bh.square().sum(), wh, retain_graph=True)[0]
    reference_loss = (t.T @ kl @ t).square().sum() + (t.T @ bl).square().sum()
    gl = torch.autograd.grad(reference_loss, wh)[0]
    torch.testing.assert_close(gh, gl, atol=2e-10, rtol=2e-10)


def test_reference_tables_shared_by_family_and_no_hot_path_tabulation(monkeypatch):
    class TwoFields(MixedElementAssembler):
        fields = [Field("u", "v", 3, basis="hierarchical"),
                  Field("p", "q", 3, basis="hierarchical")]

        def forward(self, u, v, p, q):
            return u * v + p * q + p * v + u * q

    original = Triangle.get_hierarchical_basis_fns
    calls = []
    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(Triangle, "get_hierarchical_basis_fns", counted)
    asm = TwoFields.from_mesh(patch_mesh())
    assert len(calls) == 1
    assert asm.ref_val["u__triangle"].data_ptr() == asm.ref_val["p__triangle"].data_ptr()
    def fail(*args, **kwargs):
        raise AssertionError("basis must not be evaluated in the assembly hot path")
    monkeypatch.setattr(Triangle, "get_hierarchical_basis_fns", fail)
    asm(batch_size=1)
    asm.assemble_vector(func=lambda v, q: v + q, batch_size=2)


def test_distinct_family_reference_tables():
    class TwoFamilies(MixedElementAssembler):
        fields = [Field("u", "v", 3, basis="hierarchical"), Field("p", "q", 3)]
    asm = TwoFamilies.from_mesh(patch_mesh())
    assert not torch.allclose(asm.ref_val["u__triangle"], asm.ref_val["p__triangle"])
    assert "u__triangle" in asm.field_sign and "p__triangle" not in asm.field_sign


@pytest.mark.parametrize("operation", ["points"])
def test_nodal_only_helpers_fail_clearly(operation):
    asm = poisson(3, "hierarchical").from_mesh(patch_mesh(3))
    args = ["u"]
    if operation == "restrict":
        args.append(torch.zeros(asm.n_points))
    elif operation == "prolong":
        args.append(torch.zeros(asm.layout.n_nodes("u")))
    with pytest.raises(NotImplementedError, match="modal coefficients"):
        getattr(asm.layout, operation)(*args)
    with pytest.raises(NotImplementedError, match="modal coefficients"):
        asm.layout.dof_mask("u", where=lambda x: x[:, 0] == 0)


def test_invalid_family_and_nontriangle():
    with pytest.raises(ValueError, match="basis"):
        Field("u", "v", basis="unknown")
    mesh = Mesh(meshio.Mesh([[0., 0.], [1., 0.], [0., 1.], [1., 1.]],
                            [("quad", [[0, 1, 2, 3]])])).double()
    with pytest.raises(NotImplementedError, match="triangles only"):
        poisson(2, "hierarchical").from_mesh(mesh)


def test_dtype_movement_and_state_dict():
    mesh = patch_mesh()
    asm = poisson(3, "hierarchical").from_mesh(mesh)
    state = asm.state_dict()
    assert "field_sign.u__triangle" in state
    other = poisson(3, "hierarchical").from_mesh(mesh)
    other.load_state_dict(state)
    torch.testing.assert_close(asm().to_dense(), other().to_dense())
    k = asm().to_dense()
    b = asm.assemble_vector()
    # __call__ follows geometry dtype; cast the complete assembler first.
    asm.float()
    assert asm.field_sign["u__triangle"].dtype == torch.float32
    torch.testing.assert_close(asm().to_dense().double(), k, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(asm.assemble_vector().double(), b, atol=2e-7, rtol=2e-5)
