"""Opt-in triangle ElementAssembler with nodal geometry and modal fields."""

import copy

import pytest
import torch

from tensormesh import ElementAssembler, Field, MixedElementAssembler
from tensormesh.assemble import (LaplaceElementAssembler, MassElementAssembler,
                                LinearElasticityElementAssembler, J2Plasticity)
from tensormesh.distributed import distributed
from tests.assemble._triangle_fixtures import patch_mesh, poisson


@pytest.fixture(autouse=True)
def double_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def make_element(cls, mesh, p=3, **kwargs):
    return cls.from_mesh(mesh, basis="hierarchical", field_order=p,
                         quadrature_order=2*p, **kwargs)


def assert_operator(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=5e-12)


@pytest.mark.parametrize("p,project", [(1, "reduce"), (3, "sparse"), (4, "reduce")])
def test_triangle_element_mass_stiffness_match_mixed(p, project):
    mesh = patch_mesh()
    modal = poisson(p, "hierarchical").from_mesh(mesh, quadrature_order=2*p)
    for cls, form in ((MassElementAssembler, lambda u, v: u*v),
                      (LaplaceElementAssembler, lambda gradu, gradv: (gradu*gradv).sum())):
        asm = make_element(cls, mesh, p, project=project)
        expected = modal(func=form).to_dense()
        actual = asm(batch_size=3).to_dense()
        assert_operator(actual, expected)
        assert_operator(asm().to_dense(), actual)
        assert asm.n_points == len(mesh.points)
        assert asm().shape == (modal.n_dofs, modal.n_dofs)
        torch.testing.assert_close(asm.elements["triangle"], mesh.elements())


def test_triangle_element_vector_elasticity_on_curved_geometry():
    class Elasticity(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=2, basis="hierarchical")]

        def forward(self, gradu, gradv):
            eu, ev = .5*(gradu+gradu.T), .5*(gradv+gradv.T)
            mu, lam = 23./(2*1.3), 23.*.3/(1.3*.4)
            return 2*mu*(eu*ev).sum() + lam*gradu.trace()*gradv.trace()

    mesh = patch_mesh(2, curved=True)
    asm = make_element(LinearElasticityElementAssembler, mesh, E=23., nu=.3)
    expected = Elasticity.from_mesh(mesh, quadrature_order=6)(batch_size=3).to_dense()
    assert_operator(asm(batch_size=3).to_dense(), expected)


def test_triangle_element_nonsymmetric_form_and_nodal_data_channels():
    class Weighted(ElementAssembler):
        def forward(self, u, gradv, kappa, gradkappa, cell, alpha, x):
            return alpha*(1+kappa+gradkappa.sum()+cell+x.sum())*u*gradv[0]

    mesh = patch_mesh(2, curved=True)
    asm = make_element(Weighted, mesh)
    mixed = poisson(3, "hierarchical").from_mesh(mesh, quadrature_order=6)
    kwargs = dict(point_data={"kappa": mesh.points[:, 0].square()},
                  element_data={"cell": torch.tensor([.2, .4], dtype=torch.float64)},
                  scalar_data={"alpha": 1.7})
    # Element uses u for rows; Mixed uses the test field v for rows.
    expected = mixed(func=lambda v, gradu, kappa, gradkappa, cell, alpha, x:
                     alpha*(1+kappa+gradkappa.sum()+cell+x.sum())*v*gradu[0],
                     **kwargs).to_dense()
    actual = asm(batch_size=3, **kwargs).to_dense()
    assert_operator(actual, expected)
    assert (actual-actual.T).abs().max() > 1e-6

    trans = asm.transformation["triangle6"]
    x = torch.einsum("qb,ebd->eqd", trans.shape_val, mesh.points[mesh.elements()])
    history = 1+x[:, :, 0]
    actual = asm(func=lambda u, gradv, history: history*u*gradv[0],
                 quadrature_data={"history": history}, batch_size=3).to_dense()
    expected = mixed(func=lambda v, gradu, x: (1+x[0])*v*gradu[0]).to_dense()
    assert_operator(actual, expected)


def test_triangle_element_repeated_geometry_and_coefficient_autograd():
    class Weighted(ElementAssembler):
        def forward(self, u, v, gradu, gradv, kappa, gradkappa, alpha):
            return alpha*(kappa*(gradu*gradv).sum() + gradkappa.sum()*u*v)

    mesh = patch_mesh(2, curved=True)
    asm = make_element(Weighted, mesh)
    mixed = poisson(3, "hierarchical").from_mesh(mesh, quadrature_order=6)
    points = mesh.points.detach().clone().requires_grad_()
    kappa = torch.linspace(.7, 1.3, len(points), dtype=torch.float64, requires_grad=True)
    alpha = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    for _ in range(2):
        out = asm(points, point_data={"kappa": kappa}, scalar_data={"alpha": alpha})
        gradients = torch.autograd.grad(out.values.square().sum(), (points, kappa, alpha))
        # Mixed's existing scalar_data conversion detaches tensors; capture
        # alpha in this reference instead of changing that assembler's API.
        reference = mixed(points, point_data={"kappa": kappa},
                          func=lambda u, v, gradu, gradv, kappa, gradkappa:
                          alpha*(kappa*(gradu*gradv).sum() + gradkappa.sum()*u*v))
        expected = torch.autograd.grad(reference.values.square().sum(), (points, kappa, alpha))
        for actual, target in zip(gradients, expected):
            assert_operator(actual, target)


def test_triangle_element_reuse_checkpoint_and_geometry_changes():
    mesh = patch_mesh()
    asm = make_element(MassElementAssembler, mesh)
    expected = asm().to_dense()
    state = copy.deepcopy(asm.state_dict())
    clone = make_element(MassElementAssembler, mesh)
    clone.load_state_dict(state)
    assert_operator(clone().to_dense(), expected)
    clone.transformation["triangle"].points.mul_(1.2)
    assert_operator(clone().to_dense(), expected*1.2**2)
    clone.load_state_dict(state)
    assert_operator(clone().to_dense(), expected)
    clone.float()
    torch.testing.assert_close(clone().to_dense().double(), expected, rtol=2e-5, atol=2e-7)
    other = LaplaceElementAssembler.from_assembler(asm)
    assert other._hierarchical is asm._hierarchical
    assert other().layout_signature == asm().layout_signature
    assert_operator(other().to_dense(), make_element(LaplaceElementAssembler, patch_mesh())().to_dense())


def test_triangle_element_lagrange_defaults_and_field_order_inference():
    class Scaled(MassElementAssembler):
        def __post_init__(self, scale):
            self.scale = scale

        def forward(self, u, v):
            return self.scale*u*v

    mesh = patch_mesh(2)
    default = Scaled.from_mesh(mesh, 4, "reduce", 2.5)
    explicit = Scaled.from_mesh(mesh, 4, "reduce", 2.5, basis="lagrange", field_order=2)
    assert_operator(default().to_dense(), explicit().to_dense())
    assert default.state_dict().keys() == explicit.state_dict().keys()
    modal = Scaled.from_mesh(mesh, 4, "reduce", 2.5, basis="hierarchical")
    assert modal._hierarchical.order == 2
    assert_operator(modal().to_dense(), make_element(Scaled, mesh, 2, scale=2.5)().to_dense())
    inferred = MassElementAssembler.from_mesh(mesh, basis="hierarchical")
    assert inferred.transformation["triangle6"].quadrature_order == 2

    class Legacy(MassElementAssembler):
        def __init__(self, projector, transformation, elements, edges):
            super().__init__(projector, transformation, elements, edges)

    legacy = Legacy.from_mesh(mesh)
    assert_operator(Legacy.from_assembler(legacy)().to_dense(), legacy().to_dense())


def test_triangle_element_unsupported_assembly_paths():
    mesh = patch_mesh()
    with pytest.raises(NotImplementedError, match="energy"):
        make_element(MassElementAssembler, mesh).energy()
    with pytest.raises(NotImplementedError, match="distributed"):
        distributed(MassElementAssembler).from_mesh(object(), basis="hierarchical")
    with pytest.raises(NotImplementedError, match="state updates"):
        J2Plasticity.from_mesh(mesh, basis="hierarchical")
    with pytest.raises(NotImplementedError, match="state updates"):
        J2Plasticity.from_assembler(make_element(MassElementAssembler, mesh))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_triangle_element_cuda_construction_and_movement():
    mesh = patch_mesh(2, curved=True)
    expected = make_element(LaplaceElementAssembler, mesh)().to_dense()
    asm = make_element(LaplaceElementAssembler, mesh.cuda())
    assert_operator(asm().to_dense().cpu(), expected)
    asm.cpu()
    assert_operator(asm().to_dense(), expected)
