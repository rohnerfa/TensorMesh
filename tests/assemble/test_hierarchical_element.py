"""Opt-in hierarchical ElementAssembler; the nodal API remains unchanged."""

import copy

import meshio
import numpy as np
import pytest
import torch

from tensormesh import ElementAssembler, Field, Mesh, MixedElementAssembler
from tensormesh.assemble import (LaplaceElementAssembler, MassElementAssembler,
                                LinearElasticityElementAssembler)
from tensormesh.assemble import _hierarchical_element as tables_module
from tensormesh.distributed import distributed

from _tetrahedron_fixtures import (patch_mesh, orientation_mesh, poisson,
                                  scalar_reference, change_of_basis)


@pytest.fixture(autouse=True)
def double_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


def make_element(cls, mesh, p, **kwargs):
    return cls.from_mesh(mesh, quadrature_order=2*p, basis="hierarchical",
                         field_order=p, **kwargs)


def assert_operator(actual, expected, tolerance=3e-12):
    torch.testing.assert_close(actual, expected, rtol=tolerance,
                               atol=tolerance * max(1., float(expected.detach().abs().max())))


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("project", ["reduce", "sparse"])
def test_scalar_mass_and_stiffness_match_mixed_and_independent_integrals(p, project):
    mesh = patch_mesh()
    mixed = poisson(p).from_mesh(mesh, quadrature_order=2*p)
    mass_ref, stiffness_ref, _ = scalar_reference(mixed)
    for cls, reference in ((MassElementAssembler, mass_ref),
                           (LaplaceElementAssembler, stiffness_ref)):
        asm = make_element(cls, mesh, p, project=project)
        result = asm(batch_size=3).to_dense()
        assert_operator(result, reference)
        if cls is LaplaceElementAssembler:
            assert_operator(result, mixed(batch_size=3).to_dense())
        assert asm.n_points == len(mesh.points)
        assert result.shape == (mixed.n_dofs, mixed.n_dofs)
        torch.testing.assert_close(asm.elements['tetra'], mesh.elements())


@pytest.mark.parametrize("p", [2, 3, 4])
@pytest.mark.parametrize("geometry,curved", [(1, False), (2, True)])
def test_vector_elasticity_keeps_block_forward_contract(p, geometry, curved):
    class Elasticity(MixedElementAssembler):
        fields = [Field("u", "v", p, components=3, basis="hierarchical")]

        def forward(self, gradu, gradv):
            eu, ev = .5*(gradu+gradu.T), .5*(gradv+gradv.T)
            mu, lam = 23./(2*1.3), 23.*.3/(1.3*.4)
            return 2*mu*(eu*ev).sum() + lam*gradu.trace()*gradv.trace()

    mesh = patch_mesh(geometry, curved)
    asm = make_element(LinearElasticityElementAssembler, mesh, p, E=23., nu=.3)
    expected = Elasticity.from_mesh(mesh, quadrature_order=2*p)(batch_size=3).to_dense()
    assert_operator(asm(batch_size=3).to_dense(), expected)


def test_all_orientations_match_mixed_and_batching():
    mesh = orientation_mesh()
    asm = make_element(MassElementAssembler, mesh, 3)
    mixed = poisson(3).from_mesh(mesh, quadrature_order=6)
    expected = mixed(func=lambda u, v: u*v, batch_size=3).to_dense()
    assert_operator(asm(batch_size=3).to_dense(), expected)
    unbatched = asm().to_dense()
    for batch in (1, 4, None, 100):
        assert_operator(asm(batch_size=batch).to_dense(), unbatched)


def test_nonsymmetric_convention_and_all_data_channels():
    class Ref(ElementAssembler):
        def forward(self, u, gradv, kappa, gradkappa, cell, history, alpha, x):
            assert u.ndim == 0 and gradv.shape == (3,)
            return alpha*(1+kappa+gradkappa.sum()+cell+history+x.sum())*u*gradv[0]

    class Mixed(MixedElementAssembler):
        fields = [Field("trial", "test", 3, basis="hierarchical")]

        def forward(self, test, gradtrial, kappa, gradkappa, cell, history, alpha, x):
            # Mixed uses test rows; Element uses the letter u for rows.
            return alpha*(1+kappa+gradkappa.sum()+cell+history+x.sum())*test*gradtrial[0]

    mesh = patch_mesh(2, True)
    asm = make_element(Ref, mesh, 3)
    mixed = Mixed.from_mesh(mesh, quadrature_order=6)
    nodal = mesh.points[:, 0].square()
    nq = asm.transformation['tetra10'].n_quadrature
    history = torch.linspace(.1, .8, 2*nq).reshape(2, nq)
    # Mixed has no quadrature_data channel: integrate identical nodal coefficients
    # first, and independently compare the history-weighted Element result below.
    kw = dict(point_data={'kappa': nodal}, element_data={'cell': torch.tensor([.2, .4])},
              scalar_data={'alpha': 1.7})
    actual = asm(func=lambda u, gradv, kappa, gradkappa, cell, alpha, x:
                 alpha*(1+kappa+gradkappa.sum()+cell+x.sum())*u*gradv[0],
                 batch_size=3, **kw).to_dense()
    reference = mixed(func=lambda test, gradtrial, kappa, gradkappa, cell, alpha, x:
                      alpha*(1+kappa+gradkappa.sum()+cell+x.sum())*test*gradtrial[0],
                      batch_size=3, **kw).to_dense()
    assert_operator(actual, reference)
    assert (actual-actual.T).abs().max() > 1e-6
    full = asm(quadrature_data={'history': history}, **kw).to_dense()
    assert_operator(asm(quadrature_data={'history': history}, batch_size=3, **kw).to_dense(), full)
    assert not torch.allclose(full, actual)


def test_supported_nodal_change_of_basis():
    mesh = patch_mesh(2, True)
    nodal = poisson(2, family='lagrange').from_mesh(mesh, quadrature_order=8)
    modal = poisson(2).from_mesh(mesh, quadrature_order=8)
    T = change_of_basis(nodal, modal)
    asm = MassElementAssembler.from_mesh(mesh, quadrature_order=8, basis='hierarchical')
    nodal_mass = MassElementAssembler.from_mesh(mesh, quadrature_order=8)().to_dense()
    assert_operator(asm().to_dense(), T.T @ nodal_mass @ T)


def test_caches_reuse_geometry_but_invalidate_on_coordinate_changes(monkeypatch):
    mesh = patch_mesh()
    asm = make_element(LaplaceElementAssembler, mesh, 3)
    inverse = tables_module.small_inverse
    calls = []

    def counted(jac):
        calls.append(jac.shape)
        return inverse(jac)

    monkeypatch.setattr(tables_module, 'small_inverse', counted)
    initial = asm().to_dense()
    assert len(calls) == 1
    assert_operator(asm().to_dense(), initial)
    assert_operator(asm(batch_size=3).to_dense(), initial)
    assert len(calls) == 1
    trans = asm.transformation['tetra']
    moved = trans.points.clone()*1.2
    assert_operator(asm(moved).to_dense(), initial*1.2)
    assert len(calls) == 2
    # No explicit points override: in-place changes must still invalidate.
    trans.points.mul_(1.1)
    assert_operator(asm().to_dense(), initial*1.32)
    assert len(calls) == 3
    asm.float()
    assert_operator(asm().to_dense().double(), initial*1.32, tolerance=2e-5)
    assert len(calls) == 4


def test_chunked_calls_do_not_materialize_full_physical_cache(monkeypatch):
    asm = make_element(LaplaceElementAssembler, patch_mesh(), 3)
    field = asm._hierarchical.blocks['tetra']
    chunked = asm(batch_size=3).to_dense()
    assert '_grad' not in field._buffers
    assert '_inverse' not in field._buffers
    assert '_values' not in field._buffers
    # Reference tabulation happens only during construction, even when chunking.
    from tensormesh.element import Tetrahedron
    monkeypatch.setattr(Tetrahedron, 'get_hierarchical_basis_fns',
                        lambda *args, **kwargs: pytest.fail('reference re-tabulation'))
    assert_operator(asm().to_dense(), chunked)


def test_repeated_autograd_for_geometry_nodal_and_scalar_data():
    class Weighted(ElementAssembler):
        def forward(self, u, v, gradu, gradv, kappa, gradkappa, alpha):
            return alpha*(kappa*(gradu*gradv).sum() + gradkappa.sum()*u*v)

    class Mixed(MixedElementAssembler):
        fields = [Field('u', 'v', 3, basis='hierarchical')]

        def forward(self, u, v, gradu, gradv, kappa, gradkappa):
            return 1.2*(kappa*(gradu*gradv).sum() + gradkappa.sum()*u*v)

    mesh = patch_mesh(2, True)
    asm = make_element(Weighted, mesh, 3)
    mixed = Mixed.from_mesh(mesh, quadrature_order=6)
    points = mesh.points.detach().clone().requires_grad_()
    kappa = torch.linspace(.7, 1.3, len(points), requires_grad=True)
    alpha = torch.tensor(1.2, requires_grad=True)
    for _ in range(2):
        out = asm(points, point_data={'kappa': kappa}, scalar_data={'alpha': alpha})
        grad = torch.autograd.grad(out.values.square().sum(), (points, kappa, alpha))
        ref = mixed(points, point_data={'kappa': kappa})
        ref_grad = torch.autograd.grad(ref.values.square().sum(), (points, kappa))
        for a, b in zip(grad, ref_grad):
            assert_operator(a, b, tolerance=2e-10)
        assert grad[2].abs() > 0
        assert '_grad' not in asm._hierarchical.blocks['tetra10']._buffers


def test_from_assembler_layout_identity_and_warm_state_roundtrip():
    mesh = patch_mesh()
    asm = make_element(LaplaceElementAssembler, mesh, 3)
    expected = asm().to_dense()
    signature = asm().layout_signature
    mass = MassElementAssembler.from_assembler(asm)
    assert mass._hierarchical is asm._hierarchical
    assert mass.transformation is asm.transformation
    assert mass().layout_signature == signature
    assert_operator(mass().to_dense(), make_element(MassElementAssembler, mesh, 3)().to_dense())
    cold = make_element(LaplaceElementAssembler, mesh, 3)
    cold.load_state_dict(copy.deepcopy(asm.state_dict()))
    assert_operator(cold().to_dense(), expected)
    # Loading into an already warm instance must invalidate disposable tables.
    other = make_element(LaplaceElementAssembler, patch_mesh(), 3)
    other(other.transformation['tetra'].points*1.5)
    other.load_state_dict(copy.deepcopy(asm.state_dict()))
    assert_operator(other().to_dense(), expected)


def test_nodal_defaults_and_positional_constructor_parameters_remain_unchanged():
    class Scaled(ElementAssembler):
        def __post_init__(self, factor, *, marker):
            self.factor, self.marker = factor, marker

        def forward(self, u, v):
            return self.factor*u*v

    mesh = patch_mesh(2)
    default = Scaled.from_mesh(mesh, 4, 'reduce', 2.5, marker='legacy')
    explicit = Scaled.from_mesh(mesh, 4, 'reduce', 2.5, marker='legacy',
                                basis='lagrange', field_order=2)
    assert default._hierarchical is None and default.marker == 'legacy'
    assert_operator(default().to_dense(), explicit().to_dense())
    assert default.state_dict().keys() == explicit.state_dict().keys()
    assert not any('_hierarchical' in key for key in default.state_dict())
    assert default.transformation['tetra10'].quadrature_order == 4
    hierarchical = Scaled.from_mesh(mesh, 4, 'reduce', 2.5, marker='new', basis='hierarchical')
    assert hierarchical._hierarchical.order == 2
    assert hierarchical.transformation['tetra10'].quadrature_order == 4
    # Do not silently change the old degree-two quadrature default.
    default_quadrature = MassElementAssembler.from_mesh(mesh, basis='hierarchical')
    assert default_quadrature.transformation['tetra10'].quadrature_order == 2


def test_legacy_constructor_override_receives_no_new_keywords():
    class Legacy(MassElementAssembler):
        def __init__(self, projector, transformation, elements, edges):
            super().__init__(projector, transformation, elements, edges)

    asm = Legacy.from_mesh(patch_mesh())
    assert_operator(Legacy.from_assembler(asm)().to_dense(), asm().to_dense())


def test_cubic_geometry_and_field_degree_inference():
    from tensormesh.element import Tetrahedron
    linear = patch_mesh()
    reference = Tetrahedron.get_basis(3, torch.float64)
    weights = torch.cat((1-reference.sum(1, keepdim=True), reference), dim=1)
    samples = torch.einsum('qb,ebd->eqd', weights, linear.points[linear.elements()])
    points, inverse = np.unique(
        samples.numpy().round(13).reshape(-1, 3), axis=0, return_inverse=True)
    cells = inverse.reshape(2, 20)
    mesh = Mesh(meshio.Mesh(points, [('tetra20', cells)]), reorder=False).double()
    asm = LinearElasticityElementAssembler.from_mesh(
        mesh, basis='hierarchical', quadrature_order=4, E=23., nu=.3)
    assert asm._hierarchical.order == 3
    assert asm.transformation['tetra20'].basis_order == 3
    modal = poisson(3).from_mesh(mesh, quadrature_order=4)
    laplace = LaplaceElementAssembler.from_assembler(asm)
    assert_operator(laplace().to_dense(), modal().to_dense())


def test_complex_nodal_coefficients_keep_real_field_tables():
    mesh = patch_mesh()
    asm = make_element(MassElementAssembler, mesh, 3).type(torch.complex128)
    kappa = torch.full((len(mesh.points),), 1+2j, dtype=torch.complex128)
    expected = make_element(MassElementAssembler, mesh, 3)().to_dense()*(1+2j)
    actual = asm(func=lambda u, v, kappa: kappa*u*v, point_data={'kappa': kappa}).to_dense()
    assert_operator(actual, expected)
    assert not asm._hierarchical.blocks['tetra'].ref_val.is_complex()


@pytest.mark.parametrize('order', [0, -1, True, 2.5])
def test_invalid_field_orders(order):
    with pytest.raises(ValueError, match='field_order'):
        MassElementAssembler.from_mesh(patch_mesh(), basis='hierarchical', field_order=order)


def test_unsupported_combinations_fail_explicitly():
    mesh = patch_mesh()
    with pytest.raises(ValueError, match='basis'):
        MassElementAssembler.from_mesh(mesh, basis='unknown')
    with pytest.raises(ValueError, match='different field_order'):
        MassElementAssembler.from_mesh(mesh, field_order=3)
    triangle = Mesh(meshio.Mesh(np.array([[0., 0.], [1., 0.], [0., 1.]]),
                               [('triangle', np.array([[0, 1, 2]]))]))
    with pytest.raises(NotImplementedError, match='tetrahedra only'):
        MassElementAssembler.from_mesh(triangle, basis='hierarchical')
    with pytest.raises(NotImplementedError, match='energy'):
        make_element(MassElementAssembler, mesh, 3).energy()
    with pytest.raises(NotImplementedError, match='distributed'):
        distributed(MassElementAssembler).from_mesh(object(), basis='hierarchical')
    from tensormesh.assemble import J2Plasticity
    with pytest.raises(NotImplementedError, match='state updates'):
        J2Plasticity.from_mesh(mesh, basis='hierarchical')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_construction_and_movement():
    mesh = patch_mesh(2, True)
    expected = make_element(LaplaceElementAssembler, mesh, 3)().to_dense()
    asm = make_element(LaplaceElementAssembler, mesh.cuda(), 3)
    assert_operator(asm().to_dense().cpu(), expected)
    asm.cpu()
    assert_operator(asm().to_dense(), expected)
