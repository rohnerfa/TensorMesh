"""Grouped tetrahedral matrices and loads against independent cell integration."""

import pytest
import torch

from tensormesh import Condenser, Field, MixedElementAssembler
from tensormesh.assemble.projector import ReduceProjector, SparseProjector
from tensormesh.element import Tetrahedron
from tensormesh.nn import BufferDict
from tests.assemble._tetrahedron_fixtures import (
    DTYPE, ORIENTATIONS, cell_tables, change_of_basis, field_indices,
    independent_quadrature, orientation_mesh, patch_mesh, poisson,
    scalar_reference, scatter_matrix,
)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5, 6])
def test_affine_scalar_operators_match_cellwise_integration(p):
    asm = poisson(p).from_mesh(patch_mesh(), quadrature_order=max(8, 2 * p))
    mass, stiffness, load = scalar_reference(asm)
    actual_mass = asm(func=lambda u, v: u * v).to_dense()
    actual_stiffness = asm().to_dense()
    torch.testing.assert_close(actual_mass, mass, atol=2e-14, rtol=3e-11)
    torch.testing.assert_close(actual_stiffness, stiffness, atol=3e-13, rtol=3e-11)
    torch.testing.assert_close(asm.assemble_vector(), load, atol=3e-14, rtol=3e-11)
    torch.testing.assert_close(actual_stiffness, actual_stiffness.T, atol=2e-14, rtol=2e-13)
    assert torch.linalg.eigvalsh(actual_mass).min() > 0
    constant = torch.zeros(asm.n_dofs, dtype=DTYPE)
    constant[asm.field_conn["u__tetra"][:, :4]] = 1
    torch.testing.assert_close(actual_stiffness @ constant, torch.zeros_like(constant),
                               atol=4e-13, rtol=0)


def test_all_24_classes_preserve_public_cell_order_and_match_reference():
    from tensormesh.element.hierarchical_tetrahedron import TETRAHEDRON_ORIENTATIONS

    mesh = orientation_mesh()
    conn = mesh.elements().clone()
    asm = poisson(4).from_mesh(mesh)
    assert tuple(TETRAHEDRON_ORIENTATIONS) == ORIENTATIONS
    assert asm.grouped_tetrahedra
    torch.testing.assert_close(asm.elements["tetra"], conn)
    torch.testing.assert_close(asm.cell_orientation["tetra"], torch.arange(23, -1, -1))
    assert len(asm.grouping.groups) == 24
    assert all(group.n_cells == 1 for group in asm.grouping.groups)
    assert "u__tetra" not in asm.field_sign
    for group in asm.grouping.groups:
        assert group.orientation == ORIENTATIONS[group.class_id]
        cells = asm.group_cells[group.key]
        assert asm.cell_orientation[group.element_type][cells].eq(group.class_id).all()
        # Vertex coefficients retain the local slots of the original geometry.
        torch.testing.assert_close(asm.field_conn["u__tetra"][cells, :4], conn[cells, :4])
    mass, stiffness, load = scalar_reference(asm)
    torch.testing.assert_close(asm(func=lambda u, v: u * v).to_dense(), mass,
                               atol=3e-14, rtol=3e-11)
    torch.testing.assert_close(asm(batch_size=17, element_batch_size=2).to_dense(), stiffness,
                               atol=5e-13, rtol=3e-11)
    torch.testing.assert_close(asm.assemble_vector(batch_size=17), load,
                               atol=3e-12, rtol=3e-11)


@pytest.mark.parametrize("element_batch_size", [None, 7])
@pytest.mark.parametrize("form", ["mass", "poisson", "load"])
def test_cell_chunks_span_orientations_in_one_pass_per_chunk(monkeypatch, element_batch_size, form):
    asm = poisson(3).from_mesh(orientation_mesh())
    mass, stiffness, load = scalar_reference(asm)
    calls = []
    run_pass = asm._run_pass

    def counted(fn, params, alpha, beta, values, grads, data, zeros, eyes):
        table = grads["u"] if form == "poisson" else values["u"]
        calls.append(table.shape[:2])
        return run_pass(fn, params, alpha, beta, values, grads, data, zeros, eyes)

    monkeypatch.setattr(asm, "_run_pass", counted)
    if form == "load":
        actual = asm.assemble_vector(element_batch_size=element_batch_size)
        expected = load
    else:
        actual = asm(func=(lambda u, v: u * v) if form == "mass" else None,
                     element_batch_size=element_batch_size).to_dense()
        expected = mass if form == "mass" else stiffness
    torch.testing.assert_close(actual, expected, atol=4e-13, rtol=3e-11)
    # 24 different orientations: default is ONE pass, not 24 one-cell passes.
    sizes = [24] if element_batch_size is None else [7, 7, 7, 3]
    assert calls == [(size, asm.transformation["tetra"].n_quadrature) for size in sizes]


@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize("geometry,curved", [(1, False), (2, False), (2, True)])
def test_supported_lagrange_change_of_basis(p, geometry, curved):
    mesh = patch_mesh(geometry, curved)
    modal = poisson(p).from_mesh(mesh, quadrature_order=8)
    nodal = poisson(p, "lagrange").from_mesh(mesh, quadrature_order=8)
    transform = change_of_basis(nodal, modal)
    for form in (None, lambda u, v: u * v):
        expected = transform.T @ nodal(func=form).to_dense() @ transform
        torch.testing.assert_close(modal(func=form).to_dense(), expected,
                                   atol=4e-13, rtol=3e-11)
    torch.testing.assert_close(modal.assemble_vector(), transform.T @ nodal.assemble_vector(),
                               atol=4e-14, rtol=3e-11)


def mixed_form(pressure_family="hierarchical", pressure_order=2):
    class Mixed(MixedElementAssembler):
        fields = [Field("u", "v", 4, components=3, basis="hierarchical"),
                  Field("p", "q", pressure_order, basis=pressure_family)]

        def forward(self, u, v, gradu, gradv, p, q, gradp, gradq, x):
            return ((gradu * gradv).sum() + (u * v).sum()
                    + p * gradv.diagonal().sum() + 2 * q * gradu.diagonal().sum()
                    + (gradp * gradq).sum() + p * q
                    + x[0] * (q * u[0] - 3 * p * v[2]))

        def forward_vector(self, v, gradv, q, gradq, x):
            return (x * v).sum() + gradv[0, 1] + x[0] * q + gradq[1]

    return Mixed


def _mixed_reference(asm):
    weights, reference = independent_quadrature(10)
    matrix = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=DTYPE)
    load = torch.zeros(asm.n_dofs, dtype=DTYPE)
    for cell in range(len(asm.elements[asm.element_types[0]])):
        x, w, tables, _ = cell_tables(asm, cell, reference, weights)
        u, gu = tables["u"]
        p, gp = tables["p"]
        iu, ip = field_indices(asm, "u", cell), field_indices(asm, "p", cell)
        uu = (torch.einsum("qi,qj,q->ij", u, u, w)
              + torch.einsum("qid,qjd,q->ij", gu, gu, w))
        pp = (torch.einsum("qi,qj,q->ij", p, p, w)
              + torch.einsum("qid,qjd,q->ij", gp, gp, w))
        up = torch.einsum("qic,qj,q->icj", gu, p, w)
        up[:, 2, :] -= 3 * torch.einsum("qi,qj,q,q->ij", u, p, x[:, 0], w)
        pu = 2 * torch.einsum("qi,qjc,q->ijc", p, gu, w)
        pu[:, :, 0] += torch.einsum("qi,qj,q,q->ij", p, u, x[:, 0], w)
        scatter_matrix(matrix, iu, iu, torch.kron(uu.contiguous(), torch.eye(3, dtype=DTYPE)))
        scatter_matrix(matrix, ip, ip, pp)
        scatter_matrix(matrix, iu, ip, up.reshape(len(iu), len(ip)))
        scatter_matrix(matrix, ip, iu, pu.reshape(len(ip), len(iu)))
        fu = torch.einsum("qi,qc,q->ic", u, x, w)
        fu[:, 0] += torch.einsum("qi,q->i", gu[:, :, 1], w)
        load[iu] += fu.reshape(-1)
        load[ip] += torch.einsum("qi,q,q->i", p, x[:, 0], w) + gp[:, :, 1].T @ w
    return matrix, load


@pytest.mark.parametrize("pressure_family,pressure_order", [
    ("hierarchical", 1), ("hierarchical", 3), ("lagrange", 1), ("lagrange", 2),
])
def test_nonsymmetric_rectangular_vector_scalar_blocks(pressure_family, pressure_order):
    asm = mixed_form(pressure_family, pressure_order).from_mesh(patch_mesh())
    matrix, load = _mixed_reference(asm)
    actual = asm(batch_size=19, element_batch_size=1)
    torch.testing.assert_close(actual.to_dense(), matrix, atol=5e-13, rtol=3e-11)
    torch.testing.assert_close(asm.assemble_vector(batch_size=19, element_batch_size=1), load,
                               atol=4e-14, rtol=3e-11)
    off = asm.layout.offsets["p"]
    assert matrix[:off, off:].shape[0] != matrix[:off, off:].shape[1]
    assert not torch.allclose(matrix[:off, off:], matrix[off:, :off].T)
    # Executed signatures may leave entire declared blocks inactive.
    only_pressure = asm(func=lambda p, q: p * q).to_dense()
    assert torch.count_nonzero(only_pressure[:off]) == 0
    assert torch.count_nonzero(only_pressure[:, :off]) == 0
    pressure_load = asm.assemble_vector(func=lambda q, x: x[0] * q)
    assert torch.count_nonzero(pressure_load[:off]) == 0
    torch.testing.assert_close(pressure_load[off:],
                               asm.assemble_vector(func=lambda q, x: x[0] * q,
                                                   batch_size=7)[off:])


def data_form(components=1):
    class DataForm(MixedElementAssembler):
        fields = [Field("u", "v", 3, components=components, basis="hierarchical")]

        def forward(self, u, v, w, gradw, a, grada, tag):
            coefficient = 1 + w.square().sum() + 0.2 * gradw.square().sum()
            return (u * v).sum() * (coefficient + a + 0.3 * grada.sum() + tag)

        def forward_vector(self, v, gradv, w, gradw, a, tag):
            return (w * v).sum() + 0.2 * (gradw * gradv).sum() + (a + tag) * v.sum()

    return DataForm


def _data_reference(asm, coefficients, point_values, tags):
    weights, reference = independent_quadrature(12)
    matrix = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=DTYPE)
    load = torch.zeros(asm.n_dofs, dtype=DTYPE)
    components = asm.fields[0].components
    kind = asm.element_types[0]
    for cell in range(len(asm.elements[kind])):
        _, w, tables, (gv, gg) = cell_tables(asm, cell, reference, weights)
        value, grad = tables["u"]
        conn = asm.field_conn[f"u__{kind}"][cell]
        local_coefficients = coefficients[conn].reshape(len(conn), components)
        data = value @ local_coefficients
        gradient = torch.einsum("qbd,bc->qcd", grad, local_coefficients)
        nodal = point_values[asm.elements[kind][cell]]
        a = gv @ nodal
        grada = torch.einsum("qbd,b->qd", gg, nodal)
        factor = (1 + data.square().sum(-1) + 0.2 * gradient.square().sum((-1, -2))
                  + a + 0.3 * grada.sum(-1) + tags[cell])
        local = torch.einsum("qi,qj,q,q->ij", value, value, factor, w)
        indices = field_indices(asm, "u", cell)
        scatter_matrix(matrix, indices, indices,
                       torch.kron(local.contiguous(), torch.eye(components, dtype=DTYPE)))
        rhs = (torch.einsum("qb,qc,q->bc", value, data, w)
               + 0.2 * torch.einsum("qbd,qcd,q->bc", grad, gradient, w)
               + torch.einsum("qb,q,q->b", value, a + tags[cell], w)[:, None])
        load[indices] += rhs.reshape(-1)
    return matrix, load


@pytest.mark.parametrize("components", [1, 3])
def test_nonlinear_field_point_and_original_order_element_data_with_uneven_chunks(components):
    asm = data_form(components).from_mesh(orientation_mesh([17, 0, 17, 9, 0, 17]),
                                        quadrature_order=12)
    rng = torch.Generator().manual_seed(791)
    shape = (asm.layout.n_nodes("u"),) if components == 1 else (asm.layout.n_nodes("u"), components)
    coefficients = 0.2 * torch.randn(shape, dtype=DTYPE, generator=rng)
    point_values = 1 + 0.1 * asm._points[:, 0] - 0.2 * asm._points[:, 1] + asm._points[:, 2]
    tags = torch.tensor([3., 1., 7., 2., 5., 11.], dtype=DTYPE)
    inputs = {"field_data": {"w": ("u", coefficients)},
              "point_data": {"a": point_values}, "element_data": {"tag": {"tetra": tags}}}
    matrix, load = _data_reference(asm, coefficients, point_values, tags)
    assert sorted(group.n_cells for group in asm.grouping.groups) == [1, 2, 3]
    for qchunk, echunk in [(-1, None), (None, 2), (31, 2), (47, 1)]:
        torch.testing.assert_close(asm(**inputs, batch_size=qchunk,
                                       element_batch_size=echunk).to_dense(), matrix,
                                   atol=3e-13, rtol=3e-11)
        torch.testing.assert_close(asm.assemble_vector(**inputs, batch_size=qchunk,
                                                      element_batch_size=echunk), load,
                                   atol=3e-13, rtol=3e-11)


def test_one_occupied_class_with_nondivisible_element_chunks():
    asm = poisson(4).from_mesh(orientation_mesh([7] * 5))
    assert len(asm.grouping.groups) == 1
    matrix, load = asm().to_dense(), asm.assemble_vector()
    torch.testing.assert_close(asm(batch_size=17, element_batch_size=2).to_dense(), matrix,
                               atol=3e-13, rtol=3e-11)
    torch.testing.assert_close(asm.assemble_vector(batch_size=17, element_batch_size=2), load,
                               atol=3e-13, rtol=3e-11)


@pytest.mark.parametrize("size", [0, -1, -3, 1.5, True])
def test_invalid_element_batch_size_rejected_for_both_forms(size):
    asm = poisson(2).from_mesh(patch_mesh())
    for assemble in (asm, asm.assemble_vector):
        with pytest.raises((TypeError, ValueError), match="element_batch_size"):
            assemble(element_batch_size=size)


def test_csr_source_order_and_duplicate_sums_for_matrix_and_load():
    asm = mixed_form().from_mesh(patch_mesh())
    for beta in asm.fields:
        source_conn = torch.cat([asm.field_conn[f"{beta.trial}__{kind}"].reshape(-1)
                                 for kind in asm.element_types])
        projector = asm.grouping.vector_projector[beta.trial]
        assert isinstance(projector, SparseProjector)
        assert projector.projection.layout == torch.sparse_csr
        source = torch.arange(len(source_conn) * beta.components, dtype=DTYPE).reshape(
            len(source_conn), beta.components) / 13
        expected = torch.zeros(asm.layout.n_nodes(beta.trial), beta.components, dtype=DTYPE)
        for slot, destination in enumerate(source_conn):
            expected[destination] += source[slot]
        torch.testing.assert_close(projector(source), expected)
        for alpha in asm.fields:
            key = f"{beta.trial}__{alpha.trial}"
            projector = asm.grouping.pair_projector[key]
            assert isinstance(projector, SparseProjector)
            assert projector.projection.layout == torch.sparse_csr
            size = projector.projection.shape[1]
            assert projector.from_shape == (size,)
            source = torch.arange(size * beta.components * alpha.components, dtype=DTYPE)
            source = source.reshape(size, beta.components, alpha.components) / 29
            expected = torch.zeros(asm.n_dofs, asm.n_dofs, dtype=DTYPE)
            offset = 0
            for kind in asm.element_types:
                for cell in range(len(asm.elements[kind])):
                    rows = field_indices(asm, beta.trial, cell, kind)
                    cols = field_indices(asm, alpha.trial, cell, kind)
                    nb, na = len(rows) // beta.components, len(cols) // alpha.components
                    local = source[offset:offset + nb * na].reshape(
                        nb, na, beta.components, alpha.components)
                    local = local.permute(0, 2, 1, 3).reshape(len(rows), len(cols))
                    scatter_matrix(expected, rows, cols, local)
                    offset += nb * na
            assert offset == size
            actual = torch.zeros_like(expected)
            actual[asm.pair_rows[key], asm.pair_cols[key]] = projector(source).reshape(-1)
            torch.testing.assert_close(actual, expected, atol=3e-11, rtol=2e-14)


def _cache_form():
    class Shared(MixedElementAssembler):
        fields = [Field("u", "v", 4, basis="hierarchical"),
                  Field("p", "q", 4, basis="hierarchical"), Field("s", "t", 2)]

        def forward(self, u, v, p, q, s, t):
            return u * v + p * q + s * t + u * q + s * v

        def forward_vector(self, v, q, t, x):
            return (1 + x[0]) * v + x[1] * q + t

    return Shared


def test_registered_shared_tables_and_csr_hot_path_without_index_add(monkeypatch):
    calls = []
    original = Tetrahedron.get_hierarchical_basis_fns

    def counted(*args, **kwargs):
        calls.append(kwargs["orientation"])
        return original(*args, **kwargs)

    monkeypatch.setattr(Tetrahedron, "get_hierarchical_basis_fns", counted)
    asm = _cache_form().from_mesh(orientation_mesh([8, 0, 8]))
    assert len(calls) == 2
    assert isinstance(asm.grouping.ref_val, BufferDict)
    assert isinstance(asm.grouping.ref_grad, BufferDict)
    assert isinstance(asm.grouping.cells, BufferDict)
    assert isinstance(asm.grouping.pair_projector, torch.nn.ModuleDict)
    assert isinstance(asm.grouping.vector_projector, torch.nn.ModuleDict)
    ukey = asm.grouping.table_keys["u__tetra"]
    assert ukey == asm.grouping.table_keys["p__tetra"]
    assert asm.grouping.ref_val[ukey].shape[0] == 2
    assert asm.grouping.ref_val[ukey].ndim == 3
    assert asm.grouping.ref_grad[ukey].ndim == 4
    assert len(asm.grouping.ref_val) == 1
    torch.testing.assert_close(asm.grouping.table_index["tetra"], torch.tensor([1, 0, 1]))
    # Lagrange fields retain the existing orientation-independent cache and
    # every cell chunk reads this same shared table.
    assert asm.ref_val["s__tetra"].ndim == 2
    assert [key for key in asm.ref_val if key.startswith("s__")] == ["s__tetra"]
    expected_matrix, expected_load = asm().to_dense(), asm.assemble_vector()

    def fail(*args, **kwargs):
        raise AssertionError("Grouped assembly must reuse tables and CSR projection")

    monkeypatch.setattr(Tetrahedron, "get_hierarchical_basis_fns", fail)
    monkeypatch.setattr(ReduceProjector, "__call__", fail)
    monkeypatch.setattr(torch.Tensor, "index_add_", fail)
    torch.testing.assert_close(asm(batch_size=17, element_batch_size=1).to_dense(), expected_matrix)
    torch.testing.assert_close(asm.assemble_vector(batch_size=17, element_batch_size=1), expected_load)


def test_autograd_coordinates_and_modal_coefficients_finite_differences_and_reassembly():
    class Nonlinear(MixedElementAssembler):
        fields = [Field("u", "v", 4, basis="hierarchical")]

        def forward(self, u, v, gradu, gradv, w, gradw, x):
            return (1 + w ** 2 + 0.1 * gradw.square().sum() + x[0] ** 2) * (
                u * v + (gradu * gradv).sum())

        def forward_vector(self, v, gradv, w, gradw):
            return w ** 2 * v + 0.1 * (gradw * gradv).sum()

    asm = Nonlinear.from_mesh(patch_mesh(2, curved=True), quadrature_order=8)
    rng = torch.Generator().manual_seed(817)
    points = asm._points.detach().clone().requires_grad_()
    coefficients = (0.2 * torch.randn(asm.n_dofs, dtype=DTYPE, generator=rng)).requires_grad_()
    dx = torch.randn(points.shape, dtype=DTYPE, generator=rng)
    dc = torch.randn(coefficients.shape, dtype=DTYPE, generator=rng)
    probe = torch.randn(asm.n_dofs, asm.n_dofs, dtype=DTYPE, generator=rng)
    load_probe = torch.randn(asm.n_dofs, dtype=DTYPE, generator=rng)
    table_pointers = {key: value.data_ptr() for key, value in asm.grouping.ref_val.items()}

    def objective(x, c):
        inputs = {"points": x, "field_data": {"w": ("u", c)},
                  "batch_size": 17, "element_batch_size": 2}
        return ((asm(**inputs).to_dense() * probe).sum()
                + (asm.assemble_vector(**inputs) * load_probe).sum())

    # Backpropagate twice without retain_graph: no previous geometry graph may
    # survive in a persistent cache or in a packed local-contribution buffer.
    for _ in range(2):
        gx, gc = torch.autograd.grad(objective(points, coefficients), (points, coefficients))
        h = 2e-6
        for direction, derivative in [("points", (gx * dx).sum()),
                                      ("coefficients", (gc * dc).sum())]:
            xstep = h * dx if direction == "points" else torch.zeros_like(points)
            cstep = h * dc if direction == "coefficients" else torch.zeros_like(coefficients)
            with torch.no_grad():
                difference = (objective(points + xstep, coefficients + cstep)
                              - objective(points - xstep, coefficients - cstep)) / (2 * h)
            torch.testing.assert_close(derivative, difference, atol=3e-7, rtol=3e-6)
    assert table_pointers == {key: value.data_ptr() for key, value in asm.grouping.ref_val.items()}
    assert all(not table.requires_grad for table in asm.grouping.ref_val.values())


def test_grouped_sparse_pattern_is_stable_for_condenser_reuse():
    asm = poisson(4, components=3).from_mesh(patch_mesh())
    first = asm(func=lambda u, v: (u * v).sum())
    second = asm(func=lambda u, v: 2 * (u * v).sum(), element_batch_size=1)
    assert first.has_same_layout(second)
    boundary = torch.zeros(asm.n_dofs, dtype=torch.bool)
    boundary[:3] = True
    condenser = Condenser(boundary)
    rhs = torch.ones(asm.n_dofs, dtype=DTYPE)
    k1, b1 = condenser(first, rhs)
    k2, b2 = condenser(second, rhs)
    torch.testing.assert_close(k2.to_dense(), 2 * k1.to_dense())
    torch.testing.assert_close(b2, b1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_group_cache_state_roundtrip_and_dtype_device_movement(device):
    mesh = orientation_mesh([17, 0, 17])
    asm = _cache_form().from_mesh(mesh)
    matrix, load = asm().to_dense(), asm.assemble_vector()
    other = _cache_form().from_mesh(mesh)
    state = asm.state_dict()
    assert any(key.startswith("grouping.ref_val.") for key in state)
    assert any(key.startswith("grouping.pair_projector.") for key in state)
    other.load_state_dict(state)
    assert other.grouping.table_keys == asm.grouping.table_keys
    other.to(device=device, dtype=torch.float32)
    for table in other.grouping.ref_val.values():
        assert table.device.type == device and table.dtype == torch.float32
    for buffer in (*other.group_cells.values(), *other.cell_orientation.values(),
                   *other.grouping.table_index.values(),
                   *other.field_conn.values()):
        assert buffer.device.type == device and buffer.dtype == torch.int64
    for projector in (*other.grouping.pair_projector.values(),
                      *other.grouping.vector_projector.values()):
        assert projector.projection.dtype == torch.float32
        assert projector.projection.crow_indices().dtype in (torch.int32, torch.int64)
    torch.testing.assert_close(other().to_dense().cpu().double(), matrix, atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(other.assemble_vector().cpu().double(), load, atol=3e-6, rtol=3e-5)
    other.double()
    torch.testing.assert_close(other().to_dense().cpu(), matrix, atol=2e-7, rtol=2e-5)
    torch.testing.assert_close(other.assemble_vector().cpu(), load, atol=2e-7, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_grouped_setup_from_cuda_quadratic_mesh_with_hierarchical_lagrange_fields():
    form = mixed_form(pressure_family="lagrange", pressure_order=1)
    cpu = form.from_mesh(patch_mesh(2, curved=True), quadrature_order=12)
    expected_matrix, expected_load = cpu().to_dense(), cpu.assemble_vector()
    # Construct directly from CUDA geometry: this exercises topology discovery
    # and P1 corner-node numbering, rather than moving an existing CPU assembler.
    gpu = form.from_mesh(patch_mesh(2, curved=True).cuda(), quadrature_order=12)
    assert gpu.grouped_tetrahedra
    for buffer in (*gpu.elements.values(), *gpu.field_conn.values(),
                   *gpu.field_node_ids.values(), *gpu.cell_orientation.values(),
                   *gpu.group_cells.values()):
        assert buffer.is_cuda and buffer.dtype == torch.int64
    assert all(table.is_cuda and table.dtype == DTYPE for table in gpu.grouping.ref_val.values())
    for projector in (*gpu.grouping.pair_projector.values(),
                      *gpu.grouping.vector_projector.values()):
        assert projector.projection.is_cuda
        assert projector.projection.layout == torch.sparse_csr
    actual_matrix = gpu(batch_size=37, element_batch_size=2).to_dense()
    actual_load = gpu.assemble_vector(batch_size=37, element_batch_size=2)
    assert actual_matrix.is_cuda and actual_load.is_cuda
    torch.testing.assert_close(actual_matrix.cpu(), expected_matrix, atol=4e-12, rtol=3e-10)
    torch.testing.assert_close(actual_load.cpu(), expected_load, atol=3e-13, rtol=3e-10)
