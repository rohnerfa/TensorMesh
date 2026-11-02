"""Benchmark numerical oracles and streaming reconstruction, not speed thresholds."""

import json

import pytest
import torch

from examples.poisson.benchmark_hierarchical_tetrahedra import (
    assembler_class, cube_mesh, data_for, main, numerical_diagnostics,
    reconstruct, run_benchmark,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("orientation", ["balanced", "single", "skewed"])
def test_generated_cube_has_unit_volume_and_conforming_interior_faces(orientation):
    mesh = cube_mesh(2, orientation)
    cells = mesh.elements()
    coordinates = mesh.points[cells]
    jac = coordinates[:, 1:] - coordinates[:, :1]
    volumes = torch.linalg.det(jac).abs() / 6
    torch.testing.assert_close(volumes, torch.full_like(volumes, 1 / 48))
    faces = cells[:, torch.tensor([[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]])]
    faces, counts = torch.unique(faces.reshape(-1, 3).sort(-1).values,
                                 dim=0, return_counts=True)
    assert bool(((counts == 1) | (counts == 2)).all())
    assert int((counts == 1).sum()) == 48
    boundary = mesh.points[faces[counts == 1]]
    # Each singleton is on a cube face: there are no unmatched interior facets.
    on_plane = ((boundary == 0).all(1) | (boundary == 1).all(1)).any(1)
    assert bool(on_plane.all())


@pytest.mark.parametrize("form", ["mass", "poisson", "mixed", "field_data"])
@pytest.mark.parametrize("reference_mode", ["auto", "cell"])
def test_stage_reconstruction_matches_chunked_public_assembly_and_analytic_cube(form, reference_mode):
    mesh = cube_mesh(1)
    asm = assembler_class(2, form).from_mesh(mesh)
    field_data = data_for(asm, form, 0.5)
    kwargs = {"field_data": field_data, "batch_size": 3, "element_batch_size": 4,
              "reference_mode": reference_mode}
    with torch.no_grad():
        matrix = asm(**kwargs)
        load = asm.assemble_vector(**kwargs)
        reconstructed, matrix_stages = reconstruct(asm, True, **kwargs)
        vector, vector_stages = reconstruct(asm, False, **kwargs)
        torch.testing.assert_close(reconstructed.row, matrix.row)
        torch.testing.assert_close(reconstructed.col, matrix.col)
        torch.testing.assert_close(reconstructed.values, matrix.values, atol=2e-12, rtol=2e-12)
        torch.testing.assert_close(vector, load, atol=2e-12, rtol=2e-12)
        assert numerical_diagnostics(asm, matrix, load, form, 0.5)["passed"]
    for stages in (matrix_stages, vector_stages):
        assert {"geometry_data_and_physical_gradients", "local_buffer_packing",
                "csr_projection"} <= stages.keys()
        assert {"reference_form_evaluation", "cell_form_evaluation"} & stages.keys()
        assert {"reference_contraction", "cell_quadrature"} & stages.keys()
    shared_matrix = reference_mode == "auto" and form == "mass"
    assert ("reference_contraction" in matrix_stages) == shared_matrix


def test_p4_face_and_cell_modes_reconstruct_with_promoted_quadrature():
    asm = assembler_class(4, "poisson").from_mesh(cube_mesh(1, "skewed"), quadrature_order=6)
    assert asm.transformation["tetra"].quadrature_order == 7
    with torch.no_grad():
        matrix = asm(batch_size=5, element_batch_size=2)
        result, _ = reconstruct(asm, batch_size=5, element_batch_size=2)
        torch.testing.assert_close(result.values, matrix.values, atol=2e-12, rtol=2e-12)
        assert numerical_diagnostics(asm, matrix, asm.assemble_vector(), "poisson")["passed"]


def test_report_has_analytic_checks_reassembly_and_same_rule_lagrange_comparisons():
    report = run_benchmark(degrees=(2,), forms=("mass", "field_data"), repeats=1, warmup=0,
                           batch_size=4, element_batch_size=1, quadrature_order=5,
                           compare_lagrange=True)
    assert report["status"] == "completed", report["cases"]
    assert len(report["cases"]) == 4
    assert report["source_sha256"]
    assert not report["gpu_performance_measured"]
    for modal, nodal in zip(report["cases"][::2], report["cases"][1::2]):
        rule = modal["quadrature"]["tetra"]
        assert rule["requested_exactness_degree"] == 5
        assert rule["actual_rule_index"] == 6
        assert rule["sha256"] == nodal["quadrature"]["tetra"]["sha256"]
        assert modal["reconstruction"]["matrix_and_load_match_public_assembly"]
        assert sum(modal["orientation_group_histogram"].values()) == 6
        assert modal["storage"]["reference_tables_have_no_element_axis"]
        assert modal["cell_batching"] == "original_order_across_orientations"
        assert modal["storage"]["cell_batches_per_quadrature_sweep"] == {"tetra": 6}
        strategy = "orientation_shared" if modal["form"] == "mass" else "cellwise"
        assert modal["reference_integration"]["matrix"]["strategy"] == strategy
        assert "oriented_reference_tabulation" in modal["setup"]["exclusive_cpu_stages_seconds"]
        for case in (modal, nodal):
            assert case["numerical"]["passed"]
            assert not case["autograd_enabled"]
            timing = case["timings"]["full_matrix_assembly"]
            assert len(timing["samples_seconds"]) == 1
            assert "first_call_seconds" in timing
            assert timing["gpu_memory"]["status"] == "unavailable"
            if case["form"] == "field_data":
                changed = case["changing_field_data"]["diagnostics"]
                assert changed[0]["actual"]["linear_coordinate_energy"] != \
                    changed[1]["actual"]["linear_coordinate_energy"]
    json.dumps(report, allow_nan=False)


def test_cli_writes_machine_readable_report_and_separately_solves(tmp_path, capsys):
    output = tmp_path / "report.json"
    assert main(["--degrees", "2", "--forms", "poisson", "--repeats", "1", "--warmup", "0",
                 "--element-batch-size", "0", "--solve", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert report["cases"][0]["solve"]["status"] == "solved"


def test_float32_mixed_and_changed_field_data_numerical_diagnostics():
    report = run_benchmark(degrees=(2,), forms=("mixed", "field_data"), dtype="float32",
                           repeats=1, warmup=0, batch_size=3, element_batch_size=1)
    assert report["status"] == "completed", report["cases"]
    assert all(case["numerical"]["passed"] for case in report["cases"])


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires unavailable CUDA")
def test_requested_cuda_reports_unavailable_without_fabricated_measurements():
    report = run_benchmark(device="cuda", degrees=(1,), forms=("mass",))
    assert report["status"] == "unavailable"
    assert report["cases"] == []
    assert "CUDA unavailable" in report["reason"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_benchmark_reconstructs_and_checks_changed_coefficient_energy():
    report = run_benchmark(device="cuda", degrees=(2,), forms=("poisson", "field_data"),
                           repeats=1, warmup=0, batch_size=3, element_batch_size=1)
    assert report["status"] == "completed", report["cases"]
    for case in report["cases"]:
        assert case["numerical"]["passed"]
        assert case["reconstruction"]["matrix_and_load_match_public_assembly"]
        memory = case["timings"]["full_matrix_assembly"]["gpu_memory"]
        assert memory["status"] == "measured"
        assert "peak_allocated_bytes" in memory and "peak_reserved_bytes" in memory
