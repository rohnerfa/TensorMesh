"""Performance instrumentation must preserve numerical outputs and source isolation."""

import json

import pytest
import torch

from examples.poisson._tetrahedron_profile import profile_matrix
from tests.assemble._tetrahedron_fixtures import orientation_mesh, poisson


@pytest.mark.parametrize("mode", ["auto", "cell"])
def test_matrix_profiling_preserves_chunked_assembly_and_restores_methods(tmp_path, mode):
    asm = poisson(3).from_mesh(orientation_mesh([0, 5, 17, 0, 5, 17, 23]))
    fn = lambda: asm(func=lambda u, v: u * v, batch_size=17,
                     element_batch_size=5, reference_mode=mode)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            expected = fn()
            names = set(vars(asm))
            grouping_names = set(vars(asm.grouping))
            original_projector_call = type(asm.grouping.pair_projector["u__u"]).__call__
            result = profile_matrix(asm, fn, asm.device, 2, 1, tmp_path / "trace.json")
            torch.testing.assert_close(fn().values, expected.values)
        assert set(vars(asm)) == names
        assert set(vars(asm.grouping)) == grouping_names
        assert type(asm.grouping.pair_projector["u__u"]).__call__ is original_projector_call
        assert result["output_equivalence"] == "passed"
        assert {"module_dtype_conversion", "module_device_conversion", "weak_form_evaluation",
                "quadrature_contraction", "global_projection", "other_assembly"} <= result["stages"].keys()
        assert result["projection_probe"]["u__u"]["equivalence"] == "passed"
        assert result["projection_probe"]["u__u"]["source_entries"] == 7 * 20**2
        # Exclusive samples partition the instrumented call, including residual overhead.
        for i, total in enumerate(result["instrumented_total"]["samples_seconds"]):
            assert sum(s["samples_seconds"][i] for s in result["stages"].values()) == pytest.approx(total)
        trace = json.loads((tmp_path / "trace.json").read_text())
        assert any(event.get("name") == "assembly::global_projection" for event in trace["traceEvents"])
        json.dumps(result, allow_nan=False)
    finally:
        torch.set_num_threads(previous)
