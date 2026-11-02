"""Reference-integrand storage estimates shared by the two benchmark drivers.

No TensorMesh import: the process-isolated driver must import its selected
checkout before inspecting an assembler. Call only after assembly measurements.
"""


def reference_report(asm, fn, matrix, batch_size, element_batch_size,
                     reference_mode="auto", field_data=None):
    if not asm.grouped_tetrahedra:
        return {"strategy": "existing_nodal_path"}
    params = asm._classify_params(fn, {"x": asm._points}, {}, {}, field_data or {})
    grouping = asm.grouping
    eligible = grouping.shares_reference(params)
    shared = grouping.shares_reference(params, reference_mode)
    report = {"mode": reference_mode, "eligible": eligible,
              "strategy": "orientation_shared" if shared else "cellwise",
              "estimate_scope": "final weak-form integrand only; not allocator peak or workspace"}
    if not eligible:
        return report
    if matrix:
        pairs = asm._executed_pairs(asm.fields, params)
    else:
        active = {f.trial for _, role, f in params if role == "test_val"}
        pairs = [(None, f) for f in asm.fields if f.trial in active]
    by_type = {}
    for kind in grouping.element_types:
        count = len(asm.elements[kind])
        step = min(element_batch_size or count, count)
        plans = grouping.reference_chunks(kind, element_batch_size) if shared else None
        occupied = sum(g.element_type == kind for g in grouping.groups)
        nq = asm.transformation[kind].n_quadrature
        nq = nq if batch_size in (-1, None) else min(nq, batch_size)
        selected_bytes = cell_bytes = 0
        for chunk, start in enumerate(range(0, count, step)):
            cells = min(step, count - start)
            for alpha, beta in pairs:
                fields = [beta] if alpha is None else [beta, alpha]
                width = 1
                for field in fields:
                    width *= asm.field_conn[f"{field.trial}__{kind}"].shape[1] * field.components
                oriented = any(f.basis == "hierarchical" for f in fields)
                cell_rows = cells if oriented and occupied > 1 else 1
                rows = plans[chunk].n_orientations if shared and oriented else cell_rows
                selected_bytes = max(selected_bytes, rows * nq * width * asm._points.element_size())
                cell_bytes = max(cell_bytes, cell_rows * nq * width * asm._points.element_size())
        by_type[kind] = {
            "max_integrand_bytes": selected_bytes,
            "cell_mode_max_integrand_bytes": cell_bytes,
            "cell_chunks": (count + step - 1) // step,
            "max_orientations_per_chunk": max(p.n_orientations for p in plans) if plans else None,
            "max_packing_buckets": max(len(p.buckets) for p in plans) if plans else None,
            "max_padded_cell_slots": max(p.padded_cells for p in plans) if plans else None}
    report["element_types"] = by_type
    return report
