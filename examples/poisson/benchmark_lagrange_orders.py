"""Paired serial tetrahedron assembly benchmark in isolated checkouts.

    python examples/poisson/benchmark_lagrange_orders.py --baseline /path/to/upstream

One CPU thread, float64, deterministic affine mesh, equal quadrature for equal
field degree. Mesh generation, verification and the first form call are untimed.
Setup measures constructing both forms; assembly measures them separately.
"""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def worker(n, feature):
    import meshio
    import torch
    import tensormesh
    from tensormesh import ElementAssembler, Mesh, NodeAssembler
    from tensormesh.element import Tetrahedron

    assert Path(tensormesh.__file__).resolve().parent == Path.cwd() / "tensormesh"
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float64)

    class Laplace(ElementAssembler):
        def forward(self, gradu, gradv):
            return gradu @ gradv

    class Source(NodeAssembler):
        def forward(self, v, x):
            return v * (1 + x.sum())

    # Six tetrahedra per cube. Integer lattice coordinates also identify
    # elevated nodal DOFs independently of the implementation being timed.
    corners = []
    for origin in itertools.product(range(n), repeat=3):
        for axes in itertools.permutations(range(3)):
            path = [list(origin)]
            for axis in axes:
                pt = path[-1].copy()
                pt[axis] += 1
                path.append(pt)
            corners.append(path)
    corners = torch.tensor(corners)
    perm = torch.rand(len(corners), 4, generator=torch.Generator().manual_seed(582)).argsort(1)
    corners = corners.gather(1, perm[..., None].expand_as(corners))

    def mesh(p):
        ref = Tetrahedron.get_basis(p, torch.float64)
        bary = (p * torch.cat((1 - ref.sum(1, keepdim=True), ref), 1)).round().long()
        lattice = bary @ corners
        points, ids = torch.unique(lattice.reshape(-1, 3), dim=0, return_inverse=True)
        cells = ids.reshape(len(corners), -1)
        kind = "tetra" if p == 1 else f"tetra{cells.shape[1]}"
        m = Mesh(meshio.Mesh((points.double() / (n*p)).numpy(), [(kind, cells.numpy())]),
                 reorder=False).double()
        return m, points

    def elapsed(fn, count):
        start = time.perf_counter_ns()
        for _ in range(count):
            fn()
        return (time.perf_counter_ns() - start) / (count * 1e6)

    def digest(t):
        return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()

    meshes = {p: mesh(p) for p in (1, 2, 3)}
    times, checks = {}, {}
    for geom, p in [(1, 1), (2, 2), (3, 3)] + ([(1, 3)] if feature else []):
        m, pts = meshes[geom]
        args = dict(quadrature_order=2*p)
        if p != geom:
            args["field_order"] = p

        def setup():
            a = Laplace.from_mesh(m, **args)
            b = Source.from_mesh(m, **args) if p == geom else Source.from_assembler(a)
            return a, b

        a, b = setup()
        K, f = a(), b()
        if geom == p:
            checks[p] = [digest(x) for x in (K.row, K.col, K.values, f)]
            if p == 3:
                ref_K, ref_f = K, f
        else:
            positions = (a.field_layout().points("u") * (n*p)).round().long()
            lookup = {tuple(pos): i for i, pos in enumerate(meshes[p][1].tolist())}
            ids = torch.tensor([lookup[tuple(pos)] for pos in positions.tolist()])
            assert len(ids.unique()) == len(ref_f)
            keys = ids[K.row] * len(f) + ids[K.col]
            idx = keys.argsort()
            assert torch.equal(keys[idx], ref_K.row * len(f) + ref_K.col)
            torch.testing.assert_close(K.values[idx], ref_K.values, atol=2e-12, rtol=2e-12)
            torch.testing.assert_close(f, ref_f[ids], atol=2e-12, rtol=2e-12)
        stats = {}
        for name, fn, count in [("setup", setup, 1), ("matrix", a, 5), ("load", b, 5)]:
            fn()
            stats[name] = statistics.median(elapsed(fn, count) for _ in range(7))
        times[f"P{p}/P{geom}"] = stats
    return {"times": times, "checks": checks, "cells": len(corners)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--cells-per-axis", type=int, default=4)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--json", type=Path, help="write all paired observations (milliseconds)")
    ap.add_argument("--worker", choices=["baseline", "feature"], help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        print(json.dumps(worker(args.cells_per_axis, args.worker == "feature")))
        return
    if not args.baseline or args.pairs < 1 or args.cells_per_axis < 1:
        ap.error("provide --baseline and positive mesh and repetition counts")
    script = Path(__file__).resolve()
    roots = {"feature": script.parents[2], "baseline": args.baseline.resolve()}
    threads = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                "VECLIB_MAXIMUM_THREADS")}
    pairs = []
    for i in range(args.pairs):
        pair = {}
        for role in (("baseline", "feature") if i % 2 == 0 else ("feature", "baseline")):
            proc = subprocess.run(
                [sys.executable, str(script), "--worker", role, "--cells-per-axis", str(args.cells_per_axis)],
                cwd=roots[role], env={**os.environ, **threads, "PYTHONHASHSEED": "0",
                                      "PYTHONPATH": str(roots[role])}, capture_output=True, text=True)
            if proc.returncode:
                raise RuntimeError(proc.stderr)
            pair[role] = json.loads(proc.stdout.splitlines()[-1])
        assert pair["feature"]["checks"] == pair["baseline"]["checks"], "nodal output changed"
        pairs.append(pair)
    if args.json:
        args.json.write_text(json.dumps(pairs, indent=2))
    print(f"{pairs[0]['baseline']['cells']} tetrahedra, {len(pairs)} alternating CPU pairs")
    print("field/geometry  stage    upstream median [min,max] ms | feature median [min,max] ms | paired Δ")
    for key in pairs[0]["feature"]["times"]:
        for stage in ("setup", "matrix", "load"):
            # Upstream cannot construct P3/P1. Compare with the same P3 space
            # on explicitly elevated affine cells; label that comparison.
            base_key = "P3/P3" if key == "P3/P1" else key
            base = [r["baseline"]["times"][base_key][stage] for r in pairs]
            feat = [r["feature"]["times"][key][stage] for r in pairs]
            fmt = lambda x: f"{statistics.median(x):.3f} [{min(x):.3f},{max(x):.3f}]"
            delta = statistics.median(100*(f/b-1) for b, f in zip(base, feat))
            print(f"{key:8s} {stage:7s} {fmt(base):26s} {fmt(feat):26s} {delta:+.1f}%"
                  + (" (upstream P3/P3)" if key != base_key else ""))


if __name__ == "__main__":
    main()
