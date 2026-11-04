"""Nodal P3 Allen-Cahn, with degree-12 quadrature and otherwise original settings.

Run from the repository root:
    python examples/diffusion/allen-cahn/ac_convex_concave_p3.py --device cpu
"""
import os
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from tqdm import tqdm

from tensormesh import Mesh
from tensormesh.dataset import PoissonMultiFrequency
from ac_convex_concave import KAssembler, RAssembler, parse_arguments


def simulate(device="cpu", chara_length=0.02, steps=200, max_iter=50, dt=1e-6, epsilon=220,
             seed=0, n_freq=24, order=3, quadrature_order=12):
    mesh = Mesh.gen_rectangle(chara_length=chara_length, element_type="tri", order=order).to(device=device)

    generator = torch.Generator(device=mesh.points.device).manual_seed(seed)
    a = torch.empty(n_freq, n_freq, device=mesh.points.device).uniform_(-1, 1, generator=generator)
    dataset = PoissonMultiFrequency(a=a, r=1)
    cold = dataset.source_term(mesh.points)
    print(f"mesh: {mesh.n_points} nodes, {mesh.n_elements} P{order} triangles on {device}", flush=True)

    params = dict(dt=dt, epsilon=epsilon)
    R_asm = RAssembler.from_mesh(mesh, quadrature_order=quadrature_order, **params)
    K_asm = KAssembler.from_mesh(mesh, quadrature_order=quadrature_order, **params)

    cs = [cold]

    with tqdm(range(steps), desc="Time stepping") as pbar:
        for _ in pbar:
            c = cold
            rnorm = torch.tensor(float("inf"))
            for newton_step in range(max_iter):
                point_data = {"c": c, "cold": cold}
                K = K_asm(mesh.points, point_data=point_data)
                R = R_asm(mesh.points, point_data=point_data)

                c = c + K.solve(-R)

                rnorm = torch.linalg.norm(R)
                if rnorm < 1e-10:
                    break

            cold = c
            cs.append(cold)
            pbar.set_postfix({"rnorm": float(rnorm), "newton_iter": newton_step + 1})

    return mesh, cs


def main():
    args = parse_arguments()
    print(f"PyTorch {torch.__version__}, CPU threads: {torch.get_num_threads()}", flush=True)
    start = perf_counter()
    mesh, cs = simulate(device=args.device, chara_length=args.chara_length,
                        steps=args.steps, max_iter=args.max_iter, dt=args.dt,
                        epsilon=args.epsilon, seed=args.seed, n_freq=args.n_freq)
    print(f"Simulation wall time (mesh + setup + stepping): {perf_counter() - start:.3f} s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"Allen-Cahn-convex-concave-P3-dt{args.dt:g}.mp4")
    start = perf_counter()
    mesh.plot(values={"phi": cs}, show_mesh=False, dt=args.dt, save_path=out)
    print(f"Video rendering wall time: {perf_counter() - start:.3f} s", flush=True)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
