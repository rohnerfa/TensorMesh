"""Nodal P2 Allen-Cahn with degree-8 quadrature and original physical settings.

Run from the repository root:
    python examples/diffusion/allen-cahn/ac_convex_concave_p2.py --device cpu
"""
import os
from time import perf_counter

from ac_convex_concave_p3 import simulate
from ac_convex_concave import parse_arguments
import torch


def main():
    args = parse_arguments()
    print(f"PyTorch {torch.__version__}, CPU threads: {torch.get_num_threads()}", flush=True)
    start = perf_counter()
    mesh, cs = simulate(device=args.device, chara_length=args.chara_length,
                        steps=args.steps, max_iter=args.max_iter, dt=args.dt,
                        epsilon=args.epsilon, seed=args.seed, n_freq=args.n_freq,
                        order=2, quadrature_order=8)
    print(f"Simulation wall time (mesh + setup + stepping): {perf_counter() - start:.3f} s", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"Allen-Cahn-convex-concave-P2-dt{args.dt:g}.mp4")
    start = perf_counter()
    mesh.plot(values={"phi": cs}, show_mesh=False, dt=args.dt, save_path=out)
    print(f"Video rendering wall time: {perf_counter() - start:.3f} s", flush=True)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
