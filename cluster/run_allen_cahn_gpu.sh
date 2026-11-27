#!/usr/bin/env bash
# Run inside an existing Euler GPU-backed JupyterLab terminal allocation.
set -euo pipefail
cd "$(dirname "$0")/.."
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable: use an allocated GPU session"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python examples/diffusion/allen-cahn/benchmark_four_way.py \
  --upstream ../baseline --current . \
  --device cuda --linear-solver torch-cg-jacobi --threads 2 \
  --order 4 --chara-length 0.05 --input cluster/inputs/p4/input.npz \
  --steps 2 --repeats 1 --output dist/euler-allen-cahn-p4-smoke "$@"
