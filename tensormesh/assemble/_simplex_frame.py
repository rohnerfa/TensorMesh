"""Private geometry connectivity for canonical hierarchical simplex frames."""

from functools import lru_cache
from itertools import permutations
from math import comb

import torch

from ..element import element_type2element, element_type2order


def _canonical_reference(points, perm):
    """Express original-frame reference points in sorted-corner coordinates."""
    bary = torch.cat((1 - points.sum(-1, keepdim=True), points), -1)
    if perm.ndim == 1:  # one frame, shared [Q,D] rule
        return bary[:, perm[1:]]
    return bary.gather(-1, perm[:, None, 1:].expand(-1, points.shape[1], -1))


@lru_cache(maxsize=None)
def _geometry_permutations(element, order):
    """CPU lookup tables, independent of cell IDs and physical coordinates."""
    nv = element.n_vertex
    nodes = element.get_basis(order, torch.float64)

    # Integer barycentric weights in TensorMesh node order, including the
    # weight of vertex zero. Only complete equispaced simplices are supported.
    bary = torch.cat((1 - nodes.sum(dim=1, keepdim=True), nodes), dim=1) * order
    keys = bary.round().long()
    slots = {tuple(key): i for i, key in enumerate(keys.tolist())}
    if (not torch.allclose(bary, keys.double(), atol=1e-10, rtol=0)
            or (keys < 0).any() or (keys.sum(1) != order).any()
            or not torch.equal(keys[:nv], order * torch.eye(nv, dtype=keys.dtype, device=keys.device))
            or len(slots) != len(nodes) or len(nodes) != comb(order + nv - 1, nv - 1)):
        raise ValueError(f"unsupported simplex reference lattice for {element.__name__} P{order}")

    # At most 24 rows. Index them by the tuple of original corner slots.
    row_for_perm = keys.new_full((nv,) * nv, -1)
    columns = []
    old_keys = torch.empty_like(keys)
    for row, perm in enumerate(permutations(range(nv))):
        old_keys[:, perm] = keys  # lambda_old[perm[j]] = lambda_canonical[j]
        columns.append([slots[tuple(key)] for key in old_keys.tolist()])
        row_for_perm[perm] = row
    columns = keys.new_tensor(columns)
    slots_in_order = torch.arange(len(nodes), device=columns.device)
    if not torch.equal(columns.sort(dim=1).values, slots_in_order.expand_as(columns)):
        raise ValueError(f"non-bijective reference-node permutation for {element.__name__} P{order}")
    return row_for_perm, columns


def _canonical_simplex(
    kind: str, cells: torch.Tensor, n_points: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return private geometry rows and their canonical-to-original corner slots.

    ``cells`` is one block in TensorMesh's internal geometry-node order.
    Cell rows, mesh point IDs, device and connectivity dtype are preserved.
    ``perm[e, j]`` is the old corner slot of canonical corner ``j`` in cell
    ``e``. It also maps canonical reference vertices into the original frame.
    """
    if kind not in element_type2order or not kind.startswith(("triangle", "tetra")):
        raise NotImplementedError(f"unsupported simplex geometry type: {kind!r}")
    element, order = element_type2element(kind), element_type2order[kind]
    nv = element.n_vertex
    width = comb(order + nv - 1, nv - 1)
    if not isinstance(cells, torch.Tensor) or cells.ndim != 2 or cells.shape[1] != width:
        raise ValueError(f"{kind} geometry connectivity must have shape [n_cells, {width}]")
    if cells.dtype not in (torch.int32, torch.int64):
        raise ValueError("geometry connectivity must contain integer point IDs")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    if cells.numel():
        low, high = torch.aminmax(cells)
        if low.item() < 0 or high.item() >= n_points:
            raise ValueError("geometry point IDs must lie in [0, n_points)")

    corners, perm = cells[:, :nv].sort(dim=1)
    if (corners[:, 1:] == corners[:, :-1]).any():
        raise ValueError("simplex corners must have distinct point IDs")
    if order == 1:
        return corners, perm
    row_for_perm, columns = _geometry_permutations(element, order)
    rows = row_for_perm.to(cells.device)[tuple(perm.unbind(dim=1))]
    return cells.gather(1, columns.to(cells.device)[rows]), perm
