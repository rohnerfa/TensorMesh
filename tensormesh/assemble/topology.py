r"""Shared mesh-topology helpers for the assemblers.

The sparsity pattern of a Galerkin matrix is the set of unique
``(row, col)`` node pairs touched by any element. :func:`build_edges`
computes that pattern — together with the per-element scatter indices —
for both the square single-space case (:class:`~tensormesh.assemble.ElementAssembler`)
and the rectangular two-space case (the off-diagonal blocks of a mixed
multi-field form).

:func:`lagrange_dofmap` builds the global DOF numbering of a Lagrange
space of **arbitrary order on a mesh of any (single) order** from the
mesh *topology* — vertices, unique edges, cell interiors — instead of
reusing mesh nodes. This is what decouples the field order from the
mesh order (generalized Taylor–Hood pairs, quadratic fields on linear
gmsh imports); :func:`lagrange_boundary_mask` marks its boundary DOFs
by facet incidence, without any geometric predicate.

:func:`hierarchical_dofmap` shares the vertex/edge entity numbering for
triangle modal spaces, keeping edge modes in degree order and returning
parity-dependent orientation signs instead of reversing nodal indices.
Tetrahedral modal spaces use :func:`hierarchical_tetrahedron_dofmap` and
vertex-oriented reference evaluators, with no coefficient signs.
"""
from typing import Dict, List, Mapping, NamedTuple, Tuple

import numpy as np
import scipy.sparse
import torch

from ..element import (TETRAHEDRON_ORIENTATIONS, Tetrahedron, Triangle,
                       element_type2element, element_type2order)

__all__ = ["build_edges", "lagrange_dofmap", "lagrange_boundary_mask", "boundary_facet_mask",
           "LagrangeDofMap", "HierarchicalDofMap", "hierarchical_dofmap",
           "hierarchical_boundary_mask", "HierarchicalTetrahedronDofMap",
           "hierarchical_tetrahedron_dofmap", "hierarchical_tetrahedron_boundary_mask"]


def build_edges(
    conn_pairs: Mapping[str, Tuple[torch.Tensor, torch.Tensor]],
    shape: Tuple[int, int],
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    r"""Deduplicate the ``(row, col)`` basis-pair indices of a bilinear form.

    For every element type, every element contributes all pairs of one
    row-connectivity entry and one column-connectivity entry. The unique
    pairs form the COO pattern (the "edges") of the assembled sparse
    matrix; the returned per-slot edge ids drive the
    :class:`~tensormesh.assemble.projector.ReduceProjector` scatter.

    Row and column connectivity may come from two *different* function
    spaces (e.g. the velocity–pressure coupling block of a Taylor–Hood
    form), so the pattern may be rectangular.

    Parameters
    ----------
    conn_pairs : Mapping[str, Tuple[torch.Tensor, torch.Tensor]]
        Maps each element type to ``(row_conn, col_conn)`` integer tensors
        of shape ``[n_element, n_row_basis]`` / ``[n_element, n_col_basis]``
        (same ``n_element``), holding node indices in ``range(shape[0])``
        and ``range(shape[1])`` respectively.
    shape : Tuple[int, int]
        ``(n_row_nodes, n_col_nodes)`` bounds of the index space.

    Returns
    -------
    edges : torch.Tensor
        Long tensor of shape ``[2, n_edges]`` listing the unique
        ``(row, col)`` pairs in CSR (row-major) order.
    elem_eids : Dict[str, torch.Tensor]
        Maps each element type to a long tensor of shape
        ``[n_element * n_row_basis * n_col_basis]`` holding the edge id of
        every ``(element, i, j)`` slot, flattened in ``(e, i, j)`` order —
        exactly the flatten order used by
        ``ReduceProjector(from_shape=(n_element, n_row_basis, n_col_basis))``.
    """
    flat: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for element_type, (row_conn, col_conn) in conn_pairs.items():
        assert row_conn.shape[0] == col_conn.shape[0], (
            f"row/col connectivity of '{element_type}' disagree on n_element: "
            f"{row_conn.shape[0]} vs {col_conn.shape[0]}"
        )
        n_row_basis, n_col_basis = row_conn.shape[1], col_conn.shape[1]
        elem_u = row_conn[:, :, None].expand(-1, -1, n_col_basis).reshape(-1)
        elem_v = col_conn[:, None, :].expand(-1, n_row_basis, -1).reshape(-1)
        # .copy(): scipy's fancy indexing rejects the non-writeable views
        # .numpy() can return
        flat[element_type] = (
            elem_u.cpu().numpy().copy(),
            elem_v.cpu().numpy().copy(),
        )

    all_u = np.concatenate([u for u, _ in flat.values()])
    all_v = np.concatenate([v for _, v in flat.values()])

    tmp = scipy.sparse.coo_matrix((  # used to remove duplicated edges
        np.ones_like(all_u),  # data
        (all_u, all_v),       # (row, col)
    ), shape=shape).tocsr().tocoo()
    edge_u, edge_v = tmp.row, tmp.col
    num_edges = len(edge_u)
    eids_csr = scipy.sparse.coo_matrix((
        np.arange(num_edges), (edge_u, edge_v)
    ), shape=shape).tocsr()

    elem_eids = {
        element_type: torch.from_numpy(np.array(eids_csr[u, v]).ravel()).long()
        for element_type, (u, v) in flat.items()
    }
    edges = torch.from_numpy(np.stack([edge_u, edge_v], 0)).long()
    return edges, elem_eids


class LagrangeDofMap(NamedTuple):
    r"""Global DOF numbering of a Lagrange space built from mesh topology.

    Attributes
    ----------
    conn : Dict[str, torch.Tensor]
        Per element type, the field-local connectivity ``[n_element, n_basis_f]``:
        slot ``i`` holds the global DOF of the field's ``i``-th reference
        basis function on that element.
    n_dofs : int
        Total number of scalar DOF carriers of the space.
    vertex_ids : torch.Tensor
        Mesh point ids of the vertex DOFs, sorted — ``[n_vertex_dofs]``.
        Vertex DOFs occupy the leading ``[0, n_vertex_dofs)`` block.
    ref_nodes : Dict[str, torch.Tensor]
        Per element type, the field's reference node coordinates
        ``[n_basis_f, D]`` (float64) — slot-aligned with ``conn``; used to
        evaluate interpolation tables and to push DOF coordinates through
        the isoparametric geometry map.
    """
    conn: Dict[str, torch.Tensor]
    n_dofs: int
    vertex_ids: torch.Tensor
    ref_nodes: Dict[str, torch.Tensor]


class HierarchicalDofMap(NamedTuple):
    r"""Triangle modal connectivity and local-to-global orientation factors.

    ``conn[type][K, i]`` indexes the global coefficient of local function i.
    For a global coefficient vector u, use
    ``u_local = sign[type] * u[conn[type]]`` before basis evaluation.
    Equivalently multiply each local basis function by its sign. Signs must
    act on both test and trial functions in assembly, including load vectors.

    Numbering is compact vertices, unique sorted edges (ascending degree
    within each edge), then private cell interiors. Outputs are CPU tensors:
    conn and vertex_ids are long; sign is int8 (+1 or -1), ready to cast
    once to the assembly dtype/device. vertex_ids records the original mesh
    point IDs in sorted order. Unlike LagrangeDofMap, there are no ref_nodes:
    modal coefficients do not represent values at interpolation points.
    """
    conn: Dict[str, torch.Tensor]
    sign: Dict[str, torch.Tensor]
    n_dofs: int
    vertex_ids: torch.Tensor


def _number_vertices_and_edges(per_etype, n_points, order):
    """Shared entity numbering; values are (element class, corner IDs)."""
    vertex_ids = torch.unique(torch.cat([c.reshape(-1) for _, c in per_etype.values()]))
    g2l = torch.full((n_points,), -1, dtype=torch.long)
    g2l[vertex_ids] = torch.arange(vertex_ids.shape[0], dtype=torch.long)
    edge_ids = {}
    n_global_edges = 0
    if order >= 2:
        chunks, sizes = [], []
        for element_type, (elem_cls, corners) in per_etype.items():
            pairs = corners[:, elem_cls.edge]
            chunks.append(torch.sort(pairs, dim=-1).values.reshape(-1, 2))
            sizes.append((element_type, pairs.shape[0], pairs.shape[1]))
        unique_edges, inverse = torch.unique(torch.cat(chunks, dim=0), dim=0,
                                            return_inverse=True)
        n_global_edges = unique_edges.shape[0]
        offset = 0
        for element_type, n_element, n_le in sizes:
            edge_ids[element_type] = inverse[offset:offset + n_element * n_le] \
                .reshape(n_element, n_le)
            offset += n_element * n_le
    return vertex_ids, g2l, edge_ids, n_global_edges


def hierarchical_dofmap(elements: Mapping[str, torch.Tensor],
                        n_points: int, order: int) -> HierarchicalDofMap:
    r"""Build the uniform-order hierarchical H1 DOF map for triangles only.

    The API parallels lagrange_dofmap. Only the first three connectivity
    columns (corner vertices) are read, independently of geometry order.
    Local ordering matches Triangle.get_hierarchical_basis_fns: vertices,
    edges (1,2), (0,2), (0,1) with degrees n=2,...,order, then interiors.
    Edge direction is canonical from smaller to larger mesh vertex ID.
    Reversed edges get (-1)**n, never a reversal of mode indices.

    Entity numbering is shared with lagrange_dofmap, but no nodal basis
    construction or reference-node classification is needed. Setup is
    vectorized over cells; only element types are iterated in Python.
    Connectivity may reside on any device; outputs are on CPU, as used by
    the assemblers' topology setup. Unsupported element types are rejected.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("triangle order must be an integer >= 1")
    if not elements:
        raise ValueError("elements must contain at least one triangle type")
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    per_etype = {}
    for element_type, value in elements.items():
        elem_cls = element_type2element(element_type)
        if elem_cls is not Triangle:
            raise NotImplementedError("hierarchical DOF maps support triangles only")
        if value.ndim != 2 or value.shape[1] < 3:
            raise ValueError("triangle connectivity must have shape [n_element, >=3]")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("triangle connectivity must contain integer point IDs")
        corners = value[:, :3].detach().cpu().long()
        if ((corners < 0) | (corners >= n_points)).any():
            raise ValueError("triangle vertex IDs must lie in [0, n_points)")
        sorted_corners = corners.sort(dim=1).values
        if (sorted_corners[:, 1:] == sorted_corners[:, :-1]).any():
            raise ValueError("a triangle must have three distinct vertex IDs")
        per_etype[element_type] = (elem_cls, corners)

    vertex_ids, g2l, edge_ids, n_edges = _number_vertices_and_edges(per_etype, n_points, order)
    n_vertex = vertex_ids.numel()
    n_interior = (order - 1) * (order - 2) // 2
    interior_base = n_vertex + n_edges * (order - 1)
    n_basis = (order + 1) * (order + 2) // 2
    conn, sign = {}, {}
    for element_type, (elem_cls, corners) in per_etype.items():
        n_element = corners.shape[0]
        out = torch.empty((n_element, n_basis), dtype=torch.long)
        factors = torch.ones((n_element, n_basis), dtype=torch.int8)
        out[:, :3] = g2l[corners]
        if order >= 2:
            modes = torch.arange(order - 1)
            out[:, 3:3 * order] = (n_vertex + edge_ids[element_type][..., None]
                                   * (order - 1) + modes).reshape(n_element, 3 * (order - 1))
            pairs = corners[:, elem_cls.edge]
            reversed_edge = pairs[..., 0] > pairs[..., 1]
            odd_degree = (modes + 2) % 2 == 1
            factors[:, 3:3 * order] = torch.where(
                reversed_edge[..., None] & odd_degree, -1, 1
            ).reshape(n_element, 3 * (order - 1)).to(torch.int8)
        out[:, 3 * order:] = (interior_base + torch.arange(n_element)[:, None]
                              * n_interior + torch.arange(n_interior))
        conn[element_type], sign[element_type] = out, factors
        interior_base += n_element * n_interior
    return HierarchicalDofMap(conn, sign, interior_base, vertex_ids)


class _TetrahedronTopology(NamedTuple):
    """CPU corner/entity data shared across field orders; treat as immutable.

    Entity IDs are lexicographic in original mesh point IDs. Per-type rows
    retain original cell order. ``face_representatives[f]`` contains an
    element-type index (in ``corners`` insertion order), original cell ID,
    and local face slot of the first incidence of face f.
    """
    vertex_ids: torch.Tensor
    corners: Dict[str, torch.Tensor]
    vertex_conn: Dict[str, torch.Tensor]
    edges: torch.Tensor
    faces: torch.Tensor
    edge_ids: Dict[str, torch.Tensor]
    face_ids: Dict[str, torch.Tensor]
    face_counts: torch.Tensor
    orientation: Dict[str, torch.Tensor]
    groups: Dict[str, Dict[int, torch.Tensor]]
    face_representatives: torch.Tensor
    boundary_faces: torch.Tensor
    n_points: int


class HierarchicalTetrahedronDofMap(NamedTuple):
    """Uniform-order tetrahedral modal connectivity without orientation signs.

    Attributes
    ----------
    conn : Dict[str, torch.Tensor]
        CPU long connectivity ``[n_element, (p+1)*(p+2)*(p+3)//6]``.
        Local ordering is vertices, six edge blocks, four face blocks, then
        private cell modes, matching the oriented tetrahedral evaluator.
    n_dofs : int
        Number of scalar coefficients, in compact vertex / unique edge /
        unique face / private cell order. Interiors follow original cell
        order within each element type, then mapping insertion order.
    vertex_ids : torch.Tensor
        Sorted original mesh point IDs of the compact vertex coefficients.
    topology : object
        Reusable private CPU topology, independent of field degree. Pass it
        to :func:`hierarchical_tetrahedron_dofmap` for another order on the
        same mesh. Treat its tensors as immutable.

    Notes
    -----
    Edge/face coordinates are oriented inside the basis evaluator, so
    coefficients are gathered directly with ``conn``. Modal coefficients
    are not nodal values; there are no coefficient coordinates or signs.
    """
    conn: Dict[str, torch.Tensor]
    n_dofs: int
    vertex_ids: torch.Tensor
    topology: object


def _tetrahedron_corners(elements, n_points):
    """Validate corner connectivity independently of geometry's other nodes."""
    if isinstance(n_points, bool) or not isinstance(n_points, int) or n_points < 0:
        raise ValueError("n_points must be a nonnegative integer")
    if not elements:
        raise ValueError("elements must contain at least one tetrahedron type")
    corners = {}
    for kind, value in elements.items():
        if kind not in element_type2order:
            raise ValueError(f"unknown element type '{kind}'")
        if element_type2element(kind) is not Tetrahedron:
            raise NotImplementedError("hierarchical tetrahedron topology supports tetrahedra only")
        if not isinstance(value, torch.Tensor) or value.ndim != 2 or value.shape[1] < 4:
            raise ValueError("tetrahedron connectivity must have shape [n_element, >=4]")
        if value.dtype not in (torch.int32, torch.int64):
            raise ValueError("tetrahedron connectivity must contain integer point IDs")
        ids = value[:, :4].detach().cpu().long()
        if ((ids < 0) | (ids >= n_points)).any():
            raise ValueError("tetrahedron vertex IDs must lie in [0, n_points)")
        ordered = ids.sort(dim=1).values
        if (ordered[:, 1:] == ordered[:, :-1]).any():
            raise ValueError("a tetrahedron must have four distinct vertex IDs")
        corners[kind] = ids
    return corners


def _tetrahedron_topology(elements: Mapping[str, torch.Tensor], n_points: int
                          ) -> _TetrahedronTopology:
    """Build reusable tetrahedral entities and orientation groups on CPU.

    Only the first four connectivity columns are read. All entities are
    built, including edges/faces for linear fields. Groups contain stable
    original cell indices per type and occupied lexicographic class ID.
    Setup loops over types/classes, never over individual cells.
    """
    corners = {kind: ids.clone() for kind, ids in
               _tetrahedron_corners(elements, n_points).items()}
    vertex_ids = torch.unique(torch.cat([ids.reshape(-1) for ids in corners.values()]))
    vertex_conn = {kind: torch.searchsorted(vertex_ids, ids) for kind, ids in corners.items()}

    def entities(table):
        width = table.shape[1]
        chunks = [ids[:, table].sort(dim=-1).values.reshape(-1, width)
                  for ids in corners.values()]
        unique, inverse, counts = torch.unique(torch.cat(chunks), dim=0,
                                               return_inverse=True, return_counts=True)
        mapping, offset = {}, 0
        for kind, ids in corners.items():
            size = ids.shape[0] * table.shape[0]
            mapping[kind] = inverse[offset:offset + size].reshape(-1, table.shape[0])
            offset += size
        return unique, mapping, counts, inverse

    edges, edge_ids, _, _ = entities(Tetrahedron.edge)
    faces, face_ids, face_counts, face_inverse = entities(torch.tensor(Tetrahedron.face))
    # Stable grouping of face incidences yields the first original incident cell.
    incidence_order = torch.argsort(face_inverse, stable=True)
    starts = face_counts.cumsum(0) - face_counts
    representatives = []
    for type_id, ids in enumerate(corners.values()):
        n = ids.shape[0]
        representatives.append(torch.stack((
            torch.full((n * 4,), type_id, dtype=torch.long),
            torch.arange(n).repeat_interleave(4), torch.arange(4).repeat(n)), dim=-1))
    face_representatives = torch.cat(representatives)[incidence_order[starts]]

    # Base-4 codes preserve lexicographic permutation order.
    weights = torch.tensor([64, 16, 4, 1])
    codes = (torch.tensor(TETRAHEDRON_ORIENTATIONS) * weights).sum(-1)
    orientation, groups = {}, {}
    for kind, ids in corners.items():
        classes = torch.searchsorted(codes, (ids.argsort(dim=-1) * weights).sum(-1))
        orientation[kind] = classes
        groups[kind] = {c: torch.nonzero(classes == c, as_tuple=True)[0]
                        for c in classes.unique().tolist()}
    return _TetrahedronTopology(vertex_ids, corners, vertex_conn, edges, faces,
                                edge_ids, face_ids, face_counts, orientation, groups,
                                face_representatives, face_counts == 1, n_points)


def hierarchical_tetrahedron_dofmap(elements: Mapping[str, torch.Tensor],
                                    n_points: int, order: int, *, topology=None
                                    ) -> HierarchicalTetrahedronDofMap:
    """Build uniform-order hierarchical H1 tetrahedral coefficient numbering.

    Parameters
    ----------
    elements : Mapping[str, torch.Tensor]
        Geometry connectivity ``[n_element, >=4]`` per tetrahedral type.
        Only original corner IDs are used, independently of geometry order.
        Inputs may reside on any device; all outputs are CPU long tensors.
    n_points : int
        Number of mesh points, bounding the original corner IDs.
    order : int
        Field polynomial degree, at least 1.
    topology : object, optional
        Private topology from a previous result on the same mesh. Reuses
        all entity IDs, incidence and orientation groups across orders.
        Connectivity is validated against its stored corners; changing
        geometry coordinates or noncorner nodes does not invalidate it.

    Returns
    -------
    HierarchicalTetrahedronDofMap
        Connectivity, scalar dimension, original vertex IDs and topology.

    Notes
    -----
    There are p-1 modes per edge, (p-1)(p-2)/2 per face for p>=3, and
    (p-1)(p-2)(p-3)/6 private modes per cell for p>=4. Mode indices follow
    :class:`~tensormesh.element.HierarchicalTetrahedronBasisFns`. Global
    coefficients require no additional signs or face transformations.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("tetrahedron order must be an integer >= 1")
    if topology is None:
        topology = _tetrahedron_topology(elements, n_points)
    else:
        corners = _tetrahedron_corners(elements, n_points)
        if (not isinstance(topology, _TetrahedronTopology) or topology.n_points != n_points
                or tuple(corners) != tuple(topology.corners)
                or any(not torch.equal(ids, topology.corners[kind])
                       for kind, ids in corners.items())):
            raise ValueError("topology must match the mesh point count and ordered cell corners")
    ne = order - 1
    nf = (order - 1) * (order - 2) // 2 if order >= 3 else 0
    nc = (order - 1) * (order - 2) * (order - 3) // 6 if order >= 4 else 0
    edge_base = topology.vertex_ids.numel()
    face_base = edge_base + topology.edges.shape[0] * ne
    cell_base = face_base + topology.faces.shape[0] * nf
    conn = {}
    for kind, vertices in topology.vertex_conn.items():
        n = vertices.shape[0]
        edge_dofs = edge_base + topology.edge_ids[kind][..., None] * ne + torch.arange(ne)
        face_dofs = face_base + topology.face_ids[kind][..., None] * nf + torch.arange(nf)
        cell_dofs = cell_base + torch.arange(n)[:, None] * nc + torch.arange(nc)
        conn[kind] = torch.cat((vertices, edge_dofs.reshape(n, 6 * ne),
                               face_dofs.reshape(n, 4 * nf), cell_dofs), dim=1)
        cell_base += n * nc
    return HierarchicalTetrahedronDofMap(conn, cell_base, topology.vertex_ids, topology)


def hierarchical_tetrahedron_boundary_mask(conn: Mapping[str, torch.Tensor],
                                           order: int, n_dofs: int) -> torch.Tensor:
    """Mark the closure of boundary faces in tetrahedral modal connectivity.

    Parameters
    ----------
    conn : Mapping[str, torch.Tensor]
        Field connectivity from :func:`hierarchical_tetrahedron_dofmap`.
        All tensors must share a device; the first four slots carry vertices.
    order : int
        Field polynomial degree, at least 1.
    n_dofs : int
        Number of scalar coefficients (size of the returned mask).

    Returns
    -------
    torch.Tensor
        Boolean mask on the connectivity device. A face is on the boundary
        iff it has exactly one incident cell. Its vertices, edge modes and
        face modes are marked; private cell modes are never marked.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("tetrahedron order must be an integer >= 1")
    if isinstance(n_dofs, bool) or not isinstance(n_dofs, int) or n_dofs < 0:
        raise ValueError("n_dofs must be a nonnegative integer")
    _tetrahedron_corners(conn, n_dofs)
    device = next(iter(conn.values())).device
    n_basis = (order + 1) * (order + 2) * (order + 3) // 6
    nf = (order - 1) * (order - 2) // 2 if order >= 3 else 0
    traces = []
    for f, face in enumerate(Tetrahedron.face):
        slots = list(face)
        for e, edge in enumerate(Tetrahedron.edge.tolist()):
            if all(v in face for v in edge):
                slots.extend(range(4 + e * (order - 1), 4 + (e + 1) * (order - 1)))
        slots.extend(range(4 + 6 * (order - 1) + f * nf,
                           4 + 6 * (order - 1) + (f + 1) * nf))
        traces.append(slots)
    slots = torch.tensor(traces, dtype=torch.long, device=device)
    faces = torch.tensor(Tetrahedron.face, device=device)
    keys, carriers = [], []
    for value in conn.values():
        if value.shape[1] != n_basis:
            raise ValueError("hierarchical connectivity width must match the polynomial order")
        if value.device != device:
            raise ValueError("connectivity tensors must share a device")
        if ((value < 0) | (value >= n_dofs)).any():
            raise ValueError("coefficient IDs must lie in [0, n_dofs)")
        keys.append(value[:, faces].sort(dim=-1).values.reshape(-1, 3))
        carriers.append(value[:, slots].reshape(-1, slots.shape[1]))
    _, inverse, counts = torch.unique(torch.cat(keys), dim=0, return_inverse=True,
                                       return_counts=True)
    mask = torch.zeros(n_dofs, dtype=torch.bool, device=device)
    mask[torch.cat(carriers)[counts[inverse] == 1].reshape(-1)] = True
    return mask


def _classify_reference_nodes(elem_cls, order: int):
    """Slot classification of the reference Lagrange nodes.

    Thin alias of :meth:`tensormesh.Element.classify_nodes` (kept so the
    DOF-map code reads locally); see there for the returned structure.
    """
    return elem_cls.classify_nodes(order)


def lagrange_dofmap(elements: Mapping[str, torch.Tensor],
                    n_points: int,
                    order: int) -> LagrangeDofMap:
    r"""Build the DOF map of an order-``order`` Lagrange space on a mesh.

    Only the corner-vertex columns of ``elements`` are read (corner nodes
    come first in every TensorMesh connectivity), so the mesh may be of
    any order — the field order is fully decoupled from it. DOFs are laid
    out as ``[vertices | edges | cell interiors]``:

    * one DOF per mesh vertex used by the cells;
    * ``order - 1`` DOFs per unique edge, **oriented** from the lower to
      the higher global vertex id, with the element-local slot flipped
      whenever the element traverses the edge backwards — this is what
      makes the space :math:`C^0`-conforming for ``order >= 3``;
    * per-cell interior DOFs (2D cells; 1D cells reuse the edge slots).

    Scope: any order for 1D/2D elements; in 3D only spaces whose nodes sit
    on vertices and edges (e.g. P2 tetrahedra) — face DOFs need the
    orientation layer planned with the H(div)/H(curl) work and raise
    ``NotImplementedError`` here.

    Parameters
    ----------
    elements : Mapping[str, torch.Tensor]
        Mesh connectivity per element type, ``[n_element, n_basis_mesh]``.
    n_points : int
        Number of mesh points (bounds the vertex ids).
    order : int
        Polynomial order of the Lagrange space, ``>= 1``.
    """
    per_etype = {}
    for element_type, value in elements.items():
        elem_cls = element_type2element(element_type)
        slots, n_interior = _classify_reference_nodes(elem_cls, order)
        if elem_cls.dim == 3 and n_interior > 0:
            raise NotImplementedError(
                f"order-{order} Lagrange DOFs on '{element_type}' include "
                f"face/interior nodes; 3D face-DOF orientation is not "
                f"implemented yet (see ROADMAP) — 3D supports vertex+edge "
                f"spaces such as P2 tetrahedra"
            )
        corners = value[:, :elem_cls.n_vertex].long()        # [E, n_vertex] mesh ids
        per_etype[element_type] = (elem_cls, slots, n_interior, corners)

    vertex_ids, g2l, edge_ids, n_global_edges = _number_vertices_and_edges(
        {key: (elem_cls, corners) for key, (elem_cls, _, _, corners) in per_etype.items()},
        n_points, order)
    n_vertex_dofs = vertex_ids.shape[0]

    edge_base = n_vertex_dofs
    interior_base = n_vertex_dofs + n_global_edges * (order - 1)

    conn, ref_nodes = {}, {}
    for element_type, (elem_cls, slots, n_interior, corners) in per_etype.items():
        n_element = corners.shape[0]
        out = torch.empty(n_element, len(slots), dtype=torch.long)
        for i, (kind, a, j) in enumerate(slots):
            if kind == "vertex":
                out[:, i] = g2l[corners[:, a]]
            elif kind == "edge":
                ga = corners[:, elem_cls.edge[a, 0]]
                gb = corners[:, elem_cls.edge[a, 1]]
                jj = torch.where(ga > gb,
                                 torch.tensor(order - j, dtype=torch.long),
                                 torch.tensor(j, dtype=torch.long))
                out[:, i] = edge_base + edge_ids[element_type][:, a] * (order - 1) + (jj - 1)
            else:  # interior: per-cell, orientation-free
                out[:, i] = interior_base + torch.arange(n_element, dtype=torch.long) \
                    * n_interior + a
        conn[element_type] = out
        ref_nodes[element_type] = elem_cls.get_basis(order, torch.float64)
        interior_base += n_element * n_interior

    return LagrangeDofMap(conn=conn, n_dofs=interior_base,
                          vertex_ids=vertex_ids, ref_nodes=ref_nodes)


def lagrange_boundary_mask(conn: Mapping[str, torch.Tensor],
                           order: int,
                           n_dofs: int) -> torch.Tensor:
    r"""Boolean mask of the boundary DOFs of a Lagrange space — topological.

    A facet is on the boundary iff it is referenced by exactly one cell
    (the standard incidence criterion — no geometric tolerance and no
    ``is_boundary`` point data needed). All DOFs sitting on a boundary
    facet are marked, including edge/face nodes of higher-order spaces.

    Parameters
    ----------
    conn : Mapping[str, torch.Tensor]
        Field-local connectivity per element type (``LagrangeDofMap.conn``,
        or a mesh connectivity for mesh-node-carried spaces).
    order : int
        Polynomial order of the space (selects the facet node table).
    n_dofs : int
        Size of the returned mask.
    """
    keys, facet_nodes, widths = [], [], set()
    for element_type, value in conn.items():
        elem_cls = element_type2element(element_type)
        facet_local = elem_cls.get_facet(order)
        if isinstance(facet_local, tuple):
            raise NotImplementedError(
                f"mixed facet types of '{element_type}' are not supported by "
                f"the topological boundary detection yet"
            )
        n_corner = int((facet_local < elem_cls.n_vertex).sum(dim=1)[0])
        assert ((facet_local < elem_cls.n_vertex).sum(dim=1) == n_corner).all()
        widths.add(n_corner)
        corner_dofs = value[:, facet_local[:, :n_corner]]      # [E, n_facet, w]
        keys.append(torch.sort(corner_dofs, dim=-1).values.reshape(-1, n_corner))
        facet_nodes.append(value[:, facet_local].reshape(-1, facet_local.shape[1]))
    if len(widths) != 1:
        raise NotImplementedError(
            "meshes mixing facet arities (e.g. tetrahedra + hexahedra) are "
            "not supported by the topological boundary detection yet"
        )
    keys = torch.cat(keys, dim=0)                              # [n_all_facets, w]
    _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True,
                                      return_counts=True)
    is_boundary_facet = counts[inverse] == 1                   # [n_all_facets]
    mask = torch.zeros(n_dofs, dtype=torch.bool)
    boundary_nodes = torch.cat(facet_nodes, dim=0)[is_boundary_facet].reshape(-1)
    mask[boundary_nodes] = True
    return mask


def hierarchical_boundary_mask(conn: Mapping[str, torch.Tensor],
                               order: int, n_dofs: int) -> torch.Tensor:
    """Mark triangle boundary vertices and all boundary edge modes, never bubbles.

    Only endpoint incidence identifies boundary edges; no modal coefficient
    is interpreted as a point value. Local slots follow the hierarchical
    vertex/edge/interior ordering, independently of geometry order.
    """
    if isinstance(order, bool) or not isinstance(order, int) or order < 1:
        raise ValueError("triangle order must be an integer >= 1")
    if not conn:
        raise ValueError("conn must contain at least one triangle type")
    keys, traces = [], []
    device = next(iter(conn.values())).device
    edges = Triangle.edge.to(device)
    modes = 3 + torch.arange(3, device=device)[:, None] * (order - 1) \
        + torch.arange(order - 1, device=device)
    slots = torch.cat((edges, modes), dim=1)
    for kind, value in conn.items():
        if element_type2element(kind) is not Triangle:
            raise NotImplementedError("hierarchical boundary masks support triangles only")
        if value.ndim != 2 or value.shape[1] != (order + 1) * (order + 2) // 2:
            raise ValueError("hierarchical connectivity width must match the polynomial order")
        keys.append(value[:, edges].sort(dim=-1).values.reshape(-1, 2))
        traces.append(value[:, slots].reshape(-1, order + 1))
    _, inverse, counts = torch.unique(torch.cat(keys), dim=0, return_inverse=True,
                                      return_counts=True)
    mask = torch.zeros(n_dofs, dtype=torch.bool, device=device)
    mask[torch.cat(traces)[counts[inverse] == 1].reshape(-1)] = True
    return mask


def boundary_facet_mask(conn: Mapping[str, torch.Tensor]
                        ) -> Dict[str, Tuple[torch.Tensor, ...]]:
    r"""Which local facets of each cell lie on the boundary — by incidence.

    A facet is on the boundary iff exactly one cell references it. Facets
    are identified by their sorted corner nodes, so cells of different
    types that share a facet (a tetrahedron and a pyramid, say) are
    matched, and triangular facets are only compared with triangular ones.

    The answer is relative to the cells in ``conn``: a facet shared with a
    cell that is left out counts as a boundary facet.

    Parameters
    ----------
    conn : Mapping[str, torch.Tensor]
        Connectivity ``[E, B]`` per element type; the order of each type is
        read from its name.

    Returns
    -------
    Dict[str, Tuple[torch.Tensor, ...]]
        Per element type, one boolean tensor ``[E, n_facet]`` per facet
        shape, aligned with :meth:`~tensormesh.element.Element.element_to_facet`:
        ``(mask,)`` for cells with a single facet shape, ``(tri_mask,
        quad_mask)`` for prisms and pyramids.
    """
    groups: Dict[int, List[Tuple[str, int, torch.Tensor]]] = {}
    n_slots: Dict[str, int] = {}
    for element_type, value in conn.items():
        elem_cls = element_type2element(element_type)
        tables = elem_cls.get_facet(element_type2order[element_type])
        tables = tables if isinstance(tables, tuple) else (tables,)
        n_slots[element_type] = len(tables)
        for slot, table in enumerate(tables):
            table = torch.as_tensor(table)
            is_corner = table < elem_cls.n_vertex                    # [n_facet, B_f]
            n_corner = int(is_corner[0].sum())
            assert (is_corner.sum(dim=1) == n_corner).all()
            corners = table[is_corner].reshape(table.shape[0], n_corner)
            keys = torch.sort(value[:, corners.to(value.device)], dim=-1).values   # [E, n_facet, w]
            groups.setdefault(n_corner, []).append((element_type, slot, keys))

    masks: Dict[str, List[torch.Tensor]] = {k: [None] * n for k, n in n_slots.items()}   # type: ignore
    for n_corner, items in groups.items():
        keys = torch.cat([k.reshape(-1, n_corner) for _, _, k in items], dim=0)
        _, inverse, counts = torch.unique(keys, dim=0, return_inverse=True,
                                          return_counts=True)
        once = counts[inverse] == 1
        offset = 0
        for element_type, slot, k in items:
            n = k.shape[0] * k.shape[1]
            masks[element_type][slot] = once[offset:offset + n].reshape(k.shape[0], k.shape[1])
            offset += n
    return {k: tuple(v) for k, v in masks.items()}
