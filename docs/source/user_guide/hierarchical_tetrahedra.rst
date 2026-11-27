Hierarchical tetrahedral bases
==============================

TensorMesh provides a vertex-oriented, non-nodal hierarchical basis of
:math:`P_p` on tetrahedra for every integer :math:`p\geq1`. The evaluator
and topological coefficient map are independent of geometry order. Select
``Field(..., basis="hierarchical")`` in :class:`~tensormesh.MixedElementAssembler`
to assemble these spaces. This chapter describes their mathematical and API
contract; see :doc:`mixed_assembly` for grouped integration and chunking.
For interpolation, evaluation and nonzero boundary lifting, see
:doc:`hierarchical_tetrahedron_fields`.
Single-field bilinear forms can also opt into this basis through
``ElementAssembler.from_mesh(..., basis="hierarchical", field_order=p)``;
see :ref:`forms-hierarchical-element` for quadrature and compatibility limits.

Reference element and local layout
----------------------------------

The reference vertices are ``(0,0,0), (1,0,0), (0,1,0), (0,0,1)``, with
barycentric coordinates :math:`(\lambda_0,\lambda_1,\lambda_2,\lambda_3)
=(1-x-y-z,x,y,z)`. Local functions follow the existing
:class:`~tensormesh.Tetrahedron` entity tables:

* Four vertex functions, in local slot order.
* Six edge blocks: ``(2,3), (1,3), (1,2), (0,3), (0,2), (0,1)``.
* Four face blocks: ``(1,2,3), (0,2,3), (0,1,3), (0,1,2)``.
* Private cell-interior functions.

.. list-table:: Modes per entity
   :header-rows: 1

   * - Entity
     - Count
     - First degree
   * - Vertex
     - 1
     - 1
   * - Edge
     - :math:`p-1`
     - 2
   * - Face
     - :math:`(p-1)(p-2)/2`
     - 3
   * - Cell
     - :math:`(p-1)(p-2)(p-3)/6`
     - 4

Counts below an entity's first degree are zero. The total local dimension is
:math:`B=(p+1)(p+2)(p+3)/6`. Nesting is by entity and mode identity: higher
order adds modes within each block, so the complete array is **not**
prefix-nested.

Polynomial formulas and normalization
-------------------------------------

Write the homogenized Jacobi polynomial as

.. math::

   H_n^{\alpha,\beta}(s,t)=t^n P_n^{(\alpha,\beta)}(s/t).

The implementation evaluates its polynomial three-term recurrence and
analytic derivatives, without dividing by :math:`t`. Thus these formulas
are valid at collapsed coordinates, vertices, edges and faces as well as
interior points.

Vertex functions are :math:`\lambda_a`. For a canonically ordered edge
:math:`(a,b)`, the degree-:math:`n` mode is

.. math::

   E_n=-\frac{2}{n-1}\lambda_a\lambda_b
       H_{n-2}^{1,1}(\lambda_b-\lambda_a,\lambda_a+\lambda_b),
       \qquad n=2,\ldots,p.

On that edge, with :math:`z=\lambda_b-\lambda_a`, this is
:math:`[P_n(z)-P_{n-2}(z)]/(2n-1)`, whose derivative in :math:`z` is
:math:`P_{n-1}(z)`. Reversing the coordinate arguments multiplies the mode
by :math:`(-1)^n`; it does not reverse mode indices.

For a canonically ordered face :math:`(a,b,c)`, set
:math:`s_{ab}=\lambda_a+\lambda_b` and
:math:`s_f=s_{ab}+\lambda_c`. Its interior modes are

.. math::

   F_{ij}=\lambda_a\lambda_b\lambda_c
       H_i^{1,1}(\lambda_b-\lambda_a,s_{ab})
       H_j^{2i+3,1}(2\lambda_c-s_f,s_f),
       \qquad i+j\leq p-3.

Ordering is increasing total :math:`i+j`, then ascending :math:`i`, with
:math:`j=\mathrm{total}-i`. On the face, :math:`s_f=1`: this defines the
canonical triangular face evaluator in barycentric coordinates
:math:`(\lambda_a,\lambda_b,\lambda_c)`. The private interior convention
of an independently numbered 2D triangle must not be substituted without
aligning these canonical coordinate arguments.

For cell modes, use the **fixed local order** :math:`(0,1,2,3)`, and let
:math:`s_{01}=\lambda_0+\lambda_1`, :math:`s_{012}=s_{01}+\lambda_2`:

.. math::

   C_{ijk}=\lambda_0\lambda_1\lambda_2\lambda_3
       H_i^{1,1}(\lambda_1-\lambda_0,s_{01})
       H_j^{2i+3,1}(2\lambda_2-s_{012},s_{012})
       H_k^{2(i+j)+5,1}(2\lambda_3-1,1),
       \qquad i+j+k\leq p-4.

Ordering is increasing :math:`i+j+k`, then ascending :math:`i`, then
ascending :math:`j`, with :math:`k=\mathrm{total}-i-j`.

These formulas define TensorMesh's normalization. They follow NGSolve's
vertex-oriented construction principle, but do not promise identical
coefficients to NGSolve's current implementation. The implementation uses
TensorMesh's Jacobi recurrence; the design reference is
`NGSolve's H1 high-order elements
<https://github.com/NGSolve/ngsolve/blob/1cb0200fa087ea0af1645e1de3feb309fa998e0c/fem/h1hofe_impl.hpp>`_.

Canonical orientation classes
-----------------------------

For the cell's four original global corner IDs, ``orientation`` is the
tuple of **local slots** that sorts those IDs in ascending order. For
example, corner IDs ``[19, 2, 41, 7]`` give ``(1, 3, 0, 2)``.
``TETRAHEDRON_ORIENTATIONS`` is the lexicographic tuple of the 24
permutations of ``range(4)``; its index is the class ID in ``0,...,23``.

Within each local edge and face block, coordinate arguments are sorted by
their ranks in this tuple. Vertex functions and private cell bubbles stay
in local order. Reference points and geometry connectivity keep their
original meaning. Classes depend only on relative corner IDs, independently
of degree, coordinates, Jacobian sign or outward normals.

Shared-face values and tangential derivatives then agree when coefficients
are gathered directly with ``conn``. **Do not apply triangle-style signs
afterward**: edge and face orientation is already in the evaluator. H1
conformity does not require continuity of the normal derivative.

Changing global vertex numbering can change face modes by more than a
permutation. Compare physical fields or operators through a verified change
of basis; raw coefficients need not match after a simple DOF permutation.

Evaluating functions and gradients
----------------------------------

.. code-block:: python

   import torch
   from tensormesh.element import Tetrahedron

   basis = Tetrahedron.get_hierarchical_basis_fns(
       4, torch.float64, orientation=(1, 3, 0, 2))
   points = torch.tensor([[0.2, 0.3, 0.1], [0., 0., 0.]], dtype=torch.float64)
   values = basis.map(points)          # [2, 35]
   gradients = basis.grad().map(points)  # [2, 3, 35]

The evaluator also supports arbitrary leading batch axes and empty batches.
It is a PyTorch module: ``.to()``, ``.float()``, ``.double()`` and CUDA
movement set its tabulation dtype/device. Input points are converted to
that dtype/device. Each factory call returns an independent module; an
assembler can cache shared reference tables per occupied orientation class.
Analytic first gradients remain differentiable with respect to input points;
``grad().grad()`` is not implemented.

Topology and coefficient numbering
-----------------------------------

.. code-block:: python

   from tensormesh.assemble.topology import (
       hierarchical_tetrahedron_dofmap,
       hierarchical_tetrahedron_boundary_mask,
   )

   elements = {"tetra": torch.tensor([[0, 1, 2, 3], [4, 3, 1, 2]])}
   mapping = hierarchical_tetrahedron_dofmap(elements, n_points=5, order=4)
   assert mapping.n_dofs == 55  # 5 vertices + 9*3 edges + 7*3 faces + 2 cells
   boundary = hierarchical_tetrahedron_boundary_mask(mapping.conn, 4, mapping.n_dofs)
   higher = hierarchical_tetrahedron_dofmap(
       elements, n_points=5, order=6, topology=mapping.topology)
   assert higher.topology is mapping.topology

The result's ``conn`` tensors have shape ``[n_element, B]``. Its
``vertex_ids`` records the original IDs of the compact, sorted vertex block.
Global numbering is

.. code-block:: text

   vertices | lexicographic edges | lexicographic faces | private cell modes

Modes within each entity use the local mode order above. For a pure
tetrahedral mesh with :math:`V` vertices, :math:`E` edges, :math:`F` faces
and :math:`T` cells, the global scalar dimension is

.. math::

   N=V+(p-1)E+\frac{(p-1)(p-2)}2F
       +\frac{(p-1)(p-2)(p-3)}6T,

with face and cell terms zero below their first supported degrees. Private
cell blocks follow original cell order within each element type, then
mapping insertion order. Scheduling orientation groups cannot affect this
coefficient layout.

All entity discovery and grouping happens on CPU. Inputs may reside on
another device; outputs are CPU long tensors. Only the first four geometry
connectivity columns are read: noncorner high-order nodes do not become
field coefficients. Corner IDs must be distinct within each cell and lie
in ``[0, n_points)``. Other shapes and invalid field orders are rejected.
Empty cell blocks are supported. A mixed-geometry-order restriction imposed
by an assembler is separate from this topological builder.

The private ``topology`` object holds all entities even for p=1, along with
face incidence, representative incident cells and stable per-type orientation
groups. Treat it as immutable. Reuse checks the mesh point count, element
type order and original corner connectivity; coordinates and noncorner
geometry nodes may change without rebuilding it.

The boundary mask uses incidence of faces identified by the first four
vertex carriers. Exactly-one-cell faces are boundary faces; their full
vertex/edge/face closure is selected, and cell-interior coefficients are
excluded. The mask is returned on the connectivity device. Coefficients
are modal amplitudes, not values attached to fictitious points: nonzero
Dirichlet data needs a trace interpolation/lifting procedure. Filling every
coefficient with one does not represent a constant field.

``MixedElementAssembler`` selects the correct class table for every occupied
orientation and uses unsigned coefficients. The same topology and reference
tables are reused across fields of matching degree. Matrix and load assembly
use reduction projection of original-cell-order local contributions, through the
same ``ReduceProjector`` used for Lagrange mixed assembly. Cell chunks span
orientations; neither connectivity nor cell data is reordered by orientation.

Numerical coverage
------------------

CPU float64 tests check all monomials and their gradients through p=6, with
extended p=8 and p=10 checks; independent SciPy Jacobi formulas; autograd
at interior and boundary points; entity support and exact mode nesting;
all 24 orientation classes; and physical shared-face values/tangential
derivatives at p=4 and p=6. Topology tests cover noncontiguous original IDs,
high-order geometry corners, original cell order, reusable groups, global
dimensions and boundary closure. CUDA value/gradient and topology-device
tests are guarded when CUDA is unavailable.
