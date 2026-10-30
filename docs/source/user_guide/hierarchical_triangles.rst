Hierarchical triangle basis
===========================

The reference triangle supports an optional, non-nodal hierarchical basis
of :math:`P_p`, for integer :math:`p\geq1`. Existing Lagrange functions and
geometry defaults are unchanged.

.. code-block:: python

   import torch
   from tensormesh.element import Triangle

   basis = Triangle.get_hierarchical_basis_fns(6, dtype=torch.float64)
   reference = torch.tensor([[0.2, 0.3], [0.0, 1.0]], dtype=torch.float64)
   values = basis.map(reference)             # [2, 28]
   gradients = basis.grad().map(reference)   # [2, 2, 28]

Construction and ordering
-------------------------

Functions are ordered as three barycentric vertex functions, edge blocks
in ``Triangle.edge`` order ``(1,2), (0,2), (0,1)``, then interior bubbles.
An edge ``(a,b)`` carries modes :math:`n=2,\ldots,p`:

.. math::

   \phi_n^{ab}=-\frac{2\lambda_a\lambda_b}{n-1}
   (\lambda_a+\lambda_b)^{n-2}
   P_{n-2}^{(1,1)}\!\left(
   \frac{\lambda_b-\lambda_a}{\lambda_a+\lambda_b}\right).

These are homogenized integrated Legendre functions. Reversing the edge
multiplies mode :math:`n` by :math:`(-1)^n`: mode indices must **not** be
reversed as they would be for nodal edge points.

Interior functions multiply :math:`\lambda_0\lambda_1\lambda_2` by scaled
Jacobi products; see the evaluator's docstring for the precise formula.
Homogenized three-term recurrences and analytic derivative recurrences
avoid monomial coefficient expansion, Vandermonde pseudoinversion, and
division by collapsed coordinates, including at vertices. Values and
gradients have the existing ``Polynomials.map`` layouts.

The functions are nested as :math:`p` increases, but arrays are not nested
prefixes: adding edge modes moves the following edge/interior blocks.
These are NGSolve-style constructions, not a claim of identical NGSolve
normalization or coefficient ordering.

Signed topological connectivity
-------------------------------

.. code-block:: python

   from tensormesh.assemble.topology import hierarchical_dofmap

   cells = {"triangle": torch.tensor([[0, 1, 2], [3, 2, 1]])}
   dofs = hierarchical_dofmap(cells, n_points=4, order=6)
   coefficients = torch.zeros(dofs.n_dofs, dtype=torch.float64)
   local = dofs.sign["triangle"].to(coefficients) * coefficients[dofs.conn["triangle"]]
   field_values = local @ values.T

``conn[type]`` and ``sign[type]`` have shape ``[n_element, n_basis]``.
Global numbering is compact vertices, unique lexicographically sorted
edges with ascending-degree modes, then private interiors for each cell.
The canonical edge direction runs from smaller to larger mesh vertex ID.
Only geometry's corner connectivity is read. Outputs are CPU long indices
and int8 signs, ready to move/cast once for assembly.

``hierarchical_boundary_mask`` marks boundary vertices and every boundary
edge mode by endpoint incidence, never cell-interior bubbles. Modal
coefficients do not have interpolation-point coordinates and must not be
initialized or constrained by assigning nodal point values directly.

Mixed assembly
--------------

Select the family explicitly; Lagrange remains the default. Hierarchical
fields always use signed topological numbering, even when their degree
equals geometry order. Different fields may mix families, orders and
component counts. Geometry and ``point_data`` retain their nodal Lagrange
interpretation.

.. code-block:: python

   from tensormesh import Condenser, Field, Mesh, MixedElementAssembler

   class Poisson(MixedElementAssembler):
       fields = [Field("u", "v", order=4, basis="hierarchical")]

       def forward(self, gradu, gradv):
           return (gradu * gradv).sum()

       def forward_vector(self, v, x):
           return (1 + x[0]) * v

   mesh = Mesh.gen_rectangle(chara_length=0.3, order=1, element_type="tri").double()
   assembler = Poisson.from_mesh(mesh)
   layout = assembler.layout
   condenser = Condenser(layout.boundary_mask("u"))
   reduced_matrix, reduced_load = condenser(assembler(), assembler.assemble_vector())
   coefficients = condenser.recover(reduced_matrix.solve(reduced_load))
   mesh_values = layout.prolong("u", coefficients)

Reference tables are shared across elements and tabulated once per
family/order/element type in each assembler construction. Orientation
signs act on integrated local contributions before global projection, on
both test and trial sides. The batched integration and ``ReduceProjector``
are unchanged. For hierarchical ``field_data``, supply global modal
coefficients: gathered coefficients receive signs before interpolation.

Initialization and boundary conditions
--------------------------------------

``layout.interpolate("u", fn)`` interpolates a physical function into
coefficients using a reference modal collocation solve, not a monomial
Vandermonde or an L2 projection. ``fn`` takes flat coordinates ``[N,D]``
and returns ``[N]`` or ``[N,c]``; constants may broadcast. Shared
coefficients are oriented and averaged. On curved geometry the expansion
is polynomial in reference coordinates.

For nonzero Dirichlet data, expand traces instead of assigning boundary
point values to modal coefficients:

.. code-block:: python

   boundary = layout.boundary_mask("u")
   lifting = layout.dirichlet_values("u", lambda x: x[:, 0] ** 2)
   condenser = Condenser(boundary, lifting)

The lifting is a full block vector. Only boundary values are sampled,
using Chebyshev-Lobatto points along each boundary edge; bubbles,
unconstrained coefficients and other fields are zero. Polynomial traces
through field degree are exact up to roundoff; other traces are
interpolated. For vector fields, ``component=k`` selects one scalar
component. These helpers currently address the whole topological boundary,
not tagged subsets.

``layout.restrict`` interpolates the mesh's nodal FE data into modal
coefficients; it cannot recover information absent from low-order input.
``layout.prolong`` evaluates coefficients at geometry nodes for plotting:
it is not an identity even when geometry and field orders coincide.
Low-order mesh vertices alone cannot display interior bubbles.

Modal ``points``, ``node_ids``, ``dof_index`` and coordinate-predicate
``dof_mask`` are deliberately unsupported. Use explicit coefficient masks
or topological boundary masks. A scalar passed to ``layout.cat`` fills
coefficients, not a constant physical field.

Arbitrary-point evaluation
--------------------------

.. code-block:: python

   samples = layout.evaluate("u", coefficients, reference)
   gradients = layout.evaluate("u", coefficients, reference, gradient=True)
   # Physical points within explicitly supplied cells:
   samples = layout.evaluate("u", coefficients, physical_points,
                             cell_indices=[0, 2], physical=True)

Points can be shared ``[Q,D]`` or per-cell ``[E_selected,Q,D]``. Evaluation
returns ``[E_selected,Q,...]``, with a final physical derivative axis for
``gradient=True``. Physical triangle evaluation uses batched Newton
inversion, seeded by the corner-affine map, including curved geometry.
Outside points and nonconvergence raise errors; geometry must be valid and
invertible. This is not a global point locator. Supply ``element_type``
for mixed-element meshes. Coefficients must be on the mesh's device.

Quadrature and device support
------------------------------

Triangle quadrature degree is independent of field degree. On affine
constant-coefficient problems stiffness needs degree ``2p-2`` and mass
``2p``; the assembler defaults to ``2*max(field.order)``. Curved geometry,
variable coefficients and nonlinear forms need quadrature-convergence
checks. See :doc:`elements_and_quadrature` for the generated Gaussian rules.

The supported device workflow is to construct on CPU and transfer the
complete assembler and coefficient vectors to CUDA. CUDA-specific public
assembly/evaluation tests are guarded when no CUDA device is available.
Reference values/gradients and orientation buffers follow module device
and dtype conversions. Physical inversion and high-order float64 work
require backend support; MPS does not support float64.

The runnable examples ``examples/poisson/compare_hierarchical.py`` and
``examples/poisson/benchmark_hierarchical.py`` compare matched Lagrange and
hierarchical spaces. Benchmark setup, tabulation, integration, orientation,
projection and solve timings are separate; accuracy/conditioning are not
inferred from throughput. GPU recurrence tabulation can be slower because
of many small kernel launches. Solver comparisons depend on backend,
stopping tolerances and coefficient scaling, not just basis tabulation.
