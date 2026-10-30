Hierarchical tetrahedron field operations
============================================================

The ``BlockLayout`` attached to a ``MixedElementAssembler`` supports interpolation,
evaluation, mesh-data transfer and nonzero Dirichlet data for
``Field("u", "v", order=p, basis="hierarchical")``. Geometry remains nodal and
may be affine or curved quadratic; field degree is independent of geometry degree.
Scalar fields and componentwise vector H¹ fields use the same entity operations.

Interpolation and coefficient meaning
-------------------------------------

.. code-block:: python

   layout = assembler.layout
   coefficients = layout.interpolate("u", lambda x: 1 + x[:, 0] ** 2)
   mesh_values = layout.prolong("u", coefficients)

``interpolate`` receives a callable taking flat physical coordinates ``[N, 3]``
and returning ``[N]`` or ``[N, components]``. Scalars/constants may broadcast.
The result contains modal coefficients, not values at fictitious modal nodes.
In particular, filling every coefficient with one does not represent the
constant function one. ``layout.points`` and coordinate-predicate DOF selection
are therefore unavailable for hierarchical fields.

The construction fits the complete mesh skeleton before its interior:

#. Evaluate each used vertex once.
#. Fit each canonical edge's residual after subtracting the vertex expansion.
#. Fit each canonical face's residual after subtracting its vertex/edge traces.
#. Fit each cell's residual after subtracting its complete boundary expansion.

Shared entities use one representative incident cell. Their coefficients are
expanded once, rather than obtained by averaging independent cell fits.
Edges and faces use increasing global vertex IDs for their coordinate arguments;
cell bubbles retain the fixed local-vertex convention. Connectivity is gathered
directly: tetrahedra do not use the triangle path's additional parity signs.

All solves use the recurrence basis. For degree ``p``, edge samples are the
interior Chebyshev–Lobatto points
``t_j = (1 - cos(pi*j/p))/2``, ``j = 1,...,p-1``. Face samples have barycentric
coordinates ``(a,b,c)/p`` for positive integers with ``a+b+c=p``; cell samples
are ``(a,b,c,d)/p`` with every integer positive and sum ``p``. Tuples are ordered
lexicographically in the first two or three integers. Thus each residual solve
is square and has exactly the number of modes belonging to that entity.
The test suite checks these recurrence-basis systems' condition numbers through
degree six and at degrees eight and ten. The largest measured 2-norm condition
numbers over the local entities, in CPU float64, are:

.. list-table:: Residual interpolation conditioning
   :header-rows: 1

   * - Degree
     - Edge
     - Face
     - Cell
   * - 4
     - 3.40
     - 2.45
     - 1.00
   * - 6
     - 6.93
     - 10.65
     - 10.95
   * - 8
     - 11.91
     - 39.69
     - 63.90
   * - 10
     - 18.65
     - 150.41
     - 319.28

These refer to the individual entity systems, not the assembled stiffness matrix
or the complete interpolation operator. Float64 is recommended at high degree.

This is interpolation, not an L² projection. On curved geometry the represented
polynomial is in reference coordinates: a physical quadratic generally has a
quartic pullback under quadratic geometry.

Evaluation and gradients
------------------------

.. code-block:: python

   reference = torch.tensor([[0.2, 0.1, 0.3]], dtype=assembler._points.dtype,
                            device=assembler._points.device)
   values = layout.evaluate("u", coefficients, reference)
   gradients = layout.evaluate("u", coefficients, reference, gradient=True)
   selected = layout.evaluate("u", coefficients, physical_points,
                              cell_indices=[3, 1], physical=True)

Points may be shared ``[Q, 3]`` or per selected cell
``[n_selected_cells, Q, 3]``. Values have shape
``[n_selected_cells, Q, ...]``; physical gradients append a final axis of length
three. Arbitrary trailing coefficient axes, including vector components and
multiple datasets, are preserved. Selected cells may be reordered or repeated;
the returned order always matches the request. Use ``element_type`` when the
mesh contains several element types.

Reference evaluation partitions selected cells by their vertex-order class.
Each occupied class uses its oriented basis at the requested points, without
permuting geometry or quadrature coordinates. Shared query points share a
reference value table within the class.

With ``physical=True``, the caller supplies the containing cells. Affine
tetrahedra use a linear inverse. Curved tetrahedra use batched Newton iteration
initialized by the corner-affine inverse. ``max_iterations`` and ``tolerance``
control convergence; the latter is a physical residual tolerance. Invalid or
nonconvergent maps and points outside the reference tetrahedron raise
``ValueError``. This operation is not a mesh-wide point locator.

Mesh-data transfer
------------------

``layout.restrict("u", point_data)`` interpolates the geometry-order nodal FE
function defined by ``point_data``. It does not recover an analytic function
from insufficient samples. For example, vertex samples of ``x**2`` on a linear
mesh define a piecewise-linear input field, even if the target modal degree is
four. Restriction accepts floating or complex data ``[n_mesh_points, ...]``
and preserves its dtype and trailing dimensions.

``layout.prolong("u", coefficients)`` evaluates the oriented field at geometry
nodes for output. Explicit visualization samples can instead be passed to
``evaluate``. Conforming shared geometry is required, including matching
quadratic face nodes on adjacent cells. These helpers read current coordinates
and registered connectivity/orientation buffers on each call; they follow module
dtype/device transfers and do not retain old coordinate autograd graphs.

Boundary-only Dirichlet lifting
--------------------------------

.. code-block:: python

   from tensormesh import Condenser

   mask = layout.boundary_mask("u")
   lifting = layout.dirichlet_values("u", lambda x: 1 + x[:, 0] ** 2)
   condenser = Condenser(mask, lifting)
   reduced, rhs = condenser(assembler(), assembler.assemble_vector())
   solution = condenser.recover(reduced.solve(rhs))

The boundary mask contains vertices, edge modes and face modes in the closure
of faces incident to exactly one cell. Interior-only entities and private cell
modes are excluded. Lifting uses the same entity-first interpolation, restricted
to that closure: the callable is evaluated only on the boundary, and all
unconstrained coefficients and unrelated field blocks remain zero.

For vector fields, ``component=None`` expands all components. Passing
``component=k`` expects a scalar callable and writes only component ``k``.
The matching constraint mask is ``layout.boundary_mask("u", component=k)``.
Initially these operations address the complete topological boundary.
Nonpolynomial traces are interpolated, and curved traces are polynomial in
reference face coordinates. General modal Neumann/Robin facet assembly is
outside this field-helper API.

Runnable manufactured solution
------------------------------

``examples/poisson/hierarchical_tetrahedron.py`` solves a manufactured polynomial
Poisson problem with a nonzero trace on four connected tetrahedra and reports
value/gradient errors as JSON:

.. code-block:: bash

   python examples/poisson/hierarchical_tetrahedron.py --degree 4
   python examples/poisson/hierarchical_tetrahedron.py --degree 6 --device cuda

The numerical tests additionally cover a quartic volume bubble, sine-solution
p-convergence on a tetrahedral cube, conforming curved quadratic meshes,
boundary-only component lifting, arbitrary coefficient axes, state/device
movement and differentiation. CUDA cases are guarded when hardware is absent;
CPU checks do not establish CUDA performance acceptance.
