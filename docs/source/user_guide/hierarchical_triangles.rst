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

This reference-basis/topology layer does not change assembler defaults;
explicit mixed-assembler selection and modal field helpers are separate
integration functionality.
