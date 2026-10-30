Hierarchical tetrahedral assembly benchmark
===========================================

``examples/poisson/benchmark_hierarchical_tetrahedra.py`` measures the grouped
hierarchical tetrahedral path of ``MixedElementAssembler``. It emits JSON with
numerical checks, separated setup/integration/projection timings, and CUDA
allocator peaks when CUDA is selected. CPU runs provide numerical and harness
validation; they do not establish CUDA performance acceptance.

Running the benchmark
---------------------

From the repository root, a small CPU run is:

.. code-block:: bash

   python examples/poisson/benchmark_hierarchical_tetrahedra.py \
       --degrees 1 2 --forms mass poisson mixed field_data \
       --repeats 3 --warmup 1 --output dist/tetrahedra-cpu.json

The defaults sweep degrees 1, 2, 3, 4 and 6 on six tetrahedra. The cube generator
uses a conforming Freudenthal subdivision and needs no Gmsh installation.
``--subdivisions n`` creates ``6*n**3`` tetrahedra and ``(n+1)**3`` vertices.
Multiple subdivision counts are processed sequentially. Local vertex-slot
permutations select ``--orientations balanced single skewed`` without changing
the physical tetrahedra or matching face triangulations. Balanced meshes with
at least 24 tetrahedra occupy all 24 orientation classes.

A CUDA sweep, sized explicitly for the available memory, can use:

.. code-block:: bash

   python examples/poisson/benchmark_hierarchical_tetrahedra.py \
       --device cuda --dtype float64 --degrees 1 2 3 4 6 \
       --subdivisions 2 4 --orientations balanced skewed \
       --batch-size 16 --element-batch-size 32 --repeats 7 --warmup 2 \
       --output dist/tetrahedra-cuda.json

``--extended`` adds degrees 8 and 10; these are opt-in. ``--threads`` controls
PyTorch CPU threads. ``--batch-size`` counts quadrature points, with ``-1``
meaning the complete rule. ``--element-batch-size`` caps cells per occupied
orientation group, with ``0`` meaning the entire group. These parameters bound
integration scratch space, not full local matrix sources or CSR metadata.
High-order mixed forms can exhaust memory even with small chunks.

Forms and numerical checks
--------------------------

``--forms`` selects any of:

* ``mass``: scalar mass matrix and unit load; checks constant mass energy,
  mass applied to a constant against the load, and coordinate mass energy.
* ``poisson``: scalar stiffness and unit load; checks the constant nullspace,
  unit coordinate energy and unit-cube load integral.
* ``mixed``: a three-component degree-p field coupled to a scalar degree
  ``max(1,p-1)`` field, exercising rectangular component/order blocks. The form
  is gradient energy plus vector mass, twice scalar mass, and symmetric
  quarter-strength scalar/vector-sum coupling. Constant and coordinate
  energies are checked against ``13/2`` and ``31/6``, respectively.
* ``field_data``: diffusion with coefficient ``1 + rho**2``, where ``rho`` is
  interpolated from field coefficients. Reassembly alternates ``rho=x`` and
  ``rho=x+1/2`` on the same assembler. Coordinate energies and weighted loads
  are checked against ``1 + shift**2 + shift + 1/3``.

The checks use the affine vertex-mode contract directly; they do not depend on
a modal interpolation fit. All reported runs disable autograd and measure
forward execution. Backward time and retained autograd storage are outside this
benchmark's scope. Tests check numerical properties without timing thresholds.

``--solve`` additionally times homogeneous Dirichlet Poisson solves. Constraint
setup is excluded. CPU timings include a fresh SciPy sparse LU factorization
and solve, excluding CSC conversion; CUDA uses PyTorch CG through torch-sla.
The relative algebraic residual is reported. A mesh/order with no free DOFs
reports that status explicitly. This optional solve is separate from assembly
and is not a manufactured-solution convergence study.

Quadrature and Lagrange comparisons
-----------------------------------

The default exactness request is ``2*p`` (at least 3 for ``field_data``).
``--quadrature-extra`` adds exactness; ``--quadrature-order`` overrides the
request. Grouped requests 5, 6 and 7 use legacy rule indices 6, 7 and generated
rule 8 because legacy tables 5--7 underintegrate by one degree. The report
records requested exactness, assembler request, actual rule index, point count
and a hash of the actual points and weights.

``--compare-lagrange`` uses the same cube geometry and actual quadrature rule.
The Lagrange assembler request is adjusted explicitly to bypass its legacy
non-promoting dispatch. Comparisons are limited to degrees 1 and 2, for which
generalized tetrahedral Lagrange topology is supported. Higher degrees receive
an ``unsupported`` record. The established Lagrange path does not support
element batching; its report records ``element_batch_size: null``. Its matrix
and load timings include its own projection backend. Modal and nodal matrix
entries are not directly compared: the analytic physical identities are the
common checks. No speedup is extrapolated to unsupported degrees or hardware.

Timing and memory interpretation
--------------------------------

Each case retains one assembler and is released before the next case. Setup
constructs topology and tables on CPU, then transfers the complete module to
the requested device. The first construction is instrumented with exclusive
CPU categories for entity topology/DOF maps, group layout, oriented reference
tabulation, and projection/sparse-pattern construction. Remaining geometry,
nodal tables, eager geometry value caching and module setup are accounted for
in the remainder. Python instrumentation overhead is included. Global basis
and quadrature caches may already be warm from earlier cases; this is not a
fresh-process cache benchmark.

First public matrix/load calls are reported separately from warmed samples,
medians and extrema. CUDA wall timers synchronize before and after each call.
The independent stage reconstruction streams the production geometry and
tensor kernels, packs disjoint local source slices, and applies one CSR
projection per executed block. Both matrix values/pattern and loads must match
public assembly before a case passes. Its stages are:

* Geometry/data gathering and physical-gradient mapping.
* Form validation and weak-form evaluation/quadrature integration.
* Local-buffer packing.
* CSR projection.
* Final sparse matrix or block-vector layout.

Stage timers synchronize at each boundary and therefore perturb execution.
Their medians need not sum to end-to-end timings. Use the public assembly
timings for throughput comparisons. Reconstruction shares the production
tensor kernels, so the analytic cube checks provide an additional independent
numerical reference. Geometry batches are streamed; the harness does not
retain a mesh-sized collection of all mapped quadrature gradients.

CUDA memory is measured by a separate call after warmup. Baseline and peak
allocated/reserved bytes, including incremental peaks, describe PyTorch's
allocator; they are not driver memory or host RSS. The report also counts
oriented reference storage, grouped connectivity/orientation, declared matrix/load
CSR buffers, global values, and estimates local source and physical-gradient
scratch sizes. These estimates omit
weak-form intermediates and allocator overhead. CPU runs mark GPU memory as
unavailable. Requests for unavailable CUDA emit a machine-readable unavailable
report and return a nonzero exit status. Catchable allocation failures receive
``out_of_memory`` records; an operating-system process kill cannot be reported.

JSON includes mesh/source hashes, device model, software versions, dtype,
orders, DOFs, orientation histograms, executed blocks and chunk sizes. Progress
goes to stderr; stdout contains only JSON. ``--output`` writes the same report
to a file. Generated reports belong in ignored output directories, not source
control. Numerical failures produce an incomplete report and a nonzero exit
status. Successful CPU checks and guarded CUDA test skips must not be presented
as measured GPU performance.
