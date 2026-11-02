Hierarchical tetrahedral assembly benchmark
===========================================

``examples/poisson/benchmark_hierarchical_tetrahedra.py`` measures the grouped
hierarchical tetrahedral path of ``MixedElementAssembler``. It emits JSON with
numerical checks, separated setup/integration/projection timings, and CUDA
allocator peaks when CUDA is selected. CPU runs provide numerical and harness
validation; they do not establish CUDA performance acceptance.

High-order comparison against upstream
--------------------------------------

Use ``examples/poisson/benchmark_upstream_tetrahedra.py`` to answer how much
assembly time changes when switching from upstream Lagrange to hierarchical
tetrahedra at P3/P4/P6. This is a separate, process-isolated comparison against
``camlab-ethz/TensorMesh`` commit
``2d544e8bc7ee8de412ffbc768b313875c3b12895``. Select the upstream Lagrange API
with ``--upstream-assembler element`` (the default: ``ElementAssembler`` and
``NodeAssembler``) or ``--upstream-assembler mixed`` (``MixedElementAssembler``
for both matrices and loads). Both upstream choices use reduction-based global
accumulation; the feature calls grouped ``MixedElementAssembler`` with CSR
projection. The chosen baseline is recorded in JSON and printed by ``--inspect``.

Prepare the baseline once (no second package installation is needed):

.. code-block:: bash

   git clone https://github.com/camlab-ethz/TensorMesh.git ../TensorMesh-upstream
   git -C ../TensorMesh-upstream checkout --detach 2d544e8bc7ee8de412ffbc768b313875c3b12895

From the feature checkout, using the same activated environment:

.. code-block:: bash

   mkdir -p dist
   python examples/poisson/benchmark_upstream_tetrahedra.py \
       --upstream ../TensorMesh-upstream --device cuda --dtype float64 \
       --degrees 3 4 6 --forms mass poisson --subdivisions 2 4 \
       --batch-size -1 --element-batch-size 0 --repeats 7 --warmup 2 \
       --output dist/tetra-upstream-gpu.json \
       > dist/tetra-upstream-gpu.stdout.json

   python examples/poisson/benchmark_upstream_tetrahedra.py \
       --inspect dist/tetra-upstream-gpu.json

For a CPU check, use ``--device cpu --subdivisions 1 --repeats 1 --warmup 1``. The integration
test additionally accepts ``TENSORMESH_UPSTREAM=/absolute/path/to/TensorMesh-upstream``;
without this environment variable the real-upstream tests skip.

The printed table shows upstream and hierarchical matrix milliseconds, the ratio
``L/H`` (above one favors hierarchical), and ``H change``: the percentage increase
or decrease in time after switching. JSON also includes unit-load timing ratios,
raw samples, first calls, setup, and memory. Each pair must pass numerical checks
before ratios are emitted. An OOM or failed check has an explicit status instead
of a speedup. **P6 Poisson at 384 cells is skipped by default**, before mesh
construction or worker launch, because the previous Euler run exhausted GPU
memory. P6 mass and the 48-cell P6 Poisson comparison remain in the sweep.
Additional exclusions can be appended with ``--skip-case P FORM SUBDIVISIONS``.
Skipped cases are printed and recorded without timing ratios and do not mark
the report as failed. Use the same smaller ``--batch-size`` on both paths for
a separately labeled quadrature-chunked experiment.

Mixed-assembler comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~

To compare through ``MixedElementAssembler`` on both checkouts, run:

.. code-block:: bash

   python examples/poisson/benchmark_upstream_tetrahedra.py \
       --upstream ../TensorMesh-upstream --upstream-assembler mixed \
       --device cuda --dtype float64 --degrees 3 4 6 \
       --forms mass poisson --subdivisions 2 4 \
       --batch-size -1 --element-batch-size 0 --repeats 7 --warmup 2 \
       --output dist/tetra-upstream-cell-batched-gpu.json \
       > dist/tetra-upstream-cell-batched-gpu.stdout.json

   python examples/poisson/benchmark_upstream_tetrahedra.py \
       --inspect dist/tetra-upstream-cell-batched-gpu.json

This uses the same scalar mass/Poisson forms and a single degree-p field on each
side. Upstream declares ``Field("u", "v", p)``; the feature selects the
hierarchical basis. The field degree matches the nodal mesh degree, so upstream
uses the existing mesh connectivity even at P3/P4/P6. The same quadrature and
lattice adapters and full operator-equivalence checks apply to either baseline.

Both mixed paths recompute geometry and physical field gradients on each call.
This removes the original comparison's ``ElementAssembler`` versus
``MixedElementAssembler`` execution/caching difference. Both paths now process
all cells of an element type together by default; hierarchical orientation
classes no longer imply separate integration passes. Oriented table selection,
local-buffer packing, and CSR versus reduction projection still differ. This
measures the hierarchical extension to mixed assembly, including those
implementation choices. It does not introduce new coupled Stokes forms.

``--element-batch-size N`` limits hierarchical cells per chunk across orientations;
``0`` (the default) means all cells. The pinned upstream APIs do not support this
cell limit, so upstream always uses all cells and the report explicitly labels
the hierarchical-only limit. For matched cell scheduling use ``0``; for matched
quadrature chunking use the same ``--batch-size`` supplied to both paths. Unlike
the earlier orientation-group schedule, unchunked high-order mass assembly can
materialize cell-dependent integrands and consume more GPU scratch memory.

Pass several explicit report filenames to ``--inspect`` to display both
baseline comparisons; each table identifies its upstream assembler.

Matched inputs and explicit adapters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Both workers receive the same binary mesh and quadrature arrays. The mesh has
matching degree-p geometry and fields, with all nodal points at affine positions:
faces are straight. Integer barycentric coordinates produce the internal
TensorMesh ordering and exactly shared edge/face nodes. The default balanced
orientation distribution occupies all 24 classes at 48 cells. ``single`` is
available as a diagnostic distribution. No Gmsh generation is required.

Two benchmark-only runtime adapters are applied identically to both checkouts:

* Supply a positive conical Gauss-Jacobi rule with ``(p+1)**3`` points, exact
  through degree ``2*p+1``. This avoids legacy-rule underintegration and supplies
  high-order rules absent upstream. Both mass and stiffness use this same rule.
* Fix the integer side length in ``tet_linspace``. Upstream uses
  ``floor(number_of_points**(1/3))``, which rounds down for some perfect cubes
  and breaks P5/P6 interior-node generation. The adapter uses ``order+1`` or
  ``order-1``; node placement, nodal polynomial construction and evaluators
  otherwise remain upstream code.

These adapters are recorded in every worker result. Thus this is an
**equally integrated upstream assembly comparison with explicit input fixes**.
The difference includes the implementations' geometry caching, dispatch and
projection costs as well as the basis change. It measures the cost of switching
assembly paths, rather than isolating basis arithmetic alone. Upstream source
files are not edited. The default pin requires a clean checkout; an explicit
``--upstream-commit`` can select another audited revision.

Verification and isolation
~~~~~~~~~~~~~~~~~~~~~~~~~~

For every degree/form pair, separate workers first assemble on a 48-cell,
24-orientation verification mesh. A sparse evaluation matrix ``T`` maps modal
coefficients to nodal values. Shared traces are checked, followed by full sparse
matrix and load comparisons ``A_H = T.T @ A_L @ T`` and ``b_H = T.T @ b_L``.
Each timed size additionally checks constant, linear, degree-p and mixed
monomial bilinear forms and loads against analytic unit-cube integrals, nodal
Kronecker interpolation, and symmetry. Failed checks suppress timing ratios;
this also detects insufficient nodal accuracy at higher degree or float32.
The default float64 tolerance is ``2e-8`` in maximum absolute error scaled by
``max(1, max(abs(reference)))``. Float32 uses ``3e-4`` and may fail at high degree.

Each implementation/size/form runs in a fresh interpreter, sequentially on the
same requested device. Workers verify all imported TensorMesh modules belong to
their selected checkout. Reports record commits, source hashes, import paths,
mesh/rule hashes, device, software and batching settings. Source changes during
the run are rejected. Worker jobs, logs, input arrays and validation operators
are retained in a uniquely named directory beside the report.

Setup and first calls are separated from warmed synchronized wall-clock samples.
Autograd is disabled. ``--batch-size -1 --element-batch-size 0`` uses the full
rule and all cells on both paths, without implicit orientation chunks.
With the ``element`` baseline,
upstream's geometry caches are retained during warmed calls and both the matrix
and load assemblers remain resident. With ``mixed``, each implementation uses
one assembler for matrices and loads and recomputes geometry during each call.
GPU outputs are released before timing samples; CPU copies of the first outputs
are retained for later checks. Diagnostic basis tabulation runs after timing so
it cannot prewarm production caches or inflate GPU allocation measurements.
Peak allocated GPU memory includes resident tensors and production caches. Each
case starts in a fresh process, and the report is checkpointed after each worker.

The table reports the median of ``--repeats`` warmed **full matrix assembly**
calls, including host dispatch and GPU execution with synchronization. It does
not report total benchmark runtime. Every successful timing worker also runs a
first matrix and load call, ``--warmup`` calls of each, all load repetitions,
and (on CUDA) a separate matrix and load memory measurement. With seven repeats
and two warmups that is 11 matrix and 11 load calls per timing worker. Each
worker independently imports packages, initializes CUDA, constructs the
assembler and checks its operators. The default 11-case sweep uses 22 timing
workers plus 12 validation workers. JSON records whole-worker wall seconds and
total benchmark wall seconds; ``--inspect`` prints the total and worker count.

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
meaning the complete rule. ``--element-batch-size`` caps original-order cells
across orientations, with ``0`` meaning all cells of each element type. These parameters bound
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

The original ``benchmark_hierarchical_tetrahedra.py --compare-lagrange`` uses
the same linear cube geometry and actual quadrature rule within the feature checkout.
The Lagrange assembler request is adjusted explicitly to bypass its legacy
non-promoting dispatch. Comparisons are limited to degrees 1 and 2, for which
generalized tetrahedral Lagrange topology is supported. Higher degrees receive
an ``unsupported`` record. The established Lagrange path does not support
element batching; its report records ``element_batch_size: null``. Its matrix
and load timings include its own projection backend. Modal and nodal matrix
entries are not directly compared: the analytic physical identities are the
common checks. No speedup is extrapolated to unsupported degrees or hardware.
This limit concerns constructing new field connectivity on a linear mesh;
upstream matching-order nodal meshes support higher-order fields, as used by
the separate upstream comparison above.

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
CSR buffers, global values, and estimates local source, reference-gather and
physical-gradient scratch sizes. Cell-batch counts are independent of the
orientation histogram. These estimates omit
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
