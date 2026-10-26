# Performance design review

The compact sparse forward path and heterogeneous-cell corrections are documented in [SCALING_OPTIMIZATION_REPORT.md](SCALING_OPTIMIZATION_REPORT.md). Dense workspace findings below describe the earlier implementation and the retained reference path.

Subsequent native compilation and execution with Kokkos 5.1.1 are documented in [NATIVE_RUN_REPORT.md](NATIVE_RUN_REPORT.md). Earlier environment limitations below describe the review stage, not the later native run.

**Verdict: the direction is reasonable, but the current implementation is not
ready for a performance sign-off or an MKPP/MICM/KPP ranking.** This review covers
both the actual generated integration/dispatch path and the experiment used to
measure it. No CPU/GPU throughput or efficiency is inferred from unit tests.

## Findings ranked by impact

| Priority | Finding grounded in this source | Consequence | Resolution/status |
| --- | --- | --- | --- |
| Blocking | Scientific gates and complete scenario execution are not wired into the campaign executor | Labels and matching hashes do not prove the same numerical problem was solved accurately | Keep CLI results diagnostic; complete native workload/reference/calibration/accuracy integration before publishing |
| High | Compiled-path template declares `J_values`, `W_values`, and `LU_values` as `NUM_SPECIES * NUM_SPECIES` arrays | Sparse factorization still has quadratic declared scratch storage and initialization; possible CPU cache/stack and GPU local-memory costs | Compact symbolic sparse workspace is the preferred design to implement and validate; numerical path is unchanged here |
| High | The benchmark calls `MechanismRegistry::execute()` -> `RangePolicy` -> `ParamsTiledCellIntegrator` -> per-cell `integrate()` | It does not exercise the separate `execute_batched_dense_solve()` team-cooperative helper | Profile and name the actual production path; a helper's presence does not prove team/vector utilization |
| Medium | Inline-only generated objects left the heavy helpers in the consumer optimizer | Multi-minute `-O3` compilation despite linking a kernel archive | Fixed for CPU builds: real separate definitions; new runner compilation 50.1 seconds locally. Device path retains inline definitions; runtime tradeoffs still need controlled measurement |
| High | Warmup estimates previously compared timings from newly launched processes with increasing warmup counts | This cannot establish consecutive warmup stability inside the measured process | Fixed: diagnostic runs use the declared maximum native warmup count and explicitly record stability as unverified; persistent/native warmup telemetry remains required |
| High | Sum of many separately clocked short solves is treated as a one-second sample | Clock/fence overhead can remain material even after more repetitions | Require a measured overhead bound and record interval count/minimum native interval; currently a remaining native-instrumentation requirement |
| High | Pilot counts were chosen separately for each solver, then maximized without observing every solver at that common count | Cache/frequency/nonmonotonic behavior can violate the sample floor for a subject at the selected count | Fixed: all subjects are remeasured at every proposed common count; every three-observation median must pass |
| High | Calibration trusted `passed_all_gates=True` even if retained WRMS/max metrics exceeded thresholds | A faster inaccurate candidate could be selected by contradictory input evidence | Fixed: numeric gates WRMS <= 1 and max <= 5 are required in addition to scientific flags |
| Medium | Sampling controls were only partially validated | Nonsensical warmups, too few pairs, invalid bootstrap policy, or truncated balanced cycles could reach execution | Fixed: pre-execution validation; ringer requires complete permutation cycles |
| Medium | Pair checks permitted identical order positions, and report derivation had a fixed ten-block floor | Replayed/impossible order metadata or a stricter campaign policy could be accepted | Fixed: distinct positions/disjoint run IDs; campaign-specific block and duration floors in verified derivation |
| Blocking for GPU claims | Production dispatch fences after each call; TS1 also performs host-driven forcing copies inside the timed schedule | The current timer includes production synchronization/forcing work and is not a pure device-kernel metric | Preserve this production boundary; add separately labeled device-resident/kernel measurements if needed |

## Workspace and parallel design

The dense bridges occur in
`src/mkpp/templates/solver_variants/integrate.j2`, not merely in an obsolete
benchmark. For TS1, the declared three buffers total:

`3 * 210 * 210 * sizeof(double) = 1,058,400 bytes` (about 1.01 MiB per cell invocation).

This is a source-level storage calculation, not measured stack size or a claim
that every byte survives optimization or is live simultaneously. Actual stack,
register allocation, spills, local/global memory traffic and occupancy need
compiler output and profiles. The source currently does not establish that these
buffers are harmless on either CPU or GPU.

A corrected sparse workspace design should:

1. Assign deterministic compact offsets to structural Jacobian entries and LU
   fill entries during AOT symbolic analysis. Do not perform dynamic lookup in
   the integration kernel.
2. Have compiled Jacobian chunks write only those compact slots, and pass compact
   W/LU storage through factorization and stage solves. Storage should scale
   with admitted sparse structure/fill, not automatically with N squared.
3. Reuse buffers only after proving their live ranges and stage dependencies;
   do not alias W and LU just to reduce a source-level byte count.
4. Allocate/reuse workspace outside solve-only timing, with nonoverlapping
   per-cell/per-worker ownership. Select its layout from the actual CPU/GPU
   access pattern and measure it. A single layout or team size is not presumed
   best for both backends.
5. Keep original and compact implementations available for validation. Check
   RHS/Jacobian, matrix assembly, factors/residuals, complete trajectories,
   rejection behavior, determinism and failure propagation before comparing
   time and memory across small and large mechanisms.

`RangePolicy` cell parallelism can be useful when each cell has enough independent
work and local storage is controlled. A cooperative team/vector implementation
is another candidate, not an automatic improvement: dense team LU may compute
many structural zeros compared with symbolic sparse LU. Evaluate equal-chemistry,
equal-accuracy alternatives using measured time, memory and scaling. The current
large compiled scratch makes this a priority experiment.

Do not remove production fences merely to improve a stopwatch result. First
define an asynchronous API with correct data/stream lifetimes and dependency
ordering, then compare that API separately with its real consumer.

## Correct measurement contract

**Work identity.** Every paired result must bind mechanism bytes/phase properties,
units, complete member initial state and ordering, meteorology, forcing schedule,
checkpoints, horizon, common work and timing boundary. MUSICA 0.16.5's exact TS1
v1 inputs remain pinned. The native protocol still needs full member/ensemble
execution evidence; checking the first member is insufficient.

**Achieved accuracy.** Use converged independent reference trajectories and
per-species scales. Calibrate native methods/controls on held-out calibration
inputs with equal search budgets. Freeze each solver's selected controls before
measurement. Shared `rtol`/`atol` values alone do not establish equal accuracy.
A performance helper's `scientific_eligible` argument must come from evaluated,
hash-bound gates; it is not proof those gates ran.

**Boundaries.** Keep production steady-state solve (including work that the
production call actually performs), host-to-host lifecycle (including allocation,
initialization/transfers and completion), and optional kernel microbenchmarks
separate. Native process startup, AOT generation and compilation are additional
costs to report and amortize for the intended deployment, not hide or include
asymmetrically. A one-time setup cost can be worthwhile for many repeated solves;
report the actual reuse assumption.

**Warmups.** Perform/reset repeated solves in the same process, runtime, allocation
and device context that owns the measured sample. Retain native per-warmup
throughput and the stabilization decision. The diagnostic CLI now requests the
maximum declared warmups in each process; it makes no stability claim. The
standalone stability helper remains useful once real in-process observations are
available.

**Pilots.** Remeasure all subjects at the same count; three pilot observations per
subject must each form a finite distribution and its median must reach the common
floor. Use the same declared warmup policy as measurement. Preserve pilot outputs
without including them in the paired inferential sample.

**Timer quality.** Retain the number of timed intervals and measured timer-pair
cost on the actual clock/backend. For N intervals, elapsed total T and measured
per-pair overhead o, inspect `N * o / T` against a predeclared budget (for example
1%). Increasing N does not necessarily lower this ratio. Do not subtract an
unreliable guessed overhead. If a latency case is too short, report that limit
or use a separately defined valid measurement boundary; silently increasing its
cell count changes the workload. Synchronize before/after asynchronous work.

**Order and noise.** Eighteen ringer blocks complete three six-permutation cycles
for three subjects and nine two-permutation cycles for two subjects. Retain all
valid observations, failed attempts and invalidation reasons. Distinguish solver
failure from externally invalid measurement; do not retry scientific failure
until success. Bootstrap paired blocks together and report its sampling
assumptions. A seeded percentile interval does not automatically correct serial
correlation, order effects, multiple comparisons or shared-node noise. Repeat
independent campaigns and inspect drift/autocorrelation before making a claim.

**Publication.** Pass the campaign's actual `minimum_valid_blocks` and
`minimum_sample_seconds` to `derive_verified_comparison`. Its checks cover retained
pair consistency and duration floors, not missing scientific gates, timer
calibration, warmup telemetry, actual affinity or GPU execution. The diagnostic
CLI remains nonpublishable. Show absolute time, uncertainty, achieved accuracy,
failures, throughput, hardware/resources and per-case losses before any aggregate.

## Acceptance and verification

The focused suite after this review is **258 passed, zero skipped**, including
nonmonotonic shared pilots, contradictory calibration evidence, malformed
sampling policies, incomplete permutation cycles, impossible order metadata and
campaign-specific report floors. Ruff on modified source/tests and whitespace
checks pass. The exact MUSICA package input check continues to pass.

Native compilation and CPU/GPU performance profiling remain unperformed in this
environment. No generated solver, sparse numerical implementation, device launch
policy or production synchronization behavior was rewritten in this review.
Those changes need the compiled numerical and hardware checks above. The design
has explicit correctness criteria now, but the remaining implementation and
measurement gaps prevent a performance sign-off.
