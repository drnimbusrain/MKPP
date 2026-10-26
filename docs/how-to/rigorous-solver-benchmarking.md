---
type: how-to
category: how-to
tags: [benchmark, accuracy, reproducibility, micm, kpp, gpu]
---

# Put MKPP through a rigorous solver comparison

A benchmark is a scientific experiment. A passing harness test, schema validation,
or a short native run is not evidence that MKPP is faster. Publish only matched
workloads that pass chemistry, independent-reference, held-out accuracy, resource,
and repetition gates. Retain cases where MKPP loses or fails.

## Current readiness

The supplied branch is **not ready for a publishable three-solver comparison**.
The diagnostic CLI does not wire the scientific gates into its execution path.
The existing accuracy, reference, admission, and calibration helpers are building
blocks, not proof those stages ran. The following are concrete source blockers:

| Blocker | Evidence in this source tree | Required resolution |
| --- | --- | --- |
| KPP runner is absent | `runners/CMakeLists.txt` adds `kpp`, but that directory is absent | Implement a real runner from the pinned KPP revision and its generated solver; never substitute MKPP or SciPy |
| TS1 input was omitted from the archive (fixed here) | The restored CSV has the exact pinned SHA-256 `58db2925…9253` | Keep the source CSV tracked using the new narrow `.gitignore` exception |
| Scenario chemistry identities disagree | Chapman, small-strato, and SAPRC99 scenario hashes differ from `chemistry_sha256` in their manifests | Review the intended identity and version affected inputs after resolving it |
| Workloads are not supplied to runners | CLI supplies labels/repetition counts, but does not dispatch the scenario's full environment, forcing, initial state, cell sweep, or seed | Use one resolved input contract and verify each runner actually executes it |
| Chapman defaults differ | MKPP defaults to 10 x 60 s; MICM defaults to 1 x 60 s; scenario requests 86400 s and five checkpoints | Execute the same checkpoint/forcing schedule in all runners |
| Trajectory output is partial | Native runners emit one final checkpoint, not the complete requested trajectory; multi-cell output is not complete member evidence | Emit every requested checkpoint for every member, including required fixed species |
| Build/resource provenance is incomplete | MKPP reports `unavailable-at-runtime` for binary hash; MICM reports zero hashes; affinity is unspecified/external-policy | Capture actual binaries, source revisions, flags, CPU masks, thread counts, and topology outside timed regions |
| Equal-accuracy calibration is not integrated | Diagnostic CLI labels a shared configuration without performing calibration | Calibrate actual native controls on separate inputs and freeze each solver's configuration before measurement |
| Timing boundaries are incomplete | Campaign declares lifecycle but runners emit solve timing and empty lifecycle maps | Measure each declared boundary consistently or withhold that boundary |
| Mechanism routing is incomplete | One `--runner` per solver cannot select the separate MICM Chapman and TS1 executables automatically | Route by solver and mechanism, and probe supported workloads |

The canonical chemistry must explicitly declare OpenAtmos configuration version
`1.0.0`. MusicBox workload settings, the benchmark's `schema_version: 1`, and
manifest/scenario version strings are separate identities and do not certify that
format. The new benchmark gate rejects unversioned and v0 chemistry before native
execution. It checks the v1 envelope and phase membership; full upstream parsing
and chemistry audits are still needed.

The pinned TS1 JSON already declares `1.0.0`. The other three benchmark chemistry
sources are unversioned legacy conversions with incomplete phase membership;
they are now blocked. Do not certify them by adding a version field alone. Use
reviewed v1 configurations and re-admit each native binding. The MUSICA package's
v1 Chapman example has six species and seven reactions, while the current
benchmark has four species and four reactions: exchanging them changes the
scientific problem and requires new scenarios and all solver bindings.

TS1 now matches the project's pinned `musica==0.16.5` distribution byte for byte,
including its `version: 1.0.0`, phase species properties, and original initial
conditions. The prior vendored copy changed GLYOXAL's phase diffusion coefficient
from the package's `1e300` to `1e-5` and added MKPP metadata. Both differences have
been removed from the canonical file. MKPP's artifact name is supplied with
`--mechanism-id ts1` during generation. The MICM importer now preserves that large coefficient rather than substituting
its own default, and the governed CMake build selects binary64 MICM. The native
CMake source hashes and TS1 manifest/scenario identities are updated; the latter are versioned `1.1.0` to
identify this changed scientific input. This is independent of input format v1.

`musica-v1.lock.json` pins the package version, format version, input paths and
SHA-256 values. `v1-source-comparison.json` retains the previous discrepancy and
records the resolved exact match. The old repository commit is no longer claimed
as the authoritative source of these bytes. Python is required for MKPP TS1
regeneration; an unverified legacy generated header cannot serve as a fallback.

Verify the installed comparison dependency matches these inputs:

```bash
python -m pip install '.[solver-comparison]'
python scripts/check_musica_v1_inputs.py
```

The pinned MUSICA parser accepted the exact TS1 file as v1.0.0 with 210 species
and 547 reactions in this environment. That format check does not establish
trajectory equivalence or timing eligibility.

Do not change a manifest to `admitted` to get past these blockers. Admission must
reference retained numerical evidence. TS1, small-strato, and SAPRC99 currently
have only MICM/MKPP bindings. Chapman alone declares all three. An overall claim
against KPP needs an admitted KPP binding for representative larger chemistry.

## Reproduce the contract checks

Use an isolated Python environment with the project's dependencies and test tools:

```bash
python -m pip install -e . pytest hypothesis
python -m pytest tests/python/benchmark -n 0
python scripts/compare_solvers.py --list-mechanisms
python scripts/compare_solvers.py --validate \
  --campaign benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml
python scripts/compare_solvers.py --audit \
  --campaign benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml
```

`--validate` checks input structure. `--audit` checks declared asset presence,
pinned source digests, chemistry identity, and declared admission state. Its JSON
always says `publishable: false` and exits 4, because it executes no numerical
qualification. The JSON schemas are versioned under `src/mkpp/benchmark/contracts`
and included in Python wheels. Missing schemas now fail tests instead of hiding
80 contract tests behind skips.

Run repository-wide hooks in the project's configured environment before merging:

```bash
pre-commit run --all-files
```

## Stage 1: chemistry equivalence before integration

For every binding, compare canonical species, active/fixed roles, units, reaction
stoichiometry, temperature/pressure dependencies, photolysis channels, and rate
constants at the same named states. Include the initial state, trajectory states,
and at least eight seeded perturbations. Check RHS, Jacobian values and sparsity,
and an independent finite-difference or complex-step Jacobian where supported.
Exercise zero concentrations as well as positive states; the current audit-state
builder accepts only positive inputs, so zero-state testing needs a separate path.

Keep chemistry discrepancies out of solver-performance rankings. A different
rate law is a different scientific problem. Record unsupported chemistry as
unsupported, with the reason, rather than silently removing terms.

## Stage 2: equal achieved accuracy

Run an independent stiff reference (the scenario specifies SciPy Radau) at two
successively tighter tolerance settings. Integrate to every forcing discontinuity
and requested checkpoint. Reserve the existing reference budget: worst checkpoint
WRMS <= 0.05 and maximum scaled error <= 0.25; refine or withhold if it does not
converge. A three-solver agreement alone is not an independent reference.

Use each species' declared absolute scale and the shared symmetric error:

`e_i = (subject_i - reference_i) / (atol_i + rtol * max(abs(subject_i), abs(reference_i)))`

Require worst checkpoint/member WRMS <= 1 and maximum absolute scaled error <= 5,
plus finite state, successful integration, positivity, and applicable conservation
checks. Report the worst species/time/member and full error distribution. Do not
clip negative concentrations to make an error gate pass. With fixed atmospheric
reservoirs, define conservation accounting to include the reservoir/source terms;
a naive closed-system oxygen sum may not be invariant.

Use the predeclared tolerance-multiplier grid in `calibration.py` on calibration
inputs only. Collect five calibration blocks per candidate. Choose the fastest
eligible configuration using the existing tie-break policy, freeze the actual
controls and hash, and evaluate on disjoint measurement states/seeds. Equal
numerical `rtol` does not guarantee equal achieved accuracy across controllers.
Show work-precision curves as well as the selected equal-accuracy points. Include
a same-method ROS3 comparison and a separately labeled best-native-method
comparison using identical tuning budgets for every solver.

## Stage 3: a predeclared stress matrix

The new `cpu-ringer-v1.yaml` declares the four existing mechanism pairs and an
18-block sampling policy. It is a coverage plan; the current runner limitations
above must be resolved before executing its declared workload. The expanded
stress cases below are a protocol to implement and version, not generated or
executed results. Use a staged matrix to control cost:

| Dimension | Required cases | Purpose |
| --- | --- | --- |
| Mechanisms | Chapman, small-strato, SAPRC99, TS1 | Small diagnostic through representative larger systems; state exact participating solvers |
| Batch size | Each scenario's declared sizes, initially 1, 10, 100, 1000, 10000 where declared | Latency through throughput; preserve the same member set across solvers |
| Temperature | 220, 270, 310 K | Cold, middle, warm chemistry within each mechanism's supported range |
| Pressure | 20000, 80000, 101325 Pa | Pressure-dependent rates; keep atmospheric density and fixed species physically consistent |
| Forcing | Night/zero, daytime, dawn/dusk schedule, abrupt changes at declared boundaries | Restart and discontinuity handling |
| Initial state | Canonical; sparse/zero trace species; seeded log perturbations; high radical and polluted cases | Stiffness and nontrivial reaction activity |
| Accuracy | Loose, target, tight budgets; species-scaled absolute floors | Work-precision behavior and tolerance cliffs |
| Horizon | Short transient, declared standard, long day/night cycles | Startup versus accumulated drift |
| Ordering | Original member order and one shared shuffled order | Heterogeneity, SIMD tails, divergence, ordering dependence |
| Failure/restart | Step budget exhaustion, impossible tolerances, nonfinite input, recoverable solver failure | Explicit failure status, bounded execution, no fabricated state |
| CPU layout/scaling | Production layout; 1, 2, 4 cores; representative vector/batch tails | Layout and parallel scaling, labeled separately from the primary comparison |
| GPU, if available | Small through saturation batches; homogeneous and heterogeneous cells | Verified device execution, divergence, capacity, transfer cost |

Start with all mechanisms at the canonical state, full checkpoint schedule, target
accuracy, and every declared cell count. Then screen each physical dimension
individually at one small and one large batch. Run selected interactions (cold +
low pressure + radical transient; polluted + dawn; sparse + tight tolerance) and
all failures. Predeclare escalation to the full crossed matrix if the screen finds
sensitivity. Do not select favorable cases after seeing speedups.

## Stage 4: measurement controls

Use release builds, binary64, audited pinned revisions, and comparable optimizer
and floating-point policies. Report generation/compile cost and binary size
separately. Disable reduction approximations or give all solvers equivalent
chemistry. Pin the primary CPU comparison to one physical core, one thread, the
same NUMA placement, and the same idle reserved node. Thread environment variables
alone do not demonstrate physical-core affinity.

Use three pilot observations and one common repetition count long enough that
even the fastest subject meets the one-second sample floor. Restore identical
initial state before each repetition; reset/allocate outside solve-only timing for
all subjects. Include rate-update work consistently. Synchronize device work
before and after timing. Check solver status and consume the resulting state so
work cannot be optimized away.

Use 18 paired blocks for three solvers: three cycles of all six permutations.
The same count also balances nine cycles for two solvers. Seed and shuffle the
permutation order in each cycle. This balances both positions and within-block
predecessors; truncated cycles may not be perfectly balanced. Never remove a
slow valid observation. Predeclare invalidation criteria and retain invalid
attempts before recollecting the entire pair/block. A solver failure is a
scientific outcome, not a reason to keep retrying until success.

Treat complete native integration failures, schema failures, and timeouts as
failures. The diagnostic CLI writes process transcripts before decoding results,
then appends each validated measurement immediately. Prior measurements survive
a later failure. `<raw.jsonl>.transcript.jsonl` retains phase, solver, exit status,
stdout, and stderr; transcript records are not timing samples. Use a fresh output
path and campaign identity for each independent run. Partial files are explicitly
nonpublishable. The primary raw stream only includes results that pass the
requested identity and complete named checkpoint checks.

For GPU results report device-resident solve time and full host-to-host lifecycle
separately, including transfers, allocation, launch, synchronization, and any
preprocessing such as sorting. Verify the production path ran on the device;
host compilation or a GPU label is insufficient. A GPU-versus-single-core ratio
must state that resource difference. Report CPU core counts, hardware, memory,
compiler, backend, and optimized baseline configuration alongside every result.

## Stage 5: publication and regression evidence

Use `derive_verified_comparison` with the full retained `SolverRun` series and
completed scientific eligibility. It checks block identity, campaign, workload,
resources, precision, build/configuration stability, durations, synchronization,
and at least ten valid blocks. It aligns observations by block ID and withholds
unsafe ratios. The scientific-eligibility input must come from evaluated gates;
it is not a user-provided approval flag. Different solvers may use different
frozen configurations after equal-accuracy calibration.

Report directed baseline-seconds/candidate-seconds ratios, their median and seeded
95% bootstrap interval (10000 resamples), absolute seconds, cell-steps/second,
IQR/MAD, accuracy, and failure counts. Do not mix workloads in one paired series
or average ratios across incomparable mechanisms. Show per-case tables first;
any aggregate must declare weights in advance. An interval overlapping 1 provides
insufficient evidence of a speed advantage under that experiment.

Repeat the campaign independently at least once; the existing repeatability
criterion is a <=5% change in median throughput. If it fails, investigate noise
and retain both runs. For regression gates compare MKPP to its own pinned prior
version, with a predeclared meaningful regression budget and accuracy gate;
instruction counts supplement wall-clock results. The current performance
workflow is manual-only and references missing profiling assets, so it is not an
active automated performance guarantee.

Archive canonical inputs, seeds, reference attempts, native controls, versions,
binaries/hashes, build logs, topology/affinity evidence, raw outcomes, all failed
attempts, and report-generation commands. Include MKPP losses, unsupported cases,
and withheld rows in the main results. Never infer a speedup from this audit or
from unit-test runtime.

## Performance design follow-up

The diagnostic executor now measures every solver at each proposed shared pilot
count, using the same maximum declared warmup count as measurement. It no longer
compares fresh-process timing runs to claim warmup stability. Native in-process
warmup telemetry and timer-overhead evidence are still required. The ringer
policy validates full permutation cycles before execution. Verified report
derivation accepts the actual campaign block and duration floors.

See `PERFORMANCE_DESIGN_REVIEW.md` at the repository root for the dense quadratic
scratch, actual RangePolicy path, measured timing contract, and remaining CPU/GPU
profiling requirements.
