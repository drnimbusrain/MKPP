# MKPP benchmark hardening and MUSICA v1 alignment

Subsequent native compilation and execution with Kokkos 5.1.1 are documented in [NATIVE_RUN_REPORT.md](NATIVE_RUN_REPORT.md). Earlier environment limitations below describe the review stage, not the later native run.

The supplied source archive cannot yet support a fair, publishable MKPP/MICM/KPP
speed comparison. This update closes evidence-handling gaps and aligns canonical
TS1 inputs exactly with the project's pinned MUSICA distribution. It does not
claim measured solver performance.

## MUSICA is the configuration authority

- Comparison package remains pinned to **MUSICA 0.16.5** in `pyproject.toml`.
- Canonical benchmark TS1 is now the package's **v1 / version 1.0.0** configuration,
  byte for byte. Mechanism SHA-256:
  `83b997e913194dd66046e7f76d54538dbed7e717f2ecb569e73d2c5f38b71732`.
- The missing TS1 initial-condition CSV was restored from that package. Its hash
  matches the original pin exactly:
  `58db29256362fcd7fd60d47fe3d463c566dfbe14c000757202393fa7a28a9253`.
- The previous canonical copy changed GLYOXAL's phase diffusion coefficient from
  `1e300` to `1e-5` and added MKPP metadata. These changes are removed. Artifact
  naming uses `--mechanism-id ts1` independently of canonical chemistry.
- The MICM importer also substituted `1e-5` for large coefficients. It now retains
  the original finite literal, with a precision guard. Governed comparison builds
  explicitly require binary64 MICM. Importer SHA-256 pins are updated.
- TS1 manifest/scenario versions are bumped to **1.1.0** to distinguish the new
  scientific input. This is separate from MUSICA's package and format versions.
- Source lock, provenance, license, and the historical/resolved difference record
  are included under `benchmarks/solver_comparison/sources/ts1`.
- Unversioned legacy benchmark chemistry is rejected before native execution.
  The old Chapman, small-strato, and SAPRC99 cases require reviewed migration.
  Changing a version string alone does not establish v1 conformance.

Run `python scripts/check_musica_v1_inputs.py` after installing the project's
`solver-comparison` extra to check package/version/hash equality against the
installed MUSICA distribution. This check uses distribution files without
importing native MUSICA libraries.

## Benchmark integrity changes

- Three JSON contracts are tracked and included in Python wheels. The previous
  schema-dependent test skip policy is removed.
- Native results must match requested campaign/run/block/order, complete input
  hashes, every checkpoint, every named species, and successful integration.
- Runner processes have bounded execution. Transcripts retain failed/malformed
  output; completed valid measurements are appended immediately and survive a
  later failure. A partial run remains nonpublishable.
- Pair validation rejects duplicate blocks/run IDs, same-solver comparisons,
  changed frozen evidence, invalid times, mismatched precision/resources, and
  unknown/GPU backends in the primary CPU result.
- Different solvers may have different calibrated configurations; those hashes
  no longer incorrectly form the shared workload key.
- Verified report derivation aligns blocks, requires at least ten valid pairs,
  and withholds results when matching/scientific gates fail. Markdown exposes
  uncertainty; CSV quoting round-trips commas, quotes, and newlines correctly.
- Three-solver sampling covers all six permutations, balancing within-block
  predecessors as well as positions. The planned ringer campaign uses 18 blocks.
- Accuracy arithmetic handles extreme finite values without false zero error
  from overflow. Empty trajectories, duplicate calibration identities, negative
  error metrics, and nonfinite repeatability samples are rejected.
- `--audit --campaign ...` provides a reproducible offline readiness inventory.
  It never claims scientific qualification and deliberately exits 4.

## Verification performed

| Check | Outcome |
| --- | --- |
| Original benchmark suite | 104 passed, 80 skipped because schemas were absent |
| Updated benchmark suite plus retired-wrapper checks | **258 passed, 0 skipped** |
| Ruff on benchmark source and changed/new test and helper files | Passed |
| Git whitespace check | Passed |
| JSON schemas in built Python wheel | All three present |
| Four-mechanism ringer campaign input validation | Passed |
| Installed MUSICA 0.16.5 versus vendored TS1 and CSV | Exact byte equality verified |
| Upstream MUSICA parser on canonical TS1 | v1.0.0; 210 species, 547 reactions |
| MICM TS1 native binding source generation | 210 species; 547 reactions; original GLYOXAL coefficient retained |
| Native compilation, trajectory qualification, CPU/GPU timing | **Not performed** |

Reproduce the focused tests:

```bash
python -m pytest tests/python/benchmark tests/python/test_compare_micm.py -n 0
python scripts/check_musica_v1_inputs.py
python scripts/compare_solvers.py --validate \
  --campaign benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml
python scripts/compare_solvers.py --audit \
  --campaign benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml
```

The test figures describe harness checks, not solver integration accuracy or
performance. The generated MICM header was not compiled; clang-format was absent.
Full repository hooks (`pre-commit run --all-files`) were not run. CMake, gfortran,
and GPU tooling are absent from this execution environment. No surrogate solver
measurements or guessed speedups are included.

## Work still required before a speed claim

1. Implement the missing KPP runner and admit real bindings; only Chapman
   currently declares a three-way comparison, and its old chemistry is blocked.
2. Migrate/rebind the other mechanisms to reviewed MUSICA-compatible v1 inputs.
   MUSICA's six-species/seven-reaction Chapman example differs from the current
   four-species/four-reaction case; this needs an explicit new scientific case.
3. Route native executables by solver and mechanism; pass the entire resolved
   initial state, environment, forcing schedule, cell ensemble, and checkpoints.
4. Connect chemistry audits, converged independent references, equal-accuracy
   calibration, and held-out trajectory checks to the campaign executor.
5. Capture real build/hardware/affinity evidence and consistent solve/lifecycle
   timings on controlled hardware. Refresh and verify generated bindings after
   the TS1 input change; previous numerical evidence does not transfer.
6. Execute the staged stress matrix and independent repeat campaigns. Preserve
   every loss, failure, unsupported case, and withheld comparison.

The detailed matrix and publication protocol are in
`docs/how-to/rigorous-solver-benchmarking.md`. Machine-readable current input
findings are in `reports/solver-benchmark-readiness.json`. Draft/ineligible status
is intentionally retained; no mechanism has been admitted by this patch.

## Subsequent performance-design review

See `PERFORMANCE_DESIGN_REVIEW.md` for the actual dispatch-path and dense-scratch
findings, corrected shared pilots and calibration gates, explicit warmup limits,
and the native profiling acceptance criteria. Passing unit tests does not close
those performance-design findings.
