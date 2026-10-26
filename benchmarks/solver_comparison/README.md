# Solver comparison assets

This directory contains immutable, reviewed inputs and native runner sources for
fair MICM, KPP, and MKPP comparisons.

- `manifests/` records canonical chemistry and audited solver bindings.
- `scenarios/` records immutable calibration and measurement workloads.
- `campaigns/` records resource, ordering, repetition, and publication policy.
- `runners/` contains isolated native harnesses for the real named solvers.

Generated sources, build trees, raw results, and reports do not belong here.
They must be written beneath ignored build or report-output directories. Generated
headers and baseline CSV files are evidence, never canonical chemistry.

For every governed comparison, each native solver binding is generated from the
same pinned OpenAtmos mechanism and atmospheric-state inputs. MKPP must not use
a benchmark-only YAML conversion as an alternative chemistry definition.

MICM and KPP are pinned external comparison subjects. They are not linked into
MKPP production libraries and their native implementations do not imply MKPP
backend portability. MKPP-owned runner code continues to use the production host
dispatch path and Kokkos-compatible layouts.

For the source-readiness audit, staged stress matrix, and publication protocol,
see [rigorous solver benchmarking](../../docs/how-to/rigorous-solver-benchmarking.md).
The planned `cpu-ringer-v1.yaml` covers all four existing mechanism pairs and uses
18 blocks. It does not admit any new solver binding or implement the currently
missing native scientific-gate wiring.
