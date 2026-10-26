---
type: tutorial
category: tutorials
tags: [benchmark, micm, kpp, mkpp, reproducibility]
---

# Benchmarking MICM, KPP, and MKPP

This workflow is for fair, reproducible solver comparisons. A headline ratio is
only valid when it is derived from retained native-runner evidence after the
chemistry, trajectory, configuration, and resource gates have passed.

Start by inspecting and validating the immutable inputs:

```bash
python scripts/compare_solvers.py --list-mechanisms
python scripts/compare_solvers.py \
  --validate --campaign benchmarks/solver_comparison/campaigns/cpu-primary-v1.yaml
```

Each mechanism manifest identifies the canonical reaction/species source,
binding revisions, and required solvers. Every solver must receive the same
canonical chemistry and host-supplied atmospheric state (temperature, pressure,
and named forcing). TS1 is explicitly two-way—MICM and MKPP only—until KPP has
an independently admitted binding. It must never contain a KPP placeholder.

Before publishable timing, a complete campaign must verify mappings, reaction/rate-law
audit evidence, an independent reference, calibration, and trajectory accuracy.
The current diagnostic executor does not yet wire those scientific gates together. A failed or
missing runner result is retained as a failure and withholds the affected ratio;
it is never replaced by another solver or copied state.

The primary CPU result uses one physical core and one thread for every paired
solver. It requires balanced, block-paired repetitions (the primary v1 plan declares 15;
the ringer plan declares 18 for complete permutation cycles; at least 10 publishable), with stable warmups and a fixed bootstrap seed. Solve and
lifecycle timing boundaries remain separate. A ratio is always named as
`baseline_seconds / candidate_seconds`, so a value above one means the
candidate was faster than the stated baseline.

Raw JSONL is append-only. Markdown, JSON, and optional CSV reports are
reproducible derivations that preserve admission status, uncertainty intervals,
and withholding reasons. Do not infer performance from diagnostic scripts or
from a non-publishable smoke run on uncontrolled hardware.

The native campaign executor collects diagnostic evidence only. It checks returned
request identities, hashes, and complete named checkpoint state, and retains
process transcripts and completed measurements if a later invocation fails.
Reports do not derive headline ratios. Validation alone is not scientific
admission. Run the offline audit (exit code 4, always nonpublishable):

```bash
python scripts/compare_solvers.py --audit \
  --campaign benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml
```

See [the rigorous comparison protocol](../how-to/rigorous-solver-benchmarking.md)
for current blockers, the stress matrix, hardware controls, and publication gates.
