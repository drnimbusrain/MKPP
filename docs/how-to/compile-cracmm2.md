---
type: how-to
category: mechanisms
tags: [cracmm, epa, openatmos, compilation]
---

# Compile CRACMM2

MKPP includes the Community Regional Atmospheric Chemistry Multiphase Mechanism
(CRACMM2) published by the U.S. Environmental Protection Agency. The canonical
input is [mechanism.json](../../mechanisms/openatmos/cracmm2/mechanism.json),
generated from the pinned EPA reaction and species tables in the adjacent
`source` directory.

The source snapshot is pinned to commit
`8a71fdbfb2183a1e37a53eb237e265d12f09eb245` from
[USEPA/CRACMM](https://github.com/USEPA/CRACMM). Regenerate the canonical input
after updating those source files with:

```bash
python3 scripts/import_cracmm2.py \
  --species mechanisms/openatmos/cracmm2/source/cracmm2_metadata.csv \
  --reactions mechanisms/openatmos/cracmm2/source/cracmm2_rxn_metadata.csv \
  --output mechanisms/openatmos/cracmm2/mechanism.json
```

Compile the generated MKPP solver with the normal catalog command:

```bash
python3 -m mkpp.cli compile \
  mechanisms/openatmos/cracmm2/mechanism.json \
  --test-env example_env.yaml \
  --out mkpp-generated \
  --adjoint --emit-manifest --report --no-cache
```

Recognized multi-channel reaction rates are mapped to MKPP's `EP2` and `EP3`
forms; all 11 recognized pressure-dependent associations are mapped to the
standard `TROE` reaction type. The six remaining `UNKNOWN` rates are
expressions that cannot be represented safely and are listed in
`metadata.unconverted_rates` with their original expression and reason. The
current unresolved reactions are R051, R064, R067, R167, R169, and HAL_Ozone.
They need a separately implemented custom-rate path; the importer does not
substitute guessed constants.
