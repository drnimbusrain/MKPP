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

CRACMM expressions that have no exact OpenAtmos rate-law equivalent are exposed
as named host forcings. Their source expressions remain in the mechanism
metadata; they are not replaced with guessed constants.
