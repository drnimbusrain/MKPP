# Native build and run report

This report extends the earlier benchmark and performance-design reviews with actual local compilation and execution. Runs are diagnostic and are not an admitted MKPP/MICM/KPP performance comparison.

## Dependency and build changes

- Require Kokkos **5.1.1 or newer** in both CMake entry points; reject older headers in the host dispatcher. CI now pins 5.1.1.
- Build and install exact Kokkos tag 5.1.1, commit `267ebc25fc5c8b96bb321f34d78f97b1c30f8830`.
- Remove the unnecessary mandatory Fortran language from the C++ host interface. Root configuration probes Fortran before enabling it. This allows the C++ targets to configure without a Fortran compiler.
- Use MICM 3.13.0 at commit `97ac9e5d8aadd345c242722ee8274d71dfe0f73e`, with double precision enabled.

## Environment and completed checks

GCC 13.3.0, C++23, Release (`-O3 -DNDEBUG`), Linux x86-64, Kokkos Serial CPU backend, binary64. KokkosKernels is disabled in this build. No GPU execution, verified physical-core binding, or Fortran compiler is available here.

| Check | Result |
|---|---|
| Kokkos 5.1.1 native build and installation | Passed |
| Root comparison CMake configuration without Fortran | Passed |
| Fresh MKPP TS1 generation from pinned MUSICA v1 inputs | Passed; 210 species, 547 reactions |
| Generated MKPP Chapman and TS1 translation-unit compilation | Passed |
| Production MKPP host registry, Chapman integration | Compiled and ran; 1 cell, 10 × 60-second steps, all final values finite |
| Complete MKPP benchmark runner, separate `-O0` diagnostic build | Compiled and ran Chapman and TS1; TS1 reached 600 seconds with 210 finite named species |
| Direct native MICM Chapman integration | Compiled and ran; final time 600 seconds, success |
| Direct native MICM TS1 integration | Compiled and ran; final time 600 seconds, success, 210 named species |
| Python benchmark regression suite plus MICM comparison tests | 258 passed, 0 skipped |

The complete MKPP diagnostic runner compiled with `-O0` and linked the Release Kokkos libraries; its timings must not be compared with the optimized MICM runs. TS1 returned status `success`, final time 600 seconds, and 210 finite values. This checks execution, not calibrated scientific equivalence. Exact diagnostic compiler/link commands and stdout are included.

The MKPP Chapman smoke uses the real `MechanismRegistry::execute` path, Kokkos views, an explicit fence, and a host copy of the final state. Its source is included with the evidence. It is a supplemental API harness, not the governed benchmark runner.

MICM smoke runs use three warmups and three repetitions. Raw elapsed values are retained in JSON for debugging; they are not used to calculate speedups. The standalone MKPP API smoke has no timed benchmark region.

## Limits that remain

KPP was not compiled or run: its native runner implementation is absent from this checkout, and no Fortran compiler is installed. GPU performance was not tested. KokkosKernels and other Kokkos backends were not tested.

The governed MKPP runner includes an approximately 11 MiB TS1 generated header at the integration call site. Its small static archives contain no emitted math implementation symbols; compiling those archives does not establish that the full integration executable has compiled. The full optimized executable build is recorded separately below.

The generator reports that TS1's per-thread matrix storage exceeds its 262,144-byte budget. The earlier performance-design concern remains: three dense binary64 210 × 210 buffers alone require 1,058,400 bytes per solver invocation before other state. Successful CPU smoke execution would not establish GPU feasibility.

The uncalibrated final-state comparison confirms matching species names. Its largest relative discrepancy is the artificial `sink` at approximately 10⁻³⁰; the report retains that value rather than hiding it. This observation is not an accuracy-gate result.

No scientifically admitted timing comparison is claimed. The calibration/reference gates, complete trajectory output, validated runner provenance, cross-solver tolerances, and KPP availability identified in the earlier reviews still need to be completed before publishing a ranking.

## Reproduction and evidence

Use a Python virtual environment with the project and its test requirements installed, CMake, Ninja, SymPy, Jinja2, NetworkX, and netCDF4. The checked-in `reports/native-kokkos-5.1.1/reproduce.sh` records dependency checkouts and the native build commands. It expects fresh dependency destination directories. `configure.log`, `build.log`, `environment.json`, raw native stdout/stderr, and supplemental harness source are retained in the same directory. No external dependency trees or build caches are included in the deliverable.

## Build-time improvement and optimized native execution

The initial monolithic `-O3` compilation was interrupted after several minutes without completing; it was not a compiler error and does not provide a completed baseline time. A separate inline `-O1` runner object compiled in 13.564 seconds, and the inline `-O0` diagnostic runner compiled and ran successfully.

The implemented CPU fix keeps `-O3` and moves the heavy helpers behind ordinary declarations. Generated `.cpp` files now emit real external definitions from the same algebra fragments. `nm` confirms 18 TS1 helper definitions in the archive, which grew from approximately 43 KiB of inline-only objects to 744 KiB with actual compiled chemistry. No chemistry expressions or integration tolerances were changed.

| Local observation | Result |
|---|---|
| Incremental build with no source changes | 0.166 seconds; Ninja reported no work |
| New MKPP runner translation unit, `-O3` | 50.055 seconds |
| Native build of MKPP, MICM Chapman and MICM TS1, parallelism 2 | 126.422 seconds, including TS1 generation |
| Fresh CMake configuration after removing configure-time generation | 0.6 seconds, versus the previous 61.2-second configuration |
| Optimized MKPP Chapman and TS1 runs | Passed; final time 600 seconds, finite concentrations |
| Optimized MICM Chapman and TS1 runs through the CMake-built executables | Passed; final time 600 seconds, finite concentrations |
| New optimized TS1 final state versus earlier inline diagnostic final state | All 210 serialized species values exactly equal |
| Benchmark, MICM comparison and selected code-generation regression tests | 291 passed, 0 skipped |

These are local build observations with concurrent processes, not controlled compiler benchmarks. The original `-O3` baseline did not complete, so no exact build-speedup ratio is asserted. Separating functions can affect runtime optimization; a controlled runtime comparison against a completed optimized inline build remains required. The matching final state checks this smoke case only.

The separate CPU path is enabled by `MKPP_SEPARATE_HOST_KERNELS=ON` for newly generated artifacts registered through the CMake helper. Device backend configurations retain inline bodies. Older generated artifacts remain compatible and use their existing path until regenerated. The TS1 generation rule now tracks generator/template dependencies and declares implementation-header byproducts. CMake registers its known generated sources without invoking generation at configuration time, eliminating duplicate initial generation.

See [reduce-native-build-times.md](docs/how-to/reduce-native-build-times.md) for build options. Raw optimized outputs, command durations, executable hashes, symbol evidence, final-state comparison and test output are under `reports/native-kokkos-5.1.1/`.

The final CMake rule update triggered one regeneration/rebuild (122.336 seconds), followed by the 0.166-second no-op check. The final rebuild changed executable hashes; final executable smoke results and hashes are retained separately in `final-native-runs.json`.
