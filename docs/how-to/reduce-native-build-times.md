# Reduce native build times

Use Kokkos 5.1.1 or newer and a Ninja Release build. CPU libraries created by
`mkpp_add_compiled_mechanism` now compile the heavy generated chemistry helpers
separately by default (`MKPP_SEPARATE_HOST_KERNELS=ON`). The solver and helpers
retain the selected build's optimization flags, including `-O3` for Release.

Previously each generated `.cpp` included only inline definitions, which did
not emit the helpers into the static archive. The integration consumer still
optimized all the rate, Jacobian and factorization bodies in one translation
unit. Linking that archive therefore did not bound the optimization work.

The new mode emits ordinary declarations in the CPU consumer and real helper
definitions in the generated kernel objects. It uses the same generated algebra
and layout-aware accessors. CUDA, HIP, SYCL and OpenACC configurations retain
the visible device-callable definitions; this change adds no relocatable device
code requirement. GPU compile-time improvements have not been measured.

Regenerate older mechanisms to produce `host_declarations.hpp`; older generated
artifacts continue to use their existing inline implementation. Link each
mechanism's compiled target so its public compile definitions and objects agree.
Do not set the per-mechanism preprocessor definitions manually.

```bash
cmake -S . -B build/release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DKokkos_ROOT=/path/to/kokkos-5.1.1 \
  -DMKPP_ENABLE_SOLVER_COMPARISON=ON -DMKPP_SEPARATE_HOST_KERNELS=ON
cmake --build build/release --parallel 2
```

For a controlled inline baseline, configure another build directory with
`-DMKPP_SEPARATE_HOST_KERNELS=OFF`. Compare both build cost and runtime accuracy
and performance before selecting production settings. Build times are machine
and compiler dependent; the local observations are in `NATIVE_RUN_REPORT.md`.
Link-time optimization can reopen cross-unit optimization work and should be
measured separately.

For quick development, `-O1` is also a practical option: the original TS1 runner
compiled in 13.6 seconds in the local diagnostic experiment. That is a different
optimization setting and must be recorded when comparing solver performance.
Keep development and benchmark builds in separate directories. An incremental
build should reuse generated chemistry and compiled objects; the TS1 generation
rule now tracks Python and Jinja template inputs so edits regenerate correctly.

TS1 generated output can be isolated per build with
`-DMKPP_TS1_GENERATED_DIR=/absolute/path/inside/source/build/ts1-name`.
