# Atmospheric cell scaling changes

This change implements the first scaling milestone: compact numeric matrix storage, direct sparse traversal, independent cell drivers, and removal of unused domain-sized workspace. It also includes the earlier MUSICA v1 benchmark and Kokkos 5.1.1 build corrections from this review.

## Solver and memory changes

The default forward integrator stores only structural Jacobian entries, iteration-matrix entries, and LU fill. For TS1 these are 1,967, 1,975, and 4,803 binary64 values: 69,960 bytes rather than 1,058,400 bytes for three 210-by-210 matrices. This is a 15.1-fold reduction in these numeric matrix buffers, not a measurement of total process or GPU memory. Other vectors and shared immutable traversal metadata are additional storage. The dense unrolled reference remains available for validation.

Factorization uses an ordered schedule of nonzero product pairs, preserving the previous row-wise Doolittle order. Triangular substitution visits 4,593 structural off-diagonal entries instead of 43,890 dense positions. These counts are not claimed runtime speedups.

The C interface no longer allocates an unused `num_cells * (num_species + num_reactions + 16)` scratch view. For one million TS1 cells, that removes a 6.184 GB allocation. State storage still scales with the model domain; numeric solver scratch is local to active solves.

## Independent cell inputs

The Kokkos dispatch accepts optional per-cell temperature, density, relative humidity, cloud water, and runtime reaction-parameter views while retaining scalar/shared inputs for uniform box cases. Strided cell forcing rows are gathered through their actual Kokkos mapping into a bounded per-cell buffer. The generated rate and Jacobian expressions preserve temperature and background-density symbols when needed instead of folding them into compilation constants.

The host C API uses `DefaultHostExecutionSpace` for caller-owned host buffers. It validates positive finite temperature, density, and pressure, waits before returning host buffers, and uses LayoutLeft cell/parameter indexing consistently. This does not create a device-pointer C API.

The complete forcing interface is `mkpp_set_rate_parameter_ptrs(handle, values, count)`, with `count = MKPP_NUM_RATE_PARAMETERS`. Values are stored as `values[cell + num_cells * parameter]`. Matching C++ and Fortran wrappers are emitted. The older photolysis setter remains valid for mechanisms needing only photolysis slots; it explicitly rejects incomplete arrays for mechanisms that also need external reaction parameters. Species tendency output is documented as `num_cells * MKPP_NUM_SPECIES`.

## Validation

The native heterogeneous test uses different states, temperatures, and photolysis values in adjacent cells. Each batched rate and integration result is compared with an independent single-cell solve. It checks an analytic temperature-dependent rate ratio, rejects invalid meteorology, and compares compact and unrolled-reference trajectories. Both Serial and four-thread OpenMP Kokkos 5.1.1 builds are exercised. A dedicated PR workflow repeats these native checks for both CPU backends.

Random diagonally dominant systems generated from Chapman and GOCART patterns are checked against dense matrix residuals. All eight checked-in mechanisms are regenerated to keep generated artifacts consistent with the templates.

## Remaining work

GPU execution and metadata placement have not been validated here. The remaining optimizations are separate experiments: bounded per-cell solver telemetry/failure handling, carry-over step estimates, validated Jacobian reuse, scheduling by measured cell stiffness, SIMD/tiled layouts, and asynchronous device-resident model coupling. No scientifically admitted MKPP/MICM/KPP performance ranking is claimed by this PR.

The isolated compact-kernel diagnostic retained the previous fixed-temperature case and took 0.637 seconds for one TS1 cell versus approximately 7.16 seconds in the earlier dense run. After enabling physical runtime environment inputs, a later run took 1.786 seconds. These single-sample observations occurred during development with other jobs active; they are not controlled speedup measurements. The final runtime-environment result differed from the dense final state by at most 5.93e-12 relatively. Against the retained MICM final checkpoint, normalized WRMS was 0.033815 and maximum normalized error 0.278125 using atol=1e-12 and rtol=1e-8. This is a final-checkpoint diagnostic, not a complete trajectory/reference gate. Temperature-only rate coefficient caching is a useful next experiment.
