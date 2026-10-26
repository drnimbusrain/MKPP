---
type: howto
category: how_to
tags: [device, kokkos, jvals, memory-space, portability, api]
---

# How-To: Bind Runtime Rates and Use the Device-Callable Solver Path

The generated solver kernels are callable from a Kokkos device execution space
as well as the host. The single source of truth for the rate law is the same
inline code in both cases; the difference is how externally supplied rates
(photolysis frequencies, and for cloud-gated mechanisms temperature / relative
humidity / cloud water) are bound into the call.

This guide shows how to migrate a C++ consumer that previously passed a raw
`const double*` rate array, and how to verify device callability on your build.

> **C and Fortran host models are unaffected.** The C ABI
> (`mkpp_set_photolysis_ptrs(handle, const double*)`, `mkpp_integrate`) still
> takes a raw host pointer and wraps it into a host-space view internally. Only
> C++ code that calls `SolverKernels<...>` entry points directly must bind rates
> as a view.

---

## 1. Bind rate constants as a memory-space view

Every entry point now takes `Kokkos::View<const double*, memory_space>` for the
runtime rate array instead of a raw `const double*`. A raw host pointer cannot
be dereferenced inside a device kernel, so the view carries the memory space
that makes the call chain device-portable.

```cpp
#include "chapman.hpp"

using Device = Kokkos::DefaultExecutionSpace;
using Solver = mkpp::generated::chapman::SolverKernels<Device>;
using memory_space = typename Device::memory_space;

// Photolysis rates produced by a Cloud-J style driver.
std::vector<double> photo = {2.0e-5, 1.0e-3};

// Bind into a view in the kernel's memory space. On a host build this is a
// host-space view; on a CUDA/HIP build copy the rates to the device first.
Kokkos::View<double*, memory_space> jvals_scratch("jvals", photo.size());
auto host_jvals = Kokkos::create_mirror_view(jvals_scratch);
for (std::size_t i = 0; i < photo.size(); ++i) host_jvals(static_cast<int>(i)) = photo[i];
Kokkos::deep_copy(jvals_scratch, host_jvals);
Kokkos::View<const double*, memory_space> jvals(jvals_scratch);

Kokkos::View<double*, Kokkos::LayoutRight, memory_space> state("state", Solver::NUM_SPECIES);
// ... fill state ...

Solver solver;
solver.integrate(3600.0, state, jvals);   // three-argument mechanisms
Kokkos::fence();
```

For a mechanism whose rate law is gated on environment drivers (for example
GoCart), append them as trailing scalars:

```cpp
solver.integrate(dt, state, jvals, /*temp=*/298.15, /*rh=*/0.5, /*clw=*/0.0);
```

If you already hold a contiguous rate buffer in the correct memory space, wrap
it without copying:

```cpp
Kokkos::View<const double*, memory_space> jvals(rate_ptr, num_photolysis);
```

---

## 2. Mechanisms with no external rates

A mechanism whose rate law needs no runtime rate input (for example `carbon`)
never reads the array in its generated body, so you can pass an empty view and
the call is safe:

```cpp
Kokkos::View<const double*, memory_space> empty_jvals;   // extent 0
solver.integrate(dt, state, empty_jvals);
```

For mechanisms that *do* read rates, always bind a view with at least
`NUM_PHOTOLYSIS` entries — a device kernel cannot recover a missing pointer. If
you drive the solver through the host dispatcher (`execute_mechanism_*`) rather
than calling `SolverKernels` directly, an empty view is substituted with a
shared zero-filled array at the host launch boundary, so no device kernel ever
dereferences a dangling pointer.

---

## 3. Verify device callability on your build

The generated path is exercised by two always-built checks plus one
device-conditional gate:

```bash
# Host reference parity (runs everywhere): proves host-compiled + host-executed.
ctest --test-dir build -R DeviceRuntimeParity --output-on-failure

# Full build including the compile-for-device gate when a device backend exists.
cmake --preset device && cmake --build --preset device
ctest --test-dir build-device -R 'DeviceCompileGate|DeviceRuntimeParity' --output-on-failure
```

Read the state line the parity binary prints to know exactly what was validated:

```text
MKPP_VALIDATION_STATES host-compiled host-executed device-unverified
```

| Token              | Meaning                                                          |
| ------------------ | ---------------------------------------------------------------- |
| `host-compiled`    | Generated headers and host kernels compile on the host backend.  |
| `host-executed`    | Host reference integration runs and matches the anchor.          |
| `device-compiled`  | The device instantiation of the full call chain compiles.        |
| `device-executed`  | The path runs on an accelerator and matches host results.        |
| `device-unverified`| No accelerator available, so device execution was not performed. |

`device-unverified` is a non-pass result: host success alone never certifies
device callability. See the [AOT Solver API Reference](../reference/aot-solver-api.md)
for the full signature set and the [Host Model Integration guide](host_model_integration.md)
for the C and Fortran boundary that remains unchanged.

---

## 4. Adjoint / TLM on the device path

The discrete adjoint and tangent-linear integrators use the same
memory-space-typed `jvals` view. The checkpoint buffer is a fixed-array struct
captured by value, so it works inside a device kernel. For cloud-gated
mechanisms, pass the **same** environment drivers to the forward checkpoint and
to the backward adjoint / TLM passes so the recomputed rate law matches:

```cpp
Solver::CheckpointBuffer chk;
int nsteps = solver.integrate_fwd_checkpoint(dt, state, jvals, chk, temp, rh, clw);
solver.integrate_adj(dt, state_final, lambda, jvals, chk, temp, rh, clw);
```

Enable the adjoint API at build time with `-DMKPP_ENABLE_ADJOINT=ON`.
