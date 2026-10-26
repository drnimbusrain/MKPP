# Reference: AOT Kokkos ODE Solver C++ API

This reference documents the C++ functions, constants, and layout specifications exposed by MKPP generated AOT Kokkos solver headers (`mkpp-generated/*.hpp`).

---

## 1. Constants and Type Aliases

Each generated header defines namespace-scoped constants describing the mechanism size:

```cpp
namespace mechanism_name {
    // Total number of prognostic chemical species
    constexpr int NUM_SPECIES = ...;

    // Total number of chemical reactions
    constexpr int NUM_REACTIONS = ...;

    // Total non-zero entries in the symbolic Jacobian
    constexpr int NONZERO_JACOBIAN_ENTRIES = ...;
}
```

---

## 2. Primary Function Signatures

All generated solver routines are annotated with `KOKKOS_INLINE_FUNCTION` and
defined inline (in device-callable fragments included by the header), so a
single instantiation serves both host and device compilation. Every entry point
is a member of `SolverKernels<DeviceType>`, which exposes:

```cpp
template <typename DeviceType>
struct SolverKernels {
    using memory_space = typename DeviceType::memory_space;
    // CheckpointBuffer is reachable through the kernel type when the adjoint
    // API is enabled (see section 3).
#ifdef MKPP_ENABLE_ADJOINT
    using CheckpointBuffer = ::mechanism_name::CheckpointBuffer;
#endif
};
```

> **Breaking change — runtime rate binding.** Photolysis and other externally
> supplied rates are now passed as a memory-space-typed view
> `Kokkos::View<const double*, memory_space>` (`jvals_view`) rather than a raw
> host `const double*`. This is what makes the generated path device-callable:
> a raw host pointer cannot be dereferenced inside a device kernel. Host-model
> consumers must bind their rate array into a view in the same memory space as
> the state before calling any entry point. Inside each routine the raw pointer
> is recovered once at the launch boundary (`const double* jvals =
> jvals_view.data();`), so the in-kernel numerics are unchanged.

> **Breaking change — layout-aware buffer access.** State, rate, and Jacobian
> buffers are no longer indexed with a hard-coded flat `arr[i*N + j]` formula.
> Every entry point wraps its caller view in a `detail::StrideView` accessor
> (see [section 8](#8-layout-aware-accessor-contract)) that reproduces the
> view's own strides. A `Kokkos::LayoutLeft` species-major buffer and a
> `Kokkos::LayoutRight` cell-major buffer now map to the same logical elements;
> the former flat-index path silently mis-addressed one of the two.

Mechanisms whose rate law depends on environmental drivers (temperature,
relative humidity, cloud water) take those as trailing scalar arguments; the
rest use the shorter form. The examples below show the base (three-argument)
form; cloud-gated mechanisms append `const double temp, const double rh, const
double clw`.

### `compute_rates`

Evaluates chemical reaction rates from the species state and supplied rates.

```cpp
template <typename StateView, typename RateView>
KOKKOS_INLINE_FUNCTION
void compute_rates(
    const StateView& state,
    RateView& F_block,
    Kokkos::View<const double*, memory_space> jvals_view
);
```

#### Parameters
- **`state`** (`const StateView&`): 1D view or subview of species concentrations $[\text{molec/cm}^3]$, size `NUM_SPECIES`.
- **`F_block`** (`RateView&`): Output 1D view or subview for reaction rates $[\text{molec/cm}^3/\text{s}]$, size `NUM_SPECIES`.
- **`jvals_view`** (`Kokkos::View<const double*, memory_space>`): Runtime photolysis / externally supplied rate constants (from a Cloud-J style driver), bound in the kernel's memory space.

---

### `compute_jacobian`

Evaluates the unrolled symbolic chemical Jacobian entries ($J_{i,j} = \frac{\partial f_i}{\partial y_j}$).

```cpp
template <typename StateView, typename JacView>
KOKKOS_INLINE_FUNCTION
void compute_jacobian(
    const StateView& state,
    JacView& J_block,
    Kokkos::View<const double*, memory_space> jvals_view
);
```

#### Parameters
- **`state`** (`const StateView&`): 1D species concentration view.
- **`J_block`** (`JacView&`): Sparse/unrolled Jacobian representation view or scalar buffer.
- **`jvals_view`**: Rate constants view as above.

---

### `lu_decompose` & `lu_solve`

Performs branchless, zero-loop symbolic sparse LU decomposition and forward/backward substitution.

```cpp
// Decomposes (I - gamma * dt * J) in-place into L and U scalar factors
KOKKOS_INLINE_FUNCTION
void lu_decompose(/* internal scalar references */);

// Solves A * x = b for x in-place
KOKKOS_INLINE_FUNCTION
void lu_solve(/* internal scalar references */);
```

---

### `integrate`

Integrates the chemical ODE system over a step interval using the selected Rosenbrock solver tableau (`ros2`, `ros3`, `ros4`, `rodas3`, `rodas4`).

```cpp
template <typename StateView>
KOKKOS_INLINE_FUNCTION
void integrate(
    double dt_total,
    StateView& state,
    Kokkos::View<const double*, memory_space> jvals_view
);
```

For mechanisms whose rate law depends on environmental drivers, the entry point
appends them after `jvals_view`:

```cpp
void integrate(
    double dt_total,
    StateView& state,
    Kokkos::View<const double*, memory_space> jvals_view,
    const double temp,
    const double rh,
    const double clw
);
```

#### Parameters
- **`dt_total`** (`double`): Integration step interval $[s]$.
- **`state`** (`StateView&`): 1D in-out species state subview (size `NUM_SPECIES`). State array access uses RCM-permuted species ordering `state(perm[i])` for optimal bandwidth locality.
- **`jvals_view`** (`Kokkos::View<const double*, memory_space>`): Runtime photolysis rate constants view (from a Cloud-J style driver), bound in the kernel's memory space.
- **`temp`**, **`rh`**, **`clw`** (`double`): Temperature $[K]$, relative humidity $[0,1]$, cloud water $[\text{g/cm}^3]$ — present only for mechanisms that gate rates on them.

---

### `integrate_with_reduction`

Integrates the chemical ODE system with dynamic auto-reduction, evaluating species importance $I_i = \frac{|F_i|}{\text{atol}_i + \text{rtol}_i |y_i|}$ against `importance_threshold` to freeze inactive species and skip zero-flux operations.

```cpp
template <typename StateView>
KOKKOS_INLINE_FUNCTION
void integrate_with_reduction(
    double dt_total,
    StateView& state,
    Kokkos::View<const double*, memory_space> jvals_view,
    double importance_threshold
);
```

---

## 3. Adjoint & Tangent-Linear Model (TLM) C++ API

When compiled with the `--adjoint` CLI flag and enabled at build time with the
`MKPP_ENABLE_ADJOINT` definition, headers expose discrete adjoint structs and
routines. The checkpoint buffer is a lightweight fixed-array struct (not a set
of Kokkos views), so it can be captured by value inside a device kernel:

```cpp
struct CheckpointBuffer {
    static constexpr int MAX_STEPS = 200;
    static constexpr int NUM_SPECIES = /* mechanism size */;
    int num_steps = 0;
    double h[MAX_STEPS];                    // saved step size per accepted step
    double state[MAX_STEPS][NUM_SPECIES];   // saved concentrations at step entry
};
```

It is also reachable through the kernel type as
`SolverKernels<DeviceType>::CheckpointBuffer` when the adjoint API is enabled.

### `compute_adjoint`

Evaluates the transposed analytical chemical Jacobian matrix $J^T_{i,j} = \frac{\partial f_j}{\partial y_i}$ into `J_adj_block`.

```cpp
template <typename StateView, typename JacView>
KOKKOS_INLINE_FUNCTION
void compute_adjoint(
    const StateView& state,
    JacView& J_adj_block,
    Kokkos::View<const double*, memory_space> jvals_view
);
```

### `compute_tlm`

Evaluates the tangent-linear model rate derivative $dF = J\,\delta C$ for a perturbation `delta_C`.

```cpp
template <typename StateView, typename DeltaView, typename RateView>
KOKKOS_INLINE_FUNCTION
void compute_tlm(
    const StateView& state,
    const DeltaView& delta_C,
    RateView& dF_block,
    Kokkos::View<const double*, memory_space> jvals_view
);
```

### `integrate_fwd_checkpoint`

Executes forward integration and saves trajectory checkpoints into `chk`. Returns the number of accepted steps, or `-1` if `MAX_STEPS` is exceeded.

```cpp
template <typename StateView>
KOKKOS_INLINE_FUNCTION
int integrate_fwd_checkpoint(
    double dt_total,
    StateView& state,
    Kokkos::View<const double*, memory_space> jvals_view,
    CheckpointBuffer& chk
);
```

For environment-gated mechanisms the drivers follow `chk`:

```cpp
int integrate_fwd_checkpoint(
    double dt_total, StateView& state,
    Kokkos::View<const double*, memory_space> jvals_view,
    CheckpointBuffer& chk,
    const double temp, const double rh, const double clw
);
```

### `integrate_adj`

Performs discrete adjoint backward integration over saved trajectory checkpoints, updating the sensitivity vector `lambda`. The Jacobian is recomputed from each saved state (recompute-J strategy); for environment-gated mechanisms the drivers must match those used by the forward checkpoint so the recomputed rate law is identical.

```cpp
template <typename StateView, typename AdjView>
KOKKOS_INLINE_FUNCTION
void integrate_adj(
    double dt_total,
    const StateView& state_final,
    AdjView& lambda,
    Kokkos::View<const double*, memory_space> jvals_view,
    const CheckpointBuffer& chk
);
```

### `integrate_tlm`

Propagates a perturbation `delta_C` forward through the checkpointed trajectory (tangent-linear model).

```cpp
template <typename StateView, typename PertView>
KOKKOS_INLINE_FUNCTION
void integrate_tlm(
    double dt_total,
    const StateView& state_0,
    PertView& delta_C,
    Kokkos::View<const double*, memory_space> jvals_view,
    const CheckpointBuffer& chk
);
```

---

## 4. `mkpp` CLI Compiler Reference

The `mkpp compile` CLI command generates Kokkos ODE solver headers from mechanism specifications.

### Command Usage
```bash
mkpp compile <mechanism.yaml> --test-env <env.yaml> [options]
```

### CLI Arguments & Flags

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `mechanism` | string | *Required* | Path to mechanism YAML specification file |
| `--test-env` | string | *Required* | Path to test environment YAML configuration file |
| `--out` | string | `mkpp-generated/` | Output directory for generated C++ headers and manifests |
| `--solver` | choice | `ros3` | Rosenbrock solver tableau (`ros2`, `ros3`, `ros4`, `rodas3`, `rodas4`) |
| `--adjoint` | flag | `false` | Emit discrete adjoint/TLM routines and `CheckpointBuffer` |
| `--strict` | flag | `false` | Enable strict schema and constraint validation |
| `--emit-manifest` | flag | `true` | Emit metadata manifest JSON alongside C++ headers |
| `--report` | flag | `false` | Generate Markdown analysis report and species interaction graph |
| `--no-cache` | flag | `false` | Skip cache lookup and recompute symbolic matrices from scratch |
| `--dry-run` | flag | `false` | Run parsing and validation stages without emitting code |
| `--verbose` | flag | `false` | Emit progress logs to `stderr` at each compilation pipeline stage |
| `--migrate-equilibrium` | flag | `false` | Rewrite mechanism YAML replacing deprecated `PHASE_CHANGE` blocks with `EQUILIBRIUM` |
| `--convert-kinetic-units-to-si` | flag | `false` | Convert supported rate-law pre-exponential factors from `molecule cm-3` to `mol m-3`; runtime concentrations must then be SI |

### Kinetic concentration units

MKPP preserves mechanism coefficients by default and does not infer or enforce
units. Declare the concentration basis at the mechanism root with
`"concentration_unit": "molecule cm-3"` or `"concentration_unit": "mol m-3"`.
For a reaction with total reactant molecularity $n$, its Arrhenius pre-factor
`A` has units $[C]^{-(n-1)}\,s^{-1}$; first-order and photolysis rates are in
$s^{-1}$. The concentration values passed to a generated solver must use the
mechanism's declared concentration unit.

`--convert-kinetic-units-to-si` converts supported Arrhenius, Troe/falloff,
EP2/EP3, and tunneling pre-exponential factors from `molecule cm-3` to
`mol m-3`. It also converts compile-time air number density. With this flag,
the caller must provide state concentrations, emissions, and any host-supplied
concentration-dependent rates in SI units. Photolysis rates and external
host-forcing values are not converted. The flag requires an explicit
`concentration_unit`; unsupported numeric rate laws fail rather than being
silently treated as converted.

---

## 5. Supported Rosenbrock Solver Tableaus

MKPP code generation supports 5 L-stable and A-stable Rosenbrock solver variants via the `--solver` flag:

| Solver Identifier | Method Name | Stages ($S$) | Order ($ELO$) | Error Exponent ($1/ELO$) | Best Use Case |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `ros2` | ROS-2 | 2 | 2 | $1/2 = 0.500$ | High-throughput regional/global 3D ESM runs |
| `ros3` *(Default)* | ROS-3 | 3 | 3 | $1/3 \approx 0.333$ | Operational baseline (KPP parity) |
| `ros4` | ROS-4 | 4 | 4 | $1/4 = 0.250$ | High-accuracy research simulations |
| `rodas3` | RODAS-3 | 4 | 3 | $1/3 \approx 0.333$ | Stiff mechanisms with stiff mass balance |
| `rodas4` | RODAS-4 | 6 | 4 | $1/4 = 0.250$ | Extreme wildfire/pollution plume spikes |

---

## 6. High-Level Multi-Language Host Model API (`mkpp.h`, `mkpp_mod.f90`, `mkpp.hpp`)

When generating solvers with the `--host-api` flag (`python -m mkpp generate --host-api`), MKPP emits C11, C++17, and Fortran 2018 wrapper interface files alongside Kokkos solver headers, enabling zero-copy host model integration without Python dependencies at compile or run time.

### C11 ABI Interface (`mkpp.h`)
Opaque handle C ABI functions returning integer status codes (`0` for success `MKPP_SUCCESS`, `<0` for errors):

```c
/* Opaque Context Handle */
typedef struct mkpp_handle_s* mkpp_handle_t;

/* Lifecycle & State Binding */
int mkpp_create_handle(mkpp_handle_t* handle_out, size_t num_cells);
int mkpp_destroy_handle(mkpp_handle_t handle);
int mkpp_initialize(mkpp_handle_t handle);

int mkpp_set_state_ptrs(mkpp_handle_t handle, double* species_conc, const double* temp, const double* pressure, const double* air_density);
int mkpp_set_photolysis_ptrs(mkpp_handle_t handle, const double* photolysis_rates);

/* Solver Computation & Diagnostics */
int mkpp_integrate(mkpp_handle_t handle, double dt);
int mkpp_evaluate_rates(mkpp_handle_t handle, double* rates_out);
int mkpp_evaluate_jacobian(mkpp_handle_t handle, double* jac_out);

/* Metadata Lookups & Error Handling */
int mkpp_get_species_index(mkpp_handle_t handle, const char* species_name, int* index_out);
int mkpp_get_photolysis_index(mkpp_handle_t handle, const char* photo_name, int* index_out);
int mkpp_get_last_error(mkpp_handle_t handle, char* buffer, size_t buffer_len);
```

### Fortran 2018 Module (`mkpp_mod.f90`)
Modern Fortran module using `iso_c_binding` to bind 2D column-major arrays `(num_cells, num_species)` zero-copy:

```fortran
use iso_c_binding
use mkpp_mod

type(mkpp_handle_t) :: handle
real(c_double), target :: conc(num_cells, MKPP_NUM_SPECIES)
real(c_double), target :: temp(num_cells), pres(num_cells), rho(num_cells)
integer :: status

call mkpp_create_handle(handle, num_cells, status)
call mkpp_initialize(handle, status)
call mkpp_set_state_ptrs(handle, conc, temp, pres, rho, status)
call mkpp_integrate(handle, 60.0d0, status)
call mkpp_destroy_handle(handle, status)
```

### C++ Interface (`mkpp.hpp`)
RAII `mkpp::MechanismContext` wrapper class managing handle lifecycles and throwing `std::runtime_error` on failure.

---

## 7. CF Metadata Standard Table

Variables and species states in MKPP generated solvers conform to standard Climate and Forecast (CF) metadata conventions:

| Variable | Standard Name | Long Name | Units | Memory Layout |
| :--- | :--- | :--- | :--- | :--- |
| `state` | `mole_fraction_of_chemical_species_in_air` | Chemical Species Concentration Vector | $\text{molec/cm}^3$ | Layout-agnostic via `StrideView` (§8) |
| `temp` | `air_temperature` | Ambient Air Temperature | $K$ | Scalar / 1D View |
| `press` | `air_pressure` | Ambient Air Pressure | $Pa$ | Scalar / 1D View |
| `cair` | `molar_concentration_of_air` | Total Air Molar Density | $\text{molec/cm}^3$ | Scalar / 1D View |
| `reaction_rates` | `chemical_reaction_rate` | Chemical Reaction Fluxes | $\text{molec/cm}^3/\text{s}$ | 1D View |

---

## 8. Layout-Aware Accessor Contract

Every generated mechanism emits a per-mechanism header
`mkpp-generated/<mechanism>/accessor.hpp` in the namespace
`mkpp::generated::<mechanism>::detail`. It defines a non-owning,
device-callable `StrideView<T, Rank>` (rank 1 or 2) and two `make_stride_view`
builders. All generated kernel code — `compute_rates`, `compute_jacobian`,
`integrate`, and the adjoint/TLM entry points — routes state, rate, and
Jacobian buffer access through this accessor instead of a flat
`arr[i*NUM_SPECIES + j]` formula.

### Why it exists

The C-ABI and Fortran host bindings pass species-major `LayoutLeft` buffers
(stride of the species dimension = `num_cells`), while the E2E Kokkos tests
historically used cell-major `LayoutRight` scratch. A kernel that hard-codes
`row*NUM_SPECIES + col` is correct for exactly one of the two layouts and
silently mis-addresses the other (NaN, wrong Jacobian, or a non-converging
linear solve). The accessor removes that coupling: element `(i, j)` resolves to
`data[i*stride(0) + j*stride(1)]`, i.e. the view's own mapping, whatever layout
the caller bound.

### API

```cpp
// From a caller Kokkos::View (rank 1 or 2). Layout must be stride-describable:
// LayoutLeft, LayoutRight, or LayoutStride. Anything else fails at compile time
// (static_assert) rather than mis-indexing at runtime.
const auto state_acc = detail::make_stride_view(state);   // StrideView<const double, 1>
auto       jac_acc   = detail::make_stride_view(J_block); // StrideView<double, 2>

// From a contiguous in-kernel scratch array with known strides. The integrator's
// local J_values/F_values are row-major by construction, so this reproduces the
// former flat writes exactly while keeping a single indexing pathway.
auto scratch_acc = detail::make_stride_view<double, 2>(J_values, NUM_SPECIES, 1);

// Element access mirrors the view:
state_acc(i);              // rank 1
jac_acc(row, col);         // rank 2
```

`StrideView` is trivially copyable, carries a mutable→const relaxing
converting constructor (never the reverse), and is annotated
`KOKKOS_INLINE_FUNCTION`, so it is usable inside device kernels with zero
overhead over the previous flat indexing.

**Invariant:** a generated kernel must never reintroduce a raw
`[i*NUM_SPECIES + j]` index on a caller-supplied buffer. This is enforced by
`tests/contract/test_c_abi_layout.cpp` (LayoutLeft vs LayoutRight produce
identical results) and `tests/integration/e2e_validation/test_accessor_compile.cpp`.

---

## 9. Device-Callability Status Vocabulary

MKPP and its physics submodules annotate code with `KOKKOS_INLINE_FUNCTION`.
To keep every "device" claim honest and machine-checkable (no aspirational
symbol may exist without a device definition or an enforced host-only status),
each **call chain** carries one of four statuses, recorded in
`specs/024-kokkos-compliance-hardening/contracts/device-callability-status.md`
and checked by `tests/python/test_device_honesty.py`:

| Status | Meaning | Evidence required |
| :--- | :--- | :--- |
| `device-verified` | Chain instantiates and compiles under a device execution space (compile gate). | Compile-gate test id. |
| `device-unverified` | Device-capable code is present, but **no device backend is configured in CI** (this build is `OPENMP;SERIAL`). Never conflated with a host pass. | CI reports the host result and the missing-device caveat together. |
| `host-confined` | Chain is host-only by a structural limitation (e.g. cross-TU calls, host-only libraries) and cannot be device-annotated without a restructure. | Blocker documented with the restructure prerequisite. |
| `removed` | An aspirational stub was deleted; no entry point exists to misreport. | A genuine implementation returns only with its staged re-add. |

**CI caveat for this repository:** the container build (`cece-dev`) enables only
`OPENMP` and `SERIAL`. No CUDA/HIP backend exists, so **every device result in
feature 024 is reported `device-unverified` by design** — a host pass is
evidence of layout/numerics correctness, never of on-device execution. The
compile-gate test (`tests/integration/device_callability/test_device_compile_gate.cpp`)
distinguishes "compiles for a device space" from "ran on a device".

Rules enforced by the honesty test: `KOKKOS_INLINE_FUNCTION` must never be
`#define`d to `inline`; device-annotated math must use `Kokkos::{fabs,exp,log,
max,min,pow,sqrt,fmax,fmin,cbrt}` not `std::` equivalents; no `extern "C"
*_device` symbol may be declared without an in-tree definition; and no auxiliary
compute entry point may exist that cannot execute.

---

## Related Documents

- [Reaction Types & OpenAtmos YAML Schema Reference](reaction-types-and-yaml-schema.md)
- [How-To: Host Model Integration](../how-to/host_model_integration.md)
- [Reaction Kinetics & Unified Jacobian Explanation](../explanation/unified-jacobian-and-reaction-kinetics.md)
- [How-To: Create Custom Reactions](../how-to/create-custom-reactions.md)
- [AOT Solver Quickstart Tutorial](../tutorials/aot-solver-quickstart.md)
- [Solver Comparison Benchmarks](../tutorials/benchmark-mkpp-vs-micm.md)
- [AOT Symbolic LU Architecture Explanation](../explanation/aot-symbolic-lu-architecture.md)
