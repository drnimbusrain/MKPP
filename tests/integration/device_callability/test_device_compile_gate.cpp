// Device compile gate: prove that the generated solver path is callable on a
// real Kokkos device backend.
//
// This target is built ONLY when the linked Kokkos exposes a device execution
// space (CUDA/HIP/SYCL); the CMake gate otherwise skips it with an explicit
// device-unverified status rather than silently passing. The test instantiates
// the full generated call chain — rate, Jacobian, and solver entry points,
// plus the adjoint/TLM and DRG-reduction variants when those APIs are enabled —
// inside functors launched on the device execution space. Launching forces the
// device compiler to codegen every inlined fragment, so a host-only symbol or a
// raw host pointer anywhere in the chain fails the build.
//
// The gate is a compile-and-launch check: it verifies device callability, not
// numerical accuracy (host parity is covered by test_host_parity.cpp).
#include <Kokkos_Core.hpp>

#include <cstdio>
#include <string>

#include <gtest/gtest.h>

#include "chapman.hpp"
#include "gocart.hpp"
#include "saprc99.hpp"
#include "ts1.hpp"

#ifndef MKPP_DEVICE_EXEC_SPACE
#error "MKPP_DEVICE_EXEC_SPACE must be defined by the device gate build"
#endif

namespace {

using DeviceSpace = MKPP_DEVICE_EXEC_SPACE;

// Bind a photolysis rate array in the device memory space. The values are
// irrelevant to the compile gate; only the memory-space-typed view matters.
template <typename MemorySpace>
Kokkos::View<const double*, MemorySpace> make_jvals(std::size_t extent) {
    Kokkos::View<double*, MemorySpace> scratch("jvals_scratch", extent);
    Kokkos::deep_copy(scratch, 1.0e-4);
    return Kokkos::View<const double*, MemorySpace>(scratch);
}

// Dispatch to whichever integrate arity the mechanism exposes: cloud-gated
// mechanisms take temperature/rh/clw, the rest take the three-argument form.
template <typename Solver, typename StateView, typename JvalsView>
KOKKOS_INLINE_FUNCTION void call_integrate(Solver& solver, double dt, StateView& state,
                                           JvalsView jvals) {
    if constexpr (requires { solver.integrate(dt, state, jvals, 300.0, 0.5, 0.0); }) {
        solver.integrate(dt, state, jvals, 300.0, 0.5, 0.0);
    } else {
        solver.integrate(dt, state, jvals);
    }
}

template <typename Solver, typename StateView, typename OutView, typename JvalsView>
KOKKOS_INLINE_FUNCTION void call_rates(Solver& solver, StateView& state, OutView& out,
                                       JvalsView jvals) {
    if constexpr (requires { solver.compute_rates(state, out, jvals, 300.0, 0.5, 0.0); }) {
        solver.compute_rates(state, out, jvals, 300.0, 0.5, 0.0);
    } else {
        solver.compute_rates(state, out, jvals);
    }
}

template <typename Solver, typename StateView, typename JacView, typename JvalsView>
KOKKOS_INLINE_FUNCTION void call_jacobian(Solver& solver, StateView& state, JacView& jac,
                                          JvalsView jvals) {
    if constexpr (requires { solver.compute_jacobian(state, jac, jvals, 300.0, 0.5, 0.0); }) {
        solver.compute_jacobian(state, jac, jvals, 300.0, 0.5, 0.0);
    } else {
        solver.compute_jacobian(state, jac, jvals);
    }
}

template <typename Solver, typename StateView, typename JacView, typename JvalsView>
KOKKOS_INLINE_FUNCTION void call_adjoint(Solver& solver, StateView& state, JacView& jac,
                                         JvalsView jvals) {
    if constexpr (requires { solver.compute_adjoint(state, jac, jvals, 300.0, 0.5, 0.0); }) {
        solver.compute_adjoint(state, jac, jvals, 300.0, 0.5, 0.0);
    } else {
        solver.compute_adjoint(state, jac, jvals);
    }
}

template <typename Solver, typename StateView, typename DeltaView, typename OutView,
          typename JvalsView>
KOKKOS_INLINE_FUNCTION void call_tlm(Solver& solver, StateView& state, DeltaView& delta,
                                     OutView& out, JvalsView jvals) {
    if constexpr (requires { solver.compute_tlm(state, delta, out, jvals, 300.0, 0.5, 0.0); }) {
        solver.compute_tlm(state, delta, out, jvals, 300.0, 0.5, 0.0);
    } else {
        solver.compute_tlm(state, delta, out, jvals);
    }
}

template <typename Solver, typename StateView, typename JvalsView, typename Chk>
KOKKOS_INLINE_FUNCTION void call_checkpoint(Solver& solver, double dt, StateView& state,
                                            JvalsView jvals, Chk& chk) {
    if constexpr (requires { solver.integrate_fwd_checkpoint(dt, state, jvals, chk, 300.0, 0.5, 0.0); }) {
        solver.integrate_fwd_checkpoint(dt, state, jvals, chk, 300.0, 0.5, 0.0);
    } else {
        solver.integrate_fwd_checkpoint(dt, state, jvals, chk);
    }
}

template <typename Solver, typename StateView, typename AdjView, typename JvalsView, typename Chk>
KOKKOS_INLINE_FUNCTION void call_adj(Solver& solver, double dt, StateView& state, AdjView& lambda,
                                     JvalsView jvals, const Chk& chk) {
    if constexpr (requires { solver.integrate_adj(dt, state, lambda, jvals, chk, 300.0, 0.5, 0.0); }) {
        solver.integrate_adj(dt, state, lambda, jvals, chk, 300.0, 0.5, 0.0);
    } else {
        solver.integrate_adj(dt, state, lambda, jvals, chk);
    }
}

template <typename Solver, typename StateView, typename PertView, typename JvalsView, typename Chk>
KOKKOS_INLINE_FUNCTION void call_tlm_int(Solver& solver, double dt, StateView& state,
                                         PertView& delta, JvalsView jvals, const Chk& chk) {
    if constexpr (requires { solver.integrate_tlm(dt, state, delta, jvals, chk, 300.0, 0.5, 0.0); }) {
        solver.integrate_tlm(dt, state, delta, jvals, chk, 300.0, 0.5, 0.0);
    } else {
        solver.integrate_tlm(dt, state, delta, jvals, chk);
    }
}

// Exercise the whole generated call chain in device code. Every branch pulls in
// an inlined fragment: rates, Jacobian, and the solver. Adjoint/TLM and the
// DRG-reduction integrator are exercised when their APIs are enabled.
template <typename SolverKernels>
struct DeviceChainFunctor {
    using memory_space = typename SolverKernels::memory_space;
    static constexpr int kSpecies = SolverKernels::NUM_SPECIES;
    static constexpr int kPhotolysis = SolverKernels::NUM_PHOTOLYSIS;
    Kokkos::View<const double*, memory_space> m_jvals;

    KOKKOS_INLINE_FUNCTION
    void operator()(const int) const {
        SolverKernels solver;
        Kokkos::View<double*, Kokkos::LayoutRight, memory_space> state("state", kSpecies);
        Kokkos::View<double*, Kokkos::LayoutRight, memory_space> rates("rates", kSpecies);
        Kokkos::View<double**, Kokkos::LayoutRight, memory_space> jac("jac", kSpecies, kSpecies);
        Kokkos::deep_copy(state, 1.0e10);

        // Core forward chain.
        call_rates(solver, state, rates, m_jvals);
        call_jacobian(solver, state, jac, m_jvals);
        call_integrate(solver, 60.0, state, m_jvals);

#ifdef MKPP_ENABLE_ADJOINT
        // Adjoint / tangent-linear chain (recompute-J checkpoint strategy).
        Kokkos::View<double**, Kokkos::LayoutRight, memory_space> adj("adj", kSpecies, kSpecies);
        call_adjoint(solver, state, adj, m_jvals);

        Kokkos::View<double*, Kokkos::LayoutRight, memory_space> delta("delta", kSpecies);
        Kokkos::View<double*, Kokkos::LayoutRight, memory_space> dF("dF", kSpecies);
        call_tlm(solver, state, delta, dF, m_jvals);

        typename SolverKernels::CheckpointBuffer chk;
        auto state_copy = state;
        call_checkpoint(solver, 60.0, state_copy, m_jvals, chk);

        Kokkos::View<double*, Kokkos::LayoutRight, memory_space> lambda("lambda", kSpecies);
        Kokkos::deep_copy(lambda, 1.0);
        call_adj(solver, 60.0, state, lambda, m_jvals, chk);

        auto delta_copy = delta;
        call_tlm_int(solver, 60.0, state, delta_copy, m_jvals, chk);
#endif

#ifdef MKPP_ENABLE_REDUCTION
        // Directed Relationship Graph reduction integrator.
        auto reduced_state = state;
        if constexpr (requires {
                         solver.integrate_with_reduction(60.0, reduced_state, m_jvals, 1.0e-6,
                                                         300.0, 0.5, 0.0);
                     }) {
            solver.integrate_with_reduction(60.0, reduced_state, m_jvals, 1.0e-6, 300.0, 0.5, 0.0);
        } else {
            solver.integrate_with_reduction(60.0, reduced_state, m_jvals, 1.0e-6);
        }
#endif
    }
};

template <typename SolverKernels>
void run_gate(const char* label) {
    using memory_space = typename SolverKernels::memory_space;
    constexpr int kCells = 8;
    auto jvals = make_jvals<memory_space>(SolverKernels::NUM_PHOTOLYSIS);
    Kokkos::parallel_for(
        std::string("MKPP_DeviceGate_") + label,
        Kokkos::RangePolicy<DeviceSpace>(0, kCells),
        DeviceChainFunctor<SolverKernels>{jvals});
    Kokkos::fence();
}

}  // namespace

TEST(DeviceCompileGate, Chapman) {
    run_gate<mkpp::generated::chapman::SolverKernels<DeviceSpace>>("chapman");
}

TEST(DeviceCompileGate, Gocart) {
    run_gate<mkpp::generated::gocart::SolverKernels<DeviceSpace>>("gocart");
}

TEST(DeviceCompileGate, Saprc99) {
    run_gate<mkpp::generated::saprc99::SolverKernels<DeviceSpace>>("saprc99");
}

TEST(DeviceCompileGate, Ts1) {
    run_gate<mkpp::generated::ts1::SolverKernels<DeviceSpace>>("ts1");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Exception-safe runtime ownership; guard finalizes when main returns,
    // after the state tokens below are emitted (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    const int result = RUN_ALL_TESTS();
    // This binary is compiled only when a device backend is linked, so reaching
    // main means the generated call chain compiled for the device. A clean run
    // additionally means it launched on the device. Emitting the states here
    // keeps the vocabulary consistent with the runtime parity test.
    if (result == 0) {
        std::printf("MKPP_VALIDATION_STATES device-compiled device-executed\n");
        std::fflush(stdout);
    }
    return result;
}
