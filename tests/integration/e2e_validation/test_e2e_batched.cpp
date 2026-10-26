/**
 * @file test_e2e_batched.cpp
 * @brief E2E integration test suite for batched TeamPolicy execution, trajectory parity, and fault isolation
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <cmath>
#include <limits>
#include <vector>

#include "chapman.hpp"
#include "mkpp_host/dispatcher.hpp"

namespace {

using DeviceType = Kokkos::DefaultExecutionSpace;
using ChapmanSpecies = mkpp::generated::chapman::Species;
template <typename ExecutionSpace>
using ChapmanSolver = mkpp::generated::chapman::SolverKernels<ExecutionSpace>;

// The generated kernels hoist species into scalar registers via linear indexing
// from state.data(), so the species dimension MUST be contiguous.  Use
// LayoutRight (cell-major, species-contiguous) to match the ConcentrationsView
// convention used by the SciPy and per-mechanism E2E tests.  A LayoutLeft view
// strides species by num_cells, so state.data()+k reads other cells' memory and
// the ROS-3 solver never accepts a step (effectively an infinite loop).
using StateView = Kokkos::View<double****, Kokkos::LayoutRight>;

TEST(E2EBatchedTest, DynamicBatchSizesTeamPolicy) {
    std::vector<int> batch_sizes = {64, 128, 256};
    const double dt = 60.0;
    const int steps = 5;

    for (int num_cells : batch_sizes) {
        StateView state("state", num_cells, 4, 1, 1);
        auto h_state = Kokkos::create_mirror_view(state);

        for (int c = 0; c < num_cells; ++c) {
            h_state(c, ChapmanSpecies::O, 0, 0) = 1.0e8;
            h_state(c, ChapmanSpecies::O2, 0, 0) = 5.0e17;
            h_state(c, ChapmanSpecies::O3, 0, 0) = 1.0e12;
            h_state(c, ChapmanSpecies::M, 0, 0) = 2.45e19;
        }
        Kokkos::deep_copy(state, h_state);

        double jvals_data[2] = {1.0e-5, 1.0e-3};
        mkpp::host::JvalsView<Kokkos::HostSpace> jvals(jvals_data, 2);
        mkpp::host::BatchErrorStatus error_status(num_cells);

        mkpp::host::execute_mechanism_steps_batched<ChapmanSolver<DeviceType>>(
            "chapman_batch_test", state, dt, steps, jvals, error_status, 32);

        auto h_err = Kokkos::create_mirror_view(error_status.status);
        Kokkos::deep_copy(h_err, error_status.status);

        for (int c = 0; c < num_cells; ++c) {
            EXPECT_EQ(h_err(c), mkpp::host::SUCCESS) << "Cell " << c << " failed in batch size " << num_cells;
        }
    }
}

TEST(E2EBatchedTest, MultiCellTrajectoryParity) {
    const int num_cells = 128;
    const double dt = 60.0;
    const int steps = 10;
    double jvals_data[2] = {1.0e-5, 1.0e-3};
    mkpp::host::JvalsView<Kokkos::HostSpace> jvals(jvals_data, 2);

    // State array for single-cell serial execution
    StateView state_serial("state_serial", num_cells, 4, 1, 1);
    // State array for batched team execution
    StateView state_batched("state_batched", num_cells, 4, 1, 1);

    auto h_serial = Kokkos::create_mirror_view(state_serial);
    auto h_batched = Kokkos::create_mirror_view(state_batched);

    for (int c = 0; c < num_cells; ++c) {
        double factor = 1.0 + 0.01 * static_cast<double>(c % 10);
        h_serial(c, ChapmanSpecies::O, 0, 0) = 1.0e8 * factor;
        h_serial(c, ChapmanSpecies::O2, 0, 0) = 5.0e17 * factor;
        h_serial(c, ChapmanSpecies::O3, 0, 0) = 1.0e12 * factor;
        h_serial(c, ChapmanSpecies::M, 0, 0) = 2.45e19 * factor;

        h_batched(c, ChapmanSpecies::O, 0, 0) = 1.0e8 * factor;
        h_batched(c, ChapmanSpecies::O2, 0, 0) = 5.0e17 * factor;
        h_batched(c, ChapmanSpecies::O3, 0, 0) = 1.0e12 * factor;
        h_batched(c, ChapmanSpecies::M, 0, 0) = 2.45e19 * factor;
    }

    Kokkos::deep_copy(state_serial, h_serial);
    Kokkos::deep_copy(state_batched, h_batched);

    // 1. Run serial execution
    mkpp::host::execute_mechanism_steps<ChapmanSolver<DeviceType>>(
        "chapman_serial", state_serial, dt, steps, jvals);

    // 2. Run batched team execution
    mkpp::host::BatchErrorStatus error_status(num_cells);
    mkpp::host::execute_mechanism_steps_batched<ChapmanSolver<DeviceType>>(
        "chapman_batched", state_batched, dt, steps, jvals, error_status, 64);

    Kokkos::deep_copy(h_serial, state_serial);
    Kokkos::deep_copy(h_batched, state_batched);

    for (int c = 0; c < num_cells; ++c) {
        for (int sp = 0; sp < 4; ++sp) {
            double val_ser = h_serial(c, sp, 0, 0);
            double val_bat = h_batched(c, sp, 0, 0);
            double abs_err = std::abs(val_ser - val_bat);
            EXPECT_LE(abs_err, 1.0e-10) << "Parity error at cell " << c << " species " << sp;
        }
    }
}

TEST(E2EBatchedTest, FaultIsolationNonFiniteErrorInjection) {
    const int num_cells = 100;
    const double dt = 60.0;
    const int steps = 2;
    double jvals_data[2] = {1.0e-5, 1.0e-3};
    mkpp::host::JvalsView<Kokkos::HostSpace> jvals(jvals_data, 2);

    StateView state("state", num_cells, 4, 1, 1);
    auto h_state = Kokkos::create_mirror_view(state);

    for (int c = 0; c < num_cells; ++c) {
        h_state(c, ChapmanSpecies::O, 0, 0) = 1.0e8;
        h_state(c, ChapmanSpecies::O2, 0, 0) = 5.0e17;
        h_state(c, ChapmanSpecies::O3, 0, 0) = 1.0e12;
        h_state(c, ChapmanSpecies::M, 0, 0) = 2.45e19;
    }

    // Inject non-finite state (NaN) in cell 5
    h_state(5, ChapmanSpecies::O, 0, 0) = std::numeric_limits<double>::quiet_NaN();

    Kokkos::deep_copy(state, h_state);

    mkpp::host::BatchErrorStatus error_status(num_cells);

    mkpp::host::execute_mechanism_steps_batched<ChapmanSolver<DeviceType>>(
        "chapman_fault_test", state, dt, steps, jvals, error_status, 32);

    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);

    for (int c = 0; c < num_cells; ++c) {
        if (c == 5) {
            EXPECT_EQ(h_err(c), mkpp::host::ERR_NON_FINITE_STATE) << "Cell 5 should report ERR_NON_FINITE_STATE";
        } else {
            EXPECT_EQ(h_err(c), mkpp::host::SUCCESS) << "Cell " << c << " should complete successfully";
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
}
