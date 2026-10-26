/**
 * @file test_batched_production.cpp
 * @brief US3 (T026) contract test: the team+vector batched dense solve is wired into the
 *        production dispatch (MechanismRegistry) and reproduces a reference solution.
 *
 * The batched systems are built from a *generated* mechanism Jacobian (production data, via
 * SolverKernels::compute_jacobian), shifted to a diagonally-dominant iteration matrix
 * W = shift*I - J. A known exact solution x_exact is planted, the right-hand side is formed as
 * r = W*x_exact, and the production entry point MechanismRegistry::batched_dense_solve is used to
 * recover x. This proves the batched path is a real, wired-in capability (FR-008) rather than a
 * helper reachable only from unit tests, and that it matches the reference within tolerance.
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include "chapman.hpp"
#include "mkpp_host/mechanism_manager.hpp"

namespace {

constexpr int kCells = 16;
constexpr int kSpecies = 4;
constexpr double kShift = 1.0e4;

TEST(BatchedProductionDispatch, RecoversPlantedSolutionViaRegistry) {
    using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
    using Solver = mkpp::generated::chapman::SolverKernels<HostExecSpace>;
    using Species = mkpp::generated::chapman::Species;

    // --- 1. Form a real Jacobian batch from the generated mechanism (production data) ---
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> state("state", kCells, kSpecies);
    Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> jac("jac", kCells, kSpecies, kSpecies);
    Kokkos::View<double*, Kokkos::HostSpace> jvals("jvals", 2);

    for (int c = 0; c < kCells; ++c) {
        state(c, Species::O) = 1.0e8 * (c + 1);
        state(c, Species::O2) = 5.0e17 * (c + 1);
        state(c, Species::O3) = 1.0e12 * (c + 1);
        state(c, Species::M) = 2.45e19 * (c + 1);
    }
    jvals(0) = 1.0e-5;
    jvals(1) = 1.0e-3;
    Kokkos::deep_copy(jac, 0.0);

    Solver solver;
    for (int c = 0; c < kCells; ++c) {
        auto state_row = Kokkos::subview(state, c, Kokkos::ALL());
        auto jac_block = Kokkos::subview(jac, c, Kokkos::ALL(), Kokkos::ALL());
        solver.compute_jacobian(state_row, jac_block, jvals);
    }

    // --- 2. Build W = shift*I - J (device views for the team+vector solve) and plant x_exact ---
    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", kCells, kSpecies, kSpecies);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", kCells, kSpecies);
    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);

    for (int c = 0; c < kCells; ++c) {
        double x_exact[kSpecies];
        for (int i = 0; i < kSpecies; ++i) {
            x_exact[i] = static_cast<double>(i + 1) + c;  // planted, known solution
            for (int j = 0; j < kSpecies; ++j) {
                h_A(c, i, j) = (i == j) ? (kShift - jac(c, i, j)) : (-jac(c, i, j));
            }
        }
        // r = W * x_exact (host-side, so the exact solution is recoverable by construction)
        for (int i = 0; i < kSpecies; ++i) {
            double sum = 0.0;
            for (int j = 0; j < kSpecies; ++j) {
                sum += h_A(c, i, j) * x_exact[j];
            }
            h_b(c, i) = sum;
        }
    }
    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);

    // --- 3. Solve through the PRODUCTION dispatch entry point (team + vector layers) ---
    mkpp::host::BatchErrorStatus<> error_status(kCells);
    mkpp::host::MechanismRegistry::batched_dense_solve(A, b, error_status, /*team_size=*/4);

    // --- 4. Verify: recovered x matches the planted exact solution, all cells succeeded ---
    Kokkos::deep_copy(h_b, b);
    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);

    for (int c = 0; c < kCells; ++c) {
        EXPECT_EQ(h_err(c), static_cast<int>(mkpp::host::SUCCESS)) << "cell " << c;
        for (int i = 0; i < kSpecies; ++i) {
            const double expected = static_cast<double>(i + 1) + c;
            EXPECT_NEAR(h_b(c, i), expected, 1.0e-6 * (1.0 + expected))
                << "cell " << c << " species " << i;
        }
    }
}

TEST(BatchedProductionDispatch, EmptyBatchIsNoOp) {
    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", 0, kSpecies, kSpecies);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", 0, kSpecies);
    mkpp::host::BatchErrorStatus<> error_status(0);
    EXPECT_NO_THROW(mkpp::host::MechanismRegistry::batched_dense_solve(A, b, error_status));
}

TEST(BatchedProductionDispatch, SingularCellReportsError) {
    const int cells = 2;
    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", cells, kSpecies, kSpecies);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", cells, kSpecies);
    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);

    for (int c = 0; c < cells; ++c) {
        for (int i = 0; i < kSpecies; ++i) {
            for (int j = 0; j < kSpecies; ++j) {
                h_A(c, i, j) = (i == j) ? 2.0 : 0.1;
            }
            h_b(c, i) = 1.0;
        }
    }
    // Zero out cell 1 entirely → singular (zero diagonal)
    for (int i = 0; i < kSpecies; ++i) {
        for (int j = 0; j < kSpecies; ++j) {
            h_A(1, i, j) = 0.0;
        }
    }
    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);

    mkpp::host::BatchErrorStatus<> error_status(cells);
    mkpp::host::MechanismRegistry::batched_dense_solve(A, b, error_status, /*team_size=*/2);

    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);
    EXPECT_EQ(h_err(0), static_cast<int>(mkpp::host::SUCCESS));
    EXPECT_EQ(h_err(1), static_cast<int>(mkpp::host::ERR_SINGULAR_MATRIX));
}

}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // ScopeGuard honors runtime-tuning args (thread count, device selection) and finalizes
    // exception-safely on scope exit (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
}
