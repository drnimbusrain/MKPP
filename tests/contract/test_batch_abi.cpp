/**
 * @file test_batch_abi.cpp
 * @brief Contract ABI tests for batched dense solver utilities
 */

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "mkpp_host/batched_dense_solver.hpp"
#include "mkpp_host/dispatcher.hpp"

namespace {

TEST(BatchedDenseSolverTest, Solve3x3DenseSystem) {
    const size_t num_cells = 4;
    const size_t N = 3;

    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", num_cells, N, N);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", num_cells, N);
    Kokkos::View<int*> error_status("error_status", num_cells);

    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);
    auto h_err = Kokkos::create_mirror_view(error_status);

    // Setup 3x3 system:
    // [ 3.0, 2.0, -1.0 ] [ x0 ] = [  1.0 ] -> x0 = 1.0
    // [ 2.0, -2.0, 4.0 ] [ x1 ] = [ -2.0 ] -> x1 = -2.0
    // [-1.0,  0.5, -1.0] [ x2 ] = [  0.0 ] -> x2 = -4.0
    // Exact solution x = [1.0, -2.0, -4.0]
    // Verification:
    // 3(1) + 2(-2) - 1(-4) = 3 - 4 + 4 = 3 -> let's set RHS b = [3.0, 10.0, -1.0]...
    // Let's define A and x_exact = [1.0, 2.0, 3.0]
    // A * x = [ 3(1) + 2(2) - 1(3) ] = [ 3 + 4 - 3 ] = [ 4.0 ]
    //         [ 2(1) - 2(2) + 4(3) ] = [ 2 - 4 + 12] = [ 10.0 ]
    //         [-1(1) + 0.5(2)- 1(3)] = [-1 + 1 - 3 ] = [ -3.0 ]

    for (size_t c = 0; c < num_cells; ++c) {
        h_A(c, 0, 0) = 3.0;  h_A(c, 0, 1) = 2.0;  h_A(c, 0, 2) = -1.0;
        h_A(c, 1, 0) = 2.0;  h_A(c, 1, 1) = -2.0; h_A(c, 1, 2) = 4.0;
        h_A(c, 2, 0) = -1.0; h_A(c, 2, 1) = 0.5;  h_A(c, 2, 2) = -1.0;

        h_b(c, 0) = 4.0;
        h_b(c, 1) = 10.0;
        h_b(c, 2) = -3.0;
        h_err(c) = 0;
    }

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);
    Kokkos::deep_copy(error_status, h_err);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

    Kokkos::parallel_for(
        "TestSolve3x3",
        TeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type& member) {
            mkpp::host::batched_dense_solve(member, A, b, error_status);
        });

    Kokkos::deep_copy(h_b, b);
    Kokkos::deep_copy(h_err, error_status);

    for (size_t c = 0; c < num_cells; ++c) {
        EXPECT_EQ(h_err(c), 0);
        EXPECT_NEAR(h_b(c, 0), 1.0, 1.0e-12);
        EXPECT_NEAR(h_b(c, 1), 2.0, 1.0e-12);
        EXPECT_NEAR(h_b(c, 2), 3.0, 1.0e-12);
    }
}

TEST(BatchedDenseSolverTest, Solve5x5DenseSystem) {
    const size_t num_cells = 8;
    const size_t N = 5;

    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", num_cells, N, N);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", num_cells, N);
    mkpp::host::BatchErrorStatus error_status(num_cells);

    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);

    // Diagonal dominant 5x5 system
    for (size_t c = 0; c < num_cells; ++c) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                if (i == j) {
                    h_A(c, i, j) = 10.0 + static_cast<double>(i + 1);
                } else {
                    h_A(c, i, j) = 0.5 * static_cast<double>(i + j);
                }
            }
        }
        // Set b such that x_exact = [1, 1, 1, 1, 1]
        for (size_t i = 0; i < N; ++i) {
            double sum = 0.0;
            for (size_t j = 0; j < N; ++j) {
                sum += h_A(c, i, j);
            }
            h_b(c, i) = sum;
        }
    }

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

    Kokkos::parallel_for(
        "TestSolve5x5",
        TeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type& member) {
            mkpp::host::batched_dense_solve(member, A, b, error_status);
        });

    Kokkos::deep_copy(h_b, b);
    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);

    for (size_t c = 0; c < num_cells; ++c) {
        EXPECT_EQ(h_err(c), 0);
        for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(h_b(c, i), 1.0, 1.0e-12);
        }
    }
}

TEST(BatchedDenseSolverTest, SingularMatrixDetection) {
    const size_t num_cells = 4;
    const size_t N = 3;

    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", num_cells, N, N);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", num_cells, N);
    mkpp::host::BatchErrorStatus error_status(num_cells);

    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);

    for (size_t c = 0; c < num_cells; ++c) {
        h_A(c, 0, 0) = 2.0; h_A(c, 0, 1) = 1.0; h_A(c, 0, 2) = 1.0;
        h_A(c, 1, 0) = 1.0; h_A(c, 1, 1) = 3.0; h_A(c, 1, 2) = 1.0;
        h_A(c, 2, 0) = 1.0; h_A(c, 2, 1) = 1.0; h_A(c, 2, 2) = 4.0;
        h_b(c, 0) = 1.0; h_b(c, 1) = 1.0; h_b(c, 2) = 1.0;
    }

    // Inject singular matrix (all zeros in row 1) for cell 2
    h_A(2, 1, 0) = 0.0; h_A(2, 1, 1) = 0.0; h_A(2, 1, 2) = 0.0;

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

    Kokkos::parallel_for(
        "TestSingularMatrix",
        TeamPolicy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type& member) {
            mkpp::host::batched_dense_solve(member, A, b, error_status);
        });

    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);

    EXPECT_EQ(h_err(0), mkpp::host::SUCCESS);
    EXPECT_EQ(h_err(1), mkpp::host::SUCCESS);
    EXPECT_EQ(h_err(2), mkpp::host::ERR_SINGULAR_MATRIX);
    EXPECT_EQ(h_err(3), mkpp::host::SUCCESS);
}

TEST(BatchedDenseSolverTest, Solve1024CellBatch) {
    const size_t num_cells = 1024;
    const size_t N = 4;

    Kokkos::View<double***, Kokkos::LayoutLeft> A("A", num_cells, N, N);
    Kokkos::View<double**, Kokkos::LayoutLeft> b("b", num_cells, N);
    mkpp::host::BatchErrorStatus error_status(num_cells);

    auto h_A = Kokkos::create_mirror_view(A);
    auto h_b = Kokkos::create_mirror_view(b);

    for (size_t c = 0; c < num_cells; ++c) {
        for (size_t i = 0; i < N; ++i) {
            for (size_t j = 0; j < N; ++j) {
                h_A(c, i, j) = (i == j) ? 5.0 + static_cast<double>(i) : 0.2;
            }
            h_b(c, i) = 5.0 + static_cast<double>(i) + 0.2 * static_cast<double>(N - 1);
        }
    }

    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(b, h_b);

    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

    Kokkos::parallel_for(
        "TestSolve1024Cells",
        TeamPolicy(num_cells / 64, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename TeamPolicy::member_type& member) {
            mkpp::host::batched_dense_solve(member, A, b, error_status);
        });

    Kokkos::deep_copy(h_b, b);
    auto h_err = Kokkos::create_mirror_view(error_status.status);
    Kokkos::deep_copy(h_err, error_status.status);

    for (size_t c = 0; c < num_cells; ++c) {
        EXPECT_EQ(h_err(c), mkpp::host::SUCCESS);
        for (size_t i = 0; i < N; ++i) {
            EXPECT_NEAR(h_b(c, i), 1.0, 1.0e-12);
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
