#pragma once

#include <Kokkos_Core.hpp>
#include "dispatcher.hpp"

#ifdef MKPP_HAS_KOKKOS_KERNELS
#include <KokkosBatched_LU_Decl.hpp>
#include <KokkosBatched_Trsv_Decl.hpp>
#endif

namespace mkpp {
namespace host {

template <typename ErrorView>
KOKKOS_INLINE_FUNCTION void set_cell_error(const ErrorView& err, size_t cell_idx, CellErrorCode code) {
    if constexpr (requires { err.set_error(cell_idx, code); }) {
        err.set_error(cell_idx, code);
    } else {
        err(cell_idx) = static_cast<int>(code);
    }
}

/**
 * @brief Executes batched LU factorization and triangular solve for B independent N x N dense matrices.
 *
 * When KokkosKernels is enabled, each team cooperatively factors and solves one matrix at a
 * time using the full parallel hierarchy: `KokkosBatched::TeamLU` distributes the elimination
 * across the team threads, and `KokkosBatched::TeamVectorTrsv` distributes each triangular
 * substitution across both the team and vector layers. This replaces the previous
 * one-thread-per-matrix `SerialLU`/`SerialTrsv` variants, which left the vector lanes idle
 * (FR-008). Without KokkosKernels the routine falls back to an explicit unblocked LU plus
 * forward/backward substitution on a single thread per cell.
 *
 * @tparam TeamMember Kokkos TeamPolicy member type or execution handle
 * @tparam MatrixView 3D Kokkos View with shape [B, N, N] and LayoutLeft
 * @tparam VectorView 2D Kokkos View with shape [B, N] and LayoutLeft
 * @tparam ErrorView 1D Kokkos View with shape [B] storing integer error codes or BatchErrorStatus
 *
 * @param member Kokkos team member execution handle
 * @param A Batched system matrix [B, N, N] (overwritten with LU factors)
 * @param b Right-hand-side vector [B, N] (overwritten with solution x)
 * @param error_status Per-cell status flags [B]
 */
template <typename TeamMember, typename MatrixView, typename VectorView, typename ErrorView>
KOKKOS_INLINE_FUNCTION void batched_dense_solve(
    const TeamMember& member,
    const MatrixView& A,
    const VectorView& b,
    const ErrorView& error_status) {
    const int num_cells = static_cast<int>(A.extent(0));
    const int N = static_cast<int>(A.extent(1));

    const int team_rank = member.league_rank();
    const int num_teams = member.league_size();
    const int cells_per_team = (num_cells + num_teams - 1) / num_teams;
    const int begin_cell = team_rank * cells_per_team;
    const int end_cell = (begin_cell + cells_per_team < num_cells) ? (begin_cell + cells_per_team) : num_cells;

#ifdef MKPP_HAS_KOKKOS_KERNELS
    // Team-cooperative path: every team thread participates in each factorization/solve, so the
    // cell loop is a plain sequential loop (NOT a TeamThreadRange) and the batched kernels are
    // invoked at team scope. A zero pivot is detected before factorization because the unpivoted
    // TeamLU does not report mid-elimination singularity; the check is uniform across the team
    // (A is read-only here), so all threads skip the same cell together.
    for (int cell_idx = begin_cell; cell_idx < end_cell; ++cell_idx) {
        auto A_sub = Kokkos::subview(A, cell_idx, Kokkos::ALL(), Kokkos::ALL());
        auto b_sub = Kokkos::subview(b, cell_idx, Kokkos::ALL());

        bool is_singular = false;
        for (int k = 0; k < N; ++k) {
            if (Kokkos::abs(A_sub(k, k)) < 1.0e-15) {
                is_singular = true;
                break;
            }
        }
        if (is_singular) {
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
                set_cell_error(error_status, cell_idx, ERR_SINGULAR_MATRIX);
            });
            continue;
        }

        // Team-layer LU factorization (unpivoted, in place: A = L * U).
        const int lu_err = KokkosBatched::TeamLU<TeamMember, KokkosBatched::Algo::LU::Unblocked>::invoke(member, A_sub);
        if (lu_err != 0) {
            Kokkos::single(Kokkos::PerTeam(member), [&]() {
                set_cell_error(error_status, cell_idx, ERR_SINGULAR_MATRIX);
            });
            continue;
        }

        // Team+vector forward substitution (L * y = b, unit diagonal).
        KokkosBatched::TeamVectorTrsv<
            TeamMember,
            KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Trsv::Unblocked>::invoke(member, 1.0, A_sub, b_sub);

        // Team+vector back substitution (U * x = y, non-unit diagonal).
        KokkosBatched::TeamVectorTrsv<
            TeamMember,
            KokkosBatched::Uplo::Upper,
            KokkosBatched::Trans::NoTranspose,
            KokkosBatched::Diag::NonUnit,
            KokkosBatched::Algo::Trsv::Unblocked>::invoke(member, 1.0, A_sub, b_sub);
    }
#else
    // Fallback without KokkosKernels: one thread per cell, explicit unblocked LU + substitution.
    Kokkos::parallel_for(Kokkos::TeamThreadRange(member, begin_cell, end_cell), [=](const int cell_idx) {
        auto A_sub = Kokkos::subview(A, cell_idx, Kokkos::ALL(), Kokkos::ALL());
        auto b_sub = Kokkos::subview(b, cell_idx, Kokkos::ALL());

        // Unblocked LU Factorization in-place (A = L * U)
        bool is_singular = false;
        for (int k = 0; k < N; ++k) {
            double pivot = A_sub(k, k);
            if (Kokkos::abs(pivot) < 1.0e-15) {
                is_singular = true;
                break;
            }
            for (int i = k + 1; i < N; ++i) {
                A_sub(i, k) /= pivot;
                double mult = A_sub(i, k);
                for (int j = k + 1; j < N; ++j) {
                    A_sub(i, j) -= mult * A_sub(k, j);
                }
            }
        }

        if (is_singular) {
            set_cell_error(error_status, cell_idx, ERR_SINGULAR_MATRIX);
            return;
        }

        // Forward Substitution (L * y = b)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < i; ++j) {
                b_sub(i) -= A_sub(i, j) * b_sub(j);
            }
        }

        // Backward Substitution (U * x = y)
        for (int i = N - 1; i >= 0; --i) {
            for (int j = i + 1; j < N; ++j) {
                b_sub(i) -= A_sub(i, j) * b_sub(j);
            }
            double diag = A_sub(i, i);
            if (Kokkos::abs(diag) < 1.0e-15) {
                set_cell_error(error_status, cell_idx, ERR_SINGULAR_MATRIX);
                return;
            }
            b_sub(i) /= diag;
        }
    });
#endif
}

}  // namespace host
}  // namespace mkpp
