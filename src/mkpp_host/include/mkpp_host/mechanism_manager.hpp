#pragma once

#include <Kokkos_Core.hpp>
#include "dispatcher.hpp"
#include "mechanisms.hpp"

#ifdef MKPP_HAS_KOKKOS_KERNELS
#include "batched_dense_solver.hpp"
#endif

#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mkpp {
namespace host {

struct MechanismInfo {
    std::string name;
    int num_species{0};
    int num_reactions{0};
    bool requires_env{false};
};

class MechanismRegistry {
public:
    static std::vector<std::string> get_registered_mechanisms() {
        std::vector<std::string> list;
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CHAPMAN)) && __has_include("chapman.hpp")
        list.push_back("chapman");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_GOCART)) && __has_include("gocart.hpp")
        list.push_back("gocart");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99)) && __has_include("saprc99.hpp")
        list.push_back("saprc99");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_TS1)) && __has_include("ts1.hpp")
        list.push_back("ts1");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CARBON)) && __has_include("carbon.hpp")
        list.push_back("carbon");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SMALL_STRATO)) && __has_include("small_strato.hpp")
        list.push_back("small_strato");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRCNOV)) && __has_include("saprcnov.hpp")
        list.push_back("saprcnov");
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99_MINI)) && __has_include("saprc99_mini.hpp")
        list.push_back("saprc99_mini");
#endif
        return list;
    }

    static bool has_mechanism(const std::string& name) {
        auto list = get_registered_mechanisms();
        for (const auto& m : list) {
            if (m == name) return true;
        }
        return false;
    }

    static MechanismInfo get_info(const std::string& name) {
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CHAPMAN)) && __has_include("chapman.hpp")
        if (name == "chapman") {
            return {"chapman", mech_chapman::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_GOCART)) && __has_include("gocart.hpp")
        if (name == "gocart") {
            return {"gocart", mech_gocart::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, true};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99)) && __has_include("saprc99.hpp")
        if (name == "saprc99") {
            return {"saprc99", mech_saprc99::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_TS1)) && __has_include("ts1.hpp")
        if (name == "ts1") {
            return {"ts1", mech_ts1::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CARBON)) && __has_include("carbon.hpp")
        if (name == "carbon") {
            return {"carbon", mech_carbon::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SMALL_STRATO)) && __has_include("small_strato.hpp")
        if (name == "small_strato") {
            return {"small_strato", mech_small_strato::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRCNOV)) && __has_include("saprcnov.hpp")
        if (name == "saprcnov") {
            return {"saprcnov", mech_saprcnov::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99_MINI)) && __has_include("saprc99_mini.hpp")
        if (name == "saprc99_mini") {
            return {"saprc99_mini", mech_saprc99_mini::mkpp::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 0, false};
        }
#endif
        std::string err = "FATAL ERROR: Unknown mechanism '" + name + "'. Supported: ";
        auto list = get_registered_mechanisms();
        for (size_t i = 0; i < list.size(); ++i) {
            err += list[i] + (i + 1 < list.size() ? ", " : "");
        }
        std::cerr << err << std::endl;
        throw std::runtime_error(err);
    }

    template <typename StateView>
    static void execute(const std::string& name, StateView state, const HostExecutionParams& params) {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        const int num_cells = static_cast<int>(state.extent(0));

#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CHAPMAN)) && __has_include("chapman.hpp")
        if (name == "chapman") {
            using SolverType = mech_chapman::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_chapman",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_GOCART)) && __has_include("gocart.hpp")
        if (name == "gocart") {
            using SolverType = mech_gocart::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_gocart",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99)) && __has_include("saprc99.hpp")
        if (name == "saprc99") {
            using SolverType = mech_saprc99::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_saprc99",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_TS1)) && __has_include("ts1.hpp")
        if (name == "ts1") {
            using SolverType = mech_ts1::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_ts1",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_CARBON)) && __has_include("carbon.hpp")
        if (name == "carbon") {
            using SolverType = mech_carbon::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_carbon",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SMALL_STRATO)) && __has_include("small_strato.hpp")
        if (name == "small_strato") {
            using SolverType = mech_small_strato::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_small_strato",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRCNOV)) && __has_include("saprcnov.hpp")
        if (name == "saprcnov") {
            using SolverType = mech_saprcnov::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_saprcnov",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
#if (!defined(MKPP_HOST_SELECTED_MECHANISMS) || defined(MKPP_HOST_MECH_SAPRC99_MINI)) && __has_include("saprc99_mini.hpp")
        if (name == "saprc99_mini") {
            using SolverType = mech_saprc99_mini::mkpp::SolverKernels<ExecSpace>;
            Kokkos::parallel_for("MKPP_Dispatch_saprc99_mini",
                Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
                ParamsTiledCellIntegrator<SolverType, StateView>(state, params));
            Kokkos::fence();
            return;
        }
#endif
        std::string err = "FATAL ERROR: Unknown mechanism '" + name + "'. Supported: ";
        auto list = get_registered_mechanisms();
        for (size_t i = 0; i < list.size(); ++i) {
            err += list[i] + (i + 1 < list.size() ? ", " : "");
        }
        std::cerr << err << std::endl;
        throw std::runtime_error(err);
    }

#ifdef MKPP_HAS_KOKKOS_KERNELS
    /// @brief Production dispatch entry for the team+vector batched dense solve (FR-008).
    ///
    /// Solves `num_cells` independent N x N dense systems in one launch, using
    /// `KokkosBatched::TeamLU` (team layer) and `TeamVectorTrsv` (team+vector layers) so the
    /// vector lanes are engaged instead of pinning one matrix per thread. This is the wired-in
    /// production path for `batched_dense_solve`; the matrices are exactly the `(cells,species,
    /// species)` LayoutLeft Jacobian batch that `mkpp_evaluate_jacobian` produces, so the same
    /// team+vector solve serves the C-ABI and any host caller. Gated behind KokkosKernels: with
    /// the batched backend disabled there is no batched production path to exercise (E6 `enabled`).
    ///
    /// @tparam MatrixView 3D view [num_cells, N, N] in the default execution space.
    /// @tparam VectorView 2D view [num_cells, N] in the default execution space.
    /// @param A Batched system matrices (overwritten in place with LU factors).
    /// @param b Batched right-hand sides (overwritten in place with the solution x).
    /// @param error_status Per-cell status buffer (size num_cells).
    /// @param team_size Cells handled cooperatively per team; clamped to the batch size.
    template <typename MatrixView, typename VectorView>
    static void batched_dense_solve(
        const MatrixView& A,
        const VectorView& b,
        BatchErrorStatus<> error_status,
        int team_size = 64) {
        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

        const int num_cells = static_cast<int>(A.extent(0));
        if (num_cells == 0) {
            return;
        }
        if (static_cast<int>(error_status.status.extent(0)) < num_cells) {
            std::string err = "FATAL ERROR: batched_dense_solve error buffer (" +
                              std::to_string(error_status.status.extent(0)) + ") smaller than batch (" +
                              std::to_string(num_cells) + ").";
            std::cerr << err << std::endl;
            throw std::runtime_error(err);
        }

        const int team_count = (num_cells + team_size - 1) / team_size;
        TeamPolicy policy(team_count, Kokkos::AUTO);

        // Qualify with mkpp::host:: so this resolves to the free solver function rather than
        // recursing into the member function we are defining.
        Kokkos::parallel_for(
            "MKPP_Batched_Dense_Solve_Dispatch",
            policy,
            KOKKOS_LAMBDA(const typename TeamPolicy::member_type& member) {
                mkpp::host::batched_dense_solve(member, A, b, error_status);
            });
        Kokkos::fence();
    }
#endif
};

} // namespace host
} // namespace mkpp
