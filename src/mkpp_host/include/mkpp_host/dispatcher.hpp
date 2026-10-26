#pragma once
#include <Kokkos_Core.hpp>

#if KOKKOS_VERSION < 50101
#error "MKPP requires Kokkos 5.1.1 or newer"
#endif

#include <iostream>
#include <string>
#include <stdexcept>

namespace mkpp {
namespace host {

enum CellErrorCode : int {
    SUCCESS = 0,
    ERR_SINGULAR_MATRIX = 1,
    ERR_NON_FINITE_STATE = 2,
    ERR_CONVERGENCE_FAILURE = 3,
    ERR_NUMERICAL_FAILURE = 4
};

/// @brief Photolysis rate array bound in the memory space of the state it drives.
///
/// A raw host pointer cannot be dereferenced inside a device functor, so runtime
/// array inputs travel as a Kokkos view in the consuming kernel's memory space.
/// An empty view means "no photolysis supplied" and is normalised to a shared
/// zero-filled array at the host launch boundary (never inside a kernel).
template <typename MemorySpace>
using JvalsView = Kokkos::View<const double*, MemorySpace>;

namespace detail {

/// @brief Cached zero-filled photolysis array used when a caller supplies none.
/// @note Materialised once per memory space at the host launch boundary; device
///       code never allocates. The array is created at a fixed capacity and is
///       never resized, so a view handed out here can never be invalidated by
///       a later, larger request — the previous grow-in-place pattern was a
///       data race between concurrent host callers. The holder is intentionally
///       heap-allocated and never destroyed: a function-local static Kokkos
///       view would deallocate after Kokkos::finalize() during program exit,
///       which Kokkos treats as an error. The process reclaims the single
///       bounded buffer at exit.
template <typename MemorySpace>
inline JvalsView<MemorySpace> zero_jvals(std::size_t extent) {
    using MutableView = Kokkos::View<double*, MemorySpace>;
    // Sized above every mechanism's reaction-slot count (TS1 uses 547) so the
    // shared array covers all in-tree callers without growth.
    constexpr std::size_t kCapacity = 1024;
    static const MutableView& shared = *new MutableView([] {
        MutableView zeros("MKPP_ZeroJvals", kCapacity);
        Kokkos::deep_copy(zeros, 0.0);
        return zeros;
    }());
    if (extent <= shared.extent(0)) {
        return JvalsView<MemorySpace>(shared.data(), extent);
    }
    // Oversized request: hand back a dedicated, reference-counted zero array.
    // Returning the managed view itself (converted to the const view type)
    // keeps it alive for the caller, so nothing can invalidate it later.
    MutableView oversized("MKPP_ZeroJvals_oversized", extent);
    Kokkos::deep_copy(oversized, 0.0);
    return JvalsView<MemorySpace>(oversized);
}

/// @brief Replace an empty photolysis view with the shared zero-filled array.
template <typename MemorySpace>
inline JvalsView<MemorySpace> resolve_jvals(const JvalsView<MemorySpace>& jvals,
                                            std::size_t extent) {
    if (jvals.data() == nullptr || jvals.extent(0) == 0) {
        return zero_jvals<MemorySpace>(extent);
    }
    return jvals;
}

}  // namespace detail

/// @brief Runtime execution parameters for host integration dispatch.
template <typename MemorySpace>
struct BasicHostExecutionParams {
    using memory_space = MemorySpace;

    double dt{60.0};                        ///< Integration timestep size in seconds
    JvalsView<memory_space> jvals{};        ///< Cloud-J photolysis rates in the active memory space
    double temp{288.15};                    ///< Temperature in Kelvin (used for mechanisms like GOCART)
    double air_density{2.4476e19};
    Kokkos::View<const double*, memory_space> cell_temperature;
    Kokkos::View<const double*, memory_space> cell_humidity;
    Kokkos::View<const double*, memory_space> cell_density;
    Kokkos::View<const double*, memory_space> cell_cloud_water;
    Kokkos::View<const double**, Kokkos::LayoutStride, memory_space> cell_jvals;
    double rh{0.5};                         ///< Relative humidity fraction (used for mechanisms like GOCART)
    double clw{0.0};                        ///< Cloud liquid water fraction (used by cloud-gated mechanisms)
    int steps{1};                           ///< Number of integration timesteps
    std::size_t jvals_extent{512};          ///< Fallback photolysis array size when none is supplied
};

using HostExecutionParams = BasicHostExecutionParams<Kokkos::DefaultExecutionSpace::memory_space>;

/// @brief Helper trait function invoking solver integration with matching argument signature.
template <typename SolverKernelsType, typename SubStateView, typename ParamsType>
KOKKOS_INLINE_FUNCTION void integrate_cell(SolverKernelsType& solver,
                                           const ParamsType& params,
                                           SubStateView sub_state) {
    if constexpr (requires { SolverKernelsType::SUPPORTS_RUNTIME_ENVIRONMENT; }) {
        if constexpr (SolverKernelsType::SUPPORTS_RUNTIME_ENVIRONMENT) {
            if constexpr (SolverKernelsType::HAS_CLOUD_GATED) {
                solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh, params.clw, params.air_density);
            } else {
                solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh, params.air_density);
            }
        } else {
            solver.integrate(params.dt, sub_state, params.jvals);
        }
    } else if constexpr (requires { solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh, params.clw); }) {
        solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh, params.clw);
    } else if constexpr (requires { solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh); }) {
        solver.integrate(params.dt, sub_state, params.jvals, params.temp, params.rh);
    } else {
        solver.integrate(params.dt, sub_state, params.jvals);
    }
}

template <typename MemorySpace = Kokkos::DefaultExecutionSpace::memory_space>
struct BatchErrorStatus {
    using memory_space = MemorySpace;
    Kokkos::View<int*, MemorySpace> status;

    BatchErrorStatus() = default;
    explicit BatchErrorStatus(size_t num_cells)
        : status("BatchErrorStatus", num_cells) {}
    explicit BatchErrorStatus(Kokkos::View<int*, MemorySpace> s)
        : status(s) {}

    KOKKOS_INLINE_FUNCTION
    void set_error(size_t cell_idx, CellErrorCode code) const {
        status(cell_idx) = static_cast<int>(code);
    }

    KOKKOS_INLINE_FUNCTION
    int get_error(size_t cell_idx) const {
        return status(cell_idx);
    }
};

/// @brief Serial host integrator for single-thread CPU benchmark parity.
/// @tparam SolverKernelsType Generated solver kernel type callable on host.
/// @tparam StateViewType Kokkos view type containing cell-major concentrations.
/// @param state Four-dimensional concentration view with cell extent in dimension 0.
/// @param dt Integration timestep in seconds.
/// @param steps Number of timesteps to integrate per cell.
/// @param jvals Cloud-J photolysis rates in the state's memory space (empty view if none).
template <typename SolverKernelsType, typename StateViewType>
void execute_mechanism_serial_steps(StateViewType state, const double dt, const int steps,
                                    const JvalsView<typename StateViewType::memory_space>& jvals = {}) {
    const int num_cells = state.extent(0);
    auto jvals_view = detail::resolve_jvals(jvals, 512);

    for (int cell_idx = 0; cell_idx < num_cells; ++cell_idx) {
        SolverKernelsType solver;
        auto sub_state = Kokkos::subview(state, cell_idx, Kokkos::ALL(), 0, 0);
        for (int step = 0; step < steps; ++step) { solver.integrate(dt, sub_state, jvals_view); }
    }
}

/// @brief Serial host integrator for a single timestep.
/// @tparam SolverKernelsType Generated solver kernel type callable on host.
/// @tparam StateViewType Kokkos view type containing cell-major concentrations.
/// @param state Four-dimensional concentration view with cell extent in dimension 0.
/// @param dt Integration timestep in seconds.
/// @param jvals Cloud-J photolysis rates in the state's memory space (empty view if none).
template <typename SolverKernelsType, typename StateViewType>
void execute_mechanism_serial(StateViewType state, const double dt,
                              const JvalsView<typename StateViewType::memory_space>& jvals = {}) {
    execute_mechanism_serial_steps<SolverKernelsType>(state, dt, 1, jvals);
}

template <typename SolverKernelsType, typename StateViewType>
struct TiledCellTimeIntegrator {
    StateViewType m_state;
    double m_dt;
    int m_steps;
    JvalsView<typename StateViewType::memory_space> m_jvals;

    TiledCellTimeIntegrator(StateViewType s, double dt, int steps, const JvalsView<typename StateViewType::memory_space>& jvals = {})
        : m_state(s), m_dt(dt), m_steps(steps), m_jvals(jvals) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell_idx) const {
        SolverKernelsType solver;
        auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL(), 0, 0);
        for (int step = 0; step < m_steps; ++step) { solver.integrate(m_dt, sub_state, m_jvals); }
    }
};

template <typename SolverKernelsType, typename StateViewType>
struct ParamsTiledCellIntegrator {
    StateViewType m_state;
    using ParamsType = BasicHostExecutionParams<typename StateViewType::memory_space>;
    ParamsType m_params;

    ParamsTiledCellIntegrator(StateViewType s, const ParamsType& params)
        : m_state(s), m_params(params) {
        // Normalise a caller-supplied empty photolysis view to the shared
        // zero-filled array once, at the host launch boundary, so device code
        // never allocates or branches on null.
        m_params.jvals = detail::resolve_jvals(m_params.jvals, m_params.jvals_extent);
        const auto cells = m_state.extent(0);
        for (const auto extent : {m_params.cell_temperature.extent(0), m_params.cell_humidity.extent(0),
                                  m_params.cell_density.extent(0), m_params.cell_cloud_water.extent(0)}) {
            if (extent != 0 && extent != cells)
                throw std::invalid_argument("FATAL ERROR: per-cell meteorology extent does not match state");
        }
        if (m_params.cell_jvals.extent(0) != 0 && m_params.cell_jvals.extent(0) != cells)
            throw std::invalid_argument("FATAL ERROR: per-cell forcing extent does not match state");
        if constexpr (requires { SolverKernelsType::NUM_RUNTIME_PARAMETERS; }) {
            if (m_params.cell_jvals.extent(0) != 0 &&
                m_params.cell_jvals.extent(1) < SolverKernelsType::NUM_RUNTIME_PARAMETERS)
                throw std::invalid_argument("FATAL ERROR: per-cell forcing has too few runtime parameters");
        }

    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell_idx) const {
        SolverKernelsType solver;
        auto params = m_params;
        if (params.cell_temperature.extent(0)) params.temp = params.cell_temperature(cell_idx);
        if (params.cell_humidity.extent(0)) params.rh = params.cell_humidity(cell_idx);
        if (params.cell_density.extent(0)) params.air_density = params.cell_density(cell_idx);
        if (params.cell_cloud_water.extent(0)) params.clw = params.cell_cloud_water(cell_idx);
        constexpr int parameter_count = [] {
            if constexpr (requires { SolverKernelsType::NUM_RUNTIME_PARAMETERS; })
                return SolverKernelsType::NUM_RUNTIME_PARAMETERS > 0 ? SolverKernelsType::NUM_RUNTIME_PARAMETERS : 1;
            else return 1024;
        }();
        double forcing[parameter_count];
        if (params.cell_jvals.extent(0) && params.cell_jvals.extent(1)) {
            for (int slot = 0; slot < parameter_count; ++slot)
                forcing[slot] = params.cell_jvals(cell_idx, slot);
            params.jvals = JvalsView<typename StateViewType::memory_space>(forcing, parameter_count);
        }
        if constexpr (StateViewType::rank() == 4) {
            auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL(), 0, 0);
            for (int step = 0; step < m_params.steps; ++step) {
                integrate_cell(solver, params, sub_state);
            }
        } else {
            auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL());
            for (int step = 0; step < m_params.steps; ++step) {
                integrate_cell(solver, params, sub_state);
            }
        }
    }
};

template <typename SolverKernelsType, typename StateViewType, typename ErrorViewType>
struct BatchedTeamCellIntegrator {
    StateViewType m_state;
    double m_dt;
    int m_steps;
    JvalsView<typename StateViewType::memory_space> m_jvals;
    ErrorViewType m_error_status;

    BatchedTeamCellIntegrator(StateViewType s, double dt, int steps, const JvalsView<typename StateViewType::memory_space>& jvals, ErrorViewType err)
        : m_state(s), m_dt(dt), m_steps(steps), m_jvals(jvals), m_error_status(err) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const typename Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>::member_type& member) const {
        const int num_cells = static_cast<int>(m_state.extent(0));
        const int num_species = static_cast<int>(m_state.extent(1));

        const int team_rank = member.league_rank();
        const int num_teams = member.league_size();
        const int cells_per_team = (num_cells + num_teams - 1) / num_teams;
        const int begin_cell = team_rank * cells_per_team;
        const int end_cell = (begin_cell + cells_per_team < num_cells) ? (begin_cell + cells_per_team) : num_cells;

        auto state = m_state;
        auto steps = m_steps;
        auto dt = m_dt;
        auto jvals = m_jvals;
        auto err = m_error_status;

        Kokkos::parallel_for(Kokkos::TeamThreadRange(member, begin_cell, end_cell), [=](const int cell_idx) {
            auto sub_state = Kokkos::subview(state, cell_idx, Kokkos::ALL(), 0, 0);

            // Pre-kernel non-finite boundary check
            bool has_non_finite = false;
            for (int sp = 0; sp < num_species; ++sp) {
                if (!Kokkos::isfinite(sub_state(sp))) {
                    has_non_finite = true;
                    break;
                }
            }

            if (has_non_finite) {
                // Faults are recorded in the per-cell status buffer only; the
                // host launch boundary surfaces them once. Device code must
                // never stream diagnostics (non-deterministic interleaving,
                // and no device I/O in production kernels).
                if constexpr (requires { err.set_error(cell_idx, ERR_NON_FINITE_STATE); }) {
                    err.set_error(cell_idx, ERR_NON_FINITE_STATE);
                } else {
                    err(cell_idx) = static_cast<int>(ERR_NON_FINITE_STATE);
                }
                return;
            }

            SolverKernelsType solver;
            for (int step = 0; step < steps; ++step) {
                solver.integrate(dt, sub_state, jvals);
            }
        });
    }
};

/// @brief Batched host integrator using hierarchical Kokkos::TeamPolicy
template <typename SolverKernelsType, typename StateViewType, typename ErrorViewType>
void execute_mechanism_steps_batched(const std::string& name, StateViewType state, const double dt,
                                     const int steps,
                                     const JvalsView<typename StateViewType::memory_space>& jvals,
                                     ErrorViewType error_status,
                                     int team_size = 64) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using TeamPolicy = Kokkos::TeamPolicy<ExecSpace>;

    const int num_cells = static_cast<int>(state.extent(0));
    const int num_teams = (num_cells + team_size - 1) / team_size;

    TeamPolicy policy(num_teams, Kokkos::AUTO);

    auto jvals_view = detail::resolve_jvals(jvals, 512);
    Kokkos::parallel_for(
        "MKPP_Batched_Team_Dispatch_" + name,
        policy,
        BatchedTeamCellIntegrator<SolverKernelsType, StateViewType, ErrorViewType>(
            state, dt, steps, jvals_view, error_status));

    // Surface device-detected faults exactly once at the host boundary: copy
    // the status buffer back and report a single summary. The per-cell codes
    // remain available to callers through the status view.
    auto status_view = [&]() {
        if constexpr (requires { error_status.status; }) {
            return error_status.status;
        } else {
            return error_status;
        }
    }();
    auto host_status = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, status_view);
    int faulted_cells = 0;
    for (std::size_t i = 0; i < host_status.extent(0); ++i) {
        if (host_status(i) != SUCCESS) {
            ++faulted_cells;
        }
    }
    if (faulted_cells > 0) {
        std::cerr << "FATAL ERROR: batched integration of '" << name << "' reported "
                  << faulted_cells << " faulted cell(s) of " << host_status.extent(0)
                  << "; per-cell codes are in the status buffer\n";
    }
}

/// @brief Integrate all requested timesteps with a single Kokkos launch.
/// @tparam SolverKernelsType Generated solver kernel type callable in the active execution space.
/// @tparam StateViewType Kokkos view type containing cell-major concentrations.
/// @param name Mechanism name used for Kokkos profiling labels.
/// @param state Four-dimensional concentration view with cell extent in dimension 0.
/// @param dt Integration timestep in seconds.
/// @param steps Number of timesteps to integrate per cell.
/// @param jvals Cloud-J photolysis rates in the state's memory space (empty view if none).
template <typename SolverKernelsType, typename StateViewType>
void execute_mechanism_steps(const std::string& name, StateViewType state, const double dt,
                             const int steps,
                             const JvalsView<typename StateViewType::memory_space>& jvals = {}) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    const int num_cells = state.extent(0);
    auto jvals_view = detail::resolve_jvals(jvals, 512);

    Kokkos::parallel_for(
        "MKPP_Grid_Dispatch_" + name,
        Kokkos::RangePolicy<ExecSpace>(0, num_cells, Kokkos::ChunkSize(64)),
        TiledCellTimeIntegrator<SolverKernelsType, StateViewType>(state, dt, steps, jvals_view));
}

template <typename SolverKernelsType, typename StateViewType>
void execute_mechanism(const std::string& name, StateViewType state, double dt,
                       const JvalsView<typename StateViewType::memory_space>& jvals = {}) {
    execute_mechanism_steps<SolverKernelsType>(name, state, dt, 1, jvals);
}

}  // namespace host

/// @brief Core handle implementation structure for MKPP Host Model API C ABI wrappers.
struct mkpp_handle_impl {
    std::size_t num_cells{0};
    std::size_t num_species{0};
    std::size_t num_reactions{0};
    std::size_t num_photolysis{0};
    std::size_t num_rate_parameters{0};
    bool is_initialized{false};

    // Unmanaged 2D view over host model species concentrations (num_cells, num_species)
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> species_conc;

    // Unmanaged 1D views over host model meteorology & aerosol drivers (num_cells)
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> temperature;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> pressure;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> air_density;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> relative_humidity;
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> aerosol_surface_area;

    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> cloud_water;

    // Unmanaged 2D view over host model photolysis rates (num_cells, num_photolysis)
    Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> photolysis_rates;


    // Human-readable error diagnostic message for last failure
    std::string last_error_msg;

    void set_error(const std::string& msg) {
        last_error_msg = msg;
    }
};

}  // namespace mkpp
