#include <Kokkos_Core.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

// MICM includes
#include <micm/CPU.hpp>
#include <micm/process/chemical_reaction_builder.hpp>
#include <micm/util/types.hpp>

// MKPP includes
#include "chapman.hpp"

#include <cstdlib>

using ExecSpace = Kokkos::DefaultExecutionSpace;

// ---------------------------------------------------------------------------
// MKPP Functor
// ---------------------------------------------------------------------------
struct MKPPDualFunctor {
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
    using JvalsType = Kokkos::View<const double*, typename ExecSpace::memory_space>;
    ViewType m_state;
    double m_dt;
    JvalsType m_jvals;

    MKPPDualFunctor(ViewType state, double dt, JvalsType jvals)
        : m_state(state), m_dt(dt), m_jvals(jvals) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell_idx) const {
        auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL());
        mkpp::SolverKernels<ExecSpace> solver;
        solver.integrate(m_dt, sub_state, m_jvals);
    }
};

// ---------------------------------------------------------------------------
// Direct C++ Benchmark
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    int exit_code = 0;
    {
        int num_cells = 1000;
        int num_steps = 100;
        const double dt = 60.0;

        if (const char* env_cells = std::getenv("NUM_CELLS")) {
            num_cells = std::atoi(env_cells);
        }
        if (const char* env_steps = std::getenv("NUM_STEPS")) {
            num_steps = std::atoi(env_steps);
        }

        std::cout << "==========================================================================" << std::endl;
        std::cout << "      Direct C++ Benchmark: MKPP vs NCAR/MICM (Chapman Mechanism)       " << std::endl;
        std::cout << "==========================================================================" << std::endl;
        std::cout << "Grid Cells : " << num_cells << std::endl;
        std::cout << "Timesteps  : " << num_steps << std::endl;
        std::cout << "Step Size  : " << dt << " s\n" << std::endl;

        // Initial concentration values
        const double init_O  = 1.0e10;
        const double init_O2 = 2.0e10; // fixed background in MKPP
        const double init_O3 = 3.0e10;
        const double init_M  = 4.0e10; // fixed background in MKPP

        const double J1_val = 1.0e-12; // J(O2 hv)
        const double J3_val = 1.0e-4;  // J(O3 hv)

        // 1. Setup MICM System with O and O3 as active species and fixed O2, M
        // Matching MKPP Chapman system:
        // R1: O2 + hv -> 2 O          (rate = J1 * [O2])
        // R2: O + O2 + M -> O3 + M    (rate = 6e-34 * [M] * [O2] * [O])
        // R3: O3 + hv -> O + O2       (rate = J3 * [O3])
        // R4: O + O3 -> 2 O2          (rate = 8e-12 * [O] * [O3])
        micm::Species o("O"), o2("O2"), o3("O3"), m("M");
        micm::Phase gas_phase{"gas", std::vector<micm::PhaseSpecies>{o, o2, o3, m}};
        micm::System system{gas_phase};

        micm::Process r1 = micm::ChemicalReactionBuilder()
            .SetReactants({o2})
            .SetProducts({micm::StoichSpecies(o, 2), micm::StoichSpecies(o2, 1)})
            .SetRateConstant(micm::UserDefinedRateConstantParameters{.label_ = "J1"})
            .SetPhase(gas_phase)
            .Build();

        micm::Process r2 = micm::ChemicalReactionBuilder()
            .SetReactants({o, o2, m})
            .SetProducts({micm::StoichSpecies(o3, 1), micm::StoichSpecies(o2, 1), micm::StoichSpecies(m, 1)})
            .SetRateConstant(micm::ArrheniusRateConstantParameters{.A_ = 6.0e-34})
            .SetPhase(gas_phase)
            .Build();

        micm::Process r3 = micm::ChemicalReactionBuilder()
            .SetReactants({o3})
            .SetProducts({micm::StoichSpecies(o, 1), micm::StoichSpecies(o2, 1)})
            .SetRateConstant(micm::UserDefinedRateConstantParameters{.label_ = "J3"})
            .SetPhase(gas_phase)
            .Build();

        micm::Process r4 = micm::ChemicalReactionBuilder()
            .SetReactants({o, o3})
            .SetProducts({micm::StoichSpecies(o2, 2)})
            .SetRateConstant(micm::ArrheniusRateConstantParameters{.A_ = 8.0e-12})
            .SetPhase(gas_phase)
            .Build();

        std::vector<micm::Process> reactions{r1, r2, r3, r4};
        micm::RosenbrockSolverParameters params = micm::RosenbrockSolverParameters::ThreeStageRosenbrockParameters();

        auto micm_solver = micm::CpuSolverBuilder<micm::RosenbrockSolverParameters>(params)
                    .SetSystem(system)
                    .SetReactions(reactions)
                    .SetIgnoreUnusedSpecies(true)
                    .Build();

        // 2. Setup MICM multi-cell state
        auto s = micm_solver.GetState(num_cells);
        size_t micm_idx_O  = s.variable_map_.at("O");
        size_t micm_idx_O2 = s.variable_map_.at("O2");
        size_t micm_idx_O3 = s.variable_map_.at("O3");
        size_t micm_idx_M  = s.variable_map_.at("M");

        for (int cell = 0; cell < num_cells; ++cell) {
            s.conditions_[cell].temperature_ = 288.15;
            s.conditions_[cell].pressure_ = 101325.0;
            s.variables_[cell][micm_idx_O]  = init_O;
            s.variables_[cell][micm_idx_O2] = init_O2;
            s.variables_[cell][micm_idx_O3] = init_O3;
            s.variables_[cell][micm_idx_M]  = init_M;
        }

        std::vector<double> custom_j1(num_cells, J1_val);
        std::vector<double> custom_j3(num_cells, J3_val);
        s.SetCustomRateParameter("J1", custom_j1);
        s.SetCustomRateParameter("J3", custom_j3);

        micm_solver.UpdateStateParameters(s);

        // Warmup MICM
        micm_solver.Solve(dt, s);

        // Re-initialize MICM state for benchmark
        for (int cell = 0; cell < num_cells; ++cell) {
            s.variables_[cell][micm_idx_O]  = init_O;
            s.variables_[cell][micm_idx_O2] = init_O2;
            s.variables_[cell][micm_idx_O3] = init_O3;
            s.variables_[cell][micm_idx_M]  = init_M;
        }

        micm_solver.UpdateStateParameters(s);

        // Benchmark MICM (timing only solve calls)
        auto start_micm = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < num_steps; ++step) {
            micm_solver.Solve(dt, s);
        }
        auto end_micm = std::chrono::high_resolution_clock::now();
        double micm_time_ms = std::chrono::duration<double, std::milli>(end_micm - start_micm).count();

        // 3. Setup MKPP
        using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
        ViewType mkpp_state("state", num_cells, 4);
        Kokkos::View<double*, typename ExecSpace::memory_space> mkpp_jvals("jvals", 2);

        auto h_state = Kokkos::create_mirror_view(mkpp_state);
        auto h_jvals = Kokkos::create_mirror_view(mkpp_jvals);

        h_jvals(0) = J1_val; // J1
        h_jvals(1) = J3_val; // J3
        Kokkos::deep_copy(mkpp_jvals, h_jvals);

        for (int i = 0; i < num_cells; ++i) {
            h_state(i, mkpp::Species::O)  = init_O;
            h_state(i, mkpp::Species::O2) = init_O2;
            h_state(i, mkpp::Species::O3) = init_O3;
            h_state(i, mkpp::Species::M)  = init_M;
        }
        Kokkos::deep_copy(mkpp_state, h_state);

        // Warmup MKPP
        Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                             MKPPDualFunctor(mkpp_state, dt, mkpp_jvals));
        Kokkos::fence();

        // Re-initialize MKPP state for benchmark
        for (int i = 0; i < num_cells; ++i) {
            h_state(i, mkpp::Species::O)  = init_O;
            h_state(i, mkpp::Species::O2) = init_O2;
            h_state(i, mkpp::Species::O3) = init_O3;
            h_state(i, mkpp::Species::M)  = init_M;
        }
        Kokkos::deep_copy(mkpp_state, h_state);
        Kokkos::fence();

        // Benchmark MKPP (timing only kernel execution and final fence)
        auto start_mkpp = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < num_steps; ++step) {
            Kokkos::parallel_for("step", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                                 MKPPDualFunctor(mkpp_state, dt, mkpp_jvals));
        }
        Kokkos::fence();
        auto end_mkpp = std::chrono::high_resolution_clock::now();
        double mkpp_time_ms = std::chrono::duration<double, std::milli>(end_mkpp - start_mkpp).count();

        // 4. Parity Verification
        Kokkos::deep_copy(h_state, mkpp_state);

        double max_abs_err = 0.0;
        double max_rel_err = 0.0;

        for (int cell = 0; cell < num_cells; ++cell) {
            double mkpp_O  = h_state(cell, mkpp::Species::O);
            double micm_O  = s.variables_[cell][micm_idx_O];
            double mkpp_O3 = h_state(cell, mkpp::Species::O3);
            double micm_O3 = s.variables_[cell][micm_idx_O3];

            double abs_O = std::abs(mkpp_O - micm_O);
            double rel_O = abs_O / std::max(1.0e-12, std::abs(micm_O));

            double abs_O3 = std::abs(mkpp_O3 - micm_O3);
            double rel_O3 = abs_O3 / std::max(1.0e-12, std::abs(micm_O3));

            max_abs_err = std::max(max_abs_err, std::max(abs_O, abs_O3));
            max_rel_err = std::max(max_rel_err, std::max(rel_O, rel_O3));
        }

        std::cout << std::left << std::setw(28) << "Metric"
                  << std::left << std::setw(18) << "MKPP C++ (AOT)"
                  << std::left << std::setw(18) << "MICM C++"
                  << std::left << std::setw(18) << "Speedup" << std::endl;
        std::cout << std::string(80, '-') << std::endl;

        std::cout << std::left << std::setw(28) << "Execution Time (ms)"
                  << std::left << std::setw(18) << std::fixed << std::setprecision(2) << mkpp_time_ms
                  << std::left << std::setw(18) << std::fixed << std::setprecision(2) << micm_time_ms
                  << std::left << std::setw(18) << std::fixed << std::setprecision(2) << (micm_time_ms / mkpp_time_ms) << "x" << std::endl;

        double mkpp_throughput = (num_cells * num_steps) / (mkpp_time_ms / 1000.0);
        double micm_throughput = (num_cells * num_steps) / (micm_time_ms / 1000.0);

        std::cout << std::left << std::setw(28) << "Throughput (cell-st/s)"
                  << std::left << std::setw(18) << std::scientific << std::setprecision(2) << mkpp_throughput
                  << std::left << std::setw(18) << std::scientific << std::setprecision(2) << micm_throughput
                  << "--" << std::endl;

        std::cout << std::left << std::setw(28) << "Max Abs Error"
                  << std::left << std::setw(18) << std::scientific << std::setprecision(3) << max_abs_err
                  << "--                --" << std::endl;
        std::cout << std::left << std::setw(28) << "Max Rel Error"
                  << std::left << std::setw(18) << std::scientific << std::setprecision(3) << max_rel_err
                  << "--                --" << std::endl;

        std::cout << "==========================================================================" << std::endl;

        const double parity_threshold = 0.02; // 2.0% max relative error across 100 integration steps
        if (max_rel_err > parity_threshold) {
            std::cerr << "FATAL ERROR: Chapman parity check failed: max relative error "
                      << max_rel_err << " exceeds threshold " << parity_threshold << "." << std::endl;
            exit_code = 1;
        } else {
            std::cout << "SUCCESS: Chapman parity check passed (max rel error "
                      << max_rel_err << " <= " << parity_threshold << ")." << std::endl;
        }
    }
    return exit_code;
}
