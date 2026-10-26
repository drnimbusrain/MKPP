#include <Kokkos_Core.hpp>

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cstdlib>

// Generated MICM C++ TS1 Solver
#include "generated_micm_ts1.hpp"

// Generated MKPP AOT TS1 Solver
#include "ts1.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;

// ---------------------------------------------------------------------------
// MKPP TS1 Functor
// ---------------------------------------------------------------------------
struct MKPPTS1Functor {
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
    using JvalsType = Kokkos::View<const double*, typename ExecSpace::memory_space>;
    ViewType m_state;
    double m_dt;
    JvalsType m_jvals;

    MKPPTS1Functor(ViewType state, double dt, JvalsType jvals)
        : m_state(state), m_dt(dt), m_jvals(jvals) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell_idx) const {
        auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL());
        mkpp::SolverKernels<ExecSpace> solver;
        solver.integrate(m_dt, sub_state, m_jvals);
    }
};

// ---------------------------------------------------------------------------
// Direct C++ TS1 Benchmark Entry Point
// ---------------------------------------------------------------------------
int main(int argc, char* argv[]) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    int exit_code = 0;
    {
        int num_cells = 1000;
        int num_steps = 10;
        const double dt = 60.0;
        const int num_species = 210;

        if (const char* env_cells = std::getenv("NUM_CELLS")) {
            num_cells = std::atoi(env_cells);
        }
        if (const char* env_steps = std::getenv("NUM_STEPS")) {
            num_steps = std::atoi(env_steps);
        }

        std::cout << "==========================================================================" << std::endl;
        std::cout << "      Direct C++ Benchmark: MKPP vs NCAR/MICM (TS1 Mechanism)            " << std::endl;
        std::cout << "==========================================================================" << std::endl;
        std::cout << "Species    : " << num_species << std::endl;
        std::cout << "Reactions  : 547" << std::endl;
        std::cout << "Grid Cells : " << num_cells << std::endl;
        std::cout << "Timesteps  : " << num_steps << std::endl;
        std::cout << "Step Size  : " << dt << " s\n" << std::endl;

        // 1. Build MICM TS1 Solver
        std::cout << "Building MICM TS1 Solver (210 species, 547 reactions)..." << std::flush;
        auto start_build_micm = std::chrono::high_resolution_clock::now();
        auto micm_solver = build_micm_ts1_solver();
        auto end_build_micm = std::chrono::high_resolution_clock::now();
        double micm_build_ms = std::chrono::duration<double, std::milli>(end_build_micm - start_build_micm).count();
        std::cout << " done (" << micm_build_ms << " ms)\n" << std::endl;

        // 2. Benchmark MICM
        auto micm_state = micm_solver.GetState();
        std::vector<double> init_conc(num_species, 0.0);
        // Set background species (N2=10, O2=4)
        init_conc[10] = 1.9e19; // N2
        init_conc[4]  = 5.1e18; // O2

        auto s = micm_solver.GetState(num_cells);
        for (int cell = 0; cell < num_cells; ++cell) {
            s.conditions_[cell].temperature_ = 288.15;
            s.conditions_[cell].pressure_ = 101325.0;
            s.variables_[cell] = init_conc;
        }

        micm_solver.UpdateStateParameters(s);

        // Warmup MICM
        micm_solver.Solve(dt, s);

        // Re-initialize MICM state for benchmark
        for (int cell = 0; cell < num_cells; ++cell) {
            s.variables_[cell] = init_conc;
        }
        micm_solver.UpdateStateParameters(s);

        std::cout << "Running MICM TS1 Benchmark..." << std::flush;
        auto start_micm = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < num_steps; ++step) {
            micm_solver.Solve(dt, s);
        }

        auto end_micm = std::chrono::high_resolution_clock::now();
        double micm_time_ms = std::chrono::duration<double, std::milli>(end_micm - start_micm).count();
        std::cout << " done (" << micm_time_ms << " ms)\n" << std::endl;

        // 3. Setup MKPP
        using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
        ViewType mkpp_state("state", num_cells, num_species);
        Kokkos::View<double*, typename ExecSpace::memory_space> mkpp_jvals("jvals", 600);

        auto h_state = Kokkos::create_mirror_view(mkpp_state);
        auto h_jvals = Kokkos::create_mirror_view(mkpp_jvals);

        for (int k = 0; k < 600; ++k) h_jvals(k) = 1.0e-5;
        Kokkos::deep_copy(mkpp_jvals, h_jvals);

        for (int i = 0; i < num_cells; ++i) {
            for (int native_index = 0; native_index < num_species; ++native_index) {
                h_state(i, native_index) = 0.0;
            }
            h_state(i, 10) = 1.9e19; // N2
            h_state(i, 4)  = 5.1e18; // O2
        }
        Kokkos::deep_copy(mkpp_state, h_state);

        // 4. Warmup MKPP
        Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                             MKPPTS1Functor(mkpp_state, dt, mkpp_jvals));
        Kokkos::fence();

        // Re-initialize MKPP state for benchmark
        for (int i = 0; i < num_cells; ++i) {
            for (int native_index = 0; native_index < num_species; ++native_index) {
                h_state(i, native_index) = 0.0;
            }
            h_state(i, 10) = 1.9e19; // N2
            h_state(i, 4)  = 5.1e18; // O2
        }
        Kokkos::deep_copy(mkpp_state, h_state);
        Kokkos::fence();

        // 5. Benchmark MKPP
        std::cout << "Running MKPP TS1 Benchmark..." << std::flush;
        auto start_mkpp = std::chrono::high_resolution_clock::now();
        for (int step = 0; step < num_steps; ++step) {
            Kokkos::parallel_for("step", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                                 MKPPTS1Functor(mkpp_state, dt, mkpp_jvals));
        }
        Kokkos::fence();
        auto end_mkpp = std::chrono::high_resolution_clock::now();
        double mkpp_time_ms = std::chrono::duration<double, std::milli>(end_mkpp - start_mkpp).count();
        std::cout << " done (" << mkpp_time_ms << " ms)\n" << std::endl;

        // 6. Parity Verification
        Kokkos::deep_copy(h_state, mkpp_state);

        double max_abs_err = 0.0;
        double max_rel_err = 0.0;

        // This diagnostic cannot prove a canonical name mapping for the old
        // generated MKPP header, so it deliberately withholds trajectory parity.
        // The solver-comparison TS1 runner loads the audited binding instead.
        max_abs_err = std::numeric_limits<double>::quiet_NaN();
        max_rel_err = std::numeric_limits<double>::quiet_NaN();

        // 7. Results
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

        const double parity_threshold = 0.05; // 5.0% max relative error
        if (!std::isfinite(max_rel_err) || max_rel_err > parity_threshold) {
            std::cerr << "FATAL ERROR: TS1 parity check failed: max relative error "
                      << max_rel_err << " exceeds threshold " << parity_threshold << "." << std::endl;
            exit_code = 1;
        } else {
            std::cout << "SUCCESS: TS1 parity check passed (max rel error "
                      << max_rel_err << " <= " << parity_threshold << ")." << std::endl;
        }
    }
    return exit_code;
}
