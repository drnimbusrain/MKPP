#include <Kokkos_Core.hpp>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#include "saprc99.hpp"

using ExecSpace = Kokkos::DefaultExecutionSpace;

template <typename SolverKernelsType>
struct DiurnalCellFunctor {
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
    using JvalsType = Kokkos::View<const double*, typename ExecSpace::memory_space>;
    ViewType m_state;
    double m_dt;
    JvalsType m_jvals;

    DiurnalCellFunctor(ViewType state, double dt, JvalsType jvals)
        : m_state(state), m_dt(dt), m_jvals(jvals) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int cell_idx) const {
        auto sub_state = Kokkos::subview(m_state, cell_idx, Kokkos::ALL());
        SolverKernelsType solver;
        solver.integrate(m_dt, sub_state, m_jvals);
    }
};

template <typename SolverKernelsType>
double run_diurnal_benchmark(int num_cells, int num_species, int num_steps = 1440,
                             double dt = 60.0) {
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
    ViewType state("state", num_cells, num_species);

    // Zero jvals (photolysis rates set to zero for benchmark timing)
    Kokkos::View<double*, typename ExecSpace::memory_space> jvals_dev("jvals", 64);

    // Initialize initial concentrations
    auto host_state = Kokkos::create_mirror_view(state);
    for (int i = 0; i < num_cells; ++i) {
        for (int k = 0; k < num_species; ++k) {
            host_state(i, k) = 1.0e8;  // 1e8 molec/cm3 baseline
        }
    }
    Kokkos::deep_copy(state, host_state);

    // Warmup step
    Kokkos::parallel_for("warmup", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                         DiurnalCellFunctor<SolverKernelsType>(state, dt, jvals_dev));
    Kokkos::fence();

    auto start = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < num_steps; ++step) {
        Kokkos::parallel_for("diurnal_step", Kokkos::RangePolicy<ExecSpace>(0, num_cells),
                             DiurnalCellFunctor<SolverKernelsType>(state, dt, jvals_dev));
    }
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char** argv) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    {
        std::cout << "=========================================================\n";
        std::cout << "    MKPP vs Legacy KPP SAPRC-99 Benchmark Suite          \n";
        std::cout << "=========================================================\n";

        std::vector<int> cell_counts = {1, 100, 1000, 10000, 100000};

        std::cout << "\n--- Full SAPRC-99 (79 Species, Unrolled Implicit ROS-2) ---\n";
        std::cout << std::setw(12) << "Grid Cells" << std::setw(18) << "Time (ms)" << std::setw(22)
                  << "Throughput (cells/s)" << "\n";
        std::cout << "---------------------------------------------------------\n";

        for (int n_cells : cell_counts) {
            double time_ms = run_diurnal_benchmark<mkpp::SolverKernels<ExecSpace>>(n_cells, 79);
            double total_cell_steps = (double)n_cells * 1440.0;
            double throughput = total_cell_steps / (time_ms / 1000.0);

            std::cout << std::setw(12) << n_cells << std::setw(18) << std::fixed
                      << std::setprecision(2) << time_ms << std::setw(22) << std::scientific
                      << std::setprecision(2) << throughput << "\n";
        }
    }
    return 0;
}
