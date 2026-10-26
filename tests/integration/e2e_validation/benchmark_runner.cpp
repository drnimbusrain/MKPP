#include <Kokkos_Core.hpp>
#include <mkpp_host/mechanism_manager.hpp>

#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using ExecSpace = Kokkos::DefaultExecutionSpace;

double run_diurnal_benchmark(const std::string& mech_name, int num_cells, int num_species, int num_steps, double dt,
                             std::vector<double>& final_state_out) {
    using ViewType = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecSpace::memory_space>;
    ViewType state("state", num_cells, num_species);

    // Zero jvals (photolysis rates set to zero for benchmark timing)
    Kokkos::View<double*, typename ExecSpace::memory_space> jvals_dev("jvals", 64);

    // Initialize initial concentrations matching KPP initial conditions
    auto host_state = Kokkos::create_mirror_view(state);
    const double cfactor = 2.447600e+13;

    for (int i = 0; i < num_cells; ++i) {
        for (int k = 0; k < num_species; ++k) host_state(i, k) = 0.0;

        // Match saprc99 initial conditions
        if (num_species >= 79) {
            host_state(i, 1) = (6.77e-4) * cfactor;
            host_state(i, 2) = (1.16e-3) * cfactor;
            host_state(i, 3) = (3.92e-4) * cfactor;
            host_state(i, 7) = (0.2) * cfactor;
            host_state(i, 8) = (5.e-2) * cfactor;
            host_state(i, 10) = (1.167e-2) * cfactor;
            host_state(i, 68) = (1.0e-1) * cfactor;   // NO
            host_state(i, 69) = (5.0e-2) * cfactor;   // NO2
            host_state(i, 74) = (1.0e+6) * cfactor;   // AIR
            host_state(i, 75) = (2.09e+5) * cfactor;  // O2
            host_state(i, 76) = (2.0e+04) * cfactor;  // H2O
            host_state(i, 78) = (1.0e0) * cfactor;    // CH4
        } else {
            for (int k = 0; k < num_species; ++k) host_state(i, k) = 1.0e8;
        }
    }
    Kokkos::deep_copy(state, host_state);

    mkpp::host::HostExecutionParams params;
    params.dt = dt;
    params.jvals = jvals_dev;
    params.steps = 1;

    // Warmup step via host interface
    mkpp::host::MechanismRegistry::execute(mech_name, state, params);
    Kokkos::fence();

    params.steps = num_steps;
    auto start = std::chrono::high_resolution_clock::now();
    mkpp::host::MechanismRegistry::execute(mech_name, state, params);
    Kokkos::fence();
    auto end = std::chrono::high_resolution_clock::now();

    Kokkos::deep_copy(host_state, state);
    final_state_out.resize(num_species);
    for (int k = 0; k < num_species; ++k) { final_state_out[k] = host_state(0, k); }

    std::chrono::duration<double, std::milli> elapsed = end - start;
    return elapsed.count();
}

int main(int argc, char** argv) {
    int num_cells = 1000;
    int num_steps = 1440;
    std::string mode = "full";  // "full" or "lumped"
    std::string mech = "saprc99";
    std::string csv_out = "";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cells" && i + 1 < argc) num_cells = std::stoi(argv[++i]);
        if (arg == "--steps" && i + 1 < argc) num_steps = std::stoi(argv[++i]);
        if (arg == "--mode" && i + 1 < argc) mode = argv[++i];
        if (arg == "--mechanism" && i + 1 < argc) mech = argv[++i];
        if (arg == "--output" && i + 1 < argc) csv_out = argv[++i];
    }

    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    {
        std::vector<double> final_state;
        auto info = mkpp::host::MechanismRegistry::get_info(mech);
        double time_ms = run_diurnal_benchmark(
            mech, num_cells, info.num_species, num_steps, 60.0, final_state);

        std::cout << "CELLS=" << num_cells << ", STEPS=" << num_steps << ", MODE=" << mode
                  << ", TIME_MS=" << std::fixed << std::setprecision(2) << time_ms
                  << ", THROUGHPUT=" << std::scientific << std::setprecision(3)
                  << ((double)num_cells * num_steps) / (time_ms / 1000.0) << "\n";

        if (!csv_out.empty()) {
            std::ofstream out(csv_out);
            out << "species_idx,concentration\n";
            for (size_t k = 0; k < final_state.size(); ++k) {
                out << k << "," << std::scientific << std::setprecision(12) << final_state[k]
                    << "\n";
            }
            out.close();
        }
    }
    return 0;
}
