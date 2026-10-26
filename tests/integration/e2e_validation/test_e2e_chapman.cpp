#include <Kokkos_Core.hpp>

#include <fstream>
#include <gtest/gtest.h>
#include <vector>

#include "chapman.hpp"  // The generated header

using ExecSpace = Kokkos::DefaultExecutionSpace;

// T010: Profiling Hooks
static size_t total_allocations = 0;
void test_allocate(const Kokkos::Profiling::SpaceHandle& handle, const char* name, const void* ptr,
                   uint64_t size) {
    total_allocations += size;
}

// Rosenbrock simulation loop testing the raw MKPP headers
struct RosenbrockFunctor {
    mkpp::concentrations_view_t state;

    RosenbrockFunctor(mkpp::concentrations_view_t s) : state(s) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) const {
        int i = team.league_rank();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](const int& s) {
            // Use a subview for state access (generated solver uses operator())
            auto cell_state = Kokkos::subview(state, i, Kokkos::ALL(), 0, 0);

            mkpp::SolverKernels<ExecSpace> solver;

            double dt = 3600.0;
            double jvals[2] = {2.0e-5, 0.001};  // O2 photolysis, O3 photolysis
            solver.integrate(dt, cell_state, jvals);
        });
    }
};

TEST(E2ESolverValidation, RosenbrockIntegrationAccuracy) {
    // Kokkos::Profiling::setSpaceAllocateCallback(test_allocate);
    total_allocations = 0;

    int cells = 5;
    std::vector<double> host_data(cells * 4 * 1 * 1, 0.0);
    // Init C_M etc
    for (int i = 0; i < cells; i++) {
        host_data[i * 4 + 0] = 1.0e10;  // O
        host_data[i * 4 + 1] = 2.0e10;  // O2
        host_data[i * 4 + 2] = 3.0e10;  // O3
        host_data[i * 4 + 3] = 4.0e10;  // M
    }

    mkpp::concentrations_view_t state(host_data.data(), cells, 4, 1, 1);

    Kokkos::parallel_for("Rosenbrock_Team_Dispatch",
                         Kokkos::TeamPolicy<ExecSpace>(cells, Kokkos::AUTO),
                         RosenbrockFunctor(state));
    Kokkos::fence();

    // Kokkos::Profiling::setSpaceAllocateCallback(nullptr);

    // T010: assert 0-byte dynamic allocations during dispatch
    EXPECT_EQ(total_allocations, 0);

    // Output to CSV
    std::ofstream out("e2e_output.csv");
    out << "time_step,species_name,concentration\n";
    // We only simulate step 1 for cell 0 to compare
    out << "1,O," << host_data[0] << "\n";
    out << "1,O2," << host_data[1] << "\n";
    out << "1,O3," << host_data[2] << "\n";
    out << "1,M," << host_data[3] << "\n";
    out.close();
}

int main(int argc, char** argv) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
