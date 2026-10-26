#define MKPP_HOST_SELECTED_MECHANISMS
#define MKPP_HOST_MECH_CHAPMAN
#include <mkpp_host/mechanism_manager.hpp>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    Kokkos::ScopeGuard runtime(argc, argv);
    using Space = Kokkos::DefaultExecutionSpace;
    using State = Kokkos::View<double**, Kokkos::LayoutLeft, Space::memory_space>;
    State state("state", 1, 4);
    auto host = Kokkos::create_mirror_view(state);
    const std::array<double, 4> initial{2.5e9, 5.1e18, 8.0e11, 2.46e19};
    for (int i = 0; i < 4; ++i) host(0, i) = initial[i];
    Kokkos::deep_copy(state, host);
    Kokkos::View<double*, Space::memory_space> forcing("forcing", 2);
    auto host_forcing = Kokkos::create_mirror_view(forcing);
    host_forcing(0) = 1e-12;
    host_forcing(1) = 1e-4;
    Kokkos::deep_copy(forcing, host_forcing);
    mkpp::host::HostExecutionParams parameters;
    parameters.dt = 60.0;
    parameters.steps = 10;
    parameters.jvals = forcing;
    parameters.jvals_extent = 2;
    mkpp::host::MechanismRegistry::execute("chapman", state, parameters);
    Kokkos::fence();
    Kokkos::deep_copy(host, state);
    for (int i = 0; i < 4; ++i) if (!std::isfinite(host(0, i))) return 2;
    std::cout << std::setprecision(17) << "{\"kokkos_version\":" << KOKKOS_VERSION
              << ",\"backend\":\"" << Space::name() << "\",\"finite\":true,\"final_state\":[";
    for (int i = 0; i < 4; ++i) { if (i) std::cout << ','; std::cout << host(0, i); }
    std::cout << "],\"publication_eligible\":false}\n";
}
