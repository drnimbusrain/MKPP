// Verify heterogeneous LayoutLeft batches against independent single-cell solves.
#include <Kokkos_Core.hpp>
#include "mkpp.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

void check(int code) {
    if (code != MKPP_SUCCESS) throw std::runtime_error("FATAL ERROR: C API operation failed");
}
void near(double a, double b, const char* context = "state") {
    if (!std::isfinite(a) || !std::isfinite(b) ||
        std::abs(a - b) > 1e-10 * std::max({std::abs(a), std::abs(b), 1e-25}))
        { std::cerr << context << ": " << std::setprecision(17) << a << " vs " << b << "\n";
        throw std::runtime_error("FATAL ERROR: heterogeneous batch differs from independent solve"); }
}
int main(int argc, char** argv) try {
    Kokkos::ScopeGuard runtime(argc, argv);
    constexpr int cells = 3, species = MKPP_NUM_SPECIES;
    std::array<double, cells> temperature{280, 300, 320};
    std::array<double, cells> pressure{90000, 95000, 100000};
    std::array<double, cells> density{1, 2, 3};
    std::vector<double> state(cells * species), forcing(cells * MKPP_NUM_RATE_PARAMETERS);
    for (int sp = 0; sp < species; ++sp)
        for (int cell = 0; cell < cells; ++cell) state[cell + cells * sp] = 1.0 + 0.1 * cell;
    for (int slot = 0; slot < MKPP_NUM_RATE_PARAMETERS; ++slot)
        for (int cell = 0; cell < cells; ++cell) forcing[cell + cells * slot] = (slot + 1) * (cell + 1) * 1e-4;
    const auto initial = state;
    mkpp_handle_t batch{};
    check(mkpp_create_handle(&batch, cells)); check(mkpp_initialize(batch));
    check(mkpp_set_state_ptrs(batch, state.data(), temperature.data(), pressure.data(), density.data()));
    check(mkpp_set_rate_parameter_ptrs(batch, forcing.data(), MKPP_NUM_RATE_PARAMETERS));
    std::vector<double> rates(cells * species);
    check(mkpp_evaluate_rates(batch, rates.data()));
    check(mkpp_integrate(batch, 1.0));
    for (int cell = 0; cell < cells; ++cell) {
        std::vector<double> individual(species), parameters(MKPP_NUM_RATE_PARAMETERS), individual_rates(species);
        for (int sp = 0; sp < species; ++sp) individual[sp] = initial[cell + cells * sp];
        for (int slot = 0; slot < MKPP_NUM_RATE_PARAMETERS; ++slot) parameters[slot] = forcing[cell + cells * slot];
        mkpp_handle_t single{};
        check(mkpp_create_handle(&single, 1)); check(mkpp_initialize(single));
        check(mkpp_set_state_ptrs(single, individual.data(), &temperature[cell], &pressure[cell], &density[cell]));
        check(mkpp_set_rate_parameter_ptrs(single, parameters.data(), parameters.size()));
        check(mkpp_evaluate_rates(single, individual_rates.data()));
        for (int sp = 0; sp < species; ++sp) near(individual_rates[sp], rates[cell + cells * sp], "rates");
        check(mkpp_integrate(single, 1.0));
        for (int sp = 0; sp < species; ++sp) near(individual[sp], state[cell + cells * sp]);
        check(mkpp_destroy_handle(single));
    }
    const auto integrated = state;
    // Hold state and photolysis fixed so only the temperature-dependent rate changes.
    std::fill(state.begin(), state.end(), 1.0);
    std::fill(forcing.begin(), forcing.end(), 0.0);
    int o3 = -1; check(mkpp_get_species_index(batch, "O3", &o3));
    for (int cell = 0; cell < cells; ++cell) state[cell + cells * o3] = 0.0;
    check(mkpp_evaluate_rates(batch, rates.data()));
    near(rates[cells * o3] / rates[cells * o3 + 2], std::exp(-400.0 / 280.0 + 400.0 / 320.0), "temperature ratio");
    temperature[0] = 0.0;
    if (mkpp_integrate(batch, 1.0) != MKPP_ERROR_INVALID_ARGUMENT)
        throw std::runtime_error("FATAL ERROR: invalid meteorology was accepted");
    check(mkpp_destroy_handle(batch));
    std::cout << std::setprecision(17) << '[';
    for (std::size_t i = 0; i < integrated.size(); ++i) { if (i) std::cout << ','; std::cout << integrated[i]; }
    for (const auto value : rates) std::cout << ',' << value;
    std::cout << "]\n";
}
 catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
}
