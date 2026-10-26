/** Governed native MKPP runner for canonical Chapman and TS1 mechanisms. */

#include <Kokkos_Core.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <mkpp_host/mechanism_manager.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "ts1.hpp"
#include "ts1_binding_data.hpp"

namespace {

using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using StateView = Kokkos::View<double**, Kokkos::LayoutLeft, typename ExecutionSpace::memory_space>;
using Clock = std::chrono::steady_clock;

constexpr std::string_view solver_id = "mkpp";
constexpr std::string_view default_mechanism_id = "chapman";
constexpr std::array<std::string_view, 4> species_names{"O", "O2", "O3", "M"};
constexpr std::array<std::string_view, 2> photolysis_names{"J1", "J3"};

struct Options {
    std::string mechanism{default_mechanism_id};
    int cells{1};
    int steps{10};
    int warmups{1};
    int repetitions{1};
    double dt{60.0};
    double j1{1.0e-12};
    double j3{1.0e-4};
    std::array<double, 4> initial_state{2.5e9, 5.1e18, 8.0e11, 2.46e19};
    std::string campaign_id{"standalone"};
    std::string run_id{"chapman-mkpp"};
    std::string manifest_version{"1.0.0"};
    std::string manifest_sha256 = std::string(64, '0');
    std::string scenario_id{"chapman-measurement-v1"};
    std::string scenario_version{"1.0.0"};
    std::string scenario_sha256 = std::string(64, '0');
    std::string configuration_id{"mkpp-ros3"};
    std::string configuration_sha256 = std::string(64, '0');
    int block{0};
    int order_position{0};
};

[[noreturn]] void fatal(const std::string& message) {
    throw std::runtime_error("FATAL ERROR: " + message);
}

int parse_positive_int(const char* value, std::string_view name) {
    const int parsed = std::stoi(value);
    if (parsed < 1) fatal(std::string(name) + " must be positive");
    return parsed;
}

double parse_finite_double(const char* value, std::string_view name) {
    const double parsed = std::stod(value);
    if (!Kokkos::isfinite(parsed)) fatal(std::string(name) + " must be finite");
    return parsed;
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) fatal("option '" + argument + "' requires a value");
        const char* value = argv[++index];
        if (argument == "--mechanism")
            options.mechanism = value;
        else if (argument == "--cells")
            options.cells = parse_positive_int(value, "cells");
        else if (argument == "--steps")
            options.steps = parse_positive_int(value, "steps");
        else if (argument == "--warmups")
            options.warmups = parse_positive_int(value, "warmups");
        else if (argument == "--repetitions")
            options.repetitions = parse_positive_int(value, "repetitions");
        else if (argument == "--dt")
            options.dt = parse_finite_double(value, "dt");
        else if (argument == "--J1")
            options.j1 = parse_finite_double(value, "J1");
        else if (argument == "--J3")
            options.j3 = parse_finite_double(value, "J3");
        else if (argument == "--O")
            options.initial_state[0] = parse_finite_double(value, "O");
        else if (argument == "--O2")
            options.initial_state[1] = parse_finite_double(value, "O2");
        else if (argument == "--O3")
            options.initial_state[2] = parse_finite_double(value, "O3");
        else if (argument == "--M")
            options.initial_state[3] = parse_finite_double(value, "M");
        else if (argument == "--campaign-id")
            options.campaign_id = value;
        else if (argument == "--run-id")
            options.run_id = value;
        else if (argument == "--manifest-version")
            options.manifest_version = value;
        else if (argument == "--manifest-sha256")
            options.manifest_sha256 = value;
        else if (argument == "--scenario-id")
            options.scenario_id = value;
        else if (argument == "--scenario-version")
            options.scenario_version = value;
        else if (argument == "--scenario-sha256")
            options.scenario_sha256 = value;
        else if (argument == "--configuration-id")
            options.configuration_id = value;
        else if (argument == "--configuration-sha256")
            options.configuration_sha256 = value;
        else if (argument == "--block")
            options.block = std::stoi(value);
        else if (argument == "--order-position")
            options.order_position = std::stoi(value);
        else
            fatal("unknown option '" + argument + "'");
    }
    if (options.dt <= 0.0) fatal("dt must be positive");
    if (options.mechanism != "chapman" && options.mechanism != "ts1") {
        fatal("mechanism must be 'chapman' or 'ts1'");
    }
    if (options.block < 0 || options.order_position < 0)
        fatal("block and order position must be nonnegative");
    for (std::size_t index = 0; index < options.initial_state.size(); ++index) {
        if (options.initial_state[index] < 0.0) {
            fatal("initial concentration for " + std::string(species_names[index]) +
                  " must be nonnegative");
        }
    }
    return options;
}

std::string json_escape(std::string_view value) {
    std::ostringstream escaped;
    for (const char character : value) {
        switch (character) {
            case '"':
                escaped << "\\\"";
                break;
            case '\\':
                escaped << "\\\\";
                break;
            case '\n':
                escaped << "\\n";
                break;
            case '\r':
                escaped << "\\r";
                break;
            case '\t':
                escaped << "\\t";
                break;
            default:
                escaped << character;
        }
    }
    return escaped.str();
}

void reset_state(StateView state, const Options& options) {
    // The canonical species_map is name based: O->0, O2->1, O3->2, M->3.
    const std::map<std::string_view, int> species_map{
        {species_names[0], 0}, {species_names[1], 1}, {species_names[2], 2}, {species_names[3], 3}};
    auto host_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(host_state, 0.0);
    if (options.mechanism == "ts1") {
        for (int cell = 0; cell < options.cells; ++cell) {
            for (std::size_t index = 0; index < ts1_binding::initial_state.size(); ++index) {
                host_state(cell, index) = ts1_binding::initial_state[index];
            }
        }
        Kokkos::deep_copy(state, host_state);
        Kokkos::fence("synchronize TS1 reset before solve");
        return;
    }
    for (int cell = 0; cell < options.cells; ++cell) {
        for (const auto& [name, species_index] : species_map) {
            static_cast<void>(name);
            host_state(cell, species_index) = options.initial_state[species_index];
        }
    }
    Kokkos::deep_copy(state, host_state);
    Kokkos::fence("synchronize reset before solve");
}

Clock::time_point start_timing() {
    return Clock::now();
}

double stop_timing(const Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

void synchronize_completion() {
    Kokkos::fence("synchronize steady-state solve completion");
}

void execute_solve(StateView state, const Options& options, const mkpp::host::JvalsView<typename ExecutionSpace::memory_space>& jvals) {
    mkpp::host::HostExecutionParams parameters;
    parameters.dt = options.dt;
    parameters.jvals = jvals;
    parameters.jvals_extent = jvals.extent(0);
    parameters.steps = options.steps;
    if (options.mechanism == "ts1") {
        parameters.temp = 287.45;
        parameters.air_density = 42.393400541636808;
    }
    // Dispatch is deliberately by the governed mechanism name. In particular, TS1
    // reaches the generated native kernel through the host registry, never a
    // Chapman-shaped placeholder or a raw species-index adapter.
    mkpp::host::MechanismRegistry::execute(options.mechanism, state, parameters);
}

std::vector<double> copy_final_state(StateView state) {
    auto host_state = Kokkos::create_mirror_view(state);
    Kokkos::deep_copy(host_state, state);
    std::vector<double> result(state.extent(1));
    for (std::size_t index = 0; index < result.size(); ++index)
        result[index] = host_state(0, index);
    return result;
}

void emit_json_result(const Options& options, double elapsed_ms,
                      const std::vector<double>& final_state) {
    const bool ts1_schedule = options.mechanism == "ts1";
    const long long cell_steps = static_cast<long long>(options.cells) *
        (ts1_schedule ? 2 : options.steps) * options.repetitions;
    const double final_time = ts1_schedule ? 600.0 : options.steps * options.dt;
    std::cout
        << std::setprecision(17) << "{\"schema_version\":1,\"campaign_id\":\""
        << json_escape(options.campaign_id) << "\",\"run_id\":\"" << json_escape(options.run_id)
        << "\",\"block\":" << options.block << ",\"order_position\":" << options.order_position
        << ",\"solver\":{\"id\":\"mkpp\",\"version\":\"0.1.0\",\"revision\":\"workspace\","
           "\"backend\":\"Kokkos::DefaultExecutionSpace\",\"method\":\"ros3\"}"
        << ",\"manifest\":{\"id\":\"" << json_escape(options.mechanism) << "\",\"version\":\""
        << json_escape(options.manifest_version) << "\",\"sha256\":\""
        << json_escape(options.manifest_sha256) << "\"}"
        << ",\"scenario\":{\"id\":\"" << json_escape(options.scenario_id) << "\",\"version\":\""
        << json_escape(options.scenario_version) << "\",\"sha256\":\""
        << json_escape(options.scenario_sha256) << "\"}"
        << ",\"configuration\":{\"id\":\"" << json_escape(options.configuration_id)
        << "\",\"sha256\":\"" << json_escape(options.configuration_sha256)
        << "\",\"frozen\":true,\"controls\":{\"dt\":" << options.dt << "}}"
        << ",\"build\":{\"binary_sha256\":\"unavailable-at-runtime\",\"asset_sha256\":[],"
           "\"compiler\":\""
#if defined(__clang__)
        << "clang"
#elif defined(__GNUC__)
        << "gcc"
#else
        << "unknown"
#endif
        << "\",\"flags\":[],\"precision\":\"float64\"}"
        << ",\"resources\":{\"hardware_id\":\"runtime-host\",\"physical_cores\":1,\"threads\":1,"
           "\"affinity\":\"external-policy\",\"numa_policy\":\"external-policy\"}"
        << ",\"timing\":{\"boundary\":\"steady_state_solve\",\"elapsed_ms\":" << elapsed_ms
        << ",\"cell_steps\":" << cell_steps
        << ",\"clock\":\"std::chrono::steady_clock\",\"synchronized\":true,"
           "\"synchronization\":\"Kokkos::fence\",\"lifecycle_ms\":{}}"
        << ",\"state\":{\"checkpoints\":[{\"time_seconds\":" << final_time
        << ",\"values\":{";
    if (options.mechanism == "ts1") {
        for (std::size_t index = 0; index < ts1_binding::species_names.size(); ++index) {
            if (index != 0) std::cout << ',';
            std::cout << '"' << json_escape(ts1_binding::species_names[index])
                      << "\":" << final_state[index];
        }
    } else {
        std::cout << "\"O\":" << final_state[0] << ",\"O2\":" << final_state[1]
                  << ",\"O3\":" << final_state[2] << ",\"M\":" << final_state[3];
    }
    std::cout << "}}]}"
              << ",\"work\":{\"internal_steps\":{\"available\":false,"
                 "\"reason\":\"generated MKPP kernel does not expose this counter\"}}"
              << ",\"status\":\"success\",\"diagnostics\":[]}" << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // Own the Kokkos runtime lifetime through an exception-safe scope
        // guard. Constructing it before the benchmark's own argument parsing
        // lets Kokkos consume its documented runtime-tuning arguments
        // (--kokkos-num-threads, --kokkos-device-id, ...) and shrink argc so
        // parse_options only sees benchmark flags. The guard finalizes on every
        // exit path, including a throw from argument parsing, so the catch
        // handler no longer needs a manual finalize (FR-013).
        Kokkos::ScopeGuard kokkos_scope(argc, argv);
        const Options options = parse_options(argc, argv);
        {
            if (!mkpp::host::MechanismRegistry::has_mechanism(options.mechanism)) {
                fatal("MKPP " + options.mechanism +
                      " mechanism is not compiled into the host registry");
            }
            const int expected_species = options.mechanism == "ts1" ? 210 : 4;
            if (mkpp::host::MechanismRegistry::get_info(options.mechanism).num_species !=
                expected_species) {
                fatal("MKPP " + options.mechanism +
                      " species count does not match canonical mapping");
            }

            StateView state("mkpp_benchmark_state", options.cells, expected_species);
            // Generated kernels address the supplied rate array by reaction slot.
            // TS1's 123 named photolysis inputs are audited in ts1_binding.json;
            // the full 547 slots also cover its non-photolysis generated rates.
            const std::size_t forcing_count = options.mechanism == "ts1"
                                                  ? ts1_binding::reaction_parameters.size()
                                                  : photolysis_names.size();
            Kokkos::View<double*, typename ExecutionSpace::memory_space> jvals("named_forcing",
                                                                               forcing_count);
            auto host_jvals = Kokkos::create_mirror_view(jvals);
            Kokkos::deep_copy(host_jvals, 0.0);
            if (options.mechanism == "chapman") {
                host_jvals(0) = options.j1;  // J1
                host_jvals(1) = options.j3;  // J3
            } else {
                for (std::size_t index = 0; index < ts1_binding::reaction_parameters.size();
                     ++index) {
                    host_jvals(index) = ts1_binding::reaction_parameters[index];
                }
            }
            Kokkos::deep_copy(jvals, host_jvals);

            const auto execute_ts1_schedule = [&]() {
                constexpr std::array<double, 2> interval_seconds{300.0, 300.0};
                constexpr std::array<double, 2> photolysis_scale{1.0, 1.05};
                for (std::size_t segment = 0; segment < interval_seconds.size(); ++segment) {
                    for (std::size_t slot = 0; slot < 123; ++slot)
                        host_jvals(slot) = ts1_binding::reaction_parameters[slot] * photolysis_scale[segment];
                    Kokkos::deep_copy(jvals, host_jvals);
                    Options segment_options = options;
                    segment_options.dt = interval_seconds[segment];
                    segment_options.steps = 1;
                    execute_solve(state, segment_options, jvals);
                }
            };

            // Audit mode is intentionally limited to the common initial state.  It
            // lets the comparison harness distinguish a chemistry translation
            // mismatch from an integration-control mismatch without timing either
            // solver.
            if (options.mechanism == "ts1" && std::getenv("EMIT_INITIAL_RHS") != nullptr) {
                reset_state(state, options);
                StateView rhs("mkpp_initial_rhs", options.cells, expected_species);
                Kokkos::parallel_for(
                    "evaluate MKPP TS1 initial RHS", Kokkos::RangePolicy<ExecutionSpace>(0, options.cells),
                    KOKKOS_LAMBDA(const int cell) {
                        auto cell_state = Kokkos::subview(state, cell, Kokkos::ALL());
                        auto cell_rhs = Kokkos::subview(rhs, cell, Kokkos::ALL());
                        mkpp::generated::ts1::SolverKernels<ExecutionSpace>{}.compute_rates(
                            cell_state, cell_rhs, jvals);
                    });
                synchronize_completion();
                const auto initial_rhs = copy_final_state(rhs);
                std::cout << std::setprecision(17) << '{';
                for (std::size_t index = 0; index < ts1_binding::species_names.size(); ++index) {
                    if (index != 0) std::cout << ',';
                    std::cout << '"' << ts1_binding::species_names[index] << "\":" << initial_rhs[index];
                }
                std::cout << "}\n";
                return EXIT_SUCCESS;
            }

            for (int warmup = 0; warmup < options.warmups; ++warmup) {
            reset_state(state, options);
            if (options.mechanism == "ts1") execute_ts1_schedule();
            else execute_solve(state, options, jvals);
                synchronize_completion();
            }

            double elapsed_ms = 0.0;
            for (int repetition = 0; repetition < options.repetitions; ++repetition) {
                reset_state(state, options);  // Reset is deliberately outside every timed sample.
                const auto started = start_timing();
                if (options.mechanism == "ts1") execute_ts1_schedule();
                else execute_solve(state, options, jvals);
                synchronize_completion();
                elapsed_ms += stop_timing(started);
            }
            const auto final_state = copy_final_state(state);
            for (std::size_t index = 0; index < final_state.size(); ++index) {
                if (!std::isfinite(final_state[index])) {
                    fatal("MKPP TS1 solve produced a non-finite concentration at species index " +
                          std::to_string(index));
                }
            }
            emit_json_result(options, elapsed_ms, final_state);
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        // The scope guard declared in the try block owns finalization and is
        // destroyed during unwinding, so no manual finalize is needed here.
        std::cerr << (std::string_view(error.what()).starts_with("FATAL ERROR:") ? ""
                                                                                 : "FATAL ERROR: ")
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
