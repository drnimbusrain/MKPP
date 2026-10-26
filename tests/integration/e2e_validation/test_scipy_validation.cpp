#include <Kokkos_Core.hpp>

#include <cmath>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "chapman.hpp"  // Generated header

using ExecSpace = Kokkos::DefaultExecutionSpace;
using ConcentrationsView =
    Kokkos::View<double****, Kokkos::LayoutRight, typename ExecSpace::memory_space>;
using ChapmanSolver = mkpp::generated::chapman::SolverKernels<ExecSpace>;

// Minimal JSON value parser for loading reference data
// Supports: arrays of numbers, strings, nested objects (limited)
namespace {

struct ReferenceData {
    std::string mechanism;
    std::vector<std::string> species;
    std::vector<double> initial_conditions;
    std::vector<double> jvals;
    double time_start;
    double time_end;
    std::vector<double> expected_final;
};

// Skip whitespace
void skip_ws(const std::string& s, size_t& pos) {
    while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\n' || s[pos] == '\r' || s[pos] == '\t')) {
        ++pos;
    }
}

// Parse a JSON string (assumes pos is at opening quote)
std::string parse_string(const std::string& s, size_t& pos) {
    if (s[pos] != '"') return "";
    ++pos;
    std::string result;
    while (pos < s.size() && s[pos] != '"') {
        if (s[pos] == '\\') {
            ++pos;
        }
        result += s[pos];
        ++pos;
    }
    ++pos;  // skip closing quote
    return result;
}

// Parse a JSON number
double parse_number(const std::string& s, size_t& pos) {
    size_t start = pos;
    while (pos < s.size() && (std::isdigit(s[pos]) || s[pos] == '.' || s[pos] == '-' ||
                              s[pos] == '+' || s[pos] == 'e' || s[pos] == 'E')) {
        ++pos;
    }
    return std::stod(s.substr(start, pos - start));
}

// Parse a JSON array of numbers
std::vector<double> parse_number_array(const std::string& s, size_t& pos) {
    std::vector<double> result;
    if (s[pos] != '[') return result;
    ++pos;
    skip_ws(s, pos);
    while (pos < s.size() && s[pos] != ']') {
        skip_ws(s, pos);
        result.push_back(parse_number(s, pos));
        skip_ws(s, pos);
        if (s[pos] == ',') ++pos;
    }
    ++pos;  // skip ']'
    return result;
}

// Parse a JSON array of strings
std::vector<std::string> parse_string_array(const std::string& s, size_t& pos) {
    std::vector<std::string> result;
    if (s[pos] != '[') return result;
    ++pos;
    skip_ws(s, pos);
    while (pos < s.size() && s[pos] != ']') {
        skip_ws(s, pos);
        result.push_back(parse_string(s, pos));
        skip_ws(s, pos);
        if (s[pos] == ',') ++pos;
    }
    ++pos;  // skip ']'
    return result;
}

// Skip a JSON value (object, array, string, number, bool, null)
void skip_value(const std::string& s, size_t& pos) {
    skip_ws(s, pos);
    if (s[pos] == '{') {
        int depth = 1;
        ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '{') ++depth;
            else if (s[pos] == '}') --depth;
            else if (s[pos] == '"') { parse_string(s, pos); continue; }
            ++pos;
        }
    } else if (s[pos] == '[') {
        int depth = 1;
        ++pos;
        while (pos < s.size() && depth > 0) {
            if (s[pos] == '[') ++depth;
            else if (s[pos] == ']') --depth;
            else if (s[pos] == '"') { parse_string(s, pos); continue; }
            ++pos;
        }
    } else if (s[pos] == '"') {
        parse_string(s, pos);
    } else {
        // number, bool, null
        while (pos < s.size() && s[pos] != ',' && s[pos] != '}' && s[pos] != ']') {
            ++pos;
        }
    }
}

ReferenceData load_reference(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open reference file: " + path);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    ReferenceData ref;
    size_t pos = 0;
    skip_ws(content, pos);
    if (content[pos] != '{') {
        throw std::runtime_error("Invalid JSON: expected '{'");
    }
    ++pos;

    while (pos < content.size() && content[pos] != '}') {
        skip_ws(content, pos);
        if (content[pos] == '}') break;

        std::string key = parse_string(content, pos);
        skip_ws(content, pos);
        if (content[pos] == ':') ++pos;
        skip_ws(content, pos);

        if (key == "mechanism") {
            ref.mechanism = parse_string(content, pos);
        } else if (key == "species") {
            ref.species = parse_string_array(content, pos);
        } else if (key == "initial_conditions") {
            ref.initial_conditions = parse_number_array(content, pos);
        } else if (key == "jvals") {
            ref.jvals = parse_number_array(content, pos);
        } else if (key == "time_span") {
            auto span = parse_number_array(content, pos);
            ref.time_start = span[0];
            ref.time_end = span[1];
        } else if (key == "expected_final") {
            ref.expected_final = parse_number_array(content, pos);
        } else {
            skip_value(content, pos);
        }

        skip_ws(content, pos);
        if (content[pos] == ',') ++pos;
    }

    return ref;
}

}  // namespace

// Functor to run integration on a single cell
struct ScipyValidationFunctor {
    ConcentrationsView state;
    double dt;
    Kokkos::View<const double*, typename ExecSpace::memory_space> jvals_view;

    ScipyValidationFunctor(ConcentrationsView s, double dt_,
                           Kokkos::View<const double*, typename ExecSpace::memory_space> jv)
        : state(s), dt(dt_), jvals_view(jv) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const Kokkos::TeamPolicy<ExecSpace>::member_type& team) const {
        int i = team.league_rank();

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, 1), [&](const int&) {
            auto cell_state = Kokkos::subview(state, i, Kokkos::ALL(), 0, 0);

            ChapmanSolver solver;
            solver.integrate(dt, cell_state, jvals_view);
        });
    }
};

TEST(ScipyValidation, ChapmanIntegration) {
    // Load reference data
    const std::string ref_path = REFERENCE_DATA_DIR "/chapman_reference.json";
    ReferenceData ref;
    try {
        ref = load_reference(ref_path);
    } catch (const std::exception& e) {
        FAIL() << "Failed to load reference data: " << e.what()
               << "\nExpected file at: " << ref_path;
    }

    // Validate reference data dimensions
    ASSERT_EQ(ref.initial_conditions.size(), 4u)
        << "Expected 4 species (O, O2, O3, M) in initial conditions";
    ASSERT_EQ(ref.expected_final.size(), 4u)
        << "Expected 4 species in expected_final";
    ASSERT_EQ(ref.jvals.size(), 2u)
        << "Expected 2 photolysis rates (jvals)";

    // Initialize state vector for a single cell (4D view: 1 cell x 4 species x 1 x 1)
    const int num_cells = 1;
    const int num_species = 4;
    std::vector<double> host_data(num_cells * num_species * 1 * 1, 0.0);
    for (int s = 0; s < num_species; ++s) {
        host_data[s] = ref.initial_conditions[s];
    }

    ConcentrationsView state(host_data.data(), num_cells, num_species, 1, 1);

    // Prepare jvals in the memory space the kernels execute in
    double jvals_data[2] = {ref.jvals[0], ref.jvals[1]};
    Kokkos::View<const double*, typename ExecSpace::memory_space> jvals(jvals_data, 2);

    // Compute integration time
    double dt = ref.time_end - ref.time_start;

    // Run the solver
    Kokkos::parallel_for("ScipyValidation_Chapman",
                         Kokkos::TeamPolicy<ExecSpace>(num_cells, Kokkos::AUTO),
                         ScipyValidationFunctor(state, dt, jvals));
    Kokkos::fence();

    // Validation tolerances
    const double rtol = 1e-4;
    const double atol = 1.0;  // molecules/cm3

    // Compare final concentrations against SciPy reference
    bool all_passed = true;
    std::ostringstream error_report;
    error_report << "\n--- Per-species error report ---\n";
    error_report << std::scientific;

    for (int s = 0; s < num_species; ++s) {
        double computed = host_data[s];
        double expected = ref.expected_final[s];
        double abs_err = std::fabs(computed - expected);
        double rel_err = (expected != 0.0) ? abs_err / std::fabs(expected) : abs_err;

        // Tolerance: max(atol, rtol * |expected|)
        double tolerance = std::fmax(atol, rtol * std::fabs(expected));

        error_report << "  Species " << s << " (" << ref.species[s] << "): "
                     << "computed=" << computed << ", expected=" << expected
                     << ", abs_err=" << abs_err << ", rel_err=" << rel_err
                     << ", tolerance=" << tolerance;

        if (abs_err > tolerance) {
            error_report << " [FAIL]";
            all_passed = false;
        } else {
            error_report << " [PASS]";
        }
        error_report << "\n";

        EXPECT_NEAR(computed, expected, tolerance)
            << "Species " << s << " (" << ref.species[s] << ") diverged from SciPy reference.\n"
            << "  Computed:  " << computed << "\n"
            << "  Expected:  " << expected << "\n"
            << "  Abs error: " << abs_err << "\n"
            << "  Rel error: " << rel_err << "\n"
            << "  Tolerance: " << tolerance << " (max(atol=" << atol
            << ", rtol=" << rtol << " * |expected|=" << rtol * std::fabs(expected) << "))";
    }

    if (!all_passed) {
        std::cerr << error_report.str();
    }
}

int main(int argc, char** argv) {
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
