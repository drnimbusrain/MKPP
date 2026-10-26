// Host-space parity gate for the device-callable generated solver path.
//
// Binds photolysis rates as a host-space Kokkos::View (the same memory-space
// contract a device instantiation uses) and integrates the Chapman mechanism
// against the SciPy Radau reference in chapman_reference.json — the same
// rtol=1e-12 anchor validated by mkpp_scipy_validation. This proves the
// view-based jvals entry points reproduce the reference numerics on the host
// after the breaking API change.
//
// Note: kpp_baseline_chapman.csv was considered as the anchor but its final
// concentration rows are internally inconsistent with the Jacobian rows in the
// same file (the Jacobian implies J3=1e-3, under which O3 cannot remain at
// 2.757e10), and the E2ECompare target path never compares concentrations.
// The SciPy reference JSON is the validated source of truth.
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "chapman.hpp"

namespace {

using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = ExecSpace::memory_space;

// Extract a JSON array of numbers following "key": [ ... ].
std::vector<double> parse_number_array(const std::string& content, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    auto pos = content.find(needle);
    if (pos == std::string::npos) return {};
    pos = content.find('[', pos);
    if (pos == std::string::npos) return {};
    const auto end = content.find(']', pos);
    if (end == std::string::npos) return {};

    std::vector<double> values;
    std::stringstream ss(content.substr(pos + 1, end - pos - 1));
    std::string token;
    while (std::getline(ss, token, ',')) {
        // Strip whitespace and newlines around the token.
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char c) { return std::isspace(c); }),
                    token.end());
        if (!token.empty()) values.push_back(std::stod(token));
    }
    return values;
}

}  // namespace

TEST(DeviceCallabilityHostParity, ChapmanViewBoundJvalsMatchesSciPyReference) {
    const std::string ref_path = std::string(MKPP_E2E_DATA_DIR) + "/chapman_reference.json";
    std::ifstream file(ref_path);
    ASSERT_TRUE(file.is_open()) << "FATAL ERROR: Could not open reference JSON: " << ref_path;
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());

    const auto initial = parse_number_array(content, "initial_conditions");
    const auto jv = parse_number_array(content, "jvals");
    const auto expected = parse_number_array(content, "expected_final");
    ASSERT_EQ(initial.size(), 4u);
    ASSERT_EQ(jv.size(), 2u);
    ASSERT_EQ(expected.size(), 4u);

    // Host-space arrays bound as views — the memory-space contract the
    // device-callable entry points require.
    std::vector<double> host_data = initial;
    std::vector<double> host_jvals = jv;

    Kokkos::View<double*, Kokkos::LayoutRight, MemorySpace> state(host_data.data(),
                                                                  host_data.size());
    Kokkos::View<const double*, MemorySpace> jvals(host_jvals.data(), host_jvals.size());

    mkpp::generated::chapman::SolverKernels<ExecSpace> solver;
    solver.integrate(3600.0, state, jvals);
    Kokkos::fence();

    auto final_state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state);

    // Same tolerances as mkpp_scipy_validation: rtol=1e-4, atol=1 molecule/cm3.
    const double rtol = 1.0e-4;
    const double atol = 1.0;
    const std::vector<std::string> names = {"O", "O2", "O3", "M"};
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double computed = final_state(static_cast<int>(i));
        const double abs_err = std::fabs(computed - expected[i]);
        const double tolerance = std::fmax(atol, rtol * std::fabs(expected[i]));
        EXPECT_LE(abs_err, tolerance)
            << names[i] << ": got " << computed << " expected " << expected[i]
            << " (tolerance " << tolerance << ")";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    return RUN_ALL_TESTS();
}
