// Device runtime parity gate for the device-callable generated solver path.
//
// This target is built and run on EVERY configuration, host-only included. It
// separates two acceptance concerns so that host success is never mistaken for
// accelerator success:
//
//   * Host reference execution (always runs): integrates the Chapman mechanism
//     against the SciPy Radau reference and the GoCart aerosol mechanism against
//     a host self-consistency anchor. This records host-compiled + host-executed.
//
//   * Device parity (runs only when an accelerator is present): integrates the
//     same cases on the default device execution space and compares the result to
//     the host result within the project's established tolerances. When no
//     accelerator is available the test reports an explicit device-unverified
//     result and a non-pass (skipped) status rather than silently passing,
//     because compiling or running on the host does not demonstrate that the
//     generated call chain executes correctly on a device.
//
// The five validation states this encodes are: host-compiled, host-executed,
// device-compiled, device-executed, and device-unverified.
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <vector>

#include "chapman.hpp"
#include "gocart.hpp"

namespace {

// The host execution space and memory space are always available and are used
// for the reference integration regardless of whether a device backend exists.
using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
using HostMemorySpace = Kokkos::HostSpace;

// The default execution space is the accelerator when a device backend is
// compiled in, and a host space otherwise. A device-parity run is meaningful
// only when this space is genuinely a device space AND a device is present at
// runtime; otherwise the comparison would trivially compare a host run to
// itself and falsely report success.
constexpr bool kDefaultSpaceIsDevice =
    !Kokkos::SpaceAccessibility<Kokkos::DefaultExecutionSpace, HostMemorySpace>::accessible;

bool accelerator_available() { return kDefaultSpaceIsDevice && Kokkos::num_devices() > 0; }

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
        token.erase(std::remove_if(token.begin(), token.end(),
                                   [](unsigned char c) { return std::isspace(c); }),
                    token.end());
        if (!token.empty()) values.push_back(std::stod(token));
    }
    return values;
}

// Bind a photolysis-rate array into a memory space as a read-only view.
template <typename MemorySpace>
Kokkos::View<const double*, MemorySpace> make_jvals(const std::vector<double>& values) {
    Kokkos::View<double*, MemorySpace> scratch("jvals_scratch", values.size());
    auto host = Kokkos::create_mirror_view(scratch);
    for (std::size_t i = 0; i < values.size(); ++i) host(static_cast<int>(i)) = values[i];
    Kokkos::deep_copy(scratch, host);
    return Kokkos::View<const double*, MemorySpace>(scratch);
}

// Integrate Chapman on a given execution/memory space and return the final
// state copied back to the host. Photolysis rates come from the reference file.
template <typename ExecSpace>
std::vector<double> integrate_chapman(const std::vector<double>& initial,
                                      const std::vector<double>& jvals) {
    using MemorySpace = typename ExecSpace::memory_space;
    using Solver = mkpp::generated::chapman::SolverKernels<ExecSpace>;

    Kokkos::View<double*, Kokkos::LayoutRight, MemorySpace> state("state", initial.size());
    auto host_state = Kokkos::create_mirror_view(state);
    for (std::size_t i = 0; i < initial.size(); ++i) host_state(static_cast<int>(i)) = initial[i];
    Kokkos::deep_copy(state, host_state);

    auto jvals_view = make_jvals<MemorySpace>(jvals);

    Solver solver;
    solver.integrate(3600.0, state, jvals_view);
    Kokkos::fence();

    auto final_state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state);
    std::vector<double> result(initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) result[i] = final_state(static_cast<int>(i));
    return result;
}

// Integrate GoCart (environment-gated) on a given execution/memory space. No
// external reference exists, so the host run is the anchor for device parity.
template <typename ExecSpace>
std::vector<double> integrate_gocart(const std::vector<double>& initial,
                                     const std::vector<double>& jvals) {
    using MemorySpace = typename ExecSpace::memory_space;
    using Solver = mkpp::generated::gocart::SolverKernels<ExecSpace>;

    Kokkos::View<double*, Kokkos::LayoutRight, MemorySpace> state("state", initial.size());
    auto host_state = Kokkos::create_mirror_view(state);
    for (std::size_t i = 0; i < initial.size(); ++i) host_state(static_cast<int>(i)) = initial[i];
    Kokkos::deep_copy(state, host_state);

    auto jvals_view = make_jvals<MemorySpace>(jvals);

    Solver solver;
    solver.integrate(10.0, state, jvals_view, 298.15, 0.5, 0.0);
    Kokkos::fence();

    auto final_state = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, state);
    std::vector<double> result(initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) result[i] = final_state(static_cast<int>(i));
    return result;
}

// Compare a computed state against an anchor within the project's established
// Chapman tolerances (rtol=1e-4, atol=1 molecule/cm3).
void expect_within_tolerance(const std::vector<double>& computed,
                             const std::vector<double>& anchor, const char* label) {
    ASSERT_EQ(computed.size(), anchor.size()) << label;
    const double rtol = 1.0e-4;
    const double atol = 1.0;
    for (std::size_t i = 0; i < anchor.size(); ++i) {
        const double abs_err = std::fabs(computed[i] - anchor[i]);
        const double tolerance = std::fmax(atol, rtol * std::fabs(anchor[i]));
        EXPECT_LE(abs_err, tolerance)
            << label << " species " << i << ": got " << computed[i] << " anchor " << anchor[i]
            << " (tolerance " << tolerance << ")";
    }
}

// Shared reference data loaded once by the host test and reused by the device
// test so both compare against the identical anchor.
struct ReferenceData {
    std::vector<double> chapman_initial;
    std::vector<double> chapman_jvals;
    std::vector<double> chapman_expected;
    bool loaded = false;
};

ReferenceData g_reference;

bool load_reference() {
    if (g_reference.loaded) return true;
    const std::string ref_path = std::string(MKPP_E2E_DATA_DIR) + "/chapman_reference.json";
    std::ifstream file(ref_path);
    if (!file.is_open()) {
        std::fprintf(stderr, "FATAL ERROR: Could not open reference JSON: %s\n",
                     ref_path.c_str());
        return false;
    }
    const std::string content((std::istreambuf_iterator<char>(file)),
                              std::istreambuf_iterator<char>());
    g_reference.chapman_initial = parse_number_array(content, "initial_conditions");
    g_reference.chapman_jvals = parse_number_array(content, "jvals");
    g_reference.chapman_expected = parse_number_array(content, "expected_final");
    g_reference.loaded = g_reference.chapman_initial.size() == 4 &&
                         g_reference.chapman_jvals.size() == 2 &&
                         g_reference.chapman_expected.size() == 4;
    return g_reference.loaded;
}

}  // namespace

// Host reference execution: records host-compiled and host-executed. Always
// runs and must pass on every configuration.
TEST(DeviceRuntimeParity, HostReferenceExecutes) {
    ASSERT_TRUE(load_reference()) << "FATAL ERROR: Chapman reference JSON could not be parsed.";

    const auto host_chapman =
        integrate_chapman<HostExecSpace>(g_reference.chapman_initial, g_reference.chapman_jvals);
    expect_within_tolerance(host_chapman, g_reference.chapman_expected, "chapman-host");

    // GoCart exercises the environment-gated entry point (temperature, RH,
    // cloud water) so the host path covers mechanisms with external inputs too.
    const std::vector<double> gocart_initial(
        mkpp::generated::gocart::SolverKernels<HostExecSpace>::NUM_SPECIES, 1.0e8);
    const std::vector<double> gocart_jvals(
        mkpp::generated::gocart::SolverKernels<HostExecSpace>::NUM_PHOTOLYSIS, 1.0e-4);
    const auto host_gocart = integrate_gocart<HostExecSpace>(gocart_initial, gocart_jvals);
    for (std::size_t i = 0; i < host_gocart.size(); ++i) {
        ASSERT_TRUE(std::isfinite(host_gocart[i])) << "gocart-host species " << i << " is not finite";
    }
}

// Device parity: records device-compiled and device-executed when an
// accelerator is present, otherwise reports device-unverified as a non-pass.
TEST(DeviceRuntimeParity, DeviceMatchesHost) {
    if (!accelerator_available()) {
        GTEST_SKIP() << "MKPP device parity: no accelerator execution environment available "
                        "-> device-unverified (host success does not certify device execution)";
    }

    ASSERT_TRUE(load_reference());

    // Chapman: device result must reproduce the SciPy reference within tolerance.
    const auto device_chapman =
        integrate_chapman<Kokkos::DefaultExecutionSpace>(g_reference.chapman_initial,
                                                         g_reference.chapman_jvals);
    expect_within_tolerance(device_chapman, g_reference.chapman_expected, "chapman-device");

    // GoCart: device result must match the host result within tolerance.
    const std::vector<double> gocart_initial(
        mkpp::generated::gocart::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES, 1.0e8);
    const std::vector<double> gocart_jvals(
        mkpp::generated::gocart::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_PHOTOLYSIS,
        1.0e-4);
    const auto host_gocart = integrate_gocart<HostExecSpace>(gocart_initial, gocart_jvals);
    const auto device_gocart =
        integrate_gocart<Kokkos::DefaultExecutionSpace>(gocart_initial, gocart_jvals);
    expect_within_tolerance(device_gocart, host_gocart, "gocart-device-vs-host");
}

// Emits the machine-readable validation-state summary consumed by the test
// harness and CI. The tokens are: host-compiled, host-executed,
// device-compiled, device-executed, device-unverified. When no accelerator is
// present the process exits with the dedicated device-unverified code (2) so
// ctest records an explicit non-pass result instead of a silent skip.
constexpr int kDeviceUnverifiedExitCode = 2;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    // Exception-safe runtime ownership; guard finalizes when main returns,
    // after the state tokens below are emitted (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    const int result = RUN_ALL_TESTS();
    const bool device_present = accelerator_available();

    std::printf("MKPP_VALIDATION_STATES host-compiled host-executed");
    if (device_present) {
        // Reaching here with an accelerator means the device instantiation both
        // compiled and executed; correctness is asserted by the parity test.
        std::printf(" device-compiled device-executed\n");
    } else {
        std::printf(" device-unverified\n");
    }
    std::fflush(stdout);

    // A genuine assertion failure always wins: collapse any non-zero test
    // result to a plain failure code so it is never masked by the
    // device-unverified marker. Only a clean host run with no accelerator is
    // downgraded to the dedicated device-unverified exit code.
    if (result != 0) return 1;
    if (!device_present) return kDeviceUnverifiedExitCode;
    return 0;
}
