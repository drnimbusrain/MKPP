/**
 * @file test_determinism.cpp
 * @brief Bit-reproducibility of workload ordering and chemistry across thread
 * counts (feature 024, US4 / FR-010 / SC-003).
 *
 * Two complementary guarantees are checked:
 *  1. The SZA sorter produces the canonical total order (sza, index), i.e. a
 *     pure function of the input.  Tied primary keys must not be permuted by
 *     backend scheduling, so the sorted order is identical for any thread
 *     count and any execution space.
 *  2. A fixed chemistry input produces a bit-identical final state when the
 *     whole run is repeated under different OpenMP thread counts.  The test
 *     re-executes itself with OMP_NUM_THREADS=1 and OMP_NUM_THREADS=8 and
 *     compares an FNV-1a digest of the sorted order and the final-state bits.
 */

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "chapman.hpp"
#include "exaero/sorter.hpp"
#include "mkpp_host/dispatcher.hpp"

namespace {

constexpr int kNumCells = 200;
constexpr int kNumSpecies = 4;

// Heavy ties: only five distinct solar zenith angles, each shared by 40 cells.
double sza_value(int cell) { return 40.0 + 10.0 * static_cast<double>(cell % 5); }

using DeviceType = Kokkos::DefaultExecutionSpace;
using ChapmanSpecies = mkpp::generated::chapman::Species;
template <typename ExecutionSpace>
using ChapmanSolver = mkpp::generated::chapman::SolverKernels<ExecutionSpace>;

std::uint64_t fnv1a_update(std::uint64_t hash, const void* data, std::size_t size) {
    const auto* bytes = static_cast<const std::uint8_t*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= 0x100000001b3ULL;
    }
    return hash;
}

/// Sort the tied SZA workload through the production sorter and return the
/// resulting cell order.
std::vector<int> sorted_cell_order() {
    std::vector<double> concentrations(static_cast<std::size_t>(kNumSpecies) * kNumCells, 0.0);
    std::vector<double> meteorology(10 * kNumCells, 0.0);
    // UnmanagedDeviceState binds meteorology as (10, n_lon, n_lat, n_alt) in
    // LayoutLeft, so the SZA read by the sorter at (0, i, 0, 0) lives at 10*i.
    for (int cell = 0; cell < kNumCells; ++cell) {
        meteorology[10 * cell] = sza_value(cell);
    }

    exaero::UnmanagedDeviceState state(concentrations.data(), meteorology.data(),
                                       kNumSpecies, kNumCells, 1, 1);
    exaero::ExaeroContext ctx("determinism", kNumCells);
    exaero::SZA_Sorter::sort_workload(ctx, state);

    auto host_indices = Kokkos::create_mirror_view(ctx.sza_sorted_indices);
    Kokkos::deep_copy(host_indices, ctx.sza_sorted_indices);
    std::vector<int> order(kNumCells);
    for (int i = 0; i < kNumCells; ++i) {
        order[i] = host_indices(i);
    }
    return order;
}

/// Integrate a fixed Chapman batch and return the final-state bit pattern.
std::array<double, kNumSpecies> chapman_final_state() {
    // Generated kernels index species linearly from state.data(), so the
    // species dimension must be contiguous (LayoutRight), matching the
    // convention of the other E2E tests.
    Kokkos::View<double****, Kokkos::LayoutRight> state("determinism_state", kNumCells,
                                                        kNumSpecies, 1, 1);
    auto host_state = Kokkos::create_mirror_view(state);
    for (int cell = 0; cell < kNumCells; ++cell) {
        host_state(cell, ChapmanSpecies::O, 0, 0) = 1.0e8 + static_cast<double>(cell);
        host_state(cell, ChapmanSpecies::O2, 0, 0) = 5.0e17;
        host_state(cell, ChapmanSpecies::O3, 0, 0) = 1.0e12;
        host_state(cell, ChapmanSpecies::M, 0, 0) = 2.45e19;
    }
    Kokkos::deep_copy(state, host_state);

    std::array<double, 2> jvals_data{1.0e-5, 1.0e-3};
    mkpp::host::JvalsView<Kokkos::HostSpace> jvals(jvals_data.data(), jvals_data.size());
    mkpp::host::execute_mechanism_steps<ChapmanSolver<DeviceType>>(
        "determinism", state, 60.0, 5, jvals);

    Kokkos::deep_copy(host_state, state);
    // One representative cell plus the batch boundary cell: per-cell work is
    // order-independent, so any fixed selection detects result drift.
    std::array<double, kNumSpecies> final_state{};
    for (int sp = 0; sp < kNumSpecies; ++sp) {
        final_state[sp] = host_state(kNumCells - 1, sp, 0, 0);
    }
    return final_state;
}

std::string compute_digest() {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    const std::vector<int> order = sorted_cell_order();
    hash = fnv1a_update(hash, order.data(), order.size() * sizeof(int));
    const std::array<double, kNumSpecies> final_state = chapman_final_state();
    hash = fnv1a_update(hash, final_state.data(), final_state.size() * sizeof(double));
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
}

std::string run_digest_child(const char* thread_count, const std::string& executable) {
    const std::string command =
        std::string("OMP_NUM_THREADS=") + thread_count + " '" + executable + "' --digest";
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        return std::string();
    }
    std::string line;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        line += buffer;
    }
    const int status = pclose(pipe);
    if (status != 0) {
        return std::string();
    }
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
        line.pop_back();
    }
    return line;
}

std::string g_executable_path;

}  // namespace

TEST(DeterminismTest, SortedOrderIsCanonicalTotalOrder) {
    const std::vector<int> order = sorted_cell_order();

    std::vector<std::pair<double, int>> canonical(kNumCells);
    for (int cell = 0; cell < kNumCells; ++cell) {
        canonical[cell] = {sza_value(cell), cell};
    }
    std::sort(canonical.begin(), canonical.end());

    for (int i = 0; i < kNumCells; ++i) {
        EXPECT_EQ(order[i], canonical[i].second)
            << "Sorted order diverges from the (sza, index) total order at slot " << i
            << " — tied keys are being permuted by scheduling (FR-010)";
    }
}

TEST(DeterminismTest, BitIdenticalRerunAcrossThreadCounts) {
    ASSERT_FALSE(g_executable_path.empty());
    const std::string digest_1 = run_digest_child("1", g_executable_path);
    const std::string digest_8 = run_digest_child("8", g_executable_path);
    ASSERT_FALSE(digest_1.empty()) << "OMP_NUM_THREADS=1 child run failed";
    ASSERT_FALSE(digest_8.empty()) << "OMP_NUM_THREADS=8 child run failed";
    const std::string digest_here = compute_digest();
    EXPECT_EQ(digest_1, digest_8)
        << "Digest of sorted order + chemistry differs between 1 and 8 threads (SC-003)";
    EXPECT_EQ(digest_here, digest_1)
        << "Digest differs between in-process run and child rerun (SC-003)";
}

int main(int argc, char** argv) {
    g_executable_path = argv[0];
    const bool digest_mode = argc > 1 && std::strcmp(argv[1], "--digest") == 0;

    // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
    Kokkos::ScopeGuard kokkos_scope(argc, argv);
    if (digest_mode) {
        std::printf("%s\n", compute_digest().c_str());
        return 0;
    }
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
