#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <algorithm>
#include <string>
#include <vector>

#include "chapman.hpp"
#include "mkpp.h"

namespace {

constexpr std::size_t kCellCount = 5;
using ExecutionSpace = Kokkos::DefaultExecutionSpace;
using MemorySpace = typename ExecutionSpace::memory_space;
using StateView = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using RatesView = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using JacobianView = Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>;
using Solver = mkpp::generated::chapman::SolverKernels<ExecutionSpace>;

std::size_t state_index(std::size_t cell, std::size_t species, std::size_t cell_count) {
  return cell + species * cell_count;
}

std::size_t jacobian_index(std::size_t cell, std::size_t row, std::size_t column, std::size_t cell_count) {
  return cell + row * cell_count + column * cell_count * MKPP_NUM_SPECIES;
}

void initialize_state(std::vector<double>& state, std::size_t cell_count) {
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    state[state_index(cell, mkpp::generated::chapman::Species::O, cell_count)] = 1.0e8 * (cell + 1);
    state[state_index(cell, mkpp::generated::chapman::Species::O2, cell_count)] = 5.0e17 * (cell + 1);
    state[state_index(cell, mkpp::generated::chapman::Species::O3, cell_count)] = 1.0e12 * (cell + 1);
    state[state_index(cell, mkpp::generated::chapman::Species::M, cell_count)] = 2.45e19 * (cell + 1);
  }
}

void check_c_abi_layout(std::size_t cell_count) {
  std::vector<double> concentration(cell_count * MKPP_NUM_SPECIES);
  initialize_state(concentration, cell_count);
  std::vector<double> expected_state = concentration;
  std::vector<double> temperature(cell_count, 298.15);
  std::vector<double> pressure(cell_count, 101325.0);
  std::vector<double> density(cell_count, 2.45e19);
  std::vector<double> photolysis(cell_count * std::max<std::size_t>(MKPP_NUM_PHOTOLYSIS, 1), 1.0e-5);
  std::vector<double> rates(cell_count * MKPP_NUM_SPECIES, 0.0);
  std::vector<double> expected_rates(rates.size(), 0.0);
  std::vector<double> jacobian(cell_count * MKPP_NUM_SPECIES * MKPP_NUM_SPECIES, 0.0);
  std::vector<double> expected_jacobian(jacobian.size(), 0.0);

  mkpp_handle_t handle = nullptr;
  ASSERT_EQ(mkpp_create_handle(&handle, cell_count), MKPP_SUCCESS);
  ASSERT_EQ(mkpp_initialize(handle), MKPP_SUCCESS);
  ASSERT_EQ(mkpp_set_state_ptrs(handle, concentration.data(), temperature.data(), pressure.data(), density.data()),
            MKPP_SUCCESS);
  ASSERT_EQ(mkpp_set_photolysis_ptrs(handle, photolysis.data()), MKPP_SUCCESS);

  Kokkos::View<double*, MemorySpace> jvals("layout_test_jvals", MKPP_NUM_PHOTOLYSIS);
  Kokkos::deep_copy(jvals, 1.0e-5);
  StateView reference_state(expected_state.data(), cell_count, MKPP_NUM_SPECIES);
  RatesView reference_rates(expected_rates.data(), cell_count, MKPP_NUM_SPECIES);
  JacobianView reference_jacobian(expected_jacobian.data(), cell_count, MKPP_NUM_SPECIES, MKPP_NUM_SPECIES);
  Solver solver;
  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    auto state_row = Kokkos::subview(reference_state, cell, Kokkos::ALL());
    solver.integrate(1.0, state_row, jvals);
    auto rates_row = Kokkos::subview(reference_rates, cell, Kokkos::ALL());
    auto jacobian_block = Kokkos::subview(reference_jacobian, cell, Kokkos::ALL(), Kokkos::ALL());
    solver.compute_rates(state_row, rates_row, jvals);
    solver.compute_jacobian(state_row, jacobian_block, jvals);
  }

  ASSERT_EQ(mkpp_integrate(handle, 1.0), MKPP_SUCCESS);
  ASSERT_EQ(mkpp_evaluate_rates(handle, rates.data()), MKPP_SUCCESS);
  ASSERT_EQ(mkpp_evaluate_jacobian(handle, jacobian.data()), MKPP_SUCCESS);

  for (std::size_t cell = 0; cell < cell_count; ++cell) {
    for (std::size_t species = 0; species < MKPP_NUM_SPECIES; ++species) {
      const std::size_t state_offset = state_index(cell, species, cell_count);
      EXPECT_NEAR(concentration[state_offset], expected_state[state_offset],
                  1.0e-12 * std::max(std::abs(expected_state[state_offset]), 1.0));
      EXPECT_DOUBLE_EQ(rates[state_offset], expected_rates[state_offset]);
      for (std::size_t column = 0; column < MKPP_NUM_SPECIES; ++column) {
        const std::size_t index = jacobian_index(cell, species, column, cell_count);
        EXPECT_NEAR(jacobian[index], expected_jacobian[index],
              1.0e-12 * std::abs(expected_jacobian[index]) + 1.0e-30);
      }
    }
  }

  EXPECT_EQ(mkpp_destroy_handle(handle), MKPP_SUCCESS);
}

TEST(CAbiLayout, IntegrationRatesAndJacobianPreserveCellMajorLayout) {
  check_c_abi_layout(5);
  check_c_abi_layout(2 * MKPP_NUM_SPECIES);
}

TEST(CAbiLayout, NullStateBindingReportsFatalDiagnostic) {
  std::vector<double> temperature(kCellCount, 298.15);
  std::vector<double> pressure(kCellCount, 101325.0);
  std::vector<double> density(kCellCount, 2.45e19);
  mkpp_handle_t handle = nullptr;
  ASSERT_EQ(mkpp_create_handle(&handle, kCellCount), MKPP_SUCCESS);

  EXPECT_EQ(mkpp_set_state_ptrs(handle, nullptr, temperature.data(), pressure.data(), density.data()),
            MKPP_ERROR_NULL_POINTER);
  char error[256] = {};
  ASSERT_EQ(mkpp_get_last_error(handle, error, sizeof(error)), MKPP_SUCCESS);
  EXPECT_NE(std::string(error).find("FATAL ERROR:"), std::string::npos);
  EXPECT_EQ(mkpp_destroy_handle(handle), MKPP_SUCCESS);
}

TEST(CAbiLayout, InvalidCallStatesReportFatalDiagnostics) {
  mkpp_handle_t handle = nullptr;
  ASSERT_EQ(mkpp_create_handle(&handle, kCellCount), MKPP_SUCCESS);
  EXPECT_EQ(mkpp_integrate(handle, 1.0), MKPP_ERROR_INVALID_STATE);
  char error[256] = {};
  ASSERT_EQ(mkpp_get_last_error(handle, error, sizeof(error)), MKPP_SUCCESS);
  EXPECT_NE(std::string(error).find("FATAL ERROR:"), std::string::npos);
  EXPECT_EQ(mkpp_evaluate_rates(handle, nullptr), MKPP_ERROR_NULL_POINTER);
  EXPECT_EQ(mkpp_get_last_error(handle, error, sizeof(error)), MKPP_SUCCESS);
  EXPECT_NE(std::string(error).find("FATAL ERROR:"), std::string::npos);
  EXPECT_EQ(mkpp_evaluate_jacobian(handle, nullptr), MKPP_ERROR_NULL_POINTER);
  EXPECT_EQ(mkpp_get_last_error(handle, error, sizeof(error)), MKPP_SUCCESS);
  EXPECT_NE(std::string(error).find("FATAL ERROR:"), std::string::npos);
  EXPECT_EQ(mkpp_destroy_handle(handle), MKPP_SUCCESS);
}

}  // namespace

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
  Kokkos::ScopeGuard kokkos_scope(argc, argv);
  return RUN_ALL_TESTS();
}
