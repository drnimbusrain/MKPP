// Feature 024 / T007: standalone compile check for the generated layout-aware
// accessor. The generated chunks address every caller buffer through
// StrideView, so the accessor itself must (a) compile inside a
// KOKKOS_INLINE_FUNCTION context, (b) reproduce the view's own strides, and
// (c) read identical logical elements whether the caller bound LayoutLeft
// (species strided by num_cells) or LayoutRight (species contiguous).
//
// Validates: FR-002 (contract codegen-accessor.md), research R1.

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

#include "chapman.hpp"
#include "chapman/accessor.hpp"

namespace {

using mkpp::generated::chapman::detail::make_stride_view;
using mkpp::generated::chapman::detail::StrideView;

constexpr int kCells = 5;
constexpr int kSpecies = 4;

// LayoutLeft: cell-major columns -> a cell's species row is strided by kCells.
// LayoutRight: species contiguous -> the same logical read must match.
TEST(AccessorCompile, LayoutLeftRowReadsMatchLogicalValues) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("left", kCells, kSpecies);
  for (int c = 0; c < kCells; ++c)
    for (int s = 0; s < kSpecies; ++s) v(c, s) = c * 100.0 + s;

  auto row = Kokkos::subview(v, 2, Kokkos::ALL());
  auto acc = make_stride_view(row);
  for (int s = 0; s < kSpecies; ++s) {
    EXPECT_DOUBLE_EQ(acc(s), 2 * 100.0 + s) << "LayoutLeft row read mismatch at species " << s;
  }
  // The accessor must carry the view's strides, not assume contiguity.
  EXPECT_EQ(acc.stride(0), static_cast<size_t>(row.stride(0)));
}

TEST(AccessorCompile, LayoutRightRowReadsMatchLogicalValues) {
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> v("right", kCells, kSpecies);
  for (int c = 0; c < kCells; ++c)
    for (int s = 0; s < kSpecies; ++s) v(c, s) = c * 100.0 + s;

  auto row = Kokkos::subview(v, 2, Kokkos::ALL());
  auto acc = make_stride_view(row);
  for (int s = 0; s < kSpecies; ++s) {
    EXPECT_DOUBLE_EQ(acc(s), 2 * 100.0 + s) << "LayoutRight row read mismatch at species " << s;
  }
}

TEST(AccessorCompile, BothLayoutsAgreeOnEveryElement) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> vl("left", kCells, kSpecies);
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> vr("right", kCells, kSpecies);
  for (int c = 0; c < kCells; ++c)
    for (int s = 0; s < kSpecies; ++s) {
      vl(c, s) = c * 100.0 + s;
      vr(c, s) = c * 100.0 + s;
    }

  auto acc_l = make_stride_view(vl);
  auto acc_r = make_stride_view(vr);
  for (int c = 0; c < kCells; ++c)
    for (int s = 0; s < kSpecies; ++s) {
      ASSERT_DOUBLE_EQ(acc_l(c, s), acc_r(c, s)) << "layouts disagree at (" << c << "," << s << ")";
      ASSERT_DOUBLE_EQ(acc_l(c, s), c * 100.0 + s);
    }
}

TEST(AccessorCompile, WritesThroughAccessorLandInUnderlyingView) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("left", kCells, kSpecies);
  auto acc = make_stride_view(v);
  acc(3, 1) = 42.0;
  EXPECT_DOUBLE_EQ(v(3, 1), 42.0);
}

// The integrator's local scratch arrays are contiguous row-major; the
// pointer+stride factory must reproduce the former flat arr[i*N + j] exactly.
TEST(AccessorCompile, StackArrayFactoryReproducesFlatIndexing) {
  constexpr int N = kSpecies;
  double local[N * N] = {};
  auto acc = make_stride_view<double, 2>(local, N, 1);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) acc(i, j) = i * N + j;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) EXPECT_DOUBLE_EQ(local[i * N + j], i * N + j);
}

// Const-relaxation conversion must compile (mutable accessor -> const accessor).
TEST(AccessorCompile, MutableConvertsToConstAccessor) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("left", kCells, kSpecies);
  v(1, 2) = 7.0;
  auto mutable_acc = make_stride_view(v);
  StrideView<const double, 2> const_acc = mutable_acc;
  EXPECT_DOUBLE_EQ(const_acc(1, 2), 7.0);
}

// The accessor must be usable from device-callable code: a KOKKOS_INLINE_FUNCTION
// kernel that reads through it compiles for the configured execution space.
TEST(AccessorCompile, UsableInsideKokkosParallelFor) {
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> v("left", kCells, kSpecies);
  Kokkos::View<double*, Kokkos::HostSpace> sums("sums", kCells);
  for (int c = 0; c < kCells; ++c)
    for (int s = 0; s < kSpecies; ++s) v(c, s) = c * 100.0 + s;

  auto v_acc = make_stride_view(v);
  auto s_acc = make_stride_view(sums);
  Kokkos::parallel_for(
      "accessor_kernel", Kokkos::RangePolicy<>(0, kCells), KOKKOS_LAMBDA(int c) {
        double total = 0.0;
        for (int s = 0; s < kSpecies; ++s) total += v_acc(c, s);
        s_acc(c) = total;
      });
  Kokkos::fence();
  for (int c = 0; c < kCells; ++c) {
    EXPECT_DOUBLE_EQ(sums(c), c * 100.0 * kSpecies + (kSpecies * (kSpecies - 1)) / 2.0);
  }
}

TEST(GeneratedKernelLayout, RatesAndJacobianAgreeAcrossLayouts) {
  using HostExecSpace = Kokkos::DefaultHostExecutionSpace;
  using Solver = mkpp::generated::chapman::SolverKernels<HostExecSpace>;
  using Species = mkpp::generated::chapman::Species;

  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> left_state("left_state", kCells, kSpecies);
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> right_state("right_state", kCells, kSpecies);
  Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HostSpace> left_rates("left_rates", kCells, kSpecies);
  Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace> right_rates("right_rates", kCells, kSpecies);
  Kokkos::View<double***, Kokkos::LayoutLeft, Kokkos::HostSpace> left_jacobian("left_jacobian", kCells, kSpecies, kSpecies);
  Kokkos::View<double***, Kokkos::LayoutRight, Kokkos::HostSpace> right_jacobian("right_jacobian", kCells, kSpecies, kSpecies);
  Kokkos::View<double*, Kokkos::HostSpace> jvals("jvals", 2);

  for (int cell = 0; cell < kCells; ++cell) {
    left_state(cell, Species::O) = right_state(cell, Species::O) = 1.0e8 * (cell + 1);
    left_state(cell, Species::O2) = right_state(cell, Species::O2) = 5.0e17 * (cell + 1);
    left_state(cell, Species::O3) = right_state(cell, Species::O3) = 1.0e12 * (cell + 1);
    left_state(cell, Species::M) = right_state(cell, Species::M) = 2.45e19 * (cell + 1);
  }
  jvals(0) = 1.0e-5;
  jvals(1) = 1.0e-3;
  Kokkos::deep_copy(left_rates, 0.0);
  Kokkos::deep_copy(right_rates, 0.0);
  Kokkos::deep_copy(left_jacobian, 0.0);
  Kokkos::deep_copy(right_jacobian, 0.0);

  Solver solver;
  for (int cell = 0; cell < kCells; ++cell) {
    auto left_state_row = Kokkos::subview(left_state, cell, Kokkos::ALL());
    auto right_state_row = Kokkos::subview(right_state, cell, Kokkos::ALL());
    auto left_rates_row = Kokkos::subview(left_rates, cell, Kokkos::ALL());
    auto right_rates_row = Kokkos::subview(right_rates, cell, Kokkos::ALL());
    auto left_jacobian_block = Kokkos::subview(left_jacobian, cell, Kokkos::ALL(), Kokkos::ALL());
    auto right_jacobian_block = Kokkos::subview(right_jacobian, cell, Kokkos::ALL(), Kokkos::ALL());

    solver.compute_rates(left_state_row, left_rates_row, jvals);
    solver.compute_rates(right_state_row, right_rates_row, jvals);
    solver.compute_jacobian(left_state_row, left_jacobian_block, jvals);
    solver.compute_jacobian(right_state_row, right_jacobian_block, jvals);
  }

  for (int cell = 0; cell < kCells; ++cell) {
    for (int species = 0; species < kSpecies; ++species) {
      EXPECT_DOUBLE_EQ(left_rates(cell, species), right_rates(cell, species))
          << "rate layout mismatch at cell " << cell << " species " << species;
      for (int column = 0; column < kSpecies; ++column) {
        EXPECT_DOUBLE_EQ(left_jacobian(cell, species, column), right_jacobian(cell, species, column))
            << "Jacobian layout mismatch at cell " << cell << " row " << species << " column " << column;
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  // Exception-safe runtime ownership; also consumes --kokkos-* tuning args (FR-013).
  Kokkos::ScopeGuard kokkos_scope(argc, argv);
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
