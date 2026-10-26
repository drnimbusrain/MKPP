#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>

#include "chapman.hpp"

int main() {
  constexpr int N = mkpp::generated::chapman::SolverKernels<Kokkos::DefaultExecutionSpace>::NUM_SPECIES;
  // Chapman stores its sparse matrix in 13 plan-specific slots: diagonal
  // entries plus the structurally nonzero lower and upper entries. Keep this
  // order aligned with the generated factorize_plan/solve_plan functions.
  constexpr int COMPACT_SIZE = 13;
  std::array<double, COMPACT_SIZE> w{
      6.0, 2.0, 5.0, 1.0, 2.0, 0.0, 0.0, 4.0, 0.0, -1.0, 1.0, 2.0, 3.0};
  std::array<double, COMPACT_SIZE> candidate{};
  mkpp::generated::chapman::detail::factorize_plan(w.data(), candidate.data());
  for (int index = 0; index < COMPACT_SIZE; ++index) {
    if (!std::isfinite(candidate[index])) {
      std::fprintf(stderr, "Non-finite factorization value at %d: %.17g\n", index, candidate[index]);
      return 1;
    }
  }

  std::array<double, N> rhs{1.0, 2.0, 3.0, 4.0}, candidate_solution{};
  mkpp::generated::chapman::detail::solve_plan(candidate.data(), rhs.data(), candidate_solution.data());
  for (int index = 0; index < N; ++index) {
    if (!std::isfinite(candidate_solution[index])) {
      std::fprintf(stderr, "Non-finite solve value at %d: %.17g\n", index, candidate_solution[index]);
      return 2;
    }
  }

  constexpr std::array<double, N * N> dense_logical_matrix{
      6.0, 0.0, 0.0, 0.0,
      2.0, 5.0, 1.0, 2.0,
      0.0, 0.0, 4.0, 0.0,
      -1.0, 1.0, 2.0, 3.0};
  for (int row = 0; row < N; ++row) {
    double residual = -rhs[row];
    for (int column = 0; column < N; ++column) {
      residual += dense_logical_matrix[row * N + column] * candidate_solution[column];
    }
    if (std::abs(residual) > 1.0e-12) {
      std::fprintf(stderr, "Supernodal solve residual at row %d: %.17g\n", row, residual);
      return 3;
    }
  }
  return 0;
}
