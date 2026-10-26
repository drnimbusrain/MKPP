#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::chapman::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_plan(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 4;
  struct Entry { int column, slot; };
  static constexpr int lower_offsets[] = { 0, 0, 1, 3, 6 };
  static constexpr int upper_offsets[] = { 0, 0, 2, 3, 3 };
  static constexpr int diagonal[] = { 0, 2, 7, 12 };
  static constexpr Entry lower[] = {
    { 0, 1 },
    { 0, 5 },
    { 1, 6 },
    { 0, 9 },
    { 1, 10 },
    { 2, 11 },
  };
  static constexpr Entry upper[] = {
    { 2, 3 },
    { 3, 4 },
    { 3, 8 },
  };
  double work[N];
  for (int row = 0; row < N; ++row) {
    work[row] = rhs[row];
    for (int entry = lower_offsets[row]; entry < lower_offsets[row + 1]; ++entry)
      work[row] -= lu[lower[entry].slot] * work[lower[entry].column];
  }
  for (int row = N - 1; row >= 0; --row) {
    solution[row] = work[row];
    for (int entry = upper_offsets[row]; entry < upper_offsets[row + 1]; ++entry)
      solution[row] -= lu[upper[entry].slot] * solution[upper[entry].column];
    solution[row] /= lu[diagonal[row]];
  }
}
}  // namespace mkpp::generated::chapman::detail
