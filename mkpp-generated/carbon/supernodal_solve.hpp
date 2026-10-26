#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::carbon::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_plan(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 11;
  struct Entry { int column, slot; };
  static constexpr int lower_offsets[] = { 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8 };
  static constexpr int upper_offsets[] = { 0, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 8 };
  static constexpr int diagonal[] = { 0, 2, 4, 5, 8, 11, 14, 17, 20, 23, 26 };
  static constexpr Entry lower[] = {
    { 2, 6 },
    { 3, 7 },
    { 4, 10 },
    { 4, 13 },
    { 6, 16 },
    { 6, 19 },
    { 8, 22 },
    { 8, 25 },
  };
  static constexpr Entry upper[] = {
    { 2, 1 },
    { 3, 3 },
    { 6, 9 },
    { 6, 12 },
    { 8, 15 },
    { 8, 18 },
    { 10, 21 },
    { 10, 24 },
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
}  // namespace mkpp::generated::carbon::detail
