#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::small_strato::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_plan(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 7;
  struct Entry { int column, slot; };
  static constexpr int lower_offsets[] = { 0, 0, 1, 3, 5, 8, 11, 15 };
  static constexpr int upper_offsets[] = { 0, 0, 1, 4, 7, 9, 10, 10 };
  static constexpr int diagonal[] = { 0, 2, 6, 12, 19, 25, 31 };
  static constexpr Entry lower[] = {
    { 0, 1 },
    { 0, 4 },
    { 1, 5 },
    { 1, 10 },
    { 2, 11 },
    { 1, 16 },
    { 2, 17 },
    { 3, 18 },
    { 2, 22 },
    { 3, 23 },
    { 4, 24 },
    { 2, 27 },
    { 3, 28 },
    { 4, 29 },
    { 5, 30 },
  };
  static constexpr Entry upper[] = {
    { 4, 3 },
    { 3, 7 },
    { 4, 8 },
    { 6, 9 },
    { 4, 13 },
    { 5, 14 },
    { 6, 15 },
    { 5, 20 },
    { 6, 21 },
    { 6, 26 },
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
}  // namespace mkpp::generated::small_strato::detail
