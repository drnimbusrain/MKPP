#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::saprc99_mini::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_plan(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 17;
  struct Entry { int column, slot; };
  static constexpr int lower_offsets[] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 6, 8, 12, 12, 18, 24, 27 };
  static constexpr int upper_offsets[] = { 0, 0, 0, 0, 0, 0, 4, 5, 6, 8, 12, 13, 17, 20, 20, 22, 23, 23 };
  static constexpr int diagonal[] = { 0, 1, 2, 3, 4, 5, 10, 13, 17, 22, 28, 32, 41, 45, 52, 61, 66 };
  static constexpr Entry lower[] = {
    { 6, 12 },
    { 4, 15 },
    { 7, 16 },
    { 5, 20 },
    { 6, 21 },
    { 8, 27 },
    { 4, 30 },
    { 9, 31 },
    { 5, 37 },
    { 6, 38 },
    { 9, 39 },
    { 11, 40 },
    { 8, 46 },
    { 9, 47 },
    { 10, 48 },
    { 11, 49 },
    { 12, 50 },
    { 13, 51 },
    { 9, 55 },
    { 10, 56 },
    { 11, 57 },
    { 12, 58 },
    { 13, 59 },
    { 14, 60 },
    { 13, 63 },
    { 14, 64 },
    { 15, 65 },
  };
  static constexpr Entry upper[] = {
    { 6, 6 },
    { 9, 7 },
    { 12, 8 },
    { 13, 9 },
    { 13, 11 },
    { 13, 14 },
    { 10, 18 },
    { 13, 19 },
    { 11, 23 },
    { 12, 24 },
    { 13, 25 },
    { 15, 26 },
    { 13, 29 },
    { 12, 33 },
    { 13, 34 },
    { 14, 35 },
    { 15, 36 },
    { 13, 42 },
    { 14, 43 },
    { 15, 44 },
    { 15, 53 },
    { 16, 54 },
    { 16, 62 },
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
}  // namespace mkpp::generated::saprc99_mini::detail
