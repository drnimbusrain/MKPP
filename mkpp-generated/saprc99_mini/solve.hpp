#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::saprc99_mini::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_lu(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 17;
  double work[N];
  work[0] = rhs[0];
  work[1] = rhs[1];
  work[2] = rhs[2];
  work[3] = rhs[3];
  work[4] = rhs[4];
  work[5] = rhs[5];
  work[6] = rhs[6];
  work[7] = rhs[7] - lu[7 * N + 6] * work[6];
  work[8] = rhs[8] - lu[8 * N + 4] * work[4] - lu[8 * N + 7] * work[7];
  work[9] = rhs[9] - lu[9 * N + 5] * work[5] - lu[9 * N + 6] * work[6];
  work[10] = rhs[10] - lu[10 * N + 8] * work[8];
  work[11] = rhs[11] - lu[11 * N + 4] * work[4] - lu[11 * N + 9] * work[9];
  work[12] = rhs[12] - lu[12 * N + 5] * work[5] - lu[12 * N + 6] * work[6] - lu[12 * N + 9] * work[9] - lu[12 * N + 11] * work[11];
  work[13] = rhs[13];
  work[14] = rhs[14] - lu[14 * N + 8] * work[8] - lu[14 * N + 9] * work[9] - lu[14 * N + 10] * work[10] - lu[14 * N + 11] * work[11] - lu[14 * N + 12] * work[12] - lu[14 * N + 13] * work[13];
  work[15] = rhs[15] - lu[15 * N + 9] * work[9] - lu[15 * N + 10] * work[10] - lu[15 * N + 11] * work[11] - lu[15 * N + 12] * work[12] - lu[15 * N + 13] * work[13] - lu[15 * N + 14] * work[14];
  work[16] = rhs[16] - lu[16 * N + 13] * work[13] - lu[16 * N + 14] * work[14] - lu[16 * N + 15] * work[15];
  solution[16] = work[16] / lu[16 * N + 16];
  solution[15] = (work[15] - lu[15 * N + 16] * solution[16]) / lu[15 * N + 15];
  solution[14] = (work[14] - lu[14 * N + 15] * solution[15] - lu[14 * N + 16] * solution[16]) / lu[14 * N + 14];
  solution[13] = work[13] / lu[13 * N + 13];
  solution[12] = (work[12] - lu[12 * N + 13] * solution[13] - lu[12 * N + 14] * solution[14] - lu[12 * N + 15] * solution[15]) / lu[12 * N + 12];
  solution[11] = (work[11] - lu[11 * N + 12] * solution[12] - lu[11 * N + 13] * solution[13] - lu[11 * N + 14] * solution[14] - lu[11 * N + 15] * solution[15]) / lu[11 * N + 11];
  solution[10] = (work[10] - lu[10 * N + 13] * solution[13]) / lu[10 * N + 10];
  solution[9] = (work[9] - lu[9 * N + 11] * solution[11] - lu[9 * N + 12] * solution[12] - lu[9 * N + 13] * solution[13] - lu[9 * N + 15] * solution[15]) / lu[9 * N + 9];
  solution[8] = (work[8] - lu[8 * N + 10] * solution[10] - lu[8 * N + 13] * solution[13]) / lu[8 * N + 8];
  solution[7] = (work[7] - lu[7 * N + 13] * solution[13]) / lu[7 * N + 7];
  solution[6] = (work[6] - lu[6 * N + 13] * solution[13]) / lu[6 * N + 6];
  solution[5] = (work[5] - lu[5 * N + 6] * solution[6] - lu[5 * N + 9] * solution[9] - lu[5 * N + 12] * solution[12] - lu[5 * N + 13] * solution[13]) / lu[5 * N + 5];
  solution[4] = work[4] / lu[4 * N + 4];
  solution[3] = work[3] / lu[3 * N + 3];
  solution[2] = work[2] / lu[2 * N + 2];
  solution[1] = work[1] / lu[1 * N + 1];
  solution[0] = work[0] / lu[0 * N + 0];
}
}  // namespace mkpp::generated::saprc99_mini::detail
