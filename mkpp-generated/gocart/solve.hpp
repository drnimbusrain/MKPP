#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::gocart::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_lu(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 26;
  double work[N];
  work[0] = rhs[0];
  work[1] = rhs[1] - lu[1 * N + 0] * work[0];
  work[2] = rhs[2];
  work[3] = rhs[3] - lu[3 * N + 2] * work[2];
  work[4] = rhs[4];
  work[5] = rhs[5] - lu[5 * N + 4] * work[4];
  work[6] = rhs[6] - lu[6 * N + 4] * work[4] - lu[6 * N + 5] * work[5];
  work[7] = rhs[7] - lu[7 * N + 4] * work[4] - lu[7 * N + 5] * work[5] - lu[7 * N + 6] * work[6];
  work[8] = rhs[8] - lu[8 * N + 4] * work[4] - lu[8 * N + 5] * work[5] - lu[8 * N + 6] * work[6] - lu[8 * N + 7] * work[7];
  work[9] = rhs[9] - lu[9 * N + 4] * work[4] - lu[9 * N + 5] * work[5] - lu[9 * N + 6] * work[6] - lu[9 * N + 7] * work[7] - lu[9 * N + 8] * work[8];
  work[10] = rhs[10] - lu[10 * N + 4] * work[4] - lu[10 * N + 5] * work[5] - lu[10 * N + 6] * work[6] - lu[10 * N + 7] * work[7] - lu[10 * N + 8] * work[8] - lu[10 * N + 9] * work[9];
  work[11] = rhs[11];
  work[12] = rhs[12];
  work[13] = rhs[13];
  work[14] = rhs[14];
  work[15] = rhs[15];
  work[16] = rhs[16];
  work[17] = rhs[17];
  work[18] = rhs[18];
  work[19] = rhs[19];
  work[20] = rhs[20];
  work[21] = rhs[21];
  work[22] = rhs[22] - lu[22 * N + 21] * work[21];
  work[23] = rhs[23] - lu[23 * N + 11] * work[11] - lu[23 * N + 12] * work[12] - lu[23 * N + 13] * work[13] - lu[23 * N + 14] * work[14] - lu[23 * N + 15] * work[15] - lu[23 * N + 16] * work[16] - lu[23 * N + 17] * work[17] - lu[23 * N + 18] * work[18] - lu[23 * N + 19] * work[19] - lu[23 * N + 20] * work[20] - lu[23 * N + 21] * work[21] - lu[23 * N + 22] * work[22];
  work[24] = rhs[24] - lu[24 * N + 22] * work[22] - lu[24 * N + 23] * work[23];
  work[25] = rhs[25] - lu[25 * N + 24] * work[24];
  solution[25] = work[25] / lu[25 * N + 25];
  solution[24] = (work[24] - lu[24 * N + 25] * solution[25]) / lu[24 * N + 24];
  solution[23] = (work[23] - lu[23 * N + 24] * solution[24] - lu[23 * N + 25] * solution[25]) / lu[23 * N + 23];
  solution[22] = (work[22] - lu[22 * N + 23] * solution[23] - lu[22 * N + 24] * solution[24]) / lu[22 * N + 22];
  solution[21] = (work[21] - lu[21 * N + 23] * solution[23]) / lu[21 * N + 21];
  solution[20] = work[20] / lu[20 * N + 20];
  solution[19] = work[19] / lu[19 * N + 19];
  solution[18] = work[18] / lu[18 * N + 18];
  solution[17] = work[17] / lu[17 * N + 17];
  solution[16] = work[16] / lu[16 * N + 16];
  solution[15] = work[15] / lu[15 * N + 15];
  solution[14] = work[14] / lu[14 * N + 14];
  solution[13] = work[13] / lu[13 * N + 13];
  solution[12] = work[12] / lu[12 * N + 12];
  solution[11] = work[11] / lu[11 * N + 11];
  solution[10] = (work[10] - lu[10 * N + 11] * solution[11] - lu[10 * N + 12] * solution[12] - lu[10 * N + 13] * solution[13] - lu[10 * N + 14] * solution[14] - lu[10 * N + 15] * solution[15] - lu[10 * N + 16] * solution[16] - lu[10 * N + 17] * solution[17] - lu[10 * N + 18] * solution[18] - lu[10 * N + 19] * solution[19] - lu[10 * N + 20] * solution[20] - lu[10 * N + 21] * solution[21] - lu[10 * N + 22] * solution[22] - lu[10 * N + 23] * solution[23]) / lu[10 * N + 10];
  solution[9] = (work[9] - lu[9 * N + 10] * solution[10] - lu[9 * N + 11] * solution[11] - lu[9 * N + 12] * solution[12] - lu[9 * N + 13] * solution[13] - lu[9 * N + 14] * solution[14] - lu[9 * N + 15] * solution[15] - lu[9 * N + 16] * solution[16] - lu[9 * N + 17] * solution[17] - lu[9 * N + 18] * solution[18] - lu[9 * N + 19] * solution[19] - lu[9 * N + 20] * solution[20] - lu[9 * N + 21] * solution[21] - lu[9 * N + 22] * solution[22] - lu[9 * N + 23] * solution[23]) / lu[9 * N + 9];
  solution[8] = (work[8] - lu[8 * N + 9] * solution[9] - lu[8 * N + 10] * solution[10] - lu[8 * N + 11] * solution[11] - lu[8 * N + 12] * solution[12] - lu[8 * N + 13] * solution[13] - lu[8 * N + 14] * solution[14] - lu[8 * N + 15] * solution[15] - lu[8 * N + 16] * solution[16] - lu[8 * N + 17] * solution[17] - lu[8 * N + 18] * solution[18] - lu[8 * N + 19] * solution[19] - lu[8 * N + 20] * solution[20] - lu[8 * N + 21] * solution[21] - lu[8 * N + 22] * solution[22] - lu[8 * N + 23] * solution[23]) / lu[8 * N + 8];
  solution[7] = (work[7] - lu[7 * N + 8] * solution[8] - lu[7 * N + 9] * solution[9] - lu[7 * N + 10] * solution[10] - lu[7 * N + 11] * solution[11] - lu[7 * N + 12] * solution[12] - lu[7 * N + 13] * solution[13] - lu[7 * N + 14] * solution[14] - lu[7 * N + 15] * solution[15] - lu[7 * N + 16] * solution[16] - lu[7 * N + 17] * solution[17] - lu[7 * N + 18] * solution[18] - lu[7 * N + 19] * solution[19] - lu[7 * N + 20] * solution[20] - lu[7 * N + 21] * solution[21] - lu[7 * N + 22] * solution[22] - lu[7 * N + 23] * solution[23]) / lu[7 * N + 7];
  solution[6] = (work[6] - lu[6 * N + 7] * solution[7] - lu[6 * N + 8] * solution[8] - lu[6 * N + 9] * solution[9] - lu[6 * N + 10] * solution[10] - lu[6 * N + 11] * solution[11] - lu[6 * N + 12] * solution[12] - lu[6 * N + 13] * solution[13] - lu[6 * N + 14] * solution[14] - lu[6 * N + 15] * solution[15] - lu[6 * N + 16] * solution[16] - lu[6 * N + 17] * solution[17] - lu[6 * N + 18] * solution[18] - lu[6 * N + 19] * solution[19] - lu[6 * N + 20] * solution[20] - lu[6 * N + 21] * solution[21] - lu[6 * N + 22] * solution[22] - lu[6 * N + 23] * solution[23]) / lu[6 * N + 6];
  solution[5] = (work[5] - lu[5 * N + 6] * solution[6] - lu[5 * N + 7] * solution[7] - lu[5 * N + 8] * solution[8] - lu[5 * N + 9] * solution[9] - lu[5 * N + 10] * solution[10] - lu[5 * N + 11] * solution[11] - lu[5 * N + 12] * solution[12] - lu[5 * N + 13] * solution[13] - lu[5 * N + 14] * solution[14] - lu[5 * N + 15] * solution[15] - lu[5 * N + 16] * solution[16] - lu[5 * N + 17] * solution[17] - lu[5 * N + 18] * solution[18] - lu[5 * N + 19] * solution[19] - lu[5 * N + 20] * solution[20] - lu[5 * N + 21] * solution[21] - lu[5 * N + 22] * solution[22] - lu[5 * N + 23] * solution[23]) / lu[5 * N + 5];
  solution[4] = (work[4] - lu[4 * N + 11] * solution[11] - lu[4 * N + 12] * solution[12] - lu[4 * N + 13] * solution[13] - lu[4 * N + 14] * solution[14] - lu[4 * N + 15] * solution[15] - lu[4 * N + 16] * solution[16] - lu[4 * N + 17] * solution[17] - lu[4 * N + 18] * solution[18] - lu[4 * N + 19] * solution[19] - lu[4 * N + 20] * solution[20] - lu[4 * N + 21] * solution[21] - lu[4 * N + 22] * solution[22] - lu[4 * N + 23] * solution[23]) / lu[4 * N + 4];
  solution[3] = work[3] / lu[3 * N + 3];
  solution[2] = work[2] / lu[2 * N + 2];
  solution[1] = work[1] / lu[1 * N + 1];
  solution[0] = work[0] / lu[0 * N + 0];
}
}  // namespace mkpp::generated::gocart::detail
