#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::small_strato::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_lu(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 7;
  double work[N];
  work[0] = rhs[0];
  work[1] = rhs[1] - lu[1 * N + 0] * work[0];
  work[2] = rhs[2] - lu[2 * N + 0] * work[0] - lu[2 * N + 1] * work[1];
  work[3] = rhs[3] - lu[3 * N + 1] * work[1] - lu[3 * N + 2] * work[2];
  work[4] = rhs[4] - lu[4 * N + 1] * work[1] - lu[4 * N + 2] * work[2] - lu[4 * N + 3] * work[3];
  work[5] = rhs[5] - lu[5 * N + 2] * work[2] - lu[5 * N + 3] * work[3] - lu[5 * N + 4] * work[4];
  work[6] = rhs[6] - lu[6 * N + 2] * work[2] - lu[6 * N + 3] * work[3] - lu[6 * N + 4] * work[4] - lu[6 * N + 5] * work[5];
  solution[6] = work[6] / lu[6 * N + 6];
  solution[5] = (work[5] - lu[5 * N + 6] * solution[6]) / lu[5 * N + 5];
  solution[4] = (work[4] - lu[4 * N + 5] * solution[5] - lu[4 * N + 6] * solution[6]) / lu[4 * N + 4];
  solution[3] = (work[3] - lu[3 * N + 4] * solution[4] - lu[3 * N + 5] * solution[5] - lu[3 * N + 6] * solution[6]) / lu[3 * N + 3];
  solution[2] = (work[2] - lu[2 * N + 3] * solution[3] - lu[2 * N + 4] * solution[4] - lu[2 * N + 6] * solution[6]) / lu[2 * N + 2];
  solution[1] = (work[1] - lu[1 * N + 4] * solution[4]) / lu[1 * N + 1];
  solution[0] = work[0] / lu[0 * N + 0];
}
}  // namespace mkpp::generated::small_strato::detail
