#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::chapman::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
solve_lu(const double* lu, const double* rhs, double* solution) {
  constexpr int N = 4;
  double work[N];
  work[0] = rhs[0];
  work[1] = rhs[1] - lu[1 * N + 0] * work[0];
  work[2] = rhs[2] - lu[2 * N + 0] * work[0] - lu[2 * N + 1] * work[1];
  work[3] = rhs[3] - lu[3 * N + 0] * work[0] - lu[3 * N + 1] * work[1] - lu[3 * N + 2] * work[2];
  solution[3] = work[3] / lu[3 * N + 3];
  solution[2] = (work[2] - lu[2 * N + 3] * solution[3]) / lu[2 * N + 2];
  solution[1] = (work[1] - lu[1 * N + 2] * solution[2] - lu[1 * N + 3] * solution[3]) / lu[1 * N + 1];
  solution[0] = work[0] / lu[0 * N + 0];
}
}  // namespace mkpp::generated::chapman::detail
