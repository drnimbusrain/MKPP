#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::small_strato::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
factorize_plan(const double* w, double* lu) {
  struct Operation { int output, input, begin, end, diagonal; };
  struct Product { int left, right; };
  static constexpr Operation operations[] = {
    { 0, 0, 0, 0, -1 },
    { 1, 1, 0, 0, 0 },
    { 4, 4, 0, 0, 0 },
    { 2, 2, 0, 0, -1 },
    { 3, 3, 0, 0, -1 },
    { 5, 5, 0, 0, 2 },
    { 10, 10, 0, 0, 2 },
    { 16, 16, 0, 0, 2 },
    { 6, 6, 0, 0, -1 },
    { 7, 7, 0, 0, -1 },
    { 8, 8, 0, 1, -1 },
    { 9, 9, 1, 1, -1 },
    { 11, 11, 1, 1, 6 },
    { 17, 17, 1, 1, 6 },
    { 22, 21, 1, 1, 6 },
    { 27, 25, 1, 1, 6 },
    { 12, 12, 1, 2, -1 },
    { 13, 13, 2, 4, -1 },
    { 14, 14, 4, 4, -1 },
    { 15, 15, 4, 5, -1 },
    { 18, 18, 5, 6, 12 },
    { 23, -1, 6, 7, 12 },
    { 28, -1, 7, 8, 12 },
    { 19, 19, 8, 11, -1 },
    { 20, 20, 11, 12, -1 },
    { 21, -1, 12, 14, -1 },
    { 24, 22, 14, 16, 19 },
    { 29, 26, 16, 18, 19 },
    { 25, 23, 18, 20, -1 },
    { 26, 24, 20, 23, -1 },
    { 30, 27, 23, 25, 25 },
    { 31, 28, 25, 29, -1 },
  };
  static constexpr Product products[] = {
    { 5, 3 },
    { 11, 7 },
    { 10, 3 },
    { 11, 8 },
    { 11, 9 },
    { 17, 7 },
    { 22, 7 },
    { 27, 7 },
    { 16, 3 },
    { 17, 8 },
    { 18, 13 },
    { 18, 14 },
    { 17, 9 },
    { 18, 15 },
    { 22, 8 },
    { 23, 13 },
    { 27, 8 },
    { 28, 13 },
    { 23, 14 },
    { 24, 20 },
    { 22, 9 },
    { 23, 15 },
    { 24, 21 },
    { 28, 14 },
    { 29, 20 },
    { 27, 9 },
    { 28, 15 },
    { 29, 21 },
    { 30, 26 },
  };
  for (const auto& op : operations) {
    double value = op.input >= 0 ? w[op.input] : 0.0;
    for (int product = op.begin; product < op.end; ++product)
      value -= lu[products[product].left] * lu[products[product].right];
    lu[op.output] = op.diagonal >= 0 ? value / lu[op.diagonal] : value;
  }
}
}  // namespace mkpp::generated::small_strato::detail
