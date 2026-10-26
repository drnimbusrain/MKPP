#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::carbon::detail {
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
    { 1, 1, 0, 0, -1 },
    { 2, 2, 0, 0, -1 },
    { 3, 3, 0, 0, -1 },
    { 4, 4, 0, 0, -1 },
    { 6, 6, 0, 0, 4 },
    { 5, 5, 0, 0, -1 },
    { 7, 7, 0, 0, 5 },
    { 8, 8, 0, 0, -1 },
    { 9, 9, 0, 0, -1 },
    { 10, 10, 0, 0, 8 },
    { 13, 13, 0, 0, 8 },
    { 11, 11, 0, 0, -1 },
    { 12, 12, 0, 1, -1 },
    { 14, 14, 1, 2, -1 },
    { 15, 15, 2, 2, -1 },
    { 16, 16, 2, 2, 14 },
    { 19, 19, 2, 2, 14 },
    { 17, 17, 2, 2, -1 },
    { 18, 18, 2, 3, -1 },
    { 20, 20, 3, 4, -1 },
    { 21, 21, 4, 4, -1 },
    { 22, 22, 4, 4, 20 },
    { 25, 25, 4, 4, 20 },
    { 23, 23, 4, 4, -1 },
    { 24, 24, 4, 5, -1 },
    { 26, 26, 5, 6, -1 },
  };
  static constexpr Product products[] = {
    { 10, 9 },
    { 13, 9 },
    { 16, 15 },
    { 19, 15 },
    { 22, 21 },
    { 25, 21 },
  };
  for (const auto& op : operations) {
    double value = op.input >= 0 ? w[op.input] : 0.0;
    for (int product = op.begin; product < op.end; ++product)
      value -= lu[products[product].left] * lu[products[product].right];
    lu[op.output] = op.diagonal >= 0 ? value / lu[op.diagonal] : value;
  }
}
}  // namespace mkpp::generated::carbon::detail
