#pragma once
#include <Kokkos_Core.hpp>
namespace mkpp::generated::chapman::detail {
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
    { 5, 5, 0, 0, 0 },
    { 9, 9, 0, 0, 0 },
    { 2, 2, 0, 0, -1 },
    { 3, 3, 0, 0, -1 },
    { 4, 4, 0, 0, -1 },
    { 6, 6, 0, 0, 2 },
    { 10, 10, 0, 0, 2 },
    { 7, 7, 0, 1, -1 },
    { 8, 8, 1, 2, -1 },
    { 11, 11, 2, 3, 7 },
    { 12, 12, 3, 5, -1 },
  };
  static constexpr Product products[] = {
    { 6, 3 },
    { 6, 4 },
    { 10, 3 },
    { 10, 4 },
    { 11, 8 },
  };
  for (const auto& op : operations) {
    double value = op.input >= 0 ? w[op.input] : 0.0;
    for (int product = op.begin; product < op.end; ++product)
      value -= lu[products[product].left] * lu[products[product].right];
    lu[op.output] = op.diagonal >= 0 ? value / lu[op.diagonal] : value;
  }
}
}  // namespace mkpp::generated::chapman::detail
