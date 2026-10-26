#pragma once
#include "accessor.hpp"
// CPU builds link separately optimized definitions; device builds retain visible bodies.
namespace mkpp::generated::small_strato::detail {
void compute_rates_chunk_0(StrideView<const double, 1>, StrideView<double, 1>, const double*, double, double, double);
void compute_jacobian_chunk_0(StrideView<const double, 1>, StrideView<double, 2>, const double*, double, double, double);
void factorize_plan(const double*, double*);
void solve_plan(const double*, const double*, double*);
void solve_lu(const double*, const double*, double*);
}  // namespace mkpp::generated::small_strato::detail
