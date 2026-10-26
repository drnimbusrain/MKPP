#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::chapman::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
compute_rates_chunk_0(
  StrideView<const double, 1> state, StrideView<double, 1> rates,
    const double* jvals, double temp, double rh, double air_density) {
  (void)temp;
  (void)rh;
  (void)air_density;
  rates(0) = -6e-34*state(3)*state(0)*state(1) - 8e-12*state(0)*state(2) + 2.0*state(1)*jvals[0] + 1.0*state(2)*jvals[1];
  rates(1) = -6e-34*state(3)*state(0)*state(1) + 1.6e-11*state(0)*state(2) - 1.0*state(1)*jvals[0] + 1.0*state(2)*jvals[1];
  rates(2) = 6e-34*state(3)*state(0)*state(1) - 8e-12*state(0)*state(2) - 1.0*state(2)*jvals[1];
  rates(3) = 0.0;
}
}  // namespace mkpp::generated::chapman::detail
