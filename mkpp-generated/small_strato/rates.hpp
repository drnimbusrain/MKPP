#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::small_strato::detail {
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
  rates(0) = 7.11e-11*state(5)*state(1) - 1.069e-11*state(4)*state(0) + 1.0*state(4)*jvals[3] - 8.018e-17*state(0)*state(6) - 1.576e-15*state(0)*state(2) + 2.0*state(6)*jvals[0] + 1.0*state(2)*jvals[1];
  rates(1) = -7.11e-11*state(5)*state(1) - 1.2e-10*state(1)*state(2) + 1.0*state(2)*jvals[2];
  rates(2) = -6.062e-15*state(3)*state(2) + 8.018e-17*state(0)*state(6) - 1.576e-15*state(0)*state(2) - 1.2e-10*state(1)*state(2) - 1.0*state(2)*jvals[1] - 1.0*state(2)*jvals[2];
  rates(3) = -6.062e-15*state(3)*state(2) + 1.069e-11*state(4)*state(0) + 1.0*state(4)*jvals[3];
  rates(4) = 6.062e-15*state(3)*state(2) - 1.069e-11*state(4)*state(0) - 1.0*state(4)*jvals[3];
  rates(5) = 0.0;
  rates(6) = 6.062e-15*state(3)*state(2) + 1.069e-11*state(4)*state(0) - 8.018e-17*state(0)*state(6) + 3.152e-15*state(0)*state(2) + 2.4e-10*state(1)*state(2) - 1.0*state(6)*jvals[0] + 1.0*state(2)*jvals[1] + 1.0*state(2)*jvals[2];
}
}  // namespace mkpp::generated::small_strato::detail
