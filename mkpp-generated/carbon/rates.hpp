#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::carbon::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
compute_rates_chunk_0(
  StrideView<const double, 1> state, StrideView<double, 1> rates,
    const double* jvals, double temp, double rh, double air_density) {
  const double Temp = temp;
  const double RH = rh;
  const double AirDensity = air_density;
  (void)RH;
  rates(0) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp) - 9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  rates(1) = 38199.012 - 7.3679649e-14*state(1);
  rates(2) = 4.2566446e-15;
  rates(3) = 38199.012;
  rates(4) = 2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp);
  rates(5) = 9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  rates(6) = 7.3679649e-14*state(1);
  rates(7) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*state(1);
  rates(8) = -9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  rates(9) = -4.2566446e-15;
  rates(10) = -38199.012;
}
}  // namespace mkpp::generated::carbon::detail
