#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::saprc99_mini::detail {
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
  rates(0) = 4.9010355073085e-27*state(12)*state(13)*state(9)*Kokkos::pow(Temp, -2.8) - 1.8e-12*state(2)*state(0)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*state(0)*Kokkos::exp(-2470.0/Temp) - 8e-12*state(0)*state(9)*Kokkos::exp(-2060.0/Temp);
  rates(1) = -1.0*state(1)*jvals[3] + 1.0*state(11)*(1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
  rates(2) = 1.0*state(6)*jvals[1] - 1.8e-11*state(2)*state(4)*Kokkos::exp(110.0/Temp) - 1.8e-12*state(2)*state(0)*Kokkos::exp(-1370.0/Temp) + 1.0*state(3)*jvals[0];
  rates(3) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*state(7)*jvals[2] + 3.6e-11*state(2)*state(4)*Kokkos::exp(110.0/Temp) + 1.8e-12*state(2)*state(0)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*state(0)*Kokkos::exp(-2470.0/Temp) - 1.0*state(3)*jvals[0];
  rates(4) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*state(7)*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.8e-11*state(2)*state(4)*Kokkos::exp(110.0/Temp) + 1.4e-13*state(3)*state(0)*Kokkos::exp(-2470.0/Temp);
  rates(5) = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
  rates(6) = -1.0*state(6)*jvals[1];
  rates(7) = -1.0*state(7)*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*state(7)*jvals[2];
  rates(8) = 0.0;
  rates(9) = -4.9010355073085e-27*state(12)*state(13)*state(9)*Kokkos::pow(Temp, -2.8) + 1.0*state(3)*jvals[0] - 8e-12*state(0)*state(9)*Kokkos::exp(-2060.0/Temp);
  rates(10) = 2.0*state(1)*jvals[3] - 1.0*state(7)*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*state(7)*jvals[2] + 1.0*state(6)*jvals[1];
  rates(11) = -1.0*state(11)*(1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
  rates(12) = 0.0;
  rates(13) = -4.9010355073085e-27*state(12)*state(13)*state(9)*Kokkos::pow(Temp, -2.8) + 1.0*state(11)*(1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) + 1.6e-11*state(0)*state(9)*Kokkos::exp(-2060.0/Temp);
  rates(14) = 0.0;
  rates(15) = 0.0;
  rates(16) = 0.0;
}
}  // namespace mkpp::generated::saprc99_mini::detail
