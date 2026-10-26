#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::carbon::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
compute_jacobian_chunk_0(
  StrideView<const double, 1> state, StrideView<double, 2> jacobian,
    const double* jvals, double temp, double rh, double air_density) {
  const double Temp = temp;
  const double RH = rh;
  const double AirDensity = air_density;
  (void)RH;
  jacobian.at_slot(0, 0, 0) = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(1, 0, 7) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp);
  jacobian.at_slot(2, 0, 8) = -9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(3, 1, 1) = -7.3679649e-14;
  jacobian.at_slot(4, 1, 7) = -7.3679649e-14*state(1);
  jacobian.at_slot(5, 1, 9) = 4.2566446e-15;
  jacobian.at_slot(6, 1, 10) = 38199.012;
  jacobian.at_slot(7, 2, 9) = 4.2566446e-15;
  jacobian.at_slot(8, 3, 10) = 38199.012;
  jacobian.at_slot(9, 4, 0) = 2.45e-12*Kokkos::exp(-1775.0/Temp);
  jacobian.at_slot(10, 4, 7) = 2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp);
  jacobian.at_slot(11, 5, 0) = 9.6e-12*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(12, 5, 8) = 9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(13, 6, 1) = 7.3679649e-14;
  jacobian.at_slot(14, 6, 7) = 7.3679649e-14*state(1);
  jacobian.at_slot(15, 7, 0) = -2.45e-12*Kokkos::exp(-1775.0/Temp);
  jacobian.at_slot(16, 7, 1) = -7.3679649e-14;
  jacobian.at_slot(17, 7, 7) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*state(1);
  jacobian.at_slot(18, 8, 0) = -9.6e-12*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(19, 8, 8) = -9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
  jacobian.at_slot(20, 9, 9) = -4.2566446e-15;
  jacobian.at_slot(21, 10, 10) = -38199.012;
}
}  // namespace mkpp::generated::carbon::detail
