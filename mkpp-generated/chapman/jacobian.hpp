#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::chapman::detail {
#if defined(MKPP_DEFINE_HOST_KERNELS)
void
#else
KOKKOS_INLINE_FUNCTION void
#endif
compute_jacobian_chunk_0(
  StrideView<const double, 1> state, StrideView<double, 2> jacobian,
    const double* jvals, double temp, double rh, double air_density) {
  (void)temp;
  (void)rh;
  (void)air_density;
  jacobian.at_slot(0, 0, 0) = -6e-34*state(3)*state(1) - 8e-12*state(2);
  jacobian.at_slot(1, 0, 1) = -6e-34*state(3)*state(0) + 2.0*jvals[0];
  jacobian.at_slot(2, 0, 2) = -8e-12*state(0) + 1.0*jvals[1];
  jacobian.at_slot(3, 0, 3) = -6e-34*state(0)*state(1);
  jacobian.at_slot(4, 1, 0) = -6e-34*state(3)*state(1) + 1.6e-11*state(2);
  jacobian.at_slot(5, 1, 1) = -6e-34*state(3)*state(0) - 1.0*jvals[0];
  jacobian.at_slot(6, 1, 2) = 1.6e-11*state(0) + 1.0*jvals[1];
  jacobian.at_slot(7, 1, 3) = -6e-34*state(0)*state(1);
  jacobian.at_slot(8, 2, 0) = 6e-34*state(3)*state(1) - 8e-12*state(2);
  jacobian.at_slot(9, 2, 1) = 6e-34*state(3)*state(0);
  jacobian.at_slot(10, 2, 2) = -8e-12*state(0) - 1.0*jvals[1];
  jacobian.at_slot(11, 2, 3) = 6e-34*state(0)*state(1);
}
}  // namespace mkpp::generated::chapman::detail
