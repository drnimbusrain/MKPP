#pragma once
#include <Kokkos_Core.hpp>
#include "accessor.hpp"
namespace mkpp::generated::small_strato::detail {
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
  jacobian.at_slot(0, 0, 0) = -1.069e-11*state(4) - 8.018e-17*state(6) - 1.576e-15*state(2);
  jacobian.at_slot(1, 0, 1) = 7.11e-11*state(5);
  jacobian.at_slot(2, 0, 2) = -1.576e-15*state(0) + 1.0*jvals[1];
  jacobian.at_slot(3, 0, 4) = -1.069e-11*state(0) + 1.0*jvals[3];
  jacobian.at_slot(4, 0, 5) = 7.11e-11*state(1);
  jacobian.at_slot(5, 0, 6) = -8.018e-17*state(0) + 2.0*jvals[0];
  jacobian.at_slot(6, 1, 1) = -7.11e-11*state(5) - 1.2e-10*state(2);
  jacobian.at_slot(7, 1, 2) = -1.2e-10*state(1) + 1.0*jvals[2];
  jacobian.at_slot(8, 1, 5) = -7.11e-11*state(1);
  jacobian.at_slot(9, 2, 0) = 8.018e-17*state(6) - 1.576e-15*state(2);
  jacobian.at_slot(10, 2, 1) = -1.2e-10*state(2);
  jacobian.at_slot(11, 2, 2) = -6.062e-15*state(3) - 1.576e-15*state(0) - 1.2e-10*state(1) - 1.0*jvals[1] - 1.0*jvals[2];
  jacobian.at_slot(12, 2, 3) = -6.062e-15*state(2);
  jacobian.at_slot(13, 2, 6) = 8.018e-17*state(0);
  jacobian.at_slot(14, 3, 0) = 1.069e-11*state(4);
  jacobian.at_slot(15, 3, 2) = -6.062e-15*state(3);
  jacobian.at_slot(16, 3, 3) = -6.062e-15*state(2);
  jacobian.at_slot(17, 3, 4) = 1.069e-11*state(0) + 1.0*jvals[3];
  jacobian.at_slot(18, 4, 0) = -1.069e-11*state(4);
  jacobian.at_slot(19, 4, 2) = 6.062e-15*state(3);
  jacobian.at_slot(20, 4, 3) = 6.062e-15*state(2);
  jacobian.at_slot(21, 4, 4) = -1.069e-11*state(0) - 1.0*jvals[3];
  jacobian.at_slot(22, 6, 0) = 1.069e-11*state(4) - 8.018e-17*state(6) + 3.152e-15*state(2);
  jacobian.at_slot(23, 6, 1) = 2.4e-10*state(2);
  jacobian.at_slot(24, 6, 2) = 6.062e-15*state(3) + 3.152e-15*state(0) + 2.4e-10*state(1) + 1.0*jvals[1] + 1.0*jvals[2];
  jacobian.at_slot(25, 6, 3) = 6.062e-15*state(2);
  jacobian.at_slot(26, 6, 4) = 1.069e-11*state(0);
  jacobian.at_slot(27, 6, 6) = -8.018e-17*state(0) - 1.0*jvals[0];
}
}  // namespace mkpp::generated::small_strato::detail
