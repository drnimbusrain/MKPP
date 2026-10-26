#pragma once
#include <Kokkos_Core.hpp>
// CPU libraries can own the heavy helper definitions. Device builds keep the
// complete device-callable implementation visible without relocatable device code.
#if defined(MKPP_SEPARATE_HOST_KERNELS_SAPRC99_MINI)
#include "saprc99_mini/host_declarations.hpp"
#else
#include "saprc99_mini/accessor.hpp"
#include "saprc99_mini/rates.hpp"
#include "saprc99_mini/jacobian.hpp"
#include "saprc99_mini/supernodal_factorize.hpp"
#include "saprc99_mini/supernodal_solve.hpp"
#include "saprc99_mini/solve.hpp"
#endif
// Generated solver for saprc99_mini
// SZA Workload Sorted: true
// Hysteresis/Spline Continuous Transition: true
namespace mkpp::generated::saprc99_mini {
  // Pure Kokkos abstractions (no raw pragmas allowed)

  /**
   * @brief Strongly typed enum for species indexing into state vectors.
   */
  enum Species : int {
      O3 = 0,
      H2O2 = 1,
      NO = 2,
      NO2 = 3,
      NO3 = 4,
      N2O5 = 5,
      HONO = 6,
      HNO3 = 7,
      HNO4 = 8,
      O3P = 9,
      OH = 10,
      HO2 = 11,
      AIR = 12,
      O2 = 13,
      H2O = 14,
      M = 15,
      N2 = 16
  };


  // Equilibrium meteorological input indices
  enum class EquilibriumInput : int {
      Temperature = 0,
      RelativeHumidity = 1,
      COUNT = 2
  };

#ifdef MKPP_ENABLE_ADJOINT
  // Checkpoint buffer for discrete adjoint/TLM integration
  // Recompute-J strategy: only state is stored, Jacobian recomputed from saved state
  struct CheckpointBuffer {
      static constexpr int MAX_STEPS = 200;
      static constexpr int NUM_SPECIES = 17;
      int num_steps = 0;
      double h[MAX_STEPS];
      double state[MAX_STEPS][NUM_SPECIES];  // saved concentrations at step entry
  };
#endif

  // The expression-dense RHS and Jacobian are compiled in bounded units.
  // Every detail:: helper is an inline device-callable definition supplied by
  // the implementation fragments included at the top of this header, so a
  // device instantiation of SolverKernels resolves the full call chain without
  // any host-only out-of-line symbol.

  template<typename DeviceType>
  struct SolverKernels {
      using memory_space = typename DeviceType::memory_space;
      static constexpr bool SUPPORTS_RUNTIME_ENVIRONMENT = true;
      static constexpr int NUM_RUNTIME_PARAMETERS = 4;
      static constexpr bool HAS_CLOUD_GATED = false;
#ifdef MKPP_ENABLE_ADJOINT
      // Expose the checkpoint buffer through the kernel type so generic
      // device-path callers can name it without the mechanism namespace.
      using CheckpointBuffer = ::mkpp::generated::saprc99_mini::CheckpointBuffer;
#endif
      /**
       * @brief Evaluates the rate-of-change vector F(i) = dC_i / dt.
       *
       * @tparam StateView Kokkos View type for species concentrations [NUM_SPECIES].
       * @tparam RateView Kokkos View type for output rate-of-change vector [NUM_SPECIES].
       * @param state Input concentration vector [NUM_SPECIES].
       * @param F_block Output rate-of-change vector [NUM_SPECIES].
       * @param jvals Array of photolysis rate constants [NUM_PHOTOLYSIS].
       */
      template <class StateView, class RateView>
      KOKKOS_INLINE_FUNCTION void compute_rates(const StateView& state, RateView& F_block, Kokkos::View<const double*, memory_space> jvals_view, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const {
          const double* jvals = jvals_view.data();
              const auto state_accessor = detail::make_stride_view(state);
              const auto rates_accessor = detail::make_stride_view(F_block);
              detail::compute_rates_chunk_0(state_accessor, rates_accessor, jvals, temp, rh, air_density);
      }

      /**
       * @brief Evaluates the sparse analytical Jacobian matrix J(i, j) = dF_i / dC_j.
       *
       * @tparam StateView Kokkos View type for species concentrations [NUM_SPECIES].
       * @tparam JacView Kokkos View type for output 2D Jacobian matrix [NUM_SPECIES x NUM_SPECIES].
       * @param state Input concentration vector [NUM_SPECIES].
       * @param J_block Output 2D Jacobian matrix [NUM_SPECIES x NUM_SPECIES].
       * @param jvals Array of photolysis rate constants [NUM_PHOTOLYSIS].
       */
      template <class StateView, class JacView>
      KOKKOS_INLINE_FUNCTION void compute_jacobian(const StateView& state, JacView& J_block, Kokkos::View<const double*, memory_space> jvals_view, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const {
          const double* jvals = jvals_view.data();
              const auto state_accessor = detail::make_stride_view(state);
              const auto jacobian_accessor = detail::make_stride_view(J_block);
              detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, temp, rh, air_density);
      }

#ifdef MKPP_ENABLE_ADJOINT
      template <class StateView, class JacView>
      KOKKOS_INLINE_FUNCTION void compute_adjoint(const StateView& state, JacView& J_adj_block, Kokkos::View<const double*, memory_space> jvals_view, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const {
          const double* jvals = jvals_view.data();
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          // --- Sparse Analytical Adjoint Jacobian Entries J_adj_block(i, j) = J^T(i, j) ---
          // J^T(O3, O3): d(d[O3]/dt) / d[O3]
          J_adj_block(0, 0) = -1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp) - 8e-12*state(9)*Kokkos::exp(-2060.0/Temp);
          // J^T(O3, NO): d(d[NO]/dt) / d[O3]
          J_adj_block(0, 2) = -1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp);
          // J^T(O3, NO2): d(d[NO2]/dt) / d[O3]
          J_adj_block(0, 3) = 1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp);
          // J^T(O3, NO3): d(d[NO3]/dt) / d[O3]
          J_adj_block(0, 4) = 1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp);
          // J^T(O3, O3P): d(d[O3P]/dt) / d[O3]
          J_adj_block(0, 9) = -8e-12*state(9)*Kokkos::exp(-2060.0/Temp);
          // J^T(O3, O2): d(d[O2]/dt) / d[O3]
          J_adj_block(0, 13) = 1.6e-11*state(9)*Kokkos::exp(-2060.0/Temp);
          // J^T(H2O2, H2O2): d(d[H2O2]/dt) / d[H2O2]
          J_adj_block(1, 1) = -1.0*jvals[3];
          // J^T(H2O2, OH): d(d[OH]/dt) / d[H2O2]
          J_adj_block(1, 10) = 2.0*jvals[3];
          // J^T(NO, O3): d(d[O3]/dt) / d[NO]
          J_adj_block(2, 0) = -1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp);
          // J^T(NO, NO): d(d[NO]/dt) / d[NO]
          J_adj_block(2, 2) = -1.8e-11*state(4)*Kokkos::exp(110.0/Temp) - 1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp);
          // J^T(NO, NO2): d(d[NO2]/dt) / d[NO]
          J_adj_block(2, 3) = 3.6e-11*state(4)*Kokkos::exp(110.0/Temp) + 1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp);
          // J^T(NO, NO3): d(d[NO3]/dt) / d[NO]
          J_adj_block(2, 4) = -1.8e-11*state(4)*Kokkos::exp(110.0/Temp);
          // J^T(NO2, O3): d(d[O3]/dt) / d[NO2]
          J_adj_block(3, 0) = -1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp);
          // J^T(NO2, NO): d(d[NO]/dt) / d[NO2]
          J_adj_block(3, 2) = 1.0*jvals[0];
          // J^T(NO2, NO2): d(d[NO2]/dt) / d[NO2]
          J_adj_block(3, 3) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0];
          // J^T(NO2, NO3): d(d[NO3]/dt) / d[NO2]
          J_adj_block(3, 4) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp);
          // J^T(NO2, N2O5): d(d[N2O5]/dt) / d[NO2]
          J_adj_block(3, 5) = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0);
          // J^T(NO2, O3P): d(d[O3P]/dt) / d[NO2]
          J_adj_block(3, 9) = 1.0*jvals[0];
          // J^T(NO3, NO): d(d[NO]/dt) / d[NO3]
          J_adj_block(4, 2) = -1.8e-11*state(2)*Kokkos::exp(110.0/Temp);
          // J^T(NO3, NO2): d(d[NO2]/dt) / d[NO3]
          J_adj_block(4, 3) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*state(2)*Kokkos::exp(110.0/Temp);
          // J^T(NO3, NO3): d(d[NO3]/dt) / d[NO3]
          J_adj_block(4, 4) = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*state(2)*Kokkos::exp(110.0/Temp);
          // J^T(NO3, N2O5): d(d[N2O5]/dt) / d[NO3]
          J_adj_block(4, 5) = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0);
          // J^T(N2O5, NO2): d(d[NO2]/dt) / d[N2O5]
          J_adj_block(5, 3) = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // J^T(N2O5, NO3): d(d[NO3]/dt) / d[N2O5]
          J_adj_block(5, 4) = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // J^T(N2O5, N2O5): d(d[N2O5]/dt) / d[N2O5]
          J_adj_block(5, 5) = -2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // J^T(HONO, NO): d(d[NO]/dt) / d[HONO]
          J_adj_block(6, 2) = 1.0*jvals[1];
          // J^T(HONO, HONO): d(d[HONO]/dt) / d[HONO]
          J_adj_block(6, 6) = -1.0*jvals[1];
          // J^T(HONO, OH): d(d[OH]/dt) / d[HONO]
          J_adj_block(6, 10) = 1.0*jvals[1];
          // J^T(HNO3, NO2): d(d[NO2]/dt) / d[HNO3]
          J_adj_block(7, 3) = 1.0*jvals[2];
          // J^T(HNO3, NO3): d(d[NO3]/dt) / d[HNO3]
          J_adj_block(7, 4) = 1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          // J^T(HNO3, HNO3): d(d[HNO3]/dt) / d[HNO3]
          J_adj_block(7, 7) = -1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2];
          // J^T(HNO3, OH): d(d[OH]/dt) / d[HNO3]
          J_adj_block(7, 10) = -1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2];
          // J^T(O3P, O3): d(d[O3]/dt) / d[O3P]
          J_adj_block(9, 0) = 4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) - 8e-12*state(0)*Kokkos::exp(-2060.0/Temp);
          // J^T(O3P, O3P): d(d[O3P]/dt) / d[O3P]
          J_adj_block(9, 9) = -4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) - 8e-12*state(0)*Kokkos::exp(-2060.0/Temp);
          // J^T(O3P, O2): d(d[O2]/dt) / d[O3P]
          J_adj_block(9, 13) = -4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) + 1.6e-11*state(0)*Kokkos::exp(-2060.0/Temp);
          // J^T(OH, NO3): d(d[NO3]/dt) / d[OH]
          J_adj_block(10, 4) = 1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          // J^T(OH, HNO3): d(d[HNO3]/dt) / d[OH]
          J_adj_block(10, 7) = -1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          // J^T(OH, OH): d(d[OH]/dt) / d[OH]
          J_adj_block(10, 10) = -1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          // J^T(HO2, H2O2): d(d[H2O2]/dt) / d[HO2]
          J_adj_block(11, 1) = 1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          // J^T(HO2, HO2): d(d[HO2]/dt) / d[HO2]
          J_adj_block(11, 11) = -1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp);
          // J^T(HO2, O2): d(d[O2]/dt) / d[HO2]
          J_adj_block(11, 13) = 1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          // J^T(AIR, O3): d(d[O3]/dt) / d[AIR]
          J_adj_block(12, 0) = 4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8);
          // J^T(AIR, H2O2): d(d[H2O2]/dt) / d[AIR]
          J_adj_block(12, 1) = 1.7e-33*state(11)*Kokkos::exp(1000.0/Temp);
          // J^T(AIR, NO2): d(d[NO2]/dt) / d[AIR]
          J_adj_block(12, 3) = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          // J^T(AIR, NO3): d(d[NO3]/dt) / d[AIR]
          J_adj_block(12, 4) = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0));
          // J^T(AIR, N2O5): d(d[N2O5]/dt) / d[AIR]
          J_adj_block(12, 5) = -3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          // J^T(AIR, HNO3): d(d[HNO3]/dt) / d[AIR]
          J_adj_block(12, 7) = -1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0));
          // J^T(AIR, O3P): d(d[O3P]/dt) / d[AIR]
          J_adj_block(12, 9) = -4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8);
          // J^T(AIR, OH): d(d[OH]/dt) / d[AIR]
          J_adj_block(12, 10) = -1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0));
          // J^T(AIR, HO2): d(d[HO2]/dt) / d[AIR]
          J_adj_block(12, 11) = -1.7e-33*state(11)*Kokkos::exp(1000.0/Temp);
          // J^T(AIR, O2): d(d[O2]/dt) / d[AIR]
          J_adj_block(12, 13) = 1.7e-33*state(11)*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8);
          // J^T(O2, O3): d(d[O3]/dt) / d[O2]
          J_adj_block(13, 0) = 4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8);
          // J^T(O2, O3P): d(d[O3P]/dt) / d[O2]
          J_adj_block(13, 9) = -4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8);
          // J^T(O2, O2): d(d[O2]/dt) / d[O2]
          J_adj_block(13, 13) = -4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8);
      }
#endif

#ifdef MKPP_ENABLE_ADJOINT
      template <class StateView, class DeltaView, class RateView>
      KOKKOS_INLINE_FUNCTION void compute_tlm(const StateView& state, const DeltaView& delta_C, RateView& dF_block, Kokkos::View<const double*, memory_space> jvals_view, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const {
          const double* jvals = jvals_view.data();
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          dF_block(0) = 0.0;
          dF_block(0) += (-1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp) - 8e-12*state(9)*Kokkos::exp(-2060.0/Temp)) * delta_C(0);
          dF_block(0) += (-1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp)) * delta_C(2);
          dF_block(0) += (-1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp)) * delta_C(3);
          dF_block(0) += (4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) - 8e-12*state(0)*Kokkos::exp(-2060.0/Temp)) * delta_C(9);
          dF_block(0) += (4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(12);
          dF_block(0) += (4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(13);
          dF_block(1) = 0.0;
          dF_block(1) += (-1.0*jvals[3]) * delta_C(1);
          dF_block(1) += (1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) * delta_C(11);
          dF_block(1) += (1.7e-33*state(11)*Kokkos::exp(1000.0/Temp)) * delta_C(12);
          dF_block(2) = 0.0;
          dF_block(2) += (-1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp)) * delta_C(0);
          dF_block(2) += (-1.8e-11*state(4)*Kokkos::exp(110.0/Temp) - 1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp)) * delta_C(2);
          dF_block(2) += (1.0*jvals[0]) * delta_C(3);
          dF_block(2) += (-1.8e-11*state(2)*Kokkos::exp(110.0/Temp)) * delta_C(4);
          dF_block(2) += (1.0*jvals[1]) * delta_C(6);
          dF_block(3) = 0.0;
          dF_block(3) += (1.8e-12*state(2)*Kokkos::exp(-1370.0/Temp) - 1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp)) * delta_C(0);
          dF_block(3) += (3.6e-11*state(4)*Kokkos::exp(110.0/Temp) + 1.8e-12*state(0)*Kokkos::exp(-1370.0/Temp)) * delta_C(2);
          dF_block(3) += (-2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0]) * delta_C(3);
          dF_block(3) += (-2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*state(2)*Kokkos::exp(110.0/Temp)) * delta_C(4);
          dF_block(3) += (2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)) * delta_C(5);
          dF_block(3) += (1.0*jvals[2]) * delta_C(7);
          dF_block(3) += (3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)))) * delta_C(12);
          dF_block(4) = 0.0;
          dF_block(4) += (1.4e-13*state(3)*Kokkos::exp(-2470.0/Temp)) * delta_C(0);
          dF_block(4) += (-1.8e-11*state(4)*Kokkos::exp(110.0/Temp)) * delta_C(2);
          dF_block(4) += (-2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*state(0)*Kokkos::exp(-2470.0/Temp)) * delta_C(3);
          dF_block(4) += (-2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*state(2)*Kokkos::exp(110.0/Temp)) * delta_C(4);
          dF_block(4) += (2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)) * delta_C(5);
          dF_block(4) += (1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp))) * delta_C(7);
          dF_block(4) += (1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp))) * delta_C(10);
          dF_block(4) += (3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0))) * delta_C(12);
          dF_block(5) = 0.0;
          dF_block(5) += (2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)) * delta_C(3);
          dF_block(5) += (2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)) * delta_C(4);
          dF_block(5) += (-2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)) * delta_C(5);
          dF_block(5) += (-3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(3)*state(4)*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)/(1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(3)*state(4)*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp))/((1.4e-18*state(12)*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*state(12)*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(12)*state(5)*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*state(5)*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*state(12)*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)))) * delta_C(12);
          dF_block(6) = 0.0;
          dF_block(6) += (-1.0*jvals[1]) * delta_C(6);
          dF_block(7) = 0.0;
          dF_block(7) += (-1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2]) * delta_C(7);
          dF_block(7) += (-1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp))) * delta_C(10);
          dF_block(7) += (-1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0))) * delta_C(12);
          dF_block(8) = 0.0;
          dF_block(9) = 0.0;
          dF_block(9) += (-8e-12*state(9)*Kokkos::exp(-2060.0/Temp)) * delta_C(0);
          dF_block(9) += (1.0*jvals[0]) * delta_C(3);
          dF_block(9) += (-4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) - 8e-12*state(0)*Kokkos::exp(-2060.0/Temp)) * delta_C(9);
          dF_block(9) += (-4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(12);
          dF_block(9) += (-4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(13);
          dF_block(10) = 0.0;
          dF_block(10) += (2.0*jvals[3]) * delta_C(1);
          dF_block(10) += (1.0*jvals[1]) * delta_C(6);
          dF_block(10) += (-1.0*state(10)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2]) * delta_C(7);
          dF_block(10) += (-1.0*state(7)*(1.9e-33*state(12)*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp))) * delta_C(10);
          dF_block(10) += (-1.0*state(7)*state(10)*(-8.80487804878049e-51*state(12)*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*state(12)*Kokkos::exp(-715.0/Temp) + 1.0))) * delta_C(12);
          dF_block(11) = 0.0;
          dF_block(11) += (-1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp)) * delta_C(11);
          dF_block(11) += (-1.7e-33*state(11)*Kokkos::exp(1000.0/Temp)) * delta_C(12);
          dF_block(12) = 0.0;
          dF_block(13) = 0.0;
          dF_block(13) += (1.6e-11*state(9)*Kokkos::exp(-2060.0/Temp)) * delta_C(0);
          dF_block(13) += (-4.9010355073085e-27*state(12)*state(13)*Kokkos::pow(Temp, -2.8) + 1.6e-11*state(0)*Kokkos::exp(-2060.0/Temp)) * delta_C(9);
          dF_block(13) += (1.7e-33*state(12)*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) * delta_C(11);
          dF_block(13) += (1.7e-33*state(11)*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*state(13)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(12);
          dF_block(13) += (-4.9010355073085e-27*state(12)*state(9)*Kokkos::pow(Temp, -2.8)) * delta_C(13);
          dF_block(14) = 0.0;
          dF_block(15) = 0.0;
          dF_block(16) = 0.0;
      }
#endif

      template <class StateView, class MassView>
      KOKKOS_INLINE_FUNCTION void project_mass_conservation(StateView& C_projected, const StateView& C, const MassView& m_0) const {
          // C_projected = C - E^T (E E^T)^-1 (E C - m_0)
          C_projected(0) = C(0);
          C_projected(1) = C(1);
          C_projected(2) = C(2);
          C_projected(3) = C(3);
          C_projected(4) = C(4);
          C_projected(5) = C(5);
          C_projected(6) = C(6);
          C_projected(7) = C(7);
          C_projected(8) = C(8);
          C_projected(9) = C(9);
          C_projected(10) = C(10);
          C_projected(11) = C(11);
          C_projected(12) = C(12);
          C_projected(13) = C(13);
          C_projected(14) = C(14);
          C_projected(15) = C(15);
          C_projected(16) = C(16);
      }

      static constexpr int NUM_SPECIES = 17;
      static constexpr double atol[NUM_SPECIES] = { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
      static constexpr double rtol[NUM_SPECIES] = { 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06 };

      // Photolysis reactions (Cloud-J input mapping):
      //   jvals[0] = NO2 -> NO, O3P  (original A: 1.0)
      //   jvals[1] = HONO -> OH, NO  (original A: 1.0)
      //   jvals[2] = HNO3 -> OH, NO2  (original A: 1.0)
      //   jvals[3] = H2O2 -> OH  (original A: 1.0)
      static constexpr int NUM_PHOTOLYSIS = 4;

      /**
       * @brief Initial-step seed for the adaptive Rosenbrock ramp.
       *
       * The first internal substep of every integrate() call starts at
       * dt_total * initial_step_seed.  The compile-time value is baked as the
       * initializer; a host may override the member before a solve (per
       * SolverKernels instance) without regenerating this header.
       */
      double initial_step_seed = 1.0e-6;

      /**
       * @brief Performs adaptive time-stepping Rosenbrock integration over dt_total.
       *
       * @tparam StateView Kokkos View type for species concentrations [NUM_SPECIES].
       * @param dt_total Total physical time step duration [s].
       * @param state Input/output species concentration vector [NUM_SPECIES].
       * @param jvals Array of photolysis rate constants [NUM_PHOTOLYSIS].
       */
      template <class StateView>
      KOKKOS_INLINE_FUNCTION void integrate(double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const {
          const double* jvals = jvals_view.data();
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          const int NUM_SPECIES = 17;
          const auto state_accessor = detail::make_stride_view(state);
          // ROS-3 coefficients (3-stage, order 3)
          const double g = 0.435866521508459;
          const double safety = 0.9;
          const double max_growth = 6.0;
          const double min_shrink = 0.2;
          const double rejection_factor_decrease = 0.1;
          const double h_min = dt_total * 1.0e-15;
          double t = 0.0;
          // First substep of the adaptive ramp; the seed member carries the
          // compile-time default and any host-set runtime override.
          double dt = dt_total * initial_step_seed;
          bool reject_last_dt = false;
          bool reject_more_dt = false;

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // --- 0. Hoist State Values into Scalar Registers ---
          // NOTE: State access uses permuted species ordering for RCM bandwidth reduction
          const double S_0 = state(Species::N2);  // [N2]
          const double S_1 = state(Species::M);  // [M]
          const double S_2 = state(Species::H2O);  // [H2O]
          const double S_3 = state(Species::HNO4);  // [HNO4]
          const double S_4 = state(Species::HONO);  // [HONO]
          const double S_5 = state(Species::O2);  // [O2]
          const double S_6 = state(Species::HO2);  // [HO2]
          const double S_7 = state(Species::H2O2);  // [H2O2]
          const double S_8 = state(Species::OH);  // [OH]
          const double S_9 = state(Species::O3);  // [O3]
          const double S_10 = state(Species::HNO3);  // [HNO3]
          const double S_11 = state(Species::NO);  // [NO]
          const double S_12 = state(Species::O3P);  // [O3P]
          const double S_13 = state(Species::AIR);  // [AIR]
          const double S_14 = state(Species::NO3);  // [NO3]
          const double S_15 = state(Species::NO2);  // [NO2]
          const double S_16 = state(Species::N2O5);  // [N2O5]
          // The expression-dense Jacobian is evaluated by bounded compiled
          // units.  The solver still uses the same symbolic sparse LU plan.
#ifndef MKPP_USE_UNROLLED_REFERENCE
          double J_values[54] = {};
          auto jacobian_accessor = detail::StrideView<double, 2>::compact(J_values);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, temp, rh, air_density);
          double W_values[59];
          W_values[0] = inv_g_dt;
          W_values[1] = inv_g_dt;
          W_values[2] = inv_g_dt;
          W_values[3] = inv_g_dt;
          W_values[4] = inv_g_dt - J_values[33];
          W_values[5] = inv_g_dt - J_values[53];
          W_values[6] = 0.0 - J_values[51];
          W_values[7] = 0.0 - J_values[49];
          W_values[8] = 0.0 - J_values[50];
          W_values[9] = 0.0 - J_values[52];
          W_values[10] = inv_g_dt - J_values[47];
          W_values[11] = 0.0 - J_values[48];
          W_values[12] = 0.0 - J_values[7];
          W_values[13] = inv_g_dt - J_values[6];
          W_values[14] = 0.0 - J_values[8];
          W_values[15] = 0.0 - J_values[43];
          W_values[16] = 0.0 - J_values[42];
          W_values[17] = inv_g_dt - J_values[45];
          W_values[18] = 0.0 - J_values[44];
          W_values[19] = 0.0 - J_values[46];
          W_values[20] = 0.0 - J_values[5];
          W_values[21] = inv_g_dt - J_values[0];
          W_values[22] = 0.0 - J_values[1];
          W_values[23] = 0.0 - J_values[3];
          W_values[24] = 0.0 - J_values[4];
          W_values[25] = 0.0 - J_values[2];
          W_values[26] = 0.0 - J_values[35];
          W_values[27] = inv_g_dt - J_values[34];
          W_values[28] = 0.0 - J_values[36];
          W_values[29] = 0.0 - J_values[13];
          W_values[30] = 0.0 - J_values[9];
          W_values[31] = inv_g_dt - J_values[10];
          W_values[32] = 0.0 - J_values[12];
          W_values[33] = 0.0 - J_values[11];
          W_values[34] = 0.0 - J_values[41];
          W_values[35] = 0.0 - J_values[37];
          W_values[36] = inv_g_dt - J_values[39];
          W_values[37] = 0.0 - J_values[40];
          W_values[38] = 0.0 - J_values[38];
          W_values[39] = inv_g_dt;
          W_values[40] = 0.0 - J_values[27];
          W_values[41] = 0.0 - J_values[21];
          W_values[42] = 0.0 - J_values[26];
          W_values[43] = 0.0 - J_values[22];
          W_values[44] = 0.0 - J_values[28];
          W_values[45] = inv_g_dt - J_values[24];
          W_values[46] = 0.0 - J_values[23];
          W_values[47] = 0.0 - J_values[25];
          W_values[48] = 0.0 - J_values[14];
          W_values[49] = 0.0 - J_values[19];
          W_values[50] = 0.0 - J_values[15];
          W_values[51] = 0.0 - J_values[20];
          W_values[52] = 0.0 - J_values[17];
          W_values[53] = inv_g_dt - J_values[16];
          W_values[54] = 0.0 - J_values[18];
          W_values[55] = 0.0 - J_values[32];
          W_values[56] = 0.0 - J_values[30];
          W_values[57] = 0.0 - J_values[29];
          W_values[58] = inv_g_dt - J_values[31];
          double LU_values[67];
          detail::factorize_plan(W_values, LU_values);
#else
          double J_values[NUM_SPECIES * NUM_SPECIES] = {};
          auto jacobian_accessor = detail::make_stride_view<double, 2>(J_values, NUM_SPECIES, 1);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, temp, rh, air_density);
          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_4_4 = J_values[6 * NUM_SPECIES + 6];
          double J_5_5 = J_values[13 * NUM_SPECIES + 13];
          double J_5_6 = J_values[13 * NUM_SPECIES + 11];
          double J_5_9 = J_values[13 * NUM_SPECIES + 0];
          double J_5_12 = J_values[13 * NUM_SPECIES + 9];
          double J_5_13 = J_values[13 * NUM_SPECIES + 12];
          double J_6_6 = J_values[11 * NUM_SPECIES + 11];
          double J_6_13 = J_values[11 * NUM_SPECIES + 12];
          double J_7_6 = J_values[1 * NUM_SPECIES + 11];
          double J_7_7 = J_values[1 * NUM_SPECIES + 1];
          double J_7_13 = J_values[1 * NUM_SPECIES + 12];
          double J_8_4 = J_values[10 * NUM_SPECIES + 6];
          double J_8_7 = J_values[10 * NUM_SPECIES + 1];
          double J_8_8 = J_values[10 * NUM_SPECIES + 10];
          double J_8_10 = J_values[10 * NUM_SPECIES + 7];
          double J_8_13 = J_values[10 * NUM_SPECIES + 12];
          double J_9_5 = J_values[0 * NUM_SPECIES + 13];
          double J_9_9 = J_values[0 * NUM_SPECIES + 0];
          double J_9_11 = J_values[0 * NUM_SPECIES + 2];
          double J_9_12 = J_values[0 * NUM_SPECIES + 9];
          double J_9_13 = J_values[0 * NUM_SPECIES + 12];
          double J_9_15 = J_values[0 * NUM_SPECIES + 3];
          double J_10_8 = J_values[7 * NUM_SPECIES + 10];
          double J_10_10 = J_values[7 * NUM_SPECIES + 7];
          double J_10_13 = J_values[7 * NUM_SPECIES + 12];
          double J_11_4 = J_values[2 * NUM_SPECIES + 6];
          double J_11_9 = J_values[2 * NUM_SPECIES + 0];
          double J_11_11 = J_values[2 * NUM_SPECIES + 2];
          double J_11_14 = J_values[2 * NUM_SPECIES + 4];
          double J_11_15 = J_values[2 * NUM_SPECIES + 3];
          double J_12_5 = J_values[9 * NUM_SPECIES + 13];
          double J_12_9 = J_values[9 * NUM_SPECIES + 0];
          double J_12_12 = J_values[9 * NUM_SPECIES + 9];
          double J_12_13 = J_values[9 * NUM_SPECIES + 12];
          double J_12_15 = J_values[9 * NUM_SPECIES + 3];
          double J_14_8 = J_values[4 * NUM_SPECIES + 10];
          double J_14_9 = J_values[4 * NUM_SPECIES + 0];
          double J_14_10 = J_values[4 * NUM_SPECIES + 7];
          double J_14_11 = J_values[4 * NUM_SPECIES + 2];
          double J_14_13 = J_values[4 * NUM_SPECIES + 12];
          double J_14_14 = J_values[4 * NUM_SPECIES + 4];
          double J_14_15 = J_values[4 * NUM_SPECIES + 3];
          double J_14_16 = J_values[4 * NUM_SPECIES + 5];
          double J_15_9 = J_values[3 * NUM_SPECIES + 0];
          double J_15_10 = J_values[3 * NUM_SPECIES + 7];
          double J_15_11 = J_values[3 * NUM_SPECIES + 2];
          double J_15_13 = J_values[3 * NUM_SPECIES + 12];
          double J_15_14 = J_values[3 * NUM_SPECIES + 4];
          double J_15_15 = J_values[3 * NUM_SPECIES + 3];
          double J_15_16 = J_values[3 * NUM_SPECIES + 5];
          double J_16_13 = J_values[5 * NUM_SPECIES + 12];
          double J_16_14 = J_values[5 * NUM_SPECIES + 4];
          double J_16_15 = J_values[5 * NUM_SPECIES + 3];
          double J_16_16 = J_values[5 * NUM_SPECIES + 5];
          double W_values[NUM_SPECIES * NUM_SPECIES] = {};
          W_values[0 * NUM_SPECIES + 0] = inv_g_dt;
          W_values[1 * NUM_SPECIES + 1] = inv_g_dt;
          W_values[2 * NUM_SPECIES + 2] = inv_g_dt;
          W_values[3 * NUM_SPECIES + 3] = inv_g_dt;
          W_values[4 * NUM_SPECIES + 4] = inv_g_dt - J_4_4;
          W_values[5 * NUM_SPECIES + 5] = inv_g_dt - J_5_5;
          W_values[5 * NUM_SPECIES + 6] = -J_5_6;
          W_values[5 * NUM_SPECIES + 9] = -J_5_9;
          W_values[5 * NUM_SPECIES + 12] = -J_5_12;
          W_values[5 * NUM_SPECIES + 13] = -J_5_13;
          W_values[6 * NUM_SPECIES + 6] = inv_g_dt - J_6_6;
          W_values[6 * NUM_SPECIES + 13] = -J_6_13;
          W_values[7 * NUM_SPECIES + 6] = -J_7_6;
          W_values[7 * NUM_SPECIES + 7] = inv_g_dt - J_7_7;
          W_values[7 * NUM_SPECIES + 13] = -J_7_13;
          W_values[8 * NUM_SPECIES + 4] = -J_8_4;
          W_values[8 * NUM_SPECIES + 7] = -J_8_7;
          W_values[8 * NUM_SPECIES + 8] = inv_g_dt - J_8_8;
          W_values[8 * NUM_SPECIES + 10] = -J_8_10;
          W_values[8 * NUM_SPECIES + 13] = -J_8_13;
          W_values[9 * NUM_SPECIES + 5] = -J_9_5;
          W_values[9 * NUM_SPECIES + 9] = inv_g_dt - J_9_9;
          W_values[9 * NUM_SPECIES + 11] = -J_9_11;
          W_values[9 * NUM_SPECIES + 12] = -J_9_12;
          W_values[9 * NUM_SPECIES + 13] = -J_9_13;
          W_values[9 * NUM_SPECIES + 15] = -J_9_15;
          W_values[10 * NUM_SPECIES + 8] = -J_10_8;
          W_values[10 * NUM_SPECIES + 10] = inv_g_dt - J_10_10;
          W_values[10 * NUM_SPECIES + 13] = -J_10_13;
          W_values[11 * NUM_SPECIES + 4] = -J_11_4;
          W_values[11 * NUM_SPECIES + 9] = -J_11_9;
          W_values[11 * NUM_SPECIES + 11] = inv_g_dt - J_11_11;
          W_values[11 * NUM_SPECIES + 14] = -J_11_14;
          W_values[11 * NUM_SPECIES + 15] = -J_11_15;
          W_values[12 * NUM_SPECIES + 5] = -J_12_5;
          W_values[12 * NUM_SPECIES + 9] = -J_12_9;
          W_values[12 * NUM_SPECIES + 12] = inv_g_dt - J_12_12;
          W_values[12 * NUM_SPECIES + 13] = -J_12_13;
          W_values[12 * NUM_SPECIES + 15] = -J_12_15;
          W_values[13 * NUM_SPECIES + 13] = inv_g_dt;
          W_values[14 * NUM_SPECIES + 8] = -J_14_8;
          W_values[14 * NUM_SPECIES + 9] = -J_14_9;
          W_values[14 * NUM_SPECIES + 10] = -J_14_10;
          W_values[14 * NUM_SPECIES + 11] = -J_14_11;
          W_values[14 * NUM_SPECIES + 13] = -J_14_13;
          W_values[14 * NUM_SPECIES + 14] = inv_g_dt - J_14_14;
          W_values[14 * NUM_SPECIES + 15] = -J_14_15;
          W_values[14 * NUM_SPECIES + 16] = -J_14_16;
          W_values[15 * NUM_SPECIES + 9] = -J_15_9;
          W_values[15 * NUM_SPECIES + 10] = -J_15_10;
          W_values[15 * NUM_SPECIES + 11] = -J_15_11;
          W_values[15 * NUM_SPECIES + 13] = -J_15_13;
          W_values[15 * NUM_SPECIES + 14] = -J_15_14;
          W_values[15 * NUM_SPECIES + 15] = inv_g_dt - J_15_15;
          W_values[15 * NUM_SPECIES + 16] = -J_15_16;
          W_values[16 * NUM_SPECIES + 13] = -J_16_13;
          W_values[16 * NUM_SPECIES + 14] = -J_16_14;
          W_values[16 * NUM_SPECIES + 15] = -J_16_15;
          W_values[16 * NUM_SPECIES + 16] = inv_g_dt - J_16_16;
          double LU_values[NUM_SPECIES * NUM_SPECIES] = {};
          detail::factorize_lu_chunk_0(W_values, LU_values);
#endif

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F_values_1[NUM_SPECIES];
          auto rate_accessor_1 = detail::make_stride_view<double, 1>(F_values_1, 1);
          detail::compute_rates_chunk_0(state_accessor, rate_accessor_1, jvals, temp, rh, air_density);
          double F1_0 = F_values_1[16];
          double F1_1 = F_values_1[15];
          double F1_2 = F_values_1[14];
          double F1_3 = F_values_1[8];
          double F1_4 = F_values_1[6];
          double F1_5 = F_values_1[13];
          double F1_6 = F_values_1[11];
          double F1_7 = F_values_1[1];
          double F1_8 = F_values_1[10];
          double F1_9 = F_values_1[0];
          double F1_10 = F_values_1[7];
          double F1_11 = F_values_1[2];
          double F1_12 = F_values_1[9];
          double F1_13 = F_values_1[12];
          double F1_14 = F_values_1[4];
          double F1_15 = F_values_1[3];
          double F1_16 = F_values_1[5];
          double rhs_values_1[NUM_SPECIES];
          rhs_values_1[0] = F1_0;
          rhs_values_1[1] = F1_1;
          rhs_values_1[2] = F1_2;
          rhs_values_1[3] = F1_3;
          rhs_values_1[4] = F1_4;
          rhs_values_1[5] = F1_5;
          rhs_values_1[6] = F1_6;
          rhs_values_1[7] = F1_7;
          rhs_values_1[8] = F1_8;
          rhs_values_1[9] = F1_9;
          rhs_values_1[10] = F1_10;
          rhs_values_1[11] = F1_11;
          rhs_values_1[12] = F1_12;
          rhs_values_1[13] = F1_13;
          rhs_values_1[14] = F1_14;
          rhs_values_1[15] = F1_15;
          rhs_values_1[16] = F1_16;
          double K_values_1[NUM_SPECIES];
#ifndef MKPP_USE_UNROLLED_REFERENCE
          detail::solve_plan(LU_values, rhs_values_1, K_values_1);
#else
          detail::solve_lu(LU_values, rhs_values_1, K_values_1);
#endif
          double K1_0 = K_values_1[0];
          double K1_1 = K_values_1[1];
          double K1_2 = K_values_1[2];
          double K1_3 = K_values_1[3];
          double K1_4 = K_values_1[4];
          double K1_5 = K_values_1[5];
          double K1_6 = K_values_1[6];
          double K1_7 = K_values_1[7];
          double K1_8 = K_values_1[8];
          double K1_9 = K_values_1[9];
          double K1_10 = K_values_1[10];
          double K1_11 = K_values_1[11];
          double K1_12 = K_values_1[12];
          double K1_13 = K_values_1[13];
          double K1_14 = K_values_1[14];
          double K1_15 = K_values_1[15];
          double K1_16 = K_values_1[16];

          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          double Y2_4 = S_4 + K1_4;
          double Y2_5 = S_5 + K1_5;
          double Y2_6 = S_6 + K1_6;
          double Y2_7 = S_7 + K1_7;
          double Y2_8 = S_8 + K1_8;
          double Y2_9 = S_9 + K1_9;
          double Y2_10 = S_10 + K1_10;
          double Y2_11 = S_11 + K1_11;
          double Y2_12 = S_12 + K1_12;
          double Y2_13 = S_13 + K1_13;
          double Y2_14 = S_14 + K1_14;
          double Y2_15 = S_15 + K1_15;
          double Y2_16 = S_16 + K1_16;
          // Rate evaluation F2 at Y2
          double stage_state_2[NUM_SPECIES];
          stage_state_2[16] = Y2_0;
          stage_state_2[15] = Y2_1;
          stage_state_2[14] = Y2_2;
          stage_state_2[8] = Y2_3;
          stage_state_2[6] = Y2_4;
          stage_state_2[13] = Y2_5;
          stage_state_2[11] = Y2_6;
          stage_state_2[1] = Y2_7;
          stage_state_2[10] = Y2_8;
          stage_state_2[0] = Y2_9;
          stage_state_2[7] = Y2_10;
          stage_state_2[2] = Y2_11;
          stage_state_2[9] = Y2_12;
          stage_state_2[12] = Y2_13;
          stage_state_2[4] = Y2_14;
          stage_state_2[3] = Y2_15;
          stage_state_2[5] = Y2_16;
          double F_values_2[NUM_SPECIES];
          auto rate_accessor_2 = detail::make_stride_view<double, 1>(F_values_2, 1);
          auto stage_state_accessor_2 = detail::make_stride_view<double, 1>(stage_state_2, 1);
          detail::compute_rates_chunk_0(stage_state_accessor_2, rate_accessor_2, jvals, temp, rh, air_density);
          double F2_0 = F_values_2[16];
          double F2_1 = F_values_2[15];
          double F2_2 = F_values_2[14];
          double F2_3 = F_values_2[8];
          double F2_4 = F_values_2[6];
          double F2_5 = F_values_2[13];
          double F2_6 = F_values_2[11];
          double F2_7 = F_values_2[1];
          double F2_8 = F_values_2[10];
          double F2_9 = F_values_2[0];
          double F2_10 = F_values_2[7];
          double F2_11 = F_values_2[2];
          double F2_12 = F_values_2[9];
          double F2_13 = F_values_2[12];
          double F2_14 = F_values_2[4];
          double F2_15 = F_values_2[3];
          double F2_16 = F_values_2[5];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          double rhs2_7 = F2_7 + (-1.0156171083877703 / dt) * K1_7;
          double rhs2_8 = F2_8 + (-1.0156171083877703 / dt) * K1_8;
          double rhs2_9 = F2_9 + (-1.0156171083877703 / dt) * K1_9;
          double rhs2_10 = F2_10 + (-1.0156171083877703 / dt) * K1_10;
          double rhs2_11 = F2_11 + (-1.0156171083877703 / dt) * K1_11;
          double rhs2_12 = F2_12 + (-1.0156171083877703 / dt) * K1_12;
          double rhs2_13 = F2_13 + (-1.0156171083877703 / dt) * K1_13;
          double rhs2_14 = F2_14 + (-1.0156171083877703 / dt) * K1_14;
          double rhs2_15 = F2_15 + (-1.0156171083877703 / dt) * K1_15;
          double rhs2_16 = F2_16 + (-1.0156171083877703 / dt) * K1_16;
          double rhs_values_2[NUM_SPECIES];
          rhs_values_2[0] = rhs2_0;
          rhs_values_2[1] = rhs2_1;
          rhs_values_2[2] = rhs2_2;
          rhs_values_2[3] = rhs2_3;
          rhs_values_2[4] = rhs2_4;
          rhs_values_2[5] = rhs2_5;
          rhs_values_2[6] = rhs2_6;
          rhs_values_2[7] = rhs2_7;
          rhs_values_2[8] = rhs2_8;
          rhs_values_2[9] = rhs2_9;
          rhs_values_2[10] = rhs2_10;
          rhs_values_2[11] = rhs2_11;
          rhs_values_2[12] = rhs2_12;
          rhs_values_2[13] = rhs2_13;
          rhs_values_2[14] = rhs2_14;
          rhs_values_2[15] = rhs2_15;
          rhs_values_2[16] = rhs2_16;
          double K_values_2[NUM_SPECIES];
#ifndef MKPP_USE_UNROLLED_REFERENCE
          detail::solve_plan(LU_values, rhs_values_2, K_values_2);
#else
          detail::solve_lu(LU_values, rhs_values_2, K_values_2);
#endif
          double K2_0 = K_values_2[0];
          double K2_1 = K_values_2[1];
          double K2_2 = K_values_2[2];
          double K2_3 = K_values_2[3];
          double K2_4 = K_values_2[4];
          double K2_5 = K_values_2[5];
          double K2_6 = K_values_2[6];
          double K2_7 = K_values_2[7];
          double K2_8 = K_values_2[8];
          double K2_9 = K_values_2[9];
          double K2_10 = K_values_2[10];
          double K2_11 = K_values_2[11];
          double K2_12 = K_values_2[12];
          double K2_13 = K_values_2[13];
          double K2_14 = K_values_2[14];
          double K2_15 = K_values_2[15];
          double K2_16 = K_values_2[16];

          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          double Y3_4 = S_4 + K1_4;
          double Y3_5 = S_5 + K1_5;
          double Y3_6 = S_6 + K1_6;
          double Y3_7 = S_7 + K1_7;
          double Y3_8 = S_8 + K1_8;
          double Y3_9 = S_9 + K1_9;
          double Y3_10 = S_10 + K1_10;
          double Y3_11 = S_11 + K1_11;
          double Y3_12 = S_12 + K1_12;
          double Y3_13 = S_13 + K1_13;
          double Y3_14 = S_14 + K1_14;
          double Y3_15 = S_15 + K1_15;
          double Y3_16 = S_16 + K1_16;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          double rhs3_7 = F2_7 + (4.0759956452537702 / dt) * K1_7 + (9.20767942983308 / dt) * K2_7;
          double rhs3_8 = F2_8 + (4.0759956452537702 / dt) * K1_8 + (9.20767942983308 / dt) * K2_8;
          double rhs3_9 = F2_9 + (4.0759956452537702 / dt) * K1_9 + (9.20767942983308 / dt) * K2_9;
          double rhs3_10 = F2_10 + (4.0759956452537702 / dt) * K1_10 + (9.20767942983308 / dt) * K2_10;
          double rhs3_11 = F2_11 + (4.0759956452537702 / dt) * K1_11 + (9.20767942983308 / dt) * K2_11;
          double rhs3_12 = F2_12 + (4.0759956452537702 / dt) * K1_12 + (9.20767942983308 / dt) * K2_12;
          double rhs3_13 = F2_13 + (4.0759956452537702 / dt) * K1_13 + (9.20767942983308 / dt) * K2_13;
          double rhs3_14 = F2_14 + (4.0759956452537702 / dt) * K1_14 + (9.20767942983308 / dt) * K2_14;
          double rhs3_15 = F2_15 + (4.0759956452537702 / dt) * K1_15 + (9.20767942983308 / dt) * K2_15;
          double rhs3_16 = F2_16 + (4.0759956452537702 / dt) * K1_16 + (9.20767942983308 / dt) * K2_16;
          double rhs_values_3[NUM_SPECIES];
          rhs_values_3[0] = rhs3_0;
          rhs_values_3[1] = rhs3_1;
          rhs_values_3[2] = rhs3_2;
          rhs_values_3[3] = rhs3_3;
          rhs_values_3[4] = rhs3_4;
          rhs_values_3[5] = rhs3_5;
          rhs_values_3[6] = rhs3_6;
          rhs_values_3[7] = rhs3_7;
          rhs_values_3[8] = rhs3_8;
          rhs_values_3[9] = rhs3_9;
          rhs_values_3[10] = rhs3_10;
          rhs_values_3[11] = rhs3_11;
          rhs_values_3[12] = rhs3_12;
          rhs_values_3[13] = rhs3_13;
          rhs_values_3[14] = rhs3_14;
          rhs_values_3[15] = rhs3_15;
          rhs_values_3[16] = rhs3_16;
          double K_values_3[NUM_SPECIES];
#ifndef MKPP_USE_UNROLLED_REFERENCE
          detail::solve_plan(LU_values, rhs_values_3, K_values_3);
#else
          detail::solve_lu(LU_values, rhs_values_3, K_values_3);
#endif
          double K3_0 = K_values_3[0];
          double K3_1 = K_values_3[1];
          double K3_2 = K_values_3[2];
          double K3_3 = K_values_3[3];
          double K3_4 = K_values_3[4];
          double K3_5 = K_values_3[5];
          double K3_6 = K_values_3[6];
          double K3_7 = K_values_3[7];
          double K3_8 = K_values_3[8];
          double K3_9 = K_values_3[9];
          double K3_10 = K_values_3[10];
          double K3_11 = K_values_3[11];
          double K3_12 = K_values_3[12];
          double K3_13 = K_values_3[13];
          double K3_14 = K_values_3[14];
          double K3_15 = K_values_3[15];
          double K3_16 = K_values_3[16];

          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(16)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[16] + rtol[16] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(15)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[15] + rtol[15] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(14)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[14] + rtol[14] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(13)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[13] + rtol[13] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(11)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[11] + rtol[11] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_11 + K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_11 + -2.9079558716805471 * K2_11 + 0.22354069897811571 * K3_11;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_12 + K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_12 + -2.9079558716805471 * K2_12 + 0.22354069897811571 * K3_12;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_13 + K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(12)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[12] + rtol[12] * ymax;
              double yerr_i = 0.5 * K1_13 + -2.9079558716805471 * K2_13 + 0.22354069897811571 * K3_13;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_14 + K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_14 + -2.9079558716805471 * K2_14 + 0.22354069897811571 * K3_14;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_15 + K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_15 + -2.9079558716805471 * K2_15 + 0.22354069897811571 * K3_15;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_16 + K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_16 + -2.9079558716805471 * K2_16 + 0.22354069897811571 * K3_16;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 17);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(16) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(15) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(14) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(8) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(6) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(13) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(11) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              state(1) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              state(10) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              state(0) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              state(7) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              state(2) += K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              state(9) += K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              state(12) += K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              state(4) += K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              state(3) += K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              state(5) += K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              t += dt;
              double next_dt = Kokkos::fmax(h_min, Kokkos::fmin(dt * factor, dt_total));
              // MICM deliberately prevents a step-size increase immediately
              // after a rejected attempt.  This keeps the accepted-step
              // sequence stable for stiff atmospheric mechanisms.
              if (reject_last_dt) next_dt = Kokkos::fmin(next_dt, dt);
              reject_last_dt = false;
              reject_more_dt = false;
              dt = next_dt;
          } else {
              // Match MICM's rejection controller: after two consecutive
              // rejections, force a tenfold reduction instead of accepting
              // the bounded error-controller factor.
              dt *= reject_more_dt ? rejection_factor_decrease : factor;
              reject_more_dt = reject_last_dt;
              reject_last_dt = true;
          }
          } // end while (t < dt_total)
      }

#ifdef MKPP_ENABLE_REDUCTION
      template <class StateView>
      KOKKOS_INLINE_FUNCTION void integrate_with_reduction(
          double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view, double importance_threshold, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const
      {
          const double* jvals = jvals_view.data();
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          const int NUM_SPECIES = 17;
          // ROS-3 coefficients (3-stage, order 3)
          const double g = 0.435866521508459;
          const double safety = 0.9;
          const double max_growth = 6.0;
          const double min_shrink = 0.2;

          bool active[NUM_SPECIES];
          double t = 0.0;
          double dt = Kokkos::fmin(dt_total, 1.0);  // conservative initial step

          // Initialize all species as active
          active[0] = true;
          active[1] = true;
          active[2] = true;
          active[3] = true;
          active[4] = true;
          active[5] = true;
          active[6] = true;
          active[7] = true;
          active[8] = true;
          active[9] = true;
          active[10] = true;
          active[11] = true;
          active[12] = true;
          active[13] = true;
          active[14] = true;
          active[15] = true;
          active[16] = true;

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // 0. Hoist state values into scalar registers
          // NOTE: State access uses permuted species ordering
          const double S_0 = state(16);
          const double S_1 = state(15);
          const double S_2 = state(14);
          const double S_3 = state(8);
          const double S_4 = state(6);
          const double S_5 = state(13);
          const double S_6 = state(11);
          const double S_7 = state(1);
          const double S_8 = state(10);
          const double S_9 = state(0);
          const double S_10 = state(7);
          const double S_11 = state(2);
          const double S_12 = state(9);
          const double S_13 = state(12);
          const double S_14 = state(4);
          const double S_15 = state(3);
          const double S_16 = state(5);

          // 1. Stage 1 Rates (F1)
          double F1_0 = 0.0;
          double F1_1 = 0.0;
          double F1_2 = 0.0;
          double F1_3 = 0.0;
          double F1_4 = -1.0*S_4*jvals[1];
          double F1_5 = -4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) + 1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) + 1.6e-11*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_6 = -1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F1_7 = -1.0*S_7*jvals[3] + 1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F1_8 = 2.0*S_7*jvals[3] - 1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*S_10*jvals[2] + 1.0*S_4*jvals[1];
          double F1_9 = 4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) - 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp) - 8e-12*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_10 = -1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*S_10*jvals[2];
          double F1_11 = 1.0*S_4*jvals[1] - 1.8e-11*S_11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) + 1.0*S_15*jvals[0];
          double F1_12 = -4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) + 1.0*S_15*jvals[0] - 8e-12*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_13 = 0.0;
          double F1_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.8e-11*S_11*S_14*Kokkos::exp(110.0/Temp) + 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp);
          double F1_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*S_10*jvals[2] + 3.6e-11*S_11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*S_15*jvals[0];
          double F1_16 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);

          // 2. Evaluate importance and update active set
          active[0] = (Kokkos::fabs(F1_0) / (atol[0] + rtol[0] * Kokkos::fabs(state(16))) >= importance_threshold);
          active[1] = (Kokkos::fabs(F1_1) / (atol[1] + rtol[1] * Kokkos::fabs(state(15))) >= importance_threshold);
          active[2] = (Kokkos::fabs(F1_2) / (atol[2] + rtol[2] * Kokkos::fabs(state(14))) >= importance_threshold);
          active[3] = (Kokkos::fabs(F1_3) / (atol[3] + rtol[3] * Kokkos::fabs(state(8))) >= importance_threshold);
          active[4] = (Kokkos::fabs(F1_4) / (atol[4] + rtol[4] * Kokkos::fabs(state(6))) >= importance_threshold);
          active[5] = (Kokkos::fabs(F1_5) / (atol[5] + rtol[5] * Kokkos::fabs(state(13))) >= importance_threshold);
          active[6] = (Kokkos::fabs(F1_6) / (atol[6] + rtol[6] * Kokkos::fabs(state(11))) >= importance_threshold);
          active[7] = (Kokkos::fabs(F1_7) / (atol[7] + rtol[7] * Kokkos::fabs(state(1))) >= importance_threshold);
          active[8] = (Kokkos::fabs(F1_8) / (atol[8] + rtol[8] * Kokkos::fabs(state(10))) >= importance_threshold);
          active[9] = (Kokkos::fabs(F1_9) / (atol[9] + rtol[9] * Kokkos::fabs(state(0))) >= importance_threshold);
          active[10] = (Kokkos::fabs(F1_10) / (atol[10] + rtol[10] * Kokkos::fabs(state(7))) >= importance_threshold);
          active[11] = (Kokkos::fabs(F1_11) / (atol[11] + rtol[11] * Kokkos::fabs(state(2))) >= importance_threshold);
          active[12] = (Kokkos::fabs(F1_12) / (atol[12] + rtol[12] * Kokkos::fabs(state(9))) >= importance_threshold);
          active[13] = (Kokkos::fabs(F1_13) / (atol[13] + rtol[13] * Kokkos::fabs(state(12))) >= importance_threshold);
          active[14] = (Kokkos::fabs(F1_14) / (atol[14] + rtol[14] * Kokkos::fabs(state(4))) >= importance_threshold);
          active[15] = (Kokkos::fabs(F1_15) / (atol[15] + rtol[15] * Kokkos::fabs(state(3))) >= importance_threshold);
          active[16] = (Kokkos::fabs(F1_16) / (atol[16] + rtol[16] * Kokkos::fabs(state(5))) >= importance_threshold);

          // 3. Analytical Jacobian & Iteration Matrix W (identity for frozen species)
          double J_4_4 = -1.0*jvals[1];
          double J_5_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_5_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_5_9 = 1.6e-11*S_12*Kokkos::exp(-2060.0/Temp);
          double J_5_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) + 1.6e-11*S_9*Kokkos::exp(-2060.0/Temp);
          double J_5_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_6_6 = -1.7e-33*S_13*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_6_13 = -1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
          double J_7_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_7_7 = -1.0*jvals[3];
          double J_7_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
          double J_8_4 = 1.0*jvals[1];
          double J_8_7 = 2.0*jvals[3];
          double J_8_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_8_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2];
          double J_8_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_9_5 = 4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_9_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp) - 8e-12*S_12*Kokkos::exp(-2060.0/Temp);
          double J_9_11 = -1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_9_12 = 4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
          double J_9_13 = 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_9_15 = -1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
          double J_10_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_10_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2];
          double J_10_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_11_4 = 1.0*jvals[1];
          double J_11_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp);
          double J_11_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_11_14 = -1.8e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_11_15 = 1.0*jvals[0];
          double J_12_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_12_9 = -8e-12*S_12*Kokkos::exp(-2060.0/Temp);
          double J_12_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
          double J_12_13 = -4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_12_15 = 1.0*jvals[0];
          double J_14_8 = 1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_14_9 = 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
          double J_14_10 = 1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_14_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp);
          double J_14_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_14_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_14_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
          double J_14_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          double J_15_9 = 1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
          double J_15_10 = 1.0*jvals[2];
          double J_15_11 = 3.6e-11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_15_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          double J_15_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_15_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0];
          double J_15_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          double J_16_13 = -3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          double J_16_14 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
          double J_16_15 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
          double J_16_16 = -2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          double W_0_0 = active[0] ? inv_g_dt : 1.0;
          double W_1_1 = active[1] ? inv_g_dt : 1.0;
          double W_2_2 = active[2] ? inv_g_dt : 1.0;
          double W_3_3 = active[3] ? inv_g_dt : 1.0;
          double W_4_4 = active[4] ? (inv_g_dt - J_4_4) : 1.0;
          double W_5_5 = active[5] ? (inv_g_dt - J_5_5) : 1.0;
          double W_5_6 = (active[5] && active[6]) ? (-J_5_6) : 0.0;
          double W_5_9 = (active[5] && active[9]) ? (-J_5_9) : 0.0;
          double W_5_12 = (active[5] && active[12]) ? (-J_5_12) : 0.0;
          double W_5_13 = (active[5] && active[13]) ? (-J_5_13) : 0.0;
          double W_6_6 = active[6] ? (inv_g_dt - J_6_6) : 1.0;
          double W_6_13 = (active[6] && active[13]) ? (-J_6_13) : 0.0;
          double W_7_6 = (active[7] && active[6]) ? (-J_7_6) : 0.0;
          double W_7_7 = active[7] ? (inv_g_dt - J_7_7) : 1.0;
          double W_7_13 = (active[7] && active[13]) ? (-J_7_13) : 0.0;
          double W_8_4 = (active[8] && active[4]) ? (-J_8_4) : 0.0;
          double W_8_7 = (active[8] && active[7]) ? (-J_8_7) : 0.0;
          double W_8_8 = active[8] ? (inv_g_dt - J_8_8) : 1.0;
          double W_8_10 = (active[8] && active[10]) ? (-J_8_10) : 0.0;
          double W_8_13 = (active[8] && active[13]) ? (-J_8_13) : 0.0;
          double W_9_5 = (active[9] && active[5]) ? (-J_9_5) : 0.0;
          double W_9_9 = active[9] ? (inv_g_dt - J_9_9) : 1.0;
          double W_9_11 = (active[9] && active[11]) ? (-J_9_11) : 0.0;
          double W_9_12 = (active[9] && active[12]) ? (-J_9_12) : 0.0;
          double W_9_13 = (active[9] && active[13]) ? (-J_9_13) : 0.0;
          double W_9_15 = (active[9] && active[15]) ? (-J_9_15) : 0.0;
          double W_10_8 = (active[10] && active[8]) ? (-J_10_8) : 0.0;
          double W_10_10 = active[10] ? (inv_g_dt - J_10_10) : 1.0;
          double W_10_13 = (active[10] && active[13]) ? (-J_10_13) : 0.0;
          double W_11_4 = (active[11] && active[4]) ? (-J_11_4) : 0.0;
          double W_11_9 = (active[11] && active[9]) ? (-J_11_9) : 0.0;
          double W_11_11 = active[11] ? (inv_g_dt - J_11_11) : 1.0;
          double W_11_14 = (active[11] && active[14]) ? (-J_11_14) : 0.0;
          double W_11_15 = (active[11] && active[15]) ? (-J_11_15) : 0.0;
          double W_12_5 = (active[12] && active[5]) ? (-J_12_5) : 0.0;
          double W_12_9 = (active[12] && active[9]) ? (-J_12_9) : 0.0;
          double W_12_12 = active[12] ? (inv_g_dt - J_12_12) : 1.0;
          double W_12_13 = (active[12] && active[13]) ? (-J_12_13) : 0.0;
          double W_12_15 = (active[12] && active[15]) ? (-J_12_15) : 0.0;
          double W_13_13 = active[13] ? inv_g_dt : 1.0;
          double W_14_8 = (active[14] && active[8]) ? (-J_14_8) : 0.0;
          double W_14_9 = (active[14] && active[9]) ? (-J_14_9) : 0.0;
          double W_14_10 = (active[14] && active[10]) ? (-J_14_10) : 0.0;
          double W_14_11 = (active[14] && active[11]) ? (-J_14_11) : 0.0;
          double W_14_13 = (active[14] && active[13]) ? (-J_14_13) : 0.0;
          double W_14_14 = active[14] ? (inv_g_dt - J_14_14) : 1.0;
          double W_14_15 = (active[14] && active[15]) ? (-J_14_15) : 0.0;
          double W_14_16 = (active[14] && active[16]) ? (-J_14_16) : 0.0;
          double W_15_9 = (active[15] && active[9]) ? (-J_15_9) : 0.0;
          double W_15_10 = (active[15] && active[10]) ? (-J_15_10) : 0.0;
          double W_15_11 = (active[15] && active[11]) ? (-J_15_11) : 0.0;
          double W_15_13 = (active[15] && active[13]) ? (-J_15_13) : 0.0;
          double W_15_14 = (active[15] && active[14]) ? (-J_15_14) : 0.0;
          double W_15_15 = active[15] ? (inv_g_dt - J_15_15) : 1.0;
          double W_15_16 = (active[15] && active[16]) ? (-J_15_16) : 0.0;
          double W_16_13 = (active[16] && active[13]) ? (-J_16_13) : 0.0;
          double W_16_14 = (active[16] && active[14]) ? (-J_16_14) : 0.0;
          double W_16_15 = (active[16] && active[15]) ? (-J_16_15) : 0.0;
          double W_16_16 = active[16] ? (inv_g_dt - J_16_16) : 1.0;

          // 4. Symbolic LU Factorization (conditional skip for frozen species)
          double U_0_0 = W_0_0;
          double U_1_1 = W_1_1;
          double U_2_2 = W_2_2;
          double U_3_3 = W_3_3;
          double U_4_4 = W_4_4;
          double L_8_4 = (W_8_4) / U_4_4;
          double L_11_4 = (W_11_4) / U_4_4;
          double U_5_5 = W_5_5;
          double U_5_6 = W_5_6;
          double U_5_9 = W_5_9;
          double U_5_12 = W_5_12;
          double U_5_13 = W_5_13;
          double L_9_5 = (W_9_5) / U_5_5;
          double L_12_5 = (W_12_5) / U_5_5;
          double U_6_6 = W_6_6;
          double U_6_13 = W_6_13;
          double L_7_6 = (W_7_6) / U_6_6;
          double L_9_6 = (0.0 - L_9_5 * U_5_6) / U_6_6;
          double L_12_6 = (0.0 - L_12_5 * U_5_6) / U_6_6;
          double U_7_7 = W_7_7;
          double U_7_13 = W_7_13 - L_7_6 * U_6_13;
          double L_8_7 = (W_8_7) / U_7_7;
          double U_8_8 = W_8_8;
          double U_8_10 = W_8_10;
          double U_8_13 = W_8_13 - L_8_7 * U_7_13;
          double L_10_8 = (W_10_8) / U_8_8;
          double L_14_8 = (W_14_8) / U_8_8;
          double U_9_9 = W_9_9 - L_9_5 * U_5_9;
          double U_9_11 = W_9_11;
          double U_9_12 = W_9_12 - L_9_5 * U_5_12;
          double U_9_13 = W_9_13 - L_9_5 * U_5_13 - L_9_6 * U_6_13;
          double U_9_15 = W_9_15;
          double L_11_9 = (W_11_9) / U_9_9;
          double L_12_9 = (W_12_9 - L_12_5 * U_5_9) / U_9_9;
          double L_14_9 = (W_14_9) / U_9_9;
          double L_15_9 = (W_15_9) / U_9_9;
          double U_10_10 = W_10_10 - L_10_8 * U_8_10;
          double U_10_13 = W_10_13 - L_10_8 * U_8_13;
          double L_14_10 = (W_14_10 - L_14_8 * U_8_10) / U_10_10;
          double L_15_10 = (W_15_10) / U_10_10;
          double U_11_11 = W_11_11 - L_11_9 * U_9_11;
          double U_11_12 = 0.0 - L_11_9 * U_9_12;
          double U_11_13 = 0.0 - L_11_9 * U_9_13;
          double U_11_14 = W_11_14;
          double U_11_15 = W_11_15 - L_11_9 * U_9_15;
          double L_12_11 = (0.0 - L_12_9 * U_9_11) / U_11_11;
          double L_14_11 = (W_14_11 - L_14_9 * U_9_11) / U_11_11;
          double L_15_11 = (W_15_11 - L_15_9 * U_9_11) / U_11_11;
          double U_12_12 = W_12_12 - L_12_5 * U_5_12 - L_12_9 * U_9_12 - L_12_11 * U_11_12;
          double U_12_13 = W_12_13 - L_12_5 * U_5_13 - L_12_6 * U_6_13 - L_12_9 * U_9_13 - L_12_11 * U_11_13;
          double U_12_14 = 0.0 - L_12_11 * U_11_14;
          double U_12_15 = W_12_15 - L_12_9 * U_9_15 - L_12_11 * U_11_15;
          double L_14_12 = (0.0 - L_14_9 * U_9_12 - L_14_11 * U_11_12) / U_12_12;
          double L_15_12 = (0.0 - L_15_9 * U_9_12 - L_15_11 * U_11_12) / U_12_12;
          double U_13_13 = W_13_13;
          double L_14_13 = (W_14_13 - L_14_8 * U_8_13 - L_14_9 * U_9_13 - L_14_10 * U_10_13 - L_14_11 * U_11_13 - L_14_12 * U_12_13) / U_13_13;
          double L_15_13 = (W_15_13 - L_15_9 * U_9_13 - L_15_10 * U_10_13 - L_15_11 * U_11_13 - L_15_12 * U_12_13) / U_13_13;
          double L_16_13 = (W_16_13) / U_13_13;
          double U_14_14 = W_14_14 - L_14_11 * U_11_14 - L_14_12 * U_12_14;
          double U_14_15 = W_14_15 - L_14_9 * U_9_15 - L_14_11 * U_11_15 - L_14_12 * U_12_15;
          double U_14_16 = W_14_16;
          double L_15_14 = (W_15_14 - L_15_11 * U_11_14 - L_15_12 * U_12_14) / U_14_14;
          double L_16_14 = (W_16_14) / U_14_14;
          double U_15_15 = W_15_15 - L_15_9 * U_9_15 - L_15_11 * U_11_15 - L_15_12 * U_12_15 - L_15_14 * U_14_15;
          double U_15_16 = W_15_16 - L_15_14 * U_14_16;
          double L_16_15 = (W_16_15 - L_16_14 * U_14_15) / U_15_15;
          double U_16_16 = W_16_16 - L_16_14 * U_14_16 - L_16_15 * U_15_16;

          // --- Stage 1 ---
          // F1 already computed above (used for importance evaluation)
          // Block 0: K1 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y1_0 = active[0] ? (F1_0) : 0.0;
          // Block 1: K1 forward sub [M]
          double y1_1 = active[1] ? (F1_1) : 0.0;
          double y1_2 = active[2] ? (F1_2) : 0.0;
          double y1_3 = active[3] ? (F1_3) : 0.0;
          double y1_4 = active[4] ? (F1_4) : 0.0;
          double y1_5 = active[5] ? (F1_5) : 0.0;
          // Block 2: K1 forward sub [HO2]
          double y1_6 = active[6] ? (F1_6) : 0.0;
          // Block 3: K1 forward sub [H2O2, HNO3]
          double y1_7 = active[7] ? (F1_7 - L_7_6 * y1_6) : 0.0;
          // Block 4: K1 forward sub [OH]
          double y1_8 = active[8] ? (F1_8 - L_8_4 * y1_4 - L_8_7 * y1_7) : 0.0;
          double y1_9 = active[9] ? (F1_9 - L_9_5 * y1_5 - L_9_6 * y1_6) : 0.0;
          double y1_10 = active[10] ? (F1_10 - L_10_8 * y1_8) : 0.0;
          // Block 5: K1 forward sub [NO]
          double y1_11 = active[11] ? (F1_11 - L_11_4 * y1_4 - L_11_9 * y1_9) : 0.0;
          // Block 6: K1 forward sub [O3P]
          double y1_12 = active[12] ? (F1_12 - L_12_5 * y1_5 - L_12_6 * y1_6 - L_12_9 * y1_9 - L_12_11 * y1_11) : 0.0;
          double y1_13 = active[13] ? (F1_13) : 0.0;
          // Block 7: K1 forward sub [NO3]
          double y1_14 = active[14] ? (F1_14 - L_14_8 * y1_8 - L_14_9 * y1_9 - L_14_10 * y1_10 - L_14_11 * y1_11 - L_14_12 * y1_12 - L_14_13 * y1_13) : 0.0;
          // Block 8: K1 forward sub [NO2]
          double y1_15 = active[15] ? (F1_15 - L_15_9 * y1_9 - L_15_10 * y1_10 - L_15_11 * y1_11 - L_15_12 * y1_12 - L_15_13 * y1_13 - L_15_14 * y1_14) : 0.0;
          // Block 9: K1 forward sub [N2O5]
          double y1_16 = active[16] ? (F1_16 - L_16_13 * y1_13 - L_16_14 * y1_14 - L_16_15 * y1_15) : 0.0;

          // Block 9: K1 backward sub [N2O5]
          double K1_16 = active[16] ? (y1_16 / U_16_16) : 0.0;
          // Block 8: K1 backward sub [NO2]
          double K1_15 = active[15] ? ((y1_15 - U_15_16 * K1_16) / U_15_15) : 0.0;
          // Block 7: K1 backward sub [NO3]
          double K1_14 = active[14] ? ((y1_14 - U_14_15 * K1_15 - U_14_16 * K1_16) / U_14_14) : 0.0;
          // Block 0: K1 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K1_13 = active[13] ? (y1_13 / U_13_13) : 0.0;
          // Block 6: K1 backward sub [O3P]
          double K1_12 = active[12] ? ((y1_12 - U_12_13 * K1_13 - U_12_14 * K1_14 - U_12_15 * K1_15) / U_12_12) : 0.0;
          // Block 5: K1 backward sub [NO]
          double K1_11 = active[11] ? ((y1_11 - U_11_12 * K1_12 - U_11_13 * K1_13 - U_11_14 * K1_14 - U_11_15 * K1_15) / U_11_11) : 0.0;
          // Block 3: K1 backward sub [H2O2, HNO3]
          double K1_10 = active[10] ? ((y1_10 - U_10_13 * K1_13) / U_10_10) : 0.0;
          double K1_9 = active[9] ? ((y1_9 - U_9_11 * K1_11 - U_9_12 * K1_12 - U_9_13 * K1_13 - U_9_15 * K1_15) / U_9_9) : 0.0;
          // Block 4: K1 backward sub [OH]
          double K1_8 = active[8] ? ((y1_8 - U_8_10 * K1_10 - U_8_13 * K1_13) / U_8_8) : 0.0;
          double K1_7 = active[7] ? ((y1_7 - U_7_13 * K1_13) / U_7_7) : 0.0;
          // Block 2: K1 backward sub [HO2]
          double K1_6 = active[6] ? ((y1_6 - U_6_13 * K1_13) / U_6_6) : 0.0;
          double K1_5 = active[5] ? ((y1_5 - U_5_6 * K1_6 - U_5_9 * K1_9 - U_5_12 * K1_12 - U_5_13 * K1_13) / U_5_5) : 0.0;
          double K1_4 = active[4] ? (y1_4 / U_4_4) : 0.0;
          double K1_3 = active[3] ? (y1_3 / U_3_3) : 0.0;
          double K1_2 = active[2] ? (y1_2 / U_2_2) : 0.0;
          // Block 1: K1 backward sub [M]
          double K1_1 = active[1] ? (y1_1 / U_1_1) : 0.0;
          double K1_0 = active[0] ? (y1_0 / U_0_0) : 0.0;


          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          double Y2_4 = S_4 + K1_4;
          double Y2_5 = S_5 + K1_5;
          double Y2_6 = S_6 + K1_6;
          double Y2_7 = S_7 + K1_7;
          double Y2_8 = S_8 + K1_8;
          double Y2_9 = S_9 + K1_9;
          double Y2_10 = S_10 + K1_10;
          double Y2_11 = S_11 + K1_11;
          double Y2_12 = S_12 + K1_12;
          double Y2_13 = S_13 + K1_13;
          double Y2_14 = S_14 + K1_14;
          double Y2_15 = S_15 + K1_15;
          double Y2_16 = S_16 + K1_16;
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = 0.0;
          double F2_2 = 0.0;
          double F2_3 = 0.0;
          double F2_4 = -1.0*Y2_4*jvals[1];
          double F2_5 = -4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) + 1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) + 1.6e-11*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_6 = -1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F2_7 = -1.0*Y2_7*jvals[3] + 1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F2_8 = 2.0*Y2_7*jvals[3] - 1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*Y2_10*jvals[2] + 1.0*Y2_4*jvals[1];
          double F2_9 = 4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) - 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp) - 8e-12*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_10 = -1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*Y2_10*jvals[2];
          double F2_11 = 1.0*Y2_4*jvals[1] - 1.8e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) - 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) + 1.0*Y2_15*jvals[0];
          double F2_12 = -4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) + 1.0*Y2_15*jvals[0] - 8e-12*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_13 = 0.0;
          double F2_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.8e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) + 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp);
          double F2_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*Y2_10*jvals[2] + 3.6e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) + 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp) - 1.0*Y2_15*jvals[0];
          double F2_16 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          double rhs2_7 = F2_7 + (-1.0156171083877703 / dt) * K1_7;
          double rhs2_8 = F2_8 + (-1.0156171083877703 / dt) * K1_8;
          double rhs2_9 = F2_9 + (-1.0156171083877703 / dt) * K1_9;
          double rhs2_10 = F2_10 + (-1.0156171083877703 / dt) * K1_10;
          double rhs2_11 = F2_11 + (-1.0156171083877703 / dt) * K1_11;
          double rhs2_12 = F2_12 + (-1.0156171083877703 / dt) * K1_12;
          double rhs2_13 = F2_13 + (-1.0156171083877703 / dt) * K1_13;
          double rhs2_14 = F2_14 + (-1.0156171083877703 / dt) * K1_14;
          double rhs2_15 = F2_15 + (-1.0156171083877703 / dt) * K1_15;
          double rhs2_16 = F2_16 + (-1.0156171083877703 / dt) * K1_16;
          // Block 0: K2 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y2_0 = active[0] ? (rhs2_0) : 0.0;
          // Block 1: K2 forward sub [M]
          double y2_1 = active[1] ? (rhs2_1) : 0.0;
          double y2_2 = active[2] ? (rhs2_2) : 0.0;
          double y2_3 = active[3] ? (rhs2_3) : 0.0;
          double y2_4 = active[4] ? (rhs2_4) : 0.0;
          double y2_5 = active[5] ? (rhs2_5) : 0.0;
          // Block 2: K2 forward sub [HO2]
          double y2_6 = active[6] ? (rhs2_6) : 0.0;
          // Block 3: K2 forward sub [H2O2, HNO3]
          double y2_7 = active[7] ? (rhs2_7 - L_7_6 * y2_6) : 0.0;
          // Block 4: K2 forward sub [OH]
          double y2_8 = active[8] ? (rhs2_8 - L_8_4 * y2_4 - L_8_7 * y2_7) : 0.0;
          double y2_9 = active[9] ? (rhs2_9 - L_9_5 * y2_5 - L_9_6 * y2_6) : 0.0;
          double y2_10 = active[10] ? (rhs2_10 - L_10_8 * y2_8) : 0.0;
          // Block 5: K2 forward sub [NO]
          double y2_11 = active[11] ? (rhs2_11 - L_11_4 * y2_4 - L_11_9 * y2_9) : 0.0;
          // Block 6: K2 forward sub [O3P]
          double y2_12 = active[12] ? (rhs2_12 - L_12_5 * y2_5 - L_12_6 * y2_6 - L_12_9 * y2_9 - L_12_11 * y2_11) : 0.0;
          double y2_13 = active[13] ? (rhs2_13) : 0.0;
          // Block 7: K2 forward sub [NO3]
          double y2_14 = active[14] ? (rhs2_14 - L_14_8 * y2_8 - L_14_9 * y2_9 - L_14_10 * y2_10 - L_14_11 * y2_11 - L_14_12 * y2_12 - L_14_13 * y2_13) : 0.0;
          // Block 8: K2 forward sub [NO2]
          double y2_15 = active[15] ? (rhs2_15 - L_15_9 * y2_9 - L_15_10 * y2_10 - L_15_11 * y2_11 - L_15_12 * y2_12 - L_15_13 * y2_13 - L_15_14 * y2_14) : 0.0;
          // Block 9: K2 forward sub [N2O5]
          double y2_16 = active[16] ? (rhs2_16 - L_16_13 * y2_13 - L_16_14 * y2_14 - L_16_15 * y2_15) : 0.0;

          // Block 9: K2 backward sub [N2O5]
          double K2_16 = active[16] ? (y2_16 / U_16_16) : 0.0;
          // Block 8: K2 backward sub [NO2]
          double K2_15 = active[15] ? ((y2_15 - U_15_16 * K2_16) / U_15_15) : 0.0;
          // Block 7: K2 backward sub [NO3]
          double K2_14 = active[14] ? ((y2_14 - U_14_15 * K2_15 - U_14_16 * K2_16) / U_14_14) : 0.0;
          // Block 0: K2 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K2_13 = active[13] ? (y2_13 / U_13_13) : 0.0;
          // Block 6: K2 backward sub [O3P]
          double K2_12 = active[12] ? ((y2_12 - U_12_13 * K2_13 - U_12_14 * K2_14 - U_12_15 * K2_15) / U_12_12) : 0.0;
          // Block 5: K2 backward sub [NO]
          double K2_11 = active[11] ? ((y2_11 - U_11_12 * K2_12 - U_11_13 * K2_13 - U_11_14 * K2_14 - U_11_15 * K2_15) / U_11_11) : 0.0;
          // Block 3: K2 backward sub [H2O2, HNO3]
          double K2_10 = active[10] ? ((y2_10 - U_10_13 * K2_13) / U_10_10) : 0.0;
          double K2_9 = active[9] ? ((y2_9 - U_9_11 * K2_11 - U_9_12 * K2_12 - U_9_13 * K2_13 - U_9_15 * K2_15) / U_9_9) : 0.0;
          // Block 4: K2 backward sub [OH]
          double K2_8 = active[8] ? ((y2_8 - U_8_10 * K2_10 - U_8_13 * K2_13) / U_8_8) : 0.0;
          double K2_7 = active[7] ? ((y2_7 - U_7_13 * K2_13) / U_7_7) : 0.0;
          // Block 2: K2 backward sub [HO2]
          double K2_6 = active[6] ? ((y2_6 - U_6_13 * K2_13) / U_6_6) : 0.0;
          double K2_5 = active[5] ? ((y2_5 - U_5_6 * K2_6 - U_5_9 * K2_9 - U_5_12 * K2_12 - U_5_13 * K2_13) / U_5_5) : 0.0;
          double K2_4 = active[4] ? (y2_4 / U_4_4) : 0.0;
          double K2_3 = active[3] ? (y2_3 / U_3_3) : 0.0;
          double K2_2 = active[2] ? (y2_2 / U_2_2) : 0.0;
          // Block 1: K2 backward sub [M]
          double K2_1 = active[1] ? (y2_1 / U_1_1) : 0.0;
          double K2_0 = active[0] ? (y2_0 / U_0_0) : 0.0;


          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          double Y3_4 = S_4 + K1_4;
          double Y3_5 = S_5 + K1_5;
          double Y3_6 = S_6 + K1_6;
          double Y3_7 = S_7 + K1_7;
          double Y3_8 = S_8 + K1_8;
          double Y3_9 = S_9 + K1_9;
          double Y3_10 = S_10 + K1_10;
          double Y3_11 = S_11 + K1_11;
          double Y3_12 = S_12 + K1_12;
          double Y3_13 = S_13 + K1_13;
          double Y3_14 = S_14 + K1_14;
          double Y3_15 = S_15 + K1_15;
          double Y3_16 = S_16 + K1_16;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          double rhs3_7 = F2_7 + (4.0759956452537702 / dt) * K1_7 + (9.20767942983308 / dt) * K2_7;
          double rhs3_8 = F2_8 + (4.0759956452537702 / dt) * K1_8 + (9.20767942983308 / dt) * K2_8;
          double rhs3_9 = F2_9 + (4.0759956452537702 / dt) * K1_9 + (9.20767942983308 / dt) * K2_9;
          double rhs3_10 = F2_10 + (4.0759956452537702 / dt) * K1_10 + (9.20767942983308 / dt) * K2_10;
          double rhs3_11 = F2_11 + (4.0759956452537702 / dt) * K1_11 + (9.20767942983308 / dt) * K2_11;
          double rhs3_12 = F2_12 + (4.0759956452537702 / dt) * K1_12 + (9.20767942983308 / dt) * K2_12;
          double rhs3_13 = F2_13 + (4.0759956452537702 / dt) * K1_13 + (9.20767942983308 / dt) * K2_13;
          double rhs3_14 = F2_14 + (4.0759956452537702 / dt) * K1_14 + (9.20767942983308 / dt) * K2_14;
          double rhs3_15 = F2_15 + (4.0759956452537702 / dt) * K1_15 + (9.20767942983308 / dt) * K2_15;
          double rhs3_16 = F2_16 + (4.0759956452537702 / dt) * K1_16 + (9.20767942983308 / dt) * K2_16;
          // Block 0: K3 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y3_0 = active[0] ? (rhs3_0) : 0.0;
          // Block 1: K3 forward sub [M]
          double y3_1 = active[1] ? (rhs3_1) : 0.0;
          double y3_2 = active[2] ? (rhs3_2) : 0.0;
          double y3_3 = active[3] ? (rhs3_3) : 0.0;
          double y3_4 = active[4] ? (rhs3_4) : 0.0;
          double y3_5 = active[5] ? (rhs3_5) : 0.0;
          // Block 2: K3 forward sub [HO2]
          double y3_6 = active[6] ? (rhs3_6) : 0.0;
          // Block 3: K3 forward sub [H2O2, HNO3]
          double y3_7 = active[7] ? (rhs3_7 - L_7_6 * y3_6) : 0.0;
          // Block 4: K3 forward sub [OH]
          double y3_8 = active[8] ? (rhs3_8 - L_8_4 * y3_4 - L_8_7 * y3_7) : 0.0;
          double y3_9 = active[9] ? (rhs3_9 - L_9_5 * y3_5 - L_9_6 * y3_6) : 0.0;
          double y3_10 = active[10] ? (rhs3_10 - L_10_8 * y3_8) : 0.0;
          // Block 5: K3 forward sub [NO]
          double y3_11 = active[11] ? (rhs3_11 - L_11_4 * y3_4 - L_11_9 * y3_9) : 0.0;
          // Block 6: K3 forward sub [O3P]
          double y3_12 = active[12] ? (rhs3_12 - L_12_5 * y3_5 - L_12_6 * y3_6 - L_12_9 * y3_9 - L_12_11 * y3_11) : 0.0;
          double y3_13 = active[13] ? (rhs3_13) : 0.0;
          // Block 7: K3 forward sub [NO3]
          double y3_14 = active[14] ? (rhs3_14 - L_14_8 * y3_8 - L_14_9 * y3_9 - L_14_10 * y3_10 - L_14_11 * y3_11 - L_14_12 * y3_12 - L_14_13 * y3_13) : 0.0;
          // Block 8: K3 forward sub [NO2]
          double y3_15 = active[15] ? (rhs3_15 - L_15_9 * y3_9 - L_15_10 * y3_10 - L_15_11 * y3_11 - L_15_12 * y3_12 - L_15_13 * y3_13 - L_15_14 * y3_14) : 0.0;
          // Block 9: K3 forward sub [N2O5]
          double y3_16 = active[16] ? (rhs3_16 - L_16_13 * y3_13 - L_16_14 * y3_14 - L_16_15 * y3_15) : 0.0;

          // Block 9: K3 backward sub [N2O5]
          double K3_16 = active[16] ? (y3_16 / U_16_16) : 0.0;
          // Block 8: K3 backward sub [NO2]
          double K3_15 = active[15] ? ((y3_15 - U_15_16 * K3_16) / U_15_15) : 0.0;
          // Block 7: K3 backward sub [NO3]
          double K3_14 = active[14] ? ((y3_14 - U_14_15 * K3_15 - U_14_16 * K3_16) / U_14_14) : 0.0;
          // Block 0: K3 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K3_13 = active[13] ? (y3_13 / U_13_13) : 0.0;
          // Block 6: K3 backward sub [O3P]
          double K3_12 = active[12] ? ((y3_12 - U_12_13 * K3_13 - U_12_14 * K3_14 - U_12_15 * K3_15) / U_12_12) : 0.0;
          // Block 5: K3 backward sub [NO]
          double K3_11 = active[11] ? ((y3_11 - U_11_12 * K3_12 - U_11_13 * K3_13 - U_11_14 * K3_14 - U_11_15 * K3_15) / U_11_11) : 0.0;
          // Block 3: K3 backward sub [H2O2, HNO3]
          double K3_10 = active[10] ? ((y3_10 - U_10_13 * K3_13) / U_10_10) : 0.0;
          double K3_9 = active[9] ? ((y3_9 - U_9_11 * K3_11 - U_9_12 * K3_12 - U_9_13 * K3_13 - U_9_15 * K3_15) / U_9_9) : 0.0;
          // Block 4: K3 backward sub [OH]
          double K3_8 = active[8] ? ((y3_8 - U_8_10 * K3_10 - U_8_13 * K3_13) / U_8_8) : 0.0;
          double K3_7 = active[7] ? ((y3_7 - U_7_13 * K3_13) / U_7_7) : 0.0;
          // Block 2: K3 backward sub [HO2]
          double K3_6 = active[6] ? ((y3_6 - U_6_13 * K3_13) / U_6_6) : 0.0;
          double K3_5 = active[5] ? ((y3_5 - U_5_6 * K3_6 - U_5_9 * K3_9 - U_5_12 * K3_12 - U_5_13 * K3_13) / U_5_5) : 0.0;
          double K3_4 = active[4] ? (y3_4 / U_4_4) : 0.0;
          double K3_3 = active[3] ? (y3_3 / U_3_3) : 0.0;
          double K3_2 = active[2] ? (y3_2 / U_2_2) : 0.0;
          // Block 1: K3 backward sub [M]
          double K3_1 = active[1] ? (y3_1 / U_1_1) : 0.0;
          double K3_0 = active[0] ? (y3_0 / U_0_0) : 0.0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(16)), Kokkos::fabs(Ynew_i));
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(15)), Kokkos::fabs(Ynew_i));
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(14)), Kokkos::fabs(Ynew_i));
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(13)), Kokkos::fabs(Ynew_i));
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(11)), Kokkos::fabs(Ynew_i));
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_11 + K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              double sci = atol[11] + rtol[11] * ymax;
              double yerr_i = 0.5 * K1_11 + -2.9079558716805471 * K2_11 + 0.22354069897811571 * K3_11;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_12 + K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              double sci = atol[12] + rtol[12] * ymax;
              double yerr_i = 0.5 * K1_12 + -2.9079558716805471 * K2_12 + 0.22354069897811571 * K3_12;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_13 + K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(12)), Kokkos::fabs(Ynew_i));
              double sci = atol[13] + rtol[13] * ymax;
              double yerr_i = 0.5 * K1_13 + -2.9079558716805471 * K2_13 + 0.22354069897811571 * K3_13;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_14 + K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              double sci = atol[14] + rtol[14] * ymax;
              double yerr_i = 0.5 * K1_14 + -2.9079558716805471 * K2_14 + 0.22354069897811571 * K3_14;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_15 + K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              double sci = atol[15] + rtol[15] * ymax;
              double yerr_i = 0.5 * K1_15 + -2.9079558716805471 * K2_15 + 0.22354069897811571 * K3_15;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_16 + K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              double sci = atol[16] + rtol[16] * ymax;
              double yerr_i = 0.5 * K1_16 + -2.9079558716805471 * K2_16 + 0.22354069897811571 * K3_16;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 17);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              if (active[0]) state(16) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              if (active[1]) state(15) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              if (active[2]) state(14) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              if (active[3]) state(8) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              if (active[4]) state(6) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              if (active[5]) state(13) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              if (active[6]) state(11) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              if (active[7]) state(1) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              if (active[8]) state(10) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              if (active[9]) state(0) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              if (active[10]) state(7) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              if (active[11]) state(2) += K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              if (active[12]) state(9) += K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              if (active[13]) state(12) += K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              if (active[14]) state(4) += K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              if (active[15]) state(3) += K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              if (active[16]) state(5) += K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              t += dt;
              dt *= factor;
          } else {
              dt *= factor;
          }
          } // end while (t < dt_total)
      }
#endif
#ifdef MKPP_ENABLE_ADJOINT


      // Forward integration with checkpointing for adjoint/TLM
      // Saves step size h and state at each accepted step into CheckpointBuffer.
      // Returns number of accepted steps, or -1 if MAX_STEPS exceeded.
      template <class StateView>
      KOKKOS_INLINE_FUNCTION int integrate_fwd_checkpoint(
          double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view,
          CheckpointBuffer& chk, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const
      {
          const double* jvals = jvals_view.data();
          const int NUM_SPECIES = 17;
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          // ROS-3 coefficients (3-stage, order 3)
          const double g = 0.435866521508459;
          const double safety = 0.9;
          const double max_growth = 6.0;
          const double min_shrink = 0.2;
          int ierr = 0;
          chk.num_steps = 0;
          double t = 0.0;
          double dt = Kokkos::fmin(dt_total, 1.0);  // conservative initial step

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // 0. Hoist state values into scalar registers
          // NOTE: State access uses permuted species ordering
          const double S_0 = state(Species::N2);  // [N2]
          const double S_1 = state(Species::M);  // [M]
          const double S_2 = state(Species::H2O);  // [H2O]
          const double S_3 = state(Species::HNO4);  // [HNO4]
          const double S_4 = state(Species::HONO);  // [HONO]
          const double S_5 = state(Species::O2);  // [O2]
          const double S_6 = state(Species::HO2);  // [HO2]
          const double S_7 = state(Species::H2O2);  // [H2O2]
          const double S_8 = state(Species::OH);  // [OH]
          const double S_9 = state(Species::O3);  // [O3]
          const double S_10 = state(Species::HNO3);  // [HNO3]
          const double S_11 = state(Species::NO);  // [NO]
          const double S_12 = state(Species::O3P);  // [O3P]
          const double S_13 = state(Species::AIR);  // [AIR]
          const double S_14 = state(Species::NO3);  // [NO3]
          const double S_15 = state(Species::NO2);  // [NO2]
          const double S_16 = state(Species::N2O5);  // [N2O5]

          // --- CSE Temporaries ---
          const double cse_tmp_0 = 1.0/Temp;
          const double cse_tmp_1 = 1.4e-18*S_12*Kokkos::exp(3.7*cse_tmp_0);
          const double cse_tmp_2 = Kokkos::pow(Kokkos::log(10.0), -2);
          const double cse_tmp_3 = 1.24621148321367e-27*S_12*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0*cse_tmp_0);

          // --- Reaction Rate Fluxes R_m ---
          const double R_0 = S_3*jvals[0];
          const double R_1 = 4.9010355073085e-27*S_12*S_13*S_9*Kokkos::pow(Temp, -2.8);
          const double R_2 = 8e-12*S_0*S_9*Kokkos::exp(-2060.0*cse_tmp_0);
          const double R_3 = 1.8e-12*S_2*S_0*Kokkos::exp(-1370.0*cse_tmp_0);
          const double R_4 = 1.4e-13*S_3*S_0*Kokkos::exp(-2470.0*cse_tmp_0);
          const double R_5 = 1.8e-11*S_2*S_4*Kokkos::exp(110.0*cse_tmp_0);
          const double R_6 = 2.8e-30*Kokkos::pow(0.45, 1.0/(cse_tmp_2*Kokkos::pow(Kokkos::log(cse_tmp_1), 2) + 1.0))*S_12*S_3*S_4*Kokkos::exp(3.5*cse_tmp_0)/(cse_tmp_1 + 1.0);
          const double R_7 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(cse_tmp_2*Kokkos::pow(Kokkos::log(cse_tmp_3), 2) + 1.0))*S_12*S_5*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0*cse_tmp_0)/(cse_tmp_3 + 1.0);
          const double R_8 = S_7*S_10*(1.9e-33*S_12*Kokkos::exp(725.0*cse_tmp_0)/(4.63414634146341e-18*S_12*Kokkos::exp(-715.0*cse_tmp_0) + 1.0) + 7.2e-15*Kokkos::exp(785.0*cse_tmp_0));
          const double R_9 = S_11*(1.7e-33*S_12*Kokkos::exp(1000.0*cse_tmp_0) + 2.3e-13*Kokkos::exp(600.0*cse_tmp_0));
          const double R_10 = S_6*jvals[1];
          const double R_11 = S_7*jvals[2];
          const double R_12 = S_1*jvals[3];

          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_4_4 = -1.0*jvals[1];
          double J_5_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_5_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_5_9 = 1.6e-11*S_12*Kokkos::exp(-2060.0/Temp);
          double J_5_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) + 1.6e-11*S_9*Kokkos::exp(-2060.0/Temp);
          double J_5_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_6_6 = -1.7e-33*S_13*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_6_13 = -1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
          double J_7_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
          double J_7_7 = -1.0*jvals[3];
          double J_7_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
          double J_8_4 = 1.0*jvals[1];
          double J_8_7 = 2.0*jvals[3];
          double J_8_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_8_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2];
          double J_8_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_9_5 = 4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_9_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp) - 8e-12*S_12*Kokkos::exp(-2060.0/Temp);
          double J_9_11 = -1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_9_12 = 4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
          double J_9_13 = 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_9_15 = -1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
          double J_10_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_10_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2];
          double J_10_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_11_4 = 1.0*jvals[1];
          double J_11_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp);
          double J_11_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_11_14 = -1.8e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_11_15 = 1.0*jvals[0];
          double J_12_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
          double J_12_9 = -8e-12*S_12*Kokkos::exp(-2060.0/Temp);
          double J_12_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
          double J_12_13 = -4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
          double J_12_15 = 1.0*jvals[0];
          double J_14_8 = 1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_14_9 = 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
          double J_14_10 = 1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
          double J_14_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp);
          double J_14_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
          double J_14_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_14_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
          double J_14_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          double J_15_9 = 1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
          double J_15_10 = 1.0*jvals[2];
          double J_15_11 = 3.6e-11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
          double J_15_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          double J_15_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*S_11*Kokkos::exp(110.0/Temp);
          double J_15_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0];
          double J_15_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          double J_16_13 = -3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
          double J_16_14 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
          double J_16_15 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
          double J_16_16 = -2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double W_0_0 = inv_g_dt;
          // Block 1: species [M]
          double W_1_1 = inv_g_dt;
          // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double W_2_2 = inv_g_dt;
          double W_3_3 = inv_g_dt;
          double W_4_4 = inv_g_dt - J_4_4;
          double W_5_5 = inv_g_dt - J_5_5;
          double W_5_6 = -J_5_6;
          double W_5_9 = -J_5_9;
          double W_5_12 = -J_5_12;
          double W_5_13 = -J_5_13;
          // Block 2: species [HO2]
          double W_6_6 = inv_g_dt - J_6_6;
          double W_6_13 = -J_6_13;
          // Block 3: species [H2O2, HNO3]
          double W_7_6 = -J_7_6;
          double W_7_7 = inv_g_dt - J_7_7;
          double W_7_13 = -J_7_13;
          // Block 4: species [OH]
          double W_8_4 = -J_8_4;
          double W_8_7 = -J_8_7;
          double W_8_8 = inv_g_dt - J_8_8;
          double W_8_10 = -J_8_10;
          double W_8_13 = -J_8_13;
          // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double W_9_5 = -J_9_5;
          double W_9_9 = inv_g_dt - J_9_9;
          double W_9_11 = -J_9_11;
          double W_9_12 = -J_9_12;
          double W_9_13 = -J_9_13;
          double W_9_15 = -J_9_15;
          // Block 3: species [H2O2, HNO3]
          double W_10_8 = -J_10_8;
          double W_10_10 = inv_g_dt - J_10_10;
          double W_10_13 = -J_10_13;
          // Block 5: species [NO]
          double W_11_4 = -J_11_4;
          double W_11_9 = -J_11_9;
          double W_11_11 = inv_g_dt - J_11_11;
          double W_11_14 = -J_11_14;
          double W_11_15 = -J_11_15;
          // Block 6: species [O3P]
          double W_12_5 = -J_12_5;
          double W_12_9 = -J_12_9;
          double W_12_12 = inv_g_dt - J_12_12;
          double W_12_13 = -J_12_13;
          double W_12_15 = -J_12_15;
          // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double W_13_13 = inv_g_dt;
          // Block 7: species [NO3]
          double W_14_8 = -J_14_8;
          double W_14_9 = -J_14_9;
          double W_14_10 = -J_14_10;
          double W_14_11 = -J_14_11;
          double W_14_13 = -J_14_13;
          double W_14_14 = inv_g_dt - J_14_14;
          double W_14_15 = -J_14_15;
          double W_14_16 = -J_14_16;
          // Block 8: species [NO2]
          double W_15_9 = -J_15_9;
          double W_15_10 = -J_15_10;
          double W_15_11 = -J_15_11;
          double W_15_13 = -J_15_13;
          double W_15_14 = -J_15_14;
          double W_15_15 = inv_g_dt - J_15_15;
          double W_15_16 = -J_15_16;
          // Block 9: species [N2O5]
          double W_16_13 = -J_16_13;
          double W_16_14 = -J_16_14;
          double W_16_15 = -J_16_15;
          double W_16_16 = inv_g_dt - J_16_16;

          // Symbolic Doolittle Sparse LU Factorization
          // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double U_0_0 = W_0_0;
          // Block 1: species [M]
          double U_1_1 = W_1_1;
          double U_2_2 = W_2_2;
          double U_3_3 = W_3_3;
          double U_4_4 = W_4_4;
          // Block 4: species [OH]
          double L_8_4 = (W_8_4) / U_4_4;
          // Block 5: species [NO]
          double L_11_4 = (W_11_4) / U_4_4;
          double U_5_5 = W_5_5;
          double U_5_6 = W_5_6;
          double U_5_9 = W_5_9;
          double U_5_12 = W_5_12;
          double U_5_13 = W_5_13;
          double L_9_5 = (W_9_5) / U_5_5;
          // Block 6: species [O3P]
          double L_12_5 = (W_12_5) / U_5_5;
          // Block 2: species [HO2]
          double U_6_6 = W_6_6;
          double U_6_13 = W_6_13;
          // Block 3: species [H2O2, HNO3]
          double L_7_6 = (W_7_6) / U_6_6;
          double L_9_6 = (0.0 - L_9_5 * U_5_6) / U_6_6;
          double L_12_6 = (0.0 - L_12_5 * U_5_6) / U_6_6;
          double U_7_7 = W_7_7;
          double U_7_13 = W_7_13 - L_7_6 * U_6_13;
          double L_8_7 = (W_8_7) / U_7_7;
          double U_8_8 = W_8_8;
          double U_8_10 = W_8_10;
          double U_8_13 = W_8_13 - L_8_7 * U_7_13;
          double L_10_8 = (W_10_8) / U_8_8;
          // Block 7: species [NO3]
          double L_14_8 = (W_14_8) / U_8_8;
          double U_9_9 = W_9_9 - L_9_5 * U_5_9;
          double U_9_11 = W_9_11;
          double U_9_12 = W_9_12 - L_9_5 * U_5_12;
          double U_9_13 = W_9_13 - L_9_5 * U_5_13 - L_9_6 * U_6_13;
          double U_9_15 = W_9_15;
          double L_11_9 = (W_11_9) / U_9_9;
          double L_12_9 = (W_12_9 - L_12_5 * U_5_9) / U_9_9;
          double L_14_9 = (W_14_9) / U_9_9;
          // Block 8: species [NO2]
          double L_15_9 = (W_15_9) / U_9_9;
          double U_10_10 = W_10_10 - L_10_8 * U_8_10;
          double U_10_13 = W_10_13 - L_10_8 * U_8_13;
          double L_14_10 = (W_14_10 - L_14_8 * U_8_10) / U_10_10;
          double L_15_10 = (W_15_10) / U_10_10;
          double U_11_11 = W_11_11 - L_11_9 * U_9_11;
          double U_11_12 = 0.0 - L_11_9 * U_9_12;
          double U_11_13 = 0.0 - L_11_9 * U_9_13;
          double U_11_14 = W_11_14;
          double U_11_15 = W_11_15 - L_11_9 * U_9_15;
          double L_12_11 = (0.0 - L_12_9 * U_9_11) / U_11_11;
          double L_14_11 = (W_14_11 - L_14_9 * U_9_11) / U_11_11;
          double L_15_11 = (W_15_11 - L_15_9 * U_9_11) / U_11_11;
          double U_12_12 = W_12_12 - L_12_5 * U_5_12 - L_12_9 * U_9_12 - L_12_11 * U_11_12;
          double U_12_13 = W_12_13 - L_12_5 * U_5_13 - L_12_6 * U_6_13 - L_12_9 * U_9_13 - L_12_11 * U_11_13;
          double U_12_14 = 0.0 - L_12_11 * U_11_14;
          double U_12_15 = W_12_15 - L_12_9 * U_9_15 - L_12_11 * U_11_15;
          double L_14_12 = (0.0 - L_14_9 * U_9_12 - L_14_11 * U_11_12) / U_12_12;
          double L_15_12 = (0.0 - L_15_9 * U_9_12 - L_15_11 * U_11_12) / U_12_12;
          double U_13_13 = W_13_13;
          double L_14_13 = (W_14_13 - L_14_8 * U_8_13 - L_14_9 * U_9_13 - L_14_10 * U_10_13 - L_14_11 * U_11_13 - L_14_12 * U_12_13) / U_13_13;
          double L_15_13 = (W_15_13 - L_15_9 * U_9_13 - L_15_10 * U_10_13 - L_15_11 * U_11_13 - L_15_12 * U_12_13) / U_13_13;
          // Block 9: species [N2O5]
          double L_16_13 = (W_16_13) / U_13_13;
          double U_14_14 = W_14_14 - L_14_11 * U_11_14 - L_14_12 * U_12_14;
          double U_14_15 = W_14_15 - L_14_9 * U_9_15 - L_14_11 * U_11_15 - L_14_12 * U_12_15;
          double U_14_16 = W_14_16;
          double L_15_14 = (W_15_14 - L_15_11 * U_11_14 - L_15_12 * U_12_14) / U_14_14;
          double L_16_14 = (W_16_14) / U_14_14;
          double U_15_15 = W_15_15 - L_15_9 * U_9_15 - L_15_11 * U_11_15 - L_15_12 * U_12_15 - L_15_14 * U_14_15;
          double U_15_16 = W_15_16 - L_15_14 * U_14_16;
          double L_16_15 = (W_16_15 - L_16_14 * U_14_15) / U_15_15;
          double U_16_16 = W_16_16 - L_16_14 * U_14_16 - L_16_15 * U_15_16;

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F1_0 = 0.0;
          double F1_1 = 0.0;
          double F1_2 = 0.0;
          double F1_3 = 0.0;
          double F1_4 = -1.0*S_4*jvals[1];
          double F1_5 = -4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) + 1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) + 1.6e-11*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_6 = -1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F1_7 = -1.0*S_7*jvals[3] + 1.0*S_6*(1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F1_8 = 2.0*S_7*jvals[3] - 1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*S_10*jvals[2] + 1.0*S_4*jvals[1];
          double F1_9 = 4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) - 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp) - 8e-12*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_10 = -1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*S_10*jvals[2];
          double F1_11 = 1.0*S_4*jvals[1] - 1.8e-11*S_11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) + 1.0*S_15*jvals[0];
          double F1_12 = -4.9010355073085e-27*S_13*S_5*S_12*Kokkos::pow(Temp, -2.8) + 1.0*S_15*jvals[0] - 8e-12*S_9*S_12*Kokkos::exp(-2060.0/Temp);
          double F1_13 = 0.0;
          double F1_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*S_10*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.8e-11*S_11*S_14*Kokkos::exp(110.0/Temp) + 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp);
          double F1_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*S_10*jvals[2] + 3.6e-11*S_11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_11*S_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*S_15*jvals[0];
          double F1_16 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // Block 0: K1 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y1_0 = F1_0;
          // Block 1: K1 forward sub [M]
          double y1_1 = F1_1;
          double y1_2 = F1_2;
          double y1_3 = F1_3;
          double y1_4 = F1_4;
          double y1_5 = F1_5;
          // Block 2: K1 forward sub [HO2]
          double y1_6 = F1_6;
          // Block 3: K1 forward sub [H2O2, HNO3]
          double y1_7 = F1_7 - L_7_6 * y1_6;
          // Block 4: K1 forward sub [OH]
          double y1_8 = F1_8 - L_8_4 * y1_4 - L_8_7 * y1_7;
          double y1_9 = F1_9 - L_9_5 * y1_5 - L_9_6 * y1_6;
          double y1_10 = F1_10 - L_10_8 * y1_8;
          // Block 5: K1 forward sub [NO]
          double y1_11 = F1_11 - L_11_4 * y1_4 - L_11_9 * y1_9;
          // Block 6: K1 forward sub [O3P]
          double y1_12 = F1_12 - L_12_5 * y1_5 - L_12_6 * y1_6 - L_12_9 * y1_9 - L_12_11 * y1_11;
          double y1_13 = F1_13;
          // Block 7: K1 forward sub [NO3]
          double y1_14 = F1_14 - L_14_8 * y1_8 - L_14_9 * y1_9 - L_14_10 * y1_10 - L_14_11 * y1_11 - L_14_12 * y1_12 - L_14_13 * y1_13;
          // Block 8: K1 forward sub [NO2]
          double y1_15 = F1_15 - L_15_9 * y1_9 - L_15_10 * y1_10 - L_15_11 * y1_11 - L_15_12 * y1_12 - L_15_13 * y1_13 - L_15_14 * y1_14;
          // Block 9: K1 forward sub [N2O5]
          double y1_16 = F1_16 - L_16_13 * y1_13 - L_16_14 * y1_14 - L_16_15 * y1_15;

          // Block 9: K1 backward sub [N2O5]
          double K1_16 = y1_16 / U_16_16;
          // Block 8: K1 backward sub [NO2]
          double K1_15 = (y1_15 - U_15_16 * K1_16) / U_15_15;
          // Block 7: K1 backward sub [NO3]
          double K1_14 = (y1_14 - U_14_15 * K1_15 - U_14_16 * K1_16) / U_14_14;
          // Block 0: K1 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K1_13 = y1_13 / U_13_13;
          // Block 6: K1 backward sub [O3P]
          double K1_12 = (y1_12 - U_12_13 * K1_13 - U_12_14 * K1_14 - U_12_15 * K1_15) / U_12_12;
          // Block 5: K1 backward sub [NO]
          double K1_11 = (y1_11 - U_11_12 * K1_12 - U_11_13 * K1_13 - U_11_14 * K1_14 - U_11_15 * K1_15) / U_11_11;
          // Block 3: K1 backward sub [H2O2, HNO3]
          double K1_10 = (y1_10 - U_10_13 * K1_13) / U_10_10;
          double K1_9 = (y1_9 - U_9_11 * K1_11 - U_9_12 * K1_12 - U_9_13 * K1_13 - U_9_15 * K1_15) / U_9_9;
          // Block 4: K1 backward sub [OH]
          double K1_8 = (y1_8 - U_8_10 * K1_10 - U_8_13 * K1_13) / U_8_8;
          double K1_7 = (y1_7 - U_7_13 * K1_13) / U_7_7;
          // Block 2: K1 backward sub [HO2]
          double K1_6 = (y1_6 - U_6_13 * K1_13) / U_6_6;
          double K1_5 = (y1_5 - U_5_6 * K1_6 - U_5_9 * K1_9 - U_5_12 * K1_12 - U_5_13 * K1_13) / U_5_5;
          double K1_4 = y1_4 / U_4_4;
          double K1_3 = y1_3 / U_3_3;
          double K1_2 = y1_2 / U_2_2;
          // Block 1: K1 backward sub [M]
          double K1_1 = y1_1 / U_1_1;
          double K1_0 = y1_0 / U_0_0;


          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          double Y2_4 = S_4 + K1_4;
          double Y2_5 = S_5 + K1_5;
          double Y2_6 = S_6 + K1_6;
          double Y2_7 = S_7 + K1_7;
          double Y2_8 = S_8 + K1_8;
          double Y2_9 = S_9 + K1_9;
          double Y2_10 = S_10 + K1_10;
          double Y2_11 = S_11 + K1_11;
          double Y2_12 = S_12 + K1_12;
          double Y2_13 = S_13 + K1_13;
          double Y2_14 = S_14 + K1_14;
          double Y2_15 = S_15 + K1_15;
          double Y2_16 = S_16 + K1_16;
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = 0.0;
          double F2_2 = 0.0;
          double F2_3 = 0.0;
          double F2_4 = -1.0*Y2_4*jvals[1];
          double F2_5 = -4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) + 1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp)) + 1.6e-11*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_6 = -1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F2_7 = -1.0*Y2_7*jvals[3] + 1.0*Y2_6*(1.7e-33*Y2_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp));
          double F2_8 = 2.0*Y2_7*jvals[3] - 1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*Y2_10*jvals[2] + 1.0*Y2_4*jvals[1];
          double F2_9 = 4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) - 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp) - 8e-12*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_10 = -1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*Y2_10*jvals[2];
          double F2_11 = 1.0*Y2_4*jvals[1] - 1.8e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) - 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) + 1.0*Y2_15*jvals[0];
          double F2_12 = -4.9010355073085e-27*Y2_13*Y2_5*Y2_12*Kokkos::pow(Temp, -2.8) + 1.0*Y2_15*jvals[0] - 8e-12*Y2_9*Y2_12*Kokkos::exp(-2060.0/Temp);
          double F2_13 = 0.0;
          double F2_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*Y2_10*Y2_8*(1.9e-33*Y2_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*Y2_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.8e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) + 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp);
          double F2_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 1.0*Y2_10*jvals[2] + 3.6e-11*Y2_11*Y2_14*Kokkos::exp(110.0/Temp) + 1.8e-12*Y2_11*Y2_9*Kokkos::exp(-1370.0/Temp) - 1.4e-13*Y2_15*Y2_9*Kokkos::exp(-2470.0/Temp) - 1.0*Y2_15*jvals[0];
          double F2_16 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_15*Y2_14*Kokkos::exp(3.5/Temp)/(1.4e-18*Y2_13*Kokkos::exp(3.7/Temp) + 1.0) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*Y2_13*Y2_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*Y2_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          double rhs2_7 = F2_7 + (-1.0156171083877703 / dt) * K1_7;
          double rhs2_8 = F2_8 + (-1.0156171083877703 / dt) * K1_8;
          double rhs2_9 = F2_9 + (-1.0156171083877703 / dt) * K1_9;
          double rhs2_10 = F2_10 + (-1.0156171083877703 / dt) * K1_10;
          double rhs2_11 = F2_11 + (-1.0156171083877703 / dt) * K1_11;
          double rhs2_12 = F2_12 + (-1.0156171083877703 / dt) * K1_12;
          double rhs2_13 = F2_13 + (-1.0156171083877703 / dt) * K1_13;
          double rhs2_14 = F2_14 + (-1.0156171083877703 / dt) * K1_14;
          double rhs2_15 = F2_15 + (-1.0156171083877703 / dt) * K1_15;
          double rhs2_16 = F2_16 + (-1.0156171083877703 / dt) * K1_16;
          // Block 0: K2 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y2_0 = rhs2_0;
          // Block 1: K2 forward sub [M]
          double y2_1 = rhs2_1;
          double y2_2 = rhs2_2;
          double y2_3 = rhs2_3;
          double y2_4 = rhs2_4;
          double y2_5 = rhs2_5;
          // Block 2: K2 forward sub [HO2]
          double y2_6 = rhs2_6;
          // Block 3: K2 forward sub [H2O2, HNO3]
          double y2_7 = rhs2_7 - L_7_6 * y2_6;
          // Block 4: K2 forward sub [OH]
          double y2_8 = rhs2_8 - L_8_4 * y2_4 - L_8_7 * y2_7;
          double y2_9 = rhs2_9 - L_9_5 * y2_5 - L_9_6 * y2_6;
          double y2_10 = rhs2_10 - L_10_8 * y2_8;
          // Block 5: K2 forward sub [NO]
          double y2_11 = rhs2_11 - L_11_4 * y2_4 - L_11_9 * y2_9;
          // Block 6: K2 forward sub [O3P]
          double y2_12 = rhs2_12 - L_12_5 * y2_5 - L_12_6 * y2_6 - L_12_9 * y2_9 - L_12_11 * y2_11;
          double y2_13 = rhs2_13;
          // Block 7: K2 forward sub [NO3]
          double y2_14 = rhs2_14 - L_14_8 * y2_8 - L_14_9 * y2_9 - L_14_10 * y2_10 - L_14_11 * y2_11 - L_14_12 * y2_12 - L_14_13 * y2_13;
          // Block 8: K2 forward sub [NO2]
          double y2_15 = rhs2_15 - L_15_9 * y2_9 - L_15_10 * y2_10 - L_15_11 * y2_11 - L_15_12 * y2_12 - L_15_13 * y2_13 - L_15_14 * y2_14;
          // Block 9: K2 forward sub [N2O5]
          double y2_16 = rhs2_16 - L_16_13 * y2_13 - L_16_14 * y2_14 - L_16_15 * y2_15;

          // Block 9: K2 backward sub [N2O5]
          double K2_16 = y2_16 / U_16_16;
          // Block 8: K2 backward sub [NO2]
          double K2_15 = (y2_15 - U_15_16 * K2_16) / U_15_15;
          // Block 7: K2 backward sub [NO3]
          double K2_14 = (y2_14 - U_14_15 * K2_15 - U_14_16 * K2_16) / U_14_14;
          // Block 0: K2 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K2_13 = y2_13 / U_13_13;
          // Block 6: K2 backward sub [O3P]
          double K2_12 = (y2_12 - U_12_13 * K2_13 - U_12_14 * K2_14 - U_12_15 * K2_15) / U_12_12;
          // Block 5: K2 backward sub [NO]
          double K2_11 = (y2_11 - U_11_12 * K2_12 - U_11_13 * K2_13 - U_11_14 * K2_14 - U_11_15 * K2_15) / U_11_11;
          // Block 3: K2 backward sub [H2O2, HNO3]
          double K2_10 = (y2_10 - U_10_13 * K2_13) / U_10_10;
          double K2_9 = (y2_9 - U_9_11 * K2_11 - U_9_12 * K2_12 - U_9_13 * K2_13 - U_9_15 * K2_15) / U_9_9;
          // Block 4: K2 backward sub [OH]
          double K2_8 = (y2_8 - U_8_10 * K2_10 - U_8_13 * K2_13) / U_8_8;
          double K2_7 = (y2_7 - U_7_13 * K2_13) / U_7_7;
          // Block 2: K2 backward sub [HO2]
          double K2_6 = (y2_6 - U_6_13 * K2_13) / U_6_6;
          double K2_5 = (y2_5 - U_5_6 * K2_6 - U_5_9 * K2_9 - U_5_12 * K2_12 - U_5_13 * K2_13) / U_5_5;
          double K2_4 = y2_4 / U_4_4;
          double K2_3 = y2_3 / U_3_3;
          double K2_2 = y2_2 / U_2_2;
          // Block 1: K2 backward sub [M]
          double K2_1 = y2_1 / U_1_1;
          double K2_0 = y2_0 / U_0_0;


          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          double Y3_4 = S_4 + K1_4;
          double Y3_5 = S_5 + K1_5;
          double Y3_6 = S_6 + K1_6;
          double Y3_7 = S_7 + K1_7;
          double Y3_8 = S_8 + K1_8;
          double Y3_9 = S_9 + K1_9;
          double Y3_10 = S_10 + K1_10;
          double Y3_11 = S_11 + K1_11;
          double Y3_12 = S_12 + K1_12;
          double Y3_13 = S_13 + K1_13;
          double Y3_14 = S_14 + K1_14;
          double Y3_15 = S_15 + K1_15;
          double Y3_16 = S_16 + K1_16;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          double rhs3_7 = F2_7 + (4.0759956452537702 / dt) * K1_7 + (9.20767942983308 / dt) * K2_7;
          double rhs3_8 = F2_8 + (4.0759956452537702 / dt) * K1_8 + (9.20767942983308 / dt) * K2_8;
          double rhs3_9 = F2_9 + (4.0759956452537702 / dt) * K1_9 + (9.20767942983308 / dt) * K2_9;
          double rhs3_10 = F2_10 + (4.0759956452537702 / dt) * K1_10 + (9.20767942983308 / dt) * K2_10;
          double rhs3_11 = F2_11 + (4.0759956452537702 / dt) * K1_11 + (9.20767942983308 / dt) * K2_11;
          double rhs3_12 = F2_12 + (4.0759956452537702 / dt) * K1_12 + (9.20767942983308 / dt) * K2_12;
          double rhs3_13 = F2_13 + (4.0759956452537702 / dt) * K1_13 + (9.20767942983308 / dt) * K2_13;
          double rhs3_14 = F2_14 + (4.0759956452537702 / dt) * K1_14 + (9.20767942983308 / dt) * K2_14;
          double rhs3_15 = F2_15 + (4.0759956452537702 / dt) * K1_15 + (9.20767942983308 / dt) * K2_15;
          double rhs3_16 = F2_16 + (4.0759956452537702 / dt) * K1_16 + (9.20767942983308 / dt) * K2_16;
          // Block 0: K3 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double y3_0 = rhs3_0;
          // Block 1: K3 forward sub [M]
          double y3_1 = rhs3_1;
          double y3_2 = rhs3_2;
          double y3_3 = rhs3_3;
          double y3_4 = rhs3_4;
          double y3_5 = rhs3_5;
          // Block 2: K3 forward sub [HO2]
          double y3_6 = rhs3_6;
          // Block 3: K3 forward sub [H2O2, HNO3]
          double y3_7 = rhs3_7 - L_7_6 * y3_6;
          // Block 4: K3 forward sub [OH]
          double y3_8 = rhs3_8 - L_8_4 * y3_4 - L_8_7 * y3_7;
          double y3_9 = rhs3_9 - L_9_5 * y3_5 - L_9_6 * y3_6;
          double y3_10 = rhs3_10 - L_10_8 * y3_8;
          // Block 5: K3 forward sub [NO]
          double y3_11 = rhs3_11 - L_11_4 * y3_4 - L_11_9 * y3_9;
          // Block 6: K3 forward sub [O3P]
          double y3_12 = rhs3_12 - L_12_5 * y3_5 - L_12_6 * y3_6 - L_12_9 * y3_9 - L_12_11 * y3_11;
          double y3_13 = rhs3_13;
          // Block 7: K3 forward sub [NO3]
          double y3_14 = rhs3_14 - L_14_8 * y3_8 - L_14_9 * y3_9 - L_14_10 * y3_10 - L_14_11 * y3_11 - L_14_12 * y3_12 - L_14_13 * y3_13;
          // Block 8: K3 forward sub [NO2]
          double y3_15 = rhs3_15 - L_15_9 * y3_9 - L_15_10 * y3_10 - L_15_11 * y3_11 - L_15_12 * y3_12 - L_15_13 * y3_13 - L_15_14 * y3_14;
          // Block 9: K3 forward sub [N2O5]
          double y3_16 = rhs3_16 - L_16_13 * y3_13 - L_16_14 * y3_14 - L_16_15 * y3_15;

          // Block 9: K3 backward sub [N2O5]
          double K3_16 = y3_16 / U_16_16;
          // Block 8: K3 backward sub [NO2]
          double K3_15 = (y3_15 - U_15_16 * K3_16) / U_15_15;
          // Block 7: K3 backward sub [NO3]
          double K3_14 = (y3_14 - U_14_15 * K3_15 - U_14_16 * K3_16) / U_14_14;
          // Block 0: K3 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
          double K3_13 = y3_13 / U_13_13;
          // Block 6: K3 backward sub [O3P]
          double K3_12 = (y3_12 - U_12_13 * K3_13 - U_12_14 * K3_14 - U_12_15 * K3_15) / U_12_12;
          // Block 5: K3 backward sub [NO]
          double K3_11 = (y3_11 - U_11_12 * K3_12 - U_11_13 * K3_13 - U_11_14 * K3_14 - U_11_15 * K3_15) / U_11_11;
          // Block 3: K3 backward sub [H2O2, HNO3]
          double K3_10 = (y3_10 - U_10_13 * K3_13) / U_10_10;
          double K3_9 = (y3_9 - U_9_11 * K3_11 - U_9_12 * K3_12 - U_9_13 * K3_13 - U_9_15 * K3_15) / U_9_9;
          // Block 4: K3 backward sub [OH]
          double K3_8 = (y3_8 - U_8_10 * K3_10 - U_8_13 * K3_13) / U_8_8;
          double K3_7 = (y3_7 - U_7_13 * K3_13) / U_7_7;
          // Block 2: K3 backward sub [HO2]
          double K3_6 = (y3_6 - U_6_13 * K3_13) / U_6_6;
          double K3_5 = (y3_5 - U_5_6 * K3_6 - U_5_9 * K3_9 - U_5_12 * K3_12 - U_5_13 * K3_13) / U_5_5;
          double K3_4 = y3_4 / U_4_4;
          double K3_3 = y3_3 / U_3_3;
          double K3_2 = y3_2 / U_2_2;
          // Block 1: K3 backward sub [M]
          double K3_1 = y3_1 / U_1_1;
          double K3_0 = y3_0 / U_0_0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(16)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[16] + rtol[16] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(15)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[15] + rtol[15] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(14)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[14] + rtol[14] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(13)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[13] + rtol[13] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(11)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[11] + rtol[11] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_11 + K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_11 + -2.9079558716805471 * K2_11 + 0.22354069897811571 * K3_11;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_12 + K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_12 + -2.9079558716805471 * K2_12 + 0.22354069897811571 * K3_12;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_13 + K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(12)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[12] + rtol[12] * ymax;
              double yerr_i = 0.5 * K1_13 + -2.9079558716805471 * K2_13 + 0.22354069897811571 * K3_13;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_14 + K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_14 + -2.9079558716805471 * K2_14 + 0.22354069897811571 * K3_14;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_15 + K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_15 + -2.9079558716805471 * K2_15 + 0.22354069897811571 * K3_15;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_16 + K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_16 + -2.9079558716805471 * K2_16 + 0.22354069897811571 * K3_16;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 17);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(16) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(15) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(14) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(8) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(6) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(13) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(11) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              state(1) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              state(10) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              state(0) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              state(7) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              state(2) += K1_11 + 6.1697947043828245 * K2_11 + -0.42772256543218573 * K3_11;
              state(9) += K1_12 + 6.1697947043828245 * K2_12 + -0.42772256543218573 * K3_12;
              state(12) += K1_13 + 6.1697947043828245 * K2_13 + -0.42772256543218573 * K3_13;
              state(4) += K1_14 + 6.1697947043828245 * K2_14 + -0.42772256543218573 * K3_14;
              state(3) += K1_15 + 6.1697947043828245 * K2_15 + -0.42772256543218573 * K3_15;
              state(5) += K1_16 + 6.1697947043828245 * K2_16 + -0.42772256543218573 * K3_16;
              // Save checkpoint data for adjoint/TLM
              if (chk.num_steps >= CheckpointBuffer::MAX_STEPS) {
                  ierr = 1;  // Exceeded MAX_STEPS budget
                  return -1;
              }
              chk.h[chk.num_steps] = dt;
              chk.state[chk.num_steps][0] = state(16);
              chk.state[chk.num_steps][1] = state(15);
              chk.state[chk.num_steps][2] = state(14);
              chk.state[chk.num_steps][3] = state(8);
              chk.state[chk.num_steps][4] = state(6);
              chk.state[chk.num_steps][5] = state(13);
              chk.state[chk.num_steps][6] = state(11);
              chk.state[chk.num_steps][7] = state(1);
              chk.state[chk.num_steps][8] = state(10);
              chk.state[chk.num_steps][9] = state(0);
              chk.state[chk.num_steps][10] = state(7);
              chk.state[chk.num_steps][11] = state(2);
              chk.state[chk.num_steps][12] = state(9);
              chk.state[chk.num_steps][13] = state(12);
              chk.state[chk.num_steps][14] = state(4);
              chk.state[chk.num_steps][15] = state(3);
              chk.state[chk.num_steps][16] = state(5);
              chk.num_steps++;
              t += dt;
              dt *= factor;
          } else {
              dt *= factor;
          }
          } // end while (t < dt_total)
          return chk.num_steps;
      }


      // Discrete adjoint backward integration through checkpointed trajectory
      // Walks backward through saved steps, accumulating adjoint variable lambda.
      template <class StateView, class AdjView>
      KOKKOS_INLINE_FUNCTION void integrate_adj(
          double dt_total, const StateView& state_final,
          AdjView& lambda, Kokkos::View<const double*, memory_space> jvals_view,
          const CheckpointBuffer& chk, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const
      {
          const double* jvals = jvals_view.data();
          // Environment conditions must match the forward checkpoint so the
          // recomputed Jacobian reflects the same rate law the state was
          // integrated under.
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          const double g = 0.435866521508459;

          // --- Discrete Adjoint: backward integration over checkpointed steps ---
          for (int step = chk.num_steps - 1; step >= 0; step--) {
              const double h = chk.h[step];
              const double inv_g_h = 1.0 / (g * h);

              // Reload checkpointed state
              const double S_0 = chk.state[step][0];
              const double S_1 = chk.state[step][1];
              const double S_2 = chk.state[step][2];
              const double S_3 = chk.state[step][3];
              const double S_4 = chk.state[step][4];
              const double S_5 = chk.state[step][5];
              const double S_6 = chk.state[step][6];
              const double S_7 = chk.state[step][7];
              const double S_8 = chk.state[step][8];
              const double S_9 = chk.state[step][9];
              const double S_10 = chk.state[step][10];
              const double S_11 = chk.state[step][11];
              const double S_12 = chk.state[step][12];
              const double S_13 = chk.state[step][13];
              const double S_14 = chk.state[step][14];
              const double S_15 = chk.state[step][15];
              const double S_16 = chk.state[step][16];

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_4_4 = -1.0*jvals[1];
              double J_5_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_5_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_5_9 = 1.6e-11*S_12*Kokkos::exp(-2060.0/Temp);
              double J_5_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) + 1.6e-11*S_9*Kokkos::exp(-2060.0/Temp);
              double J_5_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_6_6 = -1.7e-33*S_13*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_6_13 = -1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
              double J_7_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_7_7 = -1.0*jvals[3];
              double J_7_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
              double J_8_4 = 1.0*jvals[1];
              double J_8_7 = 2.0*jvals[3];
              double J_8_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_8_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2];
              double J_8_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_9_5 = 4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_9_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp) - 8e-12*S_12*Kokkos::exp(-2060.0/Temp);
              double J_9_11 = -1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_9_12 = 4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
              double J_9_13 = 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_9_15 = -1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
              double J_10_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_10_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2];
              double J_10_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_11_4 = 1.0*jvals[1];
              double J_11_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp);
              double J_11_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_11_14 = -1.8e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_11_15 = 1.0*jvals[0];
              double J_12_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_12_9 = -8e-12*S_12*Kokkos::exp(-2060.0/Temp);
              double J_12_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
              double J_12_13 = -4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_12_15 = 1.0*jvals[0];
              double J_14_8 = 1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_14_9 = 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
              double J_14_10 = 1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_14_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp);
              double J_14_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_14_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_14_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
              double J_14_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
              double J_15_9 = 1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
              double J_15_10 = 1.0*jvals[2];
              double J_15_11 = 3.6e-11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_15_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
              double J_15_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_15_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0];
              double J_15_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
              double J_16_13 = -3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
              double J_16_14 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
              double J_16_15 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
              double J_16_16 = -2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);

              // Form iteration matrix W = (1/(gamma*h))*I - J
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double W_0_0 = inv_g_h;
              // Block 1: species [M]
              double W_1_1 = inv_g_h;
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double W_2_2 = inv_g_h;
              double W_3_3 = inv_g_h;
              double W_4_4 = inv_g_h - J_4_4;
              double W_5_5 = inv_g_h - J_5_5;
              double W_5_6 = -J_5_6;
              double W_5_9 = -J_5_9;
              double W_5_12 = -J_5_12;
              double W_5_13 = -J_5_13;
              // Block 2: species [HO2]
              double W_6_6 = inv_g_h - J_6_6;
              double W_6_13 = -J_6_13;
              // Block 3: species [H2O2, HNO3]
              double W_7_6 = -J_7_6;
              double W_7_7 = inv_g_h - J_7_7;
              double W_7_13 = -J_7_13;
              // Block 4: species [OH]
              double W_8_4 = -J_8_4;
              double W_8_7 = -J_8_7;
              double W_8_8 = inv_g_h - J_8_8;
              double W_8_10 = -J_8_10;
              double W_8_13 = -J_8_13;
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double W_9_5 = -J_9_5;
              double W_9_9 = inv_g_h - J_9_9;
              double W_9_11 = -J_9_11;
              double W_9_12 = -J_9_12;
              double W_9_13 = -J_9_13;
              double W_9_15 = -J_9_15;
              // Block 3: species [H2O2, HNO3]
              double W_10_8 = -J_10_8;
              double W_10_10 = inv_g_h - J_10_10;
              double W_10_13 = -J_10_13;
              // Block 5: species [NO]
              double W_11_4 = -J_11_4;
              double W_11_9 = -J_11_9;
              double W_11_11 = inv_g_h - J_11_11;
              double W_11_14 = -J_11_14;
              double W_11_15 = -J_11_15;
              // Block 6: species [O3P]
              double W_12_5 = -J_12_5;
              double W_12_9 = -J_12_9;
              double W_12_12 = inv_g_h - J_12_12;
              double W_12_13 = -J_12_13;
              double W_12_15 = -J_12_15;
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double W_13_13 = inv_g_h;
              // Block 7: species [NO3]
              double W_14_8 = -J_14_8;
              double W_14_9 = -J_14_9;
              double W_14_10 = -J_14_10;
              double W_14_11 = -J_14_11;
              double W_14_13 = -J_14_13;
              double W_14_14 = inv_g_h - J_14_14;
              double W_14_15 = -J_14_15;
              double W_14_16 = -J_14_16;
              // Block 8: species [NO2]
              double W_15_9 = -J_15_9;
              double W_15_10 = -J_15_10;
              double W_15_11 = -J_15_11;
              double W_15_13 = -J_15_13;
              double W_15_14 = -J_15_14;
              double W_15_15 = inv_g_h - J_15_15;
              double W_15_16 = -J_15_16;
              // Block 9: species [N2O5]
              double W_16_13 = -J_16_13;
              double W_16_14 = -J_16_14;
              double W_16_15 = -J_16_15;
              double W_16_16 = inv_g_h - J_16_16;


              // Symbolic LU Factorization (recomputed at each adjoint step)
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double U_0_0 = W_0_0;
              // Block 1: species [M]
              double U_1_1 = W_1_1;
              double U_2_2 = W_2_2;
              double U_3_3 = W_3_3;
              double U_4_4 = W_4_4;
              // Block 4: species [OH]
              double L_8_4 = (W_8_4) / U_4_4;
              // Block 5: species [NO]
              double L_11_4 = (W_11_4) / U_4_4;
              double U_5_5 = W_5_5;
              double U_5_6 = W_5_6;
              double U_5_9 = W_5_9;
              double U_5_12 = W_5_12;
              double U_5_13 = W_5_13;
              double L_9_5 = (W_9_5) / U_5_5;
              // Block 6: species [O3P]
              double L_12_5 = (W_12_5) / U_5_5;
              // Block 2: species [HO2]
              double U_6_6 = W_6_6;
              double U_6_13 = W_6_13;
              // Block 3: species [H2O2, HNO3]
              double L_7_6 = (W_7_6) / U_6_6;
              double L_9_6 = (0.0 - L_9_5 * U_5_6) / U_6_6;
              double L_12_6 = (0.0 - L_12_5 * U_5_6) / U_6_6;
              double U_7_7 = W_7_7;
              double U_7_13 = W_7_13 - L_7_6 * U_6_13;
              double L_8_7 = (W_8_7) / U_7_7;
              double U_8_8 = W_8_8;
              double U_8_10 = W_8_10;
              double U_8_13 = W_8_13 - L_8_7 * U_7_13;
              double L_10_8 = (W_10_8) / U_8_8;
              // Block 7: species [NO3]
              double L_14_8 = (W_14_8) / U_8_8;
              double U_9_9 = W_9_9 - L_9_5 * U_5_9;
              double U_9_11 = W_9_11;
              double U_9_12 = W_9_12 - L_9_5 * U_5_12;
              double U_9_13 = W_9_13 - L_9_5 * U_5_13 - L_9_6 * U_6_13;
              double U_9_15 = W_9_15;
              double L_11_9 = (W_11_9) / U_9_9;
              double L_12_9 = (W_12_9 - L_12_5 * U_5_9) / U_9_9;
              double L_14_9 = (W_14_9) / U_9_9;
              // Block 8: species [NO2]
              double L_15_9 = (W_15_9) / U_9_9;
              double U_10_10 = W_10_10 - L_10_8 * U_8_10;
              double U_10_13 = W_10_13 - L_10_8 * U_8_13;
              double L_14_10 = (W_14_10 - L_14_8 * U_8_10) / U_10_10;
              double L_15_10 = (W_15_10) / U_10_10;
              double U_11_11 = W_11_11 - L_11_9 * U_9_11;
              double U_11_12 = 0.0 - L_11_9 * U_9_12;
              double U_11_13 = 0.0 - L_11_9 * U_9_13;
              double U_11_14 = W_11_14;
              double U_11_15 = W_11_15 - L_11_9 * U_9_15;
              double L_12_11 = (0.0 - L_12_9 * U_9_11) / U_11_11;
              double L_14_11 = (W_14_11 - L_14_9 * U_9_11) / U_11_11;
              double L_15_11 = (W_15_11 - L_15_9 * U_9_11) / U_11_11;
              double U_12_12 = W_12_12 - L_12_5 * U_5_12 - L_12_9 * U_9_12 - L_12_11 * U_11_12;
              double U_12_13 = W_12_13 - L_12_5 * U_5_13 - L_12_6 * U_6_13 - L_12_9 * U_9_13 - L_12_11 * U_11_13;
              double U_12_14 = 0.0 - L_12_11 * U_11_14;
              double U_12_15 = W_12_15 - L_12_9 * U_9_15 - L_12_11 * U_11_15;
              double L_14_12 = (0.0 - L_14_9 * U_9_12 - L_14_11 * U_11_12) / U_12_12;
              double L_15_12 = (0.0 - L_15_9 * U_9_12 - L_15_11 * U_11_12) / U_12_12;
              double U_13_13 = W_13_13;
              double L_14_13 = (W_14_13 - L_14_8 * U_8_13 - L_14_9 * U_9_13 - L_14_10 * U_10_13 - L_14_11 * U_11_13 - L_14_12 * U_12_13) / U_13_13;
              double L_15_13 = (W_15_13 - L_15_9 * U_9_13 - L_15_10 * U_10_13 - L_15_11 * U_11_13 - L_15_12 * U_12_13) / U_13_13;
              // Block 9: species [N2O5]
              double L_16_13 = (W_16_13) / U_13_13;
              double U_14_14 = W_14_14 - L_14_11 * U_11_14 - L_14_12 * U_12_14;
              double U_14_15 = W_14_15 - L_14_9 * U_9_15 - L_14_11 * U_11_15 - L_14_12 * U_12_15;
              double U_14_16 = W_14_16;
              double L_15_14 = (W_15_14 - L_15_11 * U_11_14 - L_15_12 * U_12_14) / U_14_14;
              double L_16_14 = (W_16_14) / U_14_14;
              double U_15_15 = W_15_15 - L_15_9 * U_9_15 - L_15_11 * U_11_15 - L_15_12 * U_12_15 - L_15_14 * U_14_15;
              double U_15_16 = W_15_16 - L_15_14 * U_14_16;
              double L_16_15 = (W_16_15 - L_16_14 * U_14_15) / U_15_15;
              double U_16_16 = W_16_16 - L_16_14 * U_14_16 - L_16_15 * U_15_16;


              // --- Adjoint stage unrolling (s=3 stages, backward) ---
              // Driving vector: lam_k = lambda_{n+1} (constant for this step)
              const double lam_0 = lambda(0);
              const double lam_1 = lambda(1);
              const double lam_2 = lambda(2);
              const double lam_3 = lambda(3);
              const double lam_4 = lambda(4);
              const double lam_5 = lambda(5);
              const double lam_6 = lambda(6);
              const double lam_7 = lambda(7);
              const double lam_8 = lambda(8);
              const double lam_9 = lambda(9);
              const double lam_10 = lambda(10);
              const double lam_11 = lambda(11);
              const double lam_12 = lambda(12);
              const double lam_13 = lambda(13);
              const double lam_14 = lambda(14);
              const double lam_15 = lambda(15);
              const double lam_16 = lambda(16);

              // Adjoint stage 3
              // v3: RHS for W^{-T} solve
              double v3_0 = -0.42772256543218573 * lam_0;
              double v3_1 = -0.42772256543218573 * lam_1;
              double v3_2 = -0.42772256543218573 * lam_2;
              double v3_3 = -0.42772256543218573 * lam_3;
              double v3_4 = -0.42772256543218573 * lam_4;
              double v3_5 = -0.42772256543218573 * lam_5;
              double v3_6 = -0.42772256543218573 * lam_6;
              double v3_7 = -0.42772256543218573 * lam_7;
              double v3_8 = -0.42772256543218573 * lam_8;
              double v3_9 = -0.42772256543218573 * lam_9;
              double v3_10 = -0.42772256543218573 * lam_10;
              double v3_11 = -0.42772256543218573 * lam_11;
              double v3_12 = -0.42772256543218573 * lam_12;
              double v3_13 = -0.42772256543218573 * lam_13;
              double v3_14 = -0.42772256543218573 * lam_14;
              double v3_15 = -0.42772256543218573 * lam_15;
              double v3_16 = -0.42772256543218573 * lam_16;
              // W^{-T} solve for u3
              // Block 0: u3 transpose forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double yt3_0 = v3_0 / U_0_0;
              // Block 1: u3 transpose forward sub [M]
              double yt3_1 = v3_1 / U_1_1;
              double yt3_2 = v3_2 / U_2_2;
              double yt3_3 = v3_3 / U_3_3;
              double yt3_4 = v3_4 / U_4_4;
              double yt3_5 = v3_5 / U_5_5;
              // Block 2: u3 transpose forward sub [HO2]
              double yt3_6 = (v3_6 - U_5_6 * yt3_5) / U_6_6;
              // Block 3: u3 transpose forward sub [H2O2, HNO3]
              double yt3_7 = v3_7 / U_7_7;
              // Block 4: u3 transpose forward sub [OH]
              double yt3_8 = v3_8 / U_8_8;
              double yt3_9 = (v3_9 - U_5_9 * yt3_5) / U_9_9;
              double yt3_10 = (v3_10 - U_8_10 * yt3_8) / U_10_10;
              // Block 5: u3 transpose forward sub [NO]
              double yt3_11 = (v3_11 - U_9_11 * yt3_9) / U_11_11;
              // Block 6: u3 transpose forward sub [O3P]
              double yt3_12 = (v3_12 - U_5_12 * yt3_5 - U_9_12 * yt3_9 - U_11_12 * yt3_11) / U_12_12;
              double yt3_13 = (v3_13 - U_5_13 * yt3_5 - U_6_13 * yt3_6 - U_7_13 * yt3_7 - U_8_13 * yt3_8 - U_9_13 * yt3_9 - U_10_13 * yt3_10 - U_11_13 * yt3_11 - U_12_13 * yt3_12) / U_13_13;
              // Block 7: u3 transpose forward sub [NO3]
              double yt3_14 = (v3_14 - U_11_14 * yt3_11 - U_12_14 * yt3_12) / U_14_14;
              // Block 8: u3 transpose forward sub [NO2]
              double yt3_15 = (v3_15 - U_9_15 * yt3_9 - U_11_15 * yt3_11 - U_12_15 * yt3_12 - U_14_15 * yt3_14) / U_15_15;
              // Block 9: u3 transpose forward sub [N2O5]
              double yt3_16 = (v3_16 - U_14_16 * yt3_14 - U_15_16 * yt3_15) / U_16_16;
              // Block 9: u3 transpose backward sub [N2O5]
              double u3_16 = yt3_16;
              // Block 8: u3 transpose backward sub [NO2]
              double u3_15 = yt3_15 - L_16_15 * u3_16;
              // Block 7: u3 transpose backward sub [NO3]
              double u3_14 = yt3_14 - L_15_14 * u3_15 - L_16_14 * u3_16;
              // Block 0: u3 transpose backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double u3_13 = yt3_13 - L_14_13 * u3_14 - L_15_13 * u3_15 - L_16_13 * u3_16;
              // Block 6: u3 transpose backward sub [O3P]
              double u3_12 = yt3_12 - L_14_12 * u3_14 - L_15_12 * u3_15;
              // Block 5: u3 transpose backward sub [NO]
              double u3_11 = yt3_11 - L_12_11 * u3_12 - L_14_11 * u3_14 - L_15_11 * u3_15;
              // Block 3: u3 transpose backward sub [H2O2, HNO3]
              double u3_10 = yt3_10 - L_14_10 * u3_14 - L_15_10 * u3_15;
              double u3_9 = yt3_9 - L_11_9 * u3_11 - L_12_9 * u3_12 - L_14_9 * u3_14 - L_15_9 * u3_15;
              // Block 4: u3 transpose backward sub [OH]
              double u3_8 = yt3_8 - L_10_8 * u3_10 - L_14_8 * u3_14;
              double u3_7 = yt3_7 - L_8_7 * u3_8;
              // Block 2: u3 transpose backward sub [HO2]
              double u3_6 = yt3_6 - L_7_6 * u3_7 - L_9_6 * u3_9 - L_12_6 * u3_12;
              double u3_5 = yt3_5 - L_9_5 * u3_9 - L_12_5 * u3_12;
              double u3_4 = yt3_4 - L_8_4 * u3_8 - L_11_4 * u3_11;
              double u3_3 = yt3_3;
              double u3_2 = yt3_2;
              // Block 1: u3 transpose backward sub [M]
              double u3_1 = yt3_1;
              double u3_0 = yt3_0;


              // Adjoint stage 2
              // v2: RHS for W^{-T} solve
              double v2_0 = 6.1697947043828245 * lam_0 + (9.20767942983308 / h) * u3_0;
              double v2_1 = 6.1697947043828245 * lam_1 + (9.20767942983308 / h) * u3_1;
              double v2_2 = 6.1697947043828245 * lam_2 + (9.20767942983308 / h) * u3_2;
              double v2_3 = 6.1697947043828245 * lam_3 + (9.20767942983308 / h) * u3_3;
              double v2_4 = 6.1697947043828245 * lam_4 + (9.20767942983308 / h) * u3_4;
              double v2_5 = 6.1697947043828245 * lam_5 + (9.20767942983308 / h) * u3_5;
              double v2_6 = 6.1697947043828245 * lam_6 + (9.20767942983308 / h) * u3_6;
              double v2_7 = 6.1697947043828245 * lam_7 + (9.20767942983308 / h) * u3_7;
              double v2_8 = 6.1697947043828245 * lam_8 + (9.20767942983308 / h) * u3_8;
              double v2_9 = 6.1697947043828245 * lam_9 + (9.20767942983308 / h) * u3_9;
              double v2_10 = 6.1697947043828245 * lam_10 + (9.20767942983308 / h) * u3_10;
              double v2_11 = 6.1697947043828245 * lam_11 + (9.20767942983308 / h) * u3_11;
              double v2_12 = 6.1697947043828245 * lam_12 + (9.20767942983308 / h) * u3_12;
              double v2_13 = 6.1697947043828245 * lam_13 + (9.20767942983308 / h) * u3_13;
              double v2_14 = 6.1697947043828245 * lam_14 + (9.20767942983308 / h) * u3_14;
              double v2_15 = 6.1697947043828245 * lam_15 + (9.20767942983308 / h) * u3_15;
              double v2_16 = 6.1697947043828245 * lam_16 + (9.20767942983308 / h) * u3_16;
              // W^{-T} solve for u2
              // Block 0: u2 transpose forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double yt2_0 = v2_0 / U_0_0;
              // Block 1: u2 transpose forward sub [M]
              double yt2_1 = v2_1 / U_1_1;
              double yt2_2 = v2_2 / U_2_2;
              double yt2_3 = v2_3 / U_3_3;
              double yt2_4 = v2_4 / U_4_4;
              double yt2_5 = v2_5 / U_5_5;
              // Block 2: u2 transpose forward sub [HO2]
              double yt2_6 = (v2_6 - U_5_6 * yt2_5) / U_6_6;
              // Block 3: u2 transpose forward sub [H2O2, HNO3]
              double yt2_7 = v2_7 / U_7_7;
              // Block 4: u2 transpose forward sub [OH]
              double yt2_8 = v2_8 / U_8_8;
              double yt2_9 = (v2_9 - U_5_9 * yt2_5) / U_9_9;
              double yt2_10 = (v2_10 - U_8_10 * yt2_8) / U_10_10;
              // Block 5: u2 transpose forward sub [NO]
              double yt2_11 = (v2_11 - U_9_11 * yt2_9) / U_11_11;
              // Block 6: u2 transpose forward sub [O3P]
              double yt2_12 = (v2_12 - U_5_12 * yt2_5 - U_9_12 * yt2_9 - U_11_12 * yt2_11) / U_12_12;
              double yt2_13 = (v2_13 - U_5_13 * yt2_5 - U_6_13 * yt2_6 - U_7_13 * yt2_7 - U_8_13 * yt2_8 - U_9_13 * yt2_9 - U_10_13 * yt2_10 - U_11_13 * yt2_11 - U_12_13 * yt2_12) / U_13_13;
              // Block 7: u2 transpose forward sub [NO3]
              double yt2_14 = (v2_14 - U_11_14 * yt2_11 - U_12_14 * yt2_12) / U_14_14;
              // Block 8: u2 transpose forward sub [NO2]
              double yt2_15 = (v2_15 - U_9_15 * yt2_9 - U_11_15 * yt2_11 - U_12_15 * yt2_12 - U_14_15 * yt2_14) / U_15_15;
              // Block 9: u2 transpose forward sub [N2O5]
              double yt2_16 = (v2_16 - U_14_16 * yt2_14 - U_15_16 * yt2_15) / U_16_16;
              // Block 9: u2 transpose backward sub [N2O5]
              double u2_16 = yt2_16;
              // Block 8: u2 transpose backward sub [NO2]
              double u2_15 = yt2_15 - L_16_15 * u2_16;
              // Block 7: u2 transpose backward sub [NO3]
              double u2_14 = yt2_14 - L_15_14 * u2_15 - L_16_14 * u2_16;
              // Block 0: u2 transpose backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double u2_13 = yt2_13 - L_14_13 * u2_14 - L_15_13 * u2_15 - L_16_13 * u2_16;
              // Block 6: u2 transpose backward sub [O3P]
              double u2_12 = yt2_12 - L_14_12 * u2_14 - L_15_12 * u2_15;
              // Block 5: u2 transpose backward sub [NO]
              double u2_11 = yt2_11 - L_12_11 * u2_12 - L_14_11 * u2_14 - L_15_11 * u2_15;
              // Block 3: u2 transpose backward sub [H2O2, HNO3]
              double u2_10 = yt2_10 - L_14_10 * u2_14 - L_15_10 * u2_15;
              double u2_9 = yt2_9 - L_11_9 * u2_11 - L_12_9 * u2_12 - L_14_9 * u2_14 - L_15_9 * u2_15;
              // Block 4: u2 transpose backward sub [OH]
              double u2_8 = yt2_8 - L_10_8 * u2_10 - L_14_8 * u2_14;
              double u2_7 = yt2_7 - L_8_7 * u2_8;
              // Block 2: u2 transpose backward sub [HO2]
              double u2_6 = yt2_6 - L_7_6 * u2_7 - L_9_6 * u2_9 - L_12_6 * u2_12;
              double u2_5 = yt2_5 - L_9_5 * u2_9 - L_12_5 * u2_12;
              double u2_4 = yt2_4 - L_8_4 * u2_8 - L_11_4 * u2_11;
              double u2_3 = yt2_3;
              double u2_2 = yt2_2;
              // Block 1: u2 transpose backward sub [M]
              double u2_1 = yt2_1;
              double u2_0 = yt2_0;


              // Adjoint stage 1
              // v1: RHS for W^{-T} solve
              double v1_0 = lam_0 + (-1.0156171083877703 / h) * u2_0 + (4.0759956452537702 / h) * u3_0;
              double v1_1 = lam_1 + (-1.0156171083877703 / h) * u2_1 + (4.0759956452537702 / h) * u3_1;
              double v1_2 = lam_2 + (-1.0156171083877703 / h) * u2_2 + (4.0759956452537702 / h) * u3_2;
              double v1_3 = lam_3 + (-1.0156171083877703 / h) * u2_3 + (4.0759956452537702 / h) * u3_3;
              double v1_4 = lam_4 + (J_4_4 * u2_4 + J_8_4 * u2_8 + J_11_4 * u2_11) + (-1.0156171083877703 / h) * u2_4 + (J_4_4 * u3_4 + J_8_4 * u3_8 + J_11_4 * u3_11) + (4.0759956452537702 / h) * u3_4;
              double v1_5 = lam_5 + (J_5_5 * u2_5 + J_9_5 * u2_9 + J_12_5 * u2_12) + (-1.0156171083877703 / h) * u2_5 + (J_5_5 * u3_5 + J_9_5 * u3_9 + J_12_5 * u3_12) + (4.0759956452537702 / h) * u3_5;
              double v1_6 = lam_6 + (J_5_6 * u2_5 + J_6_6 * u2_6 + J_7_6 * u2_7) + (-1.0156171083877703 / h) * u2_6 + (J_5_6 * u3_5 + J_6_6 * u3_6 + J_7_6 * u3_7) + (4.0759956452537702 / h) * u3_6;
              double v1_7 = lam_7 + (J_7_7 * u2_7 + J_8_7 * u2_8) + (-1.0156171083877703 / h) * u2_7 + (J_7_7 * u3_7 + J_8_7 * u3_8) + (4.0759956452537702 / h) * u3_7;
              double v1_8 = lam_8 + (J_8_8 * u2_8 + J_10_8 * u2_10 + J_14_8 * u2_14) + (-1.0156171083877703 / h) * u2_8 + (J_8_8 * u3_8 + J_10_8 * u3_10 + J_14_8 * u3_14) + (4.0759956452537702 / h) * u3_8;
              double v1_9 = lam_9 + (J_5_9 * u2_5 + J_9_9 * u2_9 + J_11_9 * u2_11 + J_12_9 * u2_12 + J_14_9 * u2_14 + J_15_9 * u2_15) + (-1.0156171083877703 / h) * u2_9 + (J_5_9 * u3_5 + J_9_9 * u3_9 + J_11_9 * u3_11 + J_12_9 * u3_12 + J_14_9 * u3_14 + J_15_9 * u3_15) + (4.0759956452537702 / h) * u3_9;
              double v1_10 = lam_10 + (J_8_10 * u2_8 + J_10_10 * u2_10 + J_14_10 * u2_14 + J_15_10 * u2_15) + (-1.0156171083877703 / h) * u2_10 + (J_8_10 * u3_8 + J_10_10 * u3_10 + J_14_10 * u3_14 + J_15_10 * u3_15) + (4.0759956452537702 / h) * u3_10;
              double v1_11 = lam_11 + (J_9_11 * u2_9 + J_11_11 * u2_11 + J_14_11 * u2_14 + J_15_11 * u2_15) + (-1.0156171083877703 / h) * u2_11 + (J_9_11 * u3_9 + J_11_11 * u3_11 + J_14_11 * u3_14 + J_15_11 * u3_15) + (4.0759956452537702 / h) * u3_11;
              double v1_12 = lam_12 + (J_5_12 * u2_5 + J_9_12 * u2_9 + J_12_12 * u2_12) + (-1.0156171083877703 / h) * u2_12 + (J_5_12 * u3_5 + J_9_12 * u3_9 + J_12_12 * u3_12) + (4.0759956452537702 / h) * u3_12;
              double v1_13 = lam_13 + (J_5_13 * u2_5 + J_6_13 * u2_6 + J_7_13 * u2_7 + J_8_13 * u2_8 + J_9_13 * u2_9 + J_10_13 * u2_10 + J_12_13 * u2_12 + J_14_13 * u2_14 + J_15_13 * u2_15 + J_16_13 * u2_16) + (-1.0156171083877703 / h) * u2_13 + (J_5_13 * u3_5 + J_6_13 * u3_6 + J_7_13 * u3_7 + J_8_13 * u3_8 + J_9_13 * u3_9 + J_10_13 * u3_10 + J_12_13 * u3_12 + J_14_13 * u3_14 + J_15_13 * u3_15 + J_16_13 * u3_16) + (4.0759956452537702 / h) * u3_13;
              double v1_14 = lam_14 + (J_11_14 * u2_11 + J_14_14 * u2_14 + J_15_14 * u2_15 + J_16_14 * u2_16) + (-1.0156171083877703 / h) * u2_14 + (J_11_14 * u3_11 + J_14_14 * u3_14 + J_15_14 * u3_15 + J_16_14 * u3_16) + (4.0759956452537702 / h) * u3_14;
              double v1_15 = lam_15 + (J_9_15 * u2_9 + J_11_15 * u2_11 + J_12_15 * u2_12 + J_14_15 * u2_14 + J_15_15 * u2_15 + J_16_15 * u2_16) + (-1.0156171083877703 / h) * u2_15 + (J_9_15 * u3_9 + J_11_15 * u3_11 + J_12_15 * u3_12 + J_14_15 * u3_14 + J_15_15 * u3_15 + J_16_15 * u3_16) + (4.0759956452537702 / h) * u3_15;
              double v1_16 = lam_16 + (J_14_16 * u2_14 + J_15_16 * u2_15 + J_16_16 * u2_16) + (-1.0156171083877703 / h) * u2_16 + (J_14_16 * u3_14 + J_15_16 * u3_15 + J_16_16 * u3_16) + (4.0759956452537702 / h) * u3_16;
              // W^{-T} solve for u1
              // Block 0: u1 transpose forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double yt1_0 = v1_0 / U_0_0;
              // Block 1: u1 transpose forward sub [M]
              double yt1_1 = v1_1 / U_1_1;
              double yt1_2 = v1_2 / U_2_2;
              double yt1_3 = v1_3 / U_3_3;
              double yt1_4 = v1_4 / U_4_4;
              double yt1_5 = v1_5 / U_5_5;
              // Block 2: u1 transpose forward sub [HO2]
              double yt1_6 = (v1_6 - U_5_6 * yt1_5) / U_6_6;
              // Block 3: u1 transpose forward sub [H2O2, HNO3]
              double yt1_7 = v1_7 / U_7_7;
              // Block 4: u1 transpose forward sub [OH]
              double yt1_8 = v1_8 / U_8_8;
              double yt1_9 = (v1_9 - U_5_9 * yt1_5) / U_9_9;
              double yt1_10 = (v1_10 - U_8_10 * yt1_8) / U_10_10;
              // Block 5: u1 transpose forward sub [NO]
              double yt1_11 = (v1_11 - U_9_11 * yt1_9) / U_11_11;
              // Block 6: u1 transpose forward sub [O3P]
              double yt1_12 = (v1_12 - U_5_12 * yt1_5 - U_9_12 * yt1_9 - U_11_12 * yt1_11) / U_12_12;
              double yt1_13 = (v1_13 - U_5_13 * yt1_5 - U_6_13 * yt1_6 - U_7_13 * yt1_7 - U_8_13 * yt1_8 - U_9_13 * yt1_9 - U_10_13 * yt1_10 - U_11_13 * yt1_11 - U_12_13 * yt1_12) / U_13_13;
              // Block 7: u1 transpose forward sub [NO3]
              double yt1_14 = (v1_14 - U_11_14 * yt1_11 - U_12_14 * yt1_12) / U_14_14;
              // Block 8: u1 transpose forward sub [NO2]
              double yt1_15 = (v1_15 - U_9_15 * yt1_9 - U_11_15 * yt1_11 - U_12_15 * yt1_12 - U_14_15 * yt1_14) / U_15_15;
              // Block 9: u1 transpose forward sub [N2O5]
              double yt1_16 = (v1_16 - U_14_16 * yt1_14 - U_15_16 * yt1_15) / U_16_16;
              // Block 9: u1 transpose backward sub [N2O5]
              double u1_16 = yt1_16;
              // Block 8: u1 transpose backward sub [NO2]
              double u1_15 = yt1_15 - L_16_15 * u1_16;
              // Block 7: u1 transpose backward sub [NO3]
              double u1_14 = yt1_14 - L_15_14 * u1_15 - L_16_14 * u1_16;
              // Block 0: u1 transpose backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double u1_13 = yt1_13 - L_14_13 * u1_14 - L_15_13 * u1_15 - L_16_13 * u1_16;
              // Block 6: u1 transpose backward sub [O3P]
              double u1_12 = yt1_12 - L_14_12 * u1_14 - L_15_12 * u1_15;
              // Block 5: u1 transpose backward sub [NO]
              double u1_11 = yt1_11 - L_12_11 * u1_12 - L_14_11 * u1_14 - L_15_11 * u1_15;
              // Block 3: u1 transpose backward sub [H2O2, HNO3]
              double u1_10 = yt1_10 - L_14_10 * u1_14 - L_15_10 * u1_15;
              double u1_9 = yt1_9 - L_11_9 * u1_11 - L_12_9 * u1_12 - L_14_9 * u1_14 - L_15_9 * u1_15;
              // Block 4: u1 transpose backward sub [OH]
              double u1_8 = yt1_8 - L_10_8 * u1_10 - L_14_8 * u1_14;
              double u1_7 = yt1_7 - L_8_7 * u1_8;
              // Block 2: u1 transpose backward sub [HO2]
              double u1_6 = yt1_6 - L_7_6 * u1_7 - L_9_6 * u1_9 - L_12_6 * u1_12;
              double u1_5 = yt1_5 - L_9_5 * u1_9 - L_12_5 * u1_12;
              double u1_4 = yt1_4 - L_8_4 * u1_8 - L_11_4 * u1_11;
              double u1_3 = yt1_3;
              double u1_2 = yt1_2;
              // Block 1: u1 transpose backward sub [M]
              double u1_1 = yt1_1;
              double u1_0 = yt1_0;


              // --- Lambda update: lambda_n = lambda_{n+1} + Σ u_i ---
              lambda(0) = lam_0 + u1_0 + u2_0 + u3_0;
              lambda(1) = lam_1 + u1_1 + u2_1 + u3_1;
              lambda(2) = lam_2 + u1_2 + u2_2 + u3_2;
              lambda(3) = lam_3 + u1_3 + u2_3 + u3_3;
              lambda(4) = lam_4 + u1_4 + u2_4 + u3_4;
              lambda(5) = lam_5 + u1_5 + u2_5 + u3_5;
              lambda(6) = lam_6 + u1_6 + u2_6 + u3_6;
              lambda(7) = lam_7 + u1_7 + u2_7 + u3_7;
              lambda(8) = lam_8 + u1_8 + u2_8 + u3_8;
              lambda(9) = lam_9 + u1_9 + u2_9 + u3_9;
              lambda(10) = lam_10 + u1_10 + u2_10 + u3_10;
              lambda(11) = lam_11 + u1_11 + u2_11 + u3_11;
              lambda(12) = lam_12 + u1_12 + u2_12 + u3_12;
              lambda(13) = lam_13 + u1_13 + u2_13 + u3_13;
              lambda(14) = lam_14 + u1_14 + u2_14 + u3_14;
              lambda(15) = lam_15 + u1_15 + u2_15 + u3_15;
              lambda(16) = lam_16 + u1_16 + u2_16 + u3_16;
          } // end for (step)
      }

      // Tangent Linear Model: forward propagation of perturbation δC
      // Uses checkpoint buffer from integrate_fwd_checkpoint().
      template <class StateView, class PertView>
      KOKKOS_INLINE_FUNCTION void integrate_tlm(
          double dt_total, const StateView& state_0,
          PertView& delta_C, Kokkos::View<const double*, memory_space> jvals_view,
          const CheckpointBuffer& chk, const double temp = 298.15, const double rh = 0.5, const double air_density = 2.46e+19) const
      {
          const double* jvals = jvals_view.data();
          // Environment conditions must match the forward checkpoint so the
          // recomputed Jacobian reflects the same rate law the state was
          // integrated under.
          const double Temp = temp;
          const double RH = rh;
          const double AirDensity = air_density;
          (void)RH;  // Reserved for future deliquescence modeling
          const double g = 0.435866521508459;

          // Local aliases for delta_C (TLM stages use dC_k naming)
          double dC_0 = delta_C(0);
          double dC_1 = delta_C(1);
          double dC_2 = delta_C(2);
          double dC_3 = delta_C(3);
          double dC_4 = delta_C(4);
          double dC_5 = delta_C(5);
          double dC_6 = delta_C(6);
          double dC_7 = delta_C(7);
          double dC_8 = delta_C(8);
          double dC_9 = delta_C(9);
          double dC_10 = delta_C(10);
          double dC_11 = delta_C(11);
          double dC_12 = delta_C(12);
          double dC_13 = delta_C(13);
          double dC_14 = delta_C(14);
          double dC_15 = delta_C(15);
          double dC_16 = delta_C(16);


          // === TLM Forward Propagation ===
          for (int step = 0; step < chk.num_steps; step++) {
              const double h = chk.h[step];
              const double inv_g_h = 1.0 / (0.435866521508459 * h);

              // Reload checkpointed state
              const double S_0 = chk.state[step][0];
              const double S_1 = chk.state[step][1];
              const double S_2 = chk.state[step][2];
              const double S_3 = chk.state[step][3];
              const double S_4 = chk.state[step][4];
              const double S_5 = chk.state[step][5];
              const double S_6 = chk.state[step][6];
              const double S_7 = chk.state[step][7];
              const double S_8 = chk.state[step][8];
              const double S_9 = chk.state[step][9];
              const double S_10 = chk.state[step][10];
              const double S_11 = chk.state[step][11];
              const double S_12 = chk.state[step][12];
              const double S_13 = chk.state[step][13];
              const double S_14 = chk.state[step][14];
              const double S_15 = chk.state[step][15];
              const double S_16 = chk.state[step][16];

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_4_4 = -1.0*jvals[1];
              double J_5_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_5_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_5_9 = 1.6e-11*S_12*Kokkos::exp(-2060.0/Temp);
              double J_5_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) + 1.6e-11*S_9*Kokkos::exp(-2060.0/Temp);
              double J_5_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp) - 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_6_6 = -1.7e-33*S_13*Kokkos::exp(1000.0/Temp) - 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_6_13 = -1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
              double J_7_6 = 1.7e-33*S_13*Kokkos::exp(1000.0/Temp) + 2.3e-13*Kokkos::exp(600.0/Temp);
              double J_7_7 = -1.0*jvals[3];
              double J_7_13 = 1.7e-33*S_6*Kokkos::exp(1000.0/Temp);
              double J_8_4 = 1.0*jvals[1];
              double J_8_7 = 2.0*jvals[3];
              double J_8_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_8_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) + 1.0*jvals[2];
              double J_8_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_9_5 = 4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_9_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp) - 8e-12*S_12*Kokkos::exp(-2060.0/Temp);
              double J_9_11 = -1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_9_12 = 4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
              double J_9_13 = 4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_9_15 = -1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
              double J_10_8 = -1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_10_10 = -1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp)) - 1.0*jvals[2];
              double J_10_13 = -1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_11_4 = 1.0*jvals[1];
              double J_11_9 = -1.8e-12*S_11*Kokkos::exp(-1370.0/Temp);
              double J_11_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp) - 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_11_14 = -1.8e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_11_15 = 1.0*jvals[0];
              double J_12_5 = -4.9010355073085e-27*S_13*S_12*Kokkos::pow(Temp, -2.8);
              double J_12_9 = -8e-12*S_12*Kokkos::exp(-2060.0/Temp);
              double J_12_12 = -4.9010355073085e-27*S_13*S_5*Kokkos::pow(Temp, -2.8) - 8e-12*S_9*Kokkos::exp(-2060.0/Temp);
              double J_12_13 = -4.9010355073085e-27*S_5*S_12*Kokkos::pow(Temp, -2.8);
              double J_12_15 = 1.0*jvals[0];
              double J_14_8 = 1.0*S_10*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_14_9 = 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
              double J_14_10 = 1.0*S_8*(1.9e-33*S_13*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0) + 7.2e-15*Kokkos::exp(785.0/Temp));
              double J_14_11 = -1.8e-11*S_14*Kokkos::exp(110.0/Temp);
              double J_14_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 1.0*S_10*S_8*(-8.80487804878049e-51*S_13*Kokkos::exp(10.0/Temp)/Kokkos::pow(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0, 2) + 1.9e-33*Kokkos::exp(725.0/Temp)/(4.63414634146341e-18*S_13*Kokkos::exp(-715.0/Temp) + 1.0));
              double J_14_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.8e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_14_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp);
              double J_14_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
              double J_15_9 = 1.8e-12*S_11*Kokkos::exp(-1370.0/Temp) - 1.4e-13*S_15*Kokkos::exp(-2470.0/Temp);
              double J_15_10 = 1.0*jvals[2];
              double J_15_11 = 3.6e-11*S_14*Kokkos::exp(110.0/Temp) + 1.8e-12*S_9*Kokkos::exp(-1370.0/Temp);
              double J_15_13 = 3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) - 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) - 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) + 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) + 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
              double J_15_14 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 3.6e-11*S_11*Kokkos::exp(110.0/Temp);
              double J_15_15 = -2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) - 1.4e-13*S_9*Kokkos::exp(-2470.0/Temp) - 1.0*jvals[0];
              double J_15_16 = 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);
              double J_16_13 = -3.92e-48*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*S_14*Kokkos::exp(7.2/Temp)/Kokkos::pow(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0, 2) + 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0) + 4.47164309881952e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_15*S_14*Kokkos::exp(3.5/Temp)*Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp))/((1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0)*Kokkos::pow(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(Kokkos::log(10.0) * Kokkos::log(10.0))) + 2.66481679740968e-39*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_16*Kokkos::pow(Temp, 7.1)*Kokkos::exp(10920.0/Temp)/Kokkos::pow(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0, 2) - 2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0) - 3.41495283971347e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_16*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)*Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp))/(Kokkos::pow(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0, 2)*(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0)*(Kokkos::log(10.0) * Kokkos::log(10.0)));
              double J_16_14 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_15*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
              double J_16_15 = 2.8e-30*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.4e-18*S_13*Kokkos::exp(3.7/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*S_14*Kokkos::exp(3.5/Temp)/(1.4e-18*S_13*Kokkos::exp(3.7/Temp) + 1.0);
              double J_16_16 = -2.13833433033195e-12*Kokkos::pow(0.45, 1.0/(Kokkos::pow(Kokkos::log(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp)), 2)/(Kokkos::log(10.0) * Kokkos::log(10.0)) + 1.0))*S_13*Kokkos::pow(Temp, 3.5)*Kokkos::exp(11000.0/Temp)/(1.24621148321367e-27*S_13*Kokkos::pow(Temp, 3.6)*Kokkos::exp(-80.0/Temp) + 1.0);

              // Form iteration matrix W = inv_g_h*I - J
              double W_0_0 = inv_g_h;
              double W_1_1 = inv_g_h;
              double W_2_2 = inv_g_h;
              double W_3_3 = inv_g_h;
              double W_4_4 = inv_g_h - J_4_4;
              double W_5_5 = inv_g_h - J_5_5;
              double W_5_6 = -J_5_6;
              double W_5_9 = -J_5_9;
              double W_5_12 = -J_5_12;
              double W_5_13 = -J_5_13;
              double W_6_6 = inv_g_h - J_6_6;
              double W_6_13 = -J_6_13;
              double W_7_6 = -J_7_6;
              double W_7_7 = inv_g_h - J_7_7;
              double W_7_13 = -J_7_13;
              double W_8_4 = -J_8_4;
              double W_8_7 = -J_8_7;
              double W_8_8 = inv_g_h - J_8_8;
              double W_8_10 = -J_8_10;
              double W_8_13 = -J_8_13;
              double W_9_5 = -J_9_5;
              double W_9_9 = inv_g_h - J_9_9;
              double W_9_11 = -J_9_11;
              double W_9_12 = -J_9_12;
              double W_9_13 = -J_9_13;
              double W_9_15 = -J_9_15;
              double W_10_8 = -J_10_8;
              double W_10_10 = inv_g_h - J_10_10;
              double W_10_13 = -J_10_13;
              double W_11_4 = -J_11_4;
              double W_11_9 = -J_11_9;
              double W_11_11 = inv_g_h - J_11_11;
              double W_11_14 = -J_11_14;
              double W_11_15 = -J_11_15;
              double W_12_5 = -J_12_5;
              double W_12_9 = -J_12_9;
              double W_12_12 = inv_g_h - J_12_12;
              double W_12_13 = -J_12_13;
              double W_12_15 = -J_12_15;
              double W_13_13 = inv_g_h;
              double W_14_8 = -J_14_8;
              double W_14_9 = -J_14_9;
              double W_14_10 = -J_14_10;
              double W_14_11 = -J_14_11;
              double W_14_13 = -J_14_13;
              double W_14_14 = inv_g_h - J_14_14;
              double W_14_15 = -J_14_15;
              double W_14_16 = -J_14_16;
              double W_15_9 = -J_15_9;
              double W_15_10 = -J_15_10;
              double W_15_11 = -J_15_11;
              double W_15_13 = -J_15_13;
              double W_15_14 = -J_15_14;
              double W_15_15 = inv_g_h - J_15_15;
              double W_15_16 = -J_15_16;
              double W_16_13 = -J_16_13;
              double W_16_14 = -J_16_14;
              double W_16_15 = -J_16_15;
              double W_16_16 = inv_g_h - J_16_16;

              // Symbolic LU Factorization
              // Block 0: species [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double U_0_0 = W_0_0;
              // Block 1: species [M]
              double U_1_1 = W_1_1;
              double U_2_2 = W_2_2;
              double U_3_3 = W_3_3;
              double U_4_4 = W_4_4;
              // Block 4: species [OH]
              double L_8_4 = (W_8_4) / U_4_4;
              // Block 5: species [NO]
              double L_11_4 = (W_11_4) / U_4_4;
              double U_5_5 = W_5_5;
              double U_5_6 = W_5_6;
              double U_5_9 = W_5_9;
              double U_5_12 = W_5_12;
              double U_5_13 = W_5_13;
              double L_9_5 = (W_9_5) / U_5_5;
              // Block 6: species [O3P]
              double L_12_5 = (W_12_5) / U_5_5;
              // Block 2: species [HO2]
              double U_6_6 = W_6_6;
              double U_6_13 = W_6_13;
              // Block 3: species [H2O2, HNO3]
              double L_7_6 = (W_7_6) / U_6_6;
              double L_9_6 = (0.0 - L_9_5 * U_5_6) / U_6_6;
              double L_12_6 = (0.0 - L_12_5 * U_5_6) / U_6_6;
              double U_7_7 = W_7_7;
              double U_7_13 = W_7_13 - L_7_6 * U_6_13;
              double L_8_7 = (W_8_7) / U_7_7;
              double U_8_8 = W_8_8;
              double U_8_10 = W_8_10;
              double U_8_13 = W_8_13 - L_8_7 * U_7_13;
              double L_10_8 = (W_10_8) / U_8_8;
              // Block 7: species [NO3]
              double L_14_8 = (W_14_8) / U_8_8;
              double U_9_9 = W_9_9 - L_9_5 * U_5_9;
              double U_9_11 = W_9_11;
              double U_9_12 = W_9_12 - L_9_5 * U_5_12;
              double U_9_13 = W_9_13 - L_9_5 * U_5_13 - L_9_6 * U_6_13;
              double U_9_15 = W_9_15;
              double L_11_9 = (W_11_9) / U_9_9;
              double L_12_9 = (W_12_9 - L_12_5 * U_5_9) / U_9_9;
              double L_14_9 = (W_14_9) / U_9_9;
              // Block 8: species [NO2]
              double L_15_9 = (W_15_9) / U_9_9;
              double U_10_10 = W_10_10 - L_10_8 * U_8_10;
              double U_10_13 = W_10_13 - L_10_8 * U_8_13;
              double L_14_10 = (W_14_10 - L_14_8 * U_8_10) / U_10_10;
              double L_15_10 = (W_15_10) / U_10_10;
              double U_11_11 = W_11_11 - L_11_9 * U_9_11;
              double U_11_12 = 0.0 - L_11_9 * U_9_12;
              double U_11_13 = 0.0 - L_11_9 * U_9_13;
              double U_11_14 = W_11_14;
              double U_11_15 = W_11_15 - L_11_9 * U_9_15;
              double L_12_11 = (0.0 - L_12_9 * U_9_11) / U_11_11;
              double L_14_11 = (W_14_11 - L_14_9 * U_9_11) / U_11_11;
              double L_15_11 = (W_15_11 - L_15_9 * U_9_11) / U_11_11;
              double U_12_12 = W_12_12 - L_12_5 * U_5_12 - L_12_9 * U_9_12 - L_12_11 * U_11_12;
              double U_12_13 = W_12_13 - L_12_5 * U_5_13 - L_12_6 * U_6_13 - L_12_9 * U_9_13 - L_12_11 * U_11_13;
              double U_12_14 = 0.0 - L_12_11 * U_11_14;
              double U_12_15 = W_12_15 - L_12_9 * U_9_15 - L_12_11 * U_11_15;
              double L_14_12 = (0.0 - L_14_9 * U_9_12 - L_14_11 * U_11_12) / U_12_12;
              double L_15_12 = (0.0 - L_15_9 * U_9_12 - L_15_11 * U_11_12) / U_12_12;
              double U_13_13 = W_13_13;
              double L_14_13 = (W_14_13 - L_14_8 * U_8_13 - L_14_9 * U_9_13 - L_14_10 * U_10_13 - L_14_11 * U_11_13 - L_14_12 * U_12_13) / U_13_13;
              double L_15_13 = (W_15_13 - L_15_9 * U_9_13 - L_15_10 * U_10_13 - L_15_11 * U_11_13 - L_15_12 * U_12_13) / U_13_13;
              // Block 9: species [N2O5]
              double L_16_13 = (W_16_13) / U_13_13;
              double U_14_14 = W_14_14 - L_14_11 * U_11_14 - L_14_12 * U_12_14;
              double U_14_15 = W_14_15 - L_14_9 * U_9_15 - L_14_11 * U_11_15 - L_14_12 * U_12_15;
              double U_14_16 = W_14_16;
              double L_15_14 = (W_15_14 - L_15_11 * U_11_14 - L_15_12 * U_12_14) / U_14_14;
              double L_16_14 = (W_16_14) / U_14_14;
              double U_15_15 = W_15_15 - L_15_9 * U_9_15 - L_15_11 * U_11_15 - L_15_12 * U_12_15 - L_15_14 * U_14_15;
              double U_15_16 = W_15_16 - L_15_14 * U_14_16;
              double L_16_15 = (W_16_15 - L_16_14 * U_14_15) / U_15_15;
              double U_16_16 = W_16_16 - L_16_14 * U_14_16 - L_16_15 * U_15_16;



              // --- TLM Stage 1 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs1_0 = 0.0;
              double tlm_rhs1_1 = 0.0;
              double tlm_rhs1_2 = 0.0;
              double tlm_rhs1_3 = 0.0;
              double tlm_rhs1_4 = J_4_4 * dC_4;
              double tlm_rhs1_5 = J_5_5 * dC_5 + J_5_6 * dC_6 + J_5_9 * dC_9 + J_5_12 * dC_12 + J_5_13 * dC_13;
              double tlm_rhs1_6 = J_6_6 * dC_6 + J_6_13 * dC_13;
              double tlm_rhs1_7 = J_7_6 * dC_6 + J_7_7 * dC_7 + J_7_13 * dC_13;
              double tlm_rhs1_8 = J_8_4 * dC_4 + J_8_7 * dC_7 + J_8_8 * dC_8 + J_8_10 * dC_10 + J_8_13 * dC_13;
              double tlm_rhs1_9 = J_9_5 * dC_5 + J_9_9 * dC_9 + J_9_11 * dC_11 + J_9_12 * dC_12 + J_9_13 * dC_13 + J_9_15 * dC_15;
              double tlm_rhs1_10 = J_10_8 * dC_8 + J_10_10 * dC_10 + J_10_13 * dC_13;
              double tlm_rhs1_11 = J_11_4 * dC_4 + J_11_9 * dC_9 + J_11_11 * dC_11 + J_11_14 * dC_14 + J_11_15 * dC_15;
              double tlm_rhs1_12 = J_12_5 * dC_5 + J_12_9 * dC_9 + J_12_12 * dC_12 + J_12_13 * dC_13 + J_12_15 * dC_15;
              double tlm_rhs1_13 = 0.0;
              double tlm_rhs1_14 = J_14_8 * dC_8 + J_14_9 * dC_9 + J_14_10 * dC_10 + J_14_11 * dC_11 + J_14_13 * dC_13 + J_14_14 * dC_14 + J_14_15 * dC_15 + J_14_16 * dC_16;
              double tlm_rhs1_15 = J_15_9 * dC_9 + J_15_10 * dC_10 + J_15_11 * dC_11 + J_15_13 * dC_13 + J_15_14 * dC_14 + J_15_15 * dC_15 + J_15_16 * dC_16;
              double tlm_rhs1_16 = J_16_13 * dC_13 + J_16_14 * dC_14 + J_16_15 * dC_15 + J_16_16 * dC_16;
              // Solve W * dK1 = tlm_rhs1
              // Block 0: dK1 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double tlm_y1_0 = tlm_rhs1_0;
              // Block 1: dK1 forward sub [M]
              double tlm_y1_1 = tlm_rhs1_1;
              double tlm_y1_2 = tlm_rhs1_2;
              double tlm_y1_3 = tlm_rhs1_3;
              double tlm_y1_4 = tlm_rhs1_4;
              double tlm_y1_5 = tlm_rhs1_5;
              // Block 2: dK1 forward sub [HO2]
              double tlm_y1_6 = tlm_rhs1_6;
              // Block 3: dK1 forward sub [H2O2, HNO3]
              double tlm_y1_7 = tlm_rhs1_7 - L_7_6 * tlm_y1_6;
              // Block 4: dK1 forward sub [OH]
              double tlm_y1_8 = tlm_rhs1_8 - L_8_4 * tlm_y1_4 - L_8_7 * tlm_y1_7;
              double tlm_y1_9 = tlm_rhs1_9 - L_9_5 * tlm_y1_5 - L_9_6 * tlm_y1_6;
              double tlm_y1_10 = tlm_rhs1_10 - L_10_8 * tlm_y1_8;
              // Block 5: dK1 forward sub [NO]
              double tlm_y1_11 = tlm_rhs1_11 - L_11_4 * tlm_y1_4 - L_11_9 * tlm_y1_9;
              // Block 6: dK1 forward sub [O3P]
              double tlm_y1_12 = tlm_rhs1_12 - L_12_5 * tlm_y1_5 - L_12_6 * tlm_y1_6 - L_12_9 * tlm_y1_9 - L_12_11 * tlm_y1_11;
              double tlm_y1_13 = tlm_rhs1_13;
              // Block 7: dK1 forward sub [NO3]
              double tlm_y1_14 = tlm_rhs1_14 - L_14_8 * tlm_y1_8 - L_14_9 * tlm_y1_9 - L_14_10 * tlm_y1_10 - L_14_11 * tlm_y1_11 - L_14_12 * tlm_y1_12 - L_14_13 * tlm_y1_13;
              // Block 8: dK1 forward sub [NO2]
              double tlm_y1_15 = tlm_rhs1_15 - L_15_9 * tlm_y1_9 - L_15_10 * tlm_y1_10 - L_15_11 * tlm_y1_11 - L_15_12 * tlm_y1_12 - L_15_13 * tlm_y1_13 - L_15_14 * tlm_y1_14;
              // Block 9: dK1 forward sub [N2O5]
              double tlm_y1_16 = tlm_rhs1_16 - L_16_13 * tlm_y1_13 - L_16_14 * tlm_y1_14 - L_16_15 * tlm_y1_15;
              // Block 9: dK1 backward sub [N2O5]
              double dK1_16 = tlm_y1_16 / U_16_16;
              // Block 8: dK1 backward sub [NO2]
              double dK1_15 = (tlm_y1_15 - U_15_16 * dK1_16) / U_15_15;
              // Block 7: dK1 backward sub [NO3]
              double dK1_14 = (tlm_y1_14 - U_14_15 * dK1_15 - U_14_16 * dK1_16) / U_14_14;
              // Block 0: dK1 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double dK1_13 = tlm_y1_13 / U_13_13;
              // Block 6: dK1 backward sub [O3P]
              double dK1_12 = (tlm_y1_12 - U_12_13 * dK1_13 - U_12_14 * dK1_14 - U_12_15 * dK1_15) / U_12_12;
              // Block 5: dK1 backward sub [NO]
              double dK1_11 = (tlm_y1_11 - U_11_12 * dK1_12 - U_11_13 * dK1_13 - U_11_14 * dK1_14 - U_11_15 * dK1_15) / U_11_11;
              // Block 3: dK1 backward sub [H2O2, HNO3]
              double dK1_10 = (tlm_y1_10 - U_10_13 * dK1_13) / U_10_10;
              double dK1_9 = (tlm_y1_9 - U_9_11 * dK1_11 - U_9_12 * dK1_12 - U_9_13 * dK1_13 - U_9_15 * dK1_15) / U_9_9;
              // Block 4: dK1 backward sub [OH]
              double dK1_8 = (tlm_y1_8 - U_8_10 * dK1_10 - U_8_13 * dK1_13) / U_8_8;
              double dK1_7 = (tlm_y1_7 - U_7_13 * dK1_13) / U_7_7;
              // Block 2: dK1 backward sub [HO2]
              double dK1_6 = (tlm_y1_6 - U_6_13 * dK1_13) / U_6_6;
              double dK1_5 = (tlm_y1_5 - U_5_6 * dK1_6 - U_5_9 * dK1_9 - U_5_12 * dK1_12 - U_5_13 * dK1_13) / U_5_5;
              double dK1_4 = tlm_y1_4 / U_4_4;
              double dK1_3 = tlm_y1_3 / U_3_3;
              double dK1_2 = tlm_y1_2 / U_2_2;
              // Block 1: dK1 backward sub [M]
              double dK1_1 = tlm_y1_1 / U_1_1;
              double dK1_0 = tlm_y1_0 / U_0_0;

              // --- TLM Stage 2 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs2_0 = (-1.0156171083877703 / h) * dK1_0;
              double tlm_rhs2_1 = (-1.0156171083877703 / h) * dK1_1;
              double tlm_rhs2_2 = (-1.0156171083877703 / h) * dK1_2;
              double tlm_rhs2_3 = (-1.0156171083877703 / h) * dK1_3;
              double tlm_rhs2_4 = J_4_4 * (dC_4 + dK1_4) + (-1.0156171083877703 / h) * dK1_4;
              double tlm_rhs2_5 = J_5_5 * (dC_5 + dK1_5) + J_5_6 * (dC_6 + dK1_6) + J_5_9 * (dC_9 + dK1_9) + J_5_12 * (dC_12 + dK1_12) + J_5_13 * (dC_13 + dK1_13) + (-1.0156171083877703 / h) * dK1_5;
              double tlm_rhs2_6 = J_6_6 * (dC_6 + dK1_6) + J_6_13 * (dC_13 + dK1_13) + (-1.0156171083877703 / h) * dK1_6;
              double tlm_rhs2_7 = J_7_6 * (dC_6 + dK1_6) + J_7_7 * (dC_7 + dK1_7) + J_7_13 * (dC_13 + dK1_13) + (-1.0156171083877703 / h) * dK1_7;
              double tlm_rhs2_8 = J_8_4 * (dC_4 + dK1_4) + J_8_7 * (dC_7 + dK1_7) + J_8_8 * (dC_8 + dK1_8) + J_8_10 * (dC_10 + dK1_10) + J_8_13 * (dC_13 + dK1_13) + (-1.0156171083877703 / h) * dK1_8;
              double tlm_rhs2_9 = J_9_5 * (dC_5 + dK1_5) + J_9_9 * (dC_9 + dK1_9) + J_9_11 * (dC_11 + dK1_11) + J_9_12 * (dC_12 + dK1_12) + J_9_13 * (dC_13 + dK1_13) + J_9_15 * (dC_15 + dK1_15) + (-1.0156171083877703 / h) * dK1_9;
              double tlm_rhs2_10 = J_10_8 * (dC_8 + dK1_8) + J_10_10 * (dC_10 + dK1_10) + J_10_13 * (dC_13 + dK1_13) + (-1.0156171083877703 / h) * dK1_10;
              double tlm_rhs2_11 = J_11_4 * (dC_4 + dK1_4) + J_11_9 * (dC_9 + dK1_9) + J_11_11 * (dC_11 + dK1_11) + J_11_14 * (dC_14 + dK1_14) + J_11_15 * (dC_15 + dK1_15) + (-1.0156171083877703 / h) * dK1_11;
              double tlm_rhs2_12 = J_12_5 * (dC_5 + dK1_5) + J_12_9 * (dC_9 + dK1_9) + J_12_12 * (dC_12 + dK1_12) + J_12_13 * (dC_13 + dK1_13) + J_12_15 * (dC_15 + dK1_15) + (-1.0156171083877703 / h) * dK1_12;
              double tlm_rhs2_13 = (-1.0156171083877703 / h) * dK1_13;
              double tlm_rhs2_14 = J_14_8 * (dC_8 + dK1_8) + J_14_9 * (dC_9 + dK1_9) + J_14_10 * (dC_10 + dK1_10) + J_14_11 * (dC_11 + dK1_11) + J_14_13 * (dC_13 + dK1_13) + J_14_14 * (dC_14 + dK1_14) + J_14_15 * (dC_15 + dK1_15) + J_14_16 * (dC_16 + dK1_16) + (-1.0156171083877703 / h) * dK1_14;
              double tlm_rhs2_15 = J_15_9 * (dC_9 + dK1_9) + J_15_10 * (dC_10 + dK1_10) + J_15_11 * (dC_11 + dK1_11) + J_15_13 * (dC_13 + dK1_13) + J_15_14 * (dC_14 + dK1_14) + J_15_15 * (dC_15 + dK1_15) + J_15_16 * (dC_16 + dK1_16) + (-1.0156171083877703 / h) * dK1_15;
              double tlm_rhs2_16 = J_16_13 * (dC_13 + dK1_13) + J_16_14 * (dC_14 + dK1_14) + J_16_15 * (dC_15 + dK1_15) + J_16_16 * (dC_16 + dK1_16) + (-1.0156171083877703 / h) * dK1_16;
              // Solve W * dK2 = tlm_rhs2
              // Block 0: dK2 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double tlm_y2_0 = tlm_rhs2_0;
              // Block 1: dK2 forward sub [M]
              double tlm_y2_1 = tlm_rhs2_1;
              double tlm_y2_2 = tlm_rhs2_2;
              double tlm_y2_3 = tlm_rhs2_3;
              double tlm_y2_4 = tlm_rhs2_4;
              double tlm_y2_5 = tlm_rhs2_5;
              // Block 2: dK2 forward sub [HO2]
              double tlm_y2_6 = tlm_rhs2_6;
              // Block 3: dK2 forward sub [H2O2, HNO3]
              double tlm_y2_7 = tlm_rhs2_7 - L_7_6 * tlm_y2_6;
              // Block 4: dK2 forward sub [OH]
              double tlm_y2_8 = tlm_rhs2_8 - L_8_4 * tlm_y2_4 - L_8_7 * tlm_y2_7;
              double tlm_y2_9 = tlm_rhs2_9 - L_9_5 * tlm_y2_5 - L_9_6 * tlm_y2_6;
              double tlm_y2_10 = tlm_rhs2_10 - L_10_8 * tlm_y2_8;
              // Block 5: dK2 forward sub [NO]
              double tlm_y2_11 = tlm_rhs2_11 - L_11_4 * tlm_y2_4 - L_11_9 * tlm_y2_9;
              // Block 6: dK2 forward sub [O3P]
              double tlm_y2_12 = tlm_rhs2_12 - L_12_5 * tlm_y2_5 - L_12_6 * tlm_y2_6 - L_12_9 * tlm_y2_9 - L_12_11 * tlm_y2_11;
              double tlm_y2_13 = tlm_rhs2_13;
              // Block 7: dK2 forward sub [NO3]
              double tlm_y2_14 = tlm_rhs2_14 - L_14_8 * tlm_y2_8 - L_14_9 * tlm_y2_9 - L_14_10 * tlm_y2_10 - L_14_11 * tlm_y2_11 - L_14_12 * tlm_y2_12 - L_14_13 * tlm_y2_13;
              // Block 8: dK2 forward sub [NO2]
              double tlm_y2_15 = tlm_rhs2_15 - L_15_9 * tlm_y2_9 - L_15_10 * tlm_y2_10 - L_15_11 * tlm_y2_11 - L_15_12 * tlm_y2_12 - L_15_13 * tlm_y2_13 - L_15_14 * tlm_y2_14;
              // Block 9: dK2 forward sub [N2O5]
              double tlm_y2_16 = tlm_rhs2_16 - L_16_13 * tlm_y2_13 - L_16_14 * tlm_y2_14 - L_16_15 * tlm_y2_15;
              // Block 9: dK2 backward sub [N2O5]
              double dK2_16 = tlm_y2_16 / U_16_16;
              // Block 8: dK2 backward sub [NO2]
              double dK2_15 = (tlm_y2_15 - U_15_16 * dK2_16) / U_15_15;
              // Block 7: dK2 backward sub [NO3]
              double dK2_14 = (tlm_y2_14 - U_14_15 * dK2_15 - U_14_16 * dK2_16) / U_14_14;
              // Block 0: dK2 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double dK2_13 = tlm_y2_13 / U_13_13;
              // Block 6: dK2 backward sub [O3P]
              double dK2_12 = (tlm_y2_12 - U_12_13 * dK2_13 - U_12_14 * dK2_14 - U_12_15 * dK2_15) / U_12_12;
              // Block 5: dK2 backward sub [NO]
              double dK2_11 = (tlm_y2_11 - U_11_12 * dK2_12 - U_11_13 * dK2_13 - U_11_14 * dK2_14 - U_11_15 * dK2_15) / U_11_11;
              // Block 3: dK2 backward sub [H2O2, HNO3]
              double dK2_10 = (tlm_y2_10 - U_10_13 * dK2_13) / U_10_10;
              double dK2_9 = (tlm_y2_9 - U_9_11 * dK2_11 - U_9_12 * dK2_12 - U_9_13 * dK2_13 - U_9_15 * dK2_15) / U_9_9;
              // Block 4: dK2 backward sub [OH]
              double dK2_8 = (tlm_y2_8 - U_8_10 * dK2_10 - U_8_13 * dK2_13) / U_8_8;
              double dK2_7 = (tlm_y2_7 - U_7_13 * dK2_13) / U_7_7;
              // Block 2: dK2 backward sub [HO2]
              double dK2_6 = (tlm_y2_6 - U_6_13 * dK2_13) / U_6_6;
              double dK2_5 = (tlm_y2_5 - U_5_6 * dK2_6 - U_5_9 * dK2_9 - U_5_12 * dK2_12 - U_5_13 * dK2_13) / U_5_5;
              double dK2_4 = tlm_y2_4 / U_4_4;
              double dK2_3 = tlm_y2_3 / U_3_3;
              double dK2_2 = tlm_y2_2 / U_2_2;
              // Block 1: dK2 backward sub [M]
              double dK2_1 = tlm_y2_1 / U_1_1;
              double dK2_0 = tlm_y2_0 / U_0_0;

              // --- TLM Stage 3 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs3_0 = (4.0759956452537702 / h) * dK1_0 + (9.20767942983308 / h) * dK2_0;
              double tlm_rhs3_1 = (4.0759956452537702 / h) * dK1_1 + (9.20767942983308 / h) * dK2_1;
              double tlm_rhs3_2 = (4.0759956452537702 / h) * dK1_2 + (9.20767942983308 / h) * dK2_2;
              double tlm_rhs3_3 = (4.0759956452537702 / h) * dK1_3 + (9.20767942983308 / h) * dK2_3;
              double tlm_rhs3_4 = J_4_4 * (dC_4 + dK1_4) + (4.0759956452537702 / h) * dK1_4 + (9.20767942983308 / h) * dK2_4;
              double tlm_rhs3_5 = J_5_5 * (dC_5 + dK1_5) + J_5_6 * (dC_6 + dK1_6) + J_5_9 * (dC_9 + dK1_9) + J_5_12 * (dC_12 + dK1_12) + J_5_13 * (dC_13 + dK1_13) + (4.0759956452537702 / h) * dK1_5 + (9.20767942983308 / h) * dK2_5;
              double tlm_rhs3_6 = J_6_6 * (dC_6 + dK1_6) + J_6_13 * (dC_13 + dK1_13) + (4.0759956452537702 / h) * dK1_6 + (9.20767942983308 / h) * dK2_6;
              double tlm_rhs3_7 = J_7_6 * (dC_6 + dK1_6) + J_7_7 * (dC_7 + dK1_7) + J_7_13 * (dC_13 + dK1_13) + (4.0759956452537702 / h) * dK1_7 + (9.20767942983308 / h) * dK2_7;
              double tlm_rhs3_8 = J_8_4 * (dC_4 + dK1_4) + J_8_7 * (dC_7 + dK1_7) + J_8_8 * (dC_8 + dK1_8) + J_8_10 * (dC_10 + dK1_10) + J_8_13 * (dC_13 + dK1_13) + (4.0759956452537702 / h) * dK1_8 + (9.20767942983308 / h) * dK2_8;
              double tlm_rhs3_9 = J_9_5 * (dC_5 + dK1_5) + J_9_9 * (dC_9 + dK1_9) + J_9_11 * (dC_11 + dK1_11) + J_9_12 * (dC_12 + dK1_12) + J_9_13 * (dC_13 + dK1_13) + J_9_15 * (dC_15 + dK1_15) + (4.0759956452537702 / h) * dK1_9 + (9.20767942983308 / h) * dK2_9;
              double tlm_rhs3_10 = J_10_8 * (dC_8 + dK1_8) + J_10_10 * (dC_10 + dK1_10) + J_10_13 * (dC_13 + dK1_13) + (4.0759956452537702 / h) * dK1_10 + (9.20767942983308 / h) * dK2_10;
              double tlm_rhs3_11 = J_11_4 * (dC_4 + dK1_4) + J_11_9 * (dC_9 + dK1_9) + J_11_11 * (dC_11 + dK1_11) + J_11_14 * (dC_14 + dK1_14) + J_11_15 * (dC_15 + dK1_15) + (4.0759956452537702 / h) * dK1_11 + (9.20767942983308 / h) * dK2_11;
              double tlm_rhs3_12 = J_12_5 * (dC_5 + dK1_5) + J_12_9 * (dC_9 + dK1_9) + J_12_12 * (dC_12 + dK1_12) + J_12_13 * (dC_13 + dK1_13) + J_12_15 * (dC_15 + dK1_15) + (4.0759956452537702 / h) * dK1_12 + (9.20767942983308 / h) * dK2_12;
              double tlm_rhs3_13 = (4.0759956452537702 / h) * dK1_13 + (9.20767942983308 / h) * dK2_13;
              double tlm_rhs3_14 = J_14_8 * (dC_8 + dK1_8) + J_14_9 * (dC_9 + dK1_9) + J_14_10 * (dC_10 + dK1_10) + J_14_11 * (dC_11 + dK1_11) + J_14_13 * (dC_13 + dK1_13) + J_14_14 * (dC_14 + dK1_14) + J_14_15 * (dC_15 + dK1_15) + J_14_16 * (dC_16 + dK1_16) + (4.0759956452537702 / h) * dK1_14 + (9.20767942983308 / h) * dK2_14;
              double tlm_rhs3_15 = J_15_9 * (dC_9 + dK1_9) + J_15_10 * (dC_10 + dK1_10) + J_15_11 * (dC_11 + dK1_11) + J_15_13 * (dC_13 + dK1_13) + J_15_14 * (dC_14 + dK1_14) + J_15_15 * (dC_15 + dK1_15) + J_15_16 * (dC_16 + dK1_16) + (4.0759956452537702 / h) * dK1_15 + (9.20767942983308 / h) * dK2_15;
              double tlm_rhs3_16 = J_16_13 * (dC_13 + dK1_13) + J_16_14 * (dC_14 + dK1_14) + J_16_15 * (dC_15 + dK1_15) + J_16_16 * (dC_16 + dK1_16) + (4.0759956452537702 / h) * dK1_16 + (9.20767942983308 / h) * dK2_16;
              // Solve W * dK3 = tlm_rhs3
              // Block 0: dK3 forward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double tlm_y3_0 = tlm_rhs3_0;
              // Block 1: dK3 forward sub [M]
              double tlm_y3_1 = tlm_rhs3_1;
              double tlm_y3_2 = tlm_rhs3_2;
              double tlm_y3_3 = tlm_rhs3_3;
              double tlm_y3_4 = tlm_rhs3_4;
              double tlm_y3_5 = tlm_rhs3_5;
              // Block 2: dK3 forward sub [HO2]
              double tlm_y3_6 = tlm_rhs3_6;
              // Block 3: dK3 forward sub [H2O2, HNO3]
              double tlm_y3_7 = tlm_rhs3_7 - L_7_6 * tlm_y3_6;
              // Block 4: dK3 forward sub [OH]
              double tlm_y3_8 = tlm_rhs3_8 - L_8_4 * tlm_y3_4 - L_8_7 * tlm_y3_7;
              double tlm_y3_9 = tlm_rhs3_9 - L_9_5 * tlm_y3_5 - L_9_6 * tlm_y3_6;
              double tlm_y3_10 = tlm_rhs3_10 - L_10_8 * tlm_y3_8;
              // Block 5: dK3 forward sub [NO]
              double tlm_y3_11 = tlm_rhs3_11 - L_11_4 * tlm_y3_4 - L_11_9 * tlm_y3_9;
              // Block 6: dK3 forward sub [O3P]
              double tlm_y3_12 = tlm_rhs3_12 - L_12_5 * tlm_y3_5 - L_12_6 * tlm_y3_6 - L_12_9 * tlm_y3_9 - L_12_11 * tlm_y3_11;
              double tlm_y3_13 = tlm_rhs3_13;
              // Block 7: dK3 forward sub [NO3]
              double tlm_y3_14 = tlm_rhs3_14 - L_14_8 * tlm_y3_8 - L_14_9 * tlm_y3_9 - L_14_10 * tlm_y3_10 - L_14_11 * tlm_y3_11 - L_14_12 * tlm_y3_12 - L_14_13 * tlm_y3_13;
              // Block 8: dK3 forward sub [NO2]
              double tlm_y3_15 = tlm_rhs3_15 - L_15_9 * tlm_y3_9 - L_15_10 * tlm_y3_10 - L_15_11 * tlm_y3_11 - L_15_12 * tlm_y3_12 - L_15_13 * tlm_y3_13 - L_15_14 * tlm_y3_14;
              // Block 9: dK3 forward sub [N2O5]
              double tlm_y3_16 = tlm_rhs3_16 - L_16_13 * tlm_y3_13 - L_16_14 * tlm_y3_14 - L_16_15 * tlm_y3_15;
              // Block 9: dK3 backward sub [N2O5]
              double dK3_16 = tlm_y3_16 / U_16_16;
              // Block 8: dK3 backward sub [NO2]
              double dK3_15 = (tlm_y3_15 - U_15_16 * dK3_16) / U_15_15;
              // Block 7: dK3 backward sub [NO3]
              double dK3_14 = (tlm_y3_14 - U_14_15 * dK3_15 - U_14_16 * dK3_16) / U_14_14;
              // Block 0: dK3 backward sub [N2, H2O, HNO4, HONO, O2, O3, AIR]
              double dK3_13 = tlm_y3_13 / U_13_13;
              // Block 6: dK3 backward sub [O3P]
              double dK3_12 = (tlm_y3_12 - U_12_13 * dK3_13 - U_12_14 * dK3_14 - U_12_15 * dK3_15) / U_12_12;
              // Block 5: dK3 backward sub [NO]
              double dK3_11 = (tlm_y3_11 - U_11_12 * dK3_12 - U_11_13 * dK3_13 - U_11_14 * dK3_14 - U_11_15 * dK3_15) / U_11_11;
              // Block 3: dK3 backward sub [H2O2, HNO3]
              double dK3_10 = (tlm_y3_10 - U_10_13 * dK3_13) / U_10_10;
              double dK3_9 = (tlm_y3_9 - U_9_11 * dK3_11 - U_9_12 * dK3_12 - U_9_13 * dK3_13 - U_9_15 * dK3_15) / U_9_9;
              // Block 4: dK3 backward sub [OH]
              double dK3_8 = (tlm_y3_8 - U_8_10 * dK3_10 - U_8_13 * dK3_13) / U_8_8;
              double dK3_7 = (tlm_y3_7 - U_7_13 * dK3_13) / U_7_7;
              // Block 2: dK3 backward sub [HO2]
              double dK3_6 = (tlm_y3_6 - U_6_13 * dK3_13) / U_6_6;
              double dK3_5 = (tlm_y3_5 - U_5_6 * dK3_6 - U_5_9 * dK3_9 - U_5_12 * dK3_12 - U_5_13 * dK3_13) / U_5_5;
              double dK3_4 = tlm_y3_4 / U_4_4;
              double dK3_3 = tlm_y3_3 / U_3_3;
              double dK3_2 = tlm_y3_2 / U_2_2;
              // Block 1: dK3 backward sub [M]
              double dK3_1 = tlm_y3_1 / U_1_1;
              double dK3_0 = tlm_y3_0 / U_0_0;

              // --- δC update: δC += Σ M_i * δK_i ---
              dC_0 += dK1_0 + 6.1697947043828245 * dK2_0 + -0.42772256543218573 * dK3_0;
              dC_1 += dK1_1 + 6.1697947043828245 * dK2_1 + -0.42772256543218573 * dK3_1;
              dC_2 += dK1_2 + 6.1697947043828245 * dK2_2 + -0.42772256543218573 * dK3_2;
              dC_3 += dK1_3 + 6.1697947043828245 * dK2_3 + -0.42772256543218573 * dK3_3;
              dC_4 += dK1_4 + 6.1697947043828245 * dK2_4 + -0.42772256543218573 * dK3_4;
              dC_5 += dK1_5 + 6.1697947043828245 * dK2_5 + -0.42772256543218573 * dK3_5;
              dC_6 += dK1_6 + 6.1697947043828245 * dK2_6 + -0.42772256543218573 * dK3_6;
              dC_7 += dK1_7 + 6.1697947043828245 * dK2_7 + -0.42772256543218573 * dK3_7;
              dC_8 += dK1_8 + 6.1697947043828245 * dK2_8 + -0.42772256543218573 * dK3_8;
              dC_9 += dK1_9 + 6.1697947043828245 * dK2_9 + -0.42772256543218573 * dK3_9;
              dC_10 += dK1_10 + 6.1697947043828245 * dK2_10 + -0.42772256543218573 * dK3_10;
              dC_11 += dK1_11 + 6.1697947043828245 * dK2_11 + -0.42772256543218573 * dK3_11;
              dC_12 += dK1_12 + 6.1697947043828245 * dK2_12 + -0.42772256543218573 * dK3_12;
              dC_13 += dK1_13 + 6.1697947043828245 * dK2_13 + -0.42772256543218573 * dK3_13;
              dC_14 += dK1_14 + 6.1697947043828245 * dK2_14 + -0.42772256543218573 * dK3_14;
              dC_15 += dK1_15 + 6.1697947043828245 * dK2_15 + -0.42772256543218573 * dK3_15;
              dC_16 += dK1_16 + 6.1697947043828245 * dK2_16 + -0.42772256543218573 * dK3_16;
          } // end for (step)

          // Write back final δC to output view
          delta_C(0) = dC_0;
          delta_C(1) = dC_1;
          delta_C(2) = dC_2;
          delta_C(3) = dC_3;
          delta_C(4) = dC_4;
          delta_C(5) = dC_5;
          delta_C(6) = dC_6;
          delta_C(7) = dC_7;
          delta_C(8) = dC_8;
          delta_C(9) = dC_9;
          delta_C(10) = dC_10;
          delta_C(11) = dC_11;
          delta_C(12) = dC_12;
          delta_C(13) = dC_13;
          delta_C(14) = dC_14;
          delta_C(15) = dC_15;
          delta_C(16) = dC_16;
      }
#endif
  };
}
