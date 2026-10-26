#pragma once
#include <Kokkos_Core.hpp>
// CPU libraries can own the heavy helper definitions. Device builds keep the
// complete device-callable implementation visible without relocatable device code.
#if defined(MKPP_SEPARATE_HOST_KERNELS_CARBON)
#include "carbon/host_declarations.hpp"
#else
#include "carbon/accessor.hpp"
#include "carbon/rates.hpp"
#include "carbon/jacobian.hpp"
#include "carbon/supernodal_factorize.hpp"
#include "carbon/supernodal_solve.hpp"
#include "carbon/solve.hpp"
#endif
// Generated solver for carbon
// SZA Workload Sorted: true
namespace mkpp::generated::carbon {
  // Pure Kokkos abstractions (no raw pragmas allowed)

  /**
   * @brief Strongly typed enum for species indexing into state vectors.
   */
  enum Species : int {
      CH4 = 0,
      CO = 1,
      PCOfromCH4 = 2,
      PCOfromNMVOC = 3,
      LCH4byOH = 4,
      LCH4byCl = 5,
      LCObyOH = 6,
      FixedOH = 7,
      FixedCl = 8,
      DummyCH4 = 9,
      DummyNMVOC = 10
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
      static constexpr int NUM_SPECIES = 11;
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
      static constexpr int NUM_RUNTIME_PARAMETERS = 0;
      static constexpr bool HAS_CLOUD_GATED = false;
#ifdef MKPP_ENABLE_ADJOINT
      // Expose the checkpoint buffer through the kernel type so generic
      // device-path callers can name it without the mechanism namespace.
      using CheckpointBuffer = ::mkpp::generated::carbon::CheckpointBuffer;
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
          // J^T(CH4, CH4): d(d[CH4]/dt) / d[CH4]
          J_adj_block(0, 0) = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
          // J^T(CH4, LCH4byOH): d(d[LCH4byOH]/dt) / d[CH4]
          J_adj_block(0, 4) = 2.45e-12*Kokkos::exp(-1775.0/Temp);
          // J^T(CH4, LCH4byCl): d(d[LCH4byCl]/dt) / d[CH4]
          J_adj_block(0, 5) = 9.6e-12*Kokkos::exp(-1360.0/Temp);
          // J^T(CH4, FixedOH): d(d[FixedOH]/dt) / d[CH4]
          J_adj_block(0, 7) = -2.45e-12*Kokkos::exp(-1775.0/Temp);
          // J^T(CH4, FixedCl): d(d[FixedCl]/dt) / d[CH4]
          J_adj_block(0, 8) = -9.6e-12*Kokkos::exp(-1360.0/Temp);
          // J^T(CO, CO): d(d[CO]/dt) / d[CO]
          J_adj_block(1, 1) = -7.3679649e-14;
          // J^T(CO, LCObyOH): d(d[LCObyOH]/dt) / d[CO]
          J_adj_block(1, 6) = 7.3679649e-14;
          // J^T(CO, FixedOH): d(d[FixedOH]/dt) / d[CO]
          J_adj_block(1, 7) = -7.3679649e-14;
          // J^T(FixedOH, CH4): d(d[CH4]/dt) / d[FixedOH]
          J_adj_block(7, 0) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp);
          // J^T(FixedOH, CO): d(d[CO]/dt) / d[FixedOH]
          J_adj_block(7, 1) = -7.3679649e-14*state(1);
          // J^T(FixedOH, LCH4byOH): d(d[LCH4byOH]/dt) / d[FixedOH]
          J_adj_block(7, 4) = 2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp);
          // J^T(FixedOH, LCObyOH): d(d[LCObyOH]/dt) / d[FixedOH]
          J_adj_block(7, 6) = 7.3679649e-14*state(1);
          // J^T(FixedOH, FixedOH): d(d[FixedOH]/dt) / d[FixedOH]
          J_adj_block(7, 7) = -2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*state(1);
          // J^T(FixedCl, CH4): d(d[CH4]/dt) / d[FixedCl]
          J_adj_block(8, 0) = -9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
          // J^T(FixedCl, LCH4byCl): d(d[LCH4byCl]/dt) / d[FixedCl]
          J_adj_block(8, 5) = 9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
          // J^T(FixedCl, FixedCl): d(d[FixedCl]/dt) / d[FixedCl]
          J_adj_block(8, 8) = -9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp);
          // J^T(DummyCH4, CO): d(d[CO]/dt) / d[DummyCH4]
          J_adj_block(9, 1) = 4.2566446e-15;
          // J^T(DummyCH4, PCOfromCH4): d(d[PCOfromCH4]/dt) / d[DummyCH4]
          J_adj_block(9, 2) = 4.2566446e-15;
          // J^T(DummyCH4, DummyCH4): d(d[DummyCH4]/dt) / d[DummyCH4]
          J_adj_block(9, 9) = -4.2566446e-15;
          // J^T(DummyNMVOC, CO): d(d[CO]/dt) / d[DummyNMVOC]
          J_adj_block(10, 1) = 38199.012;
          // J^T(DummyNMVOC, PCOfromNMVOC): d(d[PCOfromNMVOC]/dt) / d[DummyNMVOC]
          J_adj_block(10, 3) = 38199.012;
          // J^T(DummyNMVOC, DummyNMVOC): d(d[DummyNMVOC]/dt) / d[DummyNMVOC]
          J_adj_block(10, 10) = -38199.012;
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
          dF_block(0) += (-2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp)) * delta_C(0);
          dF_block(0) += (-2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp)) * delta_C(7);
          dF_block(0) += (-9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp)) * delta_C(8);
          dF_block(1) = 0.0;
          dF_block(1) += (-7.3679649e-14) * delta_C(1);
          dF_block(1) += (-7.3679649e-14*state(1)) * delta_C(7);
          dF_block(1) += (4.2566446e-15) * delta_C(9);
          dF_block(1) += (38199.012) * delta_C(10);
          dF_block(2) = 0.0;
          dF_block(2) += (4.2566446e-15) * delta_C(9);
          dF_block(3) = 0.0;
          dF_block(3) += (38199.012) * delta_C(10);
          dF_block(4) = 0.0;
          dF_block(4) += (2.45e-12*Kokkos::exp(-1775.0/Temp)) * delta_C(0);
          dF_block(4) += (2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp)) * delta_C(7);
          dF_block(5) = 0.0;
          dF_block(5) += (9.6e-12*Kokkos::exp(-1360.0/Temp)) * delta_C(0);
          dF_block(5) += (9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp)) * delta_C(8);
          dF_block(6) = 0.0;
          dF_block(6) += (7.3679649e-14) * delta_C(1);
          dF_block(6) += (7.3679649e-14*state(1)) * delta_C(7);
          dF_block(7) = 0.0;
          dF_block(7) += (-2.45e-12*Kokkos::exp(-1775.0/Temp)) * delta_C(0);
          dF_block(7) += (-7.3679649e-14) * delta_C(1);
          dF_block(7) += (-2.45e-12*state(0)*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*state(1)) * delta_C(7);
          dF_block(8) = 0.0;
          dF_block(8) += (-9.6e-12*Kokkos::exp(-1360.0/Temp)) * delta_C(0);
          dF_block(8) += (-9.6e-12*state(0)*Kokkos::exp(-1360.0/Temp)) * delta_C(8);
          dF_block(9) = 0.0;
          dF_block(9) += (-4.2566446e-15) * delta_C(9);
          dF_block(10) = 0.0;
          dF_block(10) += (-38199.012) * delta_C(10);
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
      }

      static constexpr int NUM_SPECIES = 11;
      static constexpr double atol[NUM_SPECIES] = { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
      static constexpr double rtol[NUM_SPECIES] = { 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06 };

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
          const int NUM_SPECIES = 11;
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
          const double S_0 = state(Species::PCOfromNMVOC);  // [PCOfromNMVOC]
          const double S_1 = state(Species::PCOfromCH4);  // [PCOfromCH4]
          const double S_2 = state(Species::DummyNMVOC);  // [DummyNMVOC]
          const double S_3 = state(Species::DummyCH4);  // [DummyCH4]
          const double S_4 = state(Species::CO);  // [CO]
          const double S_5 = state(Species::LCObyOH);  // [LCObyOH]
          const double S_6 = state(Species::FixedOH);  // [FixedOH]
          const double S_7 = state(Species::LCH4byOH);  // [LCH4byOH]
          const double S_8 = state(Species::CH4);  // [CH4]
          const double S_9 = state(Species::LCH4byCl);  // [LCH4byCl]
          const double S_10 = state(Species::FixedCl);  // [FixedCl]
          // The expression-dense Jacobian is evaluated by bounded compiled
          // units.  The solver still uses the same symbolic sparse LU plan.
#ifndef MKPP_USE_UNROLLED_REFERENCE
          double J_values[22] = {};
          auto jacobian_accessor = detail::StrideView<double, 2>::compact(J_values);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, temp, rh, air_density);
          double W_values[27];
          W_values[0] = inv_g_dt;
          W_values[1] = 0.0 - J_values[8];
          W_values[2] = inv_g_dt;
          W_values[3] = 0.0 - J_values[7];
          W_values[4] = inv_g_dt - J_values[21];
          W_values[5] = inv_g_dt - J_values[20];
          W_values[6] = 0.0 - J_values[6];
          W_values[7] = 0.0 - J_values[5];
          W_values[8] = inv_g_dt - J_values[3];
          W_values[9] = 0.0 - J_values[4];
          W_values[10] = 0.0 - J_values[13];
          W_values[11] = inv_g_dt;
          W_values[12] = 0.0 - J_values[14];
          W_values[13] = 0.0 - J_values[16];
          W_values[14] = inv_g_dt - J_values[17];
          W_values[15] = 0.0 - J_values[15];
          W_values[16] = 0.0 - J_values[10];
          W_values[17] = inv_g_dt;
          W_values[18] = 0.0 - J_values[9];
          W_values[19] = 0.0 - J_values[1];
          W_values[20] = inv_g_dt - J_values[0];
          W_values[21] = 0.0 - J_values[2];
          W_values[22] = 0.0 - J_values[11];
          W_values[23] = inv_g_dt;
          W_values[24] = 0.0 - J_values[12];
          W_values[25] = 0.0 - J_values[18];
          W_values[26] = inv_g_dt - J_values[19];
          double LU_values[27];
          detail::factorize_plan(W_values, LU_values);
#else
          double J_values[NUM_SPECIES * NUM_SPECIES] = {};
          auto jacobian_accessor = detail::make_stride_view<double, 2>(J_values, NUM_SPECIES, 1);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, temp, rh, air_density);
          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_0_2 = J_values[3 * NUM_SPECIES + 10];
          double J_1_3 = J_values[2 * NUM_SPECIES + 9];
          double J_2_2 = J_values[10 * NUM_SPECIES + 10];
          double J_3_3 = J_values[9 * NUM_SPECIES + 9];
          double J_4_2 = J_values[1 * NUM_SPECIES + 10];
          double J_4_3 = J_values[1 * NUM_SPECIES + 9];
          double J_4_4 = J_values[1 * NUM_SPECIES + 1];
          double J_4_6 = J_values[1 * NUM_SPECIES + 7];
          double J_5_4 = J_values[6 * NUM_SPECIES + 1];
          double J_5_6 = J_values[6 * NUM_SPECIES + 7];
          double J_6_4 = J_values[7 * NUM_SPECIES + 1];
          double J_6_6 = J_values[7 * NUM_SPECIES + 7];
          double J_6_8 = J_values[7 * NUM_SPECIES + 0];
          double J_7_6 = J_values[4 * NUM_SPECIES + 7];
          double J_7_8 = J_values[4 * NUM_SPECIES + 0];
          double J_8_6 = J_values[0 * NUM_SPECIES + 7];
          double J_8_8 = J_values[0 * NUM_SPECIES + 0];
          double J_8_10 = J_values[0 * NUM_SPECIES + 8];
          double J_9_8 = J_values[5 * NUM_SPECIES + 0];
          double J_9_10 = J_values[5 * NUM_SPECIES + 8];
          double J_10_8 = J_values[8 * NUM_SPECIES + 0];
          double J_10_10 = J_values[8 * NUM_SPECIES + 8];
          double W_values[NUM_SPECIES * NUM_SPECIES] = {};
          W_values[0 * NUM_SPECIES + 0] = inv_g_dt;
          W_values[0 * NUM_SPECIES + 2] = -J_0_2;
          W_values[1 * NUM_SPECIES + 1] = inv_g_dt;
          W_values[1 * NUM_SPECIES + 3] = -J_1_3;
          W_values[2 * NUM_SPECIES + 2] = inv_g_dt - J_2_2;
          W_values[3 * NUM_SPECIES + 3] = inv_g_dt - J_3_3;
          W_values[4 * NUM_SPECIES + 2] = -J_4_2;
          W_values[4 * NUM_SPECIES + 3] = -J_4_3;
          W_values[4 * NUM_SPECIES + 4] = inv_g_dt - J_4_4;
          W_values[4 * NUM_SPECIES + 6] = -J_4_6;
          W_values[5 * NUM_SPECIES + 4] = -J_5_4;
          W_values[5 * NUM_SPECIES + 5] = inv_g_dt;
          W_values[5 * NUM_SPECIES + 6] = -J_5_6;
          W_values[6 * NUM_SPECIES + 4] = -J_6_4;
          W_values[6 * NUM_SPECIES + 6] = inv_g_dt - J_6_6;
          W_values[6 * NUM_SPECIES + 8] = -J_6_8;
          W_values[7 * NUM_SPECIES + 6] = -J_7_6;
          W_values[7 * NUM_SPECIES + 7] = inv_g_dt;
          W_values[7 * NUM_SPECIES + 8] = -J_7_8;
          W_values[8 * NUM_SPECIES + 6] = -J_8_6;
          W_values[8 * NUM_SPECIES + 8] = inv_g_dt - J_8_8;
          W_values[8 * NUM_SPECIES + 10] = -J_8_10;
          W_values[9 * NUM_SPECIES + 8] = -J_9_8;
          W_values[9 * NUM_SPECIES + 9] = inv_g_dt;
          W_values[9 * NUM_SPECIES + 10] = -J_9_10;
          W_values[10 * NUM_SPECIES + 8] = -J_10_8;
          W_values[10 * NUM_SPECIES + 10] = inv_g_dt - J_10_10;
          double LU_values[NUM_SPECIES * NUM_SPECIES] = {};
          detail::factorize_lu_chunk_0(W_values, LU_values);
#endif

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F_values_1[NUM_SPECIES];
          auto rate_accessor_1 = detail::make_stride_view<double, 1>(F_values_1, 1);
          detail::compute_rates_chunk_0(state_accessor, rate_accessor_1, jvals, temp, rh, air_density);
          double F1_0 = F_values_1[3];
          double F1_1 = F_values_1[2];
          double F1_2 = F_values_1[10];
          double F1_3 = F_values_1[9];
          double F1_4 = F_values_1[1];
          double F1_5 = F_values_1[6];
          double F1_6 = F_values_1[7];
          double F1_7 = F_values_1[4];
          double F1_8 = F_values_1[0];
          double F1_9 = F_values_1[5];
          double F1_10 = F_values_1[8];
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
          // Rate evaluation F2 at Y2
          double stage_state_2[NUM_SPECIES];
          stage_state_2[3] = Y2_0;
          stage_state_2[2] = Y2_1;
          stage_state_2[10] = Y2_2;
          stage_state_2[9] = Y2_3;
          stage_state_2[1] = Y2_4;
          stage_state_2[6] = Y2_5;
          stage_state_2[7] = Y2_6;
          stage_state_2[4] = Y2_7;
          stage_state_2[0] = Y2_8;
          stage_state_2[5] = Y2_9;
          stage_state_2[8] = Y2_10;
          double F_values_2[NUM_SPECIES];
          auto rate_accessor_2 = detail::make_stride_view<double, 1>(F_values_2, 1);
          auto stage_state_accessor_2 = detail::make_stride_view<double, 1>(stage_state_2, 1);
          detail::compute_rates_chunk_0(stage_state_accessor_2, rate_accessor_2, jvals, temp, rh, air_density);
          double F2_0 = F_values_2[3];
          double F2_1 = F_values_2[2];
          double F2_2 = F_values_2[10];
          double F2_3 = F_values_2[9];
          double F2_4 = F_values_2[1];
          double F2_5 = F_values_2[6];
          double F2_6 = F_values_2[7];
          double F2_7 = F_values_2[4];
          double F2_8 = F_values_2[0];
          double F2_9 = F_values_2[5];
          double F2_10 = F_values_2[8];
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

          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 11);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(10) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(9) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(1) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(6) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(7) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              state(4) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              state(0) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              state(5) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              state(8) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
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
          const int NUM_SPECIES = 11;
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

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // 0. Hoist state values into scalar registers
          // NOTE: State access uses permuted species ordering
          const double S_0 = state(3);
          const double S_1 = state(2);
          const double S_2 = state(10);
          const double S_3 = state(9);
          const double S_4 = state(1);
          const double S_5 = state(6);
          const double S_6 = state(7);
          const double S_7 = state(4);
          const double S_8 = state(0);
          const double S_9 = state(5);
          const double S_10 = state(8);

          // 1. Stage 1 Rates (F1)
          double F1_0 = 38199.012;
          double F1_1 = 4.2566446e-15;
          double F1_2 = -38199.012;
          double F1_3 = -4.2566446e-15;
          double F1_4 = 38199.012 - 7.3679649e-14*S_4;
          double F1_5 = 7.3679649e-14*S_4;
          double F1_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
          double F1_7 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double F1_8 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double F1_9 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double F1_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);

          // 2. Evaluate importance and update active set
          active[0] = (Kokkos::fabs(F1_0) / (atol[0] + rtol[0] * Kokkos::fabs(state(3))) >= importance_threshold);
          active[1] = (Kokkos::fabs(F1_1) / (atol[1] + rtol[1] * Kokkos::fabs(state(2))) >= importance_threshold);
          active[2] = (Kokkos::fabs(F1_2) / (atol[2] + rtol[2] * Kokkos::fabs(state(10))) >= importance_threshold);
          active[3] = (Kokkos::fabs(F1_3) / (atol[3] + rtol[3] * Kokkos::fabs(state(9))) >= importance_threshold);
          active[4] = (Kokkos::fabs(F1_4) / (atol[4] + rtol[4] * Kokkos::fabs(state(1))) >= importance_threshold);
          active[5] = (Kokkos::fabs(F1_5) / (atol[5] + rtol[5] * Kokkos::fabs(state(6))) >= importance_threshold);
          active[6] = (Kokkos::fabs(F1_6) / (atol[6] + rtol[6] * Kokkos::fabs(state(7))) >= importance_threshold);
          active[7] = (Kokkos::fabs(F1_7) / (atol[7] + rtol[7] * Kokkos::fabs(state(4))) >= importance_threshold);
          active[8] = (Kokkos::fabs(F1_8) / (atol[8] + rtol[8] * Kokkos::fabs(state(0))) >= importance_threshold);
          active[9] = (Kokkos::fabs(F1_9) / (atol[9] + rtol[9] * Kokkos::fabs(state(5))) >= importance_threshold);
          active[10] = (Kokkos::fabs(F1_10) / (atol[10] + rtol[10] * Kokkos::fabs(state(8))) >= importance_threshold);

          // 3. Analytical Jacobian & Iteration Matrix W (identity for frozen species)
          double J_0_2 = 38199.012;
          double J_1_3 = 4.2566446e-15;
          double J_2_2 = -38199.012;
          double J_3_3 = -4.2566446e-15;
          double J_4_2 = 38199.012;
          double J_4_3 = 4.2566446e-15;
          double J_4_4 = -7.3679649e-14;
          double J_4_6 = -7.3679649e-14*S_4;
          double J_5_4 = 7.3679649e-14;
          double J_5_6 = 7.3679649e-14*S_4;
          double J_6_4 = -7.3679649e-14;
          double J_6_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
          double J_6_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp);
          double J_7_6 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double J_7_8 = 2.45e-12*Kokkos::exp(-1775.0/Temp);
          double J_8_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double J_8_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_8_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double J_9_8 = 9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_9_10 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double J_10_8 = -9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_10_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double W_0_0 = active[0] ? inv_g_dt : 1.0;
          double W_0_2 = (active[0] && active[2]) ? (-J_0_2) : 0.0;
          double W_1_1 = active[1] ? inv_g_dt : 1.0;
          double W_1_3 = (active[1] && active[3]) ? (-J_1_3) : 0.0;
          double W_2_2 = active[2] ? (inv_g_dt - J_2_2) : 1.0;
          double W_3_3 = active[3] ? (inv_g_dt - J_3_3) : 1.0;
          double W_4_2 = (active[4] && active[2]) ? (-J_4_2) : 0.0;
          double W_4_3 = (active[4] && active[3]) ? (-J_4_3) : 0.0;
          double W_4_4 = active[4] ? (inv_g_dt - J_4_4) : 1.0;
          double W_4_6 = (active[4] && active[6]) ? (-J_4_6) : 0.0;
          double W_5_4 = (active[5] && active[4]) ? (-J_5_4) : 0.0;
          double W_5_5 = active[5] ? inv_g_dt : 1.0;
          double W_5_6 = (active[5] && active[6]) ? (-J_5_6) : 0.0;
          double W_6_4 = (active[6] && active[4]) ? (-J_6_4) : 0.0;
          double W_6_6 = active[6] ? (inv_g_dt - J_6_6) : 1.0;
          double W_6_8 = (active[6] && active[8]) ? (-J_6_8) : 0.0;
          double W_7_6 = (active[7] && active[6]) ? (-J_7_6) : 0.0;
          double W_7_7 = active[7] ? inv_g_dt : 1.0;
          double W_7_8 = (active[7] && active[8]) ? (-J_7_8) : 0.0;
          double W_8_6 = (active[8] && active[6]) ? (-J_8_6) : 0.0;
          double W_8_8 = active[8] ? (inv_g_dt - J_8_8) : 1.0;
          double W_8_10 = (active[8] && active[10]) ? (-J_8_10) : 0.0;
          double W_9_8 = (active[9] && active[8]) ? (-J_9_8) : 0.0;
          double W_9_9 = active[9] ? inv_g_dt : 1.0;
          double W_9_10 = (active[9] && active[10]) ? (-J_9_10) : 0.0;
          double W_10_8 = (active[10] && active[8]) ? (-J_10_8) : 0.0;
          double W_10_10 = active[10] ? (inv_g_dt - J_10_10) : 1.0;

          // 4. Symbolic LU Factorization (conditional skip for frozen species)
          double U_0_0 = W_0_0;
          double U_0_2 = W_0_2;
          double U_1_1 = W_1_1;
          double U_1_3 = W_1_3;
          double U_2_2 = W_2_2;
          double L_4_2 = (W_4_2) / U_2_2;
          double U_3_3 = W_3_3;
          double L_4_3 = (W_4_3) / U_3_3;
          double U_4_4 = W_4_4;
          double U_4_6 = W_4_6;
          double L_5_4 = (W_5_4) / U_4_4;
          double L_6_4 = (W_6_4) / U_4_4;
          double U_5_5 = W_5_5;
          double U_5_6 = W_5_6 - L_5_4 * U_4_6;
          double U_6_6 = W_6_6 - L_6_4 * U_4_6;
          double U_6_8 = W_6_8;
          double L_7_6 = (W_7_6) / U_6_6;
          double L_8_6 = (W_8_6) / U_6_6;
          double U_7_7 = W_7_7;
          double U_7_8 = W_7_8 - L_7_6 * U_6_8;
          double U_8_8 = W_8_8 - L_8_6 * U_6_8;
          double U_8_10 = W_8_10;
          double L_9_8 = (W_9_8) / U_8_8;
          double L_10_8 = (W_10_8) / U_8_8;
          double U_9_9 = W_9_9;
          double U_9_10 = W_9_10 - L_9_8 * U_8_10;
          double U_10_10 = W_10_10 - L_10_8 * U_8_10;

          // --- Stage 1 ---
          // F1 already computed above (used for importance evaluation)
          // Block 0: K1 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y1_0 = active[0] ? (F1_0) : 0.0;
          double y1_1 = active[1] ? (F1_1) : 0.0;
          // Block 1: K1 forward sub [DummyNMVOC]
          double y1_2 = active[2] ? (F1_2) : 0.0;
          // Block 2: K1 forward sub [DummyCH4]
          double y1_3 = active[3] ? (F1_3) : 0.0;
          // Block 3: K1 forward sub [CO]
          double y1_4 = active[4] ? (F1_4 - L_4_2 * y1_2 - L_4_3 * y1_3) : 0.0;
          // Block 4: K1 forward sub [LCObyOH]
          double y1_5 = active[5] ? (F1_5 - L_5_4 * y1_4) : 0.0;
          // Block 5: K1 forward sub [FixedOH]
          double y1_6 = active[6] ? (F1_6 - L_6_4 * y1_4) : 0.0;
          double y1_7 = active[7] ? (F1_7 - L_7_6 * y1_6) : 0.0;
          double y1_8 = active[8] ? (F1_8 - L_8_6 * y1_6) : 0.0;
          // Block 6: K1 forward sub [LCH4byCl]
          double y1_9 = active[9] ? (F1_9 - L_9_8 * y1_8) : 0.0;
          // Block 7: K1 forward sub [FixedCl]
          double y1_10 = active[10] ? (F1_10 - L_10_8 * y1_8) : 0.0;

          // Block 7: K1 backward sub [FixedCl]
          double K1_10 = active[10] ? (y1_10 / U_10_10) : 0.0;
          // Block 6: K1 backward sub [LCH4byCl]
          double K1_9 = active[9] ? ((y1_9 - U_9_10 * K1_10) / U_9_9) : 0.0;
          // Block 0: K1 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K1_8 = active[8] ? ((y1_8 - U_8_10 * K1_10) / U_8_8) : 0.0;
          double K1_7 = active[7] ? ((y1_7 - U_7_8 * K1_8) / U_7_7) : 0.0;
          // Block 5: K1 backward sub [FixedOH]
          double K1_6 = active[6] ? ((y1_6 - U_6_8 * K1_8) / U_6_6) : 0.0;
          // Block 4: K1 backward sub [LCObyOH]
          double K1_5 = active[5] ? ((y1_5 - U_5_6 * K1_6) / U_5_5) : 0.0;
          // Block 3: K1 backward sub [CO]
          double K1_4 = active[4] ? ((y1_4 - U_4_6 * K1_6) / U_4_4) : 0.0;
          // Block 2: K1 backward sub [DummyCH4]
          double K1_3 = active[3] ? (y1_3 / U_3_3) : 0.0;
          // Block 1: K1 backward sub [DummyNMVOC]
          double K1_2 = active[2] ? (y1_2 / U_2_2) : 0.0;
          double K1_1 = active[1] ? ((y1_1 - U_1_3 * K1_3) / U_1_1) : 0.0;
          double K1_0 = active[0] ? ((y1_0 - U_0_2 * K1_2) / U_0_0) : 0.0;


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
          // Rate evaluation F2 at Y2
          double F2_0 = 38199.012;
          double F2_1 = 4.2566446e-15;
          double F2_2 = -38199.012;
          double F2_3 = -4.2566446e-15;
          double F2_4 = 38199.012 - 7.3679649e-14*Y2_4;
          double F2_5 = 7.3679649e-14*Y2_4;
          double F2_6 = -2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*Y2_4;
          double F2_7 = 2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp);
          double F2_8 = -2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
          double F2_9 = 9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
          double F2_10 = -9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
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
          // Block 0: K2 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y2_0 = active[0] ? (rhs2_0) : 0.0;
          double y2_1 = active[1] ? (rhs2_1) : 0.0;
          // Block 1: K2 forward sub [DummyNMVOC]
          double y2_2 = active[2] ? (rhs2_2) : 0.0;
          // Block 2: K2 forward sub [DummyCH4]
          double y2_3 = active[3] ? (rhs2_3) : 0.0;
          // Block 3: K2 forward sub [CO]
          double y2_4 = active[4] ? (rhs2_4 - L_4_2 * y2_2 - L_4_3 * y2_3) : 0.0;
          // Block 4: K2 forward sub [LCObyOH]
          double y2_5 = active[5] ? (rhs2_5 - L_5_4 * y2_4) : 0.0;
          // Block 5: K2 forward sub [FixedOH]
          double y2_6 = active[6] ? (rhs2_6 - L_6_4 * y2_4) : 0.0;
          double y2_7 = active[7] ? (rhs2_7 - L_7_6 * y2_6) : 0.0;
          double y2_8 = active[8] ? (rhs2_8 - L_8_6 * y2_6) : 0.0;
          // Block 6: K2 forward sub [LCH4byCl]
          double y2_9 = active[9] ? (rhs2_9 - L_9_8 * y2_8) : 0.0;
          // Block 7: K2 forward sub [FixedCl]
          double y2_10 = active[10] ? (rhs2_10 - L_10_8 * y2_8) : 0.0;

          // Block 7: K2 backward sub [FixedCl]
          double K2_10 = active[10] ? (y2_10 / U_10_10) : 0.0;
          // Block 6: K2 backward sub [LCH4byCl]
          double K2_9 = active[9] ? ((y2_9 - U_9_10 * K2_10) / U_9_9) : 0.0;
          // Block 0: K2 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K2_8 = active[8] ? ((y2_8 - U_8_10 * K2_10) / U_8_8) : 0.0;
          double K2_7 = active[7] ? ((y2_7 - U_7_8 * K2_8) / U_7_7) : 0.0;
          // Block 5: K2 backward sub [FixedOH]
          double K2_6 = active[6] ? ((y2_6 - U_6_8 * K2_8) / U_6_6) : 0.0;
          // Block 4: K2 backward sub [LCObyOH]
          double K2_5 = active[5] ? ((y2_5 - U_5_6 * K2_6) / U_5_5) : 0.0;
          // Block 3: K2 backward sub [CO]
          double K2_4 = active[4] ? ((y2_4 - U_4_6 * K2_6) / U_4_4) : 0.0;
          // Block 2: K2 backward sub [DummyCH4]
          double K2_3 = active[3] ? (y2_3 / U_3_3) : 0.0;
          // Block 1: K2 backward sub [DummyNMVOC]
          double K2_2 = active[2] ? (y2_2 / U_2_2) : 0.0;
          double K2_1 = active[1] ? ((y2_1 - U_1_3 * K2_3) / U_1_1) : 0.0;
          double K2_0 = active[0] ? ((y2_0 - U_0_2 * K2_2) / U_0_0) : 0.0;


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
          // Block 0: K3 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y3_0 = active[0] ? (rhs3_0) : 0.0;
          double y3_1 = active[1] ? (rhs3_1) : 0.0;
          // Block 1: K3 forward sub [DummyNMVOC]
          double y3_2 = active[2] ? (rhs3_2) : 0.0;
          // Block 2: K3 forward sub [DummyCH4]
          double y3_3 = active[3] ? (rhs3_3) : 0.0;
          // Block 3: K3 forward sub [CO]
          double y3_4 = active[4] ? (rhs3_4 - L_4_2 * y3_2 - L_4_3 * y3_3) : 0.0;
          // Block 4: K3 forward sub [LCObyOH]
          double y3_5 = active[5] ? (rhs3_5 - L_5_4 * y3_4) : 0.0;
          // Block 5: K3 forward sub [FixedOH]
          double y3_6 = active[6] ? (rhs3_6 - L_6_4 * y3_4) : 0.0;
          double y3_7 = active[7] ? (rhs3_7 - L_7_6 * y3_6) : 0.0;
          double y3_8 = active[8] ? (rhs3_8 - L_8_6 * y3_6) : 0.0;
          // Block 6: K3 forward sub [LCH4byCl]
          double y3_9 = active[9] ? (rhs3_9 - L_9_8 * y3_8) : 0.0;
          // Block 7: K3 forward sub [FixedCl]
          double y3_10 = active[10] ? (rhs3_10 - L_10_8 * y3_8) : 0.0;

          // Block 7: K3 backward sub [FixedCl]
          double K3_10 = active[10] ? (y3_10 / U_10_10) : 0.0;
          // Block 6: K3 backward sub [LCH4byCl]
          double K3_9 = active[9] ? ((y3_9 - U_9_10 * K3_10) / U_9_9) : 0.0;
          // Block 0: K3 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K3_8 = active[8] ? ((y3_8 - U_8_10 * K3_10) / U_8_8) : 0.0;
          double K3_7 = active[7] ? ((y3_7 - U_7_8 * K3_8) / U_7_7) : 0.0;
          // Block 5: K3 backward sub [FixedOH]
          double K3_6 = active[6] ? ((y3_6 - U_6_8 * K3_8) / U_6_6) : 0.0;
          // Block 4: K3 backward sub [LCObyOH]
          double K3_5 = active[5] ? ((y3_5 - U_5_6 * K3_6) / U_5_5) : 0.0;
          // Block 3: K3 backward sub [CO]
          double K3_4 = active[4] ? ((y3_4 - U_4_6 * K3_6) / U_4_4) : 0.0;
          // Block 2: K3 backward sub [DummyCH4]
          double K3_3 = active[3] ? (y3_3 / U_3_3) : 0.0;
          // Block 1: K3 backward sub [DummyNMVOC]
          double K3_2 = active[2] ? (y3_2 / U_2_2) : 0.0;
          double K3_1 = active[1] ? ((y3_1 - U_1_3 * K3_3) / U_1_1) : 0.0;
          double K3_0 = active[0] ? ((y3_0 - U_0_2 * K3_2) / U_0_0) : 0.0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 11);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              if (active[0]) state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              if (active[1]) state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              if (active[2]) state(10) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              if (active[3]) state(9) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              if (active[4]) state(1) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              if (active[5]) state(6) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              if (active[6]) state(7) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              if (active[7]) state(4) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              if (active[8]) state(0) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              if (active[9]) state(5) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              if (active[10]) state(8) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
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
          const int NUM_SPECIES = 11;
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
          const double S_0 = state(Species::PCOfromNMVOC);  // [PCOfromNMVOC]
          const double S_1 = state(Species::PCOfromCH4);  // [PCOfromCH4]
          const double S_2 = state(Species::DummyNMVOC);  // [DummyNMVOC]
          const double S_3 = state(Species::DummyCH4);  // [DummyCH4]
          const double S_4 = state(Species::CO);  // [CO]
          const double S_5 = state(Species::LCObyOH);  // [LCObyOH]
          const double S_6 = state(Species::FixedOH);  // [FixedOH]
          const double S_7 = state(Species::LCH4byOH);  // [LCH4byOH]
          const double S_8 = state(Species::CH4);  // [CH4]
          const double S_9 = state(Species::LCH4byCl);  // [LCH4byCl]
          const double S_10 = state(Species::FixedCl);  // [FixedCl]

          // --- Reaction Rate Fluxes R_m ---
          const double R_0 = 2.45e-12*S_0*Kokkos::exp(-1775.0/Temp);
          const double R_1 = 9.6e-12*S_0*Kokkos::exp(-1360.0/Temp);
          const double R_2 = 4.2566446e-15;
          const double R_3 = 7.3679649e-14*S_1;
          const double R_4 = 38199.012;

          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_0_2 = 38199.012;
          double J_1_3 = 4.2566446e-15;
          double J_2_2 = -38199.012;
          double J_3_3 = -4.2566446e-15;
          double J_4_2 = 38199.012;
          double J_4_3 = 4.2566446e-15;
          double J_4_4 = -7.3679649e-14;
          double J_4_6 = -7.3679649e-14*S_4;
          double J_5_4 = 7.3679649e-14;
          double J_5_6 = 7.3679649e-14*S_4;
          double J_6_4 = -7.3679649e-14;
          double J_6_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
          double J_6_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp);
          double J_7_6 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double J_7_8 = 2.45e-12*Kokkos::exp(-1775.0/Temp);
          double J_8_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double J_8_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_8_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double J_9_8 = 9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_9_10 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double J_10_8 = -9.6e-12*Kokkos::exp(-1360.0/Temp);
          double J_10_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double W_0_0 = inv_g_dt;
          double W_0_2 = -J_0_2;
          double W_1_1 = inv_g_dt;
          double W_1_3 = -J_1_3;
          // Block 1: species [DummyNMVOC]
          double W_2_2 = inv_g_dt - J_2_2;
          // Block 2: species [DummyCH4]
          double W_3_3 = inv_g_dt - J_3_3;
          // Block 3: species [CO]
          double W_4_2 = -J_4_2;
          double W_4_3 = -J_4_3;
          double W_4_4 = inv_g_dt - J_4_4;
          double W_4_6 = -J_4_6;
          // Block 4: species [LCObyOH]
          double W_5_4 = -J_5_4;
          double W_5_5 = inv_g_dt;
          double W_5_6 = -J_5_6;
          // Block 5: species [FixedOH]
          double W_6_4 = -J_6_4;
          double W_6_6 = inv_g_dt - J_6_6;
          double W_6_8 = -J_6_8;
          // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double W_7_6 = -J_7_6;
          double W_7_7 = inv_g_dt;
          double W_7_8 = -J_7_8;
          double W_8_6 = -J_8_6;
          double W_8_8 = inv_g_dt - J_8_8;
          double W_8_10 = -J_8_10;
          // Block 6: species [LCH4byCl]
          double W_9_8 = -J_9_8;
          double W_9_9 = inv_g_dt;
          double W_9_10 = -J_9_10;
          // Block 7: species [FixedCl]
          double W_10_8 = -J_10_8;
          double W_10_10 = inv_g_dt - J_10_10;

          // Symbolic Doolittle Sparse LU Factorization
          // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double U_0_0 = W_0_0;
          double U_0_2 = W_0_2;
          double U_1_1 = W_1_1;
          double U_1_3 = W_1_3;
          // Block 1: species [DummyNMVOC]
          double U_2_2 = W_2_2;
          // Block 3: species [CO]
          double L_4_2 = (W_4_2) / U_2_2;
          // Block 2: species [DummyCH4]
          double U_3_3 = W_3_3;
          double L_4_3 = (W_4_3) / U_3_3;
          double U_4_4 = W_4_4;
          double U_4_6 = W_4_6;
          // Block 4: species [LCObyOH]
          double L_5_4 = (W_5_4) / U_4_4;
          // Block 5: species [FixedOH]
          double L_6_4 = (W_6_4) / U_4_4;
          double U_5_5 = W_5_5;
          double U_5_6 = W_5_6 - L_5_4 * U_4_6;
          double U_6_6 = W_6_6 - L_6_4 * U_4_6;
          double U_6_8 = W_6_8;
          double L_7_6 = (W_7_6) / U_6_6;
          double L_8_6 = (W_8_6) / U_6_6;
          double U_7_7 = W_7_7;
          double U_7_8 = W_7_8 - L_7_6 * U_6_8;
          double U_8_8 = W_8_8 - L_8_6 * U_6_8;
          double U_8_10 = W_8_10;
          // Block 6: species [LCH4byCl]
          double L_9_8 = (W_9_8) / U_8_8;
          // Block 7: species [FixedCl]
          double L_10_8 = (W_10_8) / U_8_8;
          double U_9_9 = W_9_9;
          double U_9_10 = W_9_10 - L_9_8 * U_8_10;
          double U_10_10 = W_10_10 - L_10_8 * U_8_10;

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F1_0 = 38199.012;
          double F1_1 = 4.2566446e-15;
          double F1_2 = -38199.012;
          double F1_3 = -4.2566446e-15;
          double F1_4 = 38199.012 - 7.3679649e-14*S_4;
          double F1_5 = 7.3679649e-14*S_4;
          double F1_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
          double F1_7 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
          double F1_8 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double F1_9 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          double F1_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
          // Block 0: K1 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y1_0 = F1_0;
          double y1_1 = F1_1;
          // Block 1: K1 forward sub [DummyNMVOC]
          double y1_2 = F1_2;
          // Block 2: K1 forward sub [DummyCH4]
          double y1_3 = F1_3;
          // Block 3: K1 forward sub [CO]
          double y1_4 = F1_4 - L_4_2 * y1_2 - L_4_3 * y1_3;
          // Block 4: K1 forward sub [LCObyOH]
          double y1_5 = F1_5 - L_5_4 * y1_4;
          // Block 5: K1 forward sub [FixedOH]
          double y1_6 = F1_6 - L_6_4 * y1_4;
          double y1_7 = F1_7 - L_7_6 * y1_6;
          double y1_8 = F1_8 - L_8_6 * y1_6;
          // Block 6: K1 forward sub [LCH4byCl]
          double y1_9 = F1_9 - L_9_8 * y1_8;
          // Block 7: K1 forward sub [FixedCl]
          double y1_10 = F1_10 - L_10_8 * y1_8;

          // Block 7: K1 backward sub [FixedCl]
          double K1_10 = y1_10 / U_10_10;
          // Block 6: K1 backward sub [LCH4byCl]
          double K1_9 = (y1_9 - U_9_10 * K1_10) / U_9_9;
          // Block 0: K1 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K1_8 = (y1_8 - U_8_10 * K1_10) / U_8_8;
          double K1_7 = (y1_7 - U_7_8 * K1_8) / U_7_7;
          // Block 5: K1 backward sub [FixedOH]
          double K1_6 = (y1_6 - U_6_8 * K1_8) / U_6_6;
          // Block 4: K1 backward sub [LCObyOH]
          double K1_5 = (y1_5 - U_5_6 * K1_6) / U_5_5;
          // Block 3: K1 backward sub [CO]
          double K1_4 = (y1_4 - U_4_6 * K1_6) / U_4_4;
          // Block 2: K1 backward sub [DummyCH4]
          double K1_3 = y1_3 / U_3_3;
          // Block 1: K1 backward sub [DummyNMVOC]
          double K1_2 = y1_2 / U_2_2;
          double K1_1 = (y1_1 - U_1_3 * K1_3) / U_1_1;
          double K1_0 = (y1_0 - U_0_2 * K1_2) / U_0_0;


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
          // Rate evaluation F2 at Y2
          double F2_0 = 38199.012;
          double F2_1 = 4.2566446e-15;
          double F2_2 = -38199.012;
          double F2_3 = -4.2566446e-15;
          double F2_4 = 38199.012 - 7.3679649e-14*Y2_4;
          double F2_5 = 7.3679649e-14*Y2_4;
          double F2_6 = -2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*Y2_4;
          double F2_7 = 2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp);
          double F2_8 = -2.45e-12*Y2_8*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
          double F2_9 = 9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
          double F2_10 = -9.6e-12*Y2_8*Kokkos::exp(-1360.0/Temp);
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
          // Block 0: K2 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y2_0 = rhs2_0;
          double y2_1 = rhs2_1;
          // Block 1: K2 forward sub [DummyNMVOC]
          double y2_2 = rhs2_2;
          // Block 2: K2 forward sub [DummyCH4]
          double y2_3 = rhs2_3;
          // Block 3: K2 forward sub [CO]
          double y2_4 = rhs2_4 - L_4_2 * y2_2 - L_4_3 * y2_3;
          // Block 4: K2 forward sub [LCObyOH]
          double y2_5 = rhs2_5 - L_5_4 * y2_4;
          // Block 5: K2 forward sub [FixedOH]
          double y2_6 = rhs2_6 - L_6_4 * y2_4;
          double y2_7 = rhs2_7 - L_7_6 * y2_6;
          double y2_8 = rhs2_8 - L_8_6 * y2_6;
          // Block 6: K2 forward sub [LCH4byCl]
          double y2_9 = rhs2_9 - L_9_8 * y2_8;
          // Block 7: K2 forward sub [FixedCl]
          double y2_10 = rhs2_10 - L_10_8 * y2_8;

          // Block 7: K2 backward sub [FixedCl]
          double K2_10 = y2_10 / U_10_10;
          // Block 6: K2 backward sub [LCH4byCl]
          double K2_9 = (y2_9 - U_9_10 * K2_10) / U_9_9;
          // Block 0: K2 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K2_8 = (y2_8 - U_8_10 * K2_10) / U_8_8;
          double K2_7 = (y2_7 - U_7_8 * K2_8) / U_7_7;
          // Block 5: K2 backward sub [FixedOH]
          double K2_6 = (y2_6 - U_6_8 * K2_8) / U_6_6;
          // Block 4: K2 backward sub [LCObyOH]
          double K2_5 = (y2_5 - U_5_6 * K2_6) / U_5_5;
          // Block 3: K2 backward sub [CO]
          double K2_4 = (y2_4 - U_4_6 * K2_6) / U_4_4;
          // Block 2: K2 backward sub [DummyCH4]
          double K2_3 = y2_3 / U_3_3;
          // Block 1: K2 backward sub [DummyNMVOC]
          double K2_2 = y2_2 / U_2_2;
          double K2_1 = (y2_1 - U_1_3 * K2_3) / U_1_1;
          double K2_0 = (y2_0 - U_0_2 * K2_2) / U_0_0;


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
          // Block 0: K3 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double y3_0 = rhs3_0;
          double y3_1 = rhs3_1;
          // Block 1: K3 forward sub [DummyNMVOC]
          double y3_2 = rhs3_2;
          // Block 2: K3 forward sub [DummyCH4]
          double y3_3 = rhs3_3;
          // Block 3: K3 forward sub [CO]
          double y3_4 = rhs3_4 - L_4_2 * y3_2 - L_4_3 * y3_3;
          // Block 4: K3 forward sub [LCObyOH]
          double y3_5 = rhs3_5 - L_5_4 * y3_4;
          // Block 5: K3 forward sub [FixedOH]
          double y3_6 = rhs3_6 - L_6_4 * y3_4;
          double y3_7 = rhs3_7 - L_7_6 * y3_6;
          double y3_8 = rhs3_8 - L_8_6 * y3_6;
          // Block 6: K3 forward sub [LCH4byCl]
          double y3_9 = rhs3_9 - L_9_8 * y3_8;
          // Block 7: K3 forward sub [FixedCl]
          double y3_10 = rhs3_10 - L_10_8 * y3_8;

          // Block 7: K3 backward sub [FixedCl]
          double K3_10 = y3_10 / U_10_10;
          // Block 6: K3 backward sub [LCH4byCl]
          double K3_9 = (y3_9 - U_9_10 * K3_10) / U_9_9;
          // Block 0: K3 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
          double K3_8 = (y3_8 - U_8_10 * K3_10) / U_8_8;
          double K3_7 = (y3_7 - U_7_8 * K3_8) / U_7_7;
          // Block 5: K3 backward sub [FixedOH]
          double K3_6 = (y3_6 - U_6_8 * K3_8) / U_6_6;
          // Block 4: K3 backward sub [LCObyOH]
          double K3_5 = (y3_5 - U_5_6 * K3_6) / U_5_5;
          // Block 3: K3 backward sub [CO]
          double K3_4 = (y3_4 - U_4_6 * K3_6) / U_4_4;
          // Block 2: K3 backward sub [DummyCH4]
          double K3_3 = y3_3 / U_3_3;
          // Block 1: K3 backward sub [DummyNMVOC]
          double K3_2 = y3_2 / U_2_2;
          double K3_1 = (y3_1 - U_1_3 * K3_3) / U_1_1;
          double K3_0 = (y3_0 - U_0_2 * K3_2) / U_0_0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(10)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[10] + rtol[10] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(9)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[9] + rtol[9] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(7)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[7] + rtol[7] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_7 + K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_7 + -2.9079558716805471 * K2_7 + 0.22354069897811571 * K3_7;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_8 + K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_8 + -2.9079558716805471 * K2_8 + 0.22354069897811571 * K3_8;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_9 + K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_9 + -2.9079558716805471 * K2_9 + 0.22354069897811571 * K3_9;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_10 + K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(8)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[8] + rtol[8] * ymax;
              double yerr_i = 0.5 * K1_10 + -2.9079558716805471 * K2_10 + 0.22354069897811571 * K3_10;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 11);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(10) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(9) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(1) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(6) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(7) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              state(4) += K1_7 + 6.1697947043828245 * K2_7 + -0.42772256543218573 * K3_7;
              state(0) += K1_8 + 6.1697947043828245 * K2_8 + -0.42772256543218573 * K3_8;
              state(5) += K1_9 + 6.1697947043828245 * K2_9 + -0.42772256543218573 * K3_9;
              state(8) += K1_10 + 6.1697947043828245 * K2_10 + -0.42772256543218573 * K3_10;
              // Save checkpoint data for adjoint/TLM
              if (chk.num_steps >= CheckpointBuffer::MAX_STEPS) {
                  ierr = 1;  // Exceeded MAX_STEPS budget
                  return -1;
              }
              chk.h[chk.num_steps] = dt;
              chk.state[chk.num_steps][0] = state(3);
              chk.state[chk.num_steps][1] = state(2);
              chk.state[chk.num_steps][2] = state(10);
              chk.state[chk.num_steps][3] = state(9);
              chk.state[chk.num_steps][4] = state(1);
              chk.state[chk.num_steps][5] = state(6);
              chk.state[chk.num_steps][6] = state(7);
              chk.state[chk.num_steps][7] = state(4);
              chk.state[chk.num_steps][8] = state(0);
              chk.state[chk.num_steps][9] = state(5);
              chk.state[chk.num_steps][10] = state(8);
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

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_0_2 = 38199.012;
              double J_1_3 = 4.2566446e-15;
              double J_2_2 = -38199.012;
              double J_3_3 = -4.2566446e-15;
              double J_4_2 = 38199.012;
              double J_4_3 = 4.2566446e-15;
              double J_4_4 = -7.3679649e-14;
              double J_4_6 = -7.3679649e-14*S_4;
              double J_5_4 = 7.3679649e-14;
              double J_5_6 = 7.3679649e-14*S_4;
              double J_6_4 = -7.3679649e-14;
              double J_6_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
              double J_6_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp);
              double J_7_6 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
              double J_7_8 = 2.45e-12*Kokkos::exp(-1775.0/Temp);
              double J_8_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
              double J_8_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_8_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
              double J_9_8 = 9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_9_10 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
              double J_10_8 = -9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_10_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);

              // Form iteration matrix W = (1/(gamma*h))*I - J
              // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double W_0_0 = inv_g_h;
              double W_0_2 = -J_0_2;
              double W_1_1 = inv_g_h;
              double W_1_3 = -J_1_3;
              // Block 1: species [DummyNMVOC]
              double W_2_2 = inv_g_h - J_2_2;
              // Block 2: species [DummyCH4]
              double W_3_3 = inv_g_h - J_3_3;
              // Block 3: species [CO]
              double W_4_2 = -J_4_2;
              double W_4_3 = -J_4_3;
              double W_4_4 = inv_g_h - J_4_4;
              double W_4_6 = -J_4_6;
              // Block 4: species [LCObyOH]
              double W_5_4 = -J_5_4;
              double W_5_5 = inv_g_h;
              double W_5_6 = -J_5_6;
              // Block 5: species [FixedOH]
              double W_6_4 = -J_6_4;
              double W_6_6 = inv_g_h - J_6_6;
              double W_6_8 = -J_6_8;
              // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double W_7_6 = -J_7_6;
              double W_7_7 = inv_g_h;
              double W_7_8 = -J_7_8;
              double W_8_6 = -J_8_6;
              double W_8_8 = inv_g_h - J_8_8;
              double W_8_10 = -J_8_10;
              // Block 6: species [LCH4byCl]
              double W_9_8 = -J_9_8;
              double W_9_9 = inv_g_h;
              double W_9_10 = -J_9_10;
              // Block 7: species [FixedCl]
              double W_10_8 = -J_10_8;
              double W_10_10 = inv_g_h - J_10_10;


              // Symbolic LU Factorization (recomputed at each adjoint step)
              // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double U_0_0 = W_0_0;
              double U_0_2 = W_0_2;
              double U_1_1 = W_1_1;
              double U_1_3 = W_1_3;
              // Block 1: species [DummyNMVOC]
              double U_2_2 = W_2_2;
              // Block 3: species [CO]
              double L_4_2 = (W_4_2) / U_2_2;
              // Block 2: species [DummyCH4]
              double U_3_3 = W_3_3;
              double L_4_3 = (W_4_3) / U_3_3;
              double U_4_4 = W_4_4;
              double U_4_6 = W_4_6;
              // Block 4: species [LCObyOH]
              double L_5_4 = (W_5_4) / U_4_4;
              // Block 5: species [FixedOH]
              double L_6_4 = (W_6_4) / U_4_4;
              double U_5_5 = W_5_5;
              double U_5_6 = W_5_6 - L_5_4 * U_4_6;
              double U_6_6 = W_6_6 - L_6_4 * U_4_6;
              double U_6_8 = W_6_8;
              double L_7_6 = (W_7_6) / U_6_6;
              double L_8_6 = (W_8_6) / U_6_6;
              double U_7_7 = W_7_7;
              double U_7_8 = W_7_8 - L_7_6 * U_6_8;
              double U_8_8 = W_8_8 - L_8_6 * U_6_8;
              double U_8_10 = W_8_10;
              // Block 6: species [LCH4byCl]
              double L_9_8 = (W_9_8) / U_8_8;
              // Block 7: species [FixedCl]
              double L_10_8 = (W_10_8) / U_8_8;
              double U_9_9 = W_9_9;
              double U_9_10 = W_9_10 - L_9_8 * U_8_10;
              double U_10_10 = W_10_10 - L_10_8 * U_8_10;


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
              // W^{-T} solve for u3
              // Block 0: u3 transpose forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double yt3_0 = v3_0 / U_0_0;
              double yt3_1 = v3_1 / U_1_1;
              // Block 1: u3 transpose forward sub [DummyNMVOC]
              double yt3_2 = (v3_2 - U_0_2 * yt3_0) / U_2_2;
              // Block 2: u3 transpose forward sub [DummyCH4]
              double yt3_3 = (v3_3 - U_1_3 * yt3_1) / U_3_3;
              // Block 3: u3 transpose forward sub [CO]
              double yt3_4 = v3_4 / U_4_4;
              // Block 4: u3 transpose forward sub [LCObyOH]
              double yt3_5 = v3_5 / U_5_5;
              // Block 5: u3 transpose forward sub [FixedOH]
              double yt3_6 = (v3_6 - U_4_6 * yt3_4 - U_5_6 * yt3_5) / U_6_6;
              double yt3_7 = v3_7 / U_7_7;
              double yt3_8 = (v3_8 - U_6_8 * yt3_6 - U_7_8 * yt3_7) / U_8_8;
              // Block 6: u3 transpose forward sub [LCH4byCl]
              double yt3_9 = v3_9 / U_9_9;
              // Block 7: u3 transpose forward sub [FixedCl]
              double yt3_10 = (v3_10 - U_8_10 * yt3_8 - U_9_10 * yt3_9) / U_10_10;
              // Block 7: u3 transpose backward sub [FixedCl]
              double u3_10 = yt3_10;
              // Block 6: u3 transpose backward sub [LCH4byCl]
              double u3_9 = yt3_9;
              // Block 0: u3 transpose backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double u3_8 = yt3_8 - L_9_8 * u3_9 - L_10_8 * u3_10;
              double u3_7 = yt3_7;
              // Block 5: u3 transpose backward sub [FixedOH]
              double u3_6 = yt3_6 - L_7_6 * u3_7 - L_8_6 * u3_8;
              // Block 4: u3 transpose backward sub [LCObyOH]
              double u3_5 = yt3_5;
              // Block 3: u3 transpose backward sub [CO]
              double u3_4 = yt3_4 - L_5_4 * u3_5 - L_6_4 * u3_6;
              // Block 2: u3 transpose backward sub [DummyCH4]
              double u3_3 = yt3_3 - L_4_3 * u3_4;
              // Block 1: u3 transpose backward sub [DummyNMVOC]
              double u3_2 = yt3_2 - L_4_2 * u3_4;
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
              // W^{-T} solve for u2
              // Block 0: u2 transpose forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double yt2_0 = v2_0 / U_0_0;
              double yt2_1 = v2_1 / U_1_1;
              // Block 1: u2 transpose forward sub [DummyNMVOC]
              double yt2_2 = (v2_2 - U_0_2 * yt2_0) / U_2_2;
              // Block 2: u2 transpose forward sub [DummyCH4]
              double yt2_3 = (v2_3 - U_1_3 * yt2_1) / U_3_3;
              // Block 3: u2 transpose forward sub [CO]
              double yt2_4 = v2_4 / U_4_4;
              // Block 4: u2 transpose forward sub [LCObyOH]
              double yt2_5 = v2_5 / U_5_5;
              // Block 5: u2 transpose forward sub [FixedOH]
              double yt2_6 = (v2_6 - U_4_6 * yt2_4 - U_5_6 * yt2_5) / U_6_6;
              double yt2_7 = v2_7 / U_7_7;
              double yt2_8 = (v2_8 - U_6_8 * yt2_6 - U_7_8 * yt2_7) / U_8_8;
              // Block 6: u2 transpose forward sub [LCH4byCl]
              double yt2_9 = v2_9 / U_9_9;
              // Block 7: u2 transpose forward sub [FixedCl]
              double yt2_10 = (v2_10 - U_8_10 * yt2_8 - U_9_10 * yt2_9) / U_10_10;
              // Block 7: u2 transpose backward sub [FixedCl]
              double u2_10 = yt2_10;
              // Block 6: u2 transpose backward sub [LCH4byCl]
              double u2_9 = yt2_9;
              // Block 0: u2 transpose backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double u2_8 = yt2_8 - L_9_8 * u2_9 - L_10_8 * u2_10;
              double u2_7 = yt2_7;
              // Block 5: u2 transpose backward sub [FixedOH]
              double u2_6 = yt2_6 - L_7_6 * u2_7 - L_8_6 * u2_8;
              // Block 4: u2 transpose backward sub [LCObyOH]
              double u2_5 = yt2_5;
              // Block 3: u2 transpose backward sub [CO]
              double u2_4 = yt2_4 - L_5_4 * u2_5 - L_6_4 * u2_6;
              // Block 2: u2 transpose backward sub [DummyCH4]
              double u2_3 = yt2_3 - L_4_3 * u2_4;
              // Block 1: u2 transpose backward sub [DummyNMVOC]
              double u2_2 = yt2_2 - L_4_2 * u2_4;
              double u2_1 = yt2_1;
              double u2_0 = yt2_0;


              // Adjoint stage 1
              // v1: RHS for W^{-T} solve
              double v1_0 = lam_0 + (-1.0156171083877703 / h) * u2_0 + (4.0759956452537702 / h) * u3_0;
              double v1_1 = lam_1 + (-1.0156171083877703 / h) * u2_1 + (4.0759956452537702 / h) * u3_1;
              double v1_2 = lam_2 + (J_0_2 * u2_0 + J_2_2 * u2_2 + J_4_2 * u2_4) + (-1.0156171083877703 / h) * u2_2 + (J_0_2 * u3_0 + J_2_2 * u3_2 + J_4_2 * u3_4) + (4.0759956452537702 / h) * u3_2;
              double v1_3 = lam_3 + (J_1_3 * u2_1 + J_3_3 * u2_3 + J_4_3 * u2_4) + (-1.0156171083877703 / h) * u2_3 + (J_1_3 * u3_1 + J_3_3 * u3_3 + J_4_3 * u3_4) + (4.0759956452537702 / h) * u3_3;
              double v1_4 = lam_4 + (J_4_4 * u2_4 + J_5_4 * u2_5 + J_6_4 * u2_6) + (-1.0156171083877703 / h) * u2_4 + (J_4_4 * u3_4 + J_5_4 * u3_5 + J_6_4 * u3_6) + (4.0759956452537702 / h) * u3_4;
              double v1_5 = lam_5 + (-1.0156171083877703 / h) * u2_5 + (4.0759956452537702 / h) * u3_5;
              double v1_6 = lam_6 + (J_4_6 * u2_4 + J_5_6 * u2_5 + J_6_6 * u2_6 + J_7_6 * u2_7 + J_8_6 * u2_8) + (-1.0156171083877703 / h) * u2_6 + (J_4_6 * u3_4 + J_5_6 * u3_5 + J_6_6 * u3_6 + J_7_6 * u3_7 + J_8_6 * u3_8) + (4.0759956452537702 / h) * u3_6;
              double v1_7 = lam_7 + (-1.0156171083877703 / h) * u2_7 + (4.0759956452537702 / h) * u3_7;
              double v1_8 = lam_8 + (J_6_8 * u2_6 + J_7_8 * u2_7 + J_8_8 * u2_8 + J_9_8 * u2_9 + J_10_8 * u2_10) + (-1.0156171083877703 / h) * u2_8 + (J_6_8 * u3_6 + J_7_8 * u3_7 + J_8_8 * u3_8 + J_9_8 * u3_9 + J_10_8 * u3_10) + (4.0759956452537702 / h) * u3_8;
              double v1_9 = lam_9 + (-1.0156171083877703 / h) * u2_9 + (4.0759956452537702 / h) * u3_9;
              double v1_10 = lam_10 + (J_8_10 * u2_8 + J_9_10 * u2_9 + J_10_10 * u2_10) + (-1.0156171083877703 / h) * u2_10 + (J_8_10 * u3_8 + J_9_10 * u3_9 + J_10_10 * u3_10) + (4.0759956452537702 / h) * u3_10;
              // W^{-T} solve for u1
              // Block 0: u1 transpose forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double yt1_0 = v1_0 / U_0_0;
              double yt1_1 = v1_1 / U_1_1;
              // Block 1: u1 transpose forward sub [DummyNMVOC]
              double yt1_2 = (v1_2 - U_0_2 * yt1_0) / U_2_2;
              // Block 2: u1 transpose forward sub [DummyCH4]
              double yt1_3 = (v1_3 - U_1_3 * yt1_1) / U_3_3;
              // Block 3: u1 transpose forward sub [CO]
              double yt1_4 = v1_4 / U_4_4;
              // Block 4: u1 transpose forward sub [LCObyOH]
              double yt1_5 = v1_5 / U_5_5;
              // Block 5: u1 transpose forward sub [FixedOH]
              double yt1_6 = (v1_6 - U_4_6 * yt1_4 - U_5_6 * yt1_5) / U_6_6;
              double yt1_7 = v1_7 / U_7_7;
              double yt1_8 = (v1_8 - U_6_8 * yt1_6 - U_7_8 * yt1_7) / U_8_8;
              // Block 6: u1 transpose forward sub [LCH4byCl]
              double yt1_9 = v1_9 / U_9_9;
              // Block 7: u1 transpose forward sub [FixedCl]
              double yt1_10 = (v1_10 - U_8_10 * yt1_8 - U_9_10 * yt1_9) / U_10_10;
              // Block 7: u1 transpose backward sub [FixedCl]
              double u1_10 = yt1_10;
              // Block 6: u1 transpose backward sub [LCH4byCl]
              double u1_9 = yt1_9;
              // Block 0: u1 transpose backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double u1_8 = yt1_8 - L_9_8 * u1_9 - L_10_8 * u1_10;
              double u1_7 = yt1_7;
              // Block 5: u1 transpose backward sub [FixedOH]
              double u1_6 = yt1_6 - L_7_6 * u1_7 - L_8_6 * u1_8;
              // Block 4: u1 transpose backward sub [LCObyOH]
              double u1_5 = yt1_5;
              // Block 3: u1 transpose backward sub [CO]
              double u1_4 = yt1_4 - L_5_4 * u1_5 - L_6_4 * u1_6;
              // Block 2: u1 transpose backward sub [DummyCH4]
              double u1_3 = yt1_3 - L_4_3 * u1_4;
              // Block 1: u1 transpose backward sub [DummyNMVOC]
              double u1_2 = yt1_2 - L_4_2 * u1_4;
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

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_0_2 = 38199.012;
              double J_1_3 = 4.2566446e-15;
              double J_2_2 = -38199.012;
              double J_3_3 = -4.2566446e-15;
              double J_4_2 = 38199.012;
              double J_4_3 = 4.2566446e-15;
              double J_4_4 = -7.3679649e-14;
              double J_4_6 = -7.3679649e-14*S_4;
              double J_5_4 = 7.3679649e-14;
              double J_5_6 = 7.3679649e-14*S_4;
              double J_6_4 = -7.3679649e-14;
              double J_6_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp) - 7.3679649e-14*S_4;
              double J_6_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp);
              double J_7_6 = 2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
              double J_7_8 = 2.45e-12*Kokkos::exp(-1775.0/Temp);
              double J_8_6 = -2.45e-12*S_8*Kokkos::exp(-1775.0/Temp);
              double J_8_8 = -2.45e-12*Kokkos::exp(-1775.0/Temp) - 9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_8_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
              double J_9_8 = 9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_9_10 = 9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);
              double J_10_8 = -9.6e-12*Kokkos::exp(-1360.0/Temp);
              double J_10_10 = -9.6e-12*S_8*Kokkos::exp(-1360.0/Temp);

              // Form iteration matrix W = inv_g_h*I - J
              double W_0_0 = inv_g_h;
              double W_0_2 = -J_0_2;
              double W_1_1 = inv_g_h;
              double W_1_3 = -J_1_3;
              double W_2_2 = inv_g_h - J_2_2;
              double W_3_3 = inv_g_h - J_3_3;
              double W_4_2 = -J_4_2;
              double W_4_3 = -J_4_3;
              double W_4_4 = inv_g_h - J_4_4;
              double W_4_6 = -J_4_6;
              double W_5_4 = -J_5_4;
              double W_5_5 = inv_g_h;
              double W_5_6 = -J_5_6;
              double W_6_4 = -J_6_4;
              double W_6_6 = inv_g_h - J_6_6;
              double W_6_8 = -J_6_8;
              double W_7_6 = -J_7_6;
              double W_7_7 = inv_g_h;
              double W_7_8 = -J_7_8;
              double W_8_6 = -J_8_6;
              double W_8_8 = inv_g_h - J_8_8;
              double W_8_10 = -J_8_10;
              double W_9_8 = -J_9_8;
              double W_9_9 = inv_g_h;
              double W_9_10 = -J_9_10;
              double W_10_8 = -J_10_8;
              double W_10_10 = inv_g_h - J_10_10;

              // Symbolic LU Factorization
              // Block 0: species [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double U_0_0 = W_0_0;
              double U_0_2 = W_0_2;
              double U_1_1 = W_1_1;
              double U_1_3 = W_1_3;
              // Block 1: species [DummyNMVOC]
              double U_2_2 = W_2_2;
              // Block 3: species [CO]
              double L_4_2 = (W_4_2) / U_2_2;
              // Block 2: species [DummyCH4]
              double U_3_3 = W_3_3;
              double L_4_3 = (W_4_3) / U_3_3;
              double U_4_4 = W_4_4;
              double U_4_6 = W_4_6;
              // Block 4: species [LCObyOH]
              double L_5_4 = (W_5_4) / U_4_4;
              // Block 5: species [FixedOH]
              double L_6_4 = (W_6_4) / U_4_4;
              double U_5_5 = W_5_5;
              double U_5_6 = W_5_6 - L_5_4 * U_4_6;
              double U_6_6 = W_6_6 - L_6_4 * U_4_6;
              double U_6_8 = W_6_8;
              double L_7_6 = (W_7_6) / U_6_6;
              double L_8_6 = (W_8_6) / U_6_6;
              double U_7_7 = W_7_7;
              double U_7_8 = W_7_8 - L_7_6 * U_6_8;
              double U_8_8 = W_8_8 - L_8_6 * U_6_8;
              double U_8_10 = W_8_10;
              // Block 6: species [LCH4byCl]
              double L_9_8 = (W_9_8) / U_8_8;
              // Block 7: species [FixedCl]
              double L_10_8 = (W_10_8) / U_8_8;
              double U_9_9 = W_9_9;
              double U_9_10 = W_9_10 - L_9_8 * U_8_10;
              double U_10_10 = W_10_10 - L_10_8 * U_8_10;



              // --- TLM Stage 1 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs1_0 = J_0_2 * dC_2;
              double tlm_rhs1_1 = J_1_3 * dC_3;
              double tlm_rhs1_2 = J_2_2 * dC_2;
              double tlm_rhs1_3 = J_3_3 * dC_3;
              double tlm_rhs1_4 = J_4_2 * dC_2 + J_4_3 * dC_3 + J_4_4 * dC_4 + J_4_6 * dC_6;
              double tlm_rhs1_5 = J_5_4 * dC_4 + J_5_6 * dC_6;
              double tlm_rhs1_6 = J_6_4 * dC_4 + J_6_6 * dC_6 + J_6_8 * dC_8;
              double tlm_rhs1_7 = J_7_6 * dC_6 + J_7_8 * dC_8;
              double tlm_rhs1_8 = J_8_6 * dC_6 + J_8_8 * dC_8 + J_8_10 * dC_10;
              double tlm_rhs1_9 = J_9_8 * dC_8 + J_9_10 * dC_10;
              double tlm_rhs1_10 = J_10_8 * dC_8 + J_10_10 * dC_10;
              // Solve W * dK1 = tlm_rhs1
              // Block 0: dK1 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double tlm_y1_0 = tlm_rhs1_0;
              double tlm_y1_1 = tlm_rhs1_1;
              // Block 1: dK1 forward sub [DummyNMVOC]
              double tlm_y1_2 = tlm_rhs1_2;
              // Block 2: dK1 forward sub [DummyCH4]
              double tlm_y1_3 = tlm_rhs1_3;
              // Block 3: dK1 forward sub [CO]
              double tlm_y1_4 = tlm_rhs1_4 - L_4_2 * tlm_y1_2 - L_4_3 * tlm_y1_3;
              // Block 4: dK1 forward sub [LCObyOH]
              double tlm_y1_5 = tlm_rhs1_5 - L_5_4 * tlm_y1_4;
              // Block 5: dK1 forward sub [FixedOH]
              double tlm_y1_6 = tlm_rhs1_6 - L_6_4 * tlm_y1_4;
              double tlm_y1_7 = tlm_rhs1_7 - L_7_6 * tlm_y1_6;
              double tlm_y1_8 = tlm_rhs1_8 - L_8_6 * tlm_y1_6;
              // Block 6: dK1 forward sub [LCH4byCl]
              double tlm_y1_9 = tlm_rhs1_9 - L_9_8 * tlm_y1_8;
              // Block 7: dK1 forward sub [FixedCl]
              double tlm_y1_10 = tlm_rhs1_10 - L_10_8 * tlm_y1_8;
              // Block 7: dK1 backward sub [FixedCl]
              double dK1_10 = tlm_y1_10 / U_10_10;
              // Block 6: dK1 backward sub [LCH4byCl]
              double dK1_9 = (tlm_y1_9 - U_9_10 * dK1_10) / U_9_9;
              // Block 0: dK1 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double dK1_8 = (tlm_y1_8 - U_8_10 * dK1_10) / U_8_8;
              double dK1_7 = (tlm_y1_7 - U_7_8 * dK1_8) / U_7_7;
              // Block 5: dK1 backward sub [FixedOH]
              double dK1_6 = (tlm_y1_6 - U_6_8 * dK1_8) / U_6_6;
              // Block 4: dK1 backward sub [LCObyOH]
              double dK1_5 = (tlm_y1_5 - U_5_6 * dK1_6) / U_5_5;
              // Block 3: dK1 backward sub [CO]
              double dK1_4 = (tlm_y1_4 - U_4_6 * dK1_6) / U_4_4;
              // Block 2: dK1 backward sub [DummyCH4]
              double dK1_3 = tlm_y1_3 / U_3_3;
              // Block 1: dK1 backward sub [DummyNMVOC]
              double dK1_2 = tlm_y1_2 / U_2_2;
              double dK1_1 = (tlm_y1_1 - U_1_3 * dK1_3) / U_1_1;
              double dK1_0 = (tlm_y1_0 - U_0_2 * dK1_2) / U_0_0;

              // --- TLM Stage 2 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs2_0 = J_0_2 * (dC_2 + dK1_2) + (-1.0156171083877703 / h) * dK1_0;
              double tlm_rhs2_1 = J_1_3 * (dC_3 + dK1_3) + (-1.0156171083877703 / h) * dK1_1;
              double tlm_rhs2_2 = J_2_2 * (dC_2 + dK1_2) + (-1.0156171083877703 / h) * dK1_2;
              double tlm_rhs2_3 = J_3_3 * (dC_3 + dK1_3) + (-1.0156171083877703 / h) * dK1_3;
              double tlm_rhs2_4 = J_4_2 * (dC_2 + dK1_2) + J_4_3 * (dC_3 + dK1_3) + J_4_4 * (dC_4 + dK1_4) + J_4_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_4;
              double tlm_rhs2_5 = J_5_4 * (dC_4 + dK1_4) + J_5_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_5;
              double tlm_rhs2_6 = J_6_4 * (dC_4 + dK1_4) + J_6_6 * (dC_6 + dK1_6) + J_6_8 * (dC_8 + dK1_8) + (-1.0156171083877703 / h) * dK1_6;
              double tlm_rhs2_7 = J_7_6 * (dC_6 + dK1_6) + J_7_8 * (dC_8 + dK1_8) + (-1.0156171083877703 / h) * dK1_7;
              double tlm_rhs2_8 = J_8_6 * (dC_6 + dK1_6) + J_8_8 * (dC_8 + dK1_8) + J_8_10 * (dC_10 + dK1_10) + (-1.0156171083877703 / h) * dK1_8;
              double tlm_rhs2_9 = J_9_8 * (dC_8 + dK1_8) + J_9_10 * (dC_10 + dK1_10) + (-1.0156171083877703 / h) * dK1_9;
              double tlm_rhs2_10 = J_10_8 * (dC_8 + dK1_8) + J_10_10 * (dC_10 + dK1_10) + (-1.0156171083877703 / h) * dK1_10;
              // Solve W * dK2 = tlm_rhs2
              // Block 0: dK2 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double tlm_y2_0 = tlm_rhs2_0;
              double tlm_y2_1 = tlm_rhs2_1;
              // Block 1: dK2 forward sub [DummyNMVOC]
              double tlm_y2_2 = tlm_rhs2_2;
              // Block 2: dK2 forward sub [DummyCH4]
              double tlm_y2_3 = tlm_rhs2_3;
              // Block 3: dK2 forward sub [CO]
              double tlm_y2_4 = tlm_rhs2_4 - L_4_2 * tlm_y2_2 - L_4_3 * tlm_y2_3;
              // Block 4: dK2 forward sub [LCObyOH]
              double tlm_y2_5 = tlm_rhs2_5 - L_5_4 * tlm_y2_4;
              // Block 5: dK2 forward sub [FixedOH]
              double tlm_y2_6 = tlm_rhs2_6 - L_6_4 * tlm_y2_4;
              double tlm_y2_7 = tlm_rhs2_7 - L_7_6 * tlm_y2_6;
              double tlm_y2_8 = tlm_rhs2_8 - L_8_6 * tlm_y2_6;
              // Block 6: dK2 forward sub [LCH4byCl]
              double tlm_y2_9 = tlm_rhs2_9 - L_9_8 * tlm_y2_8;
              // Block 7: dK2 forward sub [FixedCl]
              double tlm_y2_10 = tlm_rhs2_10 - L_10_8 * tlm_y2_8;
              // Block 7: dK2 backward sub [FixedCl]
              double dK2_10 = tlm_y2_10 / U_10_10;
              // Block 6: dK2 backward sub [LCH4byCl]
              double dK2_9 = (tlm_y2_9 - U_9_10 * dK2_10) / U_9_9;
              // Block 0: dK2 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double dK2_8 = (tlm_y2_8 - U_8_10 * dK2_10) / U_8_8;
              double dK2_7 = (tlm_y2_7 - U_7_8 * dK2_8) / U_7_7;
              // Block 5: dK2 backward sub [FixedOH]
              double dK2_6 = (tlm_y2_6 - U_6_8 * dK2_8) / U_6_6;
              // Block 4: dK2 backward sub [LCObyOH]
              double dK2_5 = (tlm_y2_5 - U_5_6 * dK2_6) / U_5_5;
              // Block 3: dK2 backward sub [CO]
              double dK2_4 = (tlm_y2_4 - U_4_6 * dK2_6) / U_4_4;
              // Block 2: dK2 backward sub [DummyCH4]
              double dK2_3 = tlm_y2_3 / U_3_3;
              // Block 1: dK2 backward sub [DummyNMVOC]
              double dK2_2 = tlm_y2_2 / U_2_2;
              double dK2_1 = (tlm_y2_1 - U_1_3 * dK2_3) / U_1_1;
              double dK2_0 = (tlm_y2_0 - U_0_2 * dK2_2) / U_0_0;

              // --- TLM Stage 3 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs3_0 = J_0_2 * (dC_2 + dK1_2) + (4.0759956452537702 / h) * dK1_0 + (9.20767942983308 / h) * dK2_0;
              double tlm_rhs3_1 = J_1_3 * (dC_3 + dK1_3) + (4.0759956452537702 / h) * dK1_1 + (9.20767942983308 / h) * dK2_1;
              double tlm_rhs3_2 = J_2_2 * (dC_2 + dK1_2) + (4.0759956452537702 / h) * dK1_2 + (9.20767942983308 / h) * dK2_2;
              double tlm_rhs3_3 = J_3_3 * (dC_3 + dK1_3) + (4.0759956452537702 / h) * dK1_3 + (9.20767942983308 / h) * dK2_3;
              double tlm_rhs3_4 = J_4_2 * (dC_2 + dK1_2) + J_4_3 * (dC_3 + dK1_3) + J_4_4 * (dC_4 + dK1_4) + J_4_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_4 + (9.20767942983308 / h) * dK2_4;
              double tlm_rhs3_5 = J_5_4 * (dC_4 + dK1_4) + J_5_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_5 + (9.20767942983308 / h) * dK2_5;
              double tlm_rhs3_6 = J_6_4 * (dC_4 + dK1_4) + J_6_6 * (dC_6 + dK1_6) + J_6_8 * (dC_8 + dK1_8) + (4.0759956452537702 / h) * dK1_6 + (9.20767942983308 / h) * dK2_6;
              double tlm_rhs3_7 = J_7_6 * (dC_6 + dK1_6) + J_7_8 * (dC_8 + dK1_8) + (4.0759956452537702 / h) * dK1_7 + (9.20767942983308 / h) * dK2_7;
              double tlm_rhs3_8 = J_8_6 * (dC_6 + dK1_6) + J_8_8 * (dC_8 + dK1_8) + J_8_10 * (dC_10 + dK1_10) + (4.0759956452537702 / h) * dK1_8 + (9.20767942983308 / h) * dK2_8;
              double tlm_rhs3_9 = J_9_8 * (dC_8 + dK1_8) + J_9_10 * (dC_10 + dK1_10) + (4.0759956452537702 / h) * dK1_9 + (9.20767942983308 / h) * dK2_9;
              double tlm_rhs3_10 = J_10_8 * (dC_8 + dK1_8) + J_10_10 * (dC_10 + dK1_10) + (4.0759956452537702 / h) * dK1_10 + (9.20767942983308 / h) * dK2_10;
              // Solve W * dK3 = tlm_rhs3
              // Block 0: dK3 forward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double tlm_y3_0 = tlm_rhs3_0;
              double tlm_y3_1 = tlm_rhs3_1;
              // Block 1: dK3 forward sub [DummyNMVOC]
              double tlm_y3_2 = tlm_rhs3_2;
              // Block 2: dK3 forward sub [DummyCH4]
              double tlm_y3_3 = tlm_rhs3_3;
              // Block 3: dK3 forward sub [CO]
              double tlm_y3_4 = tlm_rhs3_4 - L_4_2 * tlm_y3_2 - L_4_3 * tlm_y3_3;
              // Block 4: dK3 forward sub [LCObyOH]
              double tlm_y3_5 = tlm_rhs3_5 - L_5_4 * tlm_y3_4;
              // Block 5: dK3 forward sub [FixedOH]
              double tlm_y3_6 = tlm_rhs3_6 - L_6_4 * tlm_y3_4;
              double tlm_y3_7 = tlm_rhs3_7 - L_7_6 * tlm_y3_6;
              double tlm_y3_8 = tlm_rhs3_8 - L_8_6 * tlm_y3_6;
              // Block 6: dK3 forward sub [LCH4byCl]
              double tlm_y3_9 = tlm_rhs3_9 - L_9_8 * tlm_y3_8;
              // Block 7: dK3 forward sub [FixedCl]
              double tlm_y3_10 = tlm_rhs3_10 - L_10_8 * tlm_y3_8;
              // Block 7: dK3 backward sub [FixedCl]
              double dK3_10 = tlm_y3_10 / U_10_10;
              // Block 6: dK3 backward sub [LCH4byCl]
              double dK3_9 = (tlm_y3_9 - U_9_10 * dK3_10) / U_9_9;
              // Block 0: dK3 backward sub [PCOfromNMVOC, PCOfromCH4, LCH4byOH, CH4]
              double dK3_8 = (tlm_y3_8 - U_8_10 * dK3_10) / U_8_8;
              double dK3_7 = (tlm_y3_7 - U_7_8 * dK3_8) / U_7_7;
              // Block 5: dK3 backward sub [FixedOH]
              double dK3_6 = (tlm_y3_6 - U_6_8 * dK3_8) / U_6_6;
              // Block 4: dK3 backward sub [LCObyOH]
              double dK3_5 = (tlm_y3_5 - U_5_6 * dK3_6) / U_5_5;
              // Block 3: dK3 backward sub [CO]
              double dK3_4 = (tlm_y3_4 - U_4_6 * dK3_6) / U_4_4;
              // Block 2: dK3 backward sub [DummyCH4]
              double dK3_3 = tlm_y3_3 / U_3_3;
              // Block 1: dK3 backward sub [DummyNMVOC]
              double dK3_2 = tlm_y3_2 / U_2_2;
              double dK3_1 = (tlm_y3_1 - U_1_3 * dK3_3) / U_1_1;
              double dK3_0 = (tlm_y3_0 - U_0_2 * dK3_2) / U_0_0;

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
      }
#endif
  };
}
