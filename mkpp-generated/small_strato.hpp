#pragma once
#include <Kokkos_Core.hpp>
// CPU libraries can own the heavy helper definitions. Device builds keep the
// complete device-callable implementation visible without relocatable device code.
#if defined(MKPP_SEPARATE_HOST_KERNELS_SMALL_STRATO)
#include "small_strato/host_declarations.hpp"
#else
#include "small_strato/accessor.hpp"
#include "small_strato/rates.hpp"
#include "small_strato/jacobian.hpp"
#include "small_strato/supernodal_factorize.hpp"
#include "small_strato/supernodal_solve.hpp"
#include "small_strato/solve.hpp"
#endif
// Generated solver for small_strato
// SZA Workload Sorted: true
// Hysteresis/Spline Continuous Transition: true
namespace mkpp::generated::small_strato {
  // Pure Kokkos abstractions (no raw pragmas allowed)

  /**
   * @brief Strongly typed enum for species indexing into state vectors.
   */
  enum Species : int {
      O = 0,
      O1D = 1,
      O3 = 2,
      NO = 3,
      NO2 = 4,
      M = 5,
      O2 = 6
  };

#ifdef MKPP_ENABLE_ADJOINT
  // Checkpoint buffer for discrete adjoint/TLM integration
  // Recompute-J strategy: only state is stored, Jacobian recomputed from saved state
  struct CheckpointBuffer {
      static constexpr int MAX_STEPS = 200;
      static constexpr int NUM_SPECIES = 7;
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
      static constexpr bool SUPPORTS_RUNTIME_ENVIRONMENT = false;
      static constexpr int NUM_RUNTIME_PARAMETERS = 4;
      static constexpr bool HAS_CLOUD_GATED = false;
#ifdef MKPP_ENABLE_ADJOINT
      // Expose the checkpoint buffer through the kernel type so generic
      // device-path callers can name it without the mechanism namespace.
      using CheckpointBuffer = ::mkpp::generated::small_strato::CheckpointBuffer;
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
      KOKKOS_INLINE_FUNCTION void compute_rates(const StateView& state, RateView& F_block, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
              const auto state_accessor = detail::make_stride_view(state);
              const auto rates_accessor = detail::make_stride_view(F_block);
              detail::compute_rates_chunk_0(state_accessor, rates_accessor, jvals, 0.0, 0.0, 2.46e+19);
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
      KOKKOS_INLINE_FUNCTION void compute_jacobian(const StateView& state, JacView& J_block, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
              const auto state_accessor = detail::make_stride_view(state);
              const auto jacobian_accessor = detail::make_stride_view(J_block);
              detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, 0.0, 0.0, 2.46e+19);
      }

#ifdef MKPP_ENABLE_ADJOINT
      template <class StateView, class JacView>
      KOKKOS_INLINE_FUNCTION void compute_adjoint(const StateView& state, JacView& J_adj_block, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
          // --- Sparse Analytical Adjoint Jacobian Entries J_adj_block(i, j) = J^T(i, j) ---
          // J^T(O, O): d(d[O]/dt) / d[O]
          J_adj_block(0, 0) = -1.069e-11*state(4) - 8.018e-17*state(6) - 1.576e-15*state(2);
          // J^T(O, O3): d(d[O3]/dt) / d[O]
          J_adj_block(0, 2) = 8.018e-17*state(6) - 1.576e-15*state(2);
          // J^T(O, NO): d(d[NO]/dt) / d[O]
          J_adj_block(0, 3) = 1.069e-11*state(4);
          // J^T(O, NO2): d(d[NO2]/dt) / d[O]
          J_adj_block(0, 4) = -1.069e-11*state(4);
          // J^T(O, O2): d(d[O2]/dt) / d[O]
          J_adj_block(0, 6) = 1.069e-11*state(4) - 8.018e-17*state(6) + 3.152e-15*state(2);
          // J^T(O1D, O): d(d[O]/dt) / d[O1D]
          J_adj_block(1, 0) = 7.11e-11*state(5);
          // J^T(O1D, O1D): d(d[O1D]/dt) / d[O1D]
          J_adj_block(1, 1) = -7.11e-11*state(5) - 1.2e-10*state(2);
          // J^T(O1D, O3): d(d[O3]/dt) / d[O1D]
          J_adj_block(1, 2) = -1.2e-10*state(2);
          // J^T(O1D, O2): d(d[O2]/dt) / d[O1D]
          J_adj_block(1, 6) = 2.4e-10*state(2);
          // J^T(O3, O): d(d[O]/dt) / d[O3]
          J_adj_block(2, 0) = -1.576e-15*state(0) + 1.0*jvals[1];
          // J^T(O3, O1D): d(d[O1D]/dt) / d[O3]
          J_adj_block(2, 1) = -1.2e-10*state(1) + 1.0*jvals[2];
          // J^T(O3, O3): d(d[O3]/dt) / d[O3]
          J_adj_block(2, 2) = -6.062e-15*state(3) - 1.576e-15*state(0) - 1.2e-10*state(1) - 1.0*jvals[1] - 1.0*jvals[2];
          // J^T(O3, NO): d(d[NO]/dt) / d[O3]
          J_adj_block(2, 3) = -6.062e-15*state(3);
          // J^T(O3, NO2): d(d[NO2]/dt) / d[O3]
          J_adj_block(2, 4) = 6.062e-15*state(3);
          // J^T(O3, O2): d(d[O2]/dt) / d[O3]
          J_adj_block(2, 6) = 6.062e-15*state(3) + 3.152e-15*state(0) + 2.4e-10*state(1) + 1.0*jvals[1] + 1.0*jvals[2];
          // J^T(NO, O3): d(d[O3]/dt) / d[NO]
          J_adj_block(3, 2) = -6.062e-15*state(2);
          // J^T(NO, NO): d(d[NO]/dt) / d[NO]
          J_adj_block(3, 3) = -6.062e-15*state(2);
          // J^T(NO, NO2): d(d[NO2]/dt) / d[NO]
          J_adj_block(3, 4) = 6.062e-15*state(2);
          // J^T(NO, O2): d(d[O2]/dt) / d[NO]
          J_adj_block(3, 6) = 6.062e-15*state(2);
          // J^T(NO2, O): d(d[O]/dt) / d[NO2]
          J_adj_block(4, 0) = -1.069e-11*state(0) + 1.0*jvals[3];
          // J^T(NO2, NO): d(d[NO]/dt) / d[NO2]
          J_adj_block(4, 3) = 1.069e-11*state(0) + 1.0*jvals[3];
          // J^T(NO2, NO2): d(d[NO2]/dt) / d[NO2]
          J_adj_block(4, 4) = -1.069e-11*state(0) - 1.0*jvals[3];
          // J^T(NO2, O2): d(d[O2]/dt) / d[NO2]
          J_adj_block(4, 6) = 1.069e-11*state(0);
          // J^T(M, O): d(d[O]/dt) / d[M]
          J_adj_block(5, 0) = 7.11e-11*state(1);
          // J^T(M, O1D): d(d[O1D]/dt) / d[M]
          J_adj_block(5, 1) = -7.11e-11*state(1);
          // J^T(O2, O): d(d[O]/dt) / d[O2]
          J_adj_block(6, 0) = -8.018e-17*state(0) + 2.0*jvals[0];
          // J^T(O2, O3): d(d[O3]/dt) / d[O2]
          J_adj_block(6, 2) = 8.018e-17*state(0);
          // J^T(O2, O2): d(d[O2]/dt) / d[O2]
          J_adj_block(6, 6) = -8.018e-17*state(0) - 1.0*jvals[0];
      }
#endif

#ifdef MKPP_ENABLE_ADJOINT
      template <class StateView, class DeltaView, class RateView>
      KOKKOS_INLINE_FUNCTION void compute_tlm(const StateView& state, const DeltaView& delta_C, RateView& dF_block, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
          dF_block(0) = 0.0;
          dF_block(0) += (-1.069e-11*state(4) - 8.018e-17*state(6) - 1.576e-15*state(2)) * delta_C(0);
          dF_block(0) += (7.11e-11*state(5)) * delta_C(1);
          dF_block(0) += (-1.576e-15*state(0) + 1.0*jvals[1]) * delta_C(2);
          dF_block(0) += (-1.069e-11*state(0) + 1.0*jvals[3]) * delta_C(4);
          dF_block(0) += (7.11e-11*state(1)) * delta_C(5);
          dF_block(0) += (-8.018e-17*state(0) + 2.0*jvals[0]) * delta_C(6);
          dF_block(1) = 0.0;
          dF_block(1) += (-7.11e-11*state(5) - 1.2e-10*state(2)) * delta_C(1);
          dF_block(1) += (-1.2e-10*state(1) + 1.0*jvals[2]) * delta_C(2);
          dF_block(1) += (-7.11e-11*state(1)) * delta_C(5);
          dF_block(2) = 0.0;
          dF_block(2) += (8.018e-17*state(6) - 1.576e-15*state(2)) * delta_C(0);
          dF_block(2) += (-1.2e-10*state(2)) * delta_C(1);
          dF_block(2) += (-6.062e-15*state(3) - 1.576e-15*state(0) - 1.2e-10*state(1) - 1.0*jvals[1] - 1.0*jvals[2]) * delta_C(2);
          dF_block(2) += (-6.062e-15*state(2)) * delta_C(3);
          dF_block(2) += (8.018e-17*state(0)) * delta_C(6);
          dF_block(3) = 0.0;
          dF_block(3) += (1.069e-11*state(4)) * delta_C(0);
          dF_block(3) += (-6.062e-15*state(3)) * delta_C(2);
          dF_block(3) += (-6.062e-15*state(2)) * delta_C(3);
          dF_block(3) += (1.069e-11*state(0) + 1.0*jvals[3]) * delta_C(4);
          dF_block(4) = 0.0;
          dF_block(4) += (-1.069e-11*state(4)) * delta_C(0);
          dF_block(4) += (6.062e-15*state(3)) * delta_C(2);
          dF_block(4) += (6.062e-15*state(2)) * delta_C(3);
          dF_block(4) += (-1.069e-11*state(0) - 1.0*jvals[3]) * delta_C(4);
          dF_block(5) = 0.0;
          dF_block(6) = 0.0;
          dF_block(6) += (1.069e-11*state(4) - 8.018e-17*state(6) + 3.152e-15*state(2)) * delta_C(0);
          dF_block(6) += (2.4e-10*state(2)) * delta_C(1);
          dF_block(6) += (6.062e-15*state(3) + 3.152e-15*state(0) + 2.4e-10*state(1) + 1.0*jvals[1] + 1.0*jvals[2]) * delta_C(2);
          dF_block(6) += (6.062e-15*state(2)) * delta_C(3);
          dF_block(6) += (1.069e-11*state(0)) * delta_C(4);
          dF_block(6) += (-8.018e-17*state(0) - 1.0*jvals[0]) * delta_C(6);
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
      }

      static constexpr int NUM_SPECIES = 7;
      static constexpr double atol[NUM_SPECIES] = { 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001 };
      static constexpr double rtol[NUM_SPECIES] = { 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06, 1e-06 };

      // Photolysis reactions (Cloud-J input mapping):
      //   jvals[0] = O2 -> O  (original A: 1.0)
      //   jvals[1] = O3 -> O, O2  (original A: 1.0)
      //   jvals[2] = O3 -> O1D, O2  (original A: 1.0)
      //   jvals[3] = NO2 -> NO, O  (original A: 1.0)
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
      KOKKOS_INLINE_FUNCTION void integrate(double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
          const int NUM_SPECIES = 7;
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
          const double S_0 = state(Species::M);  // [M]
          const double S_1 = state(Species::O1D);  // [O1D]
          const double S_2 = state(Species::O);  // [O]
          const double S_3 = state(Species::O2);  // [O2]
          const double S_4 = state(Species::O3);  // [O3]
          const double S_5 = state(Species::NO);  // [NO]
          const double S_6 = state(Species::NO2);  // [NO2]
          // The expression-dense Jacobian is evaluated by bounded compiled
          // units.  The solver still uses the same symbolic sparse LU plan.
#ifndef MKPP_USE_UNROLLED_REFERENCE
          double J_values[28] = {};
          auto jacobian_accessor = detail::StrideView<double, 2>::compact(J_values);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, 0.0, 0.0, 2.46e+19);
          double W_values[29];
          W_values[0] = inv_g_dt;
          W_values[1] = 0.0 - J_values[8];
          W_values[2] = inv_g_dt - J_values[6];
          W_values[3] = 0.0 - J_values[7];
          W_values[4] = 0.0 - J_values[4];
          W_values[5] = 0.0 - J_values[1];
          W_values[6] = inv_g_dt - J_values[0];
          W_values[7] = 0.0 - J_values[5];
          W_values[8] = 0.0 - J_values[2];
          W_values[9] = 0.0 - J_values[3];
          W_values[10] = 0.0 - J_values[23];
          W_values[11] = 0.0 - J_values[22];
          W_values[12] = inv_g_dt - J_values[27];
          W_values[13] = 0.0 - J_values[24];
          W_values[14] = 0.0 - J_values[25];
          W_values[15] = 0.0 - J_values[26];
          W_values[16] = 0.0 - J_values[10];
          W_values[17] = 0.0 - J_values[9];
          W_values[18] = 0.0 - J_values[13];
          W_values[19] = inv_g_dt - J_values[11];
          W_values[20] = 0.0 - J_values[12];
          W_values[21] = 0.0 - J_values[14];
          W_values[22] = 0.0 - J_values[15];
          W_values[23] = inv_g_dt - J_values[16];
          W_values[24] = 0.0 - J_values[17];
          W_values[25] = 0.0 - J_values[18];
          W_values[26] = 0.0 - J_values[19];
          W_values[27] = 0.0 - J_values[20];
          W_values[28] = inv_g_dt - J_values[21];
          double LU_values[32];
          detail::factorize_plan(W_values, LU_values);
#else
          double J_values[NUM_SPECIES * NUM_SPECIES] = {};
          auto jacobian_accessor = detail::make_stride_view<double, 2>(J_values, NUM_SPECIES, 1);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, 0.0, 0.0, 2.46e+19);
          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_1_0 = J_values[1 * NUM_SPECIES + 5];
          double J_1_1 = J_values[1 * NUM_SPECIES + 1];
          double J_1_4 = J_values[1 * NUM_SPECIES + 2];
          double J_2_0 = J_values[0 * NUM_SPECIES + 5];
          double J_2_1 = J_values[0 * NUM_SPECIES + 1];
          double J_2_2 = J_values[0 * NUM_SPECIES + 0];
          double J_2_3 = J_values[0 * NUM_SPECIES + 6];
          double J_2_4 = J_values[0 * NUM_SPECIES + 2];
          double J_2_6 = J_values[0 * NUM_SPECIES + 4];
          double J_3_1 = J_values[6 * NUM_SPECIES + 1];
          double J_3_2 = J_values[6 * NUM_SPECIES + 0];
          double J_3_3 = J_values[6 * NUM_SPECIES + 6];
          double J_3_4 = J_values[6 * NUM_SPECIES + 2];
          double J_3_5 = J_values[6 * NUM_SPECIES + 3];
          double J_3_6 = J_values[6 * NUM_SPECIES + 4];
          double J_4_1 = J_values[2 * NUM_SPECIES + 1];
          double J_4_2 = J_values[2 * NUM_SPECIES + 0];
          double J_4_3 = J_values[2 * NUM_SPECIES + 6];
          double J_4_4 = J_values[2 * NUM_SPECIES + 2];
          double J_4_5 = J_values[2 * NUM_SPECIES + 3];
          double J_5_2 = J_values[3 * NUM_SPECIES + 0];
          double J_5_4 = J_values[3 * NUM_SPECIES + 2];
          double J_5_5 = J_values[3 * NUM_SPECIES + 3];
          double J_5_6 = J_values[3 * NUM_SPECIES + 4];
          double J_6_2 = J_values[4 * NUM_SPECIES + 0];
          double J_6_4 = J_values[4 * NUM_SPECIES + 2];
          double J_6_5 = J_values[4 * NUM_SPECIES + 3];
          double J_6_6 = J_values[4 * NUM_SPECIES + 4];
          double W_values[NUM_SPECIES * NUM_SPECIES] = {};
          W_values[0 * NUM_SPECIES + 0] = inv_g_dt;
          W_values[1 * NUM_SPECIES + 0] = -J_1_0;
          W_values[1 * NUM_SPECIES + 1] = inv_g_dt - J_1_1;
          W_values[1 * NUM_SPECIES + 4] = -J_1_4;
          W_values[2 * NUM_SPECIES + 0] = -J_2_0;
          W_values[2 * NUM_SPECIES + 1] = -J_2_1;
          W_values[2 * NUM_SPECIES + 2] = inv_g_dt - J_2_2;
          W_values[2 * NUM_SPECIES + 3] = -J_2_3;
          W_values[2 * NUM_SPECIES + 4] = -J_2_4;
          W_values[2 * NUM_SPECIES + 6] = -J_2_6;
          W_values[3 * NUM_SPECIES + 1] = -J_3_1;
          W_values[3 * NUM_SPECIES + 2] = -J_3_2;
          W_values[3 * NUM_SPECIES + 3] = inv_g_dt - J_3_3;
          W_values[3 * NUM_SPECIES + 4] = -J_3_4;
          W_values[3 * NUM_SPECIES + 5] = -J_3_5;
          W_values[3 * NUM_SPECIES + 6] = -J_3_6;
          W_values[4 * NUM_SPECIES + 1] = -J_4_1;
          W_values[4 * NUM_SPECIES + 2] = -J_4_2;
          W_values[4 * NUM_SPECIES + 3] = -J_4_3;
          W_values[4 * NUM_SPECIES + 4] = inv_g_dt - J_4_4;
          W_values[4 * NUM_SPECIES + 5] = -J_4_5;
          W_values[5 * NUM_SPECIES + 2] = -J_5_2;
          W_values[5 * NUM_SPECIES + 4] = -J_5_4;
          W_values[5 * NUM_SPECIES + 5] = inv_g_dt - J_5_5;
          W_values[5 * NUM_SPECIES + 6] = -J_5_6;
          W_values[6 * NUM_SPECIES + 2] = -J_6_2;
          W_values[6 * NUM_SPECIES + 4] = -J_6_4;
          W_values[6 * NUM_SPECIES + 5] = -J_6_5;
          W_values[6 * NUM_SPECIES + 6] = inv_g_dt - J_6_6;
          double LU_values[NUM_SPECIES * NUM_SPECIES] = {};
          detail::factorize_lu_chunk_0(W_values, LU_values);
#endif

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F_values_1[NUM_SPECIES];
          auto rate_accessor_1 = detail::make_stride_view<double, 1>(F_values_1, 1);
          detail::compute_rates_chunk_0(state_accessor, rate_accessor_1, jvals, 0.0, 0.0, 2.46e+19);
          double F1_0 = F_values_1[5];
          double F1_1 = F_values_1[1];
          double F1_2 = F_values_1[0];
          double F1_3 = F_values_1[6];
          double F1_4 = F_values_1[2];
          double F1_5 = F_values_1[3];
          double F1_6 = F_values_1[4];
          double rhs_values_1[NUM_SPECIES];
          rhs_values_1[0] = F1_0;
          rhs_values_1[1] = F1_1;
          rhs_values_1[2] = F1_2;
          rhs_values_1[3] = F1_3;
          rhs_values_1[4] = F1_4;
          rhs_values_1[5] = F1_5;
          rhs_values_1[6] = F1_6;
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

          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          double Y2_4 = S_4 + K1_4;
          double Y2_5 = S_5 + K1_5;
          double Y2_6 = S_6 + K1_6;
          // Rate evaluation F2 at Y2
          double stage_state_2[NUM_SPECIES];
          stage_state_2[5] = Y2_0;
          stage_state_2[1] = Y2_1;
          stage_state_2[0] = Y2_2;
          stage_state_2[6] = Y2_3;
          stage_state_2[2] = Y2_4;
          stage_state_2[3] = Y2_5;
          stage_state_2[4] = Y2_6;
          double F_values_2[NUM_SPECIES];
          auto rate_accessor_2 = detail::make_stride_view<double, 1>(F_values_2, 1);
          auto stage_state_accessor_2 = detail::make_stride_view<double, 1>(stage_state_2, 1);
          detail::compute_rates_chunk_0(stage_state_accessor_2, rate_accessor_2, jvals, 0.0, 0.0, 2.46e+19);
          double F2_0 = F_values_2[5];
          double F2_1 = F_values_2[1];
          double F2_2 = F_values_2[0];
          double F2_3 = F_values_2[6];
          double F2_4 = F_values_2[2];
          double F2_5 = F_values_2[3];
          double F2_6 = F_values_2[4];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          double rhs_values_2[NUM_SPECIES];
          rhs_values_2[0] = rhs2_0;
          rhs_values_2[1] = rhs2_1;
          rhs_values_2[2] = rhs2_2;
          rhs_values_2[3] = rhs2_3;
          rhs_values_2[4] = rhs2_4;
          rhs_values_2[5] = rhs2_5;
          rhs_values_2[6] = rhs2_6;
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

          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          double Y3_4 = S_4 + K1_4;
          double Y3_5 = S_5 + K1_5;
          double Y3_6 = S_6 + K1_6;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          double rhs_values_3[NUM_SPECIES];
          rhs_values_3[0] = rhs3_0;
          rhs_values_3[1] = rhs3_1;
          rhs_values_3[2] = rhs3_2;
          rhs_values_3[3] = rhs3_3;
          rhs_values_3[4] = rhs3_4;
          rhs_values_3[5] = rhs3_5;
          rhs_values_3[6] = rhs3_6;
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

          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 7);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(5) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(1) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(6) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(2) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(3) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(4) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
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
          double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view, double importance_threshold) const
      {
          const double* jvals = jvals_view.data();
          const int NUM_SPECIES = 7;
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

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // 0. Hoist state values into scalar registers
          // NOTE: State access uses permuted species ordering
          const double S_0 = state(5);
          const double S_1 = state(1);
          const double S_2 = state(0);
          const double S_3 = state(6);
          const double S_4 = state(2);
          const double S_5 = state(3);
          const double S_6 = state(4);

          // 1. Stage 1 Rates (F1)
          double F1_0 = 0.0;
          double F1_1 = -7.11e-11*S_0*S_1 - 1.2e-10*S_1*S_4 + 1.0*S_4*jvals[2];
          double F1_2 = 7.11e-11*S_0*S_1 - 1.069e-11*S_6*S_2 + 1.0*S_6*jvals[3] - 8.018e-17*S_2*S_3 - 1.576e-15*S_2*S_4 + 2.0*S_3*jvals[0] + 1.0*S_4*jvals[1];
          double F1_3 = 6.062e-15*S_5*S_4 + 1.069e-11*S_6*S_2 - 8.018e-17*S_2*S_3 + 3.152e-15*S_2*S_4 + 2.4e-10*S_1*S_4 - 1.0*S_3*jvals[0] + 1.0*S_4*jvals[1] + 1.0*S_4*jvals[2];
          double F1_4 = -6.062e-15*S_5*S_4 + 8.018e-17*S_2*S_3 - 1.576e-15*S_2*S_4 - 1.2e-10*S_1*S_4 - 1.0*S_4*jvals[1] - 1.0*S_4*jvals[2];
          double F1_5 = -6.062e-15*S_5*S_4 + 1.069e-11*S_6*S_2 + 1.0*S_6*jvals[3];
          double F1_6 = 6.062e-15*S_5*S_4 - 1.069e-11*S_6*S_2 - 1.0*S_6*jvals[3];

          // 2. Evaluate importance and update active set
          active[0] = (Kokkos::fabs(F1_0) / (atol[0] + rtol[0] * Kokkos::fabs(state(5))) >= importance_threshold);
          active[1] = (Kokkos::fabs(F1_1) / (atol[1] + rtol[1] * Kokkos::fabs(state(1))) >= importance_threshold);
          active[2] = (Kokkos::fabs(F1_2) / (atol[2] + rtol[2] * Kokkos::fabs(state(0))) >= importance_threshold);
          active[3] = (Kokkos::fabs(F1_3) / (atol[3] + rtol[3] * Kokkos::fabs(state(6))) >= importance_threshold);
          active[4] = (Kokkos::fabs(F1_4) / (atol[4] + rtol[4] * Kokkos::fabs(state(2))) >= importance_threshold);
          active[5] = (Kokkos::fabs(F1_5) / (atol[5] + rtol[5] * Kokkos::fabs(state(3))) >= importance_threshold);
          active[6] = (Kokkos::fabs(F1_6) / (atol[6] + rtol[6] * Kokkos::fabs(state(4))) >= importance_threshold);

          // 3. Analytical Jacobian & Iteration Matrix W (identity for frozen species)
          double J_1_0 = -7.11e-11*S_1;
          double J_1_1 = -7.11e-11*S_0 - 1.2e-10*S_4;
          double J_1_4 = -1.2e-10*S_1 + 1.0*jvals[2];
          double J_2_0 = 7.11e-11*S_1;
          double J_2_1 = 7.11e-11*S_0;
          double J_2_2 = -1.069e-11*S_6 - 8.018e-17*S_3 - 1.576e-15*S_4;
          double J_2_3 = -8.018e-17*S_2 + 2.0*jvals[0];
          double J_2_4 = -1.576e-15*S_2 + 1.0*jvals[1];
          double J_2_6 = -1.069e-11*S_2 + 1.0*jvals[3];
          double J_3_1 = 2.4e-10*S_4;
          double J_3_2 = 1.069e-11*S_6 - 8.018e-17*S_3 + 3.152e-15*S_4;
          double J_3_3 = -8.018e-17*S_2 - 1.0*jvals[0];
          double J_3_4 = 6.062e-15*S_5 + 3.152e-15*S_2 + 2.4e-10*S_1 + 1.0*jvals[1] + 1.0*jvals[2];
          double J_3_5 = 6.062e-15*S_4;
          double J_3_6 = 1.069e-11*S_2;
          double J_4_1 = -1.2e-10*S_4;
          double J_4_2 = 8.018e-17*S_3 - 1.576e-15*S_4;
          double J_4_3 = 8.018e-17*S_2;
          double J_4_4 = -6.062e-15*S_5 - 1.576e-15*S_2 - 1.2e-10*S_1 - 1.0*jvals[1] - 1.0*jvals[2];
          double J_4_5 = -6.062e-15*S_4;
          double J_5_2 = 1.069e-11*S_6;
          double J_5_4 = -6.062e-15*S_5;
          double J_5_5 = -6.062e-15*S_4;
          double J_5_6 = 1.069e-11*S_2 + 1.0*jvals[3];
          double J_6_2 = -1.069e-11*S_6;
          double J_6_4 = 6.062e-15*S_5;
          double J_6_5 = 6.062e-15*S_4;
          double J_6_6 = -1.069e-11*S_2 - 1.0*jvals[3];
          double W_0_0 = active[0] ? inv_g_dt : 1.0;
          double W_1_0 = (active[1] && active[0]) ? (-J_1_0) : 0.0;
          double W_1_1 = active[1] ? (inv_g_dt - J_1_1) : 1.0;
          double W_1_4 = (active[1] && active[4]) ? (-J_1_4) : 0.0;
          double W_2_0 = (active[2] && active[0]) ? (-J_2_0) : 0.0;
          double W_2_1 = (active[2] && active[1]) ? (-J_2_1) : 0.0;
          double W_2_2 = active[2] ? (inv_g_dt - J_2_2) : 1.0;
          double W_2_3 = (active[2] && active[3]) ? (-J_2_3) : 0.0;
          double W_2_4 = (active[2] && active[4]) ? (-J_2_4) : 0.0;
          double W_2_6 = (active[2] && active[6]) ? (-J_2_6) : 0.0;
          double W_3_1 = (active[3] && active[1]) ? (-J_3_1) : 0.0;
          double W_3_2 = (active[3] && active[2]) ? (-J_3_2) : 0.0;
          double W_3_3 = active[3] ? (inv_g_dt - J_3_3) : 1.0;
          double W_3_4 = (active[3] && active[4]) ? (-J_3_4) : 0.0;
          double W_3_5 = (active[3] && active[5]) ? (-J_3_5) : 0.0;
          double W_3_6 = (active[3] && active[6]) ? (-J_3_6) : 0.0;
          double W_4_1 = (active[4] && active[1]) ? (-J_4_1) : 0.0;
          double W_4_2 = (active[4] && active[2]) ? (-J_4_2) : 0.0;
          double W_4_3 = (active[4] && active[3]) ? (-J_4_3) : 0.0;
          double W_4_4 = active[4] ? (inv_g_dt - J_4_4) : 1.0;
          double W_4_5 = (active[4] && active[5]) ? (-J_4_5) : 0.0;
          double W_5_2 = (active[5] && active[2]) ? (-J_5_2) : 0.0;
          double W_5_4 = (active[5] && active[4]) ? (-J_5_4) : 0.0;
          double W_5_5 = active[5] ? (inv_g_dt - J_5_5) : 1.0;
          double W_5_6 = (active[5] && active[6]) ? (-J_5_6) : 0.0;
          double W_6_2 = (active[6] && active[2]) ? (-J_6_2) : 0.0;
          double W_6_4 = (active[6] && active[4]) ? (-J_6_4) : 0.0;
          double W_6_5 = (active[6] && active[5]) ? (-J_6_5) : 0.0;
          double W_6_6 = active[6] ? (inv_g_dt - J_6_6) : 1.0;

          // 4. Symbolic LU Factorization (conditional skip for frozen species)
          double U_0_0 = W_0_0;
          double L_1_0 = (W_1_0) / U_0_0;
          double L_2_0 = (W_2_0) / U_0_0;
          double U_1_1 = W_1_1;
          double U_1_4 = W_1_4;
          double L_2_1 = (W_2_1) / U_1_1;
          double L_3_1 = (W_3_1) / U_1_1;
          double L_4_1 = (W_4_1) / U_1_1;
          double U_2_2 = W_2_2;
          double U_2_3 = W_2_3;
          double U_2_4 = W_2_4 - L_2_1 * U_1_4;
          double U_2_6 = W_2_6;
          double L_3_2 = (W_3_2) / U_2_2;
          double L_4_2 = (W_4_2) / U_2_2;
          double L_5_2 = (W_5_2) / U_2_2;
          double L_6_2 = (W_6_2) / U_2_2;
          double U_3_3 = W_3_3 - L_3_2 * U_2_3;
          double U_3_4 = W_3_4 - L_3_1 * U_1_4 - L_3_2 * U_2_4;
          double U_3_5 = W_3_5;
          double U_3_6 = W_3_6 - L_3_2 * U_2_6;
          double L_4_3 = (W_4_3 - L_4_2 * U_2_3) / U_3_3;
          double L_5_3 = (0.0 - L_5_2 * U_2_3) / U_3_3;
          double L_6_3 = (0.0 - L_6_2 * U_2_3) / U_3_3;
          double U_4_4 = W_4_4 - L_4_1 * U_1_4 - L_4_2 * U_2_4 - L_4_3 * U_3_4;
          double U_4_5 = W_4_5 - L_4_3 * U_3_5;
          double U_4_6 = 0.0 - L_4_2 * U_2_6 - L_4_3 * U_3_6;
          double L_5_4 = (W_5_4 - L_5_2 * U_2_4 - L_5_3 * U_3_4) / U_4_4;
          double L_6_4 = (W_6_4 - L_6_2 * U_2_4 - L_6_3 * U_3_4) / U_4_4;
          double U_5_5 = W_5_5 - L_5_3 * U_3_5 - L_5_4 * U_4_5;
          double U_5_6 = W_5_6 - L_5_2 * U_2_6 - L_5_3 * U_3_6 - L_5_4 * U_4_6;
          double L_6_5 = (W_6_5 - L_6_3 * U_3_5 - L_6_4 * U_4_5) / U_5_5;
          double U_6_6 = W_6_6 - L_6_2 * U_2_6 - L_6_3 * U_3_6 - L_6_4 * U_4_6 - L_6_5 * U_5_6;

          // --- Stage 1 ---
          // F1 already computed above (used for importance evaluation)
          // Block 0: K1 forward sub [M, O1D, O, O2, O3, NO2]
          double y1_0 = active[0] ? (F1_0) : 0.0;
          double y1_1 = active[1] ? (F1_1 - L_1_0 * y1_0) : 0.0;
          double y1_2 = active[2] ? (F1_2 - L_2_0 * y1_0 - L_2_1 * y1_1) : 0.0;
          double y1_3 = active[3] ? (F1_3 - L_3_1 * y1_1 - L_3_2 * y1_2) : 0.0;
          double y1_4 = active[4] ? (F1_4 - L_4_1 * y1_1 - L_4_2 * y1_2 - L_4_3 * y1_3) : 0.0;
          // Block 1: K1 forward sub [NO]
          double y1_5 = active[5] ? (F1_5 - L_5_2 * y1_2 - L_5_3 * y1_3 - L_5_4 * y1_4) : 0.0;
          double y1_6 = active[6] ? (F1_6 - L_6_2 * y1_2 - L_6_3 * y1_3 - L_6_4 * y1_4 - L_6_5 * y1_5) : 0.0;

          // Block 0: K1 backward sub [M, O1D, O, O2, O3, NO2]
          double K1_6 = active[6] ? (y1_6 / U_6_6) : 0.0;
          // Block 1: K1 backward sub [NO]
          double K1_5 = active[5] ? ((y1_5 - U_5_6 * K1_6) / U_5_5) : 0.0;
          double K1_4 = active[4] ? ((y1_4 - U_4_5 * K1_5 - U_4_6 * K1_6) / U_4_4) : 0.0;
          double K1_3 = active[3] ? ((y1_3 - U_3_4 * K1_4 - U_3_5 * K1_5 - U_3_6 * K1_6) / U_3_3) : 0.0;
          double K1_2 = active[2] ? ((y1_2 - U_2_3 * K1_3 - U_2_4 * K1_4 - U_2_6 * K1_6) / U_2_2) : 0.0;
          double K1_1 = active[1] ? ((y1_1 - U_1_4 * K1_4) / U_1_1) : 0.0;
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
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = -7.11e-11*Y2_0*Y2_1 - 1.2e-10*Y2_1*Y2_4 + 1.0*Y2_4*jvals[2];
          double F2_2 = 7.11e-11*Y2_0*Y2_1 - 1.069e-11*Y2_6*Y2_2 + 1.0*Y2_6*jvals[3] - 8.018e-17*Y2_2*Y2_3 - 1.576e-15*Y2_2*Y2_4 + 2.0*Y2_3*jvals[0] + 1.0*Y2_4*jvals[1];
          double F2_3 = 6.062e-15*Y2_5*Y2_4 + 1.069e-11*Y2_6*Y2_2 - 8.018e-17*Y2_2*Y2_3 + 3.152e-15*Y2_2*Y2_4 + 2.4e-10*Y2_1*Y2_4 - 1.0*Y2_3*jvals[0] + 1.0*Y2_4*jvals[1] + 1.0*Y2_4*jvals[2];
          double F2_4 = -6.062e-15*Y2_5*Y2_4 + 8.018e-17*Y2_2*Y2_3 - 1.576e-15*Y2_2*Y2_4 - 1.2e-10*Y2_1*Y2_4 - 1.0*Y2_4*jvals[1] - 1.0*Y2_4*jvals[2];
          double F2_5 = -6.062e-15*Y2_5*Y2_4 + 1.069e-11*Y2_6*Y2_2 + 1.0*Y2_6*jvals[3];
          double F2_6 = 6.062e-15*Y2_5*Y2_4 - 1.069e-11*Y2_6*Y2_2 - 1.0*Y2_6*jvals[3];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          // Block 0: K2 forward sub [M, O1D, O, O2, O3, NO2]
          double y2_0 = active[0] ? (rhs2_0) : 0.0;
          double y2_1 = active[1] ? (rhs2_1 - L_1_0 * y2_0) : 0.0;
          double y2_2 = active[2] ? (rhs2_2 - L_2_0 * y2_0 - L_2_1 * y2_1) : 0.0;
          double y2_3 = active[3] ? (rhs2_3 - L_3_1 * y2_1 - L_3_2 * y2_2) : 0.0;
          double y2_4 = active[4] ? (rhs2_4 - L_4_1 * y2_1 - L_4_2 * y2_2 - L_4_3 * y2_3) : 0.0;
          // Block 1: K2 forward sub [NO]
          double y2_5 = active[5] ? (rhs2_5 - L_5_2 * y2_2 - L_5_3 * y2_3 - L_5_4 * y2_4) : 0.0;
          double y2_6 = active[6] ? (rhs2_6 - L_6_2 * y2_2 - L_6_3 * y2_3 - L_6_4 * y2_4 - L_6_5 * y2_5) : 0.0;

          // Block 0: K2 backward sub [M, O1D, O, O2, O3, NO2]
          double K2_6 = active[6] ? (y2_6 / U_6_6) : 0.0;
          // Block 1: K2 backward sub [NO]
          double K2_5 = active[5] ? ((y2_5 - U_5_6 * K2_6) / U_5_5) : 0.0;
          double K2_4 = active[4] ? ((y2_4 - U_4_5 * K2_5 - U_4_6 * K2_6) / U_4_4) : 0.0;
          double K2_3 = active[3] ? ((y2_3 - U_3_4 * K2_4 - U_3_5 * K2_5 - U_3_6 * K2_6) / U_3_3) : 0.0;
          double K2_2 = active[2] ? ((y2_2 - U_2_3 * K2_3 - U_2_4 * K2_4 - U_2_6 * K2_6) / U_2_2) : 0.0;
          double K2_1 = active[1] ? ((y2_1 - U_1_4 * K2_4) / U_1_1) : 0.0;
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
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          // Block 0: K3 forward sub [M, O1D, O, O2, O3, NO2]
          double y3_0 = active[0] ? (rhs3_0) : 0.0;
          double y3_1 = active[1] ? (rhs3_1 - L_1_0 * y3_0) : 0.0;
          double y3_2 = active[2] ? (rhs3_2 - L_2_0 * y3_0 - L_2_1 * y3_1) : 0.0;
          double y3_3 = active[3] ? (rhs3_3 - L_3_1 * y3_1 - L_3_2 * y3_2) : 0.0;
          double y3_4 = active[4] ? (rhs3_4 - L_4_1 * y3_1 - L_4_2 * y3_2 - L_4_3 * y3_3) : 0.0;
          // Block 1: K3 forward sub [NO]
          double y3_5 = active[5] ? (rhs3_5 - L_5_2 * y3_2 - L_5_3 * y3_3 - L_5_4 * y3_4) : 0.0;
          double y3_6 = active[6] ? (rhs3_6 - L_6_2 * y3_2 - L_6_3 * y3_3 - L_6_4 * y3_4 - L_6_5 * y3_5) : 0.0;

          // Block 0: K3 backward sub [M, O1D, O, O2, O3, NO2]
          double K3_6 = active[6] ? (y3_6 / U_6_6) : 0.0;
          // Block 1: K3 backward sub [NO]
          double K3_5 = active[5] ? ((y3_5 - U_5_6 * K3_6) / U_5_5) : 0.0;
          double K3_4 = active[4] ? ((y3_4 - U_4_5 * K3_5 - U_4_6 * K3_6) / U_4_4) : 0.0;
          double K3_3 = active[3] ? ((y3_3 - U_3_4 * K3_4 - U_3_5 * K3_5 - U_3_6 * K3_6) / U_3_3) : 0.0;
          double K3_2 = active[2] ? ((y3_2 - U_2_3 * K3_3 - U_2_4 * K3_4 - U_2_6 * K3_6) / U_2_2) : 0.0;
          double K3_1 = active[1] ? ((y3_1 - U_1_4 * K3_4) / U_1_1) : 0.0;
          double K3_0 = active[0] ? (y3_0 / U_0_0) : 0.0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 7);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              if (active[0]) state(5) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              if (active[1]) state(1) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              if (active[2]) state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              if (active[3]) state(6) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              if (active[4]) state(2) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              if (active[5]) state(3) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              if (active[6]) state(4) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
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
          CheckpointBuffer& chk) const
      {
          const double* jvals = jvals_view.data();
          const int NUM_SPECIES = 7;
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
          const double S_0 = state(Species::M);  // [M]
          const double S_1 = state(Species::O1D);  // [O1D]
          const double S_2 = state(Species::O);  // [O]
          const double S_3 = state(Species::O2);  // [O2]
          const double S_4 = state(Species::O3);  // [O3]
          const double S_5 = state(Species::NO);  // [NO]
          const double S_6 = state(Species::NO2);  // [NO2]

          // --- Reaction Rate Fluxes R_m ---
          const double R_0 = S_6*jvals[0];
          const double R_1 = 8.018e-17*S_0*S_6;
          const double R_2 = S_2*jvals[1];
          const double R_3 = 1.576e-15*S_0*S_2;
          const double R_4 = S_2*jvals[2];
          const double R_5 = 7.11e-11*S_5*S_1;
          const double R_6 = 1.2e-10*S_1*S_2;
          const double R_7 = 6.062e-15*S_3*S_2;
          const double R_8 = 1.069e-11*S_4*S_0;
          const double R_9 = S_4*jvals[3];

          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_1_0 = -7.11e-11*S_1;
          double J_1_1 = -7.11e-11*S_0 - 1.2e-10*S_4;
          double J_1_4 = -1.2e-10*S_1 + 1.0*jvals[2];
          double J_2_0 = 7.11e-11*S_1;
          double J_2_1 = 7.11e-11*S_0;
          double J_2_2 = -1.069e-11*S_6 - 8.018e-17*S_3 - 1.576e-15*S_4;
          double J_2_3 = -8.018e-17*S_2 + 2.0*jvals[0];
          double J_2_4 = -1.576e-15*S_2 + 1.0*jvals[1];
          double J_2_6 = -1.069e-11*S_2 + 1.0*jvals[3];
          double J_3_1 = 2.4e-10*S_4;
          double J_3_2 = 1.069e-11*S_6 - 8.018e-17*S_3 + 3.152e-15*S_4;
          double J_3_3 = -8.018e-17*S_2 - 1.0*jvals[0];
          double J_3_4 = 6.062e-15*S_5 + 3.152e-15*S_2 + 2.4e-10*S_1 + 1.0*jvals[1] + 1.0*jvals[2];
          double J_3_5 = 6.062e-15*S_4;
          double J_3_6 = 1.069e-11*S_2;
          double J_4_1 = -1.2e-10*S_4;
          double J_4_2 = 8.018e-17*S_3 - 1.576e-15*S_4;
          double J_4_3 = 8.018e-17*S_2;
          double J_4_4 = -6.062e-15*S_5 - 1.576e-15*S_2 - 1.2e-10*S_1 - 1.0*jvals[1] - 1.0*jvals[2];
          double J_4_5 = -6.062e-15*S_4;
          double J_5_2 = 1.069e-11*S_6;
          double J_5_4 = -6.062e-15*S_5;
          double J_5_5 = -6.062e-15*S_4;
          double J_5_6 = 1.069e-11*S_2 + 1.0*jvals[3];
          double J_6_2 = -1.069e-11*S_6;
          double J_6_4 = 6.062e-15*S_5;
          double J_6_5 = 6.062e-15*S_4;
          double J_6_6 = -1.069e-11*S_2 - 1.0*jvals[3];
          // Block 0: species [M, O1D, O, O2, O3, NO2]
          double W_0_0 = inv_g_dt;
          double W_1_0 = -J_1_0;
          double W_1_1 = inv_g_dt - J_1_1;
          double W_1_4 = -J_1_4;
          double W_2_0 = -J_2_0;
          double W_2_1 = -J_2_1;
          double W_2_2 = inv_g_dt - J_2_2;
          double W_2_3 = -J_2_3;
          double W_2_4 = -J_2_4;
          double W_2_6 = -J_2_6;
          double W_3_1 = -J_3_1;
          double W_3_2 = -J_3_2;
          double W_3_3 = inv_g_dt - J_3_3;
          double W_3_4 = -J_3_4;
          double W_3_5 = -J_3_5;
          double W_3_6 = -J_3_6;
          double W_4_1 = -J_4_1;
          double W_4_2 = -J_4_2;
          double W_4_3 = -J_4_3;
          double W_4_4 = inv_g_dt - J_4_4;
          double W_4_5 = -J_4_5;
          // Block 1: species [NO]
          double W_5_2 = -J_5_2;
          double W_5_4 = -J_5_4;
          double W_5_5 = inv_g_dt - J_5_5;
          double W_5_6 = -J_5_6;
          // Block 0: species [M, O1D, O, O2, O3, NO2]
          double W_6_2 = -J_6_2;
          double W_6_4 = -J_6_4;
          double W_6_5 = -J_6_5;
          double W_6_6 = inv_g_dt - J_6_6;

          // Symbolic Doolittle Sparse LU Factorization
          // Block 0: species [M, O1D, O, O2, O3, NO2]
          double U_0_0 = W_0_0;
          double L_1_0 = (W_1_0) / U_0_0;
          double L_2_0 = (W_2_0) / U_0_0;
          double U_1_1 = W_1_1;
          double U_1_4 = W_1_4;
          double L_2_1 = (W_2_1) / U_1_1;
          double L_3_1 = (W_3_1) / U_1_1;
          double L_4_1 = (W_4_1) / U_1_1;
          double U_2_2 = W_2_2;
          double U_2_3 = W_2_3;
          double U_2_4 = W_2_4 - L_2_1 * U_1_4;
          double U_2_6 = W_2_6;
          double L_3_2 = (W_3_2) / U_2_2;
          double L_4_2 = (W_4_2) / U_2_2;
          // Block 1: species [NO]
          double L_5_2 = (W_5_2) / U_2_2;
          double L_6_2 = (W_6_2) / U_2_2;
          double U_3_3 = W_3_3 - L_3_2 * U_2_3;
          double U_3_4 = W_3_4 - L_3_1 * U_1_4 - L_3_2 * U_2_4;
          double U_3_5 = W_3_5;
          double U_3_6 = W_3_6 - L_3_2 * U_2_6;
          double L_4_3 = (W_4_3 - L_4_2 * U_2_3) / U_3_3;
          double L_5_3 = (0.0 - L_5_2 * U_2_3) / U_3_3;
          double L_6_3 = (0.0 - L_6_2 * U_2_3) / U_3_3;
          double U_4_4 = W_4_4 - L_4_1 * U_1_4 - L_4_2 * U_2_4 - L_4_3 * U_3_4;
          double U_4_5 = W_4_5 - L_4_3 * U_3_5;
          double U_4_6 = 0.0 - L_4_2 * U_2_6 - L_4_3 * U_3_6;
          double L_5_4 = (W_5_4 - L_5_2 * U_2_4 - L_5_3 * U_3_4) / U_4_4;
          double L_6_4 = (W_6_4 - L_6_2 * U_2_4 - L_6_3 * U_3_4) / U_4_4;
          double U_5_5 = W_5_5 - L_5_3 * U_3_5 - L_5_4 * U_4_5;
          double U_5_6 = W_5_6 - L_5_2 * U_2_6 - L_5_3 * U_3_6 - L_5_4 * U_4_6;
          double L_6_5 = (W_6_5 - L_6_3 * U_3_5 - L_6_4 * U_4_5) / U_5_5;
          double U_6_6 = W_6_6 - L_6_2 * U_2_6 - L_6_3 * U_3_6 - L_6_4 * U_4_6 - L_6_5 * U_5_6;

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F1_0 = 0.0;
          double F1_1 = -7.11e-11*S_0*S_1 - 1.2e-10*S_1*S_4 + 1.0*S_4*jvals[2];
          double F1_2 = 7.11e-11*S_0*S_1 - 1.069e-11*S_6*S_2 + 1.0*S_6*jvals[3] - 8.018e-17*S_2*S_3 - 1.576e-15*S_2*S_4 + 2.0*S_3*jvals[0] + 1.0*S_4*jvals[1];
          double F1_3 = 6.062e-15*S_5*S_4 + 1.069e-11*S_6*S_2 - 8.018e-17*S_2*S_3 + 3.152e-15*S_2*S_4 + 2.4e-10*S_1*S_4 - 1.0*S_3*jvals[0] + 1.0*S_4*jvals[1] + 1.0*S_4*jvals[2];
          double F1_4 = -6.062e-15*S_5*S_4 + 8.018e-17*S_2*S_3 - 1.576e-15*S_2*S_4 - 1.2e-10*S_1*S_4 - 1.0*S_4*jvals[1] - 1.0*S_4*jvals[2];
          double F1_5 = -6.062e-15*S_5*S_4 + 1.069e-11*S_6*S_2 + 1.0*S_6*jvals[3];
          double F1_6 = 6.062e-15*S_5*S_4 - 1.069e-11*S_6*S_2 - 1.0*S_6*jvals[3];
          // Block 0: K1 forward sub [M, O1D, O, O2, O3, NO2]
          double y1_0 = F1_0;
          double y1_1 = F1_1 - L_1_0 * y1_0;
          double y1_2 = F1_2 - L_2_0 * y1_0 - L_2_1 * y1_1;
          double y1_3 = F1_3 - L_3_1 * y1_1 - L_3_2 * y1_2;
          double y1_4 = F1_4 - L_4_1 * y1_1 - L_4_2 * y1_2 - L_4_3 * y1_3;
          // Block 1: K1 forward sub [NO]
          double y1_5 = F1_5 - L_5_2 * y1_2 - L_5_3 * y1_3 - L_5_4 * y1_4;
          double y1_6 = F1_6 - L_6_2 * y1_2 - L_6_3 * y1_3 - L_6_4 * y1_4 - L_6_5 * y1_5;

          // Block 0: K1 backward sub [M, O1D, O, O2, O3, NO2]
          double K1_6 = y1_6 / U_6_6;
          // Block 1: K1 backward sub [NO]
          double K1_5 = (y1_5 - U_5_6 * K1_6) / U_5_5;
          double K1_4 = (y1_4 - U_4_5 * K1_5 - U_4_6 * K1_6) / U_4_4;
          double K1_3 = (y1_3 - U_3_4 * K1_4 - U_3_5 * K1_5 - U_3_6 * K1_6) / U_3_3;
          double K1_2 = (y1_2 - U_2_3 * K1_3 - U_2_4 * K1_4 - U_2_6 * K1_6) / U_2_2;
          double K1_1 = (y1_1 - U_1_4 * K1_4) / U_1_1;
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
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = -7.11e-11*Y2_0*Y2_1 - 1.2e-10*Y2_1*Y2_4 + 1.0*Y2_4*jvals[2];
          double F2_2 = 7.11e-11*Y2_0*Y2_1 - 1.069e-11*Y2_6*Y2_2 + 1.0*Y2_6*jvals[3] - 8.018e-17*Y2_2*Y2_3 - 1.576e-15*Y2_2*Y2_4 + 2.0*Y2_3*jvals[0] + 1.0*Y2_4*jvals[1];
          double F2_3 = 6.062e-15*Y2_5*Y2_4 + 1.069e-11*Y2_6*Y2_2 - 8.018e-17*Y2_2*Y2_3 + 3.152e-15*Y2_2*Y2_4 + 2.4e-10*Y2_1*Y2_4 - 1.0*Y2_3*jvals[0] + 1.0*Y2_4*jvals[1] + 1.0*Y2_4*jvals[2];
          double F2_4 = -6.062e-15*Y2_5*Y2_4 + 8.018e-17*Y2_2*Y2_3 - 1.576e-15*Y2_2*Y2_4 - 1.2e-10*Y2_1*Y2_4 - 1.0*Y2_4*jvals[1] - 1.0*Y2_4*jvals[2];
          double F2_5 = -6.062e-15*Y2_5*Y2_4 + 1.069e-11*Y2_6*Y2_2 + 1.0*Y2_6*jvals[3];
          double F2_6 = 6.062e-15*Y2_5*Y2_4 - 1.069e-11*Y2_6*Y2_2 - 1.0*Y2_6*jvals[3];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs2_4 = F2_4 + (-1.0156171083877703 / dt) * K1_4;
          double rhs2_5 = F2_5 + (-1.0156171083877703 / dt) * K1_5;
          double rhs2_6 = F2_6 + (-1.0156171083877703 / dt) * K1_6;
          // Block 0: K2 forward sub [M, O1D, O, O2, O3, NO2]
          double y2_0 = rhs2_0;
          double y2_1 = rhs2_1 - L_1_0 * y2_0;
          double y2_2 = rhs2_2 - L_2_0 * y2_0 - L_2_1 * y2_1;
          double y2_3 = rhs2_3 - L_3_1 * y2_1 - L_3_2 * y2_2;
          double y2_4 = rhs2_4 - L_4_1 * y2_1 - L_4_2 * y2_2 - L_4_3 * y2_3;
          // Block 1: K2 forward sub [NO]
          double y2_5 = rhs2_5 - L_5_2 * y2_2 - L_5_3 * y2_3 - L_5_4 * y2_4;
          double y2_6 = rhs2_6 - L_6_2 * y2_2 - L_6_3 * y2_3 - L_6_4 * y2_4 - L_6_5 * y2_5;

          // Block 0: K2 backward sub [M, O1D, O, O2, O3, NO2]
          double K2_6 = y2_6 / U_6_6;
          // Block 1: K2 backward sub [NO]
          double K2_5 = (y2_5 - U_5_6 * K2_6) / U_5_5;
          double K2_4 = (y2_4 - U_4_5 * K2_5 - U_4_6 * K2_6) / U_4_4;
          double K2_3 = (y2_3 - U_3_4 * K2_4 - U_3_5 * K2_5 - U_3_6 * K2_6) / U_3_3;
          double K2_2 = (y2_2 - U_2_3 * K2_3 - U_2_4 * K2_4 - U_2_6 * K2_6) / U_2_2;
          double K2_1 = (y2_1 - U_1_4 * K2_4) / U_1_1;
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
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs3_4 = F2_4 + (4.0759956452537702 / dt) * K1_4 + (9.20767942983308 / dt) * K2_4;
          double rhs3_5 = F2_5 + (4.0759956452537702 / dt) * K1_5 + (9.20767942983308 / dt) * K2_5;
          double rhs3_6 = F2_6 + (4.0759956452537702 / dt) * K1_6 + (9.20767942983308 / dt) * K2_6;
          // Block 0: K3 forward sub [M, O1D, O, O2, O3, NO2]
          double y3_0 = rhs3_0;
          double y3_1 = rhs3_1 - L_1_0 * y3_0;
          double y3_2 = rhs3_2 - L_2_0 * y3_0 - L_2_1 * y3_1;
          double y3_3 = rhs3_3 - L_3_1 * y3_1 - L_3_2 * y3_2;
          double y3_4 = rhs3_4 - L_4_1 * y3_1 - L_4_2 * y3_2 - L_4_3 * y3_3;
          // Block 1: K3 forward sub [NO]
          double y3_5 = rhs3_5 - L_5_2 * y3_2 - L_5_3 * y3_3 - L_5_4 * y3_4;
          double y3_6 = rhs3_6 - L_6_2 * y3_2 - L_6_3 * y3_3 - L_6_4 * y3_4 - L_6_5 * y3_5;

          // Block 0: K3 backward sub [M, O1D, O, O2, O3, NO2]
          double K3_6 = y3_6 / U_6_6;
          // Block 1: K3 backward sub [NO]
          double K3_5 = (y3_5 - U_5_6 * K3_6) / U_5_5;
          double K3_4 = (y3_4 - U_4_5 * K3_5 - U_4_6 * K3_6) / U_4_4;
          double K3_3 = (y3_3 - U_3_4 * K3_4 - U_3_5 * K3_5 - U_3_6 * K3_6) / U_3_3;
          double K3_2 = (y3_2 - U_2_3 * K3_3 - U_2_4 * K3_4 - U_2_6 * K3_6) / U_2_2;
          double K3_1 = (y3_1 - U_1_4 * K3_4) / U_1_1;
          double K3_0 = y3_0 / U_0_0;


          // --- Solution update and error estimation ---
          double err_norm_sq = 0.0;
          {
              double Ynew_i = S_0 + K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(5)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[5] + rtol[5] * ymax;
              double yerr_i = 0.5 * K1_0 + -2.9079558716805471 * K2_0 + 0.22354069897811571 * K3_0;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_1 + K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_1 + -2.9079558716805471 * K2_1 + 0.22354069897811571 * K3_1;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_2 + K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(6)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[6] + rtol[6] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_4 + K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(2)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_4 + -2.9079558716805471 * K2_4 + 0.22354069897811571 * K3_4;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_5 + K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(3)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_5 + -2.9079558716805471 * K2_5 + 0.22354069897811571 * K3_5;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_6 + K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(4)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[4] + rtol[4] * ymax;
              double yerr_i = 0.5 * K1_6 + -2.9079558716805471 * K2_6 + 0.22354069897811571 * K3_6;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 7);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(5) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(1) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(6) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              state(2) += K1_4 + 6.1697947043828245 * K2_4 + -0.42772256543218573 * K3_4;
              state(3) += K1_5 + 6.1697947043828245 * K2_5 + -0.42772256543218573 * K3_5;
              state(4) += K1_6 + 6.1697947043828245 * K2_6 + -0.42772256543218573 * K3_6;
              // Save checkpoint data for adjoint/TLM
              if (chk.num_steps >= CheckpointBuffer::MAX_STEPS) {
                  ierr = 1;  // Exceeded MAX_STEPS budget
                  return -1;
              }
              chk.h[chk.num_steps] = dt;
              chk.state[chk.num_steps][0] = state(5);
              chk.state[chk.num_steps][1] = state(1);
              chk.state[chk.num_steps][2] = state(0);
              chk.state[chk.num_steps][3] = state(6);
              chk.state[chk.num_steps][4] = state(2);
              chk.state[chk.num_steps][5] = state(3);
              chk.state[chk.num_steps][6] = state(4);
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
          const CheckpointBuffer& chk) const
      {
          const double* jvals = jvals_view.data();
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

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_1_0 = -7.11e-11*S_1;
              double J_1_1 = -7.11e-11*S_0 - 1.2e-10*S_4;
              double J_1_4 = -1.2e-10*S_1 + 1.0*jvals[2];
              double J_2_0 = 7.11e-11*S_1;
              double J_2_1 = 7.11e-11*S_0;
              double J_2_2 = -1.069e-11*S_6 - 8.018e-17*S_3 - 1.576e-15*S_4;
              double J_2_3 = -8.018e-17*S_2 + 2.0*jvals[0];
              double J_2_4 = -1.576e-15*S_2 + 1.0*jvals[1];
              double J_2_6 = -1.069e-11*S_2 + 1.0*jvals[3];
              double J_3_1 = 2.4e-10*S_4;
              double J_3_2 = 1.069e-11*S_6 - 8.018e-17*S_3 + 3.152e-15*S_4;
              double J_3_3 = -8.018e-17*S_2 - 1.0*jvals[0];
              double J_3_4 = 6.062e-15*S_5 + 3.152e-15*S_2 + 2.4e-10*S_1 + 1.0*jvals[1] + 1.0*jvals[2];
              double J_3_5 = 6.062e-15*S_4;
              double J_3_6 = 1.069e-11*S_2;
              double J_4_1 = -1.2e-10*S_4;
              double J_4_2 = 8.018e-17*S_3 - 1.576e-15*S_4;
              double J_4_3 = 8.018e-17*S_2;
              double J_4_4 = -6.062e-15*S_5 - 1.576e-15*S_2 - 1.2e-10*S_1 - 1.0*jvals[1] - 1.0*jvals[2];
              double J_4_5 = -6.062e-15*S_4;
              double J_5_2 = 1.069e-11*S_6;
              double J_5_4 = -6.062e-15*S_5;
              double J_5_5 = -6.062e-15*S_4;
              double J_5_6 = 1.069e-11*S_2 + 1.0*jvals[3];
              double J_6_2 = -1.069e-11*S_6;
              double J_6_4 = 6.062e-15*S_5;
              double J_6_5 = 6.062e-15*S_4;
              double J_6_6 = -1.069e-11*S_2 - 1.0*jvals[3];

              // Form iteration matrix W = (1/(gamma*h))*I - J
              // Block 0: species [M, O1D, O, O2, O3, NO2]
              double W_0_0 = inv_g_h;
              double W_1_0 = -J_1_0;
              double W_1_1 = inv_g_h - J_1_1;
              double W_1_4 = -J_1_4;
              double W_2_0 = -J_2_0;
              double W_2_1 = -J_2_1;
              double W_2_2 = inv_g_h - J_2_2;
              double W_2_3 = -J_2_3;
              double W_2_4 = -J_2_4;
              double W_2_6 = -J_2_6;
              double W_3_1 = -J_3_1;
              double W_3_2 = -J_3_2;
              double W_3_3 = inv_g_h - J_3_3;
              double W_3_4 = -J_3_4;
              double W_3_5 = -J_3_5;
              double W_3_6 = -J_3_6;
              double W_4_1 = -J_4_1;
              double W_4_2 = -J_4_2;
              double W_4_3 = -J_4_3;
              double W_4_4 = inv_g_h - J_4_4;
              double W_4_5 = -J_4_5;
              // Block 1: species [NO]
              double W_5_2 = -J_5_2;
              double W_5_4 = -J_5_4;
              double W_5_5 = inv_g_h - J_5_5;
              double W_5_6 = -J_5_6;
              // Block 0: species [M, O1D, O, O2, O3, NO2]
              double W_6_2 = -J_6_2;
              double W_6_4 = -J_6_4;
              double W_6_5 = -J_6_5;
              double W_6_6 = inv_g_h - J_6_6;


              // Symbolic LU Factorization (recomputed at each adjoint step)
              // Block 0: species [M, O1D, O, O2, O3, NO2]
              double U_0_0 = W_0_0;
              double L_1_0 = (W_1_0) / U_0_0;
              double L_2_0 = (W_2_0) / U_0_0;
              double U_1_1 = W_1_1;
              double U_1_4 = W_1_4;
              double L_2_1 = (W_2_1) / U_1_1;
              double L_3_1 = (W_3_1) / U_1_1;
              double L_4_1 = (W_4_1) / U_1_1;
              double U_2_2 = W_2_2;
              double U_2_3 = W_2_3;
              double U_2_4 = W_2_4 - L_2_1 * U_1_4;
              double U_2_6 = W_2_6;
              double L_3_2 = (W_3_2) / U_2_2;
              double L_4_2 = (W_4_2) / U_2_2;
              // Block 1: species [NO]
              double L_5_2 = (W_5_2) / U_2_2;
              double L_6_2 = (W_6_2) / U_2_2;
              double U_3_3 = W_3_3 - L_3_2 * U_2_3;
              double U_3_4 = W_3_4 - L_3_1 * U_1_4 - L_3_2 * U_2_4;
              double U_3_5 = W_3_5;
              double U_3_6 = W_3_6 - L_3_2 * U_2_6;
              double L_4_3 = (W_4_3 - L_4_2 * U_2_3) / U_3_3;
              double L_5_3 = (0.0 - L_5_2 * U_2_3) / U_3_3;
              double L_6_3 = (0.0 - L_6_2 * U_2_3) / U_3_3;
              double U_4_4 = W_4_4 - L_4_1 * U_1_4 - L_4_2 * U_2_4 - L_4_3 * U_3_4;
              double U_4_5 = W_4_5 - L_4_3 * U_3_5;
              double U_4_6 = 0.0 - L_4_2 * U_2_6 - L_4_3 * U_3_6;
              double L_5_4 = (W_5_4 - L_5_2 * U_2_4 - L_5_3 * U_3_4) / U_4_4;
              double L_6_4 = (W_6_4 - L_6_2 * U_2_4 - L_6_3 * U_3_4) / U_4_4;
              double U_5_5 = W_5_5 - L_5_3 * U_3_5 - L_5_4 * U_4_5;
              double U_5_6 = W_5_6 - L_5_2 * U_2_6 - L_5_3 * U_3_6 - L_5_4 * U_4_6;
              double L_6_5 = (W_6_5 - L_6_3 * U_3_5 - L_6_4 * U_4_5) / U_5_5;
              double U_6_6 = W_6_6 - L_6_2 * U_2_6 - L_6_3 * U_3_6 - L_6_4 * U_4_6 - L_6_5 * U_5_6;


              // --- Adjoint stage unrolling (s=3 stages, backward) ---
              // Driving vector: lam_k = lambda_{n+1} (constant for this step)
              const double lam_0 = lambda(0);
              const double lam_1 = lambda(1);
              const double lam_2 = lambda(2);
              const double lam_3 = lambda(3);
              const double lam_4 = lambda(4);
              const double lam_5 = lambda(5);
              const double lam_6 = lambda(6);

              // Adjoint stage 3
              // v3: RHS for W^{-T} solve
              double v3_0 = -0.42772256543218573 * lam_0;
              double v3_1 = -0.42772256543218573 * lam_1;
              double v3_2 = -0.42772256543218573 * lam_2;
              double v3_3 = -0.42772256543218573 * lam_3;
              double v3_4 = -0.42772256543218573 * lam_4;
              double v3_5 = -0.42772256543218573 * lam_5;
              double v3_6 = -0.42772256543218573 * lam_6;
              // W^{-T} solve for u3
              // Block 0: u3 transpose forward sub [M, O1D, O, O2, O3, NO2]
              double yt3_0 = v3_0 / U_0_0;
              double yt3_1 = v3_1 / U_1_1;
              double yt3_2 = v3_2 / U_2_2;
              double yt3_3 = (v3_3 - U_2_3 * yt3_2) / U_3_3;
              double yt3_4 = (v3_4 - U_1_4 * yt3_1 - U_2_4 * yt3_2 - U_3_4 * yt3_3) / U_4_4;
              // Block 1: u3 transpose forward sub [NO]
              double yt3_5 = (v3_5 - U_3_5 * yt3_3 - U_4_5 * yt3_4) / U_5_5;
              double yt3_6 = (v3_6 - U_2_6 * yt3_2 - U_3_6 * yt3_3 - U_4_6 * yt3_4 - U_5_6 * yt3_5) / U_6_6;
              // Block 0: u3 transpose backward sub [M, O1D, O, O2, O3, NO2]
              double u3_6 = yt3_6;
              // Block 1: u3 transpose backward sub [NO]
              double u3_5 = yt3_5 - L_6_5 * u3_6;
              double u3_4 = yt3_4 - L_5_4 * u3_5 - L_6_4 * u3_6;
              double u3_3 = yt3_3 - L_4_3 * u3_4 - L_5_3 * u3_5 - L_6_3 * u3_6;
              double u3_2 = yt3_2 - L_3_2 * u3_3 - L_4_2 * u3_4 - L_5_2 * u3_5 - L_6_2 * u3_6;
              double u3_1 = yt3_1 - L_2_1 * u3_2 - L_3_1 * u3_3 - L_4_1 * u3_4;
              double u3_0 = yt3_0 - L_1_0 * u3_1 - L_2_0 * u3_2;


              // Adjoint stage 2
              // v2: RHS for W^{-T} solve
              double v2_0 = 6.1697947043828245 * lam_0 + (9.20767942983308 / h) * u3_0;
              double v2_1 = 6.1697947043828245 * lam_1 + (9.20767942983308 / h) * u3_1;
              double v2_2 = 6.1697947043828245 * lam_2 + (9.20767942983308 / h) * u3_2;
              double v2_3 = 6.1697947043828245 * lam_3 + (9.20767942983308 / h) * u3_3;
              double v2_4 = 6.1697947043828245 * lam_4 + (9.20767942983308 / h) * u3_4;
              double v2_5 = 6.1697947043828245 * lam_5 + (9.20767942983308 / h) * u3_5;
              double v2_6 = 6.1697947043828245 * lam_6 + (9.20767942983308 / h) * u3_6;
              // W^{-T} solve for u2
              // Block 0: u2 transpose forward sub [M, O1D, O, O2, O3, NO2]
              double yt2_0 = v2_0 / U_0_0;
              double yt2_1 = v2_1 / U_1_1;
              double yt2_2 = v2_2 / U_2_2;
              double yt2_3 = (v2_3 - U_2_3 * yt2_2) / U_3_3;
              double yt2_4 = (v2_4 - U_1_4 * yt2_1 - U_2_4 * yt2_2 - U_3_4 * yt2_3) / U_4_4;
              // Block 1: u2 transpose forward sub [NO]
              double yt2_5 = (v2_5 - U_3_5 * yt2_3 - U_4_5 * yt2_4) / U_5_5;
              double yt2_6 = (v2_6 - U_2_6 * yt2_2 - U_3_6 * yt2_3 - U_4_6 * yt2_4 - U_5_6 * yt2_5) / U_6_6;
              // Block 0: u2 transpose backward sub [M, O1D, O, O2, O3, NO2]
              double u2_6 = yt2_6;
              // Block 1: u2 transpose backward sub [NO]
              double u2_5 = yt2_5 - L_6_5 * u2_6;
              double u2_4 = yt2_4 - L_5_4 * u2_5 - L_6_4 * u2_6;
              double u2_3 = yt2_3 - L_4_3 * u2_4 - L_5_3 * u2_5 - L_6_3 * u2_6;
              double u2_2 = yt2_2 - L_3_2 * u2_3 - L_4_2 * u2_4 - L_5_2 * u2_5 - L_6_2 * u2_6;
              double u2_1 = yt2_1 - L_2_1 * u2_2 - L_3_1 * u2_3 - L_4_1 * u2_4;
              double u2_0 = yt2_0 - L_1_0 * u2_1 - L_2_0 * u2_2;


              // Adjoint stage 1
              // v1: RHS for W^{-T} solve
              double v1_0 = lam_0 + (J_1_0 * u2_1 + J_2_0 * u2_2) + (-1.0156171083877703 / h) * u2_0 + (J_1_0 * u3_1 + J_2_0 * u3_2) + (4.0759956452537702 / h) * u3_0;
              double v1_1 = lam_1 + (J_1_1 * u2_1 + J_2_1 * u2_2 + J_3_1 * u2_3 + J_4_1 * u2_4) + (-1.0156171083877703 / h) * u2_1 + (J_1_1 * u3_1 + J_2_1 * u3_2 + J_3_1 * u3_3 + J_4_1 * u3_4) + (4.0759956452537702 / h) * u3_1;
              double v1_2 = lam_2 + (J_2_2 * u2_2 + J_3_2 * u2_3 + J_4_2 * u2_4 + J_5_2 * u2_5 + J_6_2 * u2_6) + (-1.0156171083877703 / h) * u2_2 + (J_2_2 * u3_2 + J_3_2 * u3_3 + J_4_2 * u3_4 + J_5_2 * u3_5 + J_6_2 * u3_6) + (4.0759956452537702 / h) * u3_2;
              double v1_3 = lam_3 + (J_2_3 * u2_2 + J_3_3 * u2_3 + J_4_3 * u2_4) + (-1.0156171083877703 / h) * u2_3 + (J_2_3 * u3_2 + J_3_3 * u3_3 + J_4_3 * u3_4) + (4.0759956452537702 / h) * u3_3;
              double v1_4 = lam_4 + (J_1_4 * u2_1 + J_2_4 * u2_2 + J_3_4 * u2_3 + J_4_4 * u2_4 + J_5_4 * u2_5 + J_6_4 * u2_6) + (-1.0156171083877703 / h) * u2_4 + (J_1_4 * u3_1 + J_2_4 * u3_2 + J_3_4 * u3_3 + J_4_4 * u3_4 + J_5_4 * u3_5 + J_6_4 * u3_6) + (4.0759956452537702 / h) * u3_4;
              double v1_5 = lam_5 + (J_3_5 * u2_3 + J_4_5 * u2_4 + J_5_5 * u2_5 + J_6_5 * u2_6) + (-1.0156171083877703 / h) * u2_5 + (J_3_5 * u3_3 + J_4_5 * u3_4 + J_5_5 * u3_5 + J_6_5 * u3_6) + (4.0759956452537702 / h) * u3_5;
              double v1_6 = lam_6 + (J_2_6 * u2_2 + J_3_6 * u2_3 + J_5_6 * u2_5 + J_6_6 * u2_6) + (-1.0156171083877703 / h) * u2_6 + (J_2_6 * u3_2 + J_3_6 * u3_3 + J_5_6 * u3_5 + J_6_6 * u3_6) + (4.0759956452537702 / h) * u3_6;
              // W^{-T} solve for u1
              // Block 0: u1 transpose forward sub [M, O1D, O, O2, O3, NO2]
              double yt1_0 = v1_0 / U_0_0;
              double yt1_1 = v1_1 / U_1_1;
              double yt1_2 = v1_2 / U_2_2;
              double yt1_3 = (v1_3 - U_2_3 * yt1_2) / U_3_3;
              double yt1_4 = (v1_4 - U_1_4 * yt1_1 - U_2_4 * yt1_2 - U_3_4 * yt1_3) / U_4_4;
              // Block 1: u1 transpose forward sub [NO]
              double yt1_5 = (v1_5 - U_3_5 * yt1_3 - U_4_5 * yt1_4) / U_5_5;
              double yt1_6 = (v1_6 - U_2_6 * yt1_2 - U_3_6 * yt1_3 - U_4_6 * yt1_4 - U_5_6 * yt1_5) / U_6_6;
              // Block 0: u1 transpose backward sub [M, O1D, O, O2, O3, NO2]
              double u1_6 = yt1_6;
              // Block 1: u1 transpose backward sub [NO]
              double u1_5 = yt1_5 - L_6_5 * u1_6;
              double u1_4 = yt1_4 - L_5_4 * u1_5 - L_6_4 * u1_6;
              double u1_3 = yt1_3 - L_4_3 * u1_4 - L_5_3 * u1_5 - L_6_3 * u1_6;
              double u1_2 = yt1_2 - L_3_2 * u1_3 - L_4_2 * u1_4 - L_5_2 * u1_5 - L_6_2 * u1_6;
              double u1_1 = yt1_1 - L_2_1 * u1_2 - L_3_1 * u1_3 - L_4_1 * u1_4;
              double u1_0 = yt1_0 - L_1_0 * u1_1 - L_2_0 * u1_2;


              // --- Lambda update: lambda_n = lambda_{n+1} + Σ u_i ---
              lambda(0) = lam_0 + u1_0 + u2_0 + u3_0;
              lambda(1) = lam_1 + u1_1 + u2_1 + u3_1;
              lambda(2) = lam_2 + u1_2 + u2_2 + u3_2;
              lambda(3) = lam_3 + u1_3 + u2_3 + u3_3;
              lambda(4) = lam_4 + u1_4 + u2_4 + u3_4;
              lambda(5) = lam_5 + u1_5 + u2_5 + u3_5;
              lambda(6) = lam_6 + u1_6 + u2_6 + u3_6;
          } // end for (step)
      }

      // Tangent Linear Model: forward propagation of perturbation δC
      // Uses checkpoint buffer from integrate_fwd_checkpoint().
      template <class StateView, class PertView>
      KOKKOS_INLINE_FUNCTION void integrate_tlm(
          double dt_total, const StateView& state_0,
          PertView& delta_C, Kokkos::View<const double*, memory_space> jvals_view,
          const CheckpointBuffer& chk) const
      {
          const double* jvals = jvals_view.data();
          const double g = 0.435866521508459;

          // Local aliases for delta_C (TLM stages use dC_k naming)
          double dC_0 = delta_C(0);
          double dC_1 = delta_C(1);
          double dC_2 = delta_C(2);
          double dC_3 = delta_C(3);
          double dC_4 = delta_C(4);
          double dC_5 = delta_C(5);
          double dC_6 = delta_C(6);


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

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_1_0 = -7.11e-11*S_1;
              double J_1_1 = -7.11e-11*S_0 - 1.2e-10*S_4;
              double J_1_4 = -1.2e-10*S_1 + 1.0*jvals[2];
              double J_2_0 = 7.11e-11*S_1;
              double J_2_1 = 7.11e-11*S_0;
              double J_2_2 = -1.069e-11*S_6 - 8.018e-17*S_3 - 1.576e-15*S_4;
              double J_2_3 = -8.018e-17*S_2 + 2.0*jvals[0];
              double J_2_4 = -1.576e-15*S_2 + 1.0*jvals[1];
              double J_2_6 = -1.069e-11*S_2 + 1.0*jvals[3];
              double J_3_1 = 2.4e-10*S_4;
              double J_3_2 = 1.069e-11*S_6 - 8.018e-17*S_3 + 3.152e-15*S_4;
              double J_3_3 = -8.018e-17*S_2 - 1.0*jvals[0];
              double J_3_4 = 6.062e-15*S_5 + 3.152e-15*S_2 + 2.4e-10*S_1 + 1.0*jvals[1] + 1.0*jvals[2];
              double J_3_5 = 6.062e-15*S_4;
              double J_3_6 = 1.069e-11*S_2;
              double J_4_1 = -1.2e-10*S_4;
              double J_4_2 = 8.018e-17*S_3 - 1.576e-15*S_4;
              double J_4_3 = 8.018e-17*S_2;
              double J_4_4 = -6.062e-15*S_5 - 1.576e-15*S_2 - 1.2e-10*S_1 - 1.0*jvals[1] - 1.0*jvals[2];
              double J_4_5 = -6.062e-15*S_4;
              double J_5_2 = 1.069e-11*S_6;
              double J_5_4 = -6.062e-15*S_5;
              double J_5_5 = -6.062e-15*S_4;
              double J_5_6 = 1.069e-11*S_2 + 1.0*jvals[3];
              double J_6_2 = -1.069e-11*S_6;
              double J_6_4 = 6.062e-15*S_5;
              double J_6_5 = 6.062e-15*S_4;
              double J_6_6 = -1.069e-11*S_2 - 1.0*jvals[3];

              // Form iteration matrix W = inv_g_h*I - J
              double W_0_0 = inv_g_h;
              double W_1_0 = -J_1_0;
              double W_1_1 = inv_g_h - J_1_1;
              double W_1_4 = -J_1_4;
              double W_2_0 = -J_2_0;
              double W_2_1 = -J_2_1;
              double W_2_2 = inv_g_h - J_2_2;
              double W_2_3 = -J_2_3;
              double W_2_4 = -J_2_4;
              double W_2_6 = -J_2_6;
              double W_3_1 = -J_3_1;
              double W_3_2 = -J_3_2;
              double W_3_3 = inv_g_h - J_3_3;
              double W_3_4 = -J_3_4;
              double W_3_5 = -J_3_5;
              double W_3_6 = -J_3_6;
              double W_4_1 = -J_4_1;
              double W_4_2 = -J_4_2;
              double W_4_3 = -J_4_3;
              double W_4_4 = inv_g_h - J_4_4;
              double W_4_5 = -J_4_5;
              double W_5_2 = -J_5_2;
              double W_5_4 = -J_5_4;
              double W_5_5 = inv_g_h - J_5_5;
              double W_5_6 = -J_5_6;
              double W_6_2 = -J_6_2;
              double W_6_4 = -J_6_4;
              double W_6_5 = -J_6_5;
              double W_6_6 = inv_g_h - J_6_6;

              // Symbolic LU Factorization
              // Block 0: species [M, O1D, O, O2, O3, NO2]
              double U_0_0 = W_0_0;
              double L_1_0 = (W_1_0) / U_0_0;
              double L_2_0 = (W_2_0) / U_0_0;
              double U_1_1 = W_1_1;
              double U_1_4 = W_1_4;
              double L_2_1 = (W_2_1) / U_1_1;
              double L_3_1 = (W_3_1) / U_1_1;
              double L_4_1 = (W_4_1) / U_1_1;
              double U_2_2 = W_2_2;
              double U_2_3 = W_2_3;
              double U_2_4 = W_2_4 - L_2_1 * U_1_4;
              double U_2_6 = W_2_6;
              double L_3_2 = (W_3_2) / U_2_2;
              double L_4_2 = (W_4_2) / U_2_2;
              // Block 1: species [NO]
              double L_5_2 = (W_5_2) / U_2_2;
              double L_6_2 = (W_6_2) / U_2_2;
              double U_3_3 = W_3_3 - L_3_2 * U_2_3;
              double U_3_4 = W_3_4 - L_3_1 * U_1_4 - L_3_2 * U_2_4;
              double U_3_5 = W_3_5;
              double U_3_6 = W_3_6 - L_3_2 * U_2_6;
              double L_4_3 = (W_4_3 - L_4_2 * U_2_3) / U_3_3;
              double L_5_3 = (0.0 - L_5_2 * U_2_3) / U_3_3;
              double L_6_3 = (0.0 - L_6_2 * U_2_3) / U_3_3;
              double U_4_4 = W_4_4 - L_4_1 * U_1_4 - L_4_2 * U_2_4 - L_4_3 * U_3_4;
              double U_4_5 = W_4_5 - L_4_3 * U_3_5;
              double U_4_6 = 0.0 - L_4_2 * U_2_6 - L_4_3 * U_3_6;
              double L_5_4 = (W_5_4 - L_5_2 * U_2_4 - L_5_3 * U_3_4) / U_4_4;
              double L_6_4 = (W_6_4 - L_6_2 * U_2_4 - L_6_3 * U_3_4) / U_4_4;
              double U_5_5 = W_5_5 - L_5_3 * U_3_5 - L_5_4 * U_4_5;
              double U_5_6 = W_5_6 - L_5_2 * U_2_6 - L_5_3 * U_3_6 - L_5_4 * U_4_6;
              double L_6_5 = (W_6_5 - L_6_3 * U_3_5 - L_6_4 * U_4_5) / U_5_5;
              double U_6_6 = W_6_6 - L_6_2 * U_2_6 - L_6_3 * U_3_6 - L_6_4 * U_4_6 - L_6_5 * U_5_6;



              // --- TLM Stage 1 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs1_0 = 0.0;
              double tlm_rhs1_1 = J_1_0 * dC_0 + J_1_1 * dC_1 + J_1_4 * dC_4;
              double tlm_rhs1_2 = J_2_0 * dC_0 + J_2_1 * dC_1 + J_2_2 * dC_2 + J_2_3 * dC_3 + J_2_4 * dC_4 + J_2_6 * dC_6;
              double tlm_rhs1_3 = J_3_1 * dC_1 + J_3_2 * dC_2 + J_3_3 * dC_3 + J_3_4 * dC_4 + J_3_5 * dC_5 + J_3_6 * dC_6;
              double tlm_rhs1_4 = J_4_1 * dC_1 + J_4_2 * dC_2 + J_4_3 * dC_3 + J_4_4 * dC_4 + J_4_5 * dC_5;
              double tlm_rhs1_5 = J_5_2 * dC_2 + J_5_4 * dC_4 + J_5_5 * dC_5 + J_5_6 * dC_6;
              double tlm_rhs1_6 = J_6_2 * dC_2 + J_6_4 * dC_4 + J_6_5 * dC_5 + J_6_6 * dC_6;
              // Solve W * dK1 = tlm_rhs1
              // Block 0: dK1 forward sub [M, O1D, O, O2, O3, NO2]
              double tlm_y1_0 = tlm_rhs1_0;
              double tlm_y1_1 = tlm_rhs1_1 - L_1_0 * tlm_y1_0;
              double tlm_y1_2 = tlm_rhs1_2 - L_2_0 * tlm_y1_0 - L_2_1 * tlm_y1_1;
              double tlm_y1_3 = tlm_rhs1_3 - L_3_1 * tlm_y1_1 - L_3_2 * tlm_y1_2;
              double tlm_y1_4 = tlm_rhs1_4 - L_4_1 * tlm_y1_1 - L_4_2 * tlm_y1_2 - L_4_3 * tlm_y1_3;
              // Block 1: dK1 forward sub [NO]
              double tlm_y1_5 = tlm_rhs1_5 - L_5_2 * tlm_y1_2 - L_5_3 * tlm_y1_3 - L_5_4 * tlm_y1_4;
              double tlm_y1_6 = tlm_rhs1_6 - L_6_2 * tlm_y1_2 - L_6_3 * tlm_y1_3 - L_6_4 * tlm_y1_4 - L_6_5 * tlm_y1_5;
              // Block 0: dK1 backward sub [M, O1D, O, O2, O3, NO2]
              double dK1_6 = tlm_y1_6 / U_6_6;
              // Block 1: dK1 backward sub [NO]
              double dK1_5 = (tlm_y1_5 - U_5_6 * dK1_6) / U_5_5;
              double dK1_4 = (tlm_y1_4 - U_4_5 * dK1_5 - U_4_6 * dK1_6) / U_4_4;
              double dK1_3 = (tlm_y1_3 - U_3_4 * dK1_4 - U_3_5 * dK1_5 - U_3_6 * dK1_6) / U_3_3;
              double dK1_2 = (tlm_y1_2 - U_2_3 * dK1_3 - U_2_4 * dK1_4 - U_2_6 * dK1_6) / U_2_2;
              double dK1_1 = (tlm_y1_1 - U_1_4 * dK1_4) / U_1_1;
              double dK1_0 = tlm_y1_0 / U_0_0;

              // --- TLM Stage 2 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs2_0 = (-1.0156171083877703 / h) * dK1_0;
              double tlm_rhs2_1 = J_1_0 * (dC_0 + dK1_0) + J_1_1 * (dC_1 + dK1_1) + J_1_4 * (dC_4 + dK1_4) + (-1.0156171083877703 / h) * dK1_1;
              double tlm_rhs2_2 = J_2_0 * (dC_0 + dK1_0) + J_2_1 * (dC_1 + dK1_1) + J_2_2 * (dC_2 + dK1_2) + J_2_3 * (dC_3 + dK1_3) + J_2_4 * (dC_4 + dK1_4) + J_2_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_2;
              double tlm_rhs2_3 = J_3_1 * (dC_1 + dK1_1) + J_3_2 * (dC_2 + dK1_2) + J_3_3 * (dC_3 + dK1_3) + J_3_4 * (dC_4 + dK1_4) + J_3_5 * (dC_5 + dK1_5) + J_3_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_3;
              double tlm_rhs2_4 = J_4_1 * (dC_1 + dK1_1) + J_4_2 * (dC_2 + dK1_2) + J_4_3 * (dC_3 + dK1_3) + J_4_4 * (dC_4 + dK1_4) + J_4_5 * (dC_5 + dK1_5) + (-1.0156171083877703 / h) * dK1_4;
              double tlm_rhs2_5 = J_5_2 * (dC_2 + dK1_2) + J_5_4 * (dC_4 + dK1_4) + J_5_5 * (dC_5 + dK1_5) + J_5_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_5;
              double tlm_rhs2_6 = J_6_2 * (dC_2 + dK1_2) + J_6_4 * (dC_4 + dK1_4) + J_6_5 * (dC_5 + dK1_5) + J_6_6 * (dC_6 + dK1_6) + (-1.0156171083877703 / h) * dK1_6;
              // Solve W * dK2 = tlm_rhs2
              // Block 0: dK2 forward sub [M, O1D, O, O2, O3, NO2]
              double tlm_y2_0 = tlm_rhs2_0;
              double tlm_y2_1 = tlm_rhs2_1 - L_1_0 * tlm_y2_0;
              double tlm_y2_2 = tlm_rhs2_2 - L_2_0 * tlm_y2_0 - L_2_1 * tlm_y2_1;
              double tlm_y2_3 = tlm_rhs2_3 - L_3_1 * tlm_y2_1 - L_3_2 * tlm_y2_2;
              double tlm_y2_4 = tlm_rhs2_4 - L_4_1 * tlm_y2_1 - L_4_2 * tlm_y2_2 - L_4_3 * tlm_y2_3;
              // Block 1: dK2 forward sub [NO]
              double tlm_y2_5 = tlm_rhs2_5 - L_5_2 * tlm_y2_2 - L_5_3 * tlm_y2_3 - L_5_4 * tlm_y2_4;
              double tlm_y2_6 = tlm_rhs2_6 - L_6_2 * tlm_y2_2 - L_6_3 * tlm_y2_3 - L_6_4 * tlm_y2_4 - L_6_5 * tlm_y2_5;
              // Block 0: dK2 backward sub [M, O1D, O, O2, O3, NO2]
              double dK2_6 = tlm_y2_6 / U_6_6;
              // Block 1: dK2 backward sub [NO]
              double dK2_5 = (tlm_y2_5 - U_5_6 * dK2_6) / U_5_5;
              double dK2_4 = (tlm_y2_4 - U_4_5 * dK2_5 - U_4_6 * dK2_6) / U_4_4;
              double dK2_3 = (tlm_y2_3 - U_3_4 * dK2_4 - U_3_5 * dK2_5 - U_3_6 * dK2_6) / U_3_3;
              double dK2_2 = (tlm_y2_2 - U_2_3 * dK2_3 - U_2_4 * dK2_4 - U_2_6 * dK2_6) / U_2_2;
              double dK2_1 = (tlm_y2_1 - U_1_4 * dK2_4) / U_1_1;
              double dK2_0 = tlm_y2_0 / U_0_0;

              // --- TLM Stage 3 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs3_0 = (4.0759956452537702 / h) * dK1_0 + (9.20767942983308 / h) * dK2_0;
              double tlm_rhs3_1 = J_1_0 * (dC_0 + dK1_0) + J_1_1 * (dC_1 + dK1_1) + J_1_4 * (dC_4 + dK1_4) + (4.0759956452537702 / h) * dK1_1 + (9.20767942983308 / h) * dK2_1;
              double tlm_rhs3_2 = J_2_0 * (dC_0 + dK1_0) + J_2_1 * (dC_1 + dK1_1) + J_2_2 * (dC_2 + dK1_2) + J_2_3 * (dC_3 + dK1_3) + J_2_4 * (dC_4 + dK1_4) + J_2_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_2 + (9.20767942983308 / h) * dK2_2;
              double tlm_rhs3_3 = J_3_1 * (dC_1 + dK1_1) + J_3_2 * (dC_2 + dK1_2) + J_3_3 * (dC_3 + dK1_3) + J_3_4 * (dC_4 + dK1_4) + J_3_5 * (dC_5 + dK1_5) + J_3_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_3 + (9.20767942983308 / h) * dK2_3;
              double tlm_rhs3_4 = J_4_1 * (dC_1 + dK1_1) + J_4_2 * (dC_2 + dK1_2) + J_4_3 * (dC_3 + dK1_3) + J_4_4 * (dC_4 + dK1_4) + J_4_5 * (dC_5 + dK1_5) + (4.0759956452537702 / h) * dK1_4 + (9.20767942983308 / h) * dK2_4;
              double tlm_rhs3_5 = J_5_2 * (dC_2 + dK1_2) + J_5_4 * (dC_4 + dK1_4) + J_5_5 * (dC_5 + dK1_5) + J_5_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_5 + (9.20767942983308 / h) * dK2_5;
              double tlm_rhs3_6 = J_6_2 * (dC_2 + dK1_2) + J_6_4 * (dC_4 + dK1_4) + J_6_5 * (dC_5 + dK1_5) + J_6_6 * (dC_6 + dK1_6) + (4.0759956452537702 / h) * dK1_6 + (9.20767942983308 / h) * dK2_6;
              // Solve W * dK3 = tlm_rhs3
              // Block 0: dK3 forward sub [M, O1D, O, O2, O3, NO2]
              double tlm_y3_0 = tlm_rhs3_0;
              double tlm_y3_1 = tlm_rhs3_1 - L_1_0 * tlm_y3_0;
              double tlm_y3_2 = tlm_rhs3_2 - L_2_0 * tlm_y3_0 - L_2_1 * tlm_y3_1;
              double tlm_y3_3 = tlm_rhs3_3 - L_3_1 * tlm_y3_1 - L_3_2 * tlm_y3_2;
              double tlm_y3_4 = tlm_rhs3_4 - L_4_1 * tlm_y3_1 - L_4_2 * tlm_y3_2 - L_4_3 * tlm_y3_3;
              // Block 1: dK3 forward sub [NO]
              double tlm_y3_5 = tlm_rhs3_5 - L_5_2 * tlm_y3_2 - L_5_3 * tlm_y3_3 - L_5_4 * tlm_y3_4;
              double tlm_y3_6 = tlm_rhs3_6 - L_6_2 * tlm_y3_2 - L_6_3 * tlm_y3_3 - L_6_4 * tlm_y3_4 - L_6_5 * tlm_y3_5;
              // Block 0: dK3 backward sub [M, O1D, O, O2, O3, NO2]
              double dK3_6 = tlm_y3_6 / U_6_6;
              // Block 1: dK3 backward sub [NO]
              double dK3_5 = (tlm_y3_5 - U_5_6 * dK3_6) / U_5_5;
              double dK3_4 = (tlm_y3_4 - U_4_5 * dK3_5 - U_4_6 * dK3_6) / U_4_4;
              double dK3_3 = (tlm_y3_3 - U_3_4 * dK3_4 - U_3_5 * dK3_5 - U_3_6 * dK3_6) / U_3_3;
              double dK3_2 = (tlm_y3_2 - U_2_3 * dK3_3 - U_2_4 * dK3_4 - U_2_6 * dK3_6) / U_2_2;
              double dK3_1 = (tlm_y3_1 - U_1_4 * dK3_4) / U_1_1;
              double dK3_0 = tlm_y3_0 / U_0_0;

              // --- δC update: δC += Σ M_i * δK_i ---
              dC_0 += dK1_0 + 6.1697947043828245 * dK2_0 + -0.42772256543218573 * dK3_0;
              dC_1 += dK1_1 + 6.1697947043828245 * dK2_1 + -0.42772256543218573 * dK3_1;
              dC_2 += dK1_2 + 6.1697947043828245 * dK2_2 + -0.42772256543218573 * dK3_2;
              dC_3 += dK1_3 + 6.1697947043828245 * dK2_3 + -0.42772256543218573 * dK3_3;
              dC_4 += dK1_4 + 6.1697947043828245 * dK2_4 + -0.42772256543218573 * dK3_4;
              dC_5 += dK1_5 + 6.1697947043828245 * dK2_5 + -0.42772256543218573 * dK3_5;
              dC_6 += dK1_6 + 6.1697947043828245 * dK2_6 + -0.42772256543218573 * dK3_6;
          } // end for (step)

          // Write back final δC to output view
          delta_C(0) = dC_0;
          delta_C(1) = dC_1;
          delta_C(2) = dC_2;
          delta_C(3) = dC_3;
          delta_C(4) = dC_4;
          delta_C(5) = dC_5;
          delta_C(6) = dC_6;
      }
#endif
  };
}
