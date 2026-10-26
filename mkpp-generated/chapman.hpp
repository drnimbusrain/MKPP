#pragma once
#include <Kokkos_Core.hpp>
// CPU libraries can own the heavy helper definitions. Device builds keep the
// complete device-callable implementation visible without relocatable device code.
#if defined(MKPP_SEPARATE_HOST_KERNELS_CHAPMAN)
#include "chapman/host_declarations.hpp"
#else
#include "chapman/accessor.hpp"
#include "chapman/rates.hpp"
#include "chapman/jacobian.hpp"
#include "chapman/supernodal_factorize.hpp"
#include "chapman/supernodal_solve.hpp"
#include "chapman/solve.hpp"
#endif
// Generated solver for chapman
// SZA Workload Sorted: true
// Hysteresis/Spline Continuous Transition: true
namespace mkpp::generated::chapman {
  // Pure Kokkos abstractions (no raw pragmas allowed)

  /**
   * @brief Strongly typed enum for species indexing into state vectors.
   */
  enum Species : int {
      O = 0,
      O2 = 1,
      O3 = 2,
      M = 3
  };

#ifdef MKPP_ENABLE_ADJOINT
  // Checkpoint buffer for discrete adjoint/TLM integration
  // Recompute-J strategy: only state is stored, Jacobian recomputed from saved state
  struct CheckpointBuffer {
      static constexpr int MAX_STEPS = 200;
      static constexpr int NUM_SPECIES = 4;
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
      static constexpr int NUM_RUNTIME_PARAMETERS = 2;
      static constexpr bool HAS_CLOUD_GATED = false;
#ifdef MKPP_ENABLE_ADJOINT
      // Expose the checkpoint buffer through the kernel type so generic
      // device-path callers can name it without the mechanism namespace.
      using CheckpointBuffer = ::mkpp::generated::chapman::CheckpointBuffer;
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
          J_adj_block(0, 0) = -6e-34*state(3)*state(1) - 8e-12*state(2);
          // J^T(O, O2): d(d[O2]/dt) / d[O]
          J_adj_block(0, 1) = -6e-34*state(3)*state(1) + 1.6e-11*state(2);
          // J^T(O, O3): d(d[O3]/dt) / d[O]
          J_adj_block(0, 2) = 6e-34*state(3)*state(1) - 8e-12*state(2);
          // J^T(O2, O): d(d[O]/dt) / d[O2]
          J_adj_block(1, 0) = -6e-34*state(3)*state(0) + 2.0*jvals[0];
          // J^T(O2, O2): d(d[O2]/dt) / d[O2]
          J_adj_block(1, 1) = -6e-34*state(3)*state(0) - 1.0*jvals[0];
          // J^T(O2, O3): d(d[O3]/dt) / d[O2]
          J_adj_block(1, 2) = 6e-34*state(3)*state(0);
          // J^T(O3, O): d(d[O]/dt) / d[O3]
          J_adj_block(2, 0) = -8e-12*state(0) + 1.0*jvals[1];
          // J^T(O3, O2): d(d[O2]/dt) / d[O3]
          J_adj_block(2, 1) = 1.6e-11*state(0) + 1.0*jvals[1];
          // J^T(O3, O3): d(d[O3]/dt) / d[O3]
          J_adj_block(2, 2) = -8e-12*state(0) - 1.0*jvals[1];
          // J^T(M, O): d(d[O]/dt) / d[M]
          J_adj_block(3, 0) = -6e-34*state(0)*state(1);
          // J^T(M, O2): d(d[O2]/dt) / d[M]
          J_adj_block(3, 1) = -6e-34*state(0)*state(1);
          // J^T(M, O3): d(d[O3]/dt) / d[M]
          J_adj_block(3, 2) = 6e-34*state(0)*state(1);
      }
#endif

#ifdef MKPP_ENABLE_ADJOINT
      template <class StateView, class DeltaView, class RateView>
      KOKKOS_INLINE_FUNCTION void compute_tlm(const StateView& state, const DeltaView& delta_C, RateView& dF_block, Kokkos::View<const double*, memory_space> jvals_view) const {
          const double* jvals = jvals_view.data();
          dF_block(0) = 0.0;
          dF_block(0) += (-6e-34*state(3)*state(1) - 8e-12*state(2)) * delta_C(0);
          dF_block(0) += (-6e-34*state(3)*state(0) + 2.0*jvals[0]) * delta_C(1);
          dF_block(0) += (-8e-12*state(0) + 1.0*jvals[1]) * delta_C(2);
          dF_block(0) += (-6e-34*state(0)*state(1)) * delta_C(3);
          dF_block(1) = 0.0;
          dF_block(1) += (-6e-34*state(3)*state(1) + 1.6e-11*state(2)) * delta_C(0);
          dF_block(1) += (-6e-34*state(3)*state(0) - 1.0*jvals[0]) * delta_C(1);
          dF_block(1) += (1.6e-11*state(0) + 1.0*jvals[1]) * delta_C(2);
          dF_block(1) += (-6e-34*state(0)*state(1)) * delta_C(3);
          dF_block(2) = 0.0;
          dF_block(2) += (6e-34*state(3)*state(1) - 8e-12*state(2)) * delta_C(0);
          dF_block(2) += (6e-34*state(3)*state(0)) * delta_C(1);
          dF_block(2) += (-8e-12*state(0) - 1.0*jvals[1]) * delta_C(2);
          dF_block(2) += (6e-34*state(0)*state(1)) * delta_C(3);
          dF_block(3) = 0.0;
      }
#endif

      template <class StateView, class MassView>
      KOKKOS_INLINE_FUNCTION void project_mass_conservation(StateView& C_projected, const StateView& C, const MassView& m_0) const {
          // C_projected = C - E^T (E E^T)^-1 (E C - m_0)
          C_projected(0) = C(0);
          C_projected(1) = C(1);
          C_projected(2) = C(2);
          C_projected(3) = C(3);
      }

      static constexpr int NUM_SPECIES = 4;
      static constexpr double atol[NUM_SPECIES] = { 0.001, 0.001, 0.001, 0.001 };
      static constexpr double rtol[NUM_SPECIES] = { 1e-06, 1e-06, 1e-06, 1e-06 };

      // Photolysis reactions (Cloud-J input mapping):
      //   jvals[0] = O2 -> O  (original A: 1.0)
      //   jvals[1] = O3 -> O, O2  (original A: 1.0)
      static constexpr int NUM_PHOTOLYSIS = 2;

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
          const int NUM_SPECIES = 4;
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
          const double S_1 = state(Species::O3);  // [O3]
          const double S_2 = state(Species::O);  // [O]
          const double S_3 = state(Species::O2);  // [O2]
          // The expression-dense Jacobian is evaluated by bounded compiled
          // units.  The solver still uses the same symbolic sparse LU plan.
#ifndef MKPP_USE_UNROLLED_REFERENCE
          double J_values[12] = {};
          auto jacobian_accessor = detail::StrideView<double, 2>::compact(J_values);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, 0.0, 0.0, 2.46e+19);
          double W_values[13];
          W_values[0] = inv_g_dt;
          W_values[1] = 0.0 - J_values[11];
          W_values[2] = inv_g_dt - J_values[10];
          W_values[3] = 0.0 - J_values[8];
          W_values[4] = 0.0 - J_values[9];
          W_values[5] = 0.0 - J_values[3];
          W_values[6] = 0.0 - J_values[2];
          W_values[7] = inv_g_dt - J_values[0];
          W_values[8] = 0.0 - J_values[1];
          W_values[9] = 0.0 - J_values[7];
          W_values[10] = 0.0 - J_values[6];
          W_values[11] = 0.0 - J_values[4];
          W_values[12] = inv_g_dt - J_values[5];
          double LU_values[13];
          detail::factorize_plan(W_values, LU_values);
#else
          double J_values[NUM_SPECIES * NUM_SPECIES] = {};
          auto jacobian_accessor = detail::make_stride_view<double, 2>(J_values, NUM_SPECIES, 1);
          detail::compute_jacobian_chunk_0(state_accessor, jacobian_accessor, jvals, 0.0, 0.0, 2.46e+19);
          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_1_0 = J_values[2 * NUM_SPECIES + 3];
          double J_1_1 = J_values[2 * NUM_SPECIES + 2];
          double J_1_2 = J_values[2 * NUM_SPECIES + 0];
          double J_1_3 = J_values[2 * NUM_SPECIES + 1];
          double J_2_0 = J_values[0 * NUM_SPECIES + 3];
          double J_2_1 = J_values[0 * NUM_SPECIES + 2];
          double J_2_2 = J_values[0 * NUM_SPECIES + 0];
          double J_2_3 = J_values[0 * NUM_SPECIES + 1];
          double J_3_0 = J_values[1 * NUM_SPECIES + 3];
          double J_3_1 = J_values[1 * NUM_SPECIES + 2];
          double J_3_2 = J_values[1 * NUM_SPECIES + 0];
          double J_3_3 = J_values[1 * NUM_SPECIES + 1];
          double W_values[NUM_SPECIES * NUM_SPECIES] = {};
          W_values[0 * NUM_SPECIES + 0] = inv_g_dt;
          W_values[1 * NUM_SPECIES + 0] = -J_1_0;
          W_values[1 * NUM_SPECIES + 1] = inv_g_dt - J_1_1;
          W_values[1 * NUM_SPECIES + 2] = -J_1_2;
          W_values[1 * NUM_SPECIES + 3] = -J_1_3;
          W_values[2 * NUM_SPECIES + 0] = -J_2_0;
          W_values[2 * NUM_SPECIES + 1] = -J_2_1;
          W_values[2 * NUM_SPECIES + 2] = inv_g_dt - J_2_2;
          W_values[2 * NUM_SPECIES + 3] = -J_2_3;
          W_values[3 * NUM_SPECIES + 0] = -J_3_0;
          W_values[3 * NUM_SPECIES + 1] = -J_3_1;
          W_values[3 * NUM_SPECIES + 2] = -J_3_2;
          W_values[3 * NUM_SPECIES + 3] = inv_g_dt - J_3_3;
          double LU_values[NUM_SPECIES * NUM_SPECIES] = {};
          detail::factorize_lu_chunk_0(W_values, LU_values);
#endif

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F_values_1[NUM_SPECIES];
          auto rate_accessor_1 = detail::make_stride_view<double, 1>(F_values_1, 1);
          detail::compute_rates_chunk_0(state_accessor, rate_accessor_1, jvals, 0.0, 0.0, 2.46e+19);
          double F1_0 = F_values_1[3];
          double F1_1 = F_values_1[2];
          double F1_2 = F_values_1[0];
          double F1_3 = F_values_1[1];
          double rhs_values_1[NUM_SPECIES];
          rhs_values_1[0] = F1_0;
          rhs_values_1[1] = F1_1;
          rhs_values_1[2] = F1_2;
          rhs_values_1[3] = F1_3;
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

          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          // Rate evaluation F2 at Y2
          double stage_state_2[NUM_SPECIES];
          stage_state_2[3] = Y2_0;
          stage_state_2[2] = Y2_1;
          stage_state_2[0] = Y2_2;
          stage_state_2[1] = Y2_3;
          double F_values_2[NUM_SPECIES];
          auto rate_accessor_2 = detail::make_stride_view<double, 1>(F_values_2, 1);
          auto stage_state_accessor_2 = detail::make_stride_view<double, 1>(stage_state_2, 1);
          detail::compute_rates_chunk_0(stage_state_accessor_2, rate_accessor_2, jvals, 0.0, 0.0, 2.46e+19);
          double F2_0 = F_values_2[3];
          double F2_1 = F_values_2[2];
          double F2_2 = F_values_2[0];
          double F2_3 = F_values_2[1];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          double rhs_values_2[NUM_SPECIES];
          rhs_values_2[0] = rhs2_0;
          rhs_values_2[1] = rhs2_1;
          rhs_values_2[2] = rhs2_2;
          rhs_values_2[3] = rhs2_3;
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

          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          double rhs_values_3[NUM_SPECIES];
          rhs_values_3[0] = rhs3_0;
          rhs_values_3[1] = rhs3_1;
          rhs_values_3[2] = rhs3_2;
          rhs_values_3[3] = rhs3_3;
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
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 4);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(1) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
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
          const int NUM_SPECIES = 4;
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

          while (t < dt_total) {
          dt = Kokkos::min(dt, dt_total - t);
          const double inv_g_dt = 1.0 / (g * dt);

          // 0. Hoist state values into scalar registers
          // NOTE: State access uses permuted species ordering
          const double S_0 = state(3);
          const double S_1 = state(2);
          const double S_2 = state(0);
          const double S_3 = state(1);

          // 1. Stage 1 Rates (F1)
          double F1_0 = 0.0;
          double F1_1 = 6e-34*S_0*S_2*S_3 - 8e-12*S_2*S_1 - 1.0*S_1*jvals[1];
          double F1_2 = -6e-34*S_0*S_2*S_3 - 8e-12*S_2*S_1 + 2.0*S_3*jvals[0] + 1.0*S_1*jvals[1];
          double F1_3 = -6e-34*S_0*S_2*S_3 + 1.6e-11*S_2*S_1 - 1.0*S_3*jvals[0] + 1.0*S_1*jvals[1];

          // 2. Evaluate importance and update active set
          active[0] = (Kokkos::fabs(F1_0) / (atol[0] + rtol[0] * Kokkos::fabs(state(3))) >= importance_threshold);
          active[1] = (Kokkos::fabs(F1_1) / (atol[1] + rtol[1] * Kokkos::fabs(state(2))) >= importance_threshold);
          active[2] = (Kokkos::fabs(F1_2) / (atol[2] + rtol[2] * Kokkos::fabs(state(0))) >= importance_threshold);
          active[3] = (Kokkos::fabs(F1_3) / (atol[3] + rtol[3] * Kokkos::fabs(state(1))) >= importance_threshold);

          // 3. Analytical Jacobian & Iteration Matrix W (identity for frozen species)
          double J_1_0 = 6e-34*S_2*S_3;
          double J_1_1 = -8e-12*S_2 - 1.0*jvals[1];
          double J_1_2 = 6e-34*S_0*S_3 - 8e-12*S_1;
          double J_1_3 = 6e-34*S_0*S_2;
          double J_2_0 = -6e-34*S_2*S_3;
          double J_2_1 = -8e-12*S_2 + 1.0*jvals[1];
          double J_2_2 = -6e-34*S_0*S_3 - 8e-12*S_1;
          double J_2_3 = -6e-34*S_0*S_2 + 2.0*jvals[0];
          double J_3_0 = -6e-34*S_2*S_3;
          double J_3_1 = 1.6e-11*S_2 + 1.0*jvals[1];
          double J_3_2 = -6e-34*S_0*S_3 + 1.6e-11*S_1;
          double J_3_3 = -6e-34*S_0*S_2 - 1.0*jvals[0];
          double W_0_0 = active[0] ? inv_g_dt : 1.0;
          double W_1_0 = (active[1] && active[0]) ? (-J_1_0) : 0.0;
          double W_1_1 = active[1] ? (inv_g_dt - J_1_1) : 1.0;
          double W_1_2 = (active[1] && active[2]) ? (-J_1_2) : 0.0;
          double W_1_3 = (active[1] && active[3]) ? (-J_1_3) : 0.0;
          double W_2_0 = (active[2] && active[0]) ? (-J_2_0) : 0.0;
          double W_2_1 = (active[2] && active[1]) ? (-J_2_1) : 0.0;
          double W_2_2 = active[2] ? (inv_g_dt - J_2_2) : 1.0;
          double W_2_3 = (active[2] && active[3]) ? (-J_2_3) : 0.0;
          double W_3_0 = (active[3] && active[0]) ? (-J_3_0) : 0.0;
          double W_3_1 = (active[3] && active[1]) ? (-J_3_1) : 0.0;
          double W_3_2 = (active[3] && active[2]) ? (-J_3_2) : 0.0;
          double W_3_3 = active[3] ? (inv_g_dt - J_3_3) : 1.0;

          // 4. Symbolic LU Factorization (conditional skip for frozen species)
          double U_0_0 = W_0_0;
          double L_1_0 = (W_1_0) / U_0_0;
          double L_2_0 = (W_2_0) / U_0_0;
          double L_3_0 = (W_3_0) / U_0_0;
          double U_1_1 = W_1_1;
          double U_1_2 = W_1_2;
          double U_1_3 = W_1_3;
          double L_2_1 = (W_2_1) / U_1_1;
          double L_3_1 = (W_3_1) / U_1_1;
          double U_2_2 = W_2_2 - L_2_1 * U_1_2;
          double U_2_3 = W_2_3 - L_2_1 * U_1_3;
          double L_3_2 = (W_3_2 - L_3_1 * U_1_2) / U_2_2;
          double U_3_3 = W_3_3 - L_3_1 * U_1_3 - L_3_2 * U_2_3;

          // --- Stage 1 ---
          // F1 already computed above (used for importance evaluation)
          // Block 0: K1 forward sub [M, O3, O]
          double y1_0 = active[0] ? (F1_0) : 0.0;
          double y1_1 = active[1] ? (F1_1 - L_1_0 * y1_0) : 0.0;
          double y1_2 = active[2] ? (F1_2 - L_2_0 * y1_0 - L_2_1 * y1_1) : 0.0;
          // Block 1: K1 forward sub [O2]
          double y1_3 = active[3] ? (F1_3 - L_3_0 * y1_0 - L_3_1 * y1_1 - L_3_2 * y1_2) : 0.0;

          // Block 1: K1 backward sub [O2]
          double K1_3 = active[3] ? (y1_3 / U_3_3) : 0.0;
          // Block 0: K1 backward sub [M, O3, O]
          double K1_2 = active[2] ? ((y1_2 - U_2_3 * K1_3) / U_2_2) : 0.0;
          double K1_1 = active[1] ? ((y1_1 - U_1_2 * K1_2 - U_1_3 * K1_3) / U_1_1) : 0.0;
          double K1_0 = active[0] ? (y1_0 / U_0_0) : 0.0;


          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = 6e-34*Y2_0*Y2_2*Y2_3 - 8e-12*Y2_2*Y2_1 - 1.0*Y2_1*jvals[1];
          double F2_2 = -6e-34*Y2_0*Y2_2*Y2_3 - 8e-12*Y2_2*Y2_1 + 2.0*Y2_3*jvals[0] + 1.0*Y2_1*jvals[1];
          double F2_3 = -6e-34*Y2_0*Y2_2*Y2_3 + 1.6e-11*Y2_2*Y2_1 - 1.0*Y2_3*jvals[0] + 1.0*Y2_1*jvals[1];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          // Block 0: K2 forward sub [M, O3, O]
          double y2_0 = active[0] ? (rhs2_0) : 0.0;
          double y2_1 = active[1] ? (rhs2_1 - L_1_0 * y2_0) : 0.0;
          double y2_2 = active[2] ? (rhs2_2 - L_2_0 * y2_0 - L_2_1 * y2_1) : 0.0;
          // Block 1: K2 forward sub [O2]
          double y2_3 = active[3] ? (rhs2_3 - L_3_0 * y2_0 - L_3_1 * y2_1 - L_3_2 * y2_2) : 0.0;

          // Block 1: K2 backward sub [O2]
          double K2_3 = active[3] ? (y2_3 / U_3_3) : 0.0;
          // Block 0: K2 backward sub [M, O3, O]
          double K2_2 = active[2] ? ((y2_2 - U_2_3 * K2_3) / U_2_2) : 0.0;
          double K2_1 = active[1] ? ((y2_1 - U_1_2 * K2_2 - U_1_3 * K2_3) / U_1_1) : 0.0;
          double K2_0 = active[0] ? (y2_0 / U_0_0) : 0.0;


          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          // Block 0: K3 forward sub [M, O3, O]
          double y3_0 = active[0] ? (rhs3_0) : 0.0;
          double y3_1 = active[1] ? (rhs3_1 - L_1_0 * y3_0) : 0.0;
          double y3_2 = active[2] ? (rhs3_2 - L_2_0 * y3_0 - L_2_1 * y3_1) : 0.0;
          // Block 1: K3 forward sub [O2]
          double y3_3 = active[3] ? (rhs3_3 - L_3_0 * y3_0 - L_3_1 * y3_1 - L_3_2 * y3_2) : 0.0;

          // Block 1: K3 backward sub [O2]
          double K3_3 = active[3] ? (y3_3 / U_3_3) : 0.0;
          // Block 0: K3 backward sub [M, O3, O]
          double K3_2 = active[2] ? ((y3_2 - U_2_3 * K3_3) / U_2_2) : 0.0;
          double K3_1 = active[1] ? ((y3_1 - U_1_2 * K3_2 - U_1_3 * K3_3) / U_1_1) : 0.0;
          double K3_0 = active[0] ? (y3_0 / U_0_0) : 0.0;


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
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              double sci = atol[2] + rtol[2] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              double sci = atol[3] + rtol[3] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 4);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              if (active[0]) state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              if (active[1]) state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              if (active[2]) state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              if (active[3]) state(1) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
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
          const int NUM_SPECIES = 4;
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
          const double S_1 = state(Species::O3);  // [O3]
          const double S_2 = state(Species::O);  // [O]
          const double S_3 = state(Species::O2);  // [O2]

          // --- Reaction Rate Fluxes R_m ---
          const double R_0 = S_1*jvals[0];
          const double R_1 = 6e-34*S_3*S_0*S_1;
          const double R_2 = S_2*jvals[1];
          const double R_3 = 8e-12*S_0*S_2;

          // Analytical Jacobian & Iteration Matrix W = inv_g_dt*I - J (sparse)
          double J_1_0 = 6e-34*S_2*S_3;
          double J_1_1 = -8e-12*S_2 - 1.0*jvals[1];
          double J_1_2 = 6e-34*S_0*S_3 - 8e-12*S_1;
          double J_1_3 = 6e-34*S_0*S_2;
          double J_2_0 = -6e-34*S_2*S_3;
          double J_2_1 = -8e-12*S_2 + 1.0*jvals[1];
          double J_2_2 = -6e-34*S_0*S_3 - 8e-12*S_1;
          double J_2_3 = -6e-34*S_0*S_2 + 2.0*jvals[0];
          double J_3_0 = -6e-34*S_2*S_3;
          double J_3_1 = 1.6e-11*S_2 + 1.0*jvals[1];
          double J_3_2 = -6e-34*S_0*S_3 + 1.6e-11*S_1;
          double J_3_3 = -6e-34*S_0*S_2 - 1.0*jvals[0];
          // Block 0: species [M, O3, O]
          double W_0_0 = inv_g_dt;
          double W_1_0 = -J_1_0;
          double W_1_1 = inv_g_dt - J_1_1;
          double W_1_2 = -J_1_2;
          double W_1_3 = -J_1_3;
          double W_2_0 = -J_2_0;
          double W_2_1 = -J_2_1;
          double W_2_2 = inv_g_dt - J_2_2;
          double W_2_3 = -J_2_3;
          // Block 1: species [O2]
          double W_3_0 = -J_3_0;
          double W_3_1 = -J_3_1;
          double W_3_2 = -J_3_2;
          double W_3_3 = inv_g_dt - J_3_3;

          // Symbolic Doolittle Sparse LU Factorization
          // Block 0: species [M, O3, O]
          double U_0_0 = W_0_0;
          double L_1_0 = (W_1_0) / U_0_0;
          double L_2_0 = (W_2_0) / U_0_0;
          // Block 1: species [O2]
          double L_3_0 = (W_3_0) / U_0_0;
          double U_1_1 = W_1_1;
          double U_1_2 = W_1_2;
          double U_1_3 = W_1_3;
          double L_2_1 = (W_2_1) / U_1_1;
          double L_3_1 = (W_3_1) / U_1_1;
          double U_2_2 = W_2_2 - L_2_1 * U_1_2;
          double U_2_3 = W_2_3 - L_2_1 * U_1_3;
          double L_3_2 = (W_3_2 - L_3_1 * U_1_2) / U_2_2;
          double U_3_3 = W_3_3 - L_3_1 * U_1_3 - L_3_2 * U_2_3;

          // --- Stage 1 ---
          // Rate evaluation F1 at S
          double F1_0 = 0.0;
          double F1_1 = 6e-34*S_0*S_2*S_3 - 8e-12*S_2*S_1 - 1.0*S_1*jvals[1];
          double F1_2 = -6e-34*S_0*S_2*S_3 - 8e-12*S_2*S_1 + 2.0*S_3*jvals[0] + 1.0*S_1*jvals[1];
          double F1_3 = -6e-34*S_0*S_2*S_3 + 1.6e-11*S_2*S_1 - 1.0*S_3*jvals[0] + 1.0*S_1*jvals[1];
          // Block 0: K1 forward sub [M, O3, O]
          double y1_0 = F1_0;
          double y1_1 = F1_1 - L_1_0 * y1_0;
          double y1_2 = F1_2 - L_2_0 * y1_0 - L_2_1 * y1_1;
          // Block 1: K1 forward sub [O2]
          double y1_3 = F1_3 - L_3_0 * y1_0 - L_3_1 * y1_1 - L_3_2 * y1_2;

          // Block 1: K1 backward sub [O2]
          double K1_3 = y1_3 / U_3_3;
          // Block 0: K1 backward sub [M, O3, O]
          double K1_2 = (y1_2 - U_2_3 * K1_3) / U_2_2;
          double K1_1 = (y1_1 - U_1_2 * K1_2 - U_1_3 * K1_3) / U_1_1;
          double K1_0 = y1_0 / U_0_0;


          // --- Stage 2 ---
          // Intermediate state Y2
          double Y2_0 = S_0 + K1_0;
          double Y2_1 = S_1 + K1_1;
          double Y2_2 = S_2 + K1_2;
          double Y2_3 = S_3 + K1_3;
          // Rate evaluation F2 at Y2
          double F2_0 = 0.0;
          double F2_1 = 6e-34*Y2_0*Y2_2*Y2_3 - 8e-12*Y2_2*Y2_1 - 1.0*Y2_1*jvals[1];
          double F2_2 = -6e-34*Y2_0*Y2_2*Y2_3 - 8e-12*Y2_2*Y2_1 + 2.0*Y2_3*jvals[0] + 1.0*Y2_1*jvals[1];
          double F2_3 = -6e-34*Y2_0*Y2_2*Y2_3 + 1.6e-11*Y2_2*Y2_1 - 1.0*Y2_3*jvals[0] + 1.0*Y2_1*jvals[1];
          // RHS for stage 2
          double rhs2_0 = F2_0 + (-1.0156171083877703 / dt) * K1_0;
          double rhs2_1 = F2_1 + (-1.0156171083877703 / dt) * K1_1;
          double rhs2_2 = F2_2 + (-1.0156171083877703 / dt) * K1_2;
          double rhs2_3 = F2_3 + (-1.0156171083877703 / dt) * K1_3;
          // Block 0: K2 forward sub [M, O3, O]
          double y2_0 = rhs2_0;
          double y2_1 = rhs2_1 - L_1_0 * y2_0;
          double y2_2 = rhs2_2 - L_2_0 * y2_0 - L_2_1 * y2_1;
          // Block 1: K2 forward sub [O2]
          double y2_3 = rhs2_3 - L_3_0 * y2_0 - L_3_1 * y2_1 - L_3_2 * y2_2;

          // Block 1: K2 backward sub [O2]
          double K2_3 = y2_3 / U_3_3;
          // Block 0: K2 backward sub [M, O3, O]
          double K2_2 = (y2_2 - U_2_3 * K2_3) / U_2_2;
          double K2_1 = (y2_1 - U_1_2 * K2_2 - U_1_3 * K2_3) / U_1_1;
          double K2_0 = y2_0 / U_0_0;


          // --- Stage 3 ---
          // Intermediate state Y3
          double Y3_0 = S_0 + K1_0;
          double Y3_1 = S_1 + K1_1;
          double Y3_2 = S_2 + K1_2;
          double Y3_3 = S_3 + K1_3;
          // NewF=false: reuse F2 for stage 3
          // RHS for stage 3
          double rhs3_0 = F2_0 + (4.0759956452537702 / dt) * K1_0 + (9.20767942983308 / dt) * K2_0;
          double rhs3_1 = F2_1 + (4.0759956452537702 / dt) * K1_1 + (9.20767942983308 / dt) * K2_1;
          double rhs3_2 = F2_2 + (4.0759956452537702 / dt) * K1_2 + (9.20767942983308 / dt) * K2_2;
          double rhs3_3 = F2_3 + (4.0759956452537702 / dt) * K1_3 + (9.20767942983308 / dt) * K2_3;
          // Block 0: K3 forward sub [M, O3, O]
          double y3_0 = rhs3_0;
          double y3_1 = rhs3_1 - L_1_0 * y3_0;
          double y3_2 = rhs3_2 - L_2_0 * y3_0 - L_2_1 * y3_1;
          // Block 1: K3 forward sub [O2]
          double y3_3 = rhs3_3 - L_3_0 * y3_0 - L_3_1 * y3_1 - L_3_2 * y3_2;

          // Block 1: K3 backward sub [O2]
          double K3_3 = y3_3 / U_3_3;
          // Block 0: K3 backward sub [M, O3, O]
          double K3_2 = (y3_2 - U_2_3 * K3_3) / U_2_2;
          double K3_1 = (y3_1 - U_1_2 * K3_2 - U_1_3 * K3_3) / U_1_1;
          double K3_0 = y3_0 / U_0_0;


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
              double ymax = Kokkos::fmax(Kokkos::fabs(state(0)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[0] + rtol[0] * ymax;
              double yerr_i = 0.5 * K1_2 + -2.9079558716805471 * K2_2 + 0.22354069897811571 * K3_2;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          {
              double Ynew_i = S_3 + K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              double ymax = Kokkos::fmax(Kokkos::fabs(state(1)), Kokkos::fabs(Ynew_i));
              // Tolerances are stored in physical species order, while the
              // generated scalar registers may use the RCM permutation.
              double sci = atol[1] + rtol[1] * ymax;
              double yerr_i = 0.5 * K1_3 + -2.9079558716805471 * K2_3 + 0.22354069897811571 * K3_3;
              err_norm_sq += (yerr_i / sci) * (yerr_i / sci);
          }
          double err_norm = Kokkos::sqrt(err_norm_sq / 4);
          err_norm = Kokkos::fmax(err_norm, 1.0e-10);

          // Step Size Control (order 3: exponent = 1/3 = 0.33333333333333331)
          double factor = safety / Kokkos::cbrt(err_norm);
          factor = Kokkos::fmax(min_shrink, Kokkos::fmin(factor, max_growth));

          if (err_norm <= 1.0) {
              state(3) += K1_0 + 6.1697947043828245 * K2_0 + -0.42772256543218573 * K3_0;
              state(2) += K1_1 + 6.1697947043828245 * K2_1 + -0.42772256543218573 * K3_1;
              state(0) += K1_2 + 6.1697947043828245 * K2_2 + -0.42772256543218573 * K3_2;
              state(1) += K1_3 + 6.1697947043828245 * K2_3 + -0.42772256543218573 * K3_3;
              // Save checkpoint data for adjoint/TLM
              if (chk.num_steps >= CheckpointBuffer::MAX_STEPS) {
                  ierr = 1;  // Exceeded MAX_STEPS budget
                  return -1;
              }
              chk.h[chk.num_steps] = dt;
              chk.state[chk.num_steps][0] = state(3);
              chk.state[chk.num_steps][1] = state(2);
              chk.state[chk.num_steps][2] = state(0);
              chk.state[chk.num_steps][3] = state(1);
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

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_1_0 = 6e-34*S_2*S_3;
              double J_1_1 = -8e-12*S_2 - 1.0*jvals[1];
              double J_1_2 = 6e-34*S_0*S_3 - 8e-12*S_1;
              double J_1_3 = 6e-34*S_0*S_2;
              double J_2_0 = -6e-34*S_2*S_3;
              double J_2_1 = -8e-12*S_2 + 1.0*jvals[1];
              double J_2_2 = -6e-34*S_0*S_3 - 8e-12*S_1;
              double J_2_3 = -6e-34*S_0*S_2 + 2.0*jvals[0];
              double J_3_0 = -6e-34*S_2*S_3;
              double J_3_1 = 1.6e-11*S_2 + 1.0*jvals[1];
              double J_3_2 = -6e-34*S_0*S_3 + 1.6e-11*S_1;
              double J_3_3 = -6e-34*S_0*S_2 - 1.0*jvals[0];

              // Form iteration matrix W = (1/(gamma*h))*I - J
              // Block 0: species [M, O3, O]
              double W_0_0 = inv_g_h;
              double W_1_0 = -J_1_0;
              double W_1_1 = inv_g_h - J_1_1;
              double W_1_2 = -J_1_2;
              double W_1_3 = -J_1_3;
              double W_2_0 = -J_2_0;
              double W_2_1 = -J_2_1;
              double W_2_2 = inv_g_h - J_2_2;
              double W_2_3 = -J_2_3;
              // Block 1: species [O2]
              double W_3_0 = -J_3_0;
              double W_3_1 = -J_3_1;
              double W_3_2 = -J_3_2;
              double W_3_3 = inv_g_h - J_3_3;


              // Symbolic LU Factorization (recomputed at each adjoint step)
              // Block 0: species [M, O3, O]
              double U_0_0 = W_0_0;
              double L_1_0 = (W_1_0) / U_0_0;
              double L_2_0 = (W_2_0) / U_0_0;
              // Block 1: species [O2]
              double L_3_0 = (W_3_0) / U_0_0;
              double U_1_1 = W_1_1;
              double U_1_2 = W_1_2;
              double U_1_3 = W_1_3;
              double L_2_1 = (W_2_1) / U_1_1;
              double L_3_1 = (W_3_1) / U_1_1;
              double U_2_2 = W_2_2 - L_2_1 * U_1_2;
              double U_2_3 = W_2_3 - L_2_1 * U_1_3;
              double L_3_2 = (W_3_2 - L_3_1 * U_1_2) / U_2_2;
              double U_3_3 = W_3_3 - L_3_1 * U_1_3 - L_3_2 * U_2_3;


              // --- Adjoint stage unrolling (s=3 stages, backward) ---
              // Driving vector: lam_k = lambda_{n+1} (constant for this step)
              const double lam_0 = lambda(0);
              const double lam_1 = lambda(1);
              const double lam_2 = lambda(2);
              const double lam_3 = lambda(3);

              // Adjoint stage 3
              // v3: RHS for W^{-T} solve
              double v3_0 = -0.42772256543218573 * lam_0;
              double v3_1 = -0.42772256543218573 * lam_1;
              double v3_2 = -0.42772256543218573 * lam_2;
              double v3_3 = -0.42772256543218573 * lam_3;
              // W^{-T} solve for u3
              // Block 0: u3 transpose forward sub [M, O3, O]
              double yt3_0 = v3_0 / U_0_0;
              double yt3_1 = v3_1 / U_1_1;
              double yt3_2 = (v3_2 - U_1_2 * yt3_1) / U_2_2;
              // Block 1: u3 transpose forward sub [O2]
              double yt3_3 = (v3_3 - U_1_3 * yt3_1 - U_2_3 * yt3_2) / U_3_3;
              // Block 1: u3 transpose backward sub [O2]
              double u3_3 = yt3_3;
              // Block 0: u3 transpose backward sub [M, O3, O]
              double u3_2 = yt3_2 - L_3_2 * u3_3;
              double u3_1 = yt3_1 - L_2_1 * u3_2 - L_3_1 * u3_3;
              double u3_0 = yt3_0 - L_1_0 * u3_1 - L_2_0 * u3_2 - L_3_0 * u3_3;


              // Adjoint stage 2
              // v2: RHS for W^{-T} solve
              double v2_0 = 6.1697947043828245 * lam_0 + (9.20767942983308 / h) * u3_0;
              double v2_1 = 6.1697947043828245 * lam_1 + (9.20767942983308 / h) * u3_1;
              double v2_2 = 6.1697947043828245 * lam_2 + (9.20767942983308 / h) * u3_2;
              double v2_3 = 6.1697947043828245 * lam_3 + (9.20767942983308 / h) * u3_3;
              // W^{-T} solve for u2
              // Block 0: u2 transpose forward sub [M, O3, O]
              double yt2_0 = v2_0 / U_0_0;
              double yt2_1 = v2_1 / U_1_1;
              double yt2_2 = (v2_2 - U_1_2 * yt2_1) / U_2_2;
              // Block 1: u2 transpose forward sub [O2]
              double yt2_3 = (v2_3 - U_1_3 * yt2_1 - U_2_3 * yt2_2) / U_3_3;
              // Block 1: u2 transpose backward sub [O2]
              double u2_3 = yt2_3;
              // Block 0: u2 transpose backward sub [M, O3, O]
              double u2_2 = yt2_2 - L_3_2 * u2_3;
              double u2_1 = yt2_1 - L_2_1 * u2_2 - L_3_1 * u2_3;
              double u2_0 = yt2_0 - L_1_0 * u2_1 - L_2_0 * u2_2 - L_3_0 * u2_3;


              // Adjoint stage 1
              // v1: RHS for W^{-T} solve
              double v1_0 = lam_0 + (J_1_0 * u2_1 + J_2_0 * u2_2 + J_3_0 * u2_3) + (-1.0156171083877703 / h) * u2_0 + (J_1_0 * u3_1 + J_2_0 * u3_2 + J_3_0 * u3_3) + (4.0759956452537702 / h) * u3_0;
              double v1_1 = lam_1 + (J_1_1 * u2_1 + J_2_1 * u2_2 + J_3_1 * u2_3) + (-1.0156171083877703 / h) * u2_1 + (J_1_1 * u3_1 + J_2_1 * u3_2 + J_3_1 * u3_3) + (4.0759956452537702 / h) * u3_1;
              double v1_2 = lam_2 + (J_1_2 * u2_1 + J_2_2 * u2_2 + J_3_2 * u2_3) + (-1.0156171083877703 / h) * u2_2 + (J_1_2 * u3_1 + J_2_2 * u3_2 + J_3_2 * u3_3) + (4.0759956452537702 / h) * u3_2;
              double v1_3 = lam_3 + (J_1_3 * u2_1 + J_2_3 * u2_2 + J_3_3 * u2_3) + (-1.0156171083877703 / h) * u2_3 + (J_1_3 * u3_1 + J_2_3 * u3_2 + J_3_3 * u3_3) + (4.0759956452537702 / h) * u3_3;
              // W^{-T} solve for u1
              // Block 0: u1 transpose forward sub [M, O3, O]
              double yt1_0 = v1_0 / U_0_0;
              double yt1_1 = v1_1 / U_1_1;
              double yt1_2 = (v1_2 - U_1_2 * yt1_1) / U_2_2;
              // Block 1: u1 transpose forward sub [O2]
              double yt1_3 = (v1_3 - U_1_3 * yt1_1 - U_2_3 * yt1_2) / U_3_3;
              // Block 1: u1 transpose backward sub [O2]
              double u1_3 = yt1_3;
              // Block 0: u1 transpose backward sub [M, O3, O]
              double u1_2 = yt1_2 - L_3_2 * u1_3;
              double u1_1 = yt1_1 - L_2_1 * u1_2 - L_3_1 * u1_3;
              double u1_0 = yt1_0 - L_1_0 * u1_1 - L_2_0 * u1_2 - L_3_0 * u1_3;


              // --- Lambda update: lambda_n = lambda_{n+1} + Σ u_i ---
              lambda(0) = lam_0 + u1_0 + u2_0 + u3_0;
              lambda(1) = lam_1 + u1_1 + u2_1 + u3_1;
              lambda(2) = lam_2 + u1_2 + u2_2 + u3_2;
              lambda(3) = lam_3 + u1_3 + u2_3 + u3_3;
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


          // === TLM Forward Propagation ===
          for (int step = 0; step < chk.num_steps; step++) {
              const double h = chk.h[step];
              const double inv_g_h = 1.0 / (0.435866521508459 * h);

              // Reload checkpointed state
              const double S_0 = chk.state[step][0];
              const double S_1 = chk.state[step][1];
              const double S_2 = chk.state[step][2];
              const double S_3 = chk.state[step][3];

              // Recompute Jacobian at checkpointed state (recompute-J strategy)
              double J_1_0 = 6e-34*S_2*S_3;
              double J_1_1 = -8e-12*S_2 - 1.0*jvals[1];
              double J_1_2 = 6e-34*S_0*S_3 - 8e-12*S_1;
              double J_1_3 = 6e-34*S_0*S_2;
              double J_2_0 = -6e-34*S_2*S_3;
              double J_2_1 = -8e-12*S_2 + 1.0*jvals[1];
              double J_2_2 = -6e-34*S_0*S_3 - 8e-12*S_1;
              double J_2_3 = -6e-34*S_0*S_2 + 2.0*jvals[0];
              double J_3_0 = -6e-34*S_2*S_3;
              double J_3_1 = 1.6e-11*S_2 + 1.0*jvals[1];
              double J_3_2 = -6e-34*S_0*S_3 + 1.6e-11*S_1;
              double J_3_3 = -6e-34*S_0*S_2 - 1.0*jvals[0];

              // Form iteration matrix W = inv_g_h*I - J
              double W_0_0 = inv_g_h;
              double W_1_0 = -J_1_0;
              double W_1_1 = inv_g_h - J_1_1;
              double W_1_2 = -J_1_2;
              double W_1_3 = -J_1_3;
              double W_2_0 = -J_2_0;
              double W_2_1 = -J_2_1;
              double W_2_2 = inv_g_h - J_2_2;
              double W_2_3 = -J_2_3;
              double W_3_0 = -J_3_0;
              double W_3_1 = -J_3_1;
              double W_3_2 = -J_3_2;
              double W_3_3 = inv_g_h - J_3_3;

              // Symbolic LU Factorization
              // Block 0: species [M, O3, O]
              double U_0_0 = W_0_0;
              double L_1_0 = (W_1_0) / U_0_0;
              double L_2_0 = (W_2_0) / U_0_0;
              // Block 1: species [O2]
              double L_3_0 = (W_3_0) / U_0_0;
              double U_1_1 = W_1_1;
              double U_1_2 = W_1_2;
              double U_1_3 = W_1_3;
              double L_2_1 = (W_2_1) / U_1_1;
              double L_3_1 = (W_3_1) / U_1_1;
              double U_2_2 = W_2_2 - L_2_1 * U_1_2;
              double U_2_3 = W_2_3 - L_2_1 * U_1_3;
              double L_3_2 = (W_3_2 - L_3_1 * U_1_2) / U_2_2;
              double U_3_3 = W_3_3 - L_3_1 * U_1_3 - L_3_2 * U_2_3;



              // --- TLM Stage 1 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs1_0 = 0.0;
              double tlm_rhs1_1 = J_1_0 * dC_0 + J_1_1 * dC_1 + J_1_2 * dC_2 + J_1_3 * dC_3;
              double tlm_rhs1_2 = J_2_0 * dC_0 + J_2_1 * dC_1 + J_2_2 * dC_2 + J_2_3 * dC_3;
              double tlm_rhs1_3 = J_3_0 * dC_0 + J_3_1 * dC_1 + J_3_2 * dC_2 + J_3_3 * dC_3;
              // Solve W * dK1 = tlm_rhs1
              // Block 0: dK1 forward sub [M, O3, O]
              double tlm_y1_0 = tlm_rhs1_0;
              double tlm_y1_1 = tlm_rhs1_1 - L_1_0 * tlm_y1_0;
              double tlm_y1_2 = tlm_rhs1_2 - L_2_0 * tlm_y1_0 - L_2_1 * tlm_y1_1;
              // Block 1: dK1 forward sub [O2]
              double tlm_y1_3 = tlm_rhs1_3 - L_3_0 * tlm_y1_0 - L_3_1 * tlm_y1_1 - L_3_2 * tlm_y1_2;
              // Block 1: dK1 backward sub [O2]
              double dK1_3 = tlm_y1_3 / U_3_3;
              // Block 0: dK1 backward sub [M, O3, O]
              double dK1_2 = (tlm_y1_2 - U_2_3 * dK1_3) / U_2_2;
              double dK1_1 = (tlm_y1_1 - U_1_2 * dK1_2 - U_1_3 * dK1_3) / U_1_1;
              double dK1_0 = tlm_y1_0 / U_0_0;

              // --- TLM Stage 2 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs2_0 = (-1.0156171083877703 / h) * dK1_0;
              double tlm_rhs2_1 = J_1_0 * (dC_0 + dK1_0) + J_1_1 * (dC_1 + dK1_1) + J_1_2 * (dC_2 + dK1_2) + J_1_3 * (dC_3 + dK1_3) + (-1.0156171083877703 / h) * dK1_1;
              double tlm_rhs2_2 = J_2_0 * (dC_0 + dK1_0) + J_2_1 * (dC_1 + dK1_1) + J_2_2 * (dC_2 + dK1_2) + J_2_3 * (dC_3 + dK1_3) + (-1.0156171083877703 / h) * dK1_2;
              double tlm_rhs2_3 = J_3_0 * (dC_0 + dK1_0) + J_3_1 * (dC_1 + dK1_1) + J_3_2 * (dC_2 + dK1_2) + J_3_3 * (dC_3 + dK1_3) + (-1.0156171083877703 / h) * dK1_3;
              // Solve W * dK2 = tlm_rhs2
              // Block 0: dK2 forward sub [M, O3, O]
              double tlm_y2_0 = tlm_rhs2_0;
              double tlm_y2_1 = tlm_rhs2_1 - L_1_0 * tlm_y2_0;
              double tlm_y2_2 = tlm_rhs2_2 - L_2_0 * tlm_y2_0 - L_2_1 * tlm_y2_1;
              // Block 1: dK2 forward sub [O2]
              double tlm_y2_3 = tlm_rhs2_3 - L_3_0 * tlm_y2_0 - L_3_1 * tlm_y2_1 - L_3_2 * tlm_y2_2;
              // Block 1: dK2 backward sub [O2]
              double dK2_3 = tlm_y2_3 / U_3_3;
              // Block 0: dK2 backward sub [M, O3, O]
              double dK2_2 = (tlm_y2_2 - U_2_3 * dK2_3) / U_2_2;
              double dK2_1 = (tlm_y2_1 - U_1_2 * dK2_2 - U_1_3 * dK2_3) / U_1_1;
              double dK2_0 = tlm_y2_0 / U_0_0;

              // --- TLM Stage 3 ---
              // RHS: J*(δC + Σ A_{i,j}*δK_j) + Σ C_{i,j}/h*δK_j
              double tlm_rhs3_0 = (4.0759956452537702 / h) * dK1_0 + (9.20767942983308 / h) * dK2_0;
              double tlm_rhs3_1 = J_1_0 * (dC_0 + dK1_0) + J_1_1 * (dC_1 + dK1_1) + J_1_2 * (dC_2 + dK1_2) + J_1_3 * (dC_3 + dK1_3) + (4.0759956452537702 / h) * dK1_1 + (9.20767942983308 / h) * dK2_1;
              double tlm_rhs3_2 = J_2_0 * (dC_0 + dK1_0) + J_2_1 * (dC_1 + dK1_1) + J_2_2 * (dC_2 + dK1_2) + J_2_3 * (dC_3 + dK1_3) + (4.0759956452537702 / h) * dK1_2 + (9.20767942983308 / h) * dK2_2;
              double tlm_rhs3_3 = J_3_0 * (dC_0 + dK1_0) + J_3_1 * (dC_1 + dK1_1) + J_3_2 * (dC_2 + dK1_2) + J_3_3 * (dC_3 + dK1_3) + (4.0759956452537702 / h) * dK1_3 + (9.20767942983308 / h) * dK2_3;
              // Solve W * dK3 = tlm_rhs3
              // Block 0: dK3 forward sub [M, O3, O]
              double tlm_y3_0 = tlm_rhs3_0;
              double tlm_y3_1 = tlm_rhs3_1 - L_1_0 * tlm_y3_0;
              double tlm_y3_2 = tlm_rhs3_2 - L_2_0 * tlm_y3_0 - L_2_1 * tlm_y3_1;
              // Block 1: dK3 forward sub [O2]
              double tlm_y3_3 = tlm_rhs3_3 - L_3_0 * tlm_y3_0 - L_3_1 * tlm_y3_1 - L_3_2 * tlm_y3_2;
              // Block 1: dK3 backward sub [O2]
              double dK3_3 = tlm_y3_3 / U_3_3;
              // Block 0: dK3 backward sub [M, O3, O]
              double dK3_2 = (tlm_y3_2 - U_2_3 * dK3_3) / U_2_2;
              double dK3_1 = (tlm_y3_1 - U_1_2 * dK3_2 - U_1_3 * dK3_3) / U_1_1;
              double dK3_0 = tlm_y3_0 / U_0_0;

              // --- δC update: δC += Σ M_i * δK_i ---
              dC_0 += dK1_0 + 6.1697947043828245 * dK2_0 + -0.42772256543218573 * dK3_0;
              dC_1 += dK1_1 + 6.1697947043828245 * dK2_1 + -0.42772256543218573 * dK3_1;
              dC_2 += dK1_2 + 6.1697947043828245 * dK2_2 + -0.42772256543218573 * dK3_2;
              dC_3 += dK1_3 + 6.1697947043828245 * dK2_3 + -0.42772256543218573 * dK3_3;
          } // end for (step)

          // Write back final δC to output view
          delta_C(0) = dC_0;
          delta_C(1) = dC_1;
          delta_C(2) = dC_2;
          delta_C(3) = dC_3;
      }
#endif
  };
}
