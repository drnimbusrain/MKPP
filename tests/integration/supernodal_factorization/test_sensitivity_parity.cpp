#include <cmath>
#include <cstdio>
#include "chapman.hpp"

int main() {
  // Exception-safe runtime ownership: the guard finalizes when main returns (FR-013).
  Kokkos::ScopeGuard kokkos_scope;
  {
  using Solver = mkpp::generated::chapman::SolverKernels<Kokkos::DefaultExecutionSpace>;
  Kokkos::View<double*> state("state", 4), delta("delta", 4), rates("rates", 4);
  Kokkos::View<double**> jacobian("jacobian", 4, 4), adjoint("adjoint", 4, 4);
  auto hstate = Kokkos::create_mirror_view(state), hdelta = Kokkos::create_mirror_view(delta);
  for (int i = 0; i < 4; ++i) { hstate(i) = 1.0 + i; hdelta(i) = 0.25 * (i + 1); }
  Kokkos::deep_copy(state, hstate); Kokkos::deep_copy(delta, hdelta);
  Kokkos::View<double*> jvals_host("jvals", 2);
  auto hjv = Kokkos::create_mirror_view(jvals_host);
  hjv(0) = 2.0e-5; hjv(1) = 1.0e-3;
  Kokkos::deep_copy(jvals_host, hjv);
  Kokkos::View<const double*, Kokkos::DefaultExecutionSpace::memory_space> jvals(jvals_host);
  Solver solver;
  solver.compute_jacobian(state, jacobian, jvals);
  solver.compute_adjoint(state, adjoint, jvals);
  solver.compute_tlm(state, delta, rates, jvals);
  auto hj = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, jacobian);
  auto ha = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, adjoint);
  auto hr = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, rates);
  for (int i = 0; i < 4; ++i) for (int k = 0; k < 4; ++k)
    if (std::abs(hj(i,k) - ha(k,i)) > 1.0e-12) return 1;
  for (int i = 0; i < 4; ++i) { double expected = 0.0; for (int k = 0; k < 4; ++k) expected += hj(i,k) * hdelta(k); if (std::abs(hr(i) - expected) > 1.0e-12) return 2; }
  }
  return 0;
}
