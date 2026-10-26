#include "gocart.hpp"
int main() {
    // Exception-safe runtime ownership: the guard finalizes when main returns (FR-013).
    Kokkos::ScopeGuard kokkos_scope;
    {
        double* dummy_ptr = new double[26 * 10 * 10 * 1];  // 26 species now
        mkpp::concentrations_view_t view(dummy_ptr, 26, 10, 10, 1);
        delete[] dummy_ptr;
    }
    return 0;
}
