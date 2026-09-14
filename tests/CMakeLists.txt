add_subdirectory(contract)
add_subdirectory(integration/e2e_validation)
add_subdirectory(integration/solver_benchmark)
add_subdirectory(integration/supernodal_factorization)

include("${CMAKE_SOURCE_DIR}/src/mkpp_host/cmake/MKPPCompiledMechanism.cmake")

add_executable(test_host_dispatcher test_host_dispatcher.cpp)
target_link_libraries(
  test_host_dispatcher
  PRIVATE mkpp_host
          Kokkos::kokkos
          GTest::gtest
          e2e_chapman_compiled_kernels
          e2e_small_strato_compiled_kernels
          e2e_carbon_compiled_kernels
          e2e_ts1_compiled_kernels
          e2e_gocart_compiled_kernels
          e2e_saprc99_compiled_kernels
          e2e_saprcnov_compiled_kernels
            e2e_saprc99_mini_compiled_kernels
            e2e_cracmm2_compiled_kernels)
add_test(NAME test_host_dispatcher COMMAND test_host_dispatcher)
