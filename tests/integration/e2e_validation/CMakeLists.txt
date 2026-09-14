cmake_minimum_required(VERSION 3.24)

if(NOT TARGET mkpp_host)
  add_subdirectory(${CMAKE_SOURCE_DIR}/src/mkpp_host
                   ${CMAKE_BINARY_DIR}/mkpp_host)
endif()

include("${CMAKE_SOURCE_DIR}/src/mkpp_host/cmake/MKPPCompiledMechanism.cmake")

set(MECHS "chapman" "small_strato" "carbon" "ts1" "cracmm2")

foreach(MECH ${MECHS})
  string(TOUPPER "${MECH}" MECH_UPPER)
  if("${MECH}" STREQUAL "chapman")
    set(NUM_SPEC 3)
  elseif("${MECH}" STREQUAL "small_strato")
    set(NUM_SPEC 7)
  elseif("${MECH}" STREQUAL "carbon")
    set(NUM_SPEC 10)
  elseif("${MECH}" STREQUAL "gocart")
    set(NUM_SPEC 12)
  elseif("${MECH}" STREQUAL "saprc99")
    set(NUM_SPEC 82)
  elseif("${MECH}" STREQUAL "ts1")
    set(NUM_SPEC 210)
  elseif("${MECH}" STREQUAL "cracmm2")
    set(NUM_SPEC 271)
  else()
    set(NUM_SPEC 100)
  endif()

  add_executable(e2e_${MECH}_runner test_e2e_mechanisms.cpp)
  mkpp_add_compiled_mechanism(e2e_${MECH}_compiled_kernels
                              "${CMAKE_SOURCE_DIR}/mkpp-generated" ${MECH})
  target_include_directories(e2e_${MECH}_runner
                             PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
  target_link_libraries(
    e2e_${MECH}_runner PRIVATE mkpp_host Kokkos::kokkos GTest::gtest_main
                               GTest::gtest e2e_${MECH}_compiled_kernels)
  target_compile_definitions(
    e2e_${MECH}_runner
    PRIVATE MECH_HEADER=${MECH}.hpp
            MECH_SOLVER_NAMESPACE=mkpp::generated::${MECH}
            MKPP_HOST_SELECTED_MECHANISMS
            MKPP_HOST_MECH_${MECH_UPPER}
            NUM_SPECIES_MACRO=${NUM_SPEC})

  add_test(NAME E2ESolver_${MECH} COMMAND e2e_${MECH}_runner)

  # Only add the compare test if we have a KPP baseline CSV for it
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/data/kpp_baseline_${MECH}.csv)
    add_test(
      NAME E2ECompare_${MECH}
      COMMAND
        python3 ${CMAKE_CURRENT_SOURCE_DIR}/compare_baselines.py --baseline
        ${CMAKE_CURRENT_SOURCE_DIR}/data/kpp_baseline_${MECH}.csv --target
        e2e_output.csv --hpp ${CMAKE_SOURCE_DIR}/mkpp-generated/${MECH}.hpp)
    set_tests_properties(E2ECompare_${MECH} PROPERTIES DEPENDS
                                                       E2ESolver_${MECH})
  endif()
endforeach()

foreach(MECH gocart saprc99 saprcnov saprc99_mini)
  mkpp_add_compiled_mechanism(e2e_${MECH}_compiled_kernels
                              "${CMAKE_SOURCE_DIR}/mkpp-generated" ${MECH})
endforeach()

add_executable(benchmark_runner benchmark_runner.cpp)
target_include_directories(benchmark_runner
                           PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_link_libraries(
  benchmark_runner
  PRIVATE mkpp_host
          Kokkos::kokkos
          e2e_chapman_compiled_kernels
          e2e_small_strato_compiled_kernels
          e2e_carbon_compiled_kernels
          e2e_ts1_compiled_kernels
          e2e_gocart_compiled_kernels
          e2e_saprc99_compiled_kernels
          e2e_saprcnov_compiled_kernels
          e2e_saprc99_mini_compiled_kernels
          e2e_cracmm2_compiled_kernels)

# SciPy reference validation test
add_executable(test_scipy_validation test_scipy_validation.cpp)
target_include_directories(test_scipy_validation
                           PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_link_libraries(
  test_scipy_validation PRIVATE mkpp_host Kokkos::kokkos GTest::gtest_main
                                GTest::gtest e2e_chapman_compiled_kernels)
target_compile_definitions(
  test_scipy_validation
  PRIVATE REFERENCE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_test(NAME mkpp_scipy_validation COMMAND test_scipy_validation)

add_executable(test_e2e_batched test_e2e_batched.cpp)
target_include_directories(test_e2e_batched
                           PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_link_libraries(
  test_e2e_batched PRIVATE mkpp_host Kokkos::kokkos GTest::gtest
                           e2e_chapman_compiled_kernels)
add_test(NAME test_e2e_batched COMMAND test_e2e_batched)
if(MKPP_ENABLE_MICM_COMPARE)
  add_executable(benchmark_mkpp_vs_micm benchmark_mkpp_vs_micm.cpp)
  target_include_directories(
    benchmark_mkpp_vs_micm
    PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated
    PRIVATE ${CMAKE_BINARY_DIR}/_deps/micm-src/include)
  target_link_libraries(benchmark_mkpp_vs_micm PRIVATE mkpp_host Kokkos::kokkos
                                                       micm)
  add_test(NAME run_benchmark_mkpp_vs_micm COMMAND benchmark_mkpp_vs_micm)

  add_executable(benchmark_mkpp_vs_micm_ts1 benchmark_mkpp_vs_micm_ts1.cpp)
  target_include_directories(
    benchmark_mkpp_vs_micm_ts1
    PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated
    PRIVATE ${CMAKE_BINARY_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/_deps/micm-src/include)
  target_link_libraries(benchmark_mkpp_vs_micm_ts1 PRIVATE mkpp_host
                                                           Kokkos::kokkos micm)
  add_test(NAME run_benchmark_mkpp_vs_micm_ts1
           COMMAND benchmark_mkpp_vs_micm_ts1)
endif()
