# Device-callability gates for the generated solver path.
#
# Two independent checks live here. Host parity is always built: it binds photolysis rates as a
# host-space Kokkos::View and compares a Chapman integration against the reference numerics, proving
# the memory-space jvals entry points reproduce them after the breaking API change, with no
# accelerator required. The device compile gate is hardware-conditional: it instantiates the full
# generated call chain for a real Kokkos device backend, and when the linked Kokkos has no device
# execution space it is skipped with a clear status rather than silently passing, because host
# compilation alone does not demonstrate device callability.

cmake_minimum_required(VERSION 3.24)

# --- Host parity: always available, needs a mechanism header only -----------
# The device-callable fragments are included at file scope by the mechanism header, so this test
# compiles standalone without the compiled static library.
add_executable(test_device_callability_host_parity test_host_parity.cpp)
target_include_directories(test_device_callability_host_parity
                           PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_compile_definitions(
  test_device_callability_host_parity
  PRIVATE MKPP_E2E_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../e2e_validation/data")
target_link_libraries(test_device_callability_host_parity PRIVATE Kokkos::kokkos GTest::gtest
                                                                  GTest::gtest_main)
add_test(NAME DeviceCallabilityHostParity COMMAND test_device_callability_host_parity)

# --- Device runtime parity: always built, reports device-unverified ---------
# The parity test runs the host reference on every configuration and attempts a device comparison
# only when an accelerator is present at runtime. Without one it exits with the dedicated
# device-unverified code (2); SKIP_RETURN_CODE maps that exact code to an explicit "skipped"
# (non-pass) result rather than a false pass, while any genuine assertion failure (exit 1) still
# fails the test.
add_executable(test_device_runtime_parity test_device_runtime_parity.cpp)
target_include_directories(test_device_runtime_parity PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_compile_definitions(
  test_device_runtime_parity
  PRIVATE MKPP_E2E_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../e2e_validation/data")
target_link_libraries(test_device_runtime_parity PRIVATE Kokkos::kokkos GTest::gtest
                                                         GTest::gtest_main)
add_test(NAME DeviceRuntimeParity COMMAND test_device_runtime_parity)
set_tests_properties(DeviceRuntimeParity PROPERTIES SKIP_RETURN_CODE 2)

# --- Device compile gate: only when a device backend is linked --------------
if(NOT MKPP_DEVICE_GATE_ACTIVE)
  message(
    STATUS "MKPP device gate: no Kokkos device backend detected -> device-unverified (gate skipped)"
  )
  return()
endif()

# Pick the device execution space exposed by the linked Kokkos.
if(Kokkos_ENABLE_CUDA)
  set(MKPP_DEVICE_EXEC_SPACE "Kokkos::Cuda")
elseif(Kokkos_ENABLE_HIP)
  set(MKPP_DEVICE_EXEC_SPACE "Kokkos::Experimental::HIPSpace")
elseif(Kokkos_ENABLE_SYCL)
  set(MKPP_DEVICE_EXEC_SPACE "Kokkos::Experimental::SYCL")
endif()

add_executable(test_device_compile_gate test_device_compile_gate.cpp)
target_include_directories(test_device_compile_gate PRIVATE ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_compile_definitions(test_device_compile_gate
                           PRIVATE MKPP_DEVICE_EXEC_SPACE=${MKPP_DEVICE_EXEC_SPACE})
target_link_libraries(test_device_compile_gate PRIVATE mkpp_host Kokkos::kokkos GTest::gtest
                                                       GTest::gtest_main)

add_test(NAME DeviceCompileGate COMMAND test_device_compile_gate)
