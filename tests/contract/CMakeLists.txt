cmake_minimum_required(VERSION 3.24)

add_executable(test_batch_abi test_batch_abi.cpp)
target_link_libraries(test_batch_abi PRIVATE mkpp_host Kokkos::kokkos GTest::gtest)
add_test(NAME test_batch_abi COMMAND test_batch_abi)

# US3 (T026): the team+vector batched dense solve wired into the production dispatch
# (MechanismRegistry::batched_dense_solve). Only meaningful when KokkosKernels is enabled, matching
# the E6 `enabled` gate (FR-008).
if(MKPP_ENABLE_KOKKOS_KERNELS)
  add_executable(test_batched_production test_batched_production.cpp)
  target_link_libraries(test_batched_production PRIVATE mkpp_host Kokkos::kokkos GTest::gtest)
  add_test(NAME batched_production COMMAND test_batched_production)
endif()

find_package(
  Python3
  COMPONENTS Interpreter
  REQUIRED)
file(GLOB_RECURSE mkpp_codegen_inputs CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/mkpp/*.py"
     "${CMAKE_SOURCE_DIR}/src/mkpp/templates/*")
set(c_abi_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/c_abi_layout_generated")
string(MD5 c_abi_codegen_key "${CMAKE_CURRENT_BINARY_DIR}")
set(c_abi_staging_relative ".scratch/mkpp_c_abi_${c_abi_codegen_key}")
set(c_abi_staging_dir "${CMAKE_SOURCE_DIR}/${c_abi_staging_relative}")
set(c_abi_source "${c_abi_generated_dir}/mkpp_c_api.cpp")
add_custom_command(
  OUTPUT "${c_abi_source}"
  COMMAND ${CMAKE_COMMAND} -E make_directory "${c_abi_staging_dir}"
  COMMAND
    ${CMAKE_COMMAND} -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/src" ${Python3_EXECUTABLE} -m mkpp.cli
    compile "mechanisms/openatmos/chapman/mechanism.json" --test-env "example_env.yaml" --out
    "${c_abi_staging_relative}" --host-api --no-cache
  COMMAND ${CMAKE_COMMAND} -E make_directory "${c_abi_generated_dir}"
  COMMAND ${CMAKE_COMMAND} -E copy_directory "${c_abi_staging_dir}" "${c_abi_generated_dir}"
  DEPENDS ${mkpp_codegen_inputs} "${CMAKE_SOURCE_DIR}/mechanisms/openatmos/chapman/mechanism.json"
          "${CMAKE_SOURCE_DIR}/example_env.yaml"
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  VERBATIM)
set_source_files_properties("${c_abi_source}" PROPERTIES GENERATED TRUE)

add_executable(test_c_abi_layout test_c_abi_layout.cpp "${c_abi_source}")
target_include_directories(test_c_abi_layout PRIVATE "${c_abi_generated_dir}")
target_link_libraries(test_c_abi_layout PRIVATE mkpp_host Kokkos::kokkos GTest::gtest)
add_test(NAME c_abi_layout COMMAND test_c_abi_layout)
