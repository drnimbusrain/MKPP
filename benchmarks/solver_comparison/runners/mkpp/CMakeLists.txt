find_package(Python3 QUIET COMPONENTS Interpreter)
include("${CMAKE_SOURCE_DIR}/src/mkpp_host/cmake/MKPPCompiledMechanism.cmake")

set(TS1_SOURCE_ROOT "${CMAKE_SOURCE_DIR}/benchmarks/solver_comparison/sources/ts1/musica-config")
set(TS1_CONFIG "${TS1_SOURCE_ROOT}/ts1.json")
set(TS1_ENVIRONMENT "${TS1_SOURCE_ROOT}/environment.yaml")
# MKPP's path policy intentionally permits generated artifacts only beneath the source workspace. Do
# not use CMAKE_CURRENT_BINARY_DIR here: controlled Docker builds commonly place it in /tmp outside
# that policy boundary.
set(MKPP_TS1_GENERATED_DIR
    "${CMAKE_SOURCE_DIR}/build/solver_comparison/generated_ts1"
    CACHE PATH "MKPP TS1 generated artifacts (must be inside the source workspace)")
set(MKPP_TS1_GENERATED_HEADER "${MKPP_TS1_GENERATED_DIR}/ts1.hpp")

foreach(asset IN ITEMS "${TS1_CONFIG}" "${TS1_ENVIRONMENT}")
  if(NOT EXISTS "${asset}")
    message(FATAL_ERROR "FATAL ERROR: pinned OpenAtmos TS1 input is missing: ${asset}")
  endif()
endforeach()

file(SHA256 "${TS1_CONFIG}" TS1_CONFIG_SHA256)
file(SHA256 "${TS1_ENVIRONMENT}" TS1_ENVIRONMENT_SHA256)
if(NOT TS1_CONFIG_SHA256 STREQUAL
   "83b997e913194dd66046e7f76d54538dbed7e717f2ecb569e73d2c5f38b71732")
  message(FATAL_ERROR "FATAL ERROR: pinned OpenAtmos TS1 configuration hash mismatch")
endif()
if(NOT TS1_ENVIRONMENT_SHA256 STREQUAL
   "adbc312e85057066169bdcf8a93758faefbc8d604e0ec4f3f88a7334e6d8caa3")
  message(FATAL_ERROR "FATAL ERROR: pinned OpenAtmos TS1 environment hash mismatch")
endif()

# Both native solvers consume this exact OpenAtmos configuration.  The output header is a build
# artifact, not a second mechanism definition; --no-cache makes source/environment changes visible
# immediately to generated chemistry.
if(Python3_Interpreter_FOUND)
  # The source list is deterministic: register generated outputs without executing the expensive
  # generator during CMake configuration.
  set(MKPP_TS1_COMPILED_SOURCES)
  foreach(kernel IN ITEMS rates jacobian solve supernodal_factorize supernodal_solve)
    list(APPEND MKPP_TS1_COMPILED_SOURCES "${MKPP_TS1_GENERATED_DIR}/ts1/${kernel}.cpp")
  endforeach()
  file(GLOB_RECURSE MKPP_GENERATOR_DEPENDENCIES CONFIGURE_DEPENDS
       "${CMAKE_SOURCE_DIR}/src/mkpp/*.py" "${CMAKE_SOURCE_DIR}/src/mkpp/templates/*.j2")
  add_custom_command(
    OUTPUT "${MKPP_TS1_GENERATED_HEADER}" ${MKPP_TS1_COMPILED_SOURCES}
    BYPRODUCTS "${MKPP_TS1_GENERATED_DIR}/ts1/accessor.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/host_declarations.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/rates.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/jacobian.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/solve.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/supernodal_factorize.hpp"
               "${MKPP_TS1_GENERATED_DIR}/ts1/supernodal_solve.hpp"
    COMMAND "${CMAKE_COMMAND}" -E make_directory "${MKPP_TS1_GENERATED_DIR}"
    COMMAND
      "${CMAKE_COMMAND}" -E env "PYTHONPATH=${CMAKE_SOURCE_DIR}/src" "${Python3_EXECUTABLE}" -c
      "from mkpp.cli import main; main()" compile "${TS1_CONFIG}" --test-env "${TS1_ENVIRONMENT}"
      --out "${MKPP_TS1_GENERATED_DIR}" --mechanism-id ts1 --emit-manifest --no-cache
    DEPENDS "${TS1_CONFIG}" "${TS1_ENVIRONMENT}" ${MKPP_GENERATOR_DEPENDENCIES}
    COMMENT "Generating MKPP TS1 kernel from pinned OpenAtmos inputs"
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    VERBATIM)
  add_custom_target(mkpp_ts1_openatmos_binding DEPENDS "${MKPP_TS1_GENERATED_HEADER}")

  mkpp_add_compiled_mechanism(mkpp_ts1_compiled_kernels "${MKPP_TS1_GENERATED_DIR}" ts1 GENERATED
                              SOURCES ${MKPP_TS1_COMPILED_SOURCES})
  add_dependencies(mkpp_ts1_compiled_kernels mkpp_ts1_openatmos_binding)
  target_compile_features(mkpp_ts1_compiled_kernels PRIVATE cxx_std_23)
else()
  message(
    FATAL_ERROR
      "FATAL ERROR: Python is required to regenerate TS1 from pinned MUSICA v1 inputs; legacy fallback is not admitted"
  )
endif()

# The runner selects Chapman as well as TS1 through the host dispatcher.  Each selected generated
# mechanism must therefore register its implementation assets through the same artifact-discovery
# path.
mkpp_add_compiled_mechanism(mkpp_chapman_compiled_kernels "${CMAKE_SOURCE_DIR}/mkpp-generated"
                            chapman)
target_compile_features(mkpp_chapman_compiled_kernels PRIVATE cxx_std_23)

add_executable(solver_benchmark_mkpp main.cpp)
add_dependencies(solver_benchmark_mkpp mkpp_ts1_openatmos_binding)

target_compile_features(solver_benchmark_mkpp PRIVATE cxx_std_23)
target_compile_definitions(solver_benchmark_mkpp PRIVATE MKPP_HOST_SELECTED_MECHANISMS
                                                         MKPP_HOST_MECH_CHAPMAN MKPP_HOST_MECH_TS1)
target_include_directories(solver_benchmark_mkpp PRIVATE "${MKPP_TS1_GENERATED_DIR}"
                                                         ${CMAKE_SOURCE_DIR}/mkpp-generated)
target_link_libraries(solver_benchmark_mkpp PRIVATE mkpp_host Kokkos::kokkos)
target_link_libraries(solver_benchmark_mkpp PRIVATE mkpp_ts1_compiled_kernels
                                                    mkpp_chapman_compiled_kernels)
