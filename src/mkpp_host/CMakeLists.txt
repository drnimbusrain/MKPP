cmake_minimum_required(VERSION 3.24)
project(mkpp_host LANGUAGES CXX C)

# See the note in the MKPP root CMakeLists: a parent build that added MKPP via add_subdirectory
# already provides the Kokkos targets, and re-running find_package would resolve against a different
# installation.
if(NOT TARGET Kokkos::kokkos)
  find_package(Kokkos 5.1.1 REQUIRED)
endif()

add_library(mkpp_host INTERFACE)
add_library(MKPP::mkpp_host ALIAS mkpp_host)
target_compile_features(mkpp_host INTERFACE cxx_std_23)

target_include_directories(
  mkpp_host
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
            $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../mkpp-generated>
            $<INSTALL_INTERFACE:include>)
target_link_libraries(mkpp_host INTERFACE Kokkos::kokkos)

if(MKPP_ENABLE_ADJOINT)
  target_compile_definitions(mkpp_host INTERFACE MKPP_ENABLE_ADJOINT)
endif()

if(MKPP_ENABLE_KOKKOS_KERNELS)
  if(TARGET KokkosKernels::kokkoskernels)
    target_link_libraries(mkpp_host INTERFACE KokkosKernels::kokkoskernels)
  elseif(TARGET kokkoskernels)
    target_link_libraries(mkpp_host INTERFACE kokkoskernels)
  endif()
  target_compile_definitions(mkpp_host INTERFACE MKPP_HAS_KOKKOS_KERNELS)
endif()
