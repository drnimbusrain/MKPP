#!/usr/bin/env bash
set -euo pipefail
# Run from repository root. Prepare the Python environment first (see NATIVE_RUN_REPORT.md).
repo="$PWD"
deps="$(dirname "$repo")/upstream"
export PATH="$repo/.venv/bin:$PATH"
git clone --branch 5.1.1 --depth 1 https://github.com/kokkos/kokkos.git "$deps/kokkos"
git clone https://github.com/NCAR/micm.git "$deps/micm"
git -C "$deps/micm" checkout 97ac9e5d8aadd345c242722ee8274d71dfe0f73e
cmake -S "$deps/kokkos" -B "$deps/kokkos-5.1.1-build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23 \
  -DCMAKE_INSTALL_PREFIX="$deps/kokkos-5.1.1-install" \
  -DKokkos_ENABLE_SERIAL=ON -DKokkos_ENABLE_OPENMP=OFF \
  -DKokkos_ENABLE_CUDA=OFF -DKokkos_ENABLE_TESTS=OFF
cmake --build "$deps/kokkos-5.1.1-build" --parallel 2
cmake --install "$deps/kokkos-5.1.1-build"
cmake -S . -B build/native-kokkos-5.1.1 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ROOT="$deps/kokkos-5.1.1-install" -DMKPP_ENABLE_SOLVER_COMPARISON=ON \
  -DMKPP_ENABLE_KOKKOS_KERNELS=OFF -DBUILD_TESTING=OFF -DMICM_ENABLE_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_MICM="$deps/micm" -DPython3_EXECUTABLE="$repo/.venv/bin/python"
cmake --build build/native-kokkos-5.1.1 \
  --target solver_benchmark_mkpp solver_benchmark_micm solver_benchmark_micm_ts1 --parallel 1
# Diagnostic executions only; these do not create admitted performance evidence.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
build/native-kokkos-5.1.1/benchmarks/solver_comparison/runners/mkpp/solver_benchmark_mkpp \
  --mechanism chapman --cells 1 --steps 10 --dt 60 --warmups 3 --repetitions 3
build/native-kokkos-5.1.1/benchmarks/solver_comparison/runners/mkpp/solver_benchmark_mkpp \
  --mechanism ts1 --cells 1 --warmups 3 --repetitions 3
# Locate MICM executables because upstream versions can set RUNTIME_OUTPUT_DIRECTORY.
micm_chapman=$(find build/native-kokkos-5.1.1 -type f -name solver_benchmark_micm -perm -u+x -print -quit)
micm_ts1=$(find build/native-kokkos-5.1.1 -type f -name solver_benchmark_micm_ts1 -perm -u+x -print -quit)
NUM_CELLS=1 NUM_STEPS=10 DT_SECONDS=60 WARMUPS=3 REPETITIONS=3 "$micm_chapman"
NUM_CELLS=1 NUM_STEPS=10 DT_SECONDS=60 WARMUPS=3 REPETITIONS=3 "$micm_ts1"
