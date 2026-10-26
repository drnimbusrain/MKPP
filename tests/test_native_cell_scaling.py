"""Compile and execute heterogeneous C ABI batches and both LU backends."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from mkpp.codegen import generate_headers
from mkpp.parser import load_mechanism

ROOT = Path(__file__).resolve().parents[1]


def test_heterogeneous_cells_and_reference_backend(tmp_path):
    kokkos = os.environ.get("MKPP_TEST_KOKKOS_ROOT")
    if not kokkos:
        pytest.skip("set MKPP_TEST_KOKKOS_ROOT to run the native Kokkos integration gate")
    document = json.loads((ROOT / "mechanisms/openatmos/chapman/mechanism.json").read_text())
    document["reactions"][1].update({"A": 1e-3, "C": -400.0})
    source = tmp_path / "mechanism.json"
    source.write_text(json.dumps(document))
    mechanism = load_mechanism(str(source))
    generated = tmp_path / "generated"
    generate_headers(mechanism, out_dir=generated, generate_host_api=True, emit_reference_backend=True)
    harness = ROOT / "tests/native/heterogeneous_cells.cpp"
    cmake = tmp_path / "CMakeLists.txt"
    cmake.write_text(f"""cmake_minimum_required(VERSION 3.24)
project(cell_scaling LANGUAGES CXX)
find_package(Kokkos 5.1.1 REQUIRED)
include("{ROOT}/src/mkpp_host/cmake/MKPPCompiledMechanism.cmake")
mkpp_add_compiled_mechanism(kernels "{generated}" chapman)
foreach(backend IN ITEMS compact reference)
  add_executable(${{backend}} "{harness}" "{generated}/mkpp_c_api.cpp")
  target_compile_features(${{backend}} PRIVATE cxx_std_23)
  target_include_directories(${{backend}} PRIVATE "{generated}" "{ROOT}/src/mkpp_host/include")
  target_link_libraries(${{backend}} PRIVATE kernels Kokkos::kokkos)
endforeach()
target_compile_definitions(reference PRIVATE MKPP_USE_UNROLLED_REFERENCE)
""")
    executable_dir = Path(sys.executable).parent
    cmake_bin = shutil.which("cmake") or str(executable_dir / "cmake")
    ninja = shutil.which("ninja") or str(executable_dir / "ninja")
    build = tmp_path / "build"
    subprocess.run(
        [
            cmake_bin,
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DKokkos_ROOT={kokkos}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([cmake_bin, "--build", str(build), "--parallel", "2"], check=True, capture_output=True, text=True, timeout=180)
    fortran = shutil.which("gfortran")
    if fortran:
        subprocess.run(
            [fortran, "-fsyntax-only", "-J", str(build), str(generated / "mkpp_mod.f90")],
            check=True,
            capture_output=True,
            text=True,
        )
    outputs = [
        json.loads(subprocess.check_output([str(build / backend)], text=True, timeout=30)) for backend in ("compact", "reference")
    ]
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-10, atol=1e-20)
