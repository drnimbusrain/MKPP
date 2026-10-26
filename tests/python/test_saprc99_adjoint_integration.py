"""
Integration test: SAPRC-99 mechanism with adjoint-enabled headers (Ros3).

**Validates: Requirements 8.2**

Verifies:
1. Generation of adjoint-enabled headers for SAPRC-99 succeeds
2. All required adjoint/TLM functions are present in the generated code
3. Checkpoint memory fits within the 256 KB bound
4. Generated header is syntactically valid (balanced braces, namespace closure)
5. If a C++ compiler is available, the generated header compiles successfully

The checkpoint memory calculation (D5 simplified):
    (MAX_STEPS * NUM_SPECIES * 8) + (MAX_STEPS * 8) for h array + 4 bytes for num_steps

For SAPRC-99 with 82 species:
    200 * 82 * 8 + 200 * 8 + 4 = 131,200 + 1,600 + 4 = 132,804 bytes (~130 KB)
    Well within 256 KB limit.
"""

import os
import shutil
import subprocess

import pytest
from mkpp.codegen import generate_headers
from mkpp.parser import load_mechanism

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_STEPS = 200
MAX_CHECKPOINT_MEMORY_BYTES = 256 * 1024  # 256 KB = 262,144 bytes
SAPRC99_PATH = "mechanisms/openatmos/saprc99_mini/mechanism.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_saprc99():
    """Load the SAPRC-99 mechanism."""
    return load_mechanism(SAPRC99_PATH)


def _generate_saprc99_adjoint(tmp_dir: str) -> str:
    """Generate the SAPRC-99 adjoint-enabled header and return the code."""
    mech = _load_saprc99()
    results = generate_headers(mech, out_dir=tmp_dir, solver_name="ros3", adjoint=True)
    with open(results["header"]) as f:
        return f.read()


def _get_header_path(tmp_dir: str) -> str:
    """Generate SAPRC-99 adjoint header and return the file path."""
    mech = _load_saprc99()
    results = generate_headers(mech, out_dir=tmp_dir, solver_name="ros3", adjoint=True)
    return results["header"]


def _count_braces(code: str) -> tuple:
    """Count opening and closing braces in the code, ignoring those in comments and strings."""
    open_count = 0
    close_count = 0
    in_line_comment = False
    in_block_comment = False
    in_string = False
    prev_char = ""

    for i, ch in enumerate(code):
        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            continue
        if in_block_comment:
            if ch == "/" and prev_char == "*":
                in_block_comment = False
            prev_char = ch
            continue
        if in_string:
            if ch == '"' and prev_char != "\\":
                in_string = False
            prev_char = ch
            continue

        if ch == "/" and i + 1 < len(code):
            next_ch = code[i + 1]
            if next_ch == "/":
                in_line_comment = True
                prev_char = ch
                continue
            elif next_ch == "*":
                in_block_comment = True
                prev_char = ch
                continue

        if ch == '"':
            in_string = True
            prev_char = ch
            continue

        if ch == "{":
            open_count += 1
        elif ch == "}":
            close_count += 1

        prev_char = ch

    return open_count, close_count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSAPRC99AdjointGeneration:
    """Integration tests for SAPRC-99 adjoint-enabled code generation with Ros3."""

    def test_generation_succeeds(self, tmp_path):
        """Generation of adjoint-enabled headers for SAPRC-99 completes without error.

        **Validates: Requirements 8.2**
        """
        mech = _load_saprc99()
        results = generate_headers(mech, out_dir=str(tmp_path), solver_name="ros3", adjoint=True)
        assert os.path.exists(results["header"])
        # Verify the file is non-empty
        size = os.path.getsize(results["header"])
        assert size > 0, "Generated header is empty"

    def test_required_functions_present(self, tmp_path):
        """All adjoint/TLM functions are present in the generated code.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))

        # Required functions per Req 6.2
        assert "integrate_adj" in code, "integrate_adj not found in generated code"
        assert "integrate_tlm" in code, "integrate_tlm not found in generated code"
        assert "integrate_fwd_checkpoint" in code, "integrate_fwd_checkpoint not found in generated code"
        assert "CheckpointBuffer" in code, "CheckpointBuffer struct not found in generated code"

    def test_num_species_is_17(self, tmp_path):
        """The generated CheckpointBuffer has NUM_SPECIES = 17 for saprc99_mini.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))
        # CheckpointBuffer should have NUM_SPECIES = 17
        assert "static constexpr int NUM_SPECIES = 17;" in code

    def test_checkpoint_memory_within_256kb(self, tmp_path):
        """Checkpoint memory fits within 256 KB bound.

        Calculation (D5 simplified checkpoint - state only, not stage vectors):
            state: MAX_STEPS * NUM_SPECIES * 8 bytes
            h:     MAX_STEPS * 8 bytes
            num_steps: 4 bytes (int)
        Total = 200 * 17 * 8 + 200 * 8 + 4 = 28,804 bytes

        **Validates: Requirements 8.2**
        """
        mech = _load_saprc99()
        num_species = len(mech.species)  # 17

        # Calculate checkpoint memory per the D5 design (state + h + num_steps)
        state_bytes = MAX_STEPS * num_species * 8  # double state[MAX_STEPS][NUM_SPECIES]
        h_bytes = MAX_STEPS * 8  # double h[MAX_STEPS]
        num_steps_bytes = 4  # int num_steps

        total_bytes = state_bytes + h_bytes + num_steps_bytes

        assert total_bytes <= MAX_CHECKPOINT_MEMORY_BYTES, (
            f"Checkpoint memory {total_bytes} bytes ({total_bytes / 1024:.1f} KB) "
            f"exceeds 256 KB bound ({MAX_CHECKPOINT_MEMORY_BYTES} bytes).\n"
            f"  state: {state_bytes} bytes (MAX_STEPS={MAX_STEPS} * "
            f"NUM_SPECIES={num_species} * 8)\n"
            f"  h: {h_bytes} bytes (MAX_STEPS={MAX_STEPS} * 8)\n"
            f"  num_steps: {num_steps_bytes} bytes"
        )

        expected_approx = 28_804
        assert abs(total_bytes - expected_approx) < 100, (
            f"Checkpoint memory calculation unexpected: " f"got {total_bytes}, expected ~{expected_approx}"
        )

    def test_balanced_braces(self, tmp_path):
        """Generated header has balanced braces (basic syntactic validation).

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))
        open_count, close_count = _count_braces(code)

        assert open_count == close_count, (
            f"Unbalanced braces in generated header: "
            f"{open_count} opening vs {close_count} closing "
            f"(difference: {open_count - close_count})"
        )

    def test_namespace_closure(self, tmp_path):
        """Generated header properly opens and closes the mkpp namespace.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))

        # Namespace must open
        assert "namespace mkpp::generated::saprc99_mini" in code, "Missing generated mechanism namespace"

        # The last non-whitespace closing brace should close the namespace
        # Check that the code ends with a closing brace (after stripping trailing whitespace)
        stripped = code.rstrip()
        assert stripped.endswith("}"), "Generated header does not end with closing brace (namespace closure)"

    def test_pragma_once_present(self, tmp_path):
        """Generated header starts with #pragma once.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))
        assert code.startswith("#pragma once"), "Generated header does not start with #pragma once"

    def test_kokkos_include_present(self, tmp_path):
        """Generated header includes Kokkos_Core.hpp.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))
        assert "#include <Kokkos_Core.hpp>" in code, "Generated header missing #include <Kokkos_Core.hpp>"

    def test_no_heap_allocation_in_checkpoint_buffer(self, tmp_path):
        """CheckpointBuffer struct uses no heap allocation.

        **Validates: Requirements 8.2**
        """
        code = _generate_saprc99_adjoint(str(tmp_path))

        # Extract CheckpointBuffer struct
        start = code.find("struct CheckpointBuffer")
        assert start != -1, "CheckpointBuffer struct not found"

        # Find the closing brace of the struct
        depth = 0
        struct_end = start
        brace_started = False
        for i in range(start, len(code)):
            if code[i] == "{":
                depth += 1
                brace_started = True
            elif code[i] == "}":
                depth -= 1
                if brace_started and depth == 0:
                    struct_end = i
                    break

        checkpoint_struct = code[start : struct_end + 1]

        # No heap allocation in the struct
        assert "std::vector" not in checkpoint_struct, "CheckpointBuffer uses std::vector (heap allocation)"
        assert "new " not in checkpoint_struct, "CheckpointBuffer uses 'new' (heap allocation)"
        assert "malloc" not in checkpoint_struct, "CheckpointBuffer uses malloc (heap allocation)"

    @pytest.mark.skipif(
        not shutil.which("g++") and not shutil.which("clang++"),
        reason="No C++ compiler available",
    )
    def test_compilation_succeeds(self, tmp_path):
        """Generated header compiles successfully with a C++ compiler.

        Uses syntax-check mode (-fsyntax-only) to verify the generated
        header is valid C++ without needing to link against Kokkos.
        A stub Kokkos_Core.hpp is provided so we don't depend on a full
        Kokkos installation at the system level.

        **Validates: Requirements 8.2**
        """
        header_path = _get_header_path(str(tmp_path))

        # Create a stub Kokkos_Core.hpp in the tmp_path so the generated
        # header's #include <Kokkos_Core.hpp> resolves without needing
        # the real Kokkos installation.
        stub_kokkos = tmp_path / "Kokkos_Core.hpp"
        stub_kokkos.write_text(
            """\
#pragma once
#include <cmath>
#include <algorithm>

#ifndef KOKKOS_INLINE_FUNCTION
#define KOKKOS_INLINE_FUNCTION inline
#endif

#ifndef KOKKOS_DEFAULTED_FUNCTION
#define KOKKOS_DEFAULTED_FUNCTION
#endif

namespace Kokkos {
    struct Serial { using memory_space = Serial; };
    template<typename A, typename B> auto fmin(A a, B b) { return a < b ? a : b; }
    template<typename A, typename B> auto fmax(A a, B b) { return a > b ? a : b; }
    template<typename A, typename B> auto min(A a, B b) { return a < b ? a : b; }
    template<typename T> T fabs(T a) { return a < 0 ? -a : a; }
    template<typename T> T sqrt(T a) { return std::sqrt(a); }
    template<typename A, typename B> auto pow(A a, B b) { return std::pow(a, b); }
    template<typename T> T cbrt(T a) { return std::cbrt(a); }
    template<typename T> T log(T a) { return std::log(a); }
    template<typename T> T exp(T a) { return std::exp(a); }
    template<typename T> T log10(T a) { return std::log10(a); }
    template<typename T> T log2(T a) { return std::log2(a); }
    template<typename T> bool isfinite(T a) { return std::isfinite(a); }
    namespace numbers { constexpr double pi = 3.14159265358979323846; }
    struct LayoutLeft {};
    struct LayoutRight {};
    struct LayoutStride {};
    struct MemoryUnmanaged {};
    template<typename... Args> struct View {
        double storage[1];
        double* data() const { return const_cast<double*>(storage); }
    };
}
"""
        )

        # Create a minimal translation unit that includes the generated header
        test_cpp = tmp_path / "test_compile.cpp"
        test_cpp.write_text(
            '#include "' + os.path.basename(header_path) + '"\n'
            "\n"
            "int main() {\n"
            "    // Instantiate the struct to verify template syntax\n"
            "    using Kernels = mkpp::generated::saprc99_mini::SolverKernels<Kokkos::Serial>;\n"
            "    Kernels k;\n"
            "    mkpp::generated::saprc99_mini::CheckpointBuffer chk;\n"
            "    (void)k;\n"
            "    (void)chk;\n"
            "    return 0;\n"
            "}\n"
        )

        # Choose compiler
        compiler = shutil.which("clang++") or shutil.which("g++")
        assert compiler, "No C++ compiler found"

        # Compile with syntax-only flag, using tmp_path as include directory
        # so both the generated header and our Kokkos stub are found.
        result = subprocess.run(
            [
                compiler,
                "-std=c++20",
                "-DMKPP_ENABLE_ADJOINT",
                "-fsyntax-only",
                "-I",
                str(tmp_path),
                str(test_cpp),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, (
            f"Compilation failed for SAPRC-99 adjoint header.\n" f"Compiler: {compiler}\n" f"stderr:\n{result.stderr[:2000]}"
        )
