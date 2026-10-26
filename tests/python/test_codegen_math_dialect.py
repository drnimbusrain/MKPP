"""Integration test: generated artifacts speak the Kokkos math dialect.

Compiles a real mechanism that exercises fractional ``pow``/``log`` aerosol
falloff terms through the full MKPP pipeline and asserts the emitted
rate/Jacobian sources and header contain no bare C math functions or ``M_LN10``
in device-reachable code.  This guarantee is what lets the generated path be
compiled for a device execution space.

**Validates: Requirements 1.1.**
"""

import re
import tempfile
from pathlib import Path

import pytest
from mkpp.codegen import generate_headers
from mkpp.parser import load_mechanism

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# saprc99_mini carries the aerosol-falloff pow/log/M_LN10 terms that previously
# emitted bare C math; chapman keeps a fast photolysis-only path covered too.
MECHANISM_FILES = {
    "saprc99_mini": PROJECT_ROOT / "mechanisms/openatmos/saprc99_mini/mechanism.json",
    "chapman": PROJECT_ROOT / "mechanisms/openatmos/chapman/mechanism.json",
}

# A bare math call is one not preceded by "::" or a word char (so Kokkos:: and
# std:: are ignored) and not part of a longer identifier (so expr(, state( are
# ignored).
_BARE_MATH = re.compile(r"(?<![:.\w])(?:pow|log10|log2|log|exp|sqrt|fabs)\(")
_KOKKOS_MATH = re.compile(r"Kokkos::(?:pow|log10|log2|log|exp|sqrt|fabs)\(")


def _generated_artifact_text(mech_path: Path) -> str:
    mech = load_mechanism(str(mech_path))
    with tempfile.TemporaryDirectory() as tmpdir:
        results = generate_headers(mech, out_dir=tmpdir, solver_name="ros3")
        paths = [Path(results["header"])] + [Path(p) for p in results.get("compiled_sources", [])]
        return "\n".join(p.read_text() for p in paths if p.exists())


@pytest.mark.parametrize("mech_name", sorted(MECHANISM_FILES))
def test_generated_code_has_no_bare_math(mech_name):
    mech_path = MECHANISM_FILES[mech_name]
    if not mech_path.exists():
        pytest.skip(f"{mech_name} mechanism file not found at {mech_path}")

    code = _generated_artifact_text(mech_path)

    offenders = _BARE_MATH.findall(code)
    assert not offenders, f"[{mech_name}] generated code contains bare C math: {sorted(set(offenders))}"
    assert "M_LN10" not in code, f"[{mech_name}] generated code still contains the M_LN10 macro"


@pytest.mark.parametrize("mech_name", sorted(MECHANISM_FILES))
def test_falloff_mechanism_emits_kokkos_math(mech_name):
    """The falloff-bearing mechanism must actually exercise the new dialect."""
    mech_path = MECHANISM_FILES[mech_name]
    if not mech_path.exists():
        pytest.skip(f"{mech_name} mechanism file not found at {mech_path}")

    code = _generated_artifact_text(mech_path)
    kokkos_hits = _KOKKOS_MATH.findall(code)

    if mech_name == "saprc99_mini":
        assert kokkos_hits, "saprc99_mini should emit Kokkos::-qualified math for its falloff terms"
    else:
        # chapman is photolysis-only; it may legitimately have no transcendental
        # math, but must never regress to bare math (covered by the other test).
        assert isinstance(kokkos_hits, list)
