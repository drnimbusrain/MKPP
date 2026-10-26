import re

from mkpp import format_eqn as _fe
from mkpp.format_eqn import _strength_reduce_squares, format_eqn


def test_strength_reduce_powers():
    # Integer power 2 (square)
    code_sq1 = "pow(S_0, 2)"
    assert _strength_reduce_squares(code_sq1) == "(S_0 * S_0)"

    code_sq2 = "pow(state(1), 2.0)"
    assert _strength_reduce_squares(code_sq2) == "(state(1) * state(1))"

    # Integer power 3 (cube)
    code_cube = "pow(state(1), 3)"
    assert _strength_reduce_squares(code_cube) == "((state(1) * state(1)) * state(1))"

    code_cube2 = "pow(x, 3.0)"
    assert _strength_reduce_squares(code_cube2) == "((x * x) * x)"

    # Integer power 1
    code_pow1 = "pow(x, 1)"
    assert _strength_reduce_squares(code_pow1) == "x"

    code_pow1_float = "pow(x, 1.0)"
    assert _strength_reduce_squares(code_pow1_float) == "x"

    # Integer power 0
    code_pow0 = "pow(x, 0)"
    assert _strength_reduce_squares(code_pow0) == "1.0"

    code_pow0_float = "pow(x, 0.0)"
    assert _strength_reduce_squares(code_pow0_float) == "1.0"

    # Fractional exponents MUST remain unchanged as pow(...)
    code_frac1 = "pow(x, 0.5)"
    assert _strength_reduce_squares(code_frac1) == "pow(x, 0.5)"

    code_frac2 = "pow(x, 2.5)"
    assert _strength_reduce_squares(code_frac2) == "pow(x, 2.5)"


def test_fallback_third_body_uses_supplied_environment_density():
    code = format_eqn("M_density * C_A", [type("Species", (), {"name": "A"})()], air_density=42.5)
    assert "42.5" in code
    assert "2.4476e+19" not in code


_BARE_MATH = re.compile(r"(?<![:.\w])(pow|log|log10|log2|exp|sqrt|fabs)\(")


def test_format_eqn_qualifies_kokkos_math_functions():
    """Generated expressions must use Kokkos::-qualified math for device portability."""
    sp = [type("Species", (), {"name": "A"})()]

    code = format_eqn("C_A ** 2.5", sp)
    assert "Kokkos::pow(" in code
    assert _BARE_MATH.search(code) is None

    code = format_eqn("log(C_A) + exp(C_A) + sqrt(C_A)", sp)
    assert "Kokkos::log(" in code
    assert "Kokkos::exp(" in code
    assert "Kokkos::sqrt(" in code
    assert _BARE_MATH.search(code) is None


def test_kokkos_math_helper_qualifies_maps_and_preserves():
    """The dialect helper qualifies bare math, maps C constants, and is idempotent."""
    helper = getattr(_fe, "_kokkos_math", None)
    assert helper is not None, "_kokkos_math helper must exist in mkpp.format_eqn"

    # Qualification of bare math.
    assert helper("pow(x, 0.5)") == "Kokkos::pow(x, 0.5)"
    assert helper("log(x)") == "Kokkos::log(x)"
    assert helper("exp(x)") == "Kokkos::exp(x)"
    assert helper("sqrt(x)") == "Kokkos::sqrt(x)"
    assert helper("fabs(x)") == "Kokkos::fabs(x)"

    # Idempotent: never double-qualify, and leave std:: untouched.
    assert helper("Kokkos::pow(x, 2)") == "Kokkos::pow(x, 2)"
    assert helper("std::pow(x, 2)") == "std::pow(x, 2)"

    # C math constants mapped to portable spellings.
    assert helper("M_LN10") == "Kokkos::log(10.0)"
    assert helper("M_PI") == "Kokkos::numbers::pi"

    # Strength-reduced multiplies and non-math tokens are preserved.
    assert helper("(x * x)") == "(x * x)"
    assert helper("state(3)") == "state(3)"
    assert helper("expr(3)") == "expr(3)"


def test_kokkos_math_full_falloff_expression():
    """An aerosol-falloff expression becomes fully Kokkos-qualified with no C macro left."""
    helper = getattr(_fe, "_kokkos_math", None)
    assert helper is not None
    src = "pow(0.45, 1.0/(pow(log(1.4e-18*state[12]), 2)/(M_LN10 * M_LN10) + 1.0))"
    out = helper(src)
    assert "M_LN10" not in out
    assert _BARE_MATH.search(out) is None
    assert "Kokkos::log(10.0)" in out
