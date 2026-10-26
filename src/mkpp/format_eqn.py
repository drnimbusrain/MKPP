"""SymPy-to-C expression formatting and peephole optimizations.

Converts symbolic ODE right-hand-side and Jacobian expressions into
C++ code strings suitable for Kokkos device kernels.
"""

import math
import re


def _fold_numeric_falloff_powers(code: str) -> str:
    """Fold pure numeric falloff powers emitted by SymPy's C printer."""
    number = r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"
    pattern = re.compile(rf"pow\(({number}),\s*1\.0/\(1\.0 \+ ({number})/pow\(M_LN10, 2\)\)\)")

    def replace(match):
        base = float(match.group(1))
        coefficient = float(match.group(2))
        exponent = 1.0 / (1.0 + coefficient / (math.log(10.0) ** 2))
        return _clean_float_literals(f"{base ** exponent:g}")

    return pattern.sub(replace, code)


# SymPy's C printer emits unqualified math (pow/log/exp/sqrt) and C library
# macros (M_LN10, M_PI, ...).  Device compilers do not reliably resolve those
# spellings across CUDA/HIP/SYCL/OpenMP-offload, and the hand-written solver
# code already uses Kokkos:: math.  Qualify the generated expressions to the
# same dialect so the whole generated path is device-callable and consistent.
# The lookbehind keeps this idempotent: it never re-qualifies Kokkos:: or std::.
_KOKKOS_MATH_FUNCS = re.compile(r"(?<![:.\w])(pow|log10|log2|log|exp|sqrt|fabs)\(")
_KOKKOS_MATH_CONSTANTS = {
    "M_LN10": "Kokkos::log(10.0)",
    "M_LN2": "Kokkos::log(2.0)",
    "M_PI": "Kokkos::numbers::pi",
    "M_E": "Kokkos::numbers::e",
}
_KOKKOS_MATH_CONSTANT_RE = re.compile(r"\b(" + "|".join(sorted(_KOKKOS_MATH_CONSTANTS, key=len, reverse=True)) + r")\b")


def _kokkos_math(code: str) -> str:
    """Rewrite bare C math functions and macros into Kokkos:: spellings.

    Runs after strength reduction so integer powers stay as multiplies and only
    the transcendental calls (fractional ``pow``, ``log``, ``exp``, ``sqrt``,
    ``fabs``) and the C math constants are qualified.  Idempotent and safe on
    already-qualified ``Kokkos::``/``std::`` names and non-math tokens.
    """
    code = _KOKKOS_MATH_CONSTANT_RE.sub(lambda m: _KOKKOS_MATH_CONSTANTS[m.group(1)], code)
    return _KOKKOS_MATH_FUNCS.sub(r"Kokkos::\1(", code)


def _clean_float_literals(code: str) -> str:
    """Clean up verbose floating point literals (e.g. 7.9999999999999998e-12 -> 8e-12)."""
    pattern = re.compile(r"\b(?<![a-zA-Z_])\d+\.\d+(?:[eE][-+]?\d+)?\b")

    def replace(match):
        val_str = match.group(0)
        try:
            val = float(val_str)
            cleaned = str(val)
            if "." not in cleaned and "e" not in cleaned and "E" not in cleaned:
                cleaned += ".0"
            return cleaned
        except ValueError:
            return val_str

    return pattern.sub(replace, code)


def _strength_reduce_squares(code: str) -> str:
    """Replace generated pow(x, n) terms with explicit multiplication or identity."""
    term = r"(?:state\(\d+\)|Ynew_\d+|S_\d+|[a-zA-Z_][a-zA-Z0-9_]*)"
    # pow(x, 2) -> (x * x)
    code = re.sub(rf"\bpow\(({term}),\s*2(?:\.0)?\)", r"(\1 * \1)", code)
    # pow(x, 3) -> ((x * x) * x)
    code = re.sub(rf"\bpow\(({term}),\s*3(?:\.0)?\)", r"((\1 * \1) * \1)", code)
    # pow(x, 1) -> x
    code = re.sub(rf"\bpow\(({term}),\s*1(?:\.0)?\)", r"\1", code)
    # pow(x, 0) -> 1.0
    code = re.sub(rf"\bpow\(({term}),\s*0(?:\.0)?\)", r"1.0", code)
    return code


def format_eqn(
    eqn_str,
    species_list,
    state_var="state",
    use_parentheses=True,
    keep_env_symbols=False,
    temperature: float = 300.0,
    air_density: float = 2.4476e19,
):
    """Convert a symbolic ODE expression string into a C++ code string.

    Parameters
    ----------
    eqn_str : str
        SymPy-parseable expression string (e.g., from KPP rate law).
    species_list : list
        Ordered species definitions used to map C_X symbols to state indices.
    state_var : str
        Name of the state array variable in generated C++ code.
    use_parentheses : bool
        If True, emit ``state(idx)``; if False, emit ``state_idx``.
    keep_env_symbols : bool
        If False (default), environmental parameters (Temp, RH) are substituted
        with constants (Temp=300.0). This produces isothermal code suitable only
        for constant-temperature simulations.
        If True, Temp and RH are emitted as C++ variable references, and the
        generated function MUST accept temp/rh as parameters.

    WARNING
    -------
    The current default (keep_env_symbols=False) silently folds temperature to
    300 K. For temperature-dependent chemistry, callers MUST pass
    keep_env_symbols=True and ensure the generated C++ function accepts temp/rh
    as parameters.

    .. deprecated:: planned for MKPP 2.0
        The default will be inverted to keep_env_symbols=True for physical
        correctness. This requires all generated function signatures to accept
        a ``temp`` parameter unconditionally.
    """
    import sympy as sp

    # 1. Clean up double negatives and malformed trailing dots on floats
    s = str(eqn_str).replace("--", "+").replace("^+", "^").replace("**+", "**")
    s = re.sub(r"(\d+\.\d+)\.+", r"\1", s)
    if s == "0":
        return "0.0"

    # 2. Try to use SymPy's C-code generator for robust math formatting
    try:
        expr = sp.sympify(s)
        # Substitute legacy KPP dummy vars to 1.0 before C-code generation
        subs_dict = {
            # Legacy KPP dummy/fixed species (not real state variables)
            sp.Symbol("C_DummyCH4"): 1.0,
            sp.Symbol("C_DummyNMVOC"): 1.0,
            sp.Symbol("C_FixedOH"): 1.0,
            sp.Symbol("C_FixedCl"): 1.0,
            sp.Symbol("S_a"): 1.0,
            sp.Symbol("v_gas"): 1.0,
            # M_density is used only when a mechanism has no explicit AIR or
            # M species. It must come from the supplied compilation
            # environment, never from a unit-specific legacy constant.
            sp.Symbol("M_density"): sp.Symbol("AirDensity") if keep_env_symbols else air_density,
        }

        # When keep_env_symbols is False (default), substitute environmental
        # parameters with constants for backward compatibility. When True,
        # leave Temp and RH as C variable references (used when equilibrium
        # reactions provide temp/rh as explicit function parameters).
        if not keep_env_symbols:
            # Environmental parameters (not species concentrations)
            # NOTE: SUN is NOT substituted — photolysis rates are runtime J-values from Cloud-J
            subs_dict[sp.Symbol("TEMP")] = temperature
            subs_dict[sp.Symbol("temp")] = temperature
            subs_dict[sp.Symbol("Temp")] = temperature

        if keep_env_symbols:
            subs_dict[sp.Symbol("TEMP")] = sp.Symbol("Temp")
            subs_dict[sp.Symbol("temp")] = sp.Symbol("Temp")
        expr = expr.subs(subs_dict)
        s = sp.ccode(expr)
        s = _fold_numeric_falloff_powers(s)
    except Exception:
        # Fallback to regex if sympy fails
        s = re.sub(r"([a-zA-Z0-9_\(\)\.\+\-\*\/]+)\*\*(\-?\d+\.\d+|\-?\d+)", r"pow(\1, \2)", s)
        if not keep_env_symbols:
            s = s.replace("Temp", str(temperature))
        s = s.replace("S_a", "1.0")
        s = s.replace("v_gas", "1.0")
        s = _fold_numeric_falloff_powers(s)

    # 3. Map the C_X species symbols from the SymPy AST directly into the state indices or variables.
    sorted_sp = sorted(list(enumerate(species_list)), key=lambda x: len(x[1].name), reverse=True)
    for idx_s, spec in sorted_sp:
        if use_parentheses:
            repl = f"{state_var}({idx_s})"
        else:
            repl = f"{state_var}_{idx_s}"
        s = re.sub(r"\bC_" + spec.name + r"(?!\w)", repl, s)

    # 4. Map J_<idx> photolysis symbols to the jvals array (Cloud-J runtime input)
    s = re.sub(r"\bJ_(\d+)\b", r"jvals[\1]", s)

    # 5. Map Rate_<idx> symbols (PHASE_CHANGE, TUNNELING) to jvals array
    # These are externally-provided rates from host model thermodynamic solvers (e.g., ISORROPIA)
    s = re.sub(r"\bRate_(\d+)\b", r"jvals[\1]", s)

    s = _strength_reduce_squares(s)
    s = _clean_float_literals(s)
    s = _kokkos_math(s)

    return s
