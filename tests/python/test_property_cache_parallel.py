"""
Property-based tests for cache and parallelism (Component 3).

Validates:
- Property 5: Cache round-trip consistency (Requirements 3.2)
- Property 6: Content hash determinism and sensitivity (Requirements 3.1)
- Property 7: Parallel Jacobian equivalence (Requirements 3.5, 3.6)
- Property 8: CSE semantic preservation (Requirements 3.7)
- Property 9: CSE declaration ordering (Requirements 3.8)
"""

import tempfile
from pathlib import Path

import pytest
import sympy as sp
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from mkpp.cache_manager import CacheEntry, CacheKey, CacheManager
from mkpp.lowering import (
    prepare_unified_jacobian,
    prepare_unified_jacobian_parallel,
)
from mkpp.model import SymbolicLUPlan
from mkpp.parser import load_mechanism

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def random_symbolic_lu_plan(draw):
    """Generate a random SymbolicLUPlan-like object with species maps and non-zero entries."""
    n = draw(st.integers(min_value=2, max_value=8))
    species_map = [f"species_{i}" for i in range(n)]

    # Generate random non-zero Jacobian entries (diagonal always present)
    non_zero_jacobian = []
    for i in range(n):
        # Diagonal always non-zero
        non_zero_jacobian.append((i, i, f"C_species_{i} + 1.0"))
        # Random off-diagonals with ~30% density
        for j in range(n):
            if i != j and draw(st.booleans().filter(lambda _: True)):
                if draw(st.integers(min_value=0, max_value=9)) < 3:
                    coeff = draw(st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
                    if abs(coeff) > 1e-10:
                        non_zero_jacobian.append((i, j, f"{coeff} * C_species_{j}"))

    # Generate simple L and U expressions
    l_expressions = [(i, 0, f"W_{i}_0 / U_0_0") for i in range(1, n)]
    u_expressions = [(0, j, f"W_0_{j}") for j in range(n)]

    # Simple ordered LU expressions
    lu_expressions_ordered = [("U", 0, j, f"W_0_{j}") for j in range(n)]
    lu_expressions_ordered += [("L", i, 0, f"W_{i}_0 / U_0_0") for i in range(1, n)]

    # Forward substitution
    forward_sub_steps = [(i, f"b_{i}") for i in range(n)]

    # Backward substitution
    backward_sub_steps = [(i, f"y_{i} / U_{i}_{i}") for i in range(n - 1, -1, -1)]

    return SymbolicLUPlan(
        num_species=n,
        species_map=species_map,
        non_zero_jacobian=non_zero_jacobian,
        l_expressions=l_expressions,
        u_expressions=u_expressions,
        lu_expressions_ordered=lu_expressions_ordered,
        forward_sub_steps=forward_sub_steps,
        backward_sub_steps=backward_sub_steps,
    )


@st.composite
def random_cache_entry(draw):
    """Generate a random CacheEntry for cache round-trip testing."""
    lu_plan = draw(random_symbolic_lu_plan())
    n = lu_plan.num_species

    # Use simple SymPy matrices for testing serialization
    [sp.Symbol(f"C_{s}", real=True) for s in lu_plan.species_map]
    jacobian = sp.Matrix(
        [
            [sp.Float(draw(st.floats(min_value=-5, max_value=5, allow_nan=False, allow_infinity=False))) for _ in range(n)]
            for _ in range(n)
        ]
    )
    f_implicit = sp.Matrix(
        [sp.Float(draw(st.floats(min_value=-1, max_value=1, allow_nan=False, allow_infinity=False))) for _ in range(n)]
    )
    f_explicit = sp.Matrix([sp.Float(0.0) for _ in range(n)])

    key = CacheKey(yaml_hash="a" * 64, mkpp_version="test-0.1.0")
    entry = CacheEntry(
        key=key,
        species_map=lu_plan.species_map,
        jacobian_matrix=jacobian,
        lu_plan=lu_plan,
        f_implicit=f_implicit,
        f_explicit=f_explicit,
    )
    return entry


@st.composite
def random_sympy_expression(draw, max_depth=3):
    """Generate random SymPy expressions (sums, products of symbols)."""
    symbols = [sp.Symbol(f"x_{i}", real=True) for i in range(4)]

    def _build(depth):
        if depth <= 0 or draw(st.booleans()):
            # Terminal: either a symbol or a small float constant
            choice = draw(st.integers(min_value=0, max_value=5))
            if choice < 4:
                return symbols[choice]
            else:
                val = draw(st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False))
                return sp.Float(val)
        else:
            op = draw(st.sampled_from(["add", "mul", "pow"]))
            left = _build(depth - 1)
            right = _build(depth - 1)
            if op == "add":
                return left + right
            elif op == "mul":
                return left * right
            else:
                # Small integer power to avoid overflow
                exp = draw(st.integers(min_value=2, max_value=3))
                return left**exp

    # Generate 2-6 expressions
    num_exprs = draw(st.integers(min_value=2, max_value=6))
    exprs = [_build(max_depth) for _ in range(num_exprs)]
    return exprs


# ---------------------------------------------------------------------------
# Property 5: Cache round-trip consistency
# **Validates: Requirements 3.2**
# ---------------------------------------------------------------------------


@given(entry=random_cache_entry())
@settings(deadline=None)
def test_property_5_cache_round_trip_consistency(entry):
    """
    For any valid CacheEntry, serializing to disk with CacheManager.store()
    and then deserializing with CacheManager.lookup() produces a structurally
    equal object.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache_dir = Path(tmp_dir) / "cache"
        manager = CacheManager(cache_dir=cache_dir)

        # Store the entry
        manager.store(entry.key, entry)

        # Load it back
        loaded = manager.lookup(entry.key)

        # Verify the entry was loaded successfully
        assert loaded is not None, "Cache lookup returned None after store"

        # Verify structural equality
        assert loaded.key == entry.key
        assert loaded.species_map == entry.species_map
        assert loaded.lu_plan.num_species == entry.lu_plan.num_species
        assert loaded.lu_plan.species_map == entry.lu_plan.species_map
        assert loaded.lu_plan.non_zero_jacobian == entry.lu_plan.non_zero_jacobian
        assert loaded.lu_plan.l_expressions == entry.lu_plan.l_expressions
        assert loaded.lu_plan.u_expressions == entry.lu_plan.u_expressions
        assert loaded.lu_plan.forward_sub_steps == entry.lu_plan.forward_sub_steps
        assert loaded.lu_plan.backward_sub_steps == entry.lu_plan.backward_sub_steps

        # Verify matrices
        assert loaded.jacobian_matrix == entry.jacobian_matrix
        assert loaded.f_implicit == entry.f_implicit
        assert loaded.f_explicit == entry.f_explicit


# ---------------------------------------------------------------------------
# Property 6: Content hash determinism and sensitivity
# **Validates: Requirements 3.1**
# ---------------------------------------------------------------------------


@given(data=st.binary(min_size=1, max_size=1024))
@settings(deadline=None)
def test_property_6_hash_determinism(data):
    """
    Same bytes produce the same hash (idempotent). Calling compute_key twice
    on the same file contents yields the same CacheKey.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        yaml_path = Path(tmp_dir) / "test.yaml"
        yaml_path.write_bytes(data)

        manager = CacheManager()
        key1 = manager.compute_key(yaml_path)
        key2 = manager.compute_key(yaml_path)

        assert key1.yaml_hash == key2.yaml_hash
        assert key1.mkpp_version == key2.mkpp_version


@given(
    data_a=st.binary(min_size=1, max_size=512),
    data_b=st.binary(min_size=1, max_size=512),
)
@settings(deadline=None)
def test_property_6_hash_sensitivity(data_a, data_b):
    """
    Different byte sequences produce different hashes.
    """
    assume(data_a != data_b)

    with tempfile.TemporaryDirectory() as tmp_dir:
        path_a = Path(tmp_dir) / "a.yaml"
        path_b = Path(tmp_dir) / "b.yaml"
        path_a.write_bytes(data_a)
        path_b.write_bytes(data_b)

        manager = CacheManager()
        key_a = manager.compute_key(path_a)
        key_b = manager.compute_key(path_b)

        assert key_a.yaml_hash != key_b.yaml_hash


def test_cache_key_distinguishes_kinetic_unit_conversion_mode(tmp_path):
    mechanism_path = tmp_path / "mechanism.json"
    mechanism_path.write_text('{"concentration_unit": "molecule cm-3"}', encoding="utf-8")

    manager = CacheManager()
    source_units_key = manager.compute_key(mechanism_path)
    si_units_key = manager.compute_key(mechanism_path, convert_kinetic_units_to_si=True)

    assert source_units_key.yaml_hash != si_units_key.yaml_hash


# ---------------------------------------------------------------------------
# Property 7: Parallel Jacobian equivalence
# **Validates: Requirements 3.5, 3.6**
# ---------------------------------------------------------------------------

MECHANISM_DIR = Path(__file__).resolve().parent.parent.parent / "mechanisms"


@pytest.mark.parametrize("mech_file", ["small_strato.yaml", "chapman.yaml"])
def test_property_7_parallel_jacobian_equivalence(mech_file):
    """
    For existing test mechanisms, the Jacobian computed via parallel column
    differentiation is element-wise identical to the Jacobian computed sequentially.
    """
    mech_path = MECHANISM_DIR / mech_file
    if not mech_path.exists():
        pytest.skip(f"Mechanism file {mech_path} not found")

    mech = load_mechanism(str(mech_path))

    # Sequential computation
    result_seq = prepare_unified_jacobian(mech)
    J_seq = result_seq["jacobian_matrix"]

    # Parallel computation
    result_par = prepare_unified_jacobian_parallel(mech)
    J_par = result_par["jacobian_matrix"]

    # Verify dimensions match
    assert J_seq.shape == J_par.shape, f"Shape mismatch: sequential={J_seq.shape}, parallel={J_par.shape}"

    N = J_seq.shape[0]
    # Element-wise comparison using sympy.simplify on the difference
    for i in range(N):
        for j in range(N):
            diff = sp.simplify(J_seq[i, j] - J_par[i, j])
            assert diff == 0, f"Jacobian mismatch at ({i},{j}): " f"sequential={J_seq[i,j]}, parallel={J_par[i,j]}, diff={diff}"


# ---------------------------------------------------------------------------
# Property 8: CSE semantic preservation
# **Validates: Requirements 3.7**
# ---------------------------------------------------------------------------


@given(exprs=random_sympy_expression())
@settings(deadline=None)
def test_property_8_cse_semantic_preservation(exprs):
    """
    For any set of SymPy expressions, applying sympy.cse() and substituting
    the replacements back yields expressions numerically equivalent to originals.
    """
    # Apply CSE
    replacements, reduced = sp.cse(exprs, optimizations="basic")

    # Substitute replacements back into reduced expressions
    restored = list(reduced)
    # Apply replacements in reverse order (later ones may depend on earlier)
    for sym, sub_expr in reversed(replacements):
        restored = [expr.subs(sym, sub_expr) for expr in restored]

    # Verify numerical equivalence at random point
    symbols_used = set()
    for expr in exprs:
        symbols_used.update(expr.free_symbols)

    # Use a fixed evaluation point to avoid flakiness
    eval_point = {s: sp.Float(1.5) for s in symbols_used}

    for orig, rest in zip(exprs, restored):
        orig_val = orig.subs(eval_point)
        rest_val = rest.subs(eval_point)
        # Use simplify on the difference for symbolic comparison
        diff = sp.simplify(orig_val - rest_val)
        # Allow tiny floating-point rounding differences from expression rearrangement
        assert abs(complex(diff)) < 1e-12, f"CSE restoration failed: original={orig}, restored={rest}, diff={diff}"


# ---------------------------------------------------------------------------
# Property 9: CSE declaration ordering
# **Validates: Requirements 3.8**
# ---------------------------------------------------------------------------


@given(exprs=random_sympy_expression())
@settings(deadline=None)
def test_property_9_cse_declaration_ordering(exprs):
    """
    For CSE replacements from sympy.cse(), verify that each symbol is defined
    before it is referenced. In the replacement list, symbol at index i should
    not appear in any expression at index j < i.
    """
    replacements, reduced = sp.cse(exprs, optimizations="basic")

    if not replacements:
        return  # No CSE applied, nothing to check

    # Track which CSE symbols have been declared
    declared = set()

    for sym, expr in replacements:
        # Check that the expression only references previously declared CSE symbols
        expr_free = expr.free_symbols
        {s for s in expr_free if str(s).startswith("x")}

        # Filter to only CSE-generated symbols (they have form x0, x1, ...)
        cse_defined_syms = {s for s, _ in replacements}
        referenced_cse = expr_free & cse_defined_syms

        for ref_sym in referenced_cse:
            assert ref_sym in declared, (
                f"CSE ordering violation: {sym} references {ref_sym} "
                f"which has not been declared yet. "
                f"Declared so far: {declared}"
            )

        # Mark this symbol as declared
        declared.add(sym)

    # Also verify reduced expressions only reference declared CSE symbols
    for reduced_expr in reduced:
        reduced_free = reduced_expr.free_symbols if hasattr(reduced_expr, "free_symbols") else set()
        referenced_cse = reduced_free & cse_defined_syms
        for ref_sym in referenced_cse:
            assert ref_sym in declared, f"Reduced expression references undeclared CSE symbol {ref_sym}"
