"""
Property-based test for function signature invariance across all solvers.

**Validates: Requirements 8.3**

Property 11: Function signature invariance
For any solver, the template signature of integrate() and integrate_with_reduction()
in the generated header is identical (same parameter types and annotations).
"""

import re
import tempfile

from mkpp.codegen import SOLVER_COEFFICIENTS, generate_headers
from mkpp.lowering import compute_symbolic_lu_decomposition, prepare_unified_jacobian
from mkpp.model import (
    AerosolRepresentation,
    MechanismDefinition,
    PhaseMode,
    ReactionDefinition,
    SpeciesDefinition,
)


def _build_toy_mechanism():
    """Build a simple 3-species toy mechanism for code generation testing."""
    sp_a = SpeciesDefinition(name="A", phase=PhaseMode.GAS)
    sp_b = SpeciesDefinition(name="B", phase=PhaseMode.GAS)
    sp_c = SpeciesDefinition(name="C", phase=PhaseMode.GAS)
    rxn1 = ReactionDefinition(
        reaction_type="ARRHENIUS",
        reactants={"A": 1.0},
        products={"B": 1.0},
        rate_expression="k1*A",
        parameters={"A": 1e-2, "B": 0.0, "C": 0.0},
        stiff=True,
    )
    rxn2 = ReactionDefinition(
        reaction_type="ARRHENIUS",
        reactants={"B": 1.0},
        products={"C": 1.0},
        rate_expression="k2*B",
        parameters={"A": 1e-3, "B": 0.0, "C": 0.0},
        stiff=True,
    )
    return MechanismDefinition(
        name="test_sig_invariance",
        description="Toy mechanism for function signature invariance test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[sp_a, sp_b, sp_c],
        phases=[],
        reactions=[rxn1, rxn2],
    )


def _generate_code_for_solver(solver_name: str) -> str:
    """Generate C++ header for the given solver and return the code as a string."""
    mech = _build_toy_mechanism()
    lowering_data = prepare_unified_jacobian(mech)
    plan = compute_symbolic_lu_decomposition(lowering_data["jacobian_matrix"], lowering_data["species_map"])
    mech.metadata = {
        "sympy_metadata": lowering_data,
        "symbolic_lu_plan": plan,
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        artifacts = generate_headers(mech, out_dir=tmp_dir, solver_name=solver_name)
        with open(artifacts["header"]) as f:
            return f.read()


def _normalize_signature(raw: str) -> str:
    """Normalize whitespace in a signature for comparison.

    Collapses all whitespace runs to single spaces and removes spaces after '(' and before ')'.
    """
    sig = re.sub(r"\s+", " ", raw).strip()
    # Normalize "( " -> "(" and " )" -> ")"
    sig = re.sub(r"\(\s+", "(", sig)
    sig = re.sub(r"\s+\)", ")", sig)
    return sig


def _extract_integrate_signature(code: str) -> str:
    """Extract the full signature of integrate() from generated code.

    Expected form:
    template <class StateView>
    KOKKOS_INLINE_FUNCTION void integrate(double dt_total, StateView& state, const double* jvals) const
    """
    pattern = r"(template\s*<\s*class\s+StateView\s*>\s*\n\s*" r"KOKKOS_INLINE_FUNCTION\s+void\s+integrate\s*\([^)]*\)\s*const)"
    match = re.search(pattern, code)
    assert match is not None, "Could not find integrate() signature in generated code"
    return _normalize_signature(match.group(1))


def _extract_integrate_with_reduction_signature(code: str) -> str:
    """Extract the full signature of integrate_with_reduction() from generated code.

    Expected form:
    template <class StateView>
    KOKKOS_INLINE_FUNCTION void integrate_with_reduction(
        double dt_total, StateView& state, const double* jvals, double importance_threshold) const
    """
    pattern = (
        r"(template\s*<\s*class\s+StateView\s*>\s*\n\s*"
        r"KOKKOS_INLINE_FUNCTION\s+void\s+integrate_with_reduction\s*\([^)]*\)\s*const)"
    )
    match = re.search(pattern, code)
    assert match is not None, "Could not find integrate_with_reduction() signature in generated code"
    return _normalize_signature(match.group(1))


class TestFunctionSignatureInvariance:
    """
    Property 11: Function signature invariance.

    For any solver, the template signature of integrate() and integrate_with_reduction()
    in the generated header is identical (same parameter types and annotations).

    **Validates: Requirements 8.3**
    """

    def test_integrate_signature_identical_across_all_solvers(self):
        """All 5 solvers produce the same integrate() signature."""
        solver_names = list(SOLVER_COEFFICIENTS.keys())
        signatures = {}

        for solver_name in solver_names:
            code = _generate_code_for_solver(solver_name)
            signatures[solver_name] = _extract_integrate_signature(code)

        # All signatures should be identical - compare against the first one
        reference_solver = solver_names[0]
        reference_sig = signatures[reference_solver]

        for solver_name in solver_names[1:]:
            assert signatures[solver_name] == reference_sig, (
                f"integrate() signature differs between '{reference_solver}' and '{solver_name}':\n"
                f"  {reference_solver}: {reference_sig}\n"
                f"  {solver_name}: {signatures[solver_name]}"
            )

    def test_integrate_with_reduction_signature_identical_across_all_solvers(self):
        """All 5 solvers produce the same integrate_with_reduction() signature."""
        solver_names = list(SOLVER_COEFFICIENTS.keys())
        signatures = {}

        for solver_name in solver_names:
            code = _generate_code_for_solver(solver_name)
            signatures[solver_name] = _extract_integrate_with_reduction_signature(code)

        # All signatures should be identical - compare against the first one
        reference_solver = solver_names[0]
        reference_sig = signatures[reference_solver]

        for solver_name in solver_names[1:]:
            assert signatures[solver_name] == reference_sig, (
                f"integrate_with_reduction() signature differs between "
                f"'{reference_solver}' and '{solver_name}':\n"
                f"  {reference_solver}: {reference_sig}\n"
                f"  {solver_name}: {signatures[solver_name]}"
            )

    def test_integrate_signature_matches_expected_form(self):
        """The integrate() signature matches the exact expected form from the design doc."""
        expected_sig = (
            "template <class StateView> "
            "KOKKOS_INLINE_FUNCTION void integrate("
            "double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view) const"
        )

        for solver_name in SOLVER_COEFFICIENTS:
            code = _generate_code_for_solver(solver_name)
            actual_sig = _extract_integrate_signature(code)
            assert actual_sig == expected_sig, (
                f"[{solver_name}] integrate() signature doesn't match expected form:\n"
                f"  Expected: {expected_sig}\n"
                f"  Actual:   {actual_sig}"
            )

    def test_integrate_with_reduction_signature_matches_expected_form(self):
        """The integrate_with_reduction() signature matches the exact expected form."""
        expected_sig = (
            "template <class StateView> "
            "KOKKOS_INLINE_FUNCTION void integrate_with_reduction("
            "double dt_total, StateView& state, Kokkos::View<const double*, memory_space> jvals_view, "
            "double importance_threshold) const"
        )

        for solver_name in SOLVER_COEFFICIENTS:
            code = _generate_code_for_solver(solver_name)
            actual_sig = _extract_integrate_with_reduction_signature(code)
            assert actual_sig == expected_sig, (
                f"[{solver_name}] integrate_with_reduction() signature doesn't match expected form:\n"
                f"  Expected: {expected_sig}\n"
                f"  Actual:   {actual_sig}"
            )
