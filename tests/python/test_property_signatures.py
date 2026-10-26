"""
Property-based test for function signature preservation (Property 3).

# Feature: jinja2-codegen-templates, Property 3: Function signature preservation

For any valid Template_Context, rendering `header.j2` and extracting C++ function
signatures via regex must produce a set containing at minimum
`{compute_rates, compute_jacobian, integrate, integrate_with_reduction}`.
WHEN `adjoint_enabled=True`, the set SHALL additionally contain
`{integrate_fwd_checkpoint, integrate_adj, integrate_tlm}`.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.4**
"""

import re

from hypothesis import given, settings
from hypothesis import strategies as st
from mkpp.compact_plan import compact_plan
from mkpp.template_engine import TemplateEngine

# ---------------------------------------------------------------------------
# Strategies: Build minimal but valid Template_Context dicts
# ---------------------------------------------------------------------------

# Base function set always present
BASE_FUNCTIONS = {
    "compute_rates",
    "compute_jacobian",
    "integrate",
    "integrate_with_reduction",
}

# Additional functions when adjoint_enabled=True
ADJOINT_FUNCTIONS = {
    "integrate_fwd_checkpoint",
    "integrate_adj",
    "integrate_tlm",
}


@st.composite
def template_context_strategy(draw):
    """Generate a minimal but valid Template_Context dict for rendering header.j2.

    Builds:
    - N species (2-5)
    - Simple Jacobian entries (diagonal + random off-diagonal)
    - Simple LU expressions
    - Simple forward/backward substitution steps
    - Tableau (ROS-3 or ROS-2)
    - All needed context keys for header.j2 rendering
    """
    # Number of species
    n = draw(st.integers(min_value=2, max_value=5))

    # Adjoint mode
    adjoint_enabled = draw(st.booleans())

    # Has equilibrium
    has_equilibrium = draw(st.booleans())

    # Species list
    species_names = [f"SP{i}" for i in range(n)]
    species_list = [{"name": name, "index": idx, "elements": {}} for idx, name in enumerate(species_names)]

    # Jacobian entries: always include diagonal, random off-diagonal
    jacobian_entries = []
    non_zero_jac_set = set()
    jacobian_entries_state = []
    for i in range(n):
        # Diagonal always non-zero
        non_zero_jac_set.add((i, i))
        jacobian_entries.append((i, i, f"-k{i}"))
        jacobian_entries_state.append((i, i, f"-k{i}"))
    # Add some off-diagonal entries
    for i in range(n):
        for j in range(n):
            if i != j and draw(st.booleans()):
                non_zero_jac_set.add((i, j))
                jacobian_entries.append((i, j, f"k{i}_{j}"))
                jacobian_entries_state.append((i, j, f"k{i}_{j}"))

    # Needed W entries: diagonal always + entries from Jacobian
    needed_w = sorted({(i, i) for i in range(n)} | non_zero_jac_set)

    # LU expressions: simple diagonal U entries, off-diagonal L entries
    lu_expressions = []
    for i in range(n):
        lu_expressions.append(
            {
                "kind": "U",
                "i": i,
                "j": i,
                "expr": f"W_{i}_{i}",
            }
        )
    for i in range(1, n):
        for j in range(i):
            if (i, j) in non_zero_jac_set:
                lu_expressions.append(
                    {
                        "kind": "L",
                        "i": i,
                        "j": j,
                        "expr": f"W_{i}_{j} / U_{j}_{j}",
                    }
                )

    # Forward substitution steps
    forward_sub_steps = []
    for i in range(n):
        if i == 0:
            forward_sub_steps.append({"i": i, "raw_expr": f"b_{i}"})
        else:
            forward_sub_steps.append({"i": i, "raw_expr": f"b_{i} - L_{i}_0 * y_0"})

    # Backward substitution steps
    backward_sub_steps = []
    for i in range(n - 1, -1, -1):
        if i == n - 1:
            backward_sub_steps.append({"i": i, "raw_expr": f"y_{i} / U_{i}_{i}"})
        else:
            backward_sub_steps.append({"i": i, "raw_expr": f"(y_{i} - U_{i}_{i+1} * x_{i+1}) / U_{i}_{i}"})

    # Transposed substitution steps (when adjoint)
    transpose_forward_sub_steps = None
    transpose_backward_sub_steps = None
    if adjoint_enabled:
        transpose_forward_sub_steps = []
        for i in range(n):
            if i == 0:
                transpose_forward_sub_steps.append({"i": i, "raw_expr": f"b_{i}"})
            else:
                transpose_forward_sub_steps.append({"i": i, "raw_expr": f"b_{i} - U_{0}_{i} * y_0"})
        transpose_backward_sub_steps = []
        for i in range(n - 1, -1, -1):
            if i == n - 1:
                transpose_backward_sub_steps.append({"i": i, "raw_expr": f"y_{i} / U_{i}_{i}"})
            else:
                transpose_backward_sub_steps.append({"i": i, "raw_expr": f"(y_{i} - L_{i+1}_{i} * x_{i+1}) / U_{i}_{i}"})

    # Tableau: use ROS-3 structure (3 stages)
    tableau_choice = draw(st.sampled_from(["ros2", "ros3"]))
    if tableau_choice == "ros3":
        tableau = {
            "name": "ROS-3",
            "stages": 3,
            "ELO": 3.0,
            "Gamma": [0.43586652150845899942, 0.24291996454816804367, 2.1851380027664058512],
            "M": [1.0, 6.1697947043828245593, -0.42772256543218573326],
            "E": [0.5, -2.9079558716805469822, 0.22354069897811569627],
            "Alpha": [0.0, 0.43586652150845899942, 0.43586652150845899942],
            "NewF": [True, True, False],
            "A_matrix": [1.0, 1.0, 0.0],
            "C_matrix": [-1.0156171083877702092, 4.0759956452537699825, 9.2076794298330791242],
        }
    else:
        tableau = {
            "name": "ROS-2",
            "stages": 2,
            "ELO": 2.0,
            "Gamma": [1.7071067811865475244, -1.7071067811865475244],
            "M": [0.87867965644035742774, 0.29289321881345247560],
            "E": [0.29289321881345247560, 0.29289321881345247560],
            "Alpha": [0.0, 1.0],
            "NewF": [True, True],
            "A_matrix": [1.7071067811865475244],
            "C_matrix": [-1.1715728752538099024],
        }

    # F-vector expressions
    f_exprs = [f"-k{i} * S_{i}" for i in range(n)]

    # Rate expressions for compute_rates (state-based)
    rate_exprs_state = [f"-k{i} * state({i})" for i in range(n)]

    # State hoist info for integrate.j2
    state_hoist_info = [{"reg_idx": i, "name": f"SP{i}"} for i in range(n)]

    # Tolerance arrays
    tolerance_arrays = {
        "atol": [100.0] * n,
        "rtol": [0.01] * n,
    }

    # Photolysis
    has_photolysis = draw(st.booleans())
    num_photolysis = draw(st.integers(min_value=1, max_value=3)) if has_photolysis else 0
    photolysis_reactions = []
    if has_photolysis:
        for pi in range(num_photolysis):
            photolysis_reactions.append(
                {
                    "photo_idx": pi,
                    "reactants": {"SP0": 1.0},
                    "products": {"SP1": 1.0},
                    "original_A": 1.0e-5,
                }
            )

    # Adjoint entries (for compute_adjoint template)
    adjoint_entries_state = []
    for i in range(n):
        adjoint_entries_state.append((i, i, f"-k{i}"))

    # Build the context
    context = {
        "mechanism_name": "test_mech",
        "species": species_list,
        "num_species": n,
        "tableau": tableau,
        "lu_expressions": lu_expressions,
        "forward_sub_steps": forward_sub_steps,
        "backward_sub_steps": backward_sub_steps,
        "needed_w": needed_w,
        "non_zero_jac_set": non_zero_jac_set,
        "blocks": None,
        "jacobian_entries": jacobian_entries,
        "f_exprs": f_exprs,
        "permutation": None,
        "adjoint_enabled": adjoint_enabled,
        "has_equilibrium": has_equilibrium,
        "has_runtime_environment": has_equilibrium,
        "num_runtime_parameters": num_photolysis,
        "has_cloud_gated": False,
        "has_photolysis": has_photolysis,
        "num_photolysis": num_photolysis,
        "tolerance_arrays": tolerance_arrays,
        "initial_step_seed": "1.0e-6",
        "photolysis_reactions": photolysis_reactions,
        "host_interface": None,
        "equilibrium_results": None,
        "mass_projector": None,
        "rate_exprs_state": rate_exprs_state,
        "jacobian_entries_state": jacobian_entries_state,
        "adjoint_entries_state": adjoint_entries_state,
        "state_hoist_info": state_hoist_info,
        "rate_flux_cse": [],
        "rate_flux_exprs": [],
        "rate_flux_cse_hoist": [],
        "rate_flux_exprs_hoist": [],
        "mass_projector_data": None,
        "partition_metadata": None,
        "has_continuous_rxns": False,
        "equilibrium_reactions": [],
        "equilibrium_partition_data": None,
        "diagnostics_data": None,
        "suffix": "",
        "annotated_expressions": None,
        "simd_backend": "native",
        "compiled_rate_chunks": [],
        "compiled_jacobian_chunks": [],
        "compiled_lu_chunks": [],
    }

    # Include adjoint-specific context
    if adjoint_enabled:
        context["transpose_forward_sub_steps"] = transpose_forward_sub_steps
        context["transpose_backward_sub_steps"] = transpose_backward_sub_steps

    context["compact"] = compact_plan(
        n,
        jacobian_entries_state,
        None,
        {(entry["i"], entry["j"]) for entry in lu_expressions if entry["kind"] == "L"},
        {(entry["i"], entry["j"]) for entry in lu_expressions if entry["kind"] == "U"},
    )
    return context


# ---------------------------------------------------------------------------
# Helper: extract function names from rendered C++ code
# ---------------------------------------------------------------------------


def extract_function_names(rendered_code: str) -> set[str]:
    """Extract C++ function/method names from rendered header code.

    Matches patterns like:
      - KOKKOS_INLINE_FUNCTION void func_name(
      - KOKKOS_INLINE_FUNCTION int func_name(
    """
    # Match KOKKOS_INLINE_FUNCTION followed by return type and function name
    pattern = r"KOKKOS_INLINE_FUNCTION\s+(?:void|int|double)\s+(\w+)\s*\("
    matches = re.findall(pattern, rendered_code)
    return set(matches)


# ---------------------------------------------------------------------------
# Property 3: Function signature preservation
# **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.4**
# ---------------------------------------------------------------------------


@given(context=template_context_strategy())
@settings(deadline=None)
def test_property_3_function_signature_preservation(context):
    """
    For any valid Template_Context, rendering `header.j2` and extracting C++
    function signatures via regex must produce a set containing at minimum
    {compute_rates, compute_jacobian, integrate, integrate_with_reduction}.
    When adjoint_enabled=True, the set shall additionally contain
    {integrate_fwd_checkpoint, integrate_adj, integrate_tlm}.

    **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.4**
    """
    engine = TemplateEngine()
    rendered = engine.render("header.j2", context)

    # Extract function names from rendered code
    found_functions = extract_function_names(rendered)

    # Assert base functions are always present
    missing_base = BASE_FUNCTIONS - found_functions
    assert missing_base == set(), (
        f"Missing base function signatures in rendered header.\n"
        f"Expected: {sorted(BASE_FUNCTIONS)}\n"
        f"Found: {sorted(found_functions)}\n"
        f"Missing: {sorted(missing_base)}\n"
        f"adjoint_enabled={context['adjoint_enabled']}, "
        f"num_species={context['num_species']}, "
        f"has_equilibrium={context['has_equilibrium']}"
    )

    # When adjoint is enabled, assert adjoint functions are present
    if context["adjoint_enabled"]:
        missing_adjoint = ADJOINT_FUNCTIONS - found_functions
        assert missing_adjoint == set(), (
            f"Missing adjoint function signatures in rendered header.\n"
            f"Expected additional: {sorted(ADJOINT_FUNCTIONS)}\n"
            f"Found: {sorted(found_functions)}\n"
            f"Missing: {sorted(missing_adjoint)}\n"
            f"num_species={context['num_species']}, "
            f"has_equilibrium={context['has_equilibrium']}"
        )
