"""
Property-based test for error signaling (Property 5).

# Feature: jinja2-codegen-templates, Property 5: Error signaling for missing templates and undefined variables

For any template name not in the templates directory, `TemplateEngine.render`
must raise `TemplateNotFound`.

For any valid Template_Context with one required key removed, rendering must
raise `UndefinedError`.

Both exceptions must contain sufficient diagnostic info (template name or
variable name).

**Validates: Requirements 1.2, 6.3, 6.4**
"""

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from jinja2.exceptions import TemplateNotFound, UndefinedError
from mkpp.compact_plan import compact_plan
from mkpp.template_engine import TemplateEngine

# ---------------------------------------------------------------------------
# Known valid template names (to exclude from random generation)
# ---------------------------------------------------------------------------

VALID_TEMPLATE_NAMES = {
    "header.j2",
    "macros/w_matrix.j2",
    "macros/lu_factorize.j2",
    "macros/lu_solve_forward.j2",
    "macros/lu_solve_backward.j2",
    "macros/lu_solve_transpose.j2",
    "solver_variants/integrate.j2",
    "solver_variants/integrate_with_reduction.j2",
    "solver_variants/integrate_fwd_checkpoint.j2",
    "solver_variants/integrate_adj.j2",
    "solver_variants/integrate_tlm.j2",
    "kernel_functions/compute_rates.j2",
    "kernel_functions/compute_jacobian.j2",
    "kernel_functions/compute_adjoint.j2",
    "kernel_functions/compute_tlm.j2",
    "kernel_functions/project_mass_conservation.j2",
}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


@st.composite
def nonexistent_template_name(draw):
    """Generate a template name that does not exist in the templates directory.

    Produces names like 'abc123.j2', 'subdir/xyz.j2', etc.
    """
    # Generate a random prefix/subdirectory
    subdirs = draw(st.sampled_from(["", "macros/", "solver_variants/", "kernel_functions/", "nonexistent/"]))
    # Generate a random basename that won't collide with real template names
    basename = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-"),
            min_size=3,
            max_size=20,
        )
    )
    suffix = draw(st.sampled_from([".j2", ".jinja2", ".txt", ".hpp", ""]))
    name = f"{subdirs}{basename}{suffix}"
    # Ensure it's not actually a valid template
    assume(name not in VALID_TEMPLATE_NAMES)
    return name


@st.composite
def valid_template_context(draw):
    """Generate a minimal but valid Template_Context dict for rendering header.j2.

    Uses the same strategy approach as test_property_signatures.py.
    """
    n = draw(st.integers(min_value=2, max_value=5))
    adjoint_enabled = draw(st.booleans())
    has_equilibrium = draw(st.booleans())

    species_names = [f"SP{i}" for i in range(n)]
    species_list = [{"name": name, "index": idx, "elements": {}} for idx, name in enumerate(species_names)]

    # Jacobian entries: diagonal always
    jacobian_entries = []
    non_zero_jac_set = set()
    jacobian_entries_state = []
    for i in range(n):
        non_zero_jac_set.add((i, i))
        jacobian_entries.append((i, i, f"-k{i}"))
        jacobian_entries_state.append((i, i, f"-k{i}"))
    # Random off-diagonal
    for i in range(n):
        for j in range(n):
            if i != j and draw(st.booleans()):
                non_zero_jac_set.add((i, j))
                jacobian_entries.append((i, j, f"k{i}_{j}"))
                jacobian_entries_state.append((i, j, f"k{i}_{j}"))

    needed_w = sorted({(i, i) for i in range(n)} | non_zero_jac_set)

    # LU expressions
    lu_expressions = []
    for i in range(n):
        lu_expressions.append({"kind": "U", "i": i, "j": i, "expr": f"W_{i}_{i}"})
    for i in range(1, n):
        for j in range(i):
            if (i, j) in non_zero_jac_set:
                lu_expressions.append({"kind": "L", "i": i, "j": j, "expr": f"W_{i}_{j} / U_{j}_{j}"})

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

    # Transposed steps
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

    # Tableau: ROS-3
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

    f_exprs = [f"-k{i} * S_{i}" for i in range(n)]
    rate_exprs_state = [f"-k{i} * state({i})" for i in range(n)]
    tolerance_arrays = {"atol": [100.0] * n, "rtol": [0.01] * n}
    state_hoist_info = [{"reg_idx": i, "name": f"SP{i}"} for i in range(n)]

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

    adjoint_entries_state = [(i, i, f"-k{i}") for i in range(n)]

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


# Keys that header.j2 references unconditionally — removing any of these from
# a valid context causes Jinja2 StrictUndefined to raise UndefinedError.
# Note: some keys (like `species`, `num_photolysis`, `diagnostics_data`) are only
# accessed conditionally and don't raise when their guard condition is False.
# `num_species` raises TypeError (from range()) rather than UndefinedError.
REQUIRED_CONTEXT_KEYS = [
    "mechanism_name",
    "has_equilibrium",
    "adjoint_enabled",
    "tolerance_arrays",
    "initial_step_seed",
    "partition_metadata",
    "has_continuous_rxns",
    "host_interface",
    "has_photolysis",
    "mass_projector_data",
    "equilibrium_partition_data",
    "tableau",
    "lu_expressions",
    "forward_sub_steps",
    "backward_sub_steps",
    "needed_w",
    "non_zero_jac_set",
    "jacobian_entries",
    "f_exprs",
    "permutation",
    "blocks",
    "compiled_rate_chunks",
    "compiled_jacobian_chunks",
    "compiled_lu_chunks",
]


# ---------------------------------------------------------------------------
# Part 1: Missing templates raise TemplateNotFound
# **Validates: Requirements 1.2, 6.3, 6.4**
# ---------------------------------------------------------------------------


@given(template_name=nonexistent_template_name())
@settings(max_examples=100, deadline=None)
def test_property_5_missing_template_raises_template_not_found(template_name):
    """
    For any template name not present in the templates directory,
    TemplateEngine.render must raise TemplateNotFound.
    The exception must contain the template name for diagnostics.

    **Validates: Requirements 1.2, 6.3, 6.4**
    """
    engine = TemplateEngine()
    with pytest.raises(TemplateNotFound) as exc_info:
        engine.render(template_name, {})

    # Verify diagnostic info: exception should reference the template name
    assert template_name in str(exc_info.value), (
        f"TemplateNotFound exception should contain the template name '{template_name}' " f"but got: {exc_info.value}"
    )


# ---------------------------------------------------------------------------
# Part 2: Missing context variables raise UndefinedError
# **Validates: Requirements 1.2, 6.3, 6.4**
# ---------------------------------------------------------------------------


@given(
    context=valid_template_context(),
    key_to_remove=st.sampled_from(REQUIRED_CONTEXT_KEYS),
)
@settings(max_examples=100, deadline=None)
def test_property_5_missing_context_key_raises_undefined_error(context, key_to_remove):
    """
    For any valid Template_Context with one required key removed, rendering
    header.j2 must raise UndefinedError. The exception must contain the
    missing variable name for diagnostics.

    **Validates: Requirements 1.2, 6.3, 6.4**
    """
    # Skip if key doesn't exist in context (e.g. adjoint-specific keys)
    assume(key_to_remove in context)

    # Remove the key
    modified_context = {k: v for k, v in context.items() if k != key_to_remove}

    engine = TemplateEngine()
    with pytest.raises(UndefinedError) as exc_info:
        engine.render("header.j2", modified_context)

    # Verify diagnostic info: exception should reference the missing variable
    error_message = str(exc_info.value)
    assert key_to_remove in error_message, (
        f"UndefinedError should contain the missing variable name '{key_to_remove}' " f"but got: {error_message}"
    )
