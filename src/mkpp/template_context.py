"""Template context construction for Jinja2 code generation.

Assembles all data needed for Jinja2 template rendering into a flat
dictionary (Template_Context). Templates receive no Python function
calls, only pre-computed data.
"""

import math
import re
from typing import Any

from .compact_plan import compact_plan
from .factorization_plan import make_factorization_plan
from .format_eqn import format_eqn
from .model import MechanismDefinition, SymbolicLUPlan
from .rosenbrock import SOLVER_COEFFICIENTS


def build_template_context(
    mech: MechanismDefinition,
    solver_name: str = "ros3",
    lu_plan: SymbolicLUPlan | None = None,
    sympy_meta: dict | None = None,
    adjoint: bool = False,
    simd_backend: str = "native",
) -> dict[str, Any]:
    """Assemble all data needed for Jinja2 template rendering.

    Parameters
    ----------
    mech : MechanismDefinition
        Parsed mechanism definition containing species, reactions, and metadata.
    solver_name : str
        Name of the Rosenbrock solver variant (key in SOLVER_COEFFICIENTS).
    lu_plan : SymbolicLUPlan or None
        Pre-computed symbolic LU plan. If None, will be retrieved from
        mechanism metadata or computed from sympy_meta.
    sympy_meta : dict or None
        Mechanism symbolic metadata containing jacobian_matrix, f_vector, etc.
        If None, will be retrieved from mechanism metadata or computed.
    adjoint : bool
        If True, include transposed LU substitution steps for adjoint solves.

    Returns
    -------
    dict
        Flat dictionary with all keys required for Jinja2 template rendering.

    Raises
    ------
    ValueError
        If the mechanism has no species (empty mechanism).
    KeyError
        If solver_name is not found in SOLVER_COEFFICIENTS.
    """
    # Validate non-empty mechanism
    if not mech or not mech.species:
        raise ValueError("Cannot build template context for empty mechanism (no species)")

    N = len(mech.species)

    # --- Resolve tableau ---
    tableau = SOLVER_COEFFICIENTS[solver_name]

    # --- Resolve sympy_meta ---
    if sympy_meta is None:
        if getattr(mech, "metadata", None) and isinstance(mech.metadata, dict):
            sympy_meta = mech.metadata.get("sympy_metadata")
        if sympy_meta is None:
            sympy_meta = getattr(mech, "sympy_metadata", None)
        if sympy_meta is None:
            from .lowering import prepare_unified_jacobian

            sympy_meta = prepare_unified_jacobian(mech)

    # --- Resolve LU plan ---
    if lu_plan is None:
        if isinstance(getattr(mech, "metadata", None), dict):
            lu_plan = mech.metadata.get("symbolic_lu_plan")
        if lu_plan is None and sympy_meta and "symbolic_lu_plan" in sympy_meta:
            lu_plan = sympy_meta["symbolic_lu_plan"]
        if lu_plan is None and getattr(mech, "symbolic_lu_plan", None):
            lu_plan = getattr(mech, "symbolic_lu_plan", None)
        if lu_plan is None and sympy_meta and "jacobian_matrix" in sympy_meta:
            from .lowering import compute_symbolic_lu_decomposition

            lu_plan = compute_symbolic_lu_decomposition(
                sympy_meta["jacobian_matrix"],
                [s.name for s in mech.species],
            )

    # --- Compute transposed LU plan when adjoint is enabled ---
    if adjoint and lu_plan:
        if not lu_plan.transpose_forward_sub_steps:
            from .lowering import compute_transposed_lu_plan

            compute_transposed_lu_plan(lu_plan)

    # --- Determine permutation ---
    permutation = lu_plan.permutation if lu_plan and lu_plan.permutation else None
    environment_temperature = float(mech.metadata.get("temperature", "300.0"))
    environment_air_density = float(mech.metadata.get("air_density", "2.4476e19"))

    # --- Determine equilibrium and photolysis flags ---
    has_equilibrium = bool(mech.equilibrium_reactions)
    has_runtime_environment = has_equilibrium or bool(
        re.search(
            r"\b(?:Temp|TEMP|temp|M_density)\b",
            str(tuple(sympy_meta.get(key, "") for key in ("f_implicit", "f_explicit", "f_vector"))),
        )
    )
    # A cloud-gated kinetic reaction (activation_trigger naming
    # meteo.cloud_liquid_water) requires the generated solver to accept cloud
    # liquid water as a third per-cell equilibrium input
    # (EquilibriumInput.CloudLiquidWater). It also implies the equilibrium
    # parameter channel (temp/rh) is live, so runtime environment inputs are enabled;
    # otherwise the runtime parameters would not be threaded.
    has_cloud_gated = bool(getattr(mech, "has_cloud_gated_reaction", False))
    if has_cloud_gated:
        has_runtime_environment = True
    has_photolysis = False
    num_photolysis = 0
    photolysis_reactions = []
    if sympy_meta:
        num_photolysis = sympy_meta.get("photolysis_count", 0)
        photolysis_reactions = sympy_meta.get("photolysis_reactions", [])
        has_photolysis = num_photolysis > 0

    def _remap_s_indices(expr_str: str, inv_perm: list[int]) -> str:
        """Remap unpermuted species index S_i to permuted register index S_{inv_perm[i]}."""
        return re.sub(r"\bS_(\d+)\b", lambda m: f"S_{inv_perm[int(m.group(1))]}", expr_str)

    # --- Format Jacobian entries via format_eqn() ---
    jacobian_entries = []
    non_zero_jac_set = set()
    if lu_plan:
        if permutation:
            inv_p = [0] * N
            for reg_idx, orig_idx in enumerate(permutation):
                inv_p[orig_idx] = reg_idx

            for i, j, expr_str in lu_plan.non_zero_jacobian:
                non_zero_jac_set.add((i, j))
                eqn = format_eqn(
                    expr_str,
                    mech.species,
                    state_var="S",
                    use_parentheses=False,
                    keep_env_symbols=has_runtime_environment,
                    temperature=environment_temperature,
                    air_density=environment_air_density,
                )
                remapped_eqn = _remap_s_indices(eqn, inv_p)
                jacobian_entries.append((i, j, remapped_eqn))
        else:
            for i, j, expr_str in lu_plan.non_zero_jacobian:
                non_zero_jac_set.add((i, j))
                eqn = format_eqn(
                    expr_str,
                    mech.species,
                    state_var="S",
                    use_parentheses=False,
                    keep_env_symbols=has_runtime_environment,
                    temperature=environment_temperature,
                    air_density=environment_air_density,
                )
                jacobian_entries.append((i, j, eqn))

    # --- Format F-vector expressions ---
    f_exprs = []
    if sympy_meta:
        if "f_implicit" in sympy_meta and "f_explicit" in sympy_meta:
            F_raw = sympy_meta["f_implicit"] + sympy_meta["f_explicit"]
        elif "f_vector" in sympy_meta:
            F_raw = sympy_meta["f_vector"]
        else:
            F_raw = [0] * N

        if permutation:
            inv_p = [0] * N
            for reg_idx, orig_idx in enumerate(permutation):
                inv_p[orig_idx] = reg_idx

            for k in range(N):
                orig_idx = permutation[k]
                eqn = format_eqn(
                    F_raw[orig_idx],
                    mech.species,
                    state_var="S",
                    use_parentheses=False,
                    keep_env_symbols=has_runtime_environment,
                    temperature=environment_temperature,
                    air_density=environment_air_density,
                )
                remapped_eqn = _remap_s_indices(eqn, inv_p)
                f_exprs.append(remapped_eqn)
        else:
            for expr in F_raw:
                eqn = format_eqn(
                    expr,
                    mech.species,
                    state_var="S",
                    use_parentheses=False,
                    keep_env_symbols=has_runtime_environment,
                    temperature=environment_temperature,
                    air_density=environment_air_density,
                )
                f_exprs.append(eqn)

    # --- Determine W-entry requirements by scanning LU expressions ---
    needed_w = set()
    for i in range(N):
        needed_w.add((i, i))  # Diagonal always needed
    if lu_plan:
        for _kind, _i, _j, expr_str in lu_plan.lu_expressions_ordered:
            for m in re.finditer(r"W_(\d+)_(\d+)", expr_str):
                needed_w.add((int(m.group(1)), int(m.group(2))))

    # --- Serialize LU expressions ---
    lu_expressions = []
    if lu_plan:
        for kind, i, j, expr_str in lu_plan.lu_expressions_ordered:
            lu_expressions.append(
                {
                    "kind": kind,
                    "i": i,
                    "j": j,
                    "expr": expr_str,
                }
            )

    # --- Serialize forward/backward substitution steps ---
    forward_sub_steps = []
    backward_sub_steps = []
    if lu_plan:
        for i, expr_str in lu_plan.forward_sub_steps:
            forward_sub_steps.append({"i": i, "raw_expr": expr_str})
        for i, expr_str in lu_plan.backward_sub_steps:
            backward_sub_steps.append({"i": i, "raw_expr": expr_str})

    # --- Serialize transposed substitution steps (adjoint) ---
    transpose_forward_sub_steps = None
    transpose_backward_sub_steps = None
    if adjoint and lu_plan and lu_plan.transpose_forward_sub_steps:
        transpose_forward_sub_steps = []
        for i, expr_str in lu_plan.transpose_forward_sub_steps:
            transpose_forward_sub_steps.append({"i": i, "raw_expr": expr_str})
        transpose_backward_sub_steps = []
        for i, expr_str in lu_plan.transpose_backward_sub_steps:
            transpose_backward_sub_steps.append({"i": i, "raw_expr": expr_str})

    # --- Blocks ---
    blocks = None
    if lu_plan and lu_plan.blocks and len(lu_plan.blocks) > 1:
        blocks = []
        for block_indices in lu_plan.blocks:
            block_species_names = [lu_plan.species_map[idx] for idx in block_indices]
            blocks.append(
                {
                    "indices": block_indices,
                    "species_names": block_species_names,
                }
            )

    # --- Annotated expressions (for auto-reduction) ---
    annotated_expressions = None
    if lu_plan and lu_plan.annotated_expressions:
        annotated_expressions = []
        for ann_expr in lu_plan.annotated_expressions:
            annotated_expressions.append(
                {
                    "kind": ann_expr.kind,
                    "row": ann_expr.row,
                    "col": ann_expr.col,
                    "expr": ann_expr.expr,
                    "depends_on": sorted(ann_expr.depends_on),
                }
            )

    # --- Tolerance arrays ---
    # Match MICM's default solver controls for an equivalent comparison.
    # Per-species metadata remains the higher-priority override.
    default_atol = 1.0e-3
    default_rtol = 1.0e-6

    # Start with defaults
    atol_values = [default_atol] * N
    rtol_values = [default_rtol] * N

    # Override with metadata arrays if present (batch override)
    if isinstance(getattr(mech, "metadata", None), dict):
        meta_atol = mech.metadata.get("atol")
        meta_rtol = mech.metadata.get("rtol")
        if meta_atol is not None:
            for i in range(min(len(meta_atol), N)):
                atol_values[i] = meta_atol[i]
        if meta_rtol is not None:
            for i in range(min(len(meta_rtol), N)):
                rtol_values[i] = meta_rtol[i]

    # Override with per-species values (highest priority)
    for i, sp in enumerate(mech.species):
        if getattr(sp, "solver_atol", None) is not None:
            atol_values[i] = sp.solver_atol
        if getattr(sp, "solver_rtol", None) is not None:
            rtol_values[i] = sp.solver_rtol

    tolerance_arrays = {"atol": atol_values, "rtol": rtol_values}

    # --- Initial-step seed for the adaptive Rosenbrock ramp ---
    # The generated integrate() starts its first internal substep at
    # dt = dt_total * seed.  The committed default is the exact literal token
    # below, so an unset seed reproduces the shipped artifact byte-for-byte.
    # A raised seed cuts the mandatory growth substeps (cost floor) while the
    # rejection controller preserves accuracy; it must be finite and positive
    # or the integrator can stall, so reject anything else loudly.
    initial_step_seed = "1.0e-6"
    if isinstance(getattr(mech, "metadata", None), dict):
        meta_seed = mech.metadata.get("initial_step_seed")
        if meta_seed is not None:
            seed_value = float(meta_seed)
            if not math.isfinite(seed_value) or seed_value <= 0.0:
                raise ValueError(
                    f"initial_step_seed must be a finite positive fraction of the " f"chemistry interval, got {meta_seed!r}"
                )
            initial_step_seed = repr(seed_value)

    # --- Tableau serialization ---
    tableau_dict = {
        "name": tableau.name,
        "stages": tableau.stages,
        "ELO": tableau.ELO,
        "Gamma": list(tableau.Gamma),
        "M": list(tableau.M),
        "E": list(tableau.E),
        "Alpha": list(tableau.Alpha),
        "NewF": list(tableau.NewF),
        "A_matrix": list(tableau.A),
        "C_matrix": list(tableau.C),
    }

    # --- Species metadata ---
    species_list = []
    for idx, sp in enumerate(mech.species):
        species_list.append(
            {
                "name": sp.name,
                "index": idx,
                "elements": dict(sp.elements) if sp.elements else {},
            }
        )

    # State variable hoisting map (reg_idx -> species_name and original_idx)
    state_hoist_info = []
    for i in range(N):
        orig_i = permutation[i] if permutation else i
        state_hoist_info.append(
            {
                "reg_idx": i,
                "orig_idx": orig_i,
                "name": mech.species[orig_i].name if orig_i < len(mech.species) else f"Species_{orig_i}",
            }
        )

    # --- Assemble the flat context dictionary ---
    context: dict[str, Any] = {
        "mechanism_name": mech.name,
        "species": species_list,
        "num_species": N,
        "reactions": mech.reactions if getattr(mech, "reactions", None) else [],
        "num_reactions": len(mech.reactions) if getattr(mech, "reactions", None) else 0,
        "non_zero_jacobian_count": len(jacobian_entries) if jacobian_entries else N * N,
        "tableau": tableau_dict,
        "lu_expressions": lu_expressions,
        "forward_sub_steps": forward_sub_steps,
        "backward_sub_steps": backward_sub_steps,
        "needed_w": sorted(needed_w),
        "non_zero_jac_set": non_zero_jac_set,
        "blocks": blocks,
        "state_hoist_info": state_hoist_info,
        "jacobian_entries": jacobian_entries,
        "f_exprs": f_exprs,
        "permutation": permutation,
        "adjoint_enabled": adjoint,
        "has_equilibrium": has_equilibrium,
        "has_runtime_environment": has_runtime_environment,
        "environment_temperature": environment_temperature,
        "environment_air_density": environment_air_density,
        "has_cloud_gated": has_cloud_gated,
        "has_photolysis": has_photolysis,
        "num_photolysis": num_photolysis,
        "num_runtime_parameters": num_photolysis,
        "tolerance_arrays": tolerance_arrays,
        "initial_step_seed": initial_step_seed,
        "simd_backend": simd_backend,
    }

    # Conditionally include adjoint-specific data
    if adjoint:
        context["transpose_forward_sub_steps"] = transpose_forward_sub_steps
        context["transpose_backward_sub_steps"] = transpose_backward_sub_steps

    # Include annotated expressions for auto-reduction
    if annotated_expressions is not None:
        context["annotated_expressions"] = annotated_expressions

    # Include additional metadata for downstream templates
    context["photolysis_reactions"] = photolysis_reactions
    context["host_interface"] = mech.host_interface if getattr(mech, "host_interface", None) else None
    context["equilibrium_results"] = sympy_meta.get("equilibrium_results") if sympy_meta else None
    context["mass_projector"] = sympy_meta.get("mass_projector") if sympy_meta else None

    # --- Reaction rate flux vector CSE and expressions ---
    rate_flux_cse = []
    rate_flux_exprs = []
    rate_flux_cse_hoist = []
    rate_flux_exprs_hoist = []
    if sympy_meta:
        raw_cse = sympy_meta.get("rate_flux_cse", [])
        for sym, expr in raw_cse:
            eqn = format_eqn(
                expr,
                mech.species,
                state_var="state",
                use_parentheses=True,
                keep_env_symbols=has_runtime_environment,
                temperature=environment_temperature,
                air_density=environment_air_density,
            )
            rate_flux_cse.append({"symbol": str(sym), "expr": eqn})
            eqn_hoist = format_eqn(
                expr,
                mech.species,
                state_var="S",
                use_parentheses=False,
                keep_env_symbols=has_runtime_environment,
                temperature=environment_temperature,
                air_density=environment_air_density,
            )
            rate_flux_cse_hoist.append({"symbol": str(sym), "expr": eqn_hoist})

        raw_exprs = sympy_meta.get("rate_flux_exprs", [])
        for idx, expr in enumerate(raw_exprs):
            eqn = format_eqn(
                expr,
                mech.species,
                state_var="state",
                use_parentheses=True,
                keep_env_symbols=has_runtime_environment,
                temperature=environment_temperature,
                air_density=environment_air_density,
            )
            rate_flux_exprs.append({"index": idx, "expr": eqn})
            eqn_hoist = format_eqn(
                expr,
                mech.species,
                state_var="S",
                use_parentheses=False,
                keep_env_symbols=has_runtime_environment,
                temperature=environment_temperature,
                air_density=environment_air_density,
            )
            rate_flux_exprs_hoist.append({"index": idx, "expr": eqn_hoist})

    context["rate_flux_cse"] = rate_flux_cse
    context["rate_flux_exprs"] = rate_flux_exprs
    context["rate_flux_cse_hoist"] = rate_flux_cse_hoist
    context["rate_flux_exprs_hoist"] = rate_flux_exprs_hoist

    # --- Kernel function context: state(i)-based expressions ---
    # These use state_var="state" and use_parentheses=True for kernel function templates
    # (compute_rates, compute_jacobian, compute_adjoint, compute_tlm)
    rate_exprs_state = []
    if sympy_meta:
        if "f_implicit" in sympy_meta and "f_explicit" in sympy_meta:
            F_raw_state = sympy_meta["f_implicit"] + sympy_meta["f_explicit"]
        elif "f_vector" in sympy_meta:
            F_raw_state = sympy_meta["f_vector"]
        else:
            F_raw_state = [0] * N
        for expr in F_raw_state:
            eqn = format_eqn(
                expr,
                mech.species,
                state_var="state",
                use_parentheses=True,
                keep_env_symbols=has_runtime_environment,
                temperature=environment_temperature,
                air_density=environment_air_density,
            )
            rate_exprs_state.append(eqn)
    context["rate_exprs_state"] = rate_exprs_state

    jacobian_entries_state = []
    if sympy_meta and "jacobian_matrix" in sympy_meta:
        J_matrix = sympy_meta["jacobian_matrix"]
        for i in range(J_matrix.shape[0]):
            for j in range(J_matrix.shape[1]):
                if J_matrix[i, j] != 0:
                    eqn = format_eqn(
                        J_matrix[i, j],
                        mech.species,
                        state_var="state",
                        use_parentheses=True,
                        keep_env_symbols=has_runtime_environment,
                        temperature=environment_temperature,
                        air_density=environment_air_density,
                    )
                    jacobian_entries_state.append((i, j, eqn))
    context["jacobian_entries_state"] = jacobian_entries_state

    # Bound the expression-heavy code into independently compiled units.  The
    # context owns this metadata (rather than the CLI) so direct template users
    # and the compiler always select the same generated-artifact pathway.
    def _chunk(items: list[Any], size: int) -> list[list[Any]]:
        return [items[index : index + size] for index in range(0, len(items), size)]

    def _layout_aware_expression(expression: str) -> str:
        return expression

    context["num_runtime_parameters"] = max(
        [int(slot) + 1 for expression in rate_exprs_state for slot in re.findall(r"jvals\[(\d+)\]", expression)] + [num_photolysis]
    )
    context["compiled_rate_chunks"] = [
        [
            {"index": index, "expr": _layout_aware_expression(expression)}
            for index, expression in enumerate(chunk, start=chunk_index * 32)
        ]
        for chunk_index, chunk in enumerate(_chunk(rate_exprs_state, 32))
    ]
    context["compiled_jacobian_chunks"] = [
        [
            {"row": row, "column": column, "slot": chunk_index * 256 + index, "expr": _layout_aware_expression(expression)}
            for index, (row, column, expression) in enumerate(chunk)
        ]
        for chunk_index, chunk in enumerate(_chunk(jacobian_entries_state, 256))
    ]

    # Sparse LU is emitted as a separate compiled unit while retaining the
    # exact symbolic operation order selected during lowering.  L and U never
    # occupy the same matrix location, so a single packed array is sufficient.
    def _lu_expression(expression: str) -> str:
        expression = re.sub(r"\b[LU]_(\d+)_(\d+)\b", r"lu[\1 * N + \2]", expression)
        return re.sub(r"\bW_(\d+)_(\d+)\b", r"w[\1 * N + \2]", expression)

    def _forward_expression(expression: str) -> str:
        expression = _lu_expression(expression)
        expression = re.sub(r"\bb_(\d+)\b", r"rhs[\1]", expression)
        return re.sub(r"\by_(\d+)\b", r"work[\1]", expression)

    def _backward_expression(expression: str) -> str:
        expression = _lu_expression(expression)
        expression = re.sub(r"\by_(\d+)\b", r"work[\1]", expression)
        return re.sub(r"\bx_(\d+)\b", r"solution[\1]", expression)

    compiled_lu_expressions = [{**entry, "expr": _lu_expression(entry["expr"])} for entry in lu_expressions]
    factorization_plan = make_factorization_plan(
        mech.name,
        len(mech.species),
        lu_expressions,
        elimination_order=lu_plan.permutation if lu_plan and lu_plan.permutation else None,
        fill_in_count=lu_plan.fill_in_count if lu_plan else 0,
    )
    context["factorization_plan"] = factorization_plan
    context["compiled_lu_expressions"] = compiled_lu_expressions
    context["compiled_lu_chunks"] = _chunk(compiled_lu_expressions, 256)
    # A compact symbolic sparsity mask drives the portable candidate backend.
    # Unlike the reference expressions it contains no chemistry algebra.
    l_locations = {(entry["i"], entry["j"]) for entry in lu_expressions if entry["kind"] == "L"}
    u_locations = {(entry["i"], entry["j"]) for entry in lu_expressions if entry["kind"] == "U"}
    context["compact"] = compact_plan(N, jacobian_entries_state, permutation, l_locations, u_locations)
    context["plan_l_pattern"] = [int((row, column) in l_locations) for row in range(N) for column in range(N)]
    context["plan_u_pattern"] = [int((row, column) in u_locations) for row in range(N) for column in range(N)]
    context["compiled_forward_steps"] = [
        {"index": step["i"], "expr": _forward_expression(step["raw_expr"])} for step in forward_sub_steps
    ]
    context["compiled_backward_steps"] = [
        {"index": step["i"], "expr": _backward_expression(step["raw_expr"])} for step in backward_sub_steps
    ]

    adjoint_entries_state = []
    if sympy_meta:
        J_adj = sympy_meta.get("adjoint_matrix")
        if J_adj is None and "jacobian_matrix" in sympy_meta and sympy_meta["jacobian_matrix"] is not None:
            J_adj = sympy_meta["jacobian_matrix"].transpose()
        if J_adj is not None:
            for i in range(J_adj.shape[0]):
                for j in range(J_adj.shape[1]):
                    if J_adj[i, j] != 0:
                        eqn = format_eqn(
                            J_adj[i, j],
                            mech.species,
                            state_var="state",
                            use_parentheses=True,
                            keep_env_symbols=has_runtime_environment,
                            temperature=environment_temperature,
                            air_density=environment_air_density,
                        )
                        adjoint_entries_state.append((i, j, eqn))
    context["adjoint_entries_state"] = adjoint_entries_state

    # --- Mass projector data for project_mass_conservation template ---
    mass_projector_data = None
    if sympy_meta and "mass_projector" in sympy_meta:
        P = sympy_meta["mass_projector"]
        elements = sympy_meta["element_map"]
        species_map = sympy_meta["species_map"]

        # Build element->species coefficient mapping (E-matrix rows)
        element_species_coeffs = []
        for elem_idx, elem in enumerate(elements):
            coeffs = {}
            for sp_idx, sp_name in enumerate(species_map):
                s_def = next(s for s in mech.species if s.name == sp_name)
                if elem in s_def.elements:
                    coeffs[sp_idx] = s_def.elements[elem]
            element_species_coeffs.append(coeffs)

        # Build projector non-zero entries
        projector_entries = []
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                if P[i, j] != 0:
                    projector_entries.append((i, j, P[i, j]))

        mass_projector_data = {
            "element_map": list(elements),
            "species_map": list(species_map),
            "projector_entries": projector_entries,
            "element_species_coeffs": element_species_coeffs,
        }
    context["mass_projector_data"] = mass_projector_data

    # --- Additional metadata for header.j2 ---
    # Partition metadata for SZA sorting annotation
    partition_meta = getattr(mech, "partition_metadata", None)
    context["partition_metadata"] = partition_meta

    # Continuous transition annotation
    has_continuous_rxns = any(getattr(r, "continuous_transition", False) for r in mech.reactions)
    context["has_continuous_rxns"] = has_continuous_rxns

    # Equilibrium reactions metadata for compute_equilibrium_partition
    equilibrium_reactions = mech.equilibrium_reactions if mech.equilibrium_reactions else []
    context["equilibrium_reactions"] = equilibrium_reactions

    # --- Pre-compute equilibrium partition data for templates ---
    equilibrium_partition_data = None
    if mech.equilibrium_reactions and sympy_meta and "equilibrium_results" in sympy_meta:
        eq_results = sympy_meta["equilibrium_results"]
        eq_partition_blocks = []
        for eq_result in eq_results:
            # Total species sums
            total_species = []
            for element_name, species_indices in eq_result.total_species_map.items():
                if species_indices:
                    terms = [f"state({idx})" for idx in species_indices]
                    total_species.append(
                        {
                            "element_name": element_name,
                            "expr": " + ".join(terms),
                        }
                    )
            # Partition expressions
            partition_entries = []
            for species_name, expr in eq_result.partition_exprs.items():
                species_idx = next(
                    (i for i, s in enumerate(mech.species) if s.name == species_name),
                    None,
                )
                if species_idx is not None:
                    eqn = format_eqn(
                        expr,
                        mech.species,
                        state_var="state",
                        use_parentheses=True,
                        temperature=environment_temperature,
                        air_density=environment_air_density,
                    )
                    partition_entries.append(
                        {
                            "species_idx": species_idx,
                            "expr": eqn,
                        }
                    )
            eq_partition_blocks.append(
                {
                    "total_species": total_species,
                    "partition_entries": partition_entries,
                }
            )
        equilibrium_partition_data = eq_partition_blocks
    context["equilibrium_partition_data"] = equilibrium_partition_data

    # --- Diagnostics data for equilibrium mechanisms ---
    diagnostics_data = None
    if has_equilibrium and mech.equilibrium_reactions:
        eq_def = mech.equilibrium_reactions[0]  # Primary equilibrium system
        # Mass balance elements
        mass_balance_elements = []
        for element_name, sp_list in eq_def.total_species.items():
            species_indices = []
            for sp_name in sp_list:
                for si, spec in enumerate(mech.species):
                    if spec.name == sp_name:
                        species_indices.append(si)
                        break
            mass_balance_elements.append(
                {
                    "element_name": element_name,
                    "species_list": list(sp_list),
                    "species_indices": species_indices,
                }
            )
        # Charge balance species contributions
        charge_contributions = []
        for si, spec in enumerate(mech.species):
            if spec.name == "NH4a":
                charge_contributions.append({"idx": si, "sign": "+", "coeff": 1.0, "comment": "NH4+ (+1)"})
            elif spec.name in ("NO3an1", "NO3an2", "NO3an3"):
                charge_contributions.append({"idx": si, "sign": "-", "coeff": 1.0, "comment": "NO3- (-1)"})
            elif spec.name == "SO4":
                charge_contributions.append({"idx": si, "sign": "-", "coeff": 2.0, "comment": "SO4^2- (-2)"})
        diagnostics_data = {
            "mass_balance_elements": mass_balance_elements,
            "charge_contributions": charge_contributions,
        }
    context["diagnostics_data"] = diagnostics_data

    # Suffix for filename reference
    context["suffix"] = ""

    return context
