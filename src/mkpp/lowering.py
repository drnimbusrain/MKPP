import multiprocessing
import warnings
from typing import TYPE_CHECKING, Any

import networkx as nx
import sympy as sp

from .model import MechanismDefinition, ReactionDefinition, SymbolicLUPlan

if TYPE_CHECKING:
    from .model import AnnotatedLUExpression, SparsityAnalysis


def parse_sym_or_val(val, default=0.0):
    if val is None:
        return sp.Float(default)
    if isinstance(val, int | float):
        return sp.Float(val)
    s = str(val).strip().replace(" ", "")
    try:
        return sp.Float(float(s))
    except ValueError:
        return sp.Symbol(s, real=True)


def partition_reactions(mech: MechanismDefinition) -> dict[str, list[ReactionDefinition]]:
    """
    Partition reactions into implicit (stiff) and explicit (non-stiff) deterministic blocks
    using Tarjan's Strongly Connected Components (SCC) algorithm.
    """
    blocks = {"implicit": [], "explicit": []}

    # 1. Build the directed species dependency graph
    G = nx.DiGraph()
    for r in mech.reactions:
        for reactant in r.reactants:
            for product in r.products:
                G.add_edge(reactant, product)

    # 2. Find cycles (SCCs with more than 1 node, or self-loops)
    sccs = list(nx.strongly_connected_components(G))
    stiff_species = set()
    for scc in sccs:
        if len(scc) > 1:
            stiff_species.update(scc)
        elif len(scc) == 1:
            # Check for self-loop
            node = list(scc)[0]
            if G.has_edge(node, node):
                stiff_species.add(node)

    # 3. Partition reactions based on topology
    for r in mech.reactions:
        # A reaction belongs to the stiff manifold if it connects species within the stiff network
        is_stiff_topology = any(reactant in stiff_species for reactant in r.reactants) and any(
            product in stiff_species for product in r.products
        )

        # We also respect manual overrides (r.stiff) if the user forces it
        if r.stiff or is_stiff_topology:
            blocks["implicit"].append(r)
        else:
            blocks["explicit"].append(r)

    # Sort blocks deterministically by reaction type then expression
    blocks["implicit"].sort(key=lambda x: (x.reaction_type, x.rate_expression))
    blocks["explicit"].sort(key=lambda x: (x.reaction_type, x.rate_expression))

    # T026: Inject deterministic solver partition metadata
    blocks["metadata"] = {
        "sza_sorted": True,
        "micro_blocks": {"implicit": len(blocks["implicit"]), "explicit": len(blocks["explicit"])},
        "scc_count": len([s for s in sccs if len(s) > 1]),
    }

    return blocks


def prepare_adjoint_and_tlm(mech: MechanismDefinition) -> dict[str, bool]:
    """
    T015: Symbolic lowering hooks for analytical Jacobian, Adjoint, and Tangent-Linear models.
    For the MVP, this validates that the mechanism is differentiable.
    """
    # Verify no discontinuous thermodynamic operators are present
    for r in mech.reactions:
        if not r.continuous_transition and r.reaction_type.lower() in (
            "condensation",
            "phase_change",
        ):
            raise ValueError(f"Reaction {r.rate_expression} lacks continuous transition for analytical differentiation.")

    # Check that equilibrium reactions have continuous_transition = True
    if hasattr(mech, "equilibrium_reactions") and mech.equilibrium_reactions:
        for eq_def in mech.equilibrium_reactions:
            if not eq_def.continuous_transition:
                raise ValueError(
                    f"Equilibrium system '{eq_def.system}' requires continuous_transition=True " f"for analytical differentiation."
                )

    return {"adjoint_ready": True, "tlm_ready": True}


def _process_rate_vector_cse(reaction_fluxes: list[sp.Expr]) -> tuple[list[tuple[sp.Symbol, sp.Expr]], list[sp.Expr]]:
    """
    Apply SymPy Common Subexpression Elimination (CSE) across reaction rate fluxes.
    Assigns deterministic `cse_tmp_{n}` names in topological order and applies a
    reachability filter to drop unreferenced temporaries.
    """
    if not reaction_fluxes:
        return [], []

    replacements, reduced_fluxes = sp.cse(reaction_fluxes, optimizations="basic")

    # Reachability filter: collect all symbols referenced directly or indirectly by reduced_fluxes
    used_symbols = set()
    for flux_expr in reduced_fluxes:
        if isinstance(flux_expr, sp.Basic):
            used_symbols.update(flux_expr.free_symbols)

    for sym, expr in reversed(replacements):
        if sym in used_symbols:
            if isinstance(expr, sp.Basic):
                used_symbols.update(expr.free_symbols)

    kept_replacements = [item for item in replacements if item[0] in used_symbols]

    # Re-index kept replacements deterministically as cse_tmp_0, cse_tmp_1, ...
    symbol_map = {sym: sp.Symbol(f"cse_tmp_{idx}", real=True) for idx, (sym, expr) in enumerate(kept_replacements)}

    final_cse = []
    for idx, (sym, expr) in enumerate(kept_replacements):
        new_sym = symbol_map[sym]
        new_expr = expr.subs(symbol_map)
        final_cse.append((new_sym, new_expr))

    final_reduced_fluxes = [r.subs(symbol_map) if isinstance(r, sp.Basic) else r for r in reduced_fluxes]

    return final_cse, final_reduced_fluxes


def _evaluate_reaction_fluxes(mech: MechanismDefinition) -> dict[str, Any]:
    """
    Evaluates reaction rate expressions and builds symbolic implicit/explicit ODE vectors (f_implicit, f_explicit, f_total)
    and photolysis metadata from a mechanism definition.
    """
    species_symbols = {s.name: sp.Symbol(f"C_{s.name}", real=True, nonnegative=True) for s in mech.species}
    Temp = sp.Symbol("Temp", real=True, nonnegative=True)
    sp.Symbol("Press", real=True, nonnegative=True)
    M_density = sp.Symbol("M_density", real=True, nonnegative=True)
    v_gas = sp.Symbol("v_gas", real=True, nonnegative=True)
    S_a = sp.Symbol("S_a", real=True, nonnegative=True)

    df_dt_implicit = {s.name: sp.Integer(0) for s in mech.species}
    df_dt_explicit = {s.name: sp.Integer(0) for s in mech.species}
    fixed_species = set(s.name for s in mech.species if getattr(s, "role", None) == "fixed")

    blocks = partition_reactions(mech)

    photo_idx = 0
    photolysis_reactions = []
    reaction_fluxes = []

    for idx, r in enumerate(mech.reactions):
        rtype = r.reaction_type.upper()
        p = r.parameters
        flux = sp.Integer(0)

        if rtype == "PHOTOLYSIS":
            # OpenAtmos/MUSICA photolysis rates are supplied as named forcing;
            # the optional A field is only a legacy YAML placeholder.
            flux = sp.Symbol(f"J_{photo_idx}", real=True, nonnegative=True)
            photolysis_reactions.append(
                {
                    "photo_idx": photo_idx,
                    "reaction_idx": idx,
                    "reactants": r.reactants,
                    "products": r.products,
                    "original_A": str(p.get("A", 1.0)),
                }
            )
            photo_idx += 1

        elif rtype == "ARRHENIUS":
            if "A" not in p:
                raise ValueError(f"ARRHENIUS reaction {idx} missing 'A' coefficient.")
            A = parse_sym_or_val(p["A"])
            B = parse_sym_or_val(p.get("B", 0.0))
            C = parse_sym_or_val(p.get("C", 0.0))
            k_arr = A * (Temp / 300) ** B * sp.exp(C / Temp)
            flux = k_arr

        elif rtype == "DUMMYTROE":
            k0_A = parse_sym_or_val(p["k0"]["A"])
            k0_B = parse_sym_or_val(p["k0"].get("B", 0.0))
            k0_C = parse_sym_or_val(p["k0"].get("C", 0.0))
            k0_A * (Temp / 300) ** k0_B * sp.exp(k0_C / Temp)

            kinf_A = parse_sym_or_val(p["kinf"]["A"])
            kinf_B = parse_sym_or_val(p["kinf"].get("B", 0.0))
            kinf_C = parse_sym_or_val(p["kinf"].get("C", 0.0))
            kinf_A * (Temp / 300) ** kinf_B * sp.exp(kinf_C / Temp)

            parse_sym_or_val(p.get("Fc", 0.6))

        elif rtype == "TROE" or rtype == "FALLOFF":

            def get_troe_sub_params(sub_p):
                a = parse_sym_or_val(sub_p.get("A", 0.0))
                b = parse_sym_or_val(sub_p.get("B", 0.0))
                c = parse_sym_or_val(sub_p.get("C", 0.0))
                try:
                    fb = float(b)
                    fc = float(c)
                    if abs(fb) > 100.0 and abs(fc) <= 100.0:
                        b, c = c, b
                except Exception:
                    pass
                return a, b, c

            if "k0" in p and isinstance(p["k0"], dict):
                A0, B0, C0 = get_troe_sub_params(p["k0"])
            else:
                flat_k0 = {
                    "A": p.get("k0_A", 0.0),
                    "B": p.get("k0_B", 0.0),
                    "C": p.get("k0_C", 0.0),
                }
                A0, B0, C0 = get_troe_sub_params(flat_k0)

            if "kinf" in p and isinstance(p["kinf"], dict):
                A1, B1, C1 = get_troe_sub_params(p["kinf"])
            else:
                flat_kinf = {
                    "A": p.get("kinf_A", 0.0),
                    "B": p.get("kinf_B", 0.0),
                    "C": p.get("kinf_C", 0.0),
                }
                A1, B1, C1 = get_troe_sub_params(flat_kinf)

            CF = parse_sym_or_val(p.get("Fc", 0.6))

            # OpenAtmos/MUSICA and MICM use the same signed Arrhenius
            # convention for both Troe limits: exp(C / T).  Do not infer an
            # activation-energy sign from the coefficient magnitude.
            K0 = A0 * sp.exp(C0 / Temp) * (Temp / 300) ** B0
            K1 = A1 * sp.exp(C1 / Temp) * (Temp / 300) ** B1
            # MICM/OpenAtmos represents the atmospheric third body as the
            # special fixed species M. Prefer it when present; falling back to
            # AIR preserves mechanisms that use that spelling. This prevents
            # the renderer from substituting a legacy number-density constant
            # into molar-concentration mechanisms such as TS1.
            third_body = species_symbols.get("AIR", species_symbols.get("M", M_density))
            K0 = K0 * third_body
            K_ratio = K0 / K1
            F_broadening = CF ** (1.0 / (1.0 + (sp.log(K_ratio, 10)) ** 2))
            flux = (K0 / (1.0 + K_ratio)) * F_broadening

        elif rtype == "EP2":
            A0 = parse_sym_or_val(p.get("A0", 0.0))
            C0 = parse_sym_or_val(p.get("C0", 0.0))
            A2 = parse_sym_or_val(p.get("A2", 0.0))
            C2 = parse_sym_or_val(p.get("C2", 0.0))
            A3 = parse_sym_or_val(p.get("A3", 0.0))
            C3 = parse_sym_or_val(p.get("C3", 0.0))
            K0 = A0 * sp.exp(C0 / Temp)
            K2 = A2 * sp.exp(C2 / Temp)
            K3 = A3 * sp.exp(C3 / Temp) * species_symbols.get("AIR", species_symbols.get("M", M_density))
            flux = K0 + K3 / (1.0 + K3 / K2)

        elif rtype == "EP3":
            A1 = parse_sym_or_val(p.get("A1", 0.0))
            C1 = parse_sym_or_val(p.get("C1", 0.0))
            A2 = parse_sym_or_val(p.get("A2", 0.0))
            C2 = parse_sym_or_val(p.get("C2", 0.0))
            K1 = A1 * sp.exp(C1 / Temp)
            K2 = A2 * sp.exp(C2 / Temp)
            flux = K1 + K2 * species_symbols.get("AIR", species_symbols.get("M", M_density))

        elif rtype == "HETEROGENEOUS":
            if "rate_name" in p:
                flux = sp.Symbol(f"Rate_{idx}", real=True, nonnegative=True)
            else:
                gamma = parse_sym_or_val(p["gamma"])
                k_het = 0.25 * gamma * v_gas * S_a
                flux = k_het

        elif rtype == "PHASE_CHANGE":
            flux = sp.Symbol(f"Rate_{idx}", real=True)

        elif rtype == "TUNNELING":
            if "Y_spline" in p:
                flux = parse_sym_or_val(p["Y_spline"])
            elif "A" in p:
                A = parse_sym_or_val(p["A"])
                C = parse_sym_or_val(p.get("C", 0.0))
                flux = A * sp.exp(C / Temp)
            else:
                flux = sp.Symbol(f"Rate_{idx}", real=True)

        else:
            if "rate_name" in p:
                flux = sp.Symbol(f"Rate_{idx}", real=True, nonnegative=True)
            elif "A" in p:
                flux = parse_sym_or_val(p["A"])
            elif "Y_spline" in p:
                flux = parse_sym_or_val(p["Y_spline"])
            else:
                flux = sp.Symbol(f"Rate_{idx}", real=True)

        reactants_dict = r.reactants if isinstance(r.reactants, dict) else {sp_name: 1.0 for sp_name in r.reactants}
        products_dict = r.products if isinstance(r.products, dict) else {sp_name: 1.0 for sp_name in r.products}

        for reactant, stoich in reactants_dict.items():
            if reactant in species_symbols:
                flux *= species_symbols[reactant] ** sp.Integer(int(stoich))

        reaction_fluxes.append(flux)

        is_implicit = r in blocks["implicit"]

        for reactant, stoich in reactants_dict.items():
            if reactant in df_dt_implicit and reactant not in fixed_species:
                if is_implicit:
                    df_dt_implicit[reactant] -= flux * sp.Float(stoich)
                else:
                    df_dt_explicit[reactant] -= flux * sp.Float(stoich)

        for product, stoich in products_dict.items():
            if product in df_dt_implicit and product not in fixed_species:
                if is_implicit:
                    df_dt_implicit[product] += flux * sp.Float(stoich)
                else:
                    df_dt_explicit[product] += flux * sp.Float(stoich)

    # ------------------------------------------------------------------
    # Equilibrium coupling: add relaxation flux for equilibrium species
    # This runs AFTER the kinetic reaction loop so that equilibrium
    # terms are added on top of any existing kinetic contributions.
    # ------------------------------------------------------------------
    if mech.equilibrium_reactions:
        from .equilibrium.registry import get_model

        RH_sym = sp.Symbol("RH", real=True, nonnegative=True)

        for eq_def in mech.equilibrium_reactions:
            tau_eq_inv = sp.Float(eq_def.relaxation_timescale_inv)
            model = get_model(eq_def.system)

            # Build total-species symbols for each conserved element
            # total_species: dict[str, list[str]] where list = [gas, aerosol1, aerosol2, ...]
            totals: dict[str, sp.Expr] = {}
            for element_name, sp_list in eq_def.total_species.items():
                total_expr: sp.Expr = sp.Integer(0)
                for sp_name in sp_list:
                    if sp_name in species_symbols:
                        total_expr += species_symbols[sp_name]
                totals[element_name] = total_expr

            # Get partition expressions from the analytical equilibrium model
            eq_exprs = model.partition_expressions(
                totals,
                Temp,
                RH_sym,
                eq_def.regime_blending,
                eq_def.transition_width,
            )

            # Add relaxation flux: drives species toward equilibrium partition
            # df_dt[species_i] += tau_eq_inv * (eq_expr_i - C_species_i)
            # This creates a stiff relaxation handled by the Rosenbrock solver
            # via the Jacobian. Added to df_dt_implicit since equilibrium
            # coupling is stiff by nature.
            for sp_name, eq_expr in eq_exprs.items():
                if sp_name in df_dt_implicit and sp_name not in fixed_species:
                    df_dt_implicit[sp_name] += tau_eq_inv * (eq_expr - species_symbols[sp_name])

    ordered_species = [s.name for s in mech.species]
    f_implicit = sp.Matrix([df_dt_implicit[s] for s in ordered_species])
    f_explicit = sp.Matrix([df_dt_explicit[s] for s in ordered_species])
    f_total = f_implicit + f_explicit
    c_vector = sp.Matrix([species_symbols[s] for s in ordered_species])

    return {
        "species_symbols": species_symbols,
        "ordered_species": ordered_species,
        "f_implicit": f_implicit,
        "f_explicit": f_explicit,
        "f_total": f_total,
        "c_vector": c_vector,
        "photolysis_count": photo_idx,
        "photolysis_reactions": photolysis_reactions,
        "reaction_fluxes": reaction_fluxes,
    }


def prepare_unified_jacobian(mech: MechanismDefinition) -> dict[str, Any]:
    built = _evaluate_reaction_fluxes(mech)
    ordered_species = built["ordered_species"]
    f_implicit = built["f_implicit"]
    f_explicit = built["f_explicit"]
    f_total = built["f_total"]
    c_vector = built["c_vector"]

    jacobian_matrix = f_total.jacobian(c_vector)
    adjoint_matrix = jacobian_matrix.transpose()

    reaction_fluxes = built.get("reaction_fluxes", [])
    rate_flux_cse, rate_flux_exprs = _process_rate_vector_cse(reaction_fluxes)

    unique_elements = sorted(list(set(elem for s in mech.species for elem in s.elements.keys())))
    if unique_elements:
        E_matrix = sp.zeros(len(unique_elements), len(ordered_species))
        for j, sp_name in enumerate(ordered_species):
            species_def = next(s for s in mech.species if s.name == sp_name)
            for i, elem in enumerate(unique_elements):
                E_matrix[i, j] = species_def.elements.get(elem, 0)
        try:
            E_E_T = E_matrix * E_matrix.transpose()
            mass_projector = E_matrix.transpose() * E_E_T.pinv()
        except Exception:
            mass_projector = sp.zeros(len(ordered_species), len(unique_elements))
    else:
        E_matrix = sp.zeros(1, len(ordered_species))
        mass_projector = sp.zeros(len(ordered_species), 1)

    result = {
        "species_map": ordered_species,
        "f_implicit": f_implicit,
        "f_explicit": f_explicit,
        "jacobian_matrix": jacobian_matrix,
        "adjoint_matrix": adjoint_matrix,
        "mass_projector": mass_projector,
        "element_map": unique_elements,
        "photolysis_count": built["photolysis_count"],
        "photolysis_reactions": built["photolysis_reactions"],
        "rate_flux_cse": rate_flux_cse,
        "rate_flux_exprs": rate_flux_exprs,
    }

    # ------------------------------------------------------------------
    # Equilibrium symbolic results: extract equilibrium-specific Jacobian
    # entries and store EquilibriumSymbolicResult for downstream inspection.
    # The full Jacobian already includes equilibrium derivatives because
    # f_total contains the relaxation flux terms (tau_eq_inv * (eq_expr - C_i)).
    # Here we compute and store the isolated equilibrium contributions so
    # that code emission and adjoint layers can inspect them independently.
    # ------------------------------------------------------------------
    if mech.equilibrium_reactions:
        from .equilibrium.registry import get_model
        from .model import EquilibriumSymbolicResult

        RH_sym = sp.Symbol("RH", real=True, nonnegative=True)
        Temp = sp.Symbol("Temp", real=True, nonnegative=True)
        species_symbols = {s.name: sp.Symbol(f"C_{s.name}", real=True, nonnegative=True) for s in mech.species}

        eq_results = []
        for eq_def in mech.equilibrium_reactions:
            model = get_model(eq_def.system)

            # Build total-species symbols for each conserved element
            totals: dict[str, sp.Expr] = {}
            for element_name, sp_list in eq_def.total_species.items():
                total_expr: sp.Expr = sp.Integer(0)
                for sp_name in sp_list:
                    if sp_name in species_symbols:
                        total_expr += species_symbols[sp_name]
                totals[element_name] = total_expr

            # Get partition expressions from the analytical equilibrium model
            eq_exprs = model.partition_expressions(totals, Temp, RH_sym, eq_def.regime_blending, eq_def.transition_width)

            # Extract equilibrium-specific Jacobian entries (non-zero ∂eq_i/∂C_j)
            eq_jac_entries: list[tuple[int, sp.Expr]] = []
            for i, sp_i_name in enumerate(ordered_species):
                if sp_i_name in eq_exprs:
                    for j, sp_j_name in enumerate(ordered_species):
                        if sp_j_name in species_symbols:
                            deriv = sp.diff(eq_exprs[sp_i_name], species_symbols[sp_j_name])
                            if deriv != 0:
                                eq_jac_entries.append((i, j, deriv))

            # Build total_species_map: element -> species indices
            total_species_map: dict[str, list[int]] = {}
            for element_name, sp_list in eq_def.total_species.items():
                indices = [ordered_species.index(sp_name) for sp_name in sp_list if sp_name in ordered_species]
                total_species_map[element_name] = indices

            # Build regime weights and equilibrium constants for downstream use
            regime_weights = []
            eq_constants = model.equilibrium_constants(Temp)

            eq_result = EquilibriumSymbolicResult(
                partition_exprs=eq_exprs,
                jacobian_entries=eq_jac_entries,
                total_species_map=total_species_map,
                regime_weights=regime_weights,
                equilibrium_constants=eq_constants,
            )
            eq_results.append(eq_result)

        # ------------------------------------------------------------------
        # Build-time differentiability verification for equilibrium expressions.
        # Verify all partition expressions are symbolically differentiable:
        # no Derivative wrappers (SymPy couldn't differentiate), no Piecewise,
        # no Abs (non-smooth operations). This catches errors early before
        # proceeding to code emission.
        # ------------------------------------------------------------------
        from .model import CompilationError

        for eq_result in eq_results:
            for sp_name, expr in eq_result.partition_exprs.items():
                for sym_name, sym in species_symbols.items():
                    deriv = sp.diff(expr, sym)
                    # Check for unevaluated Derivative wrappers
                    # (indicates SymPy couldn't differentiate)
                    if deriv.has(sp.Derivative):
                        raise CompilationError(
                            stage="lowering",
                            message=(
                                f"Equilibrium expression for '{sp_name}' contains "
                                f"non-differentiable operation with respect to '{sym_name}'"
                            ),
                            species_name=sp_name,
                        )
                    # Check for forbidden non-smooth functions
                    if deriv.has(sp.Piecewise) or deriv.has(sp.Abs):
                        raise CompilationError(
                            stage="lowering",
                            message=(
                                f"Equilibrium expression for '{sp_name}' contains "
                                f"non-differentiable operation: Piecewise or Abs "
                                f"detected in derivative w.r.t. '{sym_name}'"
                            ),
                            species_name=sp_name,
                        )

        result["equilibrium_results"] = eq_results
        result["rh_symbol"] = RH_sym
    else:
        result["rh_symbol"] = None

    try:
        # Run sparsity analysis: fill-in prediction, RCM reordering, block detection
        N = jacobian_matrix.shape[0]
        jacobian_structure = set()
        for i in range(N):
            for j in range(N):
                if jacobian_matrix[i, j] != 0:
                    jacobian_structure.add((i, j))

        sparsity = SparsityOptimizer(jacobian_structure, N)
        analysis = sparsity.analyze()

        lu_plan = compute_symbolic_lu_decomposition(
            jacobian_matrix,
            ordered_species,
            permutation=analysis.permutation,
            blocks=analysis.blocks,
            is_block_diagonal=analysis.is_block_diagonal,
        )

        result["symbolic_lu_plan"] = lu_plan
        result["sparsity_analysis"] = analysis
    except Exception:
        # Fallback: compute LU without sparsity optimization
        try:
            lu_plan = compute_symbolic_lu_decomposition(jacobian_matrix, ordered_species)
            result["symbolic_lu_plan"] = lu_plan
        except Exception:
            pass

    return result


def _compute_jacobian_column(args: tuple[int, list[str], str]) -> tuple[int, list[str]]:
    """Worker function for multiprocessing — computes one Jacobian column."""
    j, f_total_serialized, c_j_srepr = args
    import sympy as _sp

    f_total = _sp.Matrix([_sp.sympify(expr) for expr in f_total_serialized])
    c_j = _sp.sympify(c_j_srepr)
    column = f_total.diff(c_j)
    return j, [_sp.srepr(expr) for expr in column]


def _build_f_total(mech: MechanismDefinition) -> dict[str, Any]:
    """
    Shared helper that builds f_total, c_vector, and auxiliary data from a mechanism.
    Used by both sequential and parallel Jacobian builders.
    """
    return _evaluate_reaction_fluxes(mech)


def prepare_unified_jacobian_parallel(mech: MechanismDefinition) -> dict[str, Any]:
    """Same as prepare_unified_jacobian but with parallel Jacobian column computation.

    Uses multiprocessing.Pool to compute each column df/dC_j independently, then
    assembles the full Jacobian in deterministic column order.  Falls back to
    sequential computation if multiprocessing raises an error.
    """
    built = _build_f_total(mech)
    ordered_species = built["ordered_species"]
    f_implicit = built["f_implicit"]
    f_explicit = built["f_explicit"]
    f_total = built["f_total"]
    c_vector = built["c_vector"]
    built["species_symbols"]

    N = len(ordered_species)

    # For mechanisms with < 200 species, the serialization overhead of
    # multiprocessing (pickling SymPy srepr strings, spawning workers, IPC)
    # exceeds the differentiation speedup. Fall back to sequential path.
    if N < 200:
        return prepare_unified_jacobian(mech)

    # Serialize f_total expressions for multiprocessing (SymPy objects aren't picklable)
    f_total_serialized = [sp.srepr(expr) for expr in f_total]
    c_sreprs = [sp.srepr(sym) for sym in c_vector]

    try:
        with multiprocessing.Pool() as pool:
            args = [(j, f_total_serialized, c_sreprs[j]) for j in range(N)]
            results = pool.map(_compute_jacobian_column, args)

        # Assemble in deterministic column order
        results.sort(key=lambda x: x[0])
        columns = []
        for _j, col_strs in results:
            columns.append(sp.Matrix([sp.sympify(s) for s in col_strs]))
        jacobian_matrix = sp.Matrix.hstack(*[col.reshape(N, 1) for col in columns])
    except Exception as e:
        warnings.warn(f"Parallel Jacobian computation failed, falling back to sequential: {e}")
        jacobian_matrix = f_total.jacobian(c_vector)

    adjoint_matrix = jacobian_matrix.transpose()

    # Mass conservation projector
    unique_elements = sorted(list(set(elem for s in mech.species for elem in s.elements.keys())))
    if unique_elements:
        E_matrix = sp.zeros(len(unique_elements), len(ordered_species))
        for j, sp_name in enumerate(ordered_species):
            species_def = next(s for s in mech.species if s.name == sp_name)
            for i, elem in enumerate(unique_elements):
                E_matrix[i, j] = species_def.elements.get(elem, 0)
        try:
            E_E_T = E_matrix * E_matrix.transpose()
            mass_projector = E_matrix.transpose() * E_E_T.pinv()
        except Exception:
            mass_projector = sp.zeros(len(ordered_species), len(unique_elements))
    else:
        E_matrix = sp.zeros(1, len(ordered_species))
        mass_projector = sp.zeros(len(ordered_species), 1)

    result = {
        "species_map": ordered_species,
        "f_implicit": f_implicit,
        "f_explicit": f_explicit,
        "jacobian_matrix": jacobian_matrix,
        "adjoint_matrix": adjoint_matrix,
        "mass_projector": mass_projector,
        "element_map": unique_elements,
    }

    try:
        N = jacobian_matrix.shape[0]
        jacobian_structure = set()
        for i in range(N):
            for j in range(N):
                if jacobian_matrix[i, j] != 0:
                    jacobian_structure.add((i, j))

        sparsity = SparsityOptimizer(jacobian_structure, N)
        analysis = sparsity.analyze()

        lu_plan = compute_symbolic_lu_decomposition(
            jacobian_matrix,
            ordered_species,
            permutation=analysis.permutation,
            blocks=analysis.blocks,
            is_block_diagonal=analysis.is_block_diagonal,
        )
        result["symbolic_lu_plan"] = lu_plan
        result["sparsity_analysis"] = analysis
    except Exception:
        try:
            lu_plan = compute_symbolic_lu_decomposition(jacobian_matrix, ordered_species)
            result["symbolic_lu_plan"] = lu_plan
        except Exception:
            pass

    return result


def compute_symbolic_lu_decomposition(
    J_matrix: sp.Matrix,
    species_map: list[str],
    permutation: "list[int] | None" = None,
    blocks: "list[list[int]] | None" = None,
    is_block_diagonal: bool = False,
) -> SymbolicLUPlan:
    """
    Computes build-time symbolic sparse LU factorization schedule (Doolittle method) on W = inv_g_dt * I - J.
    Extracts flat scalar L, U matrix entry expressions and forward/backward substitution steps referencing previously
    computed scalar variables.

    When is_block_diagonal=True and blocks is provided, compute per-block LU plans independently
    and combine results into a single SymbolicLUPlan with block metadata.

    When permutation is provided (but not block-diagonal), apply the permutation to J before
    computing LU and store the permutation in the resulting plan.
    """
    N = J_matrix.shape[0]
    if N == 0:
        raise ValueError("Cannot perform symbolic LU decomposition on empty matrix.")

    # Handle block-diagonal case: compute per-block LU independently
    if is_block_diagonal and blocks is not None and len(blocks) > 1:
        return _compute_block_diagonal_lu(J_matrix, species_map, blocks)

    # Handle permutation case: reorder J before LU
    if permutation is not None:
        J_perm = sp.zeros(N, N)
        for i in range(N):
            for j in range(N):
                J_perm[i, j] = J_matrix[permutation[i], permutation[j]]
        perm_species_map = [species_map[permutation[i]] for i in range(N)]
        plan = _compute_lu_core(J_perm, perm_species_map)
        plan.permutation = permutation
        plan.blocks = blocks
        return plan

    return _compute_lu_core(J_matrix, species_map)


def _compute_block_diagonal_lu(
    J_matrix: sp.Matrix,
    species_map: list[str],
    blocks: list[list[int]],
) -> SymbolicLUPlan:
    """Compute per-block LU decompositions and combine into a single plan."""
    N = J_matrix.shape[0]
    combined_non_zero_jac = []
    combined_l_exprs = []
    combined_u_exprs = []
    combined_lu_exprs_ordered = []
    combined_forward_steps = []
    combined_backward_steps = []
    total_fill_in = 0

    for block_indices in blocks:
        block_size = len(block_indices)
        # Extract sub-matrix for this block
        sub_matrix = sp.zeros(block_size, block_size)
        for bi, gi in enumerate(block_indices):
            for bj, gj in enumerate(block_indices):
                sub_matrix[bi, bj] = J_matrix[gi, gj]

        sub_species = [species_map[idx] for idx in block_indices]
        sub_plan = _compute_lu_core(sub_matrix, sub_species)

        # Map block-local indices back to global indices
        for i, j, expr_str in sub_plan.non_zero_jacobian:
            gi, gj = block_indices[i], block_indices[j]
            # Remap variable references in expression from local to global
            remapped_expr = _remap_indices(expr_str, block_indices, "W")
            combined_non_zero_jac.append((gi, gj, remapped_expr))

        for i, j, expr_str in sub_plan.l_expressions:
            gi, gj = block_indices[i], block_indices[j]
            remapped_expr = _remap_lu_expr(expr_str, block_indices)
            combined_l_exprs.append((gi, gj, remapped_expr))

        for i, j, expr_str in sub_plan.u_expressions:
            gi, gj = block_indices[i], block_indices[j]
            remapped_expr = _remap_lu_expr(expr_str, block_indices)
            combined_u_exprs.append((gi, gj, remapped_expr))

        for kind, i, j, expr_str in sub_plan.lu_expressions_ordered:
            gi, gj = block_indices[i], block_indices[j]
            remapped_expr = _remap_lu_expr(expr_str, block_indices)
            combined_lu_exprs_ordered.append((kind, gi, gj, remapped_expr))

        for i, expr_str in sub_plan.forward_sub_steps:
            gi = block_indices[i]
            remapped_expr = _remap_solve_expr(expr_str, block_indices, "b", "y", "L")
            combined_forward_steps.append((gi, remapped_expr))

        for i, expr_str in sub_plan.backward_sub_steps:
            gi = block_indices[i]
            remapped_expr = _remap_solve_expr(expr_str, block_indices, "y", "x", "U")
            combined_backward_steps.append((gi, remapped_expr))

        total_fill_in += sub_plan.fill_in_count

    return SymbolicLUPlan(
        num_species=N,
        species_map=species_map,
        non_zero_jacobian=combined_non_zero_jac,
        l_expressions=combined_l_exprs,
        u_expressions=combined_u_exprs,
        lu_expressions_ordered=combined_lu_exprs_ordered,
        forward_sub_steps=combined_forward_steps,
        backward_sub_steps=combined_backward_steps,
        blocks=blocks,
        fill_in_count=total_fill_in,
    )


def _remap_indices(expr_str: str, block_indices: list[int], prefix: str) -> str:
    """Remap local indices in a W_i_j style expression to global indices."""
    import re

    def replacer(m):
        i, j = int(m.group(1)), int(m.group(2))
        return f"{prefix}_{block_indices[i]}_{block_indices[j]}"

    return re.sub(rf"{prefix}_(\d+)_(\d+)", replacer, expr_str)


def _remap_lu_expr(expr_str: str, block_indices: list[int]) -> str:
    """Remap L_i_j, U_i_j, and W_i_j references from block-local to global indices."""
    import re

    def replacer(m):
        prefix = m.group(1)
        i, j = int(m.group(2)), int(m.group(3))
        return f"{prefix}_{block_indices[i]}_{block_indices[j]}"

    return re.sub(r"([LUW])_(\d+)_(\d+)", replacer, expr_str)


def _remap_solve_expr(expr_str: str, block_indices: list[int], rhs_prefix: str, sol_prefix: str, mat_prefix: str) -> str:
    """Remap forward/backward sub expressions from block-local to global indices."""
    import re

    def rhs_replacer(m):
        i = int(m.group(1))
        return f"{rhs_prefix}_{block_indices[i]}"

    def sol_replacer(m):
        i = int(m.group(1))
        return f"{sol_prefix}_{block_indices[i]}"

    def mat_replacer(m):
        i, j = int(m.group(1)), int(m.group(2))
        return f"{mat_prefix}_{block_indices[i]}_{block_indices[j]}"

    result = re.sub(rf"{rhs_prefix}_(\d+)", rhs_replacer, expr_str)
    result = re.sub(rf"{sol_prefix}_(\d+)", sol_replacer, result)
    result = re.sub(rf"{mat_prefix}_(\d+)_(\d+)", mat_replacer, result)
    return result


def _compute_lu_core(J_matrix: sp.Matrix, species_map: list[str]) -> SymbolicLUPlan:
    """Core LU decomposition logic on a single (possibly sub-) matrix."""
    N = J_matrix.shape[0]

    non_zero_jac = []
    non_zero_w = set()
    for i in range(N):
        non_zero_w.add((i, i))  # diagonal is always non-zero due to inv_g_dt
        for j in range(N):
            if J_matrix[i, j] != 0:
                non_zero_jac.append((i, j, str(J_matrix[i, j])))
                non_zero_w.add((i, j))

    nz_L = set((i, i) for i in range(N))
    nz_U = set()

    l_exprs = []
    u_exprs = []
    lu_exprs_ordered = []

    fill_in_count = 0

    for i in range(N):
        # Compute U(i, j) for j >= i
        for j in range(i, N):
            sub_terms = []
            for k in range(i):
                if (i, k) in nz_L and (k, j) in nz_U:
                    sub_terms.append(f"L_{i}_{k} * U_{k}_{j}")

            has_w = (i, j) in non_zero_w
            if has_w or sub_terms:
                if not has_w and sub_terms:
                    fill_in_count += 1
                nz_U.add((i, j))
                rhs = f"W_{i}_{j}" if has_w else "0.0"
                if sub_terms:
                    rhs += " - " + " - ".join(sub_terms)
                u_exprs.append((i, j, rhs))
                lu_exprs_ordered.append(("U", i, j, rhs))

        if (i, i) not in nz_U:
            raise ValueError(
                f"Singular or zero pivot encountered at species index {i} ('{species_map[i]}') "
                "during build-time symbolic LU decomposition."
            )

        # Compute L(j, i) for j > i
        for j in range(i + 1, N):
            sub_terms = []
            for k in range(i):
                if (j, k) in nz_L and (k, i) in nz_U:
                    sub_terms.append(f"L_{j}_{k} * U_{k}_{i}")

            has_w = (j, i) in non_zero_w
            if has_w or sub_terms:
                if not has_w and sub_terms:
                    fill_in_count += 1
                nz_L.add((j, i))
                rhs = f"W_{j}_{i}" if has_w else "0.0"
                if sub_terms:
                    rhs += " - " + " - ".join(sub_terms)
                l_exprs.append((j, i, f"({rhs}) / U_{i}_{i}"))
                lu_exprs_ordered.append(("L", j, i, f"({rhs}) / U_{i}_{i}"))

    # Generate forward substitution steps for L y = b
    forward_steps = []
    for i in range(N):
        sub_terms = []
        for k in range(i):
            if (i, k) in nz_L:
                sub_terms.append(f"L_{i}_{k} * y_{k}")
        if sub_terms:
            expr_str = f"b_{i} - " + " - ".join(sub_terms)
        else:
            expr_str = f"b_{i}"
        forward_steps.append((i, expr_str))

    # Generate backward substitution steps for U x = y
    backward_steps = []
    for i in range(N - 1, -1, -1):
        sub_terms = []
        for k in range(i + 1, N):
            if (i, k) in nz_U:
                sub_terms.append(f"U_{i}_{k} * x_{k}")
        if sub_terms:
            num_str = f"y_{i} - " + " - ".join(sub_terms)
            expr_str = f"({num_str}) / U_{i}_{i}"
        else:
            expr_str = f"y_{i} / U_{i}_{i}"
        backward_steps.append((i, expr_str))

    return SymbolicLUPlan(
        num_species=N,
        species_map=species_map,
        non_zero_jacobian=non_zero_jac,
        l_expressions=l_exprs,
        u_expressions=u_exprs,
        lu_expressions_ordered=lu_exprs_ordered,
        forward_sub_steps=forward_steps,
        backward_sub_steps=backward_steps,
        fill_in_count=fill_in_count,
    )


def compute_transposed_lu_plan(lu_plan: SymbolicLUPlan) -> SymbolicLUPlan:
    """
    Given an existing SymbolicLUPlan, compute the transposed substitution steps.

    For the forward LU solve:  W * x = b  =>  L * U * x = b
      - Forward sub:  L * y = b  (lower-triangular)
      - Backward sub: U * x = y  (upper-triangular)

    For the transposed solve: W^T * x = b  =>  U^T * L^T * x = b
      - Forward sub with U^T:  U^T * y = b  (U^T is lower-triangular)
      - Backward sub with L^T: L^T * x = y  (L^T is upper-triangular)

    The transposed steps are stored in:
      - lu_plan.transpose_forward_sub_steps
      - lu_plan.transpose_backward_sub_steps

    Requirements: 5.1, 5.2
    """
    N = lu_plan.num_species

    # Build non-zero structure sets for L and U from the existing plan
    nz_L = set()  # (row, col) pairs where L is non-zero
    nz_U = set()  # (row, col) pairs where U is non-zero

    for i, j, _expr in lu_plan.l_expressions:
        nz_L.add((i, j))
    # L diagonal is always 1 (unit lower triangular)
    for i in range(N):
        nz_L.add((i, i))

    for i, j, _expr in lu_plan.u_expressions:
        nz_U.add((i, j))

    # Transposed forward substitution: solve U^T * y = b
    # U^T is lower-triangular. U^T[i,j] = U[j,i].
    # For i = 0, 1, ..., N-1:
    #   U^T[i,i] * y_i + Σ_{k<i} U^T[i,k] * y_k = b_i
    #   => U[i,i] * y_i + Σ_{k<i} U[k,i] * y_k = b_i
    #   => y_i = (b_i - Σ_{k<i} U[k,i] * y_k) / U[i,i]
    transpose_forward_steps = []
    for i in range(N):
        sub_terms = []
        for k in range(i):
            # U^T[i,k] = U[k,i] — check if (k, i) is in nz_U
            if (k, i) in nz_U:
                sub_terms.append(f"U_{k}_{i} * y_{k}")

        if sub_terms:
            num_str = f"b_{i} - " + " - ".join(sub_terms)
            expr_str = f"({num_str}) / U_{i}_{i}"
        else:
            expr_str = f"b_{i} / U_{i}_{i}"
        transpose_forward_steps.append((i, expr_str))

    # Transposed backward substitution: solve L^T * x = y
    # L^T is upper-triangular. L^T[i,j] = L[j,i].
    # L is unit lower-triangular so L^T[i,i] = 1.
    # For i = N-1, N-2, ..., 0:
    #   L^T[i,i] * x_i + Σ_{k>i} L^T[i,k] * x_k = y_i
    #   => x_i + Σ_{k>i} L[k,i] * x_k = y_i
    #   => x_i = y_i - Σ_{k>i} L[k,i] * x_k
    transpose_backward_steps = []
    for i in range(N - 1, -1, -1):
        sub_terms = []
        for k in range(i + 1, N):
            # L^T[i,k] = L[k,i] — check if (k, i) is in nz_L
            if (k, i) in nz_L:
                sub_terms.append(f"L_{k}_{i} * x_{k}")

        if sub_terms:
            expr_str = f"y_{i} - " + " - ".join(sub_terms)
        else:
            expr_str = f"y_{i}"
        transpose_backward_steps.append((i, expr_str))

    # Store the transposed steps in the plan
    lu_plan.transpose_forward_sub_steps = transpose_forward_steps
    lu_plan.transpose_backward_sub_steps = transpose_backward_steps

    return lu_plan


def build_sympy_matrices(mech: MechanismDefinition) -> dict[str, Any]:
    """
    Lowering function to compute unified Jacobian and symbolic sparse LU plan.
    Attaches results to mechanism metadata.
    """
    res = prepare_unified_jacobian(mech)
    plan = res.get("symbolic_lu_plan")
    if plan is None:
        plan = compute_symbolic_lu_decomposition(res["jacobian_matrix"], res["species_map"])
        res["symbolic_lu_plan"] = plan

    if getattr(mech, "metadata", None) is None:
        mech.metadata = {}
    mech.metadata["sympy_metadata"] = res
    mech.metadata["symbolic_lu_plan"] = plan
    return res


def apply_cse_to_plan(lu_plan: SymbolicLUPlan, f_vector: sp.Matrix) -> tuple[list, list]:
    """Apply sympy.cse() to all expressions in the LU plan + rate vector.

    Returns:
        replacements: List of (Symbol, expression) tuples for CSE temporaries
        reduced: List of simplified expressions with CSE symbols substituted
    """
    all_exprs = []
    for i, j, expr_str in lu_plan.non_zero_jacobian:
        all_exprs.append(sp.sympify(expr_str))
    for expr in f_vector:
        all_exprs.append(expr)

    replacements, reduced = sp.cse(all_exprs, optimizations="basic")
    return replacements, reduced


class SparsityOptimizer:
    """Analyzes Jacobian sparsity for fill-in prediction, reordering, and block detection."""

    def __init__(self, jacobian_structure: "set[tuple[int, int]]", n: int):
        self.structure = jacobian_structure
        self.n = n

    def predict_fill_in(self) -> "set[tuple[int, int]]":
        """
        Graph-reachability fill-in prediction (symbolic Gaussian elimination).
        For Doolittle LU, position (i,j) fills in if there exists k < min(i,j)
        such that both (i,k) and (k,j) are structurally non-zero (transitively).
        """
        active = set(self.structure)
        active |= {(i, i) for i in range(self.n)}  # diagonal always present
        fill = set()

        for k in range(self.n):
            rows_with_k = [i for i in range(k + 1, self.n) if (i, k) in active]
            cols_with_k = [j for j in range(k + 1, self.n) if (k, j) in active]

            for i in rows_with_k:
                for j in cols_with_k:
                    if (i, j) not in active:
                        fill.add((i, j))
                        active.add((i, j))

        return fill

    def compute_rcm_ordering(self) -> "list[int]":
        """
        Reverse Cuthill-McKee on the symmetrized structure graph.
        Returns permutation vector p where new_index = p[old_index].
        """
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        for i, j in self.structure:
            if i != j:
                G.add_edge(i, j)

        # Handle disconnected graphs (no edges means no bandwidth to reduce)
        if G.number_of_edges() == 0:
            return list(range(self.n))

        # NetworkX provides Cuthill-McKee ordering directly
        # cuthill_mckee_ordering returns nodes in CM order; reverse for RCM
        cm_order = list(nx.utils.rcm.cuthill_mckee_ordering(G))
        # Reverse for RCM (Reverse Cuthill-McKee)
        rcm_order = list(reversed(cm_order))

        def _bw(order):
            inv = [0] * self.n
            for new_idx, old_idx in enumerate(order):
                inv[old_idx] = new_idx
            return max(abs(inv[i] - inv[j]) for i, j in self.structure) if self.structure else 0

        orig_order = list(range(self.n))
        if _bw(rcm_order) > _bw(orig_order):
            return orig_order

        return rcm_order

    def detect_blocks(self) -> "list[list[int]]":
        """
        Run Tarjan SCC on the directed Jacobian structure graph.
        Each SCC with no cross-block edges becomes an independent block.
        """
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        for i, j in self.structure:
            if i != j:
                G.add_edge(i, j)

        sccs = list(nx.strongly_connected_components(G))
        # Sort blocks deterministically by minimum index
        blocks = sorted([sorted(list(scc)) for scc in sccs], key=lambda b: b[0])
        return blocks

    def _check_block_independence(self, blocks: "list[list[int]]") -> bool:
        """
        Check if blocks are independent (no cross-block non-zeros after fill-in).
        Returns True if the system is truly block-diagonal.
        """
        if len(blocks) <= 1:
            return False  # Single block is not "block-diagonal"

        # Build a mapping from species index to block index
        species_to_block: dict[int, int] = {}
        for block_idx, block in enumerate(blocks):
            for species_idx in block:
                species_to_block[species_idx] = block_idx

        # Include fill-in positions in the check
        fill = self.predict_fill_in()
        all_positions = self.structure | fill | {(i, i) for i in range(self.n)}

        # Check for cross-block non-zeros
        for i, j in all_positions:
            if i == j:
                continue
            if species_to_block.get(i) != species_to_block.get(j):
                return False  # Cross-block coupling exists

        return True

    def analyze(self) -> "SparsityAnalysis":
        """Full sparsity analysis pipeline."""
        from .model import SparsityAnalysis

        fill = self.predict_fill_in()
        perm = self.compute_rcm_ordering()
        inv_perm = [0] * self.n
        for new_idx, old_idx in enumerate(perm):
            inv_perm[old_idx] = new_idx

        blocks = self.detect_blocks()
        is_block_diag = self._check_block_independence(blocks)

        return SparsityAnalysis(
            original_nnz=len(self.structure),
            fill_in_positions=fill,
            total_nnz_after_fill=len(self.structure) + len(fill),
            permutation=perm,
            inverse_permutation=inv_perm,
            blocks=blocks,
            is_block_diagonal=is_block_diag,
        )


def annotate_lu_expressions(lu_plan: SymbolicLUPlan) -> "list[AnnotatedLUExpression]":
    """Annotate each LU expression with the set of species indices it depends on.

    For each expression in lu_expressions_ordered, determines which species indices
    affect that entry. An expression at (row, col) directly depends on species row
    and col. Additionally, any W_i_j, L_i_j, or U_i_j references in the expression
    add species i and j to the dependency set.

    Args:
        lu_plan: The symbolic LU plan with lu_expressions_ordered populated.

    Returns:
        List of AnnotatedLUExpression with depends_on sets populated.
    """
    import re

    from .model import AnnotatedLUExpression

    annotated = []
    for kind, row, col, expr_str in lu_plan.lu_expressions_ordered:
        # Direct dependencies: the species at row and column positions
        depends_on: set[int] = {row, col}

        # Also check for references to other W/L/U entries
        for match in re.finditer(r"[WLU]_(\d+)_(\d+)", expr_str):
            depends_on.add(int(match.group(1)))
            depends_on.add(int(match.group(2)))

        annotated.append(
            AnnotatedLUExpression(
                kind=kind,
                row=row,
                col=col,
                expr=expr_str,
                depends_on=depends_on,
            )
        )

    return annotated
