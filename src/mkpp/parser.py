import json
import math
from pathlib import Path
from typing import Any

import yaml

from .model import (
    AerosolRepresentation,
    CompilationError,
    EnvironmentDefinition,
    EquilibriumDefinition,
    MechanismDefinition,
    PhaseDefinition,
    PhaseMode,
    ReactionDefinition,
    SolverMode,
    SpeciesDefinition,
)

_SAFE_YAML_LOADER = yaml.CSafeLoader if yaml.__with_libyaml__ else yaml.SafeLoader

MOLECULE_CM3_TO_MOL_M3 = 1.0e6 / 6.02214076e23
SUPPORTED_CONCENTRATION_UNITS = {"molecule cm-3", "mol m-3"}


def detect_config_format(source: str | Path) -> str:
    """Detect whether a configuration source is JSON or YAML.

    First checks file extension (.json -> json, .yaml/.yml -> yaml).
    If extension is missing or unrecognised, inspects leading content characters.
    """
    p = Path(source)
    if p.suffix.lower() == ".json":
        return "json"
    if p.suffix.lower() in (".yaml", ".yml"):
        return "yaml"

    try:
        if p.exists() and p.is_file():
            content = p.read_text(encoding="utf-8").strip()
        else:
            content = str(source).strip()
        if content.startswith("{") or content.startswith("["):
            return "json"
    except Exception:
        pass

    return "yaml"


def _normalize_species_dict(d: Any) -> dict[str, float]:
    """Normalize MICM/OpenAtmos reactant or product dictionary/list into species -> float yield."""
    if isinstance(d, list):
        result: dict[str, float] = {}
        for item in d:
            if isinstance(item, str):
                result[item] = 1.0
            elif isinstance(item, dict):
                name = item.get("species name", item.get("name"))
                if name:
                    result[str(name)] = float(item.get("coefficient", item.get("yield", 1.0)))
        return result
    if not isinstance(d, dict):
        return {}
    res = {}
    for sp, val in d.items():
        if isinstance(val, int | float):
            res[sp] = float(val)
        elif isinstance(val, dict):
            res[sp] = float(val.get("yield", 1.0))
        elif val is None:
            res[sp] = 1.0
        else:
            try:
                res[sp] = float(val)
            except (ValueError, TypeError):
                res[sp] = 1.0
    return res


def parse_mechanism_micm(
    name: str,
    data: dict[str, Any],
    *,
    convert_openatmos_activation_energy: bool = False,
    convert_kinetic_units_to_si: bool = False,
) -> MechanismDefinition:
    """Parse MICM/OpenAtmos standard dictionary into internal model."""
    if not isinstance(data, dict) or "species" not in data or not data["species"]:
        raise CompilationError(
            stage="parsing",
            message="OpenAtmos v1 data must define at least one species",
        )

    source_concentration_unit = data.get("concentration_unit")
    concentration_unit = source_concentration_unit
    if concentration_unit is not None and (
        not isinstance(concentration_unit, str) or concentration_unit not in SUPPORTED_CONCENTRATION_UNITS
    ):
        raise CompilationError(
            stage="validation",
            message=(
                f"Unsupported concentration_unit '{concentration_unit}'; "
                f"expected one of {sorted(SUPPORTED_CONCENTRATION_UNITS)}"
            ),
        )
    if convert_kinetic_units_to_si and concentration_unit is None:
        raise CompilationError(
            stage="validation",
            message="--convert-kinetic-units-to-si requires mechanism concentration_unit",
        )
    if convert_kinetic_units_to_si and concentration_unit == "molecule cm-3":
        concentration_unit = "mol m-3"

    species = []
    for s in data.get("species", []):
        if not isinstance(s, dict) or not s.get("name"):
            raise CompilationError(
                stage="parsing",
                message="Species in OpenAtmos v1 mechanism must have a name",
            )
        sp_name = s.get("name")
        # Default to GAS if not specified in basic MICM
        phase = PhaseMode.GAS
        sp_type = str(s.get("type", "")).lower()
        sp_role = str(s.get("role", "")).lower()
        # Only explicit OpenAtmos fixed roles and the conventional third-body
        # placeholders are non-evolving. Atmospheric species such as O2 and
        # H2O may look like background constituents, but TS1/MICM evolves them
        # and their tendency must therefore be part of the same ODE system.
        if sp_name in ("AIR", "M") or sp_type == "fixed" or sp_role == "fixed":
            role = "fixed"
        else:
            role = "variable"
        species.append(
            SpeciesDefinition(
                name=sp_name,
                phase=phase,
                role=role,
                solver_atol=s.get("_atol"),
                solver_rtol=s.get("_rtol"),
            )
        )

    phases = []
    for p in data.get("phases", []):
        phases.append(PhaseDefinition(name=p.get("name"), solver_mode=SolverMode.IMPLICIT))

    # Build species name set for validation
    species_names = {s.name for s in species}

    reactions = []
    equilibrium_reactions: list[EquilibriumDefinition] = []

    def micm_signed_parameters(raw: dict[str, Any]) -> dict[str, Any]:
        """Match musica/MICM's signed activation-energy representation.

        The OpenAtmos JSON serialisation carries positive activation energies;
        musica converts them to MICM's signed ``exp(C / T)`` convention when it
        constructs an Arrhenius or Troe object.  MKPP lowers the latter form,
        so apply the same conversion before symbolic rate/Jacobian generation.
        """
        converted = dict(raw)
        for key in ("C", "k0_C", "kinf_C", "C0", "C1", "C2", "C3"):
            value = converted.get(key)
            if isinstance(value, int | float):
                converted[key] = -float(value)
        for key in ("k0", "kinf"):
            value = converted.get(key)
            if isinstance(value, dict):
                nested = dict(value)
                if isinstance(nested.get("C"), int | float):
                    nested["C"] = -float(nested["C"])
                converted[key] = nested
        return converted

    def convert_rate_parameter(value: Any, reaction_name: str, parameter_name: str, order: float) -> float:
        """Convert a concentration-based rate coefficient to mol m-3 units."""
        try:
            coefficient = float(value)
        except (TypeError, ValueError) as exc:
            raise CompilationError(
                stage="validation",
                message=(
                    f"Cannot convert non-numeric {parameter_name} for reaction '{reaction_name}' " "to SI concentration units"
                ),
            ) from exc
        converted = coefficient * MOLECULE_CM3_TO_MOL_M3 ** (1.0 - order)
        if not math.isfinite(converted):
            raise CompilationError(
                stage="validation",
                message=f"SI conversion of {parameter_name} for reaction '{reaction_name}' is not finite",
            )
        return converted

    def is_numeric_parameter(value: Any) -> bool:
        if isinstance(value, bool):
            return False
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        return True

    def convert_reaction_parameters(
        reaction_type: str,
        reaction_name: str,
        reactants: dict[str, float],
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        """Convert supported rate-law pre-exponential factors to SI units."""
        order = sum(reactants.values())
        converted = dict(parameters)

        if reaction_type == "ARRHENIUS":
            if "A" in converted:
                converted["A"] = convert_rate_parameter(converted["A"], reaction_name, "A", order)
        elif reaction_type in {"TROE", "FALLOFF"}:
            for key, molecularity in (("k0", order + 1.0), ("kinf", order)):
                nested = converted.get(key)
                if isinstance(nested, dict) and "A" in nested:
                    nested = dict(nested)
                    nested["A"] = convert_rate_parameter(nested["A"], reaction_name, f"{key}.A", molecularity)
                    converted[key] = nested
                else:
                    flat_key = f"{key}_A"
                    if flat_key in converted:
                        converted[flat_key] = convert_rate_parameter(converted[flat_key], reaction_name, flat_key, molecularity)
        elif reaction_type == "TUNNELING":
            if "A" in converted:
                converted["A"] = convert_rate_parameter(converted["A"], reaction_name, "A", order)
        elif reaction_type == "EP2":
            for key, molecularity in (("A0", order), ("A2", order), ("A3", order + 1.0)):
                if key in converted:
                    converted[key] = convert_rate_parameter(converted[key], reaction_name, key, molecularity)
        elif reaction_type == "EP3":
            for key, molecularity in (("A1", order), ("A2", order + 1.0)):
                if key in converted:
                    converted[key] = convert_rate_parameter(converted[key], reaction_name, key, molecularity)
        elif reaction_type in {"PHOTOLYSIS", "HETEROGENEOUS", "PHASE_CHANGE", "SURFACE"}:
            pass
        elif reaction_type == "UNKNOWN":
            if is_numeric_parameter(converted.get("A")):
                raise CompilationError(
                    stage="validation",
                    message=(
                        f"Cannot convert numeric rate coefficient A for UNKNOWN reaction '{reaction_name}'; "
                        "declare a supported rate type"
                    ),
                )
        else:
            rate_parameter_keys = ("A", "A0", "A1", "A2", "A3", "k0", "kinf", "k0_A", "kinf_A")
            if any(
                is_numeric_parameter(converted.get(key))
                or (isinstance(converted.get(key), dict) and is_numeric_parameter(converted[key].get("A")))
                for key in rate_parameter_keys
            ):
                raise CompilationError(
                    stage="validation",
                    message=f"SI kinetic-unit conversion is not implemented for reaction type '{reaction_type}'",
                )

        return converted

    for r in data.get("reactions", []):
        rtype = r.get("type", "UNKNOWN")
        reaction_name = str(r.get("name", len(reactions)))

        if rtype == "EQUILIBRIUM":
            # Parse EQUILIBRIUM block into EquilibriumDefinition
            system = r.get("system", "")
            raw_total_species = r.get("total_species", {})
            regime_blending = r.get("regime_blending", "sigmoid")
            transition_width = r.get("transition_width", 0.05)
            activity_model = r.get("activity_model", "fixed")
            eq_constants = r.get("equilibrium_constants", {})
            continuous_transition = r.get("continuous_transition", True)

            # Flatten total_species: each element maps to [gas_species, *aerosol_species]
            total_species: dict[str, list[str]] = {}
            for element_name, element_data in raw_total_species.items():
                gas_sp = element_data.get("gas", "")
                aerosol_sp = element_data.get("aerosol", [])
                if isinstance(aerosol_sp, str):
                    aerosol_sp = [aerosol_sp]
                flat_list = [gas_sp] + aerosol_sp
                total_species[element_name] = flat_list

            # Validate all referenced species exist in the mechanism species list
            for element_name, sp_list in total_species.items():
                for sp_name in sp_list:
                    if sp_name not in species_names:
                        raise CompilationError(
                            stage="parsing",
                            message=f"EQUILIBRIUM references unknown species '{sp_name}' " f"in element '{element_name}'",
                            species_name=sp_name,
                        )

            # Validate equilibrium constants have required params (A, dH, Tref)
            for const_name, const_params in eq_constants.items():
                for required_field in ("A", "dH", "Tref"):
                    if required_field not in const_params:
                        raise CompilationError(
                            stage="validation",
                            message=f"Equilibrium constant '{const_name}' requires A, dH, Tref parameters",
                        )

            equilibrium_reactions.append(
                EquilibriumDefinition(
                    system=system,
                    total_species=total_species,
                    regime_blending=regime_blending,
                    transition_width=transition_width,
                    activity_model=activity_model,
                    equilibrium_constants=eq_constants,
                    continuous_transition=continuous_transition,
                    relaxation_timescale_inv=r.get("relaxation_timescale_inv", 1e6),
                )
            )
            # EQUILIBRIUM is not a kinetic reaction — do NOT add to reactions list
            continue

        # OpenAtmos surface uptake reactions encode their gas-phase side with
        # dedicated fields rather than the ordinary reactants/products maps.
        # Normalize them into the common representation so downstream
        # symbolic lowering applies both the concentration dependence and the
        # product stoichiometry.  This is format-level behaviour, not a
        # mechanism-specific convention.
        if rtype.upper() == "SURFACE":
            gas_species = r.get("gas-phase species")
            reactants = {str(gas_species): 1.0} if gas_species else {}
            products = _normalize_species_dict(r.get("gas-phase products", {}))
        else:
            reactants = _normalize_species_dict(r.get("reactants", {}))
            products = _normalize_species_dict(r.get("products", {}))

        # Extract all potential rate parameters instead of just A
        # For MICM compliance, parameters can include k0, kinf, Fc, gamma, etc.
        parameters = {}
        for k, v in r.items():
            if k not in ("type", "reactants", "products", "stiff", "continuous_transition"):
                parameters[k] = v
        if convert_openatmos_activation_energy:
            parameters = micm_signed_parameters(parameters)
        if convert_kinetic_units_to_si and data["concentration_unit"] == "molecule cm-3":
            parameters = convert_reaction_parameters(rtype.upper(), reaction_name, reactants, parameters)

        # Maintain backwards compat for the simple tests
        base_rate = str(r.get("A", ""))

        reactions.append(
            ReactionDefinition(
                reaction_type=rtype,
                reactants=reactants,
                products=products,
                rate_expression=base_rate,
                parameters=parameters,
                stiff=r.get("stiff", False),
                # OpenAtmos/MUSICA photolysis entries are continuous by definition;
                # older YAML mechanisms may opt out explicitly.
                continuous_transition=r.get("continuous_transition", rtype.upper() == "PHOTOLYSIS"),
            )
        )

    # Detect PHASE_CHANGE / EQUILIBRIUM conflicts: collect species from each
    phase_change_species: set[str] = set()
    for rxn in reactions:
        if rxn.reaction_type == "PHASE_CHANGE":
            phase_change_species.update(rxn.reactants.keys())
            phase_change_species.update(rxn.products.keys())

    equilibrium_species: set[str] = set()
    for eq_def in equilibrium_reactions:
        for sp_list in eq_def.total_species.values():
            equilibrium_species.update(sp_list)

    overlap = phase_change_species & equilibrium_species
    if overlap:
        conflict_name = sorted(overlap)[0]
        raise CompilationError(
            stage="validation",
            message=f"Species '{conflict_name}' cannot have both " f"PHASE_CHANGE and EQUILIBRIUM declarations",
            species_name=conflict_name,
        )

    from .model import ArrayDefinition, HostInterfaceSchema

    host_interface = None
    if "host_interface" in data and "arrays" in data["host_interface"]:
        arrays = []
        for arr_data in data["host_interface"]["arrays"]:
            arrays.append(
                ArrayDefinition(
                    name=arr_data.get("name", "unknown"),
                    rank=arr_data.get("rank", 0),
                    layout=arr_data.get("layout", "LayoutLeft"),
                    extent=arr_data.get("extent"),
                    unit=arr_data.get("unit", "unknown"),
                    ownership=arr_data.get("ownership", "host"),
                )
            )
        host_interface = HostInterfaceSchema(arrays=arrays)

    # A kinetic reaction may declare an `activation_trigger` naming a runtime
    # meteorological quantity, e.g. "meteo.cloud_liquid_water > 1.0e-6". When
    # present, the reaction's rate is gated by a smooth indicator of that
    # quantity and the generated solver must accept it as an extra per-cell
    # equilibrium input. Only cloud liquid water is currently supported; an
    # unrecognized quantity is rejected rather than silently ignored, because a
    # dropped trigger would run ungated chemistry (the exact failure R1 guards).
    _TRIGGER_QUANTITIES = {"meteo.cloud_liquid_water": "cloud_liquid_water"}
    cloud_gated = False
    for rxn in reactions:
        trigger = rxn.parameters.get("activation_trigger")
        if trigger is None:
            continue
        quantity = str(trigger).split(">")[0].strip()
        if quantity not in _TRIGGER_QUANTITIES:
            raise CompilationError(
                stage="validation",
                message=(
                    f"reaction declares unsupported activation_trigger '{trigger}'; "
                    f"supported quantities: {sorted(_TRIGGER_QUANTITIES)}"
                ),
            )
        cloud_gated = True

    return MechanismDefinition(
        name=name,
        description=data.get("description", ""),
        aerosol_representation=AerosolRepresentation.BULK,
        species=species,
        phases=phases,
        reactions=reactions,
        host_interface=host_interface,
        equilibrium_reactions=equilibrium_reactions,
        metadata=data.get("metadata", {}),
        has_cloud_gated_reaction=cloud_gated,
        concentration_unit=concentration_unit,
        kinetic_units_converted_to_si=(convert_kinetic_units_to_si and source_concentration_unit == "molecule cm-3"),
    )


def load_mechanism(path: str | Path, *, convert_kinetic_units_to_si: bool = False) -> MechanismDefinition:
    """Load an OpenAtmos v1 chemical mechanism from a JSON or YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mechanism configuration file not found: {p}")

    fmt = detect_config_format(p)
    try:
        with open(p, encoding="utf-8") as f:
            if fmt == "json":
                data = json.load(f)
            else:
                data = yaml.load(f, Loader=_SAFE_YAML_LOADER)
    except json.JSONDecodeError as e:
        raise CompilationError(
            stage="parsing",
            message=f"JSON syntax error in mechanism file '{p}': line {e.lineno}, column {e.colno} ({e.msg})",
            yaml_location=f"{p}:{e.lineno}:{e.colno}",
        ) from e
    except yaml.YAMLError as e:
        loc = str(p)
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            loc = f"{p}:{e.problem_mark.line + 1}:{e.problem_mark.column + 1}"
        raise CompilationError(
            stage="parsing",
            message=f"YAML syntax error in mechanism file '{p}': {e}",
            yaml_location=loc,
        ) from e

    if not isinstance(data, dict):
        raise CompilationError(
            stage="parsing",
            message=f"Mechanism configuration file '{p}' must contain a key-value dictionary",
            yaml_location=str(p),
        )

    # OpenAtmos/MUSICA and MICM both store Arrhenius activation terms in the
    # signed exp(C / T) convention.  Preserve the source value verbatim.
    # OpenAtmos catalog entries are conventionally named ``mechanism.json``.
    # The source document name describes the chemistry; an optional catalog
    # mechanism ID provides the stable host-facing artifact name.  This keeps
    # generated include paths stable without deriving identity from the shared
    # file stem.
    metadata = data.get("metadata", {})
    mechanism_id = metadata.get("mkpp_mechanism_id") if isinstance(metadata, dict) else None
    return parse_mechanism_micm(
        str(mechanism_id or data.get("name", p.stem)),
        data,
        convert_kinetic_units_to_si=convert_kinetic_units_to_si,
    )


def load_environment(path: str | Path) -> EnvironmentDefinition:
    """Load an OpenAtmos v1 environmental configuration from a JSON or YAML file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Environment configuration file not found: {p}")

    fmt = detect_config_format(p)
    try:
        with open(p, encoding="utf-8") as f:
            if fmt == "json":
                data = json.load(f)
            else:
                data = yaml.load(f, Loader=_SAFE_YAML_LOADER)
    except json.JSONDecodeError as e:
        raise CompilationError(
            stage="parsing",
            message=f"JSON syntax error in environment file '{p}': line {e.lineno}, column {e.colno} ({e.msg})",
            yaml_location=f"{p}:{e.lineno}:{e.colno}",
        ) from e
    except yaml.YAMLError as e:
        loc = str(p)
        if hasattr(e, "problem_mark") and e.problem_mark is not None:
            loc = f"{p}:{e.problem_mark.line + 1}:{e.problem_mark.column + 1}"
        raise CompilationError(
            stage="parsing",
            message=f"YAML syntax error in environment file '{p}': {e}",
            yaml_location=loc,
        ) from e

    if not isinstance(data, dict):
        raise CompilationError(
            stage="parsing",
            message=f"Environment configuration file '{p}' must contain a key-value dictionary",
            yaml_location=str(p),
        )

    env_block = data.get("environment", data.get("meteorology", data))

    temp = float(env_block.get("temperature", env_block.get("T", 298.15)))
    press = float(env_block.get("pressure", env_block.get("P", 101325.0)))
    air_dens = float(env_block.get("air_density", env_block.get("M", 2.46e19)))
    rh = float(env_block.get("relative_humidity", env_block.get("RH", 0.5)))
    solver_block = data.get("solver", {})
    if not isinstance(solver_block, dict):
        solver_block = {}
    solver_atol = solver_block.get("atol")
    solver_rtol = solver_block.get("rtol")
    solver_seed = solver_block.get("initial_step_seed")

    init_conc = data.get("initial_concentrations", data.get("initial_conditions", data.get("concentrations", {})))
    if not isinstance(init_conc, dict):
        init_conc = {}

    normalized_init = {str(k): float(v) for k, v in init_conc.items() if isinstance(v, int | float | str)}

    return EnvironmentDefinition(
        temperature=temp,
        pressure=press,
        air_density=air_dens,
        relative_humidity=rh,
        solver_atol=float(solver_atol) if solver_atol is not None else None,
        solver_rtol=float(solver_rtol) if solver_rtol is not None else None,
        solver_initial_step_seed=float(solver_seed) if solver_seed is not None else None,
        initial_concentrations=normalized_init,
    )
