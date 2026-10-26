import argparse
import json
import sys
import warnings
from pathlib import Path

import yaml

from .codegen import generate_headers
from .lowering import partition_reactions
from .model import CompilationError
from .parser import MOLECULE_CM3_TO_MOL_M3, load_environment, load_mechanism
from .validation import sanitize_path, validate_mechanism, validate_mpi_safety

# Species that belong to the NH4/NO3/SO4 equilibrium system
_NH4_NO3_SO4_SPECIES = frozenset({"NH3", "NH4a", "HNO3", "NO3an1", "NO3an2", "NO3an3", "SO2", "SO4"})

# Standard EQUILIBRIUM block for the NH4_NO3_SO4 system
_EQUILIBRIUM_BLOCK = {
    "type": "EQUILIBRIUM",
    "system": "NH4_NO3_SO4",
    "total_species": {
        "reduced_nitrogen": {
            "gas": "NH3",
            "aerosol": ["NH4a"],
        },
        "oxidized_nitrogen": {
            "gas": "HNO3",
            "aerosol": ["NO3an1", "NO3an2", "NO3an3"],
        },
        "sulfate": {
            "gas": "SO2",
            "aerosol": ["SO4"],
        },
    },
    "regime_blending": "sigmoid",
    "transition_width": 0.05,
    "equilibrium_constants": {
        "Kp_NH4NO3": {
            "A": 4.39e-17,
            "dH": -74735.0,
            "Tref": 298.15,
        },
        "Kp_NH4HSO4": {
            "A": 1.086e-2,
            "dH": -40000.0,
            "Tref": 298.15,
        },
        "Kp_NH42SO4": {
            "A": 1.817e-25,
            "dH": -160000.0,
            "Tref": 298.15,
        },
    },
    "continuous_transition": True,
}


def migrate_equilibrium(mech_path: str) -> str | None:
    """Migrate PHASE_CHANGE reactions for NH4/NO3/SO4 species to an EQUILIBRIUM block.

    Rewrites the mechanism YAML, removing PHASE_CHANGE reactions that affect
    NH4/NO3/SO4 species and inserting the standard EQUILIBRIUM block.

    Args:
        mech_path: Path to the mechanism YAML file.

    Returns:
        Path to the migrated YAML file, or None if no migration was needed.
    """
    with open(mech_path) as f:
        data = yaml.safe_load(f)

    if data is None:
        return None

    reactions = data.get("reactions", [])

    # Check idempotency: if EQUILIBRIUM block already exists, skip
    for rxn in reactions:
        if rxn.get("type") == "EQUILIBRIUM":
            print(
                "[migrate-equilibrium] Mechanism already contains an EQUILIBRIUM block. " "No migration needed.",
                file=sys.stderr,
            )
            return None

    # Find PHASE_CHANGE reactions that overlap with NH4/NO3/SO4 species
    migrated_reactions = []
    remaining_reactions = []

    for rxn in reactions:
        if rxn.get("type") == "PHASE_CHANGE":
            reactant_species = set(rxn.get("reactants", {}).keys())
            product_species = set(rxn.get("products", {}).keys())
            all_species = reactant_species | product_species

            if all_species & _NH4_NO3_SO4_SPECIES:
                migrated_reactions.append(rxn)
                continue

        remaining_reactions.append(rxn)

    if not migrated_reactions:
        print(
            "[migrate-equilibrium] No PHASE_CHANGE reactions found for NH4/NO3/SO4 species. " "No migration needed.",
            file=sys.stderr,
        )
        return None

    # Insert the EQUILIBRIUM block
    remaining_reactions.append(_EQUILIBRIUM_BLOCK)
    data["reactions"] = remaining_reactions

    # Write the modified YAML to a new file
    mech_path_obj = Path(mech_path)
    output_path = mech_path_obj.parent / f"{mech_path_obj.stem}_equilibrium{mech_path_obj.suffix}"

    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    # Emit deprecation warning
    warnings.warn(
        "PHASE_CHANGE reactions for NH4/NO3/SO4 species are deprecated. "
        "Use EQUILIBRIUM blocks instead. "
        f"Migrated mechanism written to: {output_path}",
        DeprecationWarning,
        stacklevel=2,
    )

    # Print summary
    migrated_species: set[str] = set()
    for rxn in migrated_reactions:
        migrated_species.update(rxn.get("reactants", {}).keys())
        migrated_species.update(rxn.get("products", {}).keys())

    print(
        f"[migrate-equilibrium] Migration summary:\n"
        f"  Removed {len(migrated_reactions)} PHASE_CHANGE reaction(s)\n"
        f"  Migrated species: {', '.join(sorted(migrated_species))}\n"
        f"  Inserted EQUILIBRIUM block (system: NH4_NO3_SO4)\n"
        f"  Output: {output_path}",
        file=sys.stderr,
    )

    return str(output_path)


def _verbose_log(stage: str, message: str, verbose: bool) -> None:
    """Emit a verbose progress line to stderr if verbose mode is enabled."""
    if verbose:
        print(f"[{stage}] {message}", file=sys.stderr)


def run_compiler(
    mech_path: str,
    env_path: str,
    out_dir: str,
    strict: bool = False,
    emit_manifest: bool = True,
    enable_drgep: bool = False,
    drgep_threshold: float = 0.05,
    report: bool = False,
    verbose: bool = False,
    dry_run: bool = False,
    no_cache: bool = False,
    solver: str = "ros3",
    adjoint: bool = False,
    migrate_equilibrium_flag: bool = False,
    simd_backend: str = "native",
    generate_host_api: bool = False,
    mechanism_id: str | None = None,
    emit_reference_backend: bool = False,
    convert_kinetic_units_to_si: bool = False,
) -> None:
    """Orchestrate the compilation pipeline."""

    mech_path = sanitize_path(mech_path)
    env_path = sanitize_path(env_path)
    out_dir = sanitize_path(out_dir)

    # Solver name to pass to code generation
    solver_name = solver

    # --- Migration step (before compilation) ---
    if migrate_equilibrium_flag:
        migrated_path = migrate_equilibrium(mech_path)
        if migrated_path is not None:
            mech_path = migrated_path

    try:
        # --drgep: fail-fast reject before any SymPy computation
        if enable_drgep:
            raise CompilationError(
                stage="validation",
                message="DRGEP is not supported.",
            )

        # --- Parsing stage ---
        _verbose_log("parsing", f"Loading mechanism from {mech_path}", verbose)

        env = load_environment(env_path)
        env_config = {
            "temperature": env.temperature,
            "pressure": env.pressure,
            "air_density": env.air_density,
            "relative_humidity": env.relative_humidity,
            "initial_concentrations": env.initial_concentrations,
        }

        validate_mpi_safety(env_config)

        mech = load_mechanism(mech_path, convert_kinetic_units_to_si=convert_kinetic_units_to_si)
        if mech.kinetic_units_converted_to_si:
            env.air_density *= MOLECULE_CM3_TO_MOL_M3
        # Artifact identity is intentionally independent from the OpenAtmos
        # document's descriptive ``name``.  Benchmark and host integrations
        # can therefore use a stable, valid C++ identifier without copying or
        # modifying the shared chemistry input.
        if mechanism_id is not None:
            if not mechanism_id.isidentifier():
                raise CompilationError(
                    stage="validation",
                    message="--mechanism-id must be a valid C++ identifier",
                )
            mech.name = mechanism_id
        if not hasattr(mech, "metadata") or mech.metadata is None:
            mech.metadata = {}
        # Preserve the compile-time test environment for code generation.
        # Mechanisms without runtime meteorology parameters embed this state in
        # their Arrhenius/Troe expressions.
        mech.metadata["temperature"] = str(env.temperature)
        mech.metadata["air_density"] = str(env.air_density)
        if env.solver_atol is not None:
            mech.metadata["atol"] = [env.solver_atol] * len(mech.species)
        if env.solver_rtol is not None:
            mech.metadata["rtol"] = [env.solver_rtol] * len(mech.species)
        if env.solver_initial_step_seed is not None:
            mech.metadata["initial_step_seed"] = env.solver_initial_step_seed

        # --- Validation stage ---
        _verbose_log("validation", "Validating mechanism schema and constraints", verbose)

        validate_mechanism(mech, strict=strict)

        from .validation import validate_fuzzer_stiffness

        dummy_max_condition = 1e5
        validate_fuzzer_stiffness(max_condition_number=dummy_max_condition)

        from .validation import validate_mass_conservation, validate_terminator_safety

        validate_terminator_safety(mech)
        validate_mass_conservation(mech)

        # --- Dry-run exit point ---
        if dry_run:
            num_species = len(getattr(mech, "species", []))
            num_reactions = len(getattr(mech, "reactions", []))
            print(
                f"[dry-run] Validation passed. " f"Species: {num_species}, Reactions: {num_reactions}. " f"No code generated.",
                file=sys.stderr,
            )
            return

        # --- Partitioning stage ---
        _verbose_log("partitioning", "Partitioning reactions into blocks", verbose)

        blocks = partition_reactions(mech)
        mech.partition_metadata = blocks.get("metadata")

        # --- Cache check and lowering stage ---
        from .cache_manager import CacheEntry, CacheManager

        cache_mgr = CacheManager()
        cache_key = cache_mgr.compute_key(
            Path(mech_path),
            convert_kinetic_units_to_si=convert_kinetic_units_to_si,
        )

        cached_entry = None
        if not no_cache:
            cached_entry = cache_mgr.lookup(cache_key)

        # --- Lowering stage ---
        from .lowering import prepare_adjoint_and_tlm, prepare_unified_jacobian

        if cached_entry is not None:
            _verbose_log("lowering", "Using cached symbolic Jacobian", verbose)
            sympy_meta = {
                "species_map": cached_entry.species_map,
                "jacobian_matrix": cached_entry.jacobian_matrix,
                "f_implicit": cached_entry.f_implicit,
                "f_explicit": cached_entry.f_explicit,
                "symbolic_lu_plan": cached_entry.lu_plan,
                "adjoint_matrix": cached_entry.jacobian_matrix.transpose() if cached_entry.jacobian_matrix is not None else None,
            }
            mech.sympy_metadata = sympy_meta
        else:
            # --- Lowering stage ---
            _verbose_log("lowering", "Computing symbolic Jacobian and LU decomposition", verbose)

            prepare_adjoint_and_tlm(mech)
            mech.sympy_metadata = prepare_unified_jacobian(mech)

            # Store to cache for next time
            sympy_meta = mech.sympy_metadata
            if sympy_meta is not None:
                entry = CacheEntry(
                    key=cache_key,
                    species_map=sympy_meta.get("species_map", []),
                    jacobian_matrix=sympy_meta.get("jacobian_matrix"),
                    lu_plan=sympy_meta.get("symbolic_lu_plan"),
                    f_implicit=sympy_meta.get("f_implicit"),
                    f_explicit=sympy_meta.get("f_explicit"),
                )
                cache_mgr.store(cache_key, entry)

        # --- Code generation stage ---
        _verbose_log("codegen", f"Generating headers to {out_dir}", verbose)

        generate_headers(
            mech,
            out_dir=out_dir,
            suffix="",
            solver_name=solver_name,
            adjoint=adjoint,
            generate_host_api=generate_host_api,
            simd_backend=simd_backend,
            emit_reference_backend=emit_reference_backend,
        )

        if report:
            from .reporting import write_report

            write_report(mech, mech.sympy_metadata, out_dir, suffix="")

        if emit_manifest:
            print(f"Manifest and headers emitted to {out_dir}")

    except CompilationError as ce:
        print(json.dumps(ce.to_dict()), file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        # Wrap unexpected errors in CompilationError for structured reporting
        error = CompilationError(
            stage="unknown",
            message=str(e),
        )
        print(json.dumps(error.to_dict()), file=sys.stderr)
        sys.exit(1)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(prog="mkpp", description="Multiphase KPP (MKPP) Engine Compiler")

    subparsers = parser.add_subparsers(dest="command", required=True)

    compile_parser = subparsers.add_parser("compile", help="Compile a mechanism into Kokkos headers")
    compile_parser.add_argument("mechanism", help="Path to the mechanism YAML/JSON file")
    compile_parser.add_argument("--test-env", required=True, help="Path to the test environment YAML/JSON file")
    compile_parser.add_argument("--out", default="mkpp-generated/", help="Output directory for generated artifacts")
    compile_parser.add_argument(
        "--mechanism-id",
        help="Stable generated-artifact identifier; does not modify the OpenAtmos input",
    )
    compile_parser.add_argument("--strict", action="store_true", help="Enable strict schema validation")
    compile_parser.add_argument("--emit-manifest", action="store_true", help="Emit metadata manifest alongside headers")
    compile_parser.add_argument("--report", action="store_true", help="Generate full mechanism analysis report and graph")
    compile_parser.add_argument(
        "--drgep",
        action="store_true",
        help="Reject compilation: DRGEP not supported.",
    )
    compile_parser.add_argument(
        "--drgep-threshold",
        type=float,
        default=0.05,
        help="DRGEP interaction threshold for pruning (0.0 to 1.0)",
    )
    compile_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit progress messages to stderr at each pipeline stage",
    )
    compile_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run parsing and validation only; do not generate code",
    )
    compile_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Skip cache lookup and recompute all symbolic matrices",
    )
    compile_parser.add_argument(
        "--solver",
        choices=["ros2", "ros3", "ros4", "rodas3", "rodas4"],
        default="ros3",
        help="Rosenbrock solver method (default: ros3)",
    )
    compile_parser.add_argument(
        "--adjoint",
        action="store_true",
        default=False,
        help="Deprecated compatibility option; enable emitted adjoint/TLM APIs with -DMKPP_ENABLE_ADJOINT=ON",
    )
    compile_parser.add_argument(
        "--migrate-equilibrium",
        action="store_true",
        default=False,
        help="Rewrite mechanism YAML replacing PHASE_CHANGE blocks for NH4/NO3/SO4 "
        "species with an equivalent EQUILIBRIUM declaration (deprecated migration tool)",
    )
    compile_parser.add_argument(
        "--simd-backend",
        choices=["native", "kokkos_batched"],
        default="native",
        help="SIMD vector engine backend for Wide<W> (default: native)",
    )
    compile_parser.add_argument(
        "--host-api",
        action="store_true",
        default=False,
        help="Generate C, C++, and Fortran host API headers and C wrapper source",
    )
    compile_parser.add_argument(
        "--convert-kinetic-units-to-si",
        action="store_true",
        help=(
            "Convert supported rate-law pre-exponential factors from a mechanism's "
            "molecule cm-3 concentration basis to mol m-3; the solver state must then use mol m-3"
        ),
    )
    compile_parser.add_argument("--emit-unrolled-reference", action="store_true")

    parsed_args = parser.parse_args(args)

    if parsed_args.command == "compile":
        run_compiler(
            mech_path=parsed_args.mechanism,
            env_path=parsed_args.test_env,
            out_dir=parsed_args.out,
            strict=parsed_args.strict,
            emit_manifest=parsed_args.emit_manifest,
            enable_drgep=getattr(parsed_args, "drgep", False),
            drgep_threshold=getattr(parsed_args, "drgep_threshold", 0.05),
            report=getattr(parsed_args, "report", False),
            verbose=getattr(parsed_args, "verbose", False),
            dry_run=getattr(parsed_args, "dry_run", False),
            no_cache=getattr(parsed_args, "no_cache", False),
            solver=parsed_args.solver,
            adjoint=parsed_args.adjoint,
            migrate_equilibrium_flag=parsed_args.migrate_equilibrium,
            simd_backend=parsed_args.simd_backend,
            generate_host_api=parsed_args.host_api,
            mechanism_id=parsed_args.mechanism_id,
            emit_reference_backend=parsed_args.emit_unrolled_reference,
            convert_kinetic_units_to_si=parsed_args.convert_kinetic_units_to_si,
        )
        sys.exit(0)


if __name__ == "__main__":
    main()
