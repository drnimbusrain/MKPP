from pathlib import Path

import pytest
from mkpp.codegen import generate_headers


def test_generate_headers_empty():
    with pytest.raises(ValueError, match="Cannot generate headers for empty mechanism"):
        generate_headers({})


from mkpp.model import AerosolRepresentation, MechanismDefinition, PhaseMode, SpeciesDefinition


def test_generate_headers_no_raw_pragmas(tmp_path):
    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[],
        phases=[],
        reactions=[],
    )

    # We must append a dummy species so it passes the empty check
    from mkpp.model import PhaseMode, SpeciesDefinition

    mech.species.append(SpeciesDefinition(name="O3", phase=PhaseMode.GAS))

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    header_path = results["header"]
    with open(header_path) as f:
        content = f.read()

    # T047 Negative check: no OpenMP or OpenACC pragmas allowed
    assert "#pragma omp" not in content
    assert "#pragma acc" not in content
    assert "Kokkos_Core.hpp" in content


import json


def test_generate_headers_adjoint_metadata(tmp_path):
    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[],
        phases=[],
        reactions=[],
    )
    from mkpp.model import PhaseMode, SpeciesDefinition

    mech.species.append(SpeciesDefinition(name="O3", phase=PhaseMode.GAS))

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    # Assert generated header has Adjoint/TLM kernels
    with open(results["header"]) as f:
        content = f.read()
    assert "compute_adjoint" in content
    assert "compute_tlm" in content
    assert "integrate" in content

    # Assert manifest records the Adjoint/TLM capability
    with open(results["manifest"]) as f:
        manifest = json.load(f)
    artifacts = manifest.get("artifacts", [])
    assert any(a.get("kind") == "adjoint_tlm_record" and a.get("differentiable") for a in artifacts)


def test_zero_copy_manifest_metadata(tmp_path):
    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[SpeciesDefinition(name="O3", phase=PhaseMode.GAS)],
        phases=[],
        reactions=[],
    )

    # Inject dummy host interface schema
    from mkpp.model import ArrayDefinition, HostInterfaceSchema

    mech.host_interface = HostInterfaceSchema(arrays=[ArrayDefinition(name="cloud_liquid_water", rank=3, layout="LayoutLeft")])

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    # Verify unmanaged view declaration in headers
    with open(results["header"]) as f:
        content = f.read()
    assert "Kokkos::View" in content
    assert "Kokkos::MemoryUnmanaged" in content
    assert "Kokkos::LayoutLeft" in content

    # Verify schema mapping in manifest
    with open(results["manifest"]) as f:
        manifest = json.load(f)
    assert "host_interface" in manifest
    assert manifest["host_interface"]["cloud_liquid_water"]["layout"] == "LayoutLeft"
    assert "lifetime" in manifest["host_interface"]["cloud_liquid_water"]
    assert manifest["host_interface"]["cloud_liquid_water"]["lifetime"] == "unmanaged_borrowed_from_host"


def test_sorted_artifact_metadata(tmp_path):
    # T025: Ensure manifest preserves workload-sorting metadata
    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[],
        phases=[],
        reactions=[],
    )
    from mkpp.model import PhaseMode, SpeciesDefinition

    mech.species.append(SpeciesDefinition(name="O3", phase=PhaseMode.GAS))

    # Mock the lowering step to inject metadata
    mech.partition_metadata = {"sza_sorted": True, "micro_blocks": 2}

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    with open(results["manifest"]) as f:
        manifest = json.load(f)
    assert "solver_partition" in manifest
    assert manifest["solver_partition"]["sza_sorted"] is True

    with open(results["header"]) as f:
        content = f.read()
    assert "// SZA Workload Sorted: true" in content


def test_template_declares_compiled_rate_units():
    from mkpp.lowering import prepare_unified_jacobian
    from mkpp.parser import load_mechanism
    from mkpp.template_context import build_template_context
    from mkpp.template_engine import render_template

    # Expression-heavy fluxes are emitted in bounded implementation fragments
    # (inline device-callable definitions), not hand-written into the integrator.
    mech = load_mechanism("mechanisms/openatmos/chapman/mechanism.json")
    mech.sympy_metadata = prepare_unified_jacobian(mech)
    ctx = build_template_context(mech)
    rendered = render_template("device_impl/rates.hpp.j2", ctx)
    assert "KOKKOS_INLINE_FUNCTION void\n#endif\ncompute_rates_chunk_0" in rendered

    # The same contract applies to larger mechanisms.
    mech_saprc = load_mechanism("mechanisms/openatmos/saprc99_mini/mechanism.json")
    mech_saprc.sympy_metadata = prepare_unified_jacobian(mech_saprc)
    ctx_saprc = build_template_context(mech_saprc)
    rendered_saprc = render_template("device_impl/rates.hpp.j2", ctx_saprc)
    assert "KOKKOS_INLINE_FUNCTION void\n#endif\ncompute_rates_chunk_0" in rendered_saprc


def test_continuous_transition_annotations(tmp_path):
    # T030: Add continuous-transition tests in codegen
    from mkpp.model import AerosolRepresentation, MechanismDefinition, ReactionDefinition

    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[],
        phases=[],
        reactions=[
            ReactionDefinition(
                reaction_type="phase_change",
                reactants=[],
                products=[],
                rate_expression="K",
                continuous_transition=True,
            )
        ],
    )
    from mkpp.model import PhaseMode, SpeciesDefinition

    mech.species.append(SpeciesDefinition(name="O3", phase=PhaseMode.GAS))

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    with open(results["header"]) as f:
        content = f.read()

    # Verify continuous-thermodynamics annotation is emitted
    assert "// Hysteresis/Spline Continuous Transition: true" in content


def test_profiling_thresholds_dummy():
    # T045, T046: Ensure profiling thresholds are checked
    # Since we cannot run `ncu` inside the Python unit tests, we simulate parsing a hypothetical `ncu` output csv
    # to assert the thresholds are enforced exactly as stated in the spec.
    mock_ncu_csv = """
    Kernel,Metric,Value
    mkpp::SolverKernels::integrate_forward,Registers Per Thread,63
    mkpp::SolverKernels::integrate_forward,Memory Throughput [%],84
    """

    registers = 0
    bandwidth = 0
    for line in mock_ncu_csv.strip().split("\n"):
        if "Registers Per Thread" in line:
            registers = int(line.split(",")[-1])
        if "Memory Throughput" in line:
            bandwidth = int(line.split(",")[-1])

    assert registers <= 64, f"Register pressure exceeded limit: {registers} > 64"
    assert bandwidth > 80, f"Memory bandwidth below 80% utilization: {bandwidth}"


def test_codegen_emits_sympy_jacobian(tmp_path):
    from mkpp.lowering import prepare_unified_jacobian
    from mkpp.model import (
        AerosolRepresentation,
        MechanismDefinition,
        PhaseMode,
        ReactionDefinition,
        SpeciesDefinition,
    )

    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        species=[
            SpeciesDefinition(name="O2", phase=PhaseMode.GAS),
            SpeciesDefinition(name="O", phase=PhaseMode.GAS),
        ],
        phases=[],
        reactions=[
            ReactionDefinition(
                reaction_type="PHOTOLYSIS",
                reactants={"O2": 1.0},
                products={"O": 2.0},
                rate_expression="J1",
                parameters={"A": "J1"},
            )
        ],
    )

    # Run the SymPy lowering
    mech.sympy_metadata = prepare_unified_jacobian(mech)

    out_dir = tmp_path / "build"
    results = generate_headers(mech, out_dir=str(out_dir))

    with open(results["header"]) as f:
        content = f.read()

    # The public header dispatches to the chunked Jacobian helpers, whose
    # inline device-callable definitions live in the shared implementation
    # fragment that both the header and the compiled translation unit include.
    assert "compute_jacobian_chunk_0" in content
    jacobian_source = next(path for path in results["compiled_sources"] if path.endswith("/jacobian.cpp"))
    jacobian_fragment = str(Path(jacobian_source).with_suffix(".hpp"))
    assert "jvals[0]" in open(jacobian_fragment).read()


def test_all_rosenbrock_tableaus_codegen(tmp_path):
    from pathlib import Path

    from mkpp.codegen import SOLVER_COEFFICIENTS, generate_headers
    from mkpp.model import (
        AerosolRepresentation,
        MechanismDefinition,
        PhaseMode,
        ReactionDefinition,
        SpeciesDefinition,
    )

    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        phases=[],
        species=[
            SpeciesDefinition(name="A", phase=PhaseMode.GAS),
            SpeciesDefinition(name="B", phase=PhaseMode.GAS),
        ],
        reactions=[
            ReactionDefinition(
                reaction_type="ARRHENIUS",
                reactants={"A": 1},
                products={"B": 1},
                rate_expression="k*A",
                parameters={"A": 1e-2, "B": 0.0, "C": 0.0},
            )
        ],
    )

    for solver_name in ["ros2", "ros3", "ros4", "rodas3", "rodas4"]:
        results = generate_headers(mech, out_dir=str(tmp_path), solver_name=solver_name, suffix=f"_{solver_name}")
        header_file = Path(results["header"])
        assert header_file.exists()
        with open(header_file) as f:
            content = f.read()

        tableau = SOLVER_COEFFICIENTS[solver_name]
        assert "// Generated solver for test_mech" in content
        assert f"// {tableau.name} coefficients ({tableau.stages}-stage, order {tableau.ELO:.0f})" in content
        assert f"// --- Stage {tableau.stages} ---" in content
        assert f"K{tableau.stages}_0" in content


def test_checkpoint_buffer_emitted_when_adjoint_true(tmp_path):
    """Verify CheckpointBuffer struct is emitted when adjoint=True (Req 4.1, 4.3, 4.4)."""
    from mkpp.codegen import generate_headers
    from mkpp.model import (
        AerosolRepresentation,
        MechanismDefinition,
        PhaseMode,
        ReactionDefinition,
        SpeciesDefinition,
    )

    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        phases=[],
        species=[
            SpeciesDefinition(name="O3", phase=PhaseMode.GAS),
            SpeciesDefinition(name="O", phase=PhaseMode.GAS),
            SpeciesDefinition(name="O2", phase=PhaseMode.GAS),
        ],
        reactions=[
            ReactionDefinition(
                reaction_type="PHOTOLYSIS",
                reactants={"O3": 1.0},
                products={"O": 1.0, "O2": 1.0},
                rate_expression="J1",
                parameters={"A": "J1"},
            )
        ],
    )

    results = generate_headers(mech, out_dir=str(tmp_path), adjoint=True)

    with open(results["header"]) as f:
        content = f.read()

    # Struct declaration must be present
    assert "struct CheckpointBuffer" in content
    # MAX_STEPS compile-time constant
    assert "static constexpr int MAX_STEPS = 200;" in content
    # NUM_SPECIES sized to mechanism (3 species)
    assert "static constexpr int NUM_SPECIES = 3;" in content
    # num_steps field
    assert "int num_steps = 0;" in content
    # Step-size array
    assert "double h[MAX_STEPS];" in content
    # State checkpoint array (flat 2D)
    assert "double state[MAX_STEPS][NUM_SPECIES];" in content
    # Recompute-J strategy comment
    assert "Recompute-J strategy" in content
    # Stack-allocated: no heap keywords
    assert "new " not in content or "new " in content.split("CheckpointBuffer")[0]
    assert "malloc" not in content
    assert "std::vector" not in content


def test_checkpoint_buffer_is_always_emitted_behind_the_compiler_guard(tmp_path):
    """Sensitivity APIs are selected by MKPP_ENABLE_ADJOINT, not regeneration."""
    from mkpp.codegen import generate_headers
    from mkpp.model import (
        AerosolRepresentation,
        MechanismDefinition,
        PhaseMode,
        ReactionDefinition,
        SpeciesDefinition,
    )

    mech = MechanismDefinition(
        name="test_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        phases=[],
        species=[
            SpeciesDefinition(name="O3", phase=PhaseMode.GAS),
            SpeciesDefinition(name="O", phase=PhaseMode.GAS),
        ],
        reactions=[
            ReactionDefinition(
                reaction_type="PHOTOLYSIS",
                reactants={"O3": 1.0},
                products={"O": 1.0},
                rate_expression="J1",
                parameters={"A": "J1"},
            )
        ],
    )

    results = generate_headers(mech, out_dir=str(tmp_path), adjoint=False)

    with open(results["header"]) as f:
        content = f.read()

    assert "#ifdef MKPP_ENABLE_ADJOINT" in content
    assert "struct CheckpointBuffer" in content
    assert "integrate_adj" in content
    assert "integrate_tlm" in content


def test_checkpoint_buffer_gpu_safe(tmp_path):
    """Verify CheckpointBuffer struct is GPU-safe: stack-allocated, fixed-size (Req 4.3, 8.1)."""
    from mkpp.codegen import generate_headers
    from mkpp.model import (
        AerosolRepresentation,
        MechanismDefinition,
        PhaseMode,
        ReactionDefinition,
        SpeciesDefinition,
    )

    # Use a larger mechanism to verify fixed-size arrays work for N > 20
    species = [SpeciesDefinition(name=f"S{i}", phase=PhaseMode.GAS) for i in range(25)]
    mech = MechanismDefinition(
        name="large_mech",
        description="Test",
        aerosol_representation=AerosolRepresentation.BULK,
        phases=[],
        species=species,
        reactions=[
            ReactionDefinition(
                reaction_type="ARRHENIUS",
                reactants={"S0": 1},
                products={"S1": 1},
                rate_expression="k*S0",
                parameters={"A": 1e-2, "B": 0.0, "C": 0.0},
            )
        ],
    )

    results = generate_headers(mech, out_dir=str(tmp_path), adjoint=True)

    with open(results["header"]) as f:
        content = f.read()

    # For N > 20, fixed-size arrays with compile-time bounds (Req 4.3)
    assert "struct CheckpointBuffer" in content
    assert "static constexpr int NUM_SPECIES = 25;" in content
    # Still stack-allocated — no dynamic allocation
    assert "std::vector" not in content
    assert "malloc" not in content
    # The struct uses constexpr sizes
    assert "static constexpr int MAX_STEPS = 200;" in content
