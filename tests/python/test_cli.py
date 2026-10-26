import pytest
from mkpp.cli import main


def test_cli_help(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "Compile a mechanism into Kokkos headers" in captured.out


def test_cli_compile_missing_args(capsys):
    with pytest.raises(SystemExit) as e:
        main(["compile"])
    assert e.value.code != 0
    captured = capsys.readouterr()
    assert "error: the following arguments are required: mechanism" in captured.err.lower()


def test_cli_compile_success(tmp_path, capsys, monkeypatch):
    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("""
    species: [{name: O3}]
    reactions: []
    """)
    test_env_file = tmp_path / "env.yaml"
    test_env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "build"

    # Simulate success
    monkeypatch.setattr("mkpp.cli.run_compiler", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as e:
        main(
            [
                "compile",
                str(mech_file),
                "--test-env",
                str(test_env_file),
                "--out",
                str(out_dir),
                "--emit-manifest",
            ]
        )
    assert e.value.code == 0


def test_cli_passes_si_kinetic_unit_conversion_flag(tmp_path, capsys, monkeypatch):
    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("""
    species: [{name: O3}]
    reactions: []
    """)
    test_env_file = tmp_path / "env.yaml"
    test_env_file.write_text("mpi: {gather_to_root: false}")
    observed = {}
    monkeypatch.setattr("mkpp.cli.run_compiler", lambda *args, **kwargs: observed.update(kwargs))

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "compile",
                str(mech_file),
                "--test-env",
                str(test_env_file),
                "--convert-kinetic-units-to-si",
            ]
        )

    assert exc_info.value.code == 0
    assert observed["convert_kinetic_units_to_si"] is True


def test_cli_compile_fatal_error(capsys, monkeypatch, tmp_path):
    # Simulate a deep failure in the pipeline
    import json

    def mock_load_mechanism(*args, **kwargs):
        raise ValueError("Simulated pipeline crash")

    monkeypatch.setattr("mkpp.cli.load_mechanism", mock_load_mechanism)

    # Create an actual temp environment file so the open(env_path) check passes
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")

    # We must patch os.getcwd to trick sanitize_path into allowing tmp_path absolute paths
    import os

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    with pytest.raises(SystemExit) as e:
        main(["compile", "mech.yaml", "--test-env", str(env_file), "--out", "build"])

    assert e.value.code == 1
    captured = capsys.readouterr()
    # After task 1.3, errors are reported as structured JSON on stderr
    error_data = json.loads(captured.err)
    assert error_data["stage"] == "unknown"
    assert "Simulated pipeline crash" in error_data["message"]
    assert captured.out == ""  # Ensure stdout is clean on crash


def test_cli_compile_strict_flag(capsys, monkeypatch, tmp_path):
    # Verify that the --strict flag is successfully parsed and passed down to validation
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    mech_file = tmp_path / "mech.yaml"
    # Create an empty mechanism which lacks a description, triggering strict mode failure
    mech_file.write_text("species: [{name: O3}]\nreactions: []")

    import os

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    with pytest.raises(SystemExit) as e:
        main(["compile", str(mech_file), "--test-env", str(env_file), "--out", "build", "--strict"])

    assert e.value.code == 1
    captured = capsys.readouterr()
    assert "Strict mode requires a mechanism description" in captured.err


# --- Task 1.4: Unit tests for CLI stabilization ---


def test_cli_import_succeeds():
    """Test: import of mkpp.cli succeeds without errors (Req 1.1)."""
    import mkpp.cli

    assert hasattr(mkpp.cli, "main")
    assert hasattr(mkpp.cli, "run_compiler")


def test_cli_compile_valid_input_no_exception(tmp_path, capsys, monkeypatch):
    """Test: mkpp compile with valid input completes without exception (Req 1.2)."""
    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}, {name: NO2}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "output"

    # Mock run_compiler to simulate a successful pipeline run
    monkeypatch.setattr("mkpp.cli.run_compiler", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as e:
        main(["compile", str(mech_file), "--test-env", str(env_file), "--out", str(out_dir)])
    assert e.value.code == 0


def test_cli_drgep_rejects_with_structured_error(tmp_path, capsys, monkeypatch):
    """Test: --drgep rejects immediately with structured error (Req 1.4)."""
    import json
    import os

    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    with pytest.raises(SystemExit) as e:
        main(["compile", str(mech_file), "--test-env", str(env_file), "--out", "build", "--drgep"])

    assert e.value.code == 1
    captured = capsys.readouterr()
    # Verify structured JSON error on stderr
    error_data = json.loads(captured.err)
    assert error_data["stage"] == "validation"
    assert "DRGEP" in error_data["message"]
    assert "not supported" in error_data["message"].lower()


def test_cli_compile_json_files(tmp_path, capsys, monkeypatch):
    """Test: mkpp compile seamlessly processes .json mechanism and test-env files."""
    import os

    mech_file = tmp_path / "mech.json"
    mech_file.write_text('{"description": "JSON test", "species": [{"name": "O3"}], "reactions": []}')
    env_file = tmp_path / "env.json"
    env_file.write_text('{"environment": {"temperature": 298.15}, "initial_concentrations": {"O3": 1e-6}}')

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    with pytest.raises(SystemExit) as e:
        main(["compile", str(mech_file), "--test-env", str(env_file), "--out", str(tmp_path / "out"), "--dry-run"])

    assert e.value.code == 0
    captured = capsys.readouterr()
    assert "[dry-run] Validation passed" in captured.err


def test_cli_dry_run_no_output_files(tmp_path, capsys, monkeypatch):
    """Test: --dry-run does not produce output files (Req 1.5)."""
    import os
    from types import SimpleNamespace

    import mkpp.validation

    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}, {name: NO2}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "output"

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    # Mock load_mechanism to return a dummy mechanism
    dummy_mech = SimpleNamespace(species=[{"name": "O3"}, {"name": "NO2"}], reactions=[])
    monkeypatch.setattr("mkpp.cli.load_mechanism", lambda *args, **kwargs: dummy_mech)

    # Mock validation functions to pass
    monkeypatch.setattr("mkpp.cli.validate_mechanism", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.validate_mpi_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_fuzzer_stiffness", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_terminator_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_mass_conservation", lambda *args, **kwargs: None)

    # Track whether generate_headers is called (it should NOT be for --dry-run)
    generate_called = []
    monkeypatch.setattr("mkpp.cli.generate_headers", lambda *args, **kwargs: generate_called.append(True))

    with pytest.raises(SystemExit) as e:
        main(
            [
                "compile",
                str(mech_file),
                "--test-env",
                str(env_file),
                "--out",
                str(out_dir),
                "--dry-run",
            ]
        )

    assert e.value.code == 0
    # generate_headers must not have been called
    assert generate_called == []
    # Output directory should not have been created
    assert not out_dir.exists()
    # stderr should contain dry-run message
    captured = capsys.readouterr()
    assert "[dry-run]" in captured.err


def test_cli_verbose_emits_stage_markers(tmp_path, capsys, monkeypatch):
    """Test: --verbose emits stage markers to stderr (Req 1.6)."""
    import os
    from types import SimpleNamespace

    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "output"

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    # Mock pipeline internals
    dummy_mech = SimpleNamespace(
        species=[{"name": "O3"}],
        reactions=[],
        partition_metadata=None,
        sympy_metadata=None,
    )
    monkeypatch.setattr("mkpp.cli.load_mechanism", lambda *args, **kwargs: dummy_mech)
    monkeypatch.setattr("mkpp.cli.validate_mechanism", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.validate_mpi_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.partition_reactions", lambda *args, **kwargs: {"metadata": None})
    monkeypatch.setattr("mkpp.cli.generate_headers", lambda *args, **kwargs: None)

    # Mock lowering functions
    monkeypatch.setattr("mkpp.lowering.prepare_adjoint_and_tlm", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.lowering.prepare_unified_jacobian", lambda *args, **kwargs: None)

    # Mock validation sub-functions imported inside run_compiler
    import mkpp.validation

    monkeypatch.setattr(mkpp.validation, "validate_fuzzer_stiffness", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_terminator_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_mass_conservation", lambda *args, **kwargs: None)

    with pytest.raises(SystemExit) as e:
        main(
            [
                "compile",
                str(mech_file),
                "--test-env",
                str(env_file),
                "--out",
                str(out_dir),
                "--verbose",
            ]
        )

    assert e.value.code == 0
    captured = capsys.readouterr()
    # Verify stage markers appear in stderr
    assert "[parsing]" in captured.err
    assert "[validation]" in captured.err
    assert "[partitioning]" in captured.err
    assert "[lowering]" in captured.err
    assert "[codegen]" in captured.err


def test_cli_adjoint_flag_threads_to_generate_headers(tmp_path, capsys, monkeypatch):
    """Test: --adjoint flag is parsed and passed to generate_headers (Req 6.1)."""
    import os
    from types import SimpleNamespace

    import mkpp.validation

    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "output"

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    dummy_mech = SimpleNamespace(
        species=[{"name": "O3"}],
        reactions=[],
        partition_metadata=None,
        sympy_metadata=None,
    )
    monkeypatch.setattr("mkpp.cli.load_mechanism", lambda *args, **kwargs: dummy_mech)
    monkeypatch.setattr("mkpp.cli.validate_mechanism", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.validate_mpi_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.partition_reactions", lambda *args, **kwargs: {"metadata": None})
    monkeypatch.setattr("mkpp.lowering.prepare_adjoint_and_tlm", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.lowering.prepare_unified_jacobian", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_fuzzer_stiffness", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_terminator_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_mass_conservation", lambda *args, **kwargs: None)

    # Capture the kwargs passed to generate_headers
    captured_kwargs = []

    def mock_generate_headers(*args, **kwargs):
        captured_kwargs.append(kwargs)

    monkeypatch.setattr("mkpp.cli.generate_headers", mock_generate_headers)

    with pytest.raises(SystemExit) as e:
        main(
            [
                "compile",
                str(mech_file),
                "--test-env",
                str(env_file),
                "--out",
                str(out_dir),
                "--adjoint",
            ]
        )

    assert e.value.code == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["adjoint"] is True


def test_cli_adjoint_flag_defaults_to_false(tmp_path, capsys, monkeypatch):
    """Test: without --adjoint flag, adjoint defaults to False (Req 6.1)."""
    import os
    from types import SimpleNamespace

    import mkpp.validation

    mech_file = tmp_path / "mech.yaml"
    mech_file.write_text("species: [{name: O3}]\nreactions: []")
    env_file = tmp_path / "env.yaml"
    env_file.write_text("mpi: {gather_to_root: false}")
    out_dir = tmp_path / "output"

    monkeypatch.setattr(os, "getcwd", lambda: "/")

    dummy_mech = SimpleNamespace(
        species=[{"name": "O3"}],
        reactions=[],
        partition_metadata=None,
        sympy_metadata=None,
    )
    monkeypatch.setattr("mkpp.cli.load_mechanism", lambda *args, **kwargs: dummy_mech)
    monkeypatch.setattr("mkpp.cli.validate_mechanism", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.validate_mpi_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.cli.partition_reactions", lambda *args, **kwargs: {"metadata": None})
    monkeypatch.setattr("mkpp.lowering.prepare_adjoint_and_tlm", lambda *args, **kwargs: None)
    monkeypatch.setattr("mkpp.lowering.prepare_unified_jacobian", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_fuzzer_stiffness", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_terminator_safety", lambda *args, **kwargs: None)
    monkeypatch.setattr(mkpp.validation, "validate_mass_conservation", lambda *args, **kwargs: None)

    captured_kwargs = []

    def mock_generate_headers(*args, **kwargs):
        captured_kwargs.append(kwargs)

    monkeypatch.setattr("mkpp.cli.generate_headers", mock_generate_headers)

    with pytest.raises(SystemExit) as e:
        main(["compile", str(mech_file), "--test-env", str(env_file), "--out", str(out_dir)])

    assert e.value.code == 0
    assert len(captured_kwargs) == 1
    assert captured_kwargs[0]["adjoint"] is False
