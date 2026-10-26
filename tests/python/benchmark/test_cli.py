"""The command line lists and validates inputs but never fabricates evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from mkpp.benchmark import cli
from mkpp.benchmark.cli import EXIT_ENVIRONMENT, main


def test_list_mechanisms_includes_the_ts1_two_way_scope(capsys: object) -> None:
    assert main(["--list-mechanisms"]) == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert '"mechanism_id": "ts1"' in output
    assert '"micm"' in output and '"mkpp"' in output


def test_campaign_validation_resolves_all_immutable_inputs(capsys: object) -> None:
    assert main(["--validate", "--campaign", "benchmarks/solver_comparison/campaigns/cpu-primary-v1.yaml"]) == 0
    assert '"campaign_id": "cpu-primary-v1"' in capsys.readouterr().out  # type: ignore[attr-defined]


def test_campaign_execution_fails_without_silent_or_partial_output(tmp_path: Path, capsys: object) -> None:
    output = tmp_path / "raw.jsonl"
    assert (
        main(["--campaign", "benchmarks/solver_comparison/campaigns/cpu-primary-v1.yaml", "--output", str(output)])
        == EXIT_ENVIRONMENT
    )
    assert not output.exists()
    assert "runner set is not exact" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_campaign_execution_retains_only_real_named_runner_results(tmp_path: Path, monkeypatch) -> None:
    campaign = tmp_path / "campaign.yaml"
    campaign.write_text(
        "\n".join(
            (
                "campaign_id: test-ts1",
                "targets:",
                "  - manifest: benchmarks/solver_comparison/manifests/ts1-v1.yaml",
                "    scenarios:",
                "      - benchmarks/solver_comparison/scenarios/ts1-calibration-v1.yaml",
                "      - benchmarks/solver_comparison/scenarios/ts1-measurement-v1.yaml",
                "resources:",
                "  primary: {physical_cores: 1, threads: 1}",
                "",
            )
        ),
        encoding="utf-8",
    )
    runners = []
    for solver in ("micm", "mkpp"):
        executable = tmp_path / solver
        executable.write_text("#!/bin/sh\n", encoding="utf-8")
        executable.chmod(0o755)
        runners.append(f"{solver}={executable}")

    def completed(command, **kwargs):
        solver = Path(command[0]).name
        identity = kwargs["env"] if solver == "micm" else dict(zip(command[1::2], command[2::2]))

        def supplied(env_name, flag):
            return identity[env_name if solver == "micm" else flag]

        measurement = cli._read_yaml(cli._ROOT / "benchmarks/solver_comparison/scenarios/ts1-measurement-v1.yaml")
        sha = "0" * 64
        result = {
            "schema_version": 1,
            "campaign_id": "test-ts1",
            "run_id": solver,
            "block": 0,
            "order_position": 0,
            "solver": {"id": solver, "version": "1", "revision": "pinned", "backend": "cpu", "method": "ros3"},
            "manifest": {"id": "ts1", "version": "1", "sha256": sha},
            "scenario": {"id": "ts1-measurement-v1", "version": "1", "sha256": sha},
            "configuration": {"id": "frozen", "sha256": sha, "frozen": True, "controls": {}},
            "build": {"binary_sha256": sha, "asset_sha256": [sha], "compiler": "c++", "flags": [], "precision": "binary64"},
            "resources": {"hardware_id": "test", "physical_cores": 1, "threads": 1, "affinity": "core", "numa_policy": "local"},
            "timing": {
                "boundary": "steady_state_solve",
                "elapsed_ms": 1.0,
                "cell_steps": 1,
                "clock": "steady",
                "synchronized": True,
                "synchronization": "fence",
                "lifecycle_ms": {},
            },
            "state": {"checkpoints": [{"time_seconds": 1.0, "values": {"O3": 1.0}}]},
            "work": {"rhs": {"available": True, "value": 1}},
            "status": "success",
            "diagnostics": [],
        }
        result.update(
            run_id=supplied("RUN_ID", "--run-id"),
            block=int(supplied("BLOCK_ID", "--block")),
            order_position=int(supplied("ORDER_POSITION", "--order-position")),
        )
        for name in ("manifest", "scenario"):
            result[name].update(
                version=supplied(name.upper() + "_VERSION", "--" + name + "-version"),
                sha256=supplied(name.upper() + "_SHA256", "--" + name + "-sha256"),
            )
        result["state"]["checkpoints"] = [
            {"time_seconds": time, "values": {name: value["value"] for name, value in measurement["initial_state"].items()}}
            for time in measurement["integration"]["checkpoints_seconds"]
        ]
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "")

    monkeypatch.setattr(cli.subprocess, "run", completed)
    output = tmp_path / "raw.jsonl"
    report = tmp_path / "summary.md"
    arguments = ["--campaign", str(campaign), "--output", str(output), "--report", str(report)]
    assert main([*arguments, *sum((["--runner", item] for item in runners), [])]) == 0
    assert [json.loads(line)["solver"]["id"] for line in output.read_text().splitlines()] == ["micm", "mkpp"]
    assert "raw_evidence_retained" in report.read_text(encoding="utf-8")


def test_offline_audit_reports_unadmitted_inputs_without_invoking_solvers(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline audit must not launch native processes")

    monkeypatch.setattr(cli.subprocess, "run", forbidden)
    assert main(["--audit", "--campaign", "benchmarks/solver_comparison/campaigns/cpu-primary-v1.yaml"]) == cli.EXIT_SCIENTIFIC
    report = json.loads(capsys.readouterr().out)
    assert report["publishable"] is False
    assert len(report["targets"]) == 2
    assert any("unversioned" in issue for target in report["targets"] for issue in target["issues"])


def test_failed_process_transcript_is_retained_before_error(tmp_path, monkeypatch):
    def failed(command, **kwargs):
        return subprocess.CompletedProcess(command, 7, "partial output", "native integration failed")

    monkeypatch.setattr(cli.subprocess, "run", failed)
    transcript = tmp_path / "transcript.jsonl"
    import pytest

    with pytest.raises(cli.CliError, match="native integration failed"):
        cli._invoke_campaign_runner(
            "micm",
            tmp_path / "micm",
            mechanism="chapman",
            campaign_id="test",
            manifest={"version": "1"},
            scenario={"scenario_id": "test", "version": "1"},
            block=0,
            order_position=0,
            repetitions=1,
            warmups=1,
            transcript=transcript,
        )
    record = json.loads(transcript.read_text())
    assert record["returncode"] == 7
    assert record["stdout"] == "partial output"
    assert record["stderr"] == "native integration failed"


def test_runner_timeout_is_bounded_and_retained(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        assert kwargs["timeout"] == 0.1
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(cli.subprocess, "run", timeout)
    transcript = tmp_path / "transcript.jsonl"
    import pytest

    with pytest.raises(cli.CliError, match="could not complete"):
        cli._invoke_campaign_runner(
            "micm",
            tmp_path / "micm",
            mechanism="chapman",
            campaign_id="test",
            manifest={"version": "1"},
            scenario={"scenario_id": "test", "version": "1"},
            block=0,
            order_position=0,
            repetitions=1,
            warmups=1,
            transcript=transcript,
            timeout_seconds=0.1,
        )
    assert json.loads(transcript.read_text())["status"] == "execution_failed"
