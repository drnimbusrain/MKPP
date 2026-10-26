"""Thin, fail-closed command line interface for solver-comparison evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from mkpp.benchmark.audit import audit_campaign_assets
from mkpp.benchmark.evidence import validate_requested_evidence
from mkpp.benchmark.manifest import validate_manifest
from mkpp.benchmark.openatmos import require_openatmos_v1
from mkpp.benchmark.policy import validate_performance_policy
from mkpp.benchmark.preflight import PreflightError, resolve_relative_asset, verify_asset_hash
from mkpp.benchmark.report import render_markdown_report
from mkpp.benchmark.sampling import append_raw_record, balanced_order, paired_repetition_count
from mkpp.benchmark.scenario import validate_scenario, validate_scenario_pair
from mkpp.benchmark.schema import SchemaValidationError

_ROOT = Path(__file__).resolve().parents[3]
_MANIFESTS = _ROOT / "benchmarks/solver_comparison/manifests"

EXIT_USAGE = 2
EXIT_ENVIRONMENT = 3
EXIT_SCIENTIFIC = 4
EXIT_RUNTIME = 5


class CliError(ValueError):
    """A user-facing error with a constitutional process exit code."""

    def __init__(self, message: str, exit_code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _progress(completed: int, total: int, label: str) -> None:
    """Write human progress to stderr without contaminating JSON evidence."""

    width = 24
    filled = width if total == 0 else round(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\r[{bar}] {completed}/{total} {label}", end="\n" if completed >= total else "", file=sys.stderr, flush=True)


def _read_yaml(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise CliError(f"path does not exist or is not a file: {path}")
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        raise CliError(f"invalid YAML at {path}: {error}") from error
    if not isinstance(document, Mapping):
        raise CliError(f"YAML document at {path} must be an object")
    return document


def _path_from_root(value: str, *, base: Path = _ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _manifest_paths() -> tuple[Path, ...]:
    return tuple(sorted(_MANIFESTS.glob("*.yaml")))


def list_mechanisms() -> tuple[dict[str, Any], ...]:
    """Return the available canonical manifests without invoking runners."""

    mechanisms: list[dict[str, Any]] = []
    for path in _manifest_paths():
        manifest = validate_manifest(_read_yaml(path))
        mechanisms.append(
            {
                "mechanism_id": manifest["mechanism_id"],
                "version": manifest["version"],
                "required_solvers": list(manifest["required_solvers"]),
                "admission_status": manifest["admission_status"],
                "manifest": path.relative_to(_ROOT).as_posix(),
            }
        )
    return tuple(mechanisms)


def validate_campaign(path: Path) -> Mapping[str, Any]:
    """Resolve and validate immutable campaign, manifest, and scenario inputs."""

    campaign = _read_yaml(path)
    if not isinstance(campaign.get("campaign_id"), str) or not campaign["campaign_id"].strip():
        raise CliError("campaign_id must be a non-empty string")
    targets = campaign.get("targets")
    if not isinstance(targets, Sequence) or isinstance(targets, str | bytes) or not targets:
        raise CliError("campaign.targets must be a non-empty sequence")
    resources = campaign.get("resources")
    if not isinstance(resources, Mapping) or not isinstance(resources.get("primary"), Mapping):
        raise CliError("campaign.resources.primary is required")
    primary = resources["primary"]
    if primary.get("backend", "cpu") != "cpu":
        raise CliError("this campaign executor requires a CPU primary resource policy")
    if any(type(primary.get(name)) is not int or primary[name] != 1 for name in ("physical_cores", "threads")):
        raise CliError("primary campaign resources require one physical core and one thread")

    resolved_targets: list[dict[str, str]] = []
    solver_counts: list[int] = []
    for index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise CliError(f"campaign.targets.{index} must be an object")
        manifest_path = _path_from_root(str(target.get("manifest", "")))
        manifest = validate_manifest(_read_yaml(manifest_path))
        scenario_values = target.get("scenarios")
        if not isinstance(scenario_values, Sequence) or isinstance(scenario_values, str | bytes) or len(scenario_values) != 2:
            raise CliError(f"campaign.targets.{index}.scenarios must name calibration and measurement scenarios")
        scenarios = tuple(_read_yaml(_path_from_root(str(value))) for value in scenario_values)
        calibration, measurement = (validate_scenario(item) for item in scenarios)
        if calibration["role"] != "calibration" or measurement["role"] != "measurement":
            raise CliError(f"campaign.targets.{index}: scenarios must be calibration then measurement")
        if calibration["scenario_id"] == measurement["scenario_id"]:
            raise CliError(f"campaign.targets.{index}: scenario identifiers must be distinct")
        if calibration["workload"]["ensemble_seed"] == measurement["workload"]["ensemble_seed"]:
            raise CliError(f"campaign.targets.{index}: calibration and measurement ensemble seeds must differ")
        if manifest["mechanism_id"].lower() == "ts1":
            validate_scenario_pair(calibration, measurement)
        if calibration["mechanism"]["id"] != manifest["mechanism_id"]:
            raise CliError(f"campaign.targets.{index}: scenario mechanism does not match manifest")
        solver_counts.append(len(manifest["required_solvers"]))
        resolved_targets.append({"mechanism_id": str(manifest["mechanism_id"]), "manifest": str(manifest_path)})
    validate_performance_policy(campaign, solver_counts)
    return {"campaign_id": campaign["campaign_id"], "targets": resolved_targets}


def _runner_map(values: Sequence[str]) -> dict[str, Path]:
    runners: dict[str, Path] = {}
    for value in values:
        solver, separator, executable = value.partition("=")
        if not separator or not solver or not executable:
            raise CliError("--runner must use solver_id=/absolute/or/relative/executable")
        if solver in runners:
            raise CliError(f"--runner supplied more than once for {solver}")
        path = Path(executable).resolve()
        if not path.is_file() or not path.stat().st_mode & 0o111:
            raise CliError(f"runner is missing or not executable: {path}", EXIT_ENVIRONMENT)
        runners[solver] = path
    return runners


def _identity_hash(document: Mapping[str, Any]) -> str:
    """Hash the exact resolved immutable input supplied to a native subject."""

    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _invoke_campaign_runner(
    solver: str,
    executable: Path,
    *,
    mechanism: str,
    campaign_id: str,
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    block: int,
    order_position: int,
    repetitions: int,
    warmups: int,
    transcript: Path | None = None,
    phase: str = "measurement",
    timeout_seconds: float = 300.0,
) -> Mapping[str, Any]:
    """Invoke one native runner with a shared frozen campaign identity."""

    configuration = {"mechanism": mechanism, "schedule": "canonical-openatmos-v1"}
    common = {
        "CAMPAIGN_ID": campaign_id,
        "RUN_ID": f"{mechanism}-{block}-{solver}",
        "BLOCK_ID": str(block),
        "ORDER_POSITION": str(order_position),
        "WARMUPS": str(warmups),
        "REPETITIONS": str(repetitions),
        "MANIFEST_VERSION": str(manifest["version"]),
        "MANIFEST_SHA256": _identity_hash(manifest),
        "SCENARIO_ID": str(scenario["scenario_id"]),
        "SCENARIO_VERSION": str(scenario["version"]),
        "SCENARIO_SHA256": _identity_hash(scenario),
        "CONFIGURATION_ID": "canonical-openatmos-v1",
        "CONFIGURATION_SHA256": _identity_hash(configuration),
    }
    command = [str(executable)]
    environment = os.environ.copy()
    if solver == "micm":
        environment.update(common)
    else:
        if solver == "mkpp":
            command += ["--mechanism", mechanism]
        command += [
            "--warmups",
            str(warmups),
            "--repetitions",
            str(repetitions),
            "--campaign-id",
            campaign_id,
            "--run-id",
            common["RUN_ID"],
            "--manifest-version",
            common["MANIFEST_VERSION"],
            "--manifest-sha256",
            common["MANIFEST_SHA256"],
            "--scenario-id",
            common["SCENARIO_ID"],
            "--scenario-version",
            common["SCENARIO_VERSION"],
            "--scenario-sha256",
            common["SCENARIO_SHA256"],
            "--configuration-id",
            common["CONFIGURATION_ID"],
            "--configuration-sha256",
            common["CONFIGURATION_SHA256"],
            "--block",
            str(block),
            "--order-position",
            str(order_position),
        ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False, env=environment, timeout=timeout_seconds)
    except (OSError, subprocess.TimeoutExpired) as error:
        if transcript is not None:
            append_raw_record(
                transcript,
                {
                    "phase": phase,
                    "solver": solver,
                    "run_id": common["RUN_ID"],
                    "status": "execution_failed",
                    "diagnostic": str(error),
                },
            )
        raise CliError(f"{solver} runner could not complete: {error}", EXIT_RUNTIME) from error
    if transcript is not None:
        append_raw_record(
            transcript,
            {
                "phase": phase,
                "solver": solver,
                "run_id": common["RUN_ID"],
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            },
        )
    if completed.returncode != 0:
        raise CliError(f"{solver} runner failed: {completed.stderr.strip()}", EXIT_RUNTIME)
    try:
        result = json.loads(completed.stdout)
        validate_requested_evidence(
            result,
            solver=solver,
            campaign_id=campaign_id,
            run_id=common["RUN_ID"],
            manifest=manifest,
            scenario=scenario,
            block=block,
            order_position=order_position,
        )
    except (json.JSONDecodeError, SchemaValidationError) as error:
        raise CliError(f"{solver} runner emitted invalid result: {error}", EXIT_RUNTIME) from error
    return result


def execute_campaign(path: Path, output: Path, runner_values: Sequence[str]) -> tuple[Mapping[str, Any], ...]:
    """Collect diagnostic native evidence, retaining failures before stopping.

    This does not execute scientific admission or produce publishable ratios.
    """

    campaign = _read_yaml(path)
    validation = validate_campaign(path)
    runners = _runner_map(runner_values)
    required = {
        solver
        for target in campaign["targets"]
        for solver in validate_manifest(_read_yaml(_path_from_root(target["manifest"])))["required_solvers"]
    }
    missing = required - set(runners)
    unexpected = set(runners) - required
    if missing or unexpected:
        detail = []
        if missing:
            detail.append(f"missing={','.join(sorted(missing))}")
        if unexpected:
            detail.append(f"unexpected={','.join(sorted(unexpected))}")
        raise CliError(f"requested native runner set is not exact ({'; '.join(detail)})", EXIT_ENVIRONMENT)

    # A MusicBox-like workload or an `openatmos` directory name is not proof
    # of the chemistry format. Reject unversioned/v0 sources before execution.
    for target in campaign["targets"]:
        manifest = _read_yaml(_path_from_root(target["manifest"]))
        for source in manifest["canonical_source"]:
            try:
                asset = resolve_relative_asset(_ROOT, source["path"])
                verify_asset_hash(asset, source["sha256"])
                if asset.suffix.lower() in {".json", ".yaml", ".yml"}:
                    require_openatmos_v1(_read_yaml(asset))
            except (PreflightError, SchemaValidationError) as error:
                raise CliError(str(error), EXIT_SCIENTIFIC) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    transcript = output.with_suffix(output.suffix + ".transcript.jsonl")
    records: list[Mapping[str, Any]] = []
    for target in campaign["targets"]:
        manifest = validate_manifest(_read_yaml(_path_from_root(target["manifest"])))
        measurement_path = _path_from_root(str(target["scenarios"][1]))
        measurement = validate_scenario(_read_yaml(measurement_path))
        sampling = campaign.get("sampling", {"paired_blocks": 1, "ordering_seed": 0, "warmups": {"minimum": 1}})
        repetitions = 1
        warmups = int(sampling["warmups"].get("maximum", sampling["warmups"]["minimum"]))
        if "sampling" in campaign:

            def pilot(solver: str, count: int) -> Sequence[float]:
                _progress(0, 0, f"{manifest['mechanism_id']} pilot {solver} repetitions={count}")
                return tuple(
                    float(
                        _invoke_campaign_runner(
                            solver,
                            runners[solver],
                            mechanism=manifest["mechanism_id"],
                            campaign_id=campaign["campaign_id"],
                            manifest=manifest,
                            scenario=measurement,
                            block=0,
                            order_position=0,
                            repetitions=count,
                            warmups=warmups,
                            transcript=transcript,
                            phase="pilot",
                        )["timing"]["elapsed_ms"]
                    )
                    / 1000.0
                    for _ in range(int(sampling["pilot_iterations"]))
                )

            repetitions = paired_repetition_count(
                lambda count: {solver: pilot(solver, count) for solver in manifest["required_solvers"]},
                minimum_seconds=float(sampling["minimum_sample_seconds"]),
            )
            # Each native process performs the declared maximum warmup count.
            # Timings from different fresh processes cannot demonstrate within-
            # process warmup stability, so do not label them as such.
            append_raw_record(
                transcript,
                {
                    "phase": "sampling_policy",
                    "mechanism": manifest["mechanism_id"],
                    "repetitions": repetitions,
                    "warmups": warmups,
                    "warmup_stability": "unverified; native per-warmup telemetry required",
                },
            )
        # These diagnostic measurements have not passed scientific admission or
        # calibration. A configuration label must never imply those gates ran.
        total_runs = int(sampling["paired_blocks"]) * len(manifest["required_solvers"])
        completed_runs = 0
        for block, order in enumerate(
            balanced_order(manifest["required_solvers"], blocks=sampling["paired_blocks"], seed=sampling["ordering_seed"])
        ):
            for order_position, solver in enumerate(order):
                result = _invoke_campaign_runner(
                    solver,
                    runners[solver],
                    mechanism=manifest["mechanism_id"],
                    campaign_id=campaign["campaign_id"],
                    manifest=manifest,
                    scenario=measurement,
                    block=block,
                    order_position=order_position,
                    repetitions=repetitions,
                    warmups=warmups,
                    transcript=transcript,
                )
                if result["solver"]["id"] != solver:
                    raise CliError(f"{solver} runner identity mismatch", EXIT_ENVIRONMENT)
                if result["manifest"]["id"] != manifest["mechanism_id"]:
                    raise CliError(f"{solver} runner manifest mismatch", EXIT_RUNTIME)
                if result["scenario"]["id"] != measurement["scenario_id"]:
                    raise CliError(f"{solver} runner scenario mismatch", EXIT_RUNTIME)
                records.append(result)
                append_raw_record(output, result)
                completed_runs += 1
                _progress(completed_runs, total_runs, f"{manifest['mechanism_id']} block {block} {solver}")
    del validation  # Validation must finish before native execution begins.
    return tuple(records)


def write_campaign_report(path: Path, report: Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Write a non-headline evidence index; ratio derivation remains separate."""

    campaign = _read_yaml(path)
    boundaries = {record["timing"]["boundary"] for record in records}
    document = {
        "mechanism_id": str(campaign["campaign_id"]),
        "admission_status": "raw_evidence_retained",
        "admission_reason": "Scientific admission and paired ratios require subsequent gate evaluation.",
        "steady_state_solve": () if "steady_state_solve" in boundaries else (),
        "lifecycle": () if "lifecycle" in boundaries else (),
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(render_markdown_report(document), encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-mechanisms", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--audit", action="store_true", help="audit declared assets; never claims scientific admission")
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--mechanism")
    parser.add_argument("--scenario", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--runner", action="append", default=[], metavar="SOLVER=PATH")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve inputs; fail closed until native campaign adapters are configured."""

    args = _parser().parse_args(argv)
    try:
        selected = sum((args.list_mechanisms, args.validate, args.audit, args.campaign is not None, args.mechanism is not None))
        if selected == 0:
            raise CliError("select --list-mechanisms, --validate --campaign, --campaign, or --mechanism")
        if args.list_mechanisms and selected != 1:
            raise CliError("--list-mechanisms cannot be combined with another command")
        if args.validate and (args.campaign is None or selected != 2):
            raise CliError("--validate requires only --campaign <path>")
        if args.audit and (args.campaign is None or selected != 2):
            raise CliError("--audit requires only --campaign <path>")
        if args.list_mechanisms:
            print(json.dumps(list_mechanisms(), indent=2, sort_keys=True))
            return 0
        if args.validate:
            if args.campaign is None:
                raise CliError("--validate requires --campaign <path>")
            print(json.dumps(validate_campaign(args.campaign), indent=2, sort_keys=True))
            return 0
        if args.audit:
            validate_campaign(args.campaign)
            print(json.dumps(audit_campaign_assets(_read_yaml(args.campaign), _ROOT), indent=2, sort_keys=True))
            return EXIT_SCIENTIFIC
        if args.mechanism is not None:
            if args.scenario is None:
                raise CliError("--mechanism requires --scenario <path>")
            scenario = validate_scenario(_read_yaml(args.scenario))
            if scenario["mechanism"]["id"] != args.mechanism:
                raise CliError("--mechanism does not match scenario mechanism.id")
            raise CliError(
                "native campaign execution is not configured; no solver result was emitted",
                EXIT_RUNTIME,
            )
        if args.output is None:
            raise CliError("--campaign execution requires --output <raw.jsonl>")
        records = execute_campaign(args.campaign, args.output, args.runner)
        if args.report is not None:
            write_campaign_report(args.campaign, args.report, records)
        print(json.dumps({"records": len(records), "raw_jsonl": str(args.output)}, sort_keys=True))
        return 0
    except CliError as error:
        print(f"FATAL ERROR: {error}", file=sys.stderr)
        return error.exit_code
    except Exception as error:
        print(f"FATAL ERROR: invalid benchmark input: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["CliError", "execute_campaign", "list_mechanisms", "main", "validate_campaign", "write_campaign_report"]
