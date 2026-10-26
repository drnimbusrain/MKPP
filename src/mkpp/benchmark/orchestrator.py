"""Fail-fast orchestration helpers for isolated native solver runners."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mkpp.benchmark.accuracy import TrajectoryAccuracyResult
from mkpp.benchmark.adapters.base import AdapterProtocol, PreparedAdapterRun
from mkpp.benchmark.admission import ChemistryAssessment
from mkpp.benchmark.calibration import FrozenConfiguration
from mkpp.benchmark.models import CampaignStatus, RunStatus, SolverId, SolverRun
from mkpp.benchmark.reference import ReferenceAttemptsResult
from mkpp.benchmark.report import validate_ts1_report
from mkpp.benchmark.schema import SchemaValidationError, validate_runner_result
from mkpp.benchmark.statistics import assess_repeatability


class NativeRunnerError(RuntimeError):
    """A retained, actionable native-runner protocol failure."""

    def __init__(self, solver_id: SolverId, diagnostic: str, *, stderr: str = "") -> None:
        safe_stderr = stderr.strip()
        suffix = f"; stderr: {safe_stderr}" if safe_stderr else ""
        super().__init__(f"FATAL ERROR: {solver_id.value} native runner {diagnostic}{suffix}")
        self.solver_id = solver_id
        self.diagnostic = diagnostic
        self.stderr = safe_stderr


@dataclass(frozen=True, slots=True)
class ChapmanFlowResult:
    """Retained outcome of the ordered Chapman scientific gates."""

    eligible: bool
    eligible_runs: tuple[SolverRun, ...]
    reasons: tuple[str, ...]

    @property
    def speedup_allowed(self) -> bool:
        return self.eligible


@dataclass(frozen=True, slots=True)
class Ts1FlowResult:
    """Mandatory two-solver TS1 result, including explicit ineligibility."""

    scientific: ChapmanFlowResult
    report: Mapping[str, Any]


_CAMPAIGN_TRANSITIONS = {
    CampaignStatus.CONFIGURED: CampaignStatus.PREFLIGHT_PASSED,
    CampaignStatus.PREFLIGHT_PASSED: CampaignStatus.ADMITTED,
    CampaignStatus.ADMITTED: CampaignStatus.REFERENCE_QUALIFIED,
    CampaignStatus.REFERENCE_QUALIFIED: CampaignStatus.CALIBRATED,
    CampaignStatus.CALIBRATED: CampaignStatus.MEASURING,
    CampaignStatus.MEASURING: CampaignStatus.ASSESSED,
    CampaignStatus.ASSESSED: CampaignStatus.REVIEWED,
    CampaignStatus.REVIEWED: CampaignStatus.PUBLISHED,
}


def advance_campaign_status(current: CampaignStatus, target: CampaignStatus | None = None) -> CampaignStatus:
    """Advance one constitutional campaign gate; skipped gates are invalid evidence."""

    expected = _CAMPAIGN_TRANSITIONS.get(current)
    if expected is None:
        raise ValueError(f"campaign cannot transition from terminal status {current.value}")
    if target is not None and target is not expected:
        raise ValueError(f"campaign cannot transition from {current.value} to {target.value}")
    return expected


def require_comparable_pairs(
    baseline_runs: tuple[SolverRun, ...],
    candidate_runs: tuple[SolverRun, ...],
    *,
    primary_cpu: bool = False,
    minimum_blocks: int = 1,
) -> tuple[int, ...]:
    """Validate the immutable matching evidence required for a paired ratio."""

    if not baseline_runs or not candidate_runs:
        raise ValueError("paired comparison requires retained runs from both solvers")
    if minimum_blocks < 1:
        raise ValueError("minimum_blocks must be positive")
    for runs in (baseline_runs, candidate_runs):
        if len({run.block_id for run in runs}) != len(runs):
            raise ValueError("duplicate block IDs cannot enter a paired ratio")
        if len({run.run_id for run in runs}) != len(runs):
            raise ValueError("duplicate run IDs cannot enter a paired ratio")
        first = runs[0]
        for run in runs:
            if run.solver_id is not first.solver_id:
                raise ValueError("each timing series must identify a single solver")
            if (
                run.campaign_id,
                run.manifest,
                run.scenario,
                run.configuration,
                run.build,
                run.resources,
                run.timing_boundary,
                run.common_work_count,
                run.repetition,
                run.matching_key_sha256,
            ) != (
                first.campaign_id,
                first.manifest,
                first.scenario,
                first.configuration,
                first.build,
                first.resources,
                first.timing_boundary,
                first.common_work_count,
                first.repetition,
                first.matching_key_sha256,
            ):
                raise ValueError(
                    "timing series changed frozen evidence (matching key, resources, timing boundary, "
                    "configuration, build, or workload) between blocks"
                )
            if (
                not math.isfinite(run.elapsed_seconds)
                or run.elapsed_seconds <= 0
                or run.common_work_count <= 0
                or not run.synchronized
                or run.exclusion_reason is not None
            ):
                raise ValueError("invalid timing, work, synchronization, or exclusion evidence")
    if baseline_runs[0].solver_id is candidate_runs[0].solver_id:
        raise ValueError("paired comparison requires two distinct solvers")
    if {run.run_id for run in baseline_runs} & {run.run_id for run in candidate_runs}:
        raise ValueError("paired solver series must have disjoint run IDs")
    baseline_by_block = {run.block_id: run for run in baseline_runs}
    candidate_by_block = {run.block_id: run for run in candidate_runs}
    if set(baseline_by_block) != set(candidate_by_block):
        raise ValueError("paired comparison requires identical retained block IDs")
    if len(baseline_by_block) < minimum_blocks:
        raise ValueError(f"paired comparison requires at least {minimum_blocks} valid blocks")
    for block_id in sorted(baseline_by_block):
        baseline = baseline_by_block[block_id]
        candidate = candidate_by_block[block_id]
        if baseline.order_position == candidate.order_position:
            raise ValueError("paired solvers cannot occupy the same order position")
        if baseline.status is not RunStatus.SUCCESS or candidate.status is not RunStatus.SUCCESS:
            raise ValueError("failed or invalid runs cannot enter a paired ratio")
        if baseline.campaign_id != candidate.campaign_id:
            raise ValueError("paired runs have different campaign identity")
        if baseline.manifest != candidate.manifest or baseline.scenario != candidate.scenario:
            raise ValueError("paired runs have different manifest or scenario identity")
        if baseline.build.precision != candidate.build.precision:
            raise ValueError("paired runs have different precision")
        if baseline.matching_key_sha256 != candidate.matching_key_sha256:
            raise ValueError("paired runs have different matching key")
        if baseline.resources != candidate.resources:
            raise ValueError("paired runs have different resources")
        if baseline.timing_boundary is not candidate.timing_boundary:
            raise ValueError("paired runs have different timing boundary")
        if baseline.common_work_count != candidate.common_work_count:
            raise ValueError("paired runs have different common work count")
        if primary_cpu and (baseline.resources.physical_cores != 1 or baseline.resources.threads != 1):
            raise ValueError("primary CPU comparisons require one physical core and one thread")
        if primary_cpu:
            backend = baseline.resources.backend.lower().replace("_", "-")
            if backend not in {"cpu", "serial-cpu", "serial", "host", "kokkos-serial", "openmp", "kokkos-openmp"}:
                raise ValueError("primary CPU comparison requires a verified CPU backend")
            if baseline.resources.affinity in {"unspecified", "external-policy", "none"}:
                raise ValueError("primary CPU comparison requires observed physical-core affinity")
    return tuple(sorted(baseline_by_block))


def require_requested_solvers(
    requested_solvers: tuple[SolverId, ...],
    adapters: Mapping[SolverId, AdapterProtocol],
) -> tuple[AdapterProtocol, ...]:
    """Resolve every requested solver or stop before any campaign work begins."""

    missing_solvers = tuple(solver for solver in requested_solvers if solver not in adapters)
    if missing_solvers:
        names = ", ".join(solver.value for solver in missing_solvers)
        raise RuntimeError(f"FATAL ERROR: missing requested solver adapters: {names}")
    return tuple(adapters[solver] for solver in requested_solvers)


def assess_chapman_flow(
    *,
    preflight_passed: bool,
    requested_solvers: tuple[SolverId, ...],
    adapters: Mapping[SolverId, AdapterProtocol],
    chemistry: Mapping[SolverId, ChemistryAssessment],
    reference: ReferenceAttemptsResult,
    frozen_configurations: Mapping[SolverId, FrozenConfiguration],
    trajectory_accuracy: Mapping[SolverId, TrajectoryAccuracyResult],
    runs: Mapping[SolverId, SolverRun],
) -> ChapmanFlowResult:
    """Apply Chapman gates in constitutional order and withhold unsafe ratios."""

    reasons: list[str] = []
    if not preflight_passed:
        return ChapmanFlowResult(False, (), ("preflight failed",))

    # This raises before admission if any requested subject is unavailable.
    require_requested_solvers(requested_solvers, adapters)
    for solver in requested_solvers:
        assessment = chemistry.get(solver)
        if assessment is None:
            reasons.append(f"{solver.value}: missing chemistry admission evidence")
        elif not assessment.eligible:
            reasons.extend(f"{solver.value}: {reason}" for reason in assessment.reasons)
    if reasons:
        return ChapmanFlowResult(False, (), tuple(reasons))

    if not reference.qualified:
        return ChapmanFlowResult(False, (), ("independent reference is unqualified",))

    eligible_runs: list[SolverRun] = []
    for solver in requested_solvers:
        frozen = frozen_configurations.get(solver)
        accuracy = trajectory_accuracy.get(solver)
        run = runs.get(solver)
        if frozen is None:
            reasons.append(f"{solver.value}: no frozen calibration configuration")
            continue
        if accuracy is None or not accuracy.eligible:
            reasons.append(f"{solver.value}: trajectory accuracy is ineligible")
            continue
        if run is None or run.status is not RunStatus.SUCCESS:
            reasons.append(f"{solver.value}: no successful retained native run")
            continue
        if run.configuration.sha256 != frozen.configuration_sha256:
            reasons.append(f"{solver.value}: measurement configuration differs from calibration")
            continue
        eligible_runs.append(run)

    eligible = not reasons and len(eligible_runs) == len(requested_solvers)
    return ChapmanFlowResult(eligible, tuple(eligible_runs) if eligible else (), tuple(reasons))


def assess_ts1_flow(
    *,
    requested_solvers: tuple[SolverId, ...],
    **flow_evidence: Any,
) -> Ts1FlowResult:
    """Enforce the mandatory MICM/MKPP TS1 scope and derive no KPP placeholder."""

    required = {SolverId.MICM, SolverId.MKPP}
    if set(requested_solvers) != required or len(requested_solvers) != 2:
        raise ValueError("TS1 requires exactly MICM and MKPP; KPP is forbidden until admitted")
    scientific = assess_chapman_flow(
        requested_solvers=requested_solvers,
        **flow_evidence,
    )
    results = [
        {
            "solver_id": solver.value,
            "status": "eligible" if scientific.eligible else "ineligible",
            "throughput": None,
        }
        for solver in requested_solvers
    ]
    comparisons = [
        {
            "baseline_solver_id": SolverId.MICM.value,
            "candidate_solver_id": SolverId.MKPP.value,
            "status": "eligible" if scientific.eligible else "withheld",
            "median_ratio": None,
        }
    ]
    report: Mapping[str, Any] = {
        "mechanism_id": "ts1",
        "required_solvers": [solver.value for solver in requested_solvers],
        "results": results,
        "comparisons": comparisons,
        "claims": ["TS1 MICM/MKPP two-way comparison"],
        "diagnostics": list(scientific.reasons),
    }
    validate_ts1_report(report)
    return Ts1FlowResult(scientific=scientific, report=report)


def invoke_json_runner(
    prepared: PreparedAdapterRun,
    solver_id: SolverId,
    *,
    timeout_seconds: float,
) -> Mapping[str, Any]:
    """Invoke one prepared runner and return its validated single JSON result.

    Process startup belongs to adapter/lifecycle evidence, never the primary solve
    timing. Runner stdout is reserved for exactly one JSON document; stderr is
    retained only as a diagnostic and cannot replace structured evidence.
    """

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if not prepared.command:
        raise NativeRunnerError(solver_id, "has an empty command")

    environment = os.environ.copy()
    environment.update(dict(prepared.environment))
    try:
        completed = subprocess.run(
            prepared.command,
            cwd=prepared.working_directory,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise NativeRunnerError(solver_id, f"timed out after {timeout_seconds:g} seconds") from exc
    except OSError as exc:
        raise NativeRunnerError(solver_id, f"could not start: {exc}") from exc

    if completed.returncode != 0:
        raise NativeRunnerError(
            solver_id,
            f"exited with code {completed.returncode}",
            stderr=completed.stderr,
        )

    output = completed.stdout.strip()
    if not output:
        raise NativeRunnerError(solver_id, "returned no JSON result", stderr=completed.stderr)
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        raise NativeRunnerError(
            solver_id,
            "did not return exactly one valid JSON document",
            stderr=completed.stderr,
        ) from exc
    if not isinstance(result, dict):
        raise NativeRunnerError(solver_id, "JSON result must be an object", stderr=completed.stderr)
    try:
        validate_runner_result(result)
    except SchemaValidationError as exc:
        raise NativeRunnerError(
            solver_id,
            f"returned schema-invalid JSON: {exc}",
            stderr=completed.stderr,
        ) from exc

    observed_solver = result["solver"]["id"]
    if observed_solver != solver_id.value:
        raise NativeRunnerError(
            solver_id,
            f"identity mismatch: runner reported '{observed_solver}'",
            stderr=completed.stderr,
        )
    return result


__all__ = [
    "ChapmanFlowResult",
    "NativeRunnerError",
    "Ts1FlowResult",
    "advance_campaign_status",
    "assess_chapman_flow",
    "assess_ts1_flow",
    "invoke_json_runner",
    "require_requested_solvers",
    "require_comparable_pairs",
    "assess_repeatability",
]
