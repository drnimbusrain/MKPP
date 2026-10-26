"""Eligibility-preserving report validation and derivation."""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mkpp.benchmark.models import SolverRun
from mkpp.benchmark.statistics import paired_ratio_summary

_TS1_SOLVERS = {"micm", "mkpp"}


def _freeze(value: Any) -> Any:
    """Recursively freeze imported evidence so reports cannot mutate it."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def read_raw_results(path: Path) -> tuple[Mapping[str, Any], ...]:
    """Read append-only JSONL evidence without rewriting or filtering records."""

    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"raw result line {line_number} is not valid JSON") from error
        if not isinstance(record, Mapping):
            raise ValueError(f"raw result line {line_number} must be an object")
        records.append(_freeze(record))
    return tuple(records)


def derive_pairwise_comparison(
    *,
    baseline_solver_id: str,
    candidate_solver_id: str,
    baseline_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
    eligible: bool,
    seed: int,
    withheld_reason: str | None = None,
) -> dict[str, Any]:
    """Derive an explicit directed ratio, or retain an explicit withholding reason."""

    direction = f"{baseline_solver_id}_seconds / {candidate_solver_id}_seconds"
    result: dict[str, Any] = {
        "baseline_solver_id": baseline_solver_id,
        "candidate_solver_id": candidate_solver_id,
        "ratio_direction": direction,
    }
    if eligible and (len(baseline_seconds) < 10 or len(candidate_seconds) < 10):
        eligible = False
        withheld_reason = "at least 10 valid paired blocks are required"
    if not eligible:
        result.update(
            status="withheld",
            median_ratio=None,
            bootstrap_interval=None,
            withheld_reason=withheld_reason or "scientific eligibility failed",
        )
        return result
    summary = paired_ratio_summary(baseline_seconds, candidate_seconds, seed=seed)
    result.update(
        status="eligible",
        median_ratio=summary.median_ratio,
        bootstrap_interval=summary.bootstrap_interval,
        paired_raw_ratios=summary.raw_ratios,
    )
    return result


def derive_verified_comparison(
    baseline_runs: tuple[SolverRun, ...],
    candidate_runs: tuple[SolverRun, ...],
    *,
    scientific_eligible: bool,
    seed: int,
    primary_cpu: bool = True,
    minimum_valid_blocks: int = 10,
    minimum_sample_seconds: float = 1.0,
) -> dict[str, Any]:
    """Check retained pairing; orchestration must supply completed science gates.

    The caller passes campaign-specific floors. This helper does not establish
    native warmup stability, timer overhead, or independent scientific evidence.
    """
    # Local import avoids the orchestration/report dependency during module load.
    from mkpp.benchmark.orchestrator import require_comparable_pairs

    reason = None if scientific_eligible else "scientific eligibility failed"
    if type(minimum_valid_blocks) is not int or minimum_valid_blocks < 10:
        raise ValueError("minimum_valid_blocks must be an integer >= 10")
    if not math.isfinite(minimum_sample_seconds) or minimum_sample_seconds < 1.0:
        raise ValueError("minimum_sample_seconds must be finite and >= 1")
    try:
        blocks = require_comparable_pairs(
            baseline_runs, candidate_runs, primary_cpu=primary_cpu, minimum_blocks=minimum_valid_blocks
        )
        if any(run.elapsed_seconds < minimum_sample_seconds for run in (*baseline_runs, *candidate_runs)):
            raise ValueError("retained measurement is shorter than the campaign sample duration floor")
    except ValueError as error:
        blocks = ()
        reason = str(error)
    baseline = {run.block_id: run for run in baseline_runs}
    candidate = {run.block_id: run for run in candidate_runs}
    return derive_pairwise_comparison(
        baseline_solver_id=baseline_runs[0].solver_id.value if baseline_runs else "unknown",
        candidate_solver_id=candidate_runs[0].solver_id.value if candidate_runs else "unknown",
        baseline_seconds=tuple(baseline[block].elapsed_seconds for block in blocks),
        candidate_seconds=tuple(candidate[block].elapsed_seconds for block in blocks),
        eligible=reason is None,
        seed=seed,
        withheld_reason=reason,
    )


def render_markdown_report(document: Mapping[str, Any]) -> str:
    """Render distinct solve/lifecycle sections; never merge their timing ratios."""

    mechanism_id = str(document.get("mechanism_id", "unknown"))
    status = str(document.get("admission_status", "unknown"))
    reason = str(document.get("admission_reason", ""))
    lines = [f"# Solver comparison: {mechanism_id}", "", f"Admission: `{status}`"]
    if reason:
        lines.append(f"Admission reason: {reason}")
    for heading, key in (("Steady-state solve", "steady_state_solve"), ("Lifecycle", "lifecycle")):
        lines.extend(("", f"## {heading}", ""))
        comparisons = document.get(key, ())
        if not comparisons:
            lines.append("No comparisons collected.")
            continue
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise ValueError(f"{key} comparison must be an object")
            direction = comparison.get("ratio_direction", "ratio direction unavailable")
            if comparison.get("status") == "eligible":
                lines.append(
                    f"- `{direction}`: {comparison.get('median_ratio')}; "
                    f"95% bootstrap interval: {comparison.get('bootstrap_interval', 'unavailable')}"
                )
            else:
                lines.append(f"- Withheld `{direction}`: {comparison.get('withheld_reason', 'ineligible')}")
    return "\n".join(lines) + "\n"


def render_json_report(document: Mapping[str, Any]) -> str:
    """Render a deterministic, lossless report representation."""

    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def render_csv_report(document: Mapping[str, Any]) -> str:
    """Render one comparison per row without collapsing timing boundaries."""

    columns = (
        "mechanism_id",
        "timing_boundary",
        "baseline_solver_id",
        "candidate_solver_id",
        "ratio_direction",
        "status",
        "median_ratio",
        "bootstrap_interval_low",
        "bootstrap_interval_high",
        "withheld_reason",
        "admission_status",
        "admission_reason",
    )
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(columns)
    mechanism_id = str(document.get("mechanism_id", "unknown"))
    admission_status = str(document.get("admission_status", "unknown"))
    admission_reason = str(document.get("admission_reason", ""))
    for boundary, key in (("steady_state_solve", "steady_state_solve"), ("lifecycle", "lifecycle")):
        for comparison in document.get(key, ()):
            if not isinstance(comparison, Mapping):
                raise ValueError(f"{key} comparison must be an object")
            interval = comparison.get("bootstrap_interval") or ("", "")
            if not isinstance(interval, Sequence) or len(interval) != 2:
                raise ValueError("bootstrap_interval must contain lower and upper bounds")
            values = (
                mechanism_id,
                boundary,
                str(comparison.get("baseline_solver_id", "")),
                str(comparison.get("candidate_solver_id", "")),
                str(comparison.get("ratio_direction", "")),
                str(comparison.get("status", "")),
                str(comparison.get("median_ratio", "")),
                str(interval[0]),
                str(interval[1]),
                str(comparison.get("withheld_reason", "")),
                admission_status,
                admission_reason,
            )
            writer.writerow(values)
    return output.getvalue()


def validate_ts1_report(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Reject any TS1 report that implies an unavailable KPP comparison."""

    if str(document.get("mechanism_id", "")).lower() != "ts1":
        raise ValueError("TS1 report must identify mechanism_id 'ts1'")
    required = document.get("required_solvers")
    if not isinstance(required, Sequence) or isinstance(required, str | bytes):
        raise ValueError("TS1 requires exactly MICM and MKPP")
    if set(required) != _TS1_SOLVERS or len(required) != 2:
        raise ValueError("TS1 requires exactly MICM and MKPP")

    results = document.get("results")
    if not isinstance(results, Sequence) or isinstance(results, str | bytes):
        raise ValueError("TS1 results must contain exactly MICM and MKPP")
    result_solvers = {entry.get("solver_id") for entry in results if isinstance(entry, Mapping)}
    if result_solvers != _TS1_SOLVERS or len(results) != 2:
        raise ValueError("TS1 KPP result or placeholder is forbidden; results require MICM and MKPP only")

    comparisons = document.get("comparisons", ())
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, str | bytes):
        raise ValueError("TS1 comparisons must be a sequence")
    for comparison in comparisons:
        if not isinstance(comparison, Mapping):
            raise ValueError("TS1 comparison must be an object")
        participants = {
            comparison.get("baseline_solver_id"),
            comparison.get("candidate_solver_id"),
        }
        if participants != _TS1_SOLVERS:
            raise ValueError("TS1 KPP comparison is forbidden")

    claims = document.get("claims", ())
    if not isinstance(claims, Sequence) or isinstance(claims, str | bytes):
        raise ValueError("TS1 claims must be a sequence")
    forbidden_phrases = ("three-way", "three solvers")
    for claim in claims:
        normalized = str(claim).lower()
        if any(phrase in normalized for phrase in forbidden_phrases) or re.search(r"(?<!m)\bkpp\b", normalized):
            raise ValueError("TS1 cannot make a three-way or KPP claim")
    return document


__all__ = [
    "derive_pairwise_comparison",
    "derive_verified_comparison",
    "read_raw_results",
    "render_csv_report",
    "render_json_report",
    "render_markdown_report",
    "validate_ts1_report",
]
