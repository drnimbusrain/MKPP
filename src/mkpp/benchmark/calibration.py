"""Deterministic equal-accuracy solver configuration selection."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

TOLERANCE_MULTIPLIERS = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class CalibrationCandidate:
    configuration_id: str
    configuration_sha256: str
    tolerance_multiplier: float
    paired_block_times_ms: tuple[float, ...]
    maximum_error: float
    wrms: float
    rejected_steps: int | None
    passed_all_gates: bool


@dataclass(frozen=True, slots=True)
class CalibrationContext:
    calibration_scenario_id: str
    calibration_seed: int
    measurement_scenario_id: str
    measurement_seed: int


@dataclass(frozen=True, slots=True)
class SelectedConfiguration:
    configuration_id: str
    configuration_sha256: str
    tolerance_multiplier: float
    median_time_ms: float
    maximum_error: float
    wrms: float
    rejected_steps: int | None


@dataclass(frozen=True, slots=True)
class FrozenConfiguration:
    configuration_id: str
    configuration_sha256: str
    tolerance_multiplier: float
    frozen: bool = True


def _validate_context(context: CalibrationContext) -> None:
    if context.calibration_scenario_id == context.measurement_scenario_id or context.calibration_seed == context.measurement_seed:
        raise ValueError("calibration and measurement scenarios and seeds must be disjoint")


def _validate_candidate(candidate: CalibrationCandidate) -> None:
    if candidate.tolerance_multiplier not in TOLERANCE_MULTIPLIERS:
        raise ValueError("candidate tolerance multiplier is outside the predeclared grid")
    if len(candidate.paired_block_times_ms) != 5:
        raise ValueError("calibration candidate must contain five paired blocks")
    numeric_values = (
        *candidate.paired_block_times_ms,
        candidate.maximum_error,
        candidate.wrms,
    )
    if any(not math.isfinite(value) for value in numeric_values):
        raise ValueError("calibration candidate metrics must be finite")
    if any(value <= 0.0 for value in candidate.paired_block_times_ms):
        raise ValueError("calibration block times must be positive")
    if candidate.maximum_error < 0 or candidate.wrms < 0:
        raise ValueError("calibration error metrics cannot be negative")
    if candidate.rejected_steps is not None and candidate.rejected_steps < 0:
        raise ValueError("rejected step count cannot be negative")


def select_equal_accuracy_configuration(
    candidates: list[CalibrationCandidate] | tuple[CalibrationCandidate, ...],
    context: CalibrationContext,
) -> SelectedConfiguration:
    """Select the fastest eligible candidate using the frozen tie-break policy."""

    _validate_context(context)
    if len({candidate.configuration_id for candidate in candidates}) != len(candidates):
        raise ValueError("calibration configuration IDs must be unique")
    for candidate in candidates:
        _validate_candidate(candidate)
    eligible = [
        candidate
        for candidate in candidates
        if candidate.passed_all_gates and candidate.maximum_error <= 5.0 and candidate.wrms <= 1.0
    ]
    if not eligible:
        raise ValueError("no eligible equal-accuracy calibration candidate")

    medians = {candidate.configuration_id: statistics.median(candidate.paired_block_times_ms) for candidate in eligible}
    fastest_median = min(medians.values())
    tied = [candidate for candidate in eligible if medians[candidate.configuration_id] <= fastest_median * 1.02]
    selected = min(
        tied,
        key=lambda candidate: (
            candidate.maximum_error,
            candidate.wrms,
            candidate.rejected_steps is None,
            candidate.rejected_steps if candidate.rejected_steps is not None else math.inf,
            candidate.configuration_id,
        ),
    )
    return SelectedConfiguration(
        configuration_id=selected.configuration_id,
        configuration_sha256=selected.configuration_sha256,
        tolerance_multiplier=selected.tolerance_multiplier,
        median_time_ms=medians[selected.configuration_id],
        maximum_error=selected.maximum_error,
        wrms=selected.wrms,
        rejected_steps=selected.rejected_steps,
    )


def freeze_selected_configuration(selected: SelectedConfiguration) -> FrozenConfiguration:
    """Create immutable hash-bound evidence before measurement begins."""

    return FrozenConfiguration(
        configuration_id=selected.configuration_id,
        configuration_sha256=selected.configuration_sha256,
        tolerance_multiplier=selected.tolerance_multiplier,
    )
