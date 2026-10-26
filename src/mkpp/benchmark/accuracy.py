"""Species-scaled trajectory accuracy assessment for solver comparisons."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from mkpp.benchmark.models import ErrorMetrics, ScaledSpeciesError

Trajectory = Mapping[int, Mapping[float, Mapping[str, float]]]


@dataclass(frozen=True, slots=True)
class TrajectoryAccuracyResult:
    """Scientific eligibility evidence for one subject trajectory."""

    scaled_errors: tuple[ScaledSpeciesError, ...]
    aggregate_errors: ErrorMetrics
    positivity_passed: bool
    conservation_passed: bool
    finite_state_passed: bool
    solver_success: bool
    eligible: bool
    diagnostics: tuple[str, ...]


def symmetric_scaled_error(observed: float, reference: float, atol: float, rtol: float) -> float:
    """Return the signed error under the shared symmetric species scale."""

    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and strictly positive")
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and strictly positive")
    # Scale first so opposite, finite binary64 extremes cannot overflow the
    # subtraction or turn an infinite denominator into a false zero error.
    scale = max(atol, abs(observed), abs(reference))
    denominator = atol / scale + rtol * (max(abs(observed), abs(reference)) / scale)
    difference = observed / scale - reference / scale
    return difference / denominator if denominator else math.copysign(math.inf, difference)


def _validate_shape(subject: Trajectory, reference: Trajectory, atols: Mapping[str, float]) -> None:
    if set(subject) != set(reference):
        raise ValueError("subject and reference ensemble members/checkpoints must match")
    for member, reference_checkpoints in reference.items():
        if not reference_checkpoints:
            raise ValueError(f"member {member}: checkpoints must not be empty")
        subject_checkpoints = subject[member]
        if set(subject_checkpoints) != set(reference_checkpoints):
            raise ValueError(f"member {member}: subject and reference checkpoints must match")
        for checkpoint, reference_state in reference_checkpoints.items():
            subject_state = subject_checkpoints[checkpoint]
            if set(subject_state) != set(reference_state):
                raise ValueError(f"member {member}, checkpoint {checkpoint}: species sets must match")
            if set(reference_state) != set(atols):
                missing = set(reference_state) - set(atols)
                extra = set(atols) - set(reference_state)
                raise ValueError(
                    f"member {member}, checkpoint {checkpoint}: atol species mismatch "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})"
                )


def assess_trajectory_accuracy(
    subject: Trajectory,
    reference: Trajectory,
    atol_by_species: Mapping[str, float],
    rtol: float,
    *,
    solver_success: bool = True,
    positivity_passed: bool = True,
    conservation_passed: bool = True,
) -> TrajectoryAccuracyResult:
    """Assess every member/checkpoint against the common WRMS and maximum gates."""

    if not subject or not reference:
        raise ValueError("subject and reference trajectories must not be empty")
    if not math.isfinite(rtol) or rtol <= 0:
        raise ValueError("rtol must be finite and strictly positive")
    if not atol_by_species:
        raise ValueError("atol_by_species must not be empty")
    for species, atol in atol_by_species.items():
        if not math.isfinite(atol) or atol <= 0:
            raise ValueError(f"atol for species {species!r} must be finite and strictly positive")

    _validate_shape(subject, reference, atol_by_species)
    scaled: list[ScaledSpeciesError] = []
    checkpoint_wrms: list[float] = []
    maximum = 0.0
    finite = True

    for member in sorted(reference):
        for checkpoint in sorted(reference[member]):
            checkpoint_errors: list[float] = []
            for species in sorted(reference[member][checkpoint]):
                observed = subject[member][checkpoint][species]
                expected = reference[member][checkpoint][species]
                if not math.isfinite(observed) or not math.isfinite(expected):
                    finite = False
                    value = math.inf
                else:
                    value = symmetric_scaled_error(observed, expected, atol_by_species[species], rtol)
                scaled.append(ScaledSpeciesError(member, checkpoint, species, value))
                checkpoint_errors.append(value)
                maximum = max(maximum, abs(value))
            checkpoint_wrms.append(math.hypot(*checkpoint_errors) / math.sqrt(len(checkpoint_errors)))

    worst_wrms = max(checkpoint_wrms, default=math.inf)
    diagnostics: list[str] = []
    if not solver_success:
        diagnostics.append("solver did not report successful integration")
    if not finite:
        diagnostics.append("subject or reference state contains a non-finite value")
    if not positivity_passed:
        diagnostics.append("positivity policy failed")
    if not conservation_passed:
        diagnostics.append("conservation check failed")
    if worst_wrms > 1.0:
        diagnostics.append(f"checkpoint WRMS error {worst_wrms:.6g} exceeds 1")
    if maximum > 5.0:
        diagnostics.append(f"maximum scaled error {maximum:.6g} exceeds 5")

    eligible = solver_success and finite and positivity_passed and conservation_passed and worst_wrms <= 1.0 and maximum <= 5.0
    return TrajectoryAccuracyResult(
        scaled_errors=tuple(scaled),
        aggregate_errors=ErrorMetrics(wrms=worst_wrms, maximum=maximum),
        positivity_passed=positivity_passed,
        conservation_passed=conservation_passed,
        finite_state_passed=finite,
        solver_success=solver_success,
        eligible=eligible,
        diagnostics=tuple(diagnostics),
    )


__all__ = ["TrajectoryAccuracyResult", "assess_trajectory_accuracy", "symmetric_scaled_error"]
