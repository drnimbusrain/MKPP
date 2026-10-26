"""Independent-reference tolerance refinement and convergence qualification."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from mkpp.benchmark.accuracy import symmetric_scaled_error
from mkpp.benchmark.models import ErrorMetrics

ReferenceTrajectory = Mapping[float, Mapping[str, float]]
_MIN_EFFECTIVE_RTOL = 5.0e-14
_MIN_INITIAL_R1_RTOL = 5.0e-13


class HigherPrecisionRequiredError(ValueError):
    """Raised when ordered reference controls cannot be represented safely."""


HigherPrecisionRequired = HigherPrecisionRequiredError


@dataclass(frozen=True, slots=True)
class ReferenceTolerancePair:
    r1_rtol: float
    r2_rtol: float
    r1_atol_by_species: dict[str, float]
    r2_atol_by_species: dict[str, float]


@dataclass(frozen=True, slots=True)
class ReferenceConvergenceResult:
    errors: ErrorMetrics
    finite: bool
    qualified: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReferenceAttemptsResult:
    qualified: bool
    selected_attempt: int | None
    reference: ReferenceTrajectory | None
    comparisons: tuple[ReferenceConvergenceResult, ...]
    diagnostics: tuple[str, ...] = ()


def build_reference_tolerance_pairs(
    subject_rtol: float,
    subject_atol_by_species: Mapping[str, float],
    max_refinements: int,
) -> tuple[ReferenceTolerancePair, ...]:
    """Build deterministic, strictly ordered R1/R2 tolerance pairs."""

    if not math.isfinite(subject_rtol) or subject_rtol <= 0:
        raise ValueError("subject rtol must be finite and strictly positive")
    if max_refinements < 1 or max_refinements > 3:
        raise ValueError("max_refinements must be between 1 and 3")
    if not subject_atol_by_species:
        raise ValueError("subject atol mapping must not be empty")
    for species, atol in subject_atol_by_species.items():
        if not math.isfinite(atol) or atol <= 0:
            raise ValueError(f"subject atol for {species!r} must be finite and strictly positive")

    first_r1 = min(1.0e-11, subject_rtol / 100.0)
    if first_r1 < _MIN_INITIAL_R1_RTOL:
        raise HigherPrecisionRequired(
            "reference tolerance floor 5e-14 cannot preserve the requested ordering; higher precision is required"
        )

    pairs: list[ReferenceTolerancePair] = []
    r1_rtol = first_r1
    r1_atols = {species: atol / 100.0 for species, atol in subject_atol_by_species.items()}
    for _ in range(max_refinements):
        r2_rtol = max(_MIN_EFFECTIVE_RTOL, r1_rtol / 10.0)
        if r2_rtol >= r1_rtol:
            raise HigherPrecisionRequired("reference tolerance floor 5e-14 prevents a stricter R2; higher precision is required")
        r2_atols = {species: atol / 10.0 for species, atol in r1_atols.items()}
        pairs.append(ReferenceTolerancePair(r1_rtol, r2_rtol, dict(r1_atols), dict(r2_atols)))
        r1_rtol = r2_rtol
        r1_atols = r2_atols
    return tuple(pairs)


def integration_segments(
    start_seconds: float, horizon_seconds: float, forcing_boundaries_seconds: Sequence[float]
) -> tuple[tuple[float, float], ...]:
    """Return integration intervals split at every forcing discontinuity."""

    if not math.isfinite(start_seconds) or not math.isfinite(horizon_seconds) or horizon_seconds <= 0:
        raise ValueError("integration interval must have finite start and positive horizon")
    end = start_seconds + horizon_seconds
    boundaries = list(forcing_boundaries_seconds)
    if any(not math.isfinite(value) for value in boundaries):
        raise ValueError("forcing boundaries must be finite")
    if boundaries != sorted(set(boundaries)):
        raise ValueError("forcing boundaries must be strictly increasing and unique")
    if any(value < start_seconds or value > end for value in boundaries):
        raise ValueError("forcing boundary lies outside integration interval")
    points = sorted(set((start_seconds, *boundaries, end)))
    return tuple(zip(points[:-1], points[1:], strict=True))


def _validate_trajectory_pair(r1: ReferenceTrajectory, r2: ReferenceTrajectory, atols: Mapping[str, float]) -> None:
    if not atols:
        raise ValueError("reference species and atol sets must not be empty")
    if not r1 or set(r1) != set(r2):
        raise ValueError("reference checkpoint sets must be non-empty and identical")
    for checkpoint, r1_state in r1.items():
        r2_state = r2[checkpoint]
        if set(r1_state) != set(r2_state):
            raise ValueError(f"checkpoint {checkpoint}: reference species sets must match")
        if set(r1_state) != set(atols):
            raise ValueError(f"checkpoint {checkpoint}: atol and reference species sets must match")


def assess_reference_convergence(
    r1: ReferenceTrajectory,
    r2: ReferenceTrajectory,
    atol_by_species: Mapping[str, float],
    subject_rtol: float,
) -> ReferenceConvergenceResult:
    """Assess one ordered reference pair under the reserved error budget."""

    _validate_trajectory_pair(r1, r2, atol_by_species)
    checkpoint_wrms: list[float] = []
    maximum = 0.0
    finite = True
    for checkpoint in sorted(r1):
        errors: list[float] = []
        for species in sorted(r1[checkpoint]):
            first = r1[checkpoint][species]
            second = r2[checkpoint][species]
            if not math.isfinite(first) or not math.isfinite(second):
                finite = False
                error = math.inf
            else:
                error = symmetric_scaled_error(second, first, atol_by_species[species], subject_rtol)
            errors.append(error)
            maximum = max(maximum, abs(error))
        checkpoint_wrms.append(math.hypot(*errors) / math.sqrt(len(errors)))
    worst_wrms = max(checkpoint_wrms)
    diagnostics: list[str] = []
    if not finite:
        diagnostics.append("reference state contains a non-finite value")
    if worst_wrms > 0.05:
        diagnostics.append(f"reference WRMS {worst_wrms:.6g} exceeds 0.05")
    if maximum > 0.25:
        diagnostics.append(f"reference maximum {maximum:.6g} exceeds 0.25")
    return ReferenceConvergenceResult(
        errors=ErrorMetrics(wrms=worst_wrms, maximum=maximum),
        finite=finite,
        qualified=finite and worst_wrms <= 0.05 and maximum <= 0.25,
        diagnostics=tuple(diagnostics),
    )


def qualify_reference_attempts(
    attempts: Sequence[ReferenceTrajectory],
    atol_by_species: Mapping[str, float],
    subject_rtol: float,
    *,
    scientific_checks_passed: bool,
) -> ReferenceAttemptsResult:
    """Select the first converged stricter trajectory, or retain an unqualified result."""

    if len(attempts) < 2:
        raise ValueError("reference qualification requires at least two ordered attempts")
    comparisons: list[ReferenceConvergenceResult] = []
    diagnostics: list[str] = []
    selected: int | None = None
    for index in range(len(attempts) - 1):
        comparison = assess_reference_convergence(attempts[index], attempts[index + 1], atol_by_species, subject_rtol)
        comparisons.append(comparison)
        if comparison.qualified:
            selected = index + 1
            break
    if not scientific_checks_passed:
        diagnostics.append("reference scientific checks failed")
        selected = None
    if selected is None:
        if any(not comparison.finite for comparison in comparisons):
            diagnostics.append("reference attempts contain non-finite state")
        diagnostics.append("reference is unqualified after ordered refinements")
    return ReferenceAttemptsResult(
        qualified=selected is not None,
        selected_attempt=selected,
        reference=attempts[selected] if selected is not None else None,
        comparisons=tuple(comparisons),
        diagnostics=tuple(diagnostics),
    )


__all__ = [
    "HigherPrecisionRequired",
    "HigherPrecisionRequiredError",
    "ReferenceAttemptsResult",
    "ReferenceConvergenceResult",
    "ReferenceTolerancePair",
    "assess_reference_convergence",
    "build_reference_tolerance_pairs",
    "integration_segments",
    "qualify_reference_attempts",
]
