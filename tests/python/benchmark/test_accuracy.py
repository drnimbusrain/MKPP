"""RED tests for shared, species-scaled trajectory accuracy gates."""

from __future__ import annotations

import math

import pytest
from mkpp.benchmark.accuracy import assess_trajectory_accuracy, symmetric_scaled_error

ATOL = {"A": 1.0e-6, "B": 1.0e-3}
RTOL = 1.0e-4


def _trajectory(a: float = 1.0, b: float = 10.0) -> dict[int, dict[float, dict[str, float]]]:
    return {0: {60.0: {"A": a, "B": b}, 120.0: {"A": a, "B": b}}}


def test_symmetric_scaled_error_uses_larger_subject_or_reference_magnitude() -> None:
    observed = 2.0
    reference = 1.0
    atol = 0.5
    rtol = 0.25

    assert symmetric_scaled_error(observed, reference, atol, rtol) == pytest.approx(1.0)
    assert symmetric_scaled_error(reference, observed, atol, rtol) == pytest.approx(-1.0)


def test_eligible_trajectory_reports_wrms_and_maximum() -> None:
    reference = _trajectory()
    subject = _trajectory(a=1.0 + 5.0e-5, b=10.0 - 5.0e-4)

    assessment = assess_trajectory_accuracy(subject, reference, ATOL, RTOL)

    assert assessment.eligible
    assert assessment.aggregate_errors.wrms <= 1.0
    assert assessment.aggregate_errors.maximum <= 5.0
    assert len(assessment.scaled_errors) == 4


def test_wrms_above_one_at_any_checkpoint_is_ineligible() -> None:
    reference = _trajectory()
    subject = _trajectory()
    subject[0][60.0] = {"A": 1.001, "B": 10.02}

    assessment = assess_trajectory_accuracy(subject, reference, ATOL, RTOL)

    assert not assessment.eligible
    assert assessment.aggregate_errors.wrms > 1.0
    assert any("WRMS" in diagnostic for diagnostic in assessment.diagnostics)


def test_maximum_above_five_cannot_be_hidden_by_many_species() -> None:
    species = {f"S{i}": 1.0 for i in range(100)}
    reference = {0: {60.0: dict(species)}}
    subject = {0: {60.0: dict(species)}}
    subject[0][60.0]["S0"] = 1.00061
    atol = {name: 1.0e-6 for name in species}

    assessment = assess_trajectory_accuracy(subject, reference, atol, 1.0e-4)

    assert assessment.aggregate_errors.wrms <= 1.0
    assert assessment.aggregate_errors.maximum > 5.0
    assert not assessment.eligible
    assert any("maximum" in diagnostic.lower() for diagnostic in assessment.diagnostics)


@pytest.mark.parametrize(
    ("flag", "value", "diagnostic"),
    [
        ("solver_success", False, "solver"),
        ("positivity_passed", False, "positivity"),
        ("conservation_passed", False, "conservation"),
    ],
)
def test_scientific_or_solver_failure_makes_accurate_state_ineligible(flag: str, value: bool, diagnostic: str) -> None:
    kwargs = {flag: value}
    assessment = assess_trajectory_accuracy(_trajectory(), _trajectory(), ATOL, RTOL, **kwargs)

    assert not assessment.eligible
    assert any(diagnostic in message.lower() for message in assessment.diagnostics)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_subject_state_is_retained_as_ineligible(bad_value: float) -> None:
    subject = _trajectory()
    subject[0][120.0]["A"] = bad_value

    assessment = assess_trajectory_accuracy(subject, _trajectory(), ATOL, RTOL)

    assert not assessment.eligible
    assert not assessment.finite_state_passed
    assert any("finite" in diagnostic.lower() for diagnostic in assessment.diagnostics)


def test_missing_checkpoint_or_species_fails_instead_of_comparing_partial_state() -> None:
    subject = _trajectory()
    del subject[0][120.0]["B"]

    with pytest.raises(ValueError, match="checkpoint|species|B"):
        assess_trajectory_accuracy(subject, _trajectory(), ATOL, RTOL)


@pytest.mark.parametrize(("atol", "rtol"), [(0.0, RTOL), (-1.0, RTOL), (1.0, 0.0)])
def test_error_scale_must_be_strictly_positive(atol: float, rtol: float) -> None:
    with pytest.raises(ValueError, match="atol|rtol|positive"):
        symmetric_scaled_error(1.0, 1.0, atol, rtol)


def test_empty_member_cannot_hide_missing_trajectory():
    with pytest.raises(ValueError, match="checkpoints.*empty"):
        assess_trajectory_accuracy({0: {}}, {0: {}}, ATOL, RTOL)


def test_binary64_extremes_do_not_turn_large_error_into_zero():
    assert symmetric_scaled_error(1e308, -1e308, 1.0, 1e-4) == pytest.approx(20000.0)
    assert symmetric_scaled_error(1e308, 0.0, 1.0, 10.0) == pytest.approx(0.1)


def test_large_scaled_errors_are_ineligible_without_overflow_crash():
    result = assess_trajectory_accuracy({0: {1.0: {"A": 1e308}}}, {0: {1.0: {"A": 0.0}}}, {"A": 1e-300}, 1e-300)
    assert not result.eligible
    assert result.aggregate_errors.wrms > 1.0
