"""RED tests for independently converged reference qualification."""

from __future__ import annotations

import math

import pytest
from mkpp.benchmark.reference import (
    HigherPrecisionRequired,
    assess_reference_convergence,
    build_reference_tolerance_pairs,
    integration_segments,
    qualify_reference_attempts,
)


def _state(a: float = 1.0, b: float = 10.0) -> dict[float, dict[str, float]]:
    return {60.0: {"A": a, "B": b}, 600.0: {"A": a, "B": b}}


def test_reference_tolerances_are_ordered_and_tied_to_subject_budget() -> None:
    pairs = build_reference_tolerance_pairs(
        subject_rtol=1.0e-6,
        subject_atol_by_species={"A": 1.0e-4, "B": 1.0e-8},
        max_refinements=3,
    )

    assert len(pairs) == 3
    assert pairs[0].r1_rtol == pytest.approx(1.0e-11)
    assert pairs[0].r2_rtol == pytest.approx(1.0e-12)
    assert pairs[0].r1_atol_by_species == pytest.approx({"A": 1.0e-6, "B": 1.0e-10})
    assert pairs[0].r2_atol_by_species == pytest.approx({"A": 1.0e-7, "B": 1.0e-11})
    assert [pair.r1_rtol for pair in pairs] == pytest.approx([1.0e-11, 1.0e-12, 1.0e-13])
    assert [pair.r2_rtol for pair in pairs] == pytest.approx([1.0e-12, 1.0e-13, 5.0e-14])
    assert all(pair.r2_rtol < pair.r1_rtol for pair in pairs)


def test_tolerance_floor_requires_higher_precision_instead_of_clamping_looser() -> None:
    with pytest.raises(HigherPrecisionRequired, match="precision|5e-14|floor"):
        build_reference_tolerance_pairs(
            subject_rtol=1.0e-11,
            subject_atol_by_species={"A": 1.0e-12},
            max_refinements=3,
        )


def test_integration_is_split_at_each_forcing_boundary() -> None:
    assert integration_segments(0.0, 600.0, [0.0, 120.0, 300.0, 600.0]) == (
        (0.0, 120.0),
        (120.0, 300.0),
        (300.0, 600.0),
    )


@pytest.mark.parametrize(
    "boundaries",
    [
        [0.0, 300.0, 120.0, 600.0],
        [0.0, 120.0, 120.0, 600.0],
        [-1.0, 600.0],
        [0.0, 601.0],
    ],
)
def test_invalid_forcing_boundaries_are_rejected(boundaries: list[float]) -> None:
    with pytest.raises(ValueError, match="forcing|boundary|interval|increasing"):
        integration_segments(0.0, 600.0, boundaries)


def test_reference_converges_only_below_wrms_and_maximum_thresholds() -> None:
    r1 = _state()
    r2 = _state(a=1.0 + 1.0e-7, b=10.0 - 1.0e-6)

    result = assess_reference_convergence(
        r1,
        r2,
        atol_by_species={"A": 1.0e-4, "B": 1.0e-3},
        subject_rtol=1.0e-4,
    )

    assert result.qualified
    assert result.errors.wrms <= 0.05
    assert result.errors.maximum <= 0.25


def test_reference_maximum_gate_prevents_wrms_dilution() -> None:
    species = {f"S{i}": 1.0 for i in range(100)}
    r1 = {60.0: dict(species)}
    r2 = {60.0: dict(species)}
    r2[60.0]["S0"] = 1.00003

    result = assess_reference_convergence(
        r1,
        r2,
        atol_by_species={name: 1.0e-6 for name in species},
        subject_rtol=1.0e-4,
    )

    assert result.errors.wrms <= 0.05
    assert result.errors.maximum > 0.25
    assert not result.qualified


def test_failed_pair_is_refined_and_first_passing_r2_is_selected() -> None:
    attempts = [
        _state(a=1.0),
        _state(a=1.01),
        _state(a=1.010001),
    ]

    qualification = qualify_reference_attempts(
        attempts,
        atol_by_species={"A": 1.0e-4, "B": 1.0e-3},
        subject_rtol=1.0e-4,
        scientific_checks_passed=True,
    )

    assert qualification.qualified
    assert qualification.selected_attempt == 2
    assert qualification.reference == attempts[2]
    assert len(qualification.comparisons) == 2


def test_exhausted_refinements_are_explicitly_unqualified() -> None:
    attempts = [_state(a=1.0), _state(a=1.1), _state(a=1.2), _state(a=1.3)]

    qualification = qualify_reference_attempts(
        attempts,
        atol_by_species={"A": 1.0e-8, "B": 1.0e-8},
        subject_rtol=1.0e-8,
        scientific_checks_passed=True,
    )

    assert not qualification.qualified
    assert qualification.reference is None
    assert qualification.selected_attempt is None
    assert any("unqualified" in message.lower() for message in qualification.diagnostics)


@pytest.mark.parametrize("scientific_checks_passed", [False])
def test_scientific_failure_rejects_numerically_converged_reference(
    scientific_checks_passed: bool,
) -> None:
    qualification = qualify_reference_attempts(
        [_state(), _state(a=1.0 + 1.0e-9)],
        atol_by_species={"A": 1.0e-4, "B": 1.0e-3},
        subject_rtol=1.0e-4,
        scientific_checks_passed=scientific_checks_passed,
    )
    assert not qualification.qualified
    assert any("scientific" in message.lower() for message in qualification.diagnostics)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_non_finite_reference_attempt_cannot_qualify(bad_value: float) -> None:
    bad = _state()
    bad[600.0]["A"] = bad_value

    qualification = qualify_reference_attempts(
        [_state(), bad],
        atol_by_species={"A": 1.0e-4, "B": 1.0e-3},
        subject_rtol=1.0e-4,
        scientific_checks_passed=True,
    )

    assert not qualification.qualified
    assert any("finite" in message.lower() for message in qualification.diagnostics)


def test_empty_reference_state_is_rejected_without_division_by_zero():
    with pytest.raises(ValueError, match="must not be empty"):
        assess_reference_convergence({1.0: {}}, {1.0: {}}, {}, 1e-6)
