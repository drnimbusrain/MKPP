"""Equal-accuracy solver configuration calibration tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from mkpp.benchmark.calibration import (
    TOLERANCE_MULTIPLIERS,
    CalibrationCandidate,
    CalibrationContext,
    freeze_selected_configuration,
    select_equal_accuracy_configuration,
)


def candidate(
    configuration_id: str,
    *,
    multiplier: float = 1.0,
    median_ms: float = 100.0,
    maximum_error: float = 0.5,
    wrms: float = 0.2,
    rejected_steps: int | None = 1,
    passed: bool = True,
) -> CalibrationCandidate:
    return CalibrationCandidate(
        configuration_id=configuration_id,
        configuration_sha256=(configuration_id[0] * 64),
        tolerance_multiplier=multiplier,
        paired_block_times_ms=(median_ms - 2.0, median_ms, median_ms + 2.0, median_ms + 1.0, median_ms - 1.0),
        maximum_error=maximum_error,
        wrms=wrms,
        rejected_steps=rejected_steps,
        passed_all_gates=passed,
    )


def context() -> CalibrationContext:
    return CalibrationContext(
        calibration_scenario_id="chapman-calibration-v1",
        calibration_seed=17,
        measurement_scenario_id="chapman-measurement-v1",
        measurement_seed=29,
    )


def test_candidate_grid_is_finite_and_predeclared() -> None:
    assert TOLERANCE_MULTIPLIERS == (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)


def test_rejects_candidate_outside_predeclared_grid() -> None:
    with pytest.raises(ValueError, match="predeclared"):
        select_equal_accuracy_configuration([candidate("a", multiplier=3.0)], context())


@pytest.mark.parametrize(
    ("calibration_id", "calibration_seed", "measurement_id", "measurement_seed"),
    [
        ("same", 17, "same", 29),
        ("calibration", 17, "measurement", 17),
    ],
)
def test_calibration_and_measurement_must_be_disjoint(
    calibration_id: str,
    calibration_seed: int,
    measurement_id: str,
    measurement_seed: int,
) -> None:
    overlapping = CalibrationContext(
        calibration_scenario_id=calibration_id,
        calibration_seed=calibration_seed,
        measurement_scenario_id=measurement_id,
        measurement_seed=measurement_seed,
    )

    with pytest.raises(ValueError, match="disjoint"):
        select_equal_accuracy_configuration([candidate("a")], overlapping)


def test_only_candidates_passing_every_accuracy_gate_are_selectable() -> None:
    selected = select_equal_accuracy_configuration(
        [
            candidate("a", median_ms=40.0, passed=False),
            candidate("b", median_ms=100.0, passed=True),
        ],
        context(),
    )

    assert selected.configuration_id == "b"


def test_requires_five_paired_calibration_blocks() -> None:
    incomplete = candidate("a")
    incomplete = CalibrationCandidate(
        configuration_id=incomplete.configuration_id,
        configuration_sha256=incomplete.configuration_sha256,
        tolerance_multiplier=incomplete.tolerance_multiplier,
        paired_block_times_ms=(98.0, 100.0, 102.0, 101.0),
        maximum_error=incomplete.maximum_error,
        wrms=incomplete.wrms,
        rejected_steps=incomplete.rejected_steps,
        passed_all_gates=incomplete.passed_all_gates,
    )

    with pytest.raises(ValueError, match="five paired blocks"):
        select_equal_accuracy_configuration([incomplete], context())


def test_selects_lowest_median_outside_two_percent_tie_window() -> None:
    selected = select_equal_accuracy_configuration(
        [candidate("a", median_ms=100.0, maximum_error=0.1), candidate("b", median_ms=97.0, maximum_error=0.9)],
        context(),
    )

    assert selected.configuration_id == "b"


def test_two_percent_tie_breaks_by_error_then_wrms_then_rejections_then_id() -> None:
    selected = select_equal_accuracy_configuration(
        [
            candidate("f", median_ms=100.0, maximum_error=0.3, wrms=0.1, rejected_steps=0),
            candidate("e", median_ms=99.0, maximum_error=0.2, wrms=0.3, rejected_steps=0),
            candidate("d", median_ms=99.0, maximum_error=0.2, wrms=0.2, rejected_steps=3),
            candidate("c", median_ms=99.0, maximum_error=0.2, wrms=0.2, rejected_steps=1),
            candidate("b", median_ms=99.0, maximum_error=0.2, wrms=0.2, rejected_steps=1),
        ],
        context(),
    )

    assert selected.configuration_id == "b"


def test_unavailable_rejected_step_counter_sorts_after_available_counter() -> None:
    selected = select_equal_accuracy_configuration(
        [candidate("a", rejected_steps=None), candidate("b", rejected_steps=4)],
        context(),
    )

    assert selected.configuration_id == "b"


def test_no_eligible_candidate_is_an_explicit_calibration_failure() -> None:
    with pytest.raises(ValueError, match="no.*eligible"):
        select_equal_accuracy_configuration([candidate("a", passed=False)], context())


def test_selected_configuration_is_hash_bound_and_immutable_before_measurement() -> None:
    selected = select_equal_accuracy_configuration([candidate("a")], context())
    frozen = freeze_selected_configuration(selected)

    assert frozen.configuration_id == "a"
    assert frozen.configuration_sha256 == "a" * 64
    assert frozen.frozen is True
    with pytest.raises(FrozenInstanceError):
        frozen.configuration_sha256 = "b" * 64


def test_duplicate_candidate_id_cannot_replace_another_candidates_timing():
    with pytest.raises(ValueError, match="IDs must be unique"):
        select_equal_accuracy_configuration([candidate("a", median_ms=100), candidate("a", median_ms=10)], context())


@pytest.mark.parametrize("field", ["maximum_error", "wrms"])
def test_negative_error_metric_is_not_calibration_evidence(field):
    with pytest.raises(ValueError, match="cannot be negative"):
        select_equal_accuracy_configuration([candidate("a", **{field: -1.0})], context())


@pytest.mark.parametrize("metrics", [{"maximum_error": 6.0}, {"wrms": 1.1}])
def test_eligibility_flag_cannot_override_failed_accuracy_metrics(metrics):
    selected = select_equal_accuracy_configuration(
        [candidate("a", median_ms=10.0, passed=True, **metrics), candidate("b", median_ms=100.0)], context()
    )
    assert selected.configuration_id == "b"
