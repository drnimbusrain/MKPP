"""Performance controls must be coherent before timing native subjects."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from mkpp.benchmark.policy import validate_performance_policy

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def policy():
    return yaml.safe_load((ROOT / "benchmarks/solver_comparison/campaigns/cpu-ringer-v1.yaml").read_text())


def test_ringer_policy_has_complete_two_and_three_solver_cycles(policy):
    validate_performance_policy(policy, [3, 2, 2, 2])


@pytest.mark.parametrize(
    "field,value",
    [
        ("pilot_iterations", 4),
        ("pilot_iterations", True),
        ("minimum_sample_seconds", float("nan")),
        ("minimum_sample_seconds", 0.5),
        ("minimum_valid_blocks", 2),
        ("paired_blocks", 5),
        ("bootstrap_resamples", 999),
        ("ordering_seed", -1),
    ],
)
def test_invalid_sampling_is_rejected(policy, field, value):
    policy["sampling"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_performance_policy(policy, [3, 2])


def test_incomplete_permutation_cycles_cannot_claim_balanced_ringer_design(policy):
    policy["sampling"]["paired_blocks"] = 15
    with pytest.raises(ValueError, match="permutation cycle"):
        validate_performance_policy(policy, [3, 2])


@pytest.mark.parametrize("field,value", [("minimum", 2), ("maximum", 1), ("stable_last_three_relative_range", float("inf"))])
def test_invalid_warmup_policy_is_rejected(policy, field, value):
    policy["sampling"]["warmups"][field] = value
    with pytest.raises(ValueError, match="warmup"):
        validate_performance_policy(policy, [3, 2])


def test_invalid_timing_boundary_and_disabled_science_gate_are_rejected(policy):
    modified = deepcopy(policy)
    modified["timing_boundaries"] = ["gpu_guess"]
    with pytest.raises(ValueError, match="timing_boundaries"):
        validate_performance_policy(modified, [3, 2])
    policy["publication"]["require_scientific_eligibility"] = False
    with pytest.raises(ValueError, match="cannot be disabled"):
        validate_performance_policy(policy, [3, 2])
