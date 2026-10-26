"""Campaign eligibility checks for comparable paired performance evidence."""

from __future__ import annotations

import pytest
from mkpp.benchmark.models import (
    BuildProvenance,
    CampaignStatus,
    HashIdentity,
    ResourcePolicy,
    RunStatus,
    SolverId,
    SolverRun,
    TimingBoundary,
)
from mkpp.benchmark.orchestrator import advance_campaign_status, assess_repeatability, require_comparable_pairs


def run(
    solver: SolverId,
    block: int,
    *,
    matching_key: str = "a" * 64,
    boundary: TimingBoundary = TimingBoundary.STEADY_STATE_SOLVE,
    cores: int = 1,
    threads: int = 1,
    elapsed: float = 2.0,
) -> SolverRun:
    identity = HashIdentity("chapman", "v1", "b" * 64)
    resources = ResourcePolicy("host", "cpu", cores, threads, "core", "local")
    return SolverRun(
        "campaign",
        f"{solver.value}-{block}",
        block,
        1,
        {SolverId.MICM: 0, SolverId.KPP: 1, SolverId.MKPP: 2}[solver],
        matching_key,
        solver,
        identity,
        identity,
        identity,
        BuildProvenance("c" * 64, (), "cc", (), "double"),
        resources,
        boundary,
        10,
        elapsed,
        "steady",
        True,
        "none",
        (),
        (),
        (),
        RunStatus.SUCCESS,
    )


def test_pairs_require_matching_key_resources_and_timing_boundary() -> None:
    baseline = (run(SolverId.MICM, 0), run(SolverId.MICM, 1))
    candidate = (run(SolverId.MKPP, 0), run(SolverId.MKPP, 1))
    assert require_comparable_pairs(baseline, candidate) == (0, 1)

    with pytest.raises(ValueError, match="matching key"):
        require_comparable_pairs(baseline, (run(SolverId.MKPP, 0, matching_key="d" * 64), candidate[1]))
    with pytest.raises(ValueError, match="resources"):
        require_comparable_pairs(baseline, (run(SolverId.MKPP, 0, threads=2), candidate[1]))
    with pytest.raises(ValueError, match="timing boundary"):
        require_comparable_pairs(baseline, (run(SolverId.MKPP, 0, boundary=TimingBoundary.LIFECYCLE), candidate[1]))


def test_primary_cpu_policy_requires_one_physical_core_and_thread() -> None:
    with pytest.raises(ValueError, match="one physical core"):
        require_comparable_pairs((run(SolverId.MICM, 0, cores=2),), (run(SolverId.MKPP, 0, cores=2),), primary_cpu=True)


def test_repeatability_is_within_five_percent_of_prior_median() -> None:
    assert assess_repeatability((100.0, 101.0), (102.0, 98.0))
    assert not assess_repeatability((100.0,), (106.0,))


def test_campaign_lifecycle_is_ordered_and_cannot_skip_scientific_gates() -> None:
    assert advance_campaign_status(CampaignStatus.CONFIGURED) is CampaignStatus.PREFLIGHT_PASSED
    with pytest.raises(ValueError, match="cannot transition"):
        advance_campaign_status(CampaignStatus.CONFIGURED, CampaignStatus.MEASURING)


def test_duplicate_blocks_cannot_silently_overwrite_a_slow_sample():
    with pytest.raises(ValueError, match="duplicate block"):
        require_comparable_pairs((run(SolverId.MICM, 0), run(SolverId.MICM, 0, elapsed=200)), (run(SolverId.MKPP, 0),))


def test_same_solver_cannot_be_presented_as_a_competitor():
    with pytest.raises(ValueError, match="distinct solvers"):
        require_comparable_pairs((run(SolverId.MICM, 0),), (run(SolverId.MICM, 0),))


def test_publication_can_require_the_minimum_valid_block_count():
    with pytest.raises(ValueError, match="at least 10"):
        require_comparable_pairs((run(SolverId.MICM, 0),), (run(SolverId.MKPP, 0),), minimum_blocks=10)


@pytest.mark.parametrize("elapsed", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_duration_cannot_enter_comparison(elapsed):
    with pytest.raises(ValueError, match="invalid timing"):
        require_comparable_pairs((run(SolverId.MICM, 0, elapsed=elapsed),), (run(SolverId.MKPP, 0),))


def test_changed_configuration_within_one_solver_series_is_rejected():
    from dataclasses import replace

    candidate = run(SolverId.MKPP, 1)
    candidate = replace(candidate, configuration=HashIdentity("tuned-on-test", "v1", "f" * 64))
    with pytest.raises(ValueError, match="frozen evidence"):
        require_comparable_pairs((run(SolverId.MICM, 0), run(SolverId.MICM, 1)), (run(SolverId.MKPP, 0), candidate))


def test_equal_accuracy_solvers_may_have_different_frozen_configurations():
    from dataclasses import replace

    candidate = replace(run(SolverId.MKPP, 0), configuration=HashIdentity("mkpp-calibrated", "v1", "f" * 64))
    assert require_comparable_pairs((run(SolverId.MICM, 0),), (candidate,)) == (0,)


@pytest.mark.parametrize("backend", ["cuda", "hip", "native"])
def test_gpu_or_unknown_backend_cannot_enter_primary_cpu_result(backend):
    from dataclasses import replace

    baseline = run(SolverId.MICM, 0)
    candidate = run(SolverId.MKPP, 0)
    resources = replace(baseline.resources, backend=backend)
    with pytest.raises(ValueError, match="verified CPU backend"):
        require_comparable_pairs(
            (replace(baseline, resources=resources),), (replace(candidate, resources=resources),), primary_cpu=True
        )


def test_two_solvers_cannot_report_the_same_interleaved_position():
    from dataclasses import replace

    baseline = run(SolverId.MICM, 0)
    candidate = replace(run(SolverId.MKPP, 0), order_position=baseline.order_position)
    with pytest.raises(ValueError, match="same order position"):
        require_comparable_pairs((baseline,), (candidate,))
