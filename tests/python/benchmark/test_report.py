"""Report derivation keeps scientific eligibility and timing boundaries visible."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mkpp.benchmark.report import (
    derive_pairwise_comparison,
    read_raw_results,
    render_csv_report,
    render_json_report,
    render_markdown_report,
)


def test_raw_reader_retains_each_jsonl_record_in_order(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text('{"run":"first"}\n{"run":"second"}\n', encoding="utf-8")
    assert tuple(dict(row) for row in read_raw_results(path)) == ({"run": "first"}, {"run": "second"})


def test_pairwise_ratio_names_its_baseline_and_withholds_ineligible_pairs() -> None:
    comparison = derive_pairwise_comparison(
        baseline_solver_id="micm",
        candidate_solver_id="mkpp",
        baseline_seconds=(4.0, 6.0) * 5,
        candidate_seconds=(2.0, 3.0) * 5,
        eligible=True,
        seed=11,
    )
    assert comparison["ratio_direction"] == "micm_seconds / mkpp_seconds"
    assert comparison["median_ratio"] == 2.0

    withheld = derive_pairwise_comparison(
        baseline_solver_id="micm",
        candidate_solver_id="mkpp",
        baseline_seconds=(4.0,),
        candidate_seconds=(2.0,),
        eligible=False,
        seed=11,
        withheld_reason="trajectory accuracy failed",
    )
    assert withheld["status"] == "withheld"
    assert withheld["median_ratio"] is None


def test_markdown_report_keeps_solve_and_lifecycle_sections_separate() -> None:
    text = render_markdown_report(
        {
            "mechanism_id": "ts1",
            "admission_status": "chemistry_verified",
            "admission_reason": "KPP is not admitted",
            "steady_state_solve": [
                {
                    "baseline_solver_id": "micm",
                    "candidate_solver_id": "mkpp",
                    "status": "eligible",
                    "median_ratio": 2.0,
                    "ratio_direction": "micm_seconds / mkpp_seconds",
                }
            ],
            "lifecycle": [
                {
                    "baseline_solver_id": "micm",
                    "candidate_solver_id": "mkpp",
                    "status": "withheld",
                    "median_ratio": None,
                    "withheld_reason": "not collected",
                }
            ],
        }
    )
    assert "## Steady-state solve" in text
    assert "## Lifecycle" in text
    assert "micm_seconds / mkpp_seconds" in text
    assert "KPP is not admitted" in text


def test_raw_reader_rejects_non_object_records(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(["not", "an", "object"]) + "\n", encoding="utf-8")
    try:
        read_raw_results(path)
    except ValueError as error:
        assert "object" in str(error)
    else:
        raise AssertionError("non-object raw evidence must be rejected")


def test_raw_reader_returns_immutable_nested_evidence(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text('{"nested":{"sample":[1]}}\n', encoding="utf-8")
    record = read_raw_results(path)[0]
    with pytest.raises(TypeError):
        record["new"] = "value"  # type: ignore[index]
    assert record["nested"]["sample"] == (1,)


def test_json_and_csv_reports_preserve_boundaries_and_uncertainty() -> None:
    document = {
        "mechanism_id": "chapman",
        "admission_status": "chemistry_verified",
        "admission_reason": "all gates passed",
        "steady_state_solve": [
            {
                "baseline_solver_id": "micm",
                "candidate_solver_id": "mkpp",
                "ratio_direction": "micm_seconds / mkpp_seconds",
                "status": "eligible",
                "median_ratio": 1.2,
                "bootstrap_interval": (1.1, 1.3),
            }
        ],
        "lifecycle": [],
    }
    json_report = json.loads(render_json_report(document))
    assert json_report["steady_state_solve"][0]["bootstrap_interval"] == [1.1, 1.3]
    assert json_report["admission_reason"] == "all gates passed"
    csv = render_csv_report(document)
    assert "steady_state_solve" in csv
    assert "1.1" in csv
    assert "lifecycle" not in csv


def test_two_samples_cannot_be_published_as_a_qualified_speedup():
    comparison = derive_pairwise_comparison(
        baseline_solver_id="kpp",
        candidate_solver_id="mkpp",
        baseline_seconds=(2.0, 3.0),
        candidate_seconds=(1.0, 1.0),
        eligible=True,
        seed=17,
    )
    assert comparison["status"] == "withheld"
    assert "10 valid paired blocks" in comparison["withheld_reason"]


def test_verified_comparison_aligns_by_block_and_rejects_duplicates():
    from mkpp.benchmark.models import SolverId
    from mkpp.benchmark.report import derive_verified_comparison

    from .test_campaign import run

    baseline = tuple(run(SolverId.MICM, block, elapsed=4.0) for block in range(10))
    candidate = tuple(run(SolverId.MKPP, block, elapsed=2.0) for block in reversed(range(10)))
    result = derive_verified_comparison(baseline, candidate, scientific_eligible=True, seed=17)
    assert result["median_ratio"] == 2.0
    assert result["bootstrap_interval"] == (2.0, 2.0)
    duplicate = derive_verified_comparison(baseline + (baseline[0],), candidate, scientific_eligible=True, seed=17)
    assert duplicate["status"] == "withheld"
    assert "duplicate" in duplicate["withheld_reason"]
    failed_science = derive_verified_comparison(baseline, candidate, scientific_eligible=False, seed=17)
    assert failed_science["median_ratio"] is None


def test_csv_roundtrips_quotes_commas_and_newlines():
    import csv
    import io

    reason = 'failed "O3", checkpoint\n60 s'
    report = render_csv_report(
        {"mechanism_id": "chapman", "steady_state_solve": [{"status": "withheld", "withheld_reason": reason}]}
    )
    rows = list(csv.DictReader(io.StringIO(report)))
    assert len(rows) == 1
    assert rows[0]["withheld_reason"] == reason


def test_retained_samples_must_meet_declared_campaign_floors():
    from mkpp.benchmark.models import SolverId
    from mkpp.benchmark.report import derive_verified_comparison

    from .test_campaign import run

    baseline = tuple(run(SolverId.MICM, block) for block in range(10))
    candidate = tuple(run(SolverId.MKPP, block, elapsed=0.001) for block in range(10))
    short = derive_verified_comparison(baseline, candidate, scientific_eligible=True, seed=17)
    assert short["median_ratio"] is None
    assert "duration floor" in short["withheld_reason"]
    incomplete = derive_verified_comparison(baseline, baseline, scientific_eligible=True, seed=17, minimum_valid_blocks=12)
    assert incomplete["median_ratio"] is None
    valid_candidate = tuple(run(SolverId.MKPP, block) for block in range(10))
    incomplete = derive_verified_comparison(baseline, valid_candidate, scientific_eligible=True, seed=17, minimum_valid_blocks=12)
    assert "at least 12" in incomplete["withheld_reason"]
