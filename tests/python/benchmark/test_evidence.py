"""Adversarial protocol evidence must never be accepted as a completed workload."""

from copy import deepcopy

import pytest
from mkpp.benchmark.evidence import validate_requested_evidence
from mkpp.benchmark.manifest import canonical_sha256
from mkpp.benchmark.schema import SchemaValidationError

from .test_runner_result_schema import valid_runner_result


@pytest.fixture
def requested_evidence():
    result = valid_runner_result()
    manifest = {"mechanism_id": "chapman", "version": "1"}
    scenario = {
        "scenario_id": "held-out",
        "version": "1",
        "initial_state": {"O": {}, "O3": {}},
        "integration": {"checkpoints_seconds": [60.0, 600.0]},
    }
    result["manifest"] = {"id": "chapman", "version": "1", "sha256": canonical_sha256(manifest)}
    result["scenario"] = {"id": "held-out", "version": "1", "sha256": canonical_sha256(scenario)}
    request = dict(
        solver="mkpp",
        campaign_id=result["campaign_id"],
        run_id=result["run_id"],
        manifest=manifest,
        scenario=scenario,
        block=0,
        order_position=1,
    )
    return result, request


def test_complete_requested_evidence_passes_protocol_checks(requested_evidence):
    result, request = requested_evidence
    validate_requested_evidence(result, **request)


@pytest.mark.parametrize("field,value", [("campaign_id", "other"), ("run_id", "stale"), ("block", 2), ("order_position", 0)])
def test_rejects_stale_or_replayed_request_identity(requested_evidence, field, value):
    result, request = requested_evidence
    result[field] = value
    with pytest.raises(SchemaValidationError, match=field):
        validate_requested_evidence(result, **request)


@pytest.mark.parametrize("section", ["manifest", "scenario"])
def test_rejects_correct_name_with_wrong_input_hash(requested_evidence, section):
    result, request = requested_evidence
    result[section]["sha256"] = "0" * 64
    with pytest.raises(SchemaValidationError, match=section):
        validate_requested_evidence(result, **request)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_time", "missing_species", "extra_species"])
def test_rejects_partial_or_mislabeled_trajectories(requested_evidence, mutation):
    result, request = requested_evidence
    checkpoints = result["state"]["checkpoints"]
    if mutation == "missing":
        checkpoints.pop(0)
    elif mutation == "duplicate":
        checkpoints.append(deepcopy(checkpoints[-1]))
    elif mutation == "wrong_time":
        checkpoints[-1]["time_seconds"] = 1.0
    elif mutation == "missing_species":
        del checkpoints[0]["values"]["O3"]
    else:
        checkpoints[0]["values"]["unmapped"] = 1.0
    with pytest.raises(SchemaValidationError, match="checkpoint"):
        validate_requested_evidence(result, **request)


def test_failed_native_integration_is_never_protocol_success(requested_evidence):
    result, request = requested_evidence
    result.update(status="failed", diagnostics=["integration did not finish"])
    with pytest.raises(SchemaValidationError, match="status"):
        validate_requested_evidence(result, **request)
