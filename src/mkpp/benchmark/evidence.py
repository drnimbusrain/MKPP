"""Bind a runner's claims to the requested scientific workload.

Passing these checks establishes protocol consistency, not scientific admission.
Chemistry audits, reference convergence, and held-out accuracy remain required.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mkpp.benchmark.manifest import canonical_sha256
from mkpp.benchmark.schema import SchemaValidationError, validate_runner_result


def validate_requested_evidence(
    result: Mapping[str, Any],
    *,
    solver: str,
    campaign_id: str,
    run_id: str,
    manifest: Mapping[str, Any],
    scenario: Mapping[str, Any],
    block: int,
    order_position: int,
) -> None:
    """Reject relabeled, partial, or unsuccessful native results.

    Parameters
    ----------
    result : Mapping
        Native result, without any orchestrator substitutions.
    solver, campaign_id, run_id : str
        Requested identities.
    manifest, scenario : Mapping
        Resolved immutable scientific inputs.
    block, order_position : int
        Requested measurement position.

    Returns
    -------
    None
        Raises ``SchemaValidationError`` on mismatched evidence.
    """
    validate_runner_result(result)
    expected = {"campaign_id": campaign_id, "run_id": run_id, "block": block, "order_position": order_position}
    for name, value in expected.items():
        if result[name] != value:
            raise SchemaValidationError(f"{name}: runner does not match requested identity")
    if result["solver"]["id"] != solver:
        raise SchemaValidationError("solver.id: runner identity mismatch")
    for name, document, identity in (
        ("manifest", manifest, manifest["mechanism_id"]),
        ("scenario", scenario, scenario["scenario_id"]),
    ):
        expected_identity = {"id": identity, "version": document["version"], "sha256": canonical_sha256(document)}
        if dict(result[name]) != expected_identity:
            raise SchemaValidationError(f"{name}: runner identity/hash does not match requested input")
    if result["status"] != "success":
        raise SchemaValidationError("status: runner did not report successful integration")
    checkpoints = result["state"]["checkpoints"]
    if [item["time_seconds"] for item in checkpoints] != list(scenario["integration"]["checkpoints_seconds"]):
        raise SchemaValidationError("state.checkpoints: runner omitted or changed requested checkpoint times")
    for checkpoint in checkpoints:
        if set(checkpoint["values"]) != set(scenario["initial_state"]):
            raise SchemaValidationError("state.checkpoints: runner species do not match requested named state")


__all__ = ["validate_requested_evidence"]
