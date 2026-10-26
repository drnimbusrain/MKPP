"""Validate declared performance controls before native processes are started."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def _integer(value: Any, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def validate_performance_policy(campaign: Mapping[str, Any], solver_counts: Sequence[int]) -> None:
    """Reject contradictory sampling, ordering, timing, and publication controls.

    A campaign without sampling remains a diagnostic smoke description. Passing
    this validator does not execute or establish any scientific/hardware gate.
    """
    boundaries = campaign.get("timing_boundaries")
    if boundaries is not None:
        if (
            not isinstance(boundaries, list)
            or not boundaries
            or any(value not in {"steady_state_solve", "lifecycle"} for value in boundaries)
            or len(set(boundaries)) != len(boundaries)
        ):
            raise ValueError("timing_boundaries must contain unique supported boundaries")
    sampling = campaign.get("sampling")
    if sampling is None:
        return
    if not isinstance(sampling, Mapping):
        raise ValueError("sampling must be an object")
    if _integer(sampling.get("pilot_iterations"), "pilot_iterations", 3) != 3:
        raise ValueError("pilot_iterations must be exactly 3")
    seconds = sampling.get("minimum_sample_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, int | float) or not math.isfinite(seconds) or seconds < 1.0:
        raise ValueError("minimum_sample_seconds must be finite and >= 1")
    minimum = _integer(sampling.get("minimum_valid_blocks"), "minimum_valid_blocks", 10)
    blocks = _integer(sampling.get("paired_blocks"), "paired_blocks", minimum)
    for name in ("ordering_seed", "bootstrap_seed"):
        _integer(sampling.get(name), name, 0)
    if sampling.get("bootstrap_resamples") != 10_000:
        raise ValueError("bootstrap_resamples must be 10000")
    warmups = sampling.get("warmups")
    if not isinstance(warmups, Mapping):
        raise ValueError("warmups must be an object")
    low = _integer(warmups.get("minimum"), "warmups.minimum", 3)
    _integer(warmups.get("maximum"), "warmups.maximum", low)
    relative = warmups.get("stable_last_three_relative_range")
    if isinstance(relative, bool) or not isinstance(relative, int | float) or not math.isfinite(relative) or not 0 <= relative <= 1:
        raise ValueError("warmup relative range must be finite and between 0 and 1")
    publication = campaign.get("publication", {})
    if not isinstance(publication, Mapping):
        raise ValueError("publication must be an object")
    for name in ("require_scientific_eligibility", "preserve_failures", "no_outlier_discard", "separate_timing_boundaries"):
        if name in publication and publication[name] is not True:
            raise ValueError(f"publication.{name} cannot be disabled")
    if publication.get("require_complete_order_cycles"):
        if any(blocks % math.factorial(count) for count in solver_counts):
            raise ValueError("paired_blocks must complete every solver permutation cycle")


__all__ = ["validate_performance_policy"]
