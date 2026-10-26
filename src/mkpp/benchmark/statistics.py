"""Robust, reproducible timing summaries that retain every valid observation."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RobustSummary:
    samples: tuple[float, ...]
    median: float
    iqr: float
    mad: float


@dataclass(frozen=True, slots=True)
class RatioSummary:
    raw_ratios: tuple[float, ...]
    median_ratio: float
    bootstrap_interval: tuple[float, float]


def _validated(samples: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in samples)
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("samples must be non-empty, finite, and positive")
    return values


def _quantile(sorted_values: tuple[float, ...], fraction: float) -> float:
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def robust_summary(samples: Sequence[float]) -> RobustSummary:
    """Summarize the full distribution; outliers are deliberately not removed."""

    values = _validated(samples)
    ordered = tuple(sorted(values))
    median = statistics.median(ordered)
    deviations = tuple(sorted(abs(value - median) for value in ordered))
    return RobustSummary(values, median, _quantile(ordered, 0.75) - _quantile(ordered, 0.25), statistics.median(deviations))


def bootstrap_median_interval(
    samples: Sequence[float],
    *,
    seed: int,
    resamples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Return a seeded percentile interval for the median using 10,000 draws by default."""

    values = _validated(samples)
    if resamples != 10_000 or not 0.0 < confidence < 1.0:
        raise ValueError("bootstrap requires 10,000 resamples and a valid confidence")
    generator = random.Random(seed)
    medians = sorted(statistics.median(generator.choices(values, k=len(values))) for _ in range(resamples))
    tail = (1.0 - confidence) / 2.0
    return _quantile(tuple(medians), tail), _quantile(tuple(medians), 1.0 - tail)


def paired_ratio_summary(
    baseline_seconds: Sequence[float],
    candidate_seconds: Sequence[float],
    *,
    seed: int,
) -> RatioSummary:
    """Calculate baseline/candidate ratios only for block-aligned observations."""

    baseline = _validated(baseline_seconds)
    candidate = _validated(candidate_seconds)
    if len(baseline) != len(candidate):
        raise ValueError("paired timing series must contain the same number of blocks")
    ratios = tuple(left / right for left, right in zip(baseline, candidate, strict=True))
    return RatioSummary(ratios, statistics.median(ratios), bootstrap_median_interval(ratios, seed=seed))


def assess_repeatability(first_throughputs: Sequence[float], second_throughputs: Sequence[float]) -> bool:
    """Assess repeat campaigns against the published five-percent median rule."""

    if not first_throughputs or not second_throughputs:
        return False
    if any(not math.isfinite(value) or value <= 0 for value in (*first_throughputs, *second_throughputs)):
        return False
    first = statistics.median(first_throughputs)
    second = statistics.median(second_throughputs)
    return first > 0.0 and second > 0.0 and abs(second - first) / first <= 0.05


__all__ = [
    "RatioSummary",
    "RobustSummary",
    "assess_repeatability",
    "bootstrap_median_interval",
    "paired_ratio_summary",
    "robust_summary",
]
