"""Deterministic sampling policy primitives for fair paired measurements."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import TypeVar

INVALID_SAMPLE = "predeclared_invalid"
_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class CollectedBlock:
    """One valid paired block plus earlier invalid evidence that triggered recollection."""

    block_id: int
    order: tuple[object, ...]
    records: tuple[object, ...]
    invalid_records: tuple[object, ...] = ()


def balanced_order(solvers: Sequence[_T], *, blocks: int, seed: int) -> tuple[tuple[_T, ...], ...]:
    """Balance positions and predecessor effects using shuffled permutation cycles."""

    if not solvers or len(set(solvers)) != len(solvers):
        raise ValueError("solvers must be a non-empty unique sequence")
    if blocks <= 0:
        raise ValueError("blocks must be positive")
    if len(solvers) > 3:
        raise ValueError("solver comparison supports at most three subjects")
    generator = random.Random(seed)
    orders = list(permutations(solvers))
    collected: list[tuple[_T, ...]] = []
    while len(collected) < blocks:
        generator.shuffle(orders)
        collected.extend(orders)
    return tuple(collected[:blocks])


def common_repetition_count(
    measure: Callable[[int], Sequence[float]],
    *,
    initial: int = 1,
    minimum_seconds: float = 1.0,
    maximum: int = 1 << 30,
) -> int:
    """Double shared work until its three-observation median reaches the policy floor."""

    if initial <= 0 or not math.isfinite(minimum_seconds) or minimum_seconds <= 0 or maximum < initial:
        raise ValueError("invalid pilot bounds")
    repetitions = initial
    while repetitions <= maximum:
        samples = tuple(measure(repetitions))
        if len(samples) != 3 or any(not math.isfinite(value) or value < 0.0 for value in samples):
            raise ValueError("pilot must return three finite non-negative durations")
        if statistics.median(samples) >= minimum_seconds:
            return repetitions
        repetitions *= 2
    raise RuntimeError("pilot did not reach minimum sample duration")


def paired_repetition_count(
    measure: Callable[[int], dict[_T, Sequence[float]]],
    *,
    minimum_seconds: float = 1.0,
    maximum: int = 1 << 30,
) -> int:
    """Pilot every subject at each common count; require every subject to pass.

    Independent pilots followed by a maximum assume runtime is monotonic. This
    procedure observes every subject at the actual final shared count instead.
    """
    if not math.isfinite(minimum_seconds) or minimum_seconds <= 0 or maximum < 1:
        raise ValueError("invalid pilot bounds")
    repetitions = 1
    subjects: set[_T] | None = None
    while repetitions <= maximum:
        samples = measure(repetitions)
        if not samples or (subjects is not None and set(samples) != subjects):
            raise ValueError("pilot subject identities must be non-empty and constant")
        subjects = set(samples)
        for durations in samples.values():
            if len(durations) != 3 or any(not math.isfinite(value) or value < 0 for value in durations):
                raise ValueError("each pilot subject must return three finite non-negative durations")
        if all(statistics.median(durations) >= minimum_seconds for durations in samples.values()):
            return repetitions
        repetitions *= 2
    raise RuntimeError("paired pilot did not reach minimum sample duration")


def stable_warmup_count(
    throughputs: Iterable[float],
    *,
    minimum: int = 3,
    maximum: int = 10,
    relative_range: float = 0.03,
) -> int | None:
    """Return first stable warmup count, or ``None`` after the declared limit."""

    if minimum < 3 or maximum < minimum or relative_range < 0:
        raise ValueError("invalid warmup policy")
    observations = tuple(throughputs)[:maximum]
    for count in range(minimum, len(observations) + 1):
        recent = observations[count - 3 : count]
        median = statistics.median(recent)
        if median > 0 and all(math.isfinite(value) for value in recent) and (max(recent) - min(recent)) / median <= relative_range:
            return count
    return None


def collect_paired_blocks(
    solvers: Sequence[_T],
    *,
    blocks: int,
    seed: int,
    measure: Callable[[int, int, _T], object | None],
    maximum_attempts: int = 10,
) -> tuple[CollectedBlock, ...]:
    """Collect complete blocks, retaining predeclared invalid evidence before retrying."""

    if maximum_attempts <= 0:
        raise ValueError("maximum attempts must be positive")
    orders = balanced_order(solvers, blocks=blocks, seed=seed)
    collected: list[CollectedBlock] = []
    for block_id, order in enumerate(orders):
        invalid_records: list[object] = []
        for attempt in range(1, maximum_attempts + 1):
            records_by_solver: dict[_T, object] = {}
            failed = False
            for solver in order:
                record = measure(block_id, attempt, solver)
                if record is None:
                    failed = True
                else:
                    records_by_solver[solver] = record
            if failed:
                invalid_records.extend(records_by_solver.get(solver) for solver in order)
                continue
            collected.append(
                CollectedBlock(
                    block_id,
                    tuple(order),
                    tuple(records_by_solver[solver] for solver in solvers),
                    tuple(invalid_records),
                )
            )
            break
        else:
            raise RuntimeError(f"block {block_id} exhausted recollection attempts")
    return tuple(collected)


def append_raw_record(path: Path, record: object) -> None:
    """Append one immutable JSON evidence record; never rewrite prior evidence."""

    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
        output.write("\n")


__all__ = [
    "CollectedBlock",
    "INVALID_SAMPLE",
    "append_raw_record",
    "balanced_order",
    "collect_paired_blocks",
    "common_repetition_count",
    "paired_repetition_count",
    "stable_warmup_count",
]
