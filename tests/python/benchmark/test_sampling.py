"""Deterministic collection-policy tests for paired benchmark samples."""

from __future__ import annotations

import json
from collections import Counter
from itertools import permutations
from pathlib import Path

from mkpp.benchmark.sampling import (
    INVALID_SAMPLE,
    append_raw_record,
    balanced_order,
    collect_paired_blocks,
    common_repetition_count,
    stable_warmup_count,
)


def test_seeded_balanced_order_is_repeatable_and_interleaved() -> None:
    first = balanced_order(("micm", "mkpp", "kpp"), blocks=6, seed=17)
    assert first == balanced_order(("micm", "mkpp", "kpp"), blocks=6, seed=17)
    assert set(first) == set(permutations(("micm", "mkpp", "kpp")))
    for position in range(3):
        assert set(Counter(order[position] for order in first).values()) == {2}
    predecessors = Counter(pair for order in first for pair in zip(order, order[1:]))
    assert set(predecessors.values()) == {2}


def test_pilot_doubles_common_work_until_three_run_median_is_long_enough() -> None:
    assert common_repetition_count(lambda count: (0.2 * count, 0.3 * count, 0.4 * count), initial=1) == 4


def test_warmup_requires_three_stable_samples_and_has_a_hard_limit() -> None:
    assert stable_warmup_count((10.0, 10.2, 10.1), minimum=3) == 3
    assert stable_warmup_count((1.0,) * 10, minimum=3) == 3
    assert stable_warmup_count((10.0, 10.0, 20.0) * 4, minimum=3) is None


def test_invalid_samples_are_labeled_for_recollection_not_discarded() -> None:
    assert INVALID_SAMPLE == "predeclared_invalid"


def test_invalid_paired_block_is_retained_then_recollected() -> None:
    calls: list[tuple[int, int, str]] = []

    def measure(block: int, attempt: int, solver: str) -> str | None:
        calls.append((block, attempt, solver))
        return None if block == 0 and attempt == 1 and solver == "mkpp" else f"{solver}-{attempt}"

    blocks = collect_paired_blocks(("micm", "mkpp"), blocks=1, seed=4, measure=measure)
    assert blocks[0].records == ("micm-2", "mkpp-2")
    assert blocks[0].invalid_records == (None, "micm-1")
    assert calls == [(0, 1, "mkpp"), (0, 1, "micm"), (0, 2, "mkpp"), (0, 2, "micm")]


def test_raw_records_are_append_only_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    append_raw_record(path, {"run": 1})
    append_raw_record(path, {"run": 2})
    assert [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] == [{"run": 1}, {"run": 2}]


def test_paired_pilot_rechecks_every_solver_at_the_final_common_count():
    from mkpp.benchmark.sampling import paired_repetition_count

    calls = []

    def measure(count):
        calls.append(count)
        # B passes at count 1 but changes at count 2, so max(independent
        # pilots) would incorrectly choose 2 after A's pilot succeeds.
        return {"A": (count / 2,) * 3, "B": ((0.2 if count == 2 else count),) * 3}

    assert paired_repetition_count(measure, maximum=8) == 4
    assert calls == [1, 2, 4]


def test_paired_pilot_rejects_missing_or_changed_subjects():
    import pytest
    from mkpp.benchmark.sampling import paired_repetition_count

    with pytest.raises(ValueError, match="subject identities"):
        paired_repetition_count(lambda count: {"A": (0.1,) * 3} if count == 1 else {"B": (1.0,) * 3})
