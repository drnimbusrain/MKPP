"""Numerical checks for compact sparse storage and factorization traversal."""

import numpy as np
import pytest
from mkpp.parser import load_mechanism
from mkpp.template_context import build_template_context


@pytest.mark.parametrize("mechanism", ["chapman", "gocart"])
def test_compact_plan_solves_diagonally_dominant_systems(mechanism):
    context = build_template_context(load_mechanism(f"mechanisms/openatmos/{mechanism}/mechanism.json"))
    plan = context["compact"]
    n = context["num_species"]
    rng = np.random.default_rng(14)
    for _ in range(5):
        matrix = np.zeros((n, n))
        for i, j in plan["w_locations"]:
            matrix[i, j] = rng.uniform(-1, 1)
        matrix[np.diag_indices(n)] = np.sum(np.abs(matrix), axis=1) + 1
        w = [matrix[i, j] for i, j in plan["w_locations"]]
        lu = np.full(len(plan["lu_slots"]), np.nan)
        for op in plan["operations"]:
            value = w[op["w"]] if op["w"] >= 0 else 0.0
            for left, right in plan["products"][op["begin"] : op["end"]]:
                value -= lu[left] * lu[right]
            lu[op["output"]] = value / lu[op["diagonal"]] if op["diagonal"] >= 0 else value
        rhs = rng.normal(size=n)
        work = rhs.copy()
        solution = np.zeros(n)
        for row in range(n):
            for column, slot in plan["lower"][plan["lower_offsets"][row] : plan["lower_offsets"][row + 1]]:
                work[row] -= lu[slot] * work[column]
        for row in reversed(range(n)):
            solution[row] = work[row]
            for column, slot in plan["upper"][plan["upper_offsets"][row] : plan["upper_offsets"][row + 1]]:
                solution[row] -= lu[slot] * solution[column]
            solution[row] /= lu[plan["diagonals"][row]]
        np.testing.assert_allclose(matrix @ solution, rhs, rtol=1e-12, atol=1e-12)
        assert np.isfinite(lu).all()


def test_environment_symbols_are_preserved_for_runtime_drivers():
    from mkpp.format_eqn import format_eqn

    expression = format_eqn("Temp + M_density", [], keep_env_symbols=True)
    assert "AirDensity" in expression
    assert "Temp" in expression
