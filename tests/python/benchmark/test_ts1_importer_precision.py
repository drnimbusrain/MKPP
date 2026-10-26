"""The TS1 importer must preserve upstream coefficients without loading MUSICA."""

import ast
import math
from pathlib import Path

import pytest


def _diffusion_renderer():
    source = Path(__file__).resolve().parents[3] / "benchmarks/solver_comparison/sources/ts1/micm-importer/import_ts1.py"
    tree = ast.parse(source.read_text())
    # Execute the actual dependency-free rendering functions. Importing the full
    # importer would require the optional native MUSICA package during unit tests.
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"number", "diffusion_lines"}]
    namespace = {"math": math}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["diffusion_lines"]


def test_musica_large_phase_coefficient_is_preserved_exactly():
    lines = _diffusion_renderer()("GLYOXAL", 1e300)
    text = "\n".join(lines)
    assert "phase_species.emplace_back(species, 1e+300);" in text
    assert "1e-05" not in text
    assert "static_assert" in text
    assert "requires binary64 MICM" in text


@pytest.mark.parametrize("value", [0.0, -1.0, math.inf, math.nan])
def test_invalid_phase_coefficient_is_rejected(value):
    with pytest.raises(ValueError, match="finite and positive"):
        _diffusion_renderer()("GLYOXAL", value)
