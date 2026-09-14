"""Source-backed admission facts for the extended benchmark mechanisms."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def mechanism(name: str) -> dict[str, object]:
    return json.loads((ROOT / "mechanisms" / "openatmos" / name / "mechanism.json").read_text(encoding="utf-8"))


def test_small_strato_preserves_background_species_and_sun_photolysis() -> None:
    document = mechanism("small_strato")
    species = {entry["name"]: entry for entry in document["species"]}  # type: ignore[index]
    reactions = document["reactions"]  # type: ignore[index]

    assert {"M", "O2"} <= set(species)
    forcing = document["metadata"]["photolysis_host_forcing_expressions"]  # type: ignore[index]
    assert any("SUN" in str(expression) for expression in forcing.values())
    assert all(reaction["type"] in {"ARRHENIUS", "PHOTOLYSIS"} for reaction in reactions)  # type: ignore[index]


def test_saprc99_source_declares_supported_complex_rate_laws_and_fixed_background() -> None:
    document = mechanism("saprc99")
    species = document["species"]  # type: ignore[index]
    reaction_types = {reaction["type"] for reaction in document["reactions"]}  # type: ignore[index]

    fixed = {entry["name"] for entry in species if entry.get("type") == "FIXED"}  # type: ignore[index]
    assert {"M", "N2", "RO2"} <= fixed
    assert {"ARRHENIUS", "PHOTOLYSIS", "TROE", "EP2", "EP3"} <= reaction_types
    assert "FALL" not in reaction_types


def test_generated_micm_bindings_are_sourced_from_openatmos_and_keep_ep_lambdas() -> None:
    small_strato = (ROOT / "benchmarks/solver_comparison/runners/micm/generated/small_strato.hpp").read_text(encoding="utf-8")
    saprc99 = (ROOT / "benchmarks/solver_comparison/runners/micm/generated/saprc99.hpp").read_text(encoding="utf-8")
    assert "Generated from canonical OpenAtmos JSON" in small_strato
    assert "Generated from canonical OpenAtmos JSON" in saprc99
    assert "LambdaRateConstantParameters" in saprc99
    assert "c.air_density_" in saprc99


def test_cracmm2_is_pinned_to_epa_source_and_preserves_multiphase_catalog() -> None:
    document = mechanism("cracmm2")
    assert document["metadata"]["source_commit"] == "8a71fdbfb2183a1e37a53eb237e265d12f09b245"
    assert len(document["species"]) == 271
    assert len(document["reactions"]) == 531
    assert {reaction["type"] for reaction in document["reactions"]} >= {
        "ARRHENIUS",
        "HETEROGENEOUS",
        "PHOTOLYSIS",
        "UNKNOWN",
    }
    assert len(document["metadata"]["photolysis_host_forcing_expressions"]) == 2
    assert {entry["phase"] for entry in document["species"]} >= {"gas", "aerosol"}
