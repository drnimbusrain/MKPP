import csv
from pathlib import Path

from scripts.import_cracmm2 import PHOTOLYSIS_RATE_NAMES, import_mechanism, rate_parameters


def test_arrhenius_rate_with_temperature_power_is_parsed() -> None:
    rate_type, parameters = rate_parameters("6.10E-34(T/300)<sup>-2.40</sup>", "R038", "CRACMM1")

    assert rate_type == "ARRHENIUS"
    assert parameters == {"A": 6.10e-34, "B": -2.40, "C": 0.0}


def test_complex_cracmm_rates_are_mapped_or_reported() -> None:
    mechanism_dir = Path(__file__).resolve().parents[2] / "mechanisms/openatmos/cracmm2/source"
    mechanism = import_mechanism(
        mechanism_dir / "cracmm2_metadata.csv",
        mechanism_dir / "cracmm2_rxn_metadata.csv",
    )
    reactions = {reaction["name"]: reaction for reaction in mechanism["reactions"]}

    assert reactions["R045"]["type"] == "EP3"
    assert reactions["R046"]["type"] == "EP3"
    troe_ids = {
        "R048",
        "R049",
        "R055",
        "R056",
        "R063",
        "R066",
        "R069",
        "R076",
        "R080",
        "R166",
        "R168",
    }
    assert {name for name, reaction in reactions.items() if reaction["type"] == "TROE"} == troe_ids
    assert all(reactions[name]["type"] == "TROE" for name in troe_ids)
    assert all(
        set(reactions[name]["k0"]) == {"A", "B", "C"}
        and set(reactions[name]["kinf"]) == {"A", "B", "C"}
        and reactions[name]["N"] == 1.0
        and reactions[name]["Fc"] == 0.6
        for name in troe_ids
    )
    assert reactions["R048"]["k0"] == {"A": 9.1e-32, "B": -1.5, "C": 0.0}
    assert reactions["R048"]["kinf"] == {"A": 3.0e-11, "B": 0.0, "C": 0.0}
    assert reactions["R057"]["type"] == "EP2"
    assert reactions["TRP57"]["type"] == "EP2"

    unconverted = mechanism["metadata"]["unconverted_rates"]
    unconverted_ids = {item["reaction_id"] for item in unconverted}
    unknown_ids = {name for name, reaction in reactions.items() if reaction["type"] == "UNKNOWN"}
    assert unconverted_ids == unknown_ids == {"R051", "R064", "R067", "R167", "R169", "HAL_Ozone"}
    assert all(item["reason"] and item["source_rate"] for item in unconverted)


def test_cracmm2_photolysis_rates_are_classified_and_named() -> None:
    mechanism_dir = Path(__file__).resolve().parents[2] / "mechanisms/openatmos/cracmm2/source"
    mechanism = import_mechanism(
        mechanism_dir / "cracmm2_metadata.csv",
        mechanism_dir / "cracmm2_rxn_metadata.csv",
    )
    reactions = mechanism["reactions"]
    photolysis = [reaction for reaction in reactions if reaction["type"] == "PHOTOLYSIS"]

    for reaction in reactions:
        for side in ("reactants", "products"):
            assert isinstance(reaction[side], list)
            assert all(
                set(term) == {"coefficient", "species name"}
                and isinstance(term["coefficient"], float)
                and isinstance(term["species name"], str)
                for term in reaction[side]
            )

    assert len(PHOTOLYSIS_RATE_NAMES) == 32
    assert len(photolysis) == 41
    assert all(reaction["name"].startswith("J_") for reaction in photolysis)
    assert all(reaction["A"] == reaction["name"] for reaction in photolysis)
    assert {
        reaction["name"] for reaction in photolysis
    } >= {"J_R001", "J_R014a", "J_TRP55", "J_TRP56", "J_T20"}
    assert mechanism["concentration_unit"] == "molecule cm-3"
    assert mechanism["metadata"]["photolysis_rate_source"].endswith("CSQY_DATA_cracmm2")

    with (mechanism_dir / "cracmm2_rxn_metadata.csv").open(newline="", encoding="utf-8") as stream:
        source_rows = list(csv.DictReader(stream))
    forcing = mechanism["metadata"]["photolysis_host_forcing_expressions"]
    for row in source_rows:
        if row["rate_constant"].strip() in PHOTOLYSIS_RATE_NAMES:
            assert forcing[f"J_{row['reaction_id'].strip()}"] == row["rate_constant"]
