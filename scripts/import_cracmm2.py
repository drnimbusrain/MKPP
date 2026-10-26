#!/usr/bin/env python3
"""Import the EPA CRACMM2 reaction and species tables into OpenAtmos JSON.

The importer consumes the pinned CSV files published by USEPA/CRACMM.  CMAQ
rate expressions that do not have an exact OpenAtmos representation remain
explicit named host forcings; they are never replaced by guessed numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

SOURCE_COMMIT = "8a71fdbfb2183a1e37a53eb237e265d12f09b245"
PHOTOLYSIS_RATE_SOURCE = "https://github.com/USEPA/CMAQ/blob/main/CCTM/src/MECHS/cracmm2/CSQY_DATA_cracmm2"
# Rates in the CRACMM2 CSQY_DATA table are photolysis rates, even when the
# reaction metadata omits hv from its reactants.
PHOTOLYSIS_RATE_NAMES = frozenset(
    {
        "O3O3P_NASA06",
        "O3O1D_NASA06",
        "H2O2_RACM2",
        "NO2_RACM2",
        "NO3NO_RACM2",
        "NO3NO2_RACM2",
        "HONO_RACM2",
        "HNO3_RACM2",
        "HNO4_RACM2",
        "HCHO_MOL_JPL19",
        "HCHO_RAD_JPL19",
        "CH3CHO_RACM2",
        "ALD_JPL19",
        "CH3COCH3A_JPL19",
        "CH3COCH3B_JPL19",
        "UALD_RACM2",
        "MEK_JGR19",
        "KET_JGR19",
        "HKET_RACM2",
        "MACR_RACM2",
        "MVK_RACM2",
        "GLYH2_RACM2",
        "GLYF_RACM2",
        "GLYHX_RACM2",
        "MGLY_RACM2",
        "BALD_RACM2",
        "OP1_RACM2",
        "PAA_RACM2",
        "ONIT_RACM2",
        "PAN1_RACM2",
        "PAN2_RACM2",
        "ACRO_09",
    }
)
TERM = re.compile(r"(?P<sign>[+-]?)\s*(?:(?P<coef>\d+(?:\.\d*)?|\.\d+)(?:\s*\*\s*)?)?(?P<name>[A-Za-z][A-Za-z0-9_]*)")
NUMBER = re.compile(r"^\s*[0-9.+-Ee]+\s*$")
NUMBER_VALUE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
ARRHENIUS_A = re.compile(rf"^\s*({NUMBER_VALUE})")
ARRHENIUS_EXPONENT = re.compile(rf"exp\(\s*({NUMBER_VALUE})/T\s*\)", re.I)
ARRHENIUS_TEMPERATURE = re.compile(rf"\(T/300\)\^({NUMBER_VALUE})", re.I)
RATE_FACTOR = re.compile(
    rf"^\s*({NUMBER_VALUE})(?:\s*\*\s*)?(?:exp\(\s*({NUMBER_VALUE})/T\s*\))?" rf"(?:\(T/300\)\^({NUMBER_VALUE}))?\s*$",
    re.I,
)
RATE_CHANNEL = re.compile(rf"k_?([0-9oi])\s*=\s*({NUMBER_VALUE}[^<\n]*)", re.I)
FALLOFF_SHAPE = re.compile(rf"n\s*=\s*({NUMBER_VALUE})\s*;\s*F\s*=\s*({NUMBER_VALUE})", re.I)
UNCONVERTED_RATE_REASONS = {
    "R051": (
        "Source marks this as a special rate expression; its three coefficients do not map unambiguously to a supported "
        "standard rate law."
    ),
    "R064": (
        "Reverse rate is defined as a temperature-dependent multiplier of the R063 falloff rate, not a standalone "
        "supported rate law."
    ),
    "R067": (
        "Reverse rate is defined as a temperature-dependent multiplier of the R066 falloff rate, not a standalone "
        "supported rate law."
    ),
    "R167": (
        "Reverse rate is defined as a temperature-dependent multiplier of the R166 falloff rate, not a standalone "
        "supported rate law."
    ),
    "R169": (
        "Reverse rate is defined as a temperature-dependent multiplier of the R168 falloff rate, not a standalone "
        "supported rate law."
    ),
    "HAL_Ozone": "Rate depends on seawater and a pressure-dependent minimum expression not represented by supported rate types.",
}


def arrhenius_factor(value: str) -> dict[str, float] | None:
    match = RATE_FACTOR.fullmatch(value.strip())
    if match is None:
        return None
    return {
        "A": float(match.group(1)),
        "B": float(match.group(3) or 0.0),
        "C": float(match.group(2) or 0.0),
    }


def structured_rate_parameters(rate: str, reaction_id: str) -> tuple[str, dict[str, object]] | None:
    clean = rate.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
    clean = clean.replace("<sub>o</sub>", "o").replace("<sub>i</sub>", "i")
    clean = clean.replace("<sub>0</sub>", "0").replace("<sub>1</sub>", "1")
    clean = clean.replace("<sub>2</sub>", "2").replace("<sub>3</sub>", "3")
    clean = clean.replace("<sup>", "^").replace("</sup>", "")
    clean = re.sub(r"<[^>]+>", "", clean).replace("\\*", "*")
    channels = {key.lower(): value.strip() for key, value in RATE_CHANNEL.findall(clean)}

    falloff_shape = FALLOFF_SHAPE.search(clean)
    if falloff_shape and "o" in channels and "i" in channels:
        low_pressure = arrhenius_factor(channels["o"])
        high_pressure = arrhenius_factor(channels["i"])
        if low_pressure is not None and high_pressure is not None:
            return (
                "TROE",
                {
                    "k0": low_pressure,
                    "kinf": high_pressure,
                    "N": float(falloff_shape.group(1)),
                    "Fc": float(falloff_shape.group(2)),
                },
            )

    if "0" in channels and "1" in channels and len(channels) == 2:
        first_channel = arrhenius_factor(channels["0"])
        second_channel = arrhenius_factor(channels["1"])
        if first_channel is not None and second_channel is not None:
            return (
                "EP3",
                {
                    "A1": first_channel["A"],
                    "C1": first_channel["C"],
                    "A2": second_channel["A"],
                    "C2": second_channel["C"],
                },
            )

    second_channel_key = {"R057": "1", "TRP57": "2"}.get(reaction_id)
    if second_channel_key and {"0", second_channel_key, "3"} <= channels.keys() and len(channels) == 3:
        channels_parsed = {key: arrhenius_factor(value) for key, value in channels.items()}
        if all(value is not None for value in channels_parsed.values()):
            return (
                "EP2",
                {
                    "A0": channels_parsed["0"]["A"],
                    "C0": channels_parsed["0"]["C"],
                    "A2": channels_parsed[second_channel_key]["A"],
                    "C2": channels_parsed[second_channel_key]["C"],
                    "A3": channels_parsed["3"]["A"],
                    "C3": channels_parsed["3"]["C"],
                },
            )

    return None


def terms(value: str) -> list[dict[str, object]]:
    result: dict[str, float] = {}
    for match in TERM.finditer(value.replace(" ", "")):
        name = match.group("name")
        if name in {"hv", "DELTA_C", "DELTA_N", "DELTA_SI"}:
            continue
        coefficient = float(match.group("coef") or 1.0)
        if match.group("sign") == "-":
            coefficient = -coefficient
        result[name] = result.get(name, 0.0) + coefficient
    return [{"coefficient": coefficient, "species name": name} for name, coefficient in result.items() if coefficient > 0.0]


def rate_parameters(rate: str, reaction_id: str, family: str) -> tuple[str, dict[str, object]]:
    if reaction_id == "R051":
        return "UNKNOWN", {"rate_name": f"K_{reaction_id}", "source_rate": rate}
    structured = structured_rate_parameters(rate, reaction_id)
    if structured is not None:
        return structured
    clean = rate.replace("<sup>", "^").replace("</sup>", "")
    clean = re.sub(r"<[^>]+>", "", clean).replace("\\*", "*").replace(" ", "")
    if family == "mixed":
        return "HETEROGENEOUS", {"rate_name": f"K_{reaction_id}"}
    if NUMBER.match(clean):
        return "ARRHENIUS", {"A": float(clean), "B": 0.0, "C": 0.0}

    coefficient = ARRHENIUS_A.match(clean)
    if coefficient:
        remainder = clean[coefficient.end() :].lstrip("*")
        exponent_matches = list(ARRHENIUS_EXPONENT.finditer(remainder))
        temperature_matches = list(ARRHENIUS_TEMPERATURE.finditer(remainder))
        if len(exponent_matches) <= 1 and len(temperature_matches) <= 1:
            for match in sorted(exponent_matches + temperature_matches, key=lambda item: item.start(), reverse=True):
                remainder = remainder[: match.start()] + remainder[match.end() :]
            remainder = remainder.replace("*", "")
            if not remainder:
                return (
                    "ARRHENIUS",
                    {
                        "A": float(coefficient.group(1)),
                        "B": float(temperature_matches[0].group(1)) if temperature_matches else 0.0,
                        "C": float(exponent_matches[0].group(1)) if exponent_matches else 0.0,
                    },
                )
    return "UNKNOWN", {"rate_name": f"K_{reaction_id}", "source_rate": rate}


def load_species(path: Path) -> list[dict[str, object]]:
    species: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            name = row["Species"].strip()
            phase = row["Phase"].strip().upper()
            species.append(
                {
                    "name": name,
                    "phase": "aerosol" if phase == "P" else "gas",
                    "role": "fixed" if name in {"M", "AIR", "XO2", "DELTA_C", "DELTA_N", "DELTA_SI"} else "variable",
                }
            )
    names = {entry["name"] for entry in species}
    for name in ("M", "AIR"):
        if name not in names:
            species.append({"name": name, "phase": "gas", "role": "fixed"})
    return species


def import_mechanism(species_path: Path, reactions_path: Path) -> dict[str, object]:
    species = load_species(species_path)
    species_names = {entry["name"] for entry in species}
    reactions: list[dict[str, object]] = []
    forcing: dict[str, str] = {}
    unconverted_rates: list[dict[str, str]] = []
    with reactions_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            reaction_id = row["reaction_id"].strip()
            reactant_text = row["reactants"].strip()
            rate_name = row["rate_constant"].strip()
            reaction_type, parameters = rate_parameters(row["rate_constant"], reaction_id, row["reaction_phase"].strip())
            is_photolysis = rate_name in PHOTOLYSIS_RATE_NAMES or "+ hv" in reactant_text.lower()
            if is_photolysis:
                reaction_type = "PHOTOLYSIS"
                parameters = {"A": f"J_{reaction_id}"}
                forcing[f"J_{reaction_id}"] = row["rate_constant"]
                reactant_text = re.sub(r"\s*\+\s*hv", "", reactant_text, flags=re.I)
            if reaction_type == "UNKNOWN":
                unconverted_rates.append(
                    {
                        "reaction_id": reaction_id,
                        "source_rate": row["rate_constant"],
                        "reason": UNCONVERTED_RATE_REASONS.get(
                            reaction_id,
                            "No supported standard rate-law parser matched the source expression.",
                        ),
                    }
                )
            reactants = terms(reactant_text)
            products = terms(row["products"])
            for term in (*reactants, *products):
                name = str(term["species name"])
                if name not in species_names:
                    species.append({"name": name, "phase": "gas", "role": "fixed"})
                    species_names.add(name)
            reactions.append(
                {
                    "name": f"J_{reaction_id}" if is_photolysis else reaction_id,
                    "type": reaction_type,
                    "reactants": reactants,
                    "products": products,
                    **parameters,
                    "source_family": row["reaction_family"],
                    "continuous_transition": reaction_type == "PHOTOLYSIS",
                }
            )
    return {
        "name": "cracmm2",
        "description": "Community Regional Atmospheric Chemistry Multiphase Mechanism CRACMM2 from USEPA.",
        "concentration_unit": "molecule cm-3",
        "species": species,
        "phases": [{"name": "gas"}, {"name": "aerosol"}],
        "reactions": reactions,
        "metadata": {
            "source": "https://github.com/USEPA/CRACMM",
            "source_commit": SOURCE_COMMIT,
            "source_species_table": "metadata/cracmm2/cracmm2_metadata.csv",
            "source_reaction_table": "chemistry/cracmm2/cracmm2_rxn_metadata.csv",
            "photolysis_rate_source": PHOTOLYSIS_RATE_SOURCE,
            "photolysis_host_forcing_expressions": forcing,
            "unconverted_rates": unconverted_rates,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--species", type=Path, required=True)
    parser.add_argument("--reactions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = import_mechanism(args.species, args.reactions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, sort_keys=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
