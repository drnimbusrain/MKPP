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
TERM = re.compile(r"(?P<sign>[+-]?)\s*(?:(?P<coef>\d+(?:\.\d*)?|\.\d+)(?:\s*\*\s*)?)?(?P<name>[A-Za-z][A-Za-z0-9_]*)")
ARRHENIUS = re.compile(r"^\s*([0-9.+-Ee]+)\s*\*?\s*exp\s*\(\s*([+-]?[0-9.]+)\s*/\s*T\s*\)\s*$", re.I)
NUMBER = re.compile(r"^\s*[0-9.+-Ee]+\s*$")


def terms(value: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for match in TERM.finditer(value.replace(" ", "")):
        name = match.group("name")
        if name in {"hv", "DELTA_C", "DELTA_N", "DELTA_SI"}:
            continue
        coefficient = float(match.group("coef") or 1.0)
        if match.group("sign") == "-":
            coefficient = -coefficient
        result[name] = result.get(name, 0.0) + coefficient
    return {name: value for name, value in result.items() if value > 0.0}


def rate_parameters(rate: str, reaction_id: str, family: str) -> tuple[str, dict[str, object]]:
    clean = re.sub(r"<[^>]+>", "", rate).replace("\\*", "*").replace(" ", "")
    if family == "mixed":
        return "HETEROGENEOUS", {"rate_name": f"K_{reaction_id}"}
    match = ARRHENIUS.match(clean)
    if match:
        return "ARRHENIUS", {"A": float(match.group(1)), "B": 0.0, "C": float(match.group(2))}
    if NUMBER.match(clean):
        return "ARRHENIUS", {"A": float(clean), "B": 0.0, "C": 0.0}
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
    with reactions_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            reaction_id = row["reaction_id"].strip()
            reactant_text = row["reactants"].strip()
            reaction_type, parameters = rate_parameters(row["rate_constant"], reaction_id, row["reaction_phase"].strip())
            if "+ hv" in reactant_text.lower():
                reaction_type = "PHOTOLYSIS"
                parameters = {"A": f"J_{reaction_id}"}
                forcing[f"J_{reaction_id}"] = row["rate_constant"]
                reactant_text = re.sub(r"\s*\+\s*hv", "", reactant_text, flags=re.I)
            reactants = terms(reactant_text)
            products = terms(row["products"])
            for name in (*reactants, *products):
                if name not in species_names:
                    species.append({"name": name, "phase": "gas", "role": "fixed"})
                    species_names.add(name)
            reactions.append(
                {
                    "name": reaction_id,
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
        "species": species,
        "phases": [{"name": "gas"}, {"name": "aerosol"}],
        "reactions": reactions,
        "metadata": {
            "source": "https://github.com/USEPA/CRACMM",
            "source_commit": SOURCE_COMMIT,
            "source_species_table": "metadata/cracmm2/cracmm2_metadata.csv",
            "source_reaction_table": "chemistry/cracmm2/cracmm2_rxn_metadata.csv",
            "photolysis_host_forcing_expressions": forcing,
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