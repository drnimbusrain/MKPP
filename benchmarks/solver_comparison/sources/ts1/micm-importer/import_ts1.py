#!/usr/bin/env python3
"""Generate the TS1 benchmark mechanism header from the musica configuration.

micm has no mechanism-configuration parser, so this script uses the musica
Python API to read MOZART-TS1 and write the governed solver-comparison binding.
every species and every reaction spelled out as ordinary micm types. The
generated header is checked in, so building and running the benchmark needs
neither musica nor network access.

    pip install musica
    benchmarks/solver_comparison/sources/ts1/micm-importer/import_ts1.py

musica ships the TS1 configuration inside the package, so no arguments are
needed. Pass --source to read a different directory, for example a musica
checkout you are editing.

The generated header holds three plain functions plus the benchmark mechanism:

  CreateGasPhase()                  -> micm::Phase
  CreateProcesses(const micm::Phase&) -> std::vector<micm::Process>
  PhaseSpeciesByName(...)           -> const micm::PhaseSpecies&
  struct Ts1                        -- Build() and InitState() for the harness

CreateGasPhase and CreateProcesses are not templates, so every solver
configuration shares one definition of the mechanism.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

try:
    import pandas as pd
    from musica import mechanism_configuration as mc
    from musica.mechanism_configuration import Arrhenius, Photolysis, Surface, Troe, UserDefined
    from musica.utils import find_config_path
except ImportError as error:  # pragma: no cover - a setup problem, not a logic one
    raise SystemExit(f"{error}. Install the musica Python package with 'pip install musica'.")

GAS_PHASE = "gas"

ARRHENIUS_PARAMETERS = ("A", "B", "C", "D", "E")
TROE_PARAMETERS = ("k0_A", "k0_B", "k0_C", "kinf_A", "kinf_B", "kinf_C", "Fc", "N")


def number(value: float) -> str:
    """Format a float as a C++ double literal that round-trips exactly."""
    value = float(value)
    if not math.isfinite(value):
        raise SystemExit(f"cannot write the non-finite value {value!r} as a C++ literal")
    text = repr(value)
    return text if any(c in text for c in ".eE") else text + ".0"


def diffusion_lines(name: str, coefficient: float) -> list[str]:
    """Preserve the MUSICA phase coefficient; reject unsupported native precision."""
    if not math.isfinite(coefficient) or coefficient <= 0:
        raise ValueError(f"{name}: diffusion coefficient must be finite and positive")
    literal = number(coefficient)
    return [
        f"      static_assert(std::numeric_limits<micm::Real>::max() >= {literal},",
        f'                    "{name}: configured diffusion coefficient requires binary64 MICM");',
        f"      phase_species.emplace_back(species, {literal});",
    ]


def cxx_string(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def new_species_expression(name: str) -> str:
    """Construct a species. Only CreateGasPhase does this, once per species."""
    return f"micm::Species({cxx_string(name)})"


def species_expression(name: str) -> str:
    """Refer to the phase's own species object, for use inside a reaction.

    A freshly constructed micm::Species would drop the third-body
    parameterization and the molecular weight, and micm would then look for a
    state variable that the phase does not expose.
    """
    return f"SpeciesByName(gas_phase, {cxx_string(name)})"


def reactant_list(components) -> str:
    """Expand reactant stoichiometry into one entry per unit."""
    names = []
    for component in components:
        if component.coefficient != int(component.coefficient):
            raise SystemExit(f"reactant {component.name} has a fractional coefficient {component.coefficient}")
        names.extend([component.name] * int(component.coefficient))
    return ", ".join(species_expression(n) for n in names)


def product_list(components) -> str:
    return ", ".join(f"micm::StoichSpecies{{ {species_expression(c.name)}, {number(c.coefficient)} }}" for c in components)


def rate_constant_expression(reaction) -> str:
    """Render the micm rate-constant parameters for one musica reaction."""
    if isinstance(reaction, Arrhenius):
        fields = ", ".join(f".{p}_ = {number(getattr(reaction, p))}" for p in ARRHENIUS_PARAMETERS)
        return f"micm::ArrheniusRateConstantParameters{{ {fields} }}"
    if isinstance(reaction, Troe):
        fields = ", ".join(f".{p}_ = {number(getattr(reaction, p))}" for p in TROE_PARAMETERS)
        return f"micm::TroeRateConstantParameters{{ {fields} }}"
    if isinstance(reaction, Photolysis | UserDefined):
        # micm represents both as a user-defined rate looked up by label.
        return (
            "micm::UserDefinedRateConstantParameters{ "
            f".label_ = {cxx_string(reaction.name)}, "
            f".scaling_factor_ = {number(reaction.scaling_factor)} }}"
        )
    if isinstance(reaction, Surface):
        return (
            f"micm::SurfaceRateConstantParameters{{ .label_ = {cxx_string(reaction.name)},"
            f" .phase_species_ = PhaseSpeciesByName(gas_phase, {cxx_string(reaction.gas_phase_species.name)}),"
            f" .reaction_probability_ = {number(reaction.reaction_probability)} }}"
        )
    raise SystemExit(
        f"the benchmark cannot represent a {type(reaction).__name__} reaction; "
        "it supports Arrhenius, Troe, Photolysis, UserDefined, and Surface"
    )


def reaction_block(reaction) -> list[str]:
    """Render one reaction as a micm::ChemicalReactionBuilder chain."""
    if isinstance(reaction, Surface):
        reactants, products = [reaction.gas_phase_species], reaction.gas_phase_products
    else:
        reactants, products = reaction.reactants, reaction.products

    return [
        "    processes.push_back(micm::ChemicalReactionBuilder()",
        f"                            .SetReactants({{ {reactant_list(reactants)} }})",
        f"                            .SetProducts({{ {product_list(products)} }})",
        f"                            .SetRateConstant({rate_constant_expression(reaction)})",
        "                            .SetPhase(gas_phase)",
        "                            .Build());",
    ]


def gas_phase_function(mechanism) -> list[str]:
    """Render CreateGasPhase, including third bodies and diffusion coefficients.

    A molecular weight belongs to the species; a diffusion coefficient belongs
    to the species within a phase, so the two come from different objects.
    """
    phases = {phase.name: phase for phase in mechanism.phases}
    if GAS_PHASE not in phases:
        raise SystemExit(f"expected a '{GAS_PHASE}' phase, found {sorted(phases)}")
    diffusion = {p.name: p.diffusion_coefficient_m2_s for p in phases[GAS_PHASE].species}

    lines = [
        f"  /// @brief The TS1 gas phase: {len(mechanism.species)} species.",
        "  inline micm::Phase CreateGasPhase()",
        "  {",
        "    std::vector<micm::PhaseSpecies> phase_species;",
        f"    phase_species.reserve({len(mechanism.species)});",
        "",
    ]

    for species in mechanism.species:
        lines.append("    {")
        lines.append(f"      auto species = {new_species_expression(species.name)};")
        if species.molecular_weight_kg_mol is not None:
            lines.append(
                "      species.SetProperty(micm::property_keys::MOLECULAR_WEIGHT,"
                f" micm::Real{{ {number(species.molecular_weight_kg_mol)} }});"
            )
        if species.is_third_body:
            lines.append("      // Parameterized on air density, so it holds no state variable.")
            lines.append("      species.SetThirdBody();")
        coefficient = diffusion.get(species.name)
        if coefficient is None:
            lines.append("      phase_species.emplace_back(species);")
        else:
            lines += diffusion_lines(species.name, coefficient)
        lines.append("    }")

    lines += [
        "",
        f"    return micm::Phase{{ {cxx_string(GAS_PHASE)}, phase_species }};",
        "  }",
    ]
    return lines


def processes_function(mechanism) -> list[str]:
    lines = [
        f"  /// @brief The TS1 reactions: {len(mechanism.reactions)} processes.",
        "  inline std::vector<micm::Process> CreateProcesses(const micm::Phase& gas_phase)",
        "  {",
        "    std::vector<micm::Process> processes;",
        f"    processes.reserve({len(mechanism.reactions)});",
        "",
    ]
    for reaction in mechanism.reactions:
        lines += reaction_block(reaction)
    lines += ["", "    return processes;", "  }"]
    return lines


def init_state_function(environment, concentrations, parameters) -> list[str]:
    """Render InitState.

    The single-value setters (state["X"] = v and SetCustomRateParameter(label, v))
    throw on a multi-cell state, so this fills one vector per value and uses the
    vector overloads.
    """
    lines = [
        "    template<class State>",
        "    static void InitState(State& state, micm::Index num_cells)",
        "    {",
        "      std::vector<micm::Real> cells(num_cells);",
        "      auto concentration = [&](const char* name, micm::Real value)",
        "      {",
        "        std::fill(cells.begin(), cells.end(), value);",
        "        state[name] = cells;",
        "      };",
        "      auto parameter = [&](const char* label, micm::Real value)",
        "      {",
        "        std::fill(cells.begin(), cells.end(), value);",
        "        state.SetCustomRateParameter(label, cells);",
        "      };",
        "",
    ]
    for name, value in concentrations.items():
        lines.append(f"      concentration({cxx_string(name)}, {number(value)});")
    lines.append("")
    for label, value in parameters.items():
        lines.append(f"      parameter({cxx_string(label)}, {number(value)});")
    lines += [
        "",
        "      for (micm::Index c = 0; c < num_cells; ++c)",
        "      {",
        f"        state.conditions_[c].temperature_ = {number(environment['temperature'])};",
        f"        state.conditions_[c].pressure_ = {number(environment['pressure'])};",
        "        // Troe rates and the third body read air density, and nothing else",
        "        // computes it.",
        "        state.conditions_[c].CalculateIdealAirDensity();",
        "      }",
        "    }",
    ]
    return lines


def forcing_schedule_function(scenario: dict) -> list[str]:
    """Render the OpenAtmos scenario's named piecewise-constant forcing table."""
    integration = scenario["integration"]
    start = float(integration["start_seconds"])
    end = start + float(integration["horizon_seconds"])
    boundaries = sorted(set((start, *map(float, integration["forcing_boundaries_seconds"]), end)))
    forcing = scenario["forcing"]
    values: dict[str, dict[float, float]] = {}
    for name, definition in forcing.items():
        if "constant" in definition:
            values[name] = {boundary: float(definition["constant"]) for boundary in boundaries[:-1]}
        else:
            values[name] = {float(point["time_seconds"]): float(point["value"]) for point in definition["points"]}
        missing = [boundary for boundary in boundaries[:-1] if boundary not in values[name]]
        if missing:
            raise SystemExit(f"forcing {name!r} lacks values at boundaries {missing}")

    lines = [
        f"    static constexpr micm::Index kForcingSegmentCount = {len(boundaries) - 1};",
        "    static constexpr double kForcingSegmentDurations[] = {"
        + ", ".join(number(right - left) for left, right in zip(boundaries[:-1], boundaries[1:], strict=True))
        + "};",
        "",
        "    template<class State>",
        "    static void ApplyForcingSegment(State& state, micm::Index num_cells, micm::Index segment)",
        "    {",
        '      if (segment >= kForcingSegmentCount) throw std::out_of_range("TS1 forcing segment");',
        "      std::vector<micm::Real> cells(num_cells);",
        "      auto parameter = [&](const char* label, micm::Real value)",
        "      {",
        "        std::fill(cells.begin(), cells.end(), value);",
        "        state.SetCustomRateParameter(label, cells);",
        "      };",
    ]
    for name in sorted(values):
        series = ", ".join(number(values[name][boundary]) for boundary in boundaries[:-1])
        lines += [
            f"      static constexpr micm::Real {name}_values[] = {{{series}}};",
            f"      parameter({cxx_string(name)}, {name}_values[segment]);",
        ]
    lines += ["    }"]
    return lines


def custom_parameter_labels(mechanism) -> list[str]:
    """List every custom rate parameter micm registers for this mechanism.

    See GetCustomParameterMap in include/micm/solver/solver_builder.inl.
    """
    labels = []
    for reaction in mechanism.reactions:
        if isinstance(reaction, Photolysis | UserDefined):
            labels.append(reaction.name)
        elif isinstance(reaction, Surface):
            labels.append(f"{reaction.name}.effective radius [m]")
            labels.append(f"{reaction.name}.particle number concentration [# m-3]")
    return labels


def read_initial_conditions(path: Path):
    """Split the initial-conditions CSV into environment, concentrations, and parameters.

    The layout matches musica's own TS1 box-model example:

      | prefix | first value                         | second value                         |
      | SURF   | effective radius (m)                | particle number concentration (# m-3)|
      | CONC   | initial concentration (mol m-3)     | unused                               |
      | ENV    | temperature (K) or pressure (Pa)    | unused                               |
      | USER   | user-defined parameter value        | unused                               |
      | PHOTO  | photolysis rate constant (s-1)      | unused                               |
    """
    table = pd.read_csv(path, header=None, names=["parameter", "first", "second"])

    environment: dict[str, float] = {}
    concentrations: dict[str, float] = {}
    parameters: dict[str, float] = {}

    for row in table.itertuples(index=False):
        prefix, _, name = str(row.parameter).partition(".")
        if prefix == "ENV":
            environment[name] = float(row.first)
        elif prefix == "CONC":
            concentrations[name] = float(row.first)
        elif prefix in ("PHOTO", "USER"):
            parameters[name] = float(row.first)
        elif prefix == "SURF":
            parameters[f"{name}.effective radius [m]"] = float(row.first)
            parameters[f"{name}.particle number concentration [# m-3]"] = float(row.second)
        else:
            raise SystemExit(f"unknown initial-condition prefix in {row.parameter!r}")

    return environment, concentrations, parameters


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="directory holding ts1.json and initial_conditions.csv (default: the copy musica ships)",
    )
    parser.add_argument("--scenario", type=Path, required=True, help="immutable OpenAtmos scenario YAML")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).parent / "ts1_mechanism.hpp",
        help="output header (default: generated MICM TS1 binding)",
    )
    args = parser.parse_args()

    source = args.source if args.source is not None else Path(find_config_path("v1", "ts1"))
    mechanism = mc.parse(str(source / "ts1.json"))
    environment, concentrations, parameters = read_initial_conditions(source / "initial_conditions.csv")
    solver_config = yaml.safe_load((source / "environment.yaml").read_text()).get("solver", {})
    solver_atol = float(solver_config.get("atol", 1.0e-3))
    solver_rtol = float(solver_config.get("rtol", 1.0e-6))
    solver_max_steps = int(solver_config.get("max_steps", 1000))
    scenario = yaml.safe_load(args.scenario.read_text())

    # A third body is parameterized on air density, so it holds no concentration.
    third_bodies = {s.name for s in mechanism.species if s.is_third_body}

    missing_concentrations = sorted(
        s.name for s in mechanism.species if s.name not in concentrations and s.name not in third_bodies
    )
    missing_parameters = sorted(label for label in custom_parameter_labels(mechanism) if label not in parameters)
    for name in missing_concentrations:
        concentrations[name] = 0.0
    for label in missing_parameters:
        parameters[label] = 0.0

    counts: dict[str, int] = {}
    for reaction in mechanism.reactions:
        kind = type(reaction).__name__
        counts[kind] = counts.get(kind, 0) + 1
    summary = ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))

    header = [
        "// Copyright (C) 2023-2026 University Corporation for Atmospheric Research",
        "// SPDX-License-Identifier: Apache-2.0",
        "//",
        "// GENERATED FILE. Do not edit. Re-run the governed TS1 importer instead.",
        "//",
        f"// The MOZART-TS1 mechanism ({mechanism.name}), as a benchmark mechanism.",
        f"// Read from the musica TS1 configuration, mechanism configuration version {mechanism.version}.",
        "//",
        f"// {len(mechanism.species)} species and {len(mechanism.reactions)} reactions ({summary}).",
        "// Use it to see how the linear solver and the rate calculations scale with",
        "// mechanism size.",
    ]
    if missing_concentrations:
        header.append(f"//\n// No concentration in the CSV, set to zero: {' '.join(missing_concentrations)}.")
    if missing_parameters:
        header.append(f"// No rate parameter in the CSV, set to zero: {' '.join(missing_parameters)}.")

    lines = header + [
        "#pragma once",
        "",
        "#include <micm/CPU.hpp>",
        "#include <micm/util/property_keys.hpp>",
        "",
        "#include <algorithm>",
        "#include <limits>",
        "#include <string_view>",
        "#include <vector>",
        "",
        "namespace bench",
        "{",
        "  namespace ts1",
        "  {",
        "    /// @brief Find a phase species by name. A surface reaction needs the phase",
        "    ///        entry, because that is what carries the diffusion coefficient.",
        "    inline const micm::PhaseSpecies& PhaseSpeciesByName(const micm::Phase& phase, std::string_view name)",
        "    {",
        "      for (const auto& phase_species : phase.phase_species_)",
        "      {",
        "        if (phase_species.species_.name_ == name)",
        "        {",
        "          return phase_species;",
        "        }",
        "      }",
        '      throw std::runtime_error("ts1: no species named \'" + std::string(name) + "\' in the phase");',
        "    }",
        "",
        "    /// @brief Find a species by name in the phase. A reaction must use the",
        "    ///        phase's own species object, because that is what carries the",
        "    ///        third-body parameterization and the molecular weight.",
        "    inline const micm::Species& SpeciesByName(const micm::Phase& phase, std::string_view name)",
        "    {",
        "      return PhaseSpeciesByName(phase, name).species_;",
        "    }",
        "",
    ]

    lines += ["  " + line if line else "" for line in gas_phase_function(mechanism)]
    lines.append("")
    lines += ["  " + line if line else "" for line in processes_function(mechanism)]
    lines += [
        "  }  // namespace ts1",
        "",
        "  struct Ts1",
        "  {",
        '    static constexpr std::string_view kName = "ts1";',
        f"    static constexpr double kSolverAtol = {number(solver_atol)};",
        f"    static constexpr double kSolverRtol = {number(solver_rtol)};",
        f"    static constexpr micm::Index kSolverMaxSteps = {solver_max_steps};",
        "",
        "    template<class Builder>",
        "    static auto Build(Builder builder)",
        "    {",
        "      auto gas_phase = ts1::CreateGasPhase();",
        "      return builder.SetSystem(micm::System(gas_phase))",
        "          .SetReactions(ts1::CreateProcesses(gas_phase))",
        "          .SetIgnoreUnusedSpecies(true)",
        "          .Build();",
        "    }",
        "",
    ]
    lines += init_state_function(environment, concentrations, parameters)
    lines.append("")
    lines += forcing_schedule_function(scenario)
    lines += [
        "  };",
        "}  // namespace bench",
    ]

    args.output.write_text("\n".join(lines) + "\n")

    # The CI format job rewrites every test/**/*.hpp, so format here and keep
    # regeneration free of spurious diffs.
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        print("clang-format not found; the generated header is unformatted", file=sys.stderr)
    else:
        subprocess.run([clang_format, "-i", "--style=file", str(args.output)], check=True)

    print(f"read {source}", file=sys.stderr)
    print(f"wrote {args.output} ({len(lines)} lines)", file=sys.stderr)
    print(f"  {len(mechanism.species)} species, {len(mechanism.reactions)} reactions ({summary})", file=sys.stderr)
    print(f"  {len(concentrations)} concentrations, {len(parameters)} custom rate parameters", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
