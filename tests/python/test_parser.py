import json
import time

import pytest
from mkpp.model import CompilationError
from mkpp.parser import (
    detect_config_format,
    load_environment,
    load_mechanism,
    parse_mechanism_micm,
)


def test_parse_complex_troe_parameters():
    data = {
        "species": [{"name": "O"}, {"name": "O2"}, {"name": "O3"}, {"name": "M"}],
        "reactions": [
            {
                "type": "TROE",
                "reactants": {"O": {}, "O2": {}},
                "products": {"O3": {}},
                "k0": {"A": 1.0, "B": 2.0, "C": 3.0},
                "kinf": {"A": 4.0, "B": 5.0, "C": 6.0},
                "Fc": 0.6,
                "N": 1.0,
            }
        ],
    }

    mech = parse_mechanism_micm("test", data)
    assert mech.reactions[0].reaction_type == "TROE"
    assert "k0" in mech.reactions[0].parameters
    assert mech.reactions[0].parameters["k0"]["A"] == 1.0
    assert mech.reactions[0].parameters["kinf"]["C"] == 6.0


def test_convert_number_density_rate_constants_to_si():
    data = {
        "concentration_unit": "molecule cm-3",
        "species": [{"name": name} for name in ("A", "B", "C", "M")],
        "reactions": [
            {
                "name": "second_order",
                "type": "ARRHENIUS",
                "reactants": {"A": 1.0, "B": 1.0},
                "products": {"C": 1.0},
                "A": 2.0e-12,
            },
            {
                "name": "third_order",
                "type": "ARRHENIUS",
                "reactants": {"A": 1.0, "B": 1.0, "M": 1.0},
                "products": {"C": 1.0},
                "A": 3.0e-30,
            },
            {
                "name": "falloff",
                "type": "TROE",
                "reactants": {"A": 1.0, "B": 1.0},
                "products": {"C": 1.0},
                "k0": {"A": 4.0e-30, "B": 0.0, "C": 0.0},
                "kinf": {"A": 5.0e-12, "B": 0.0, "C": 0.0},
            },
        ],
    }

    mechanism = parse_mechanism_micm("number_density", data, convert_kinetic_units_to_si=True)
    second_order_factor = (1.0e6 / 6.02214076e23) ** -1
    third_order_factor = (1.0e6 / 6.02214076e23) ** -2

    assert mechanism.concentration_unit == "mol m-3"
    assert mechanism.kinetic_units_converted_to_si
    assert mechanism.reactions[0].parameters["A"] == pytest.approx(2.0e-12 * second_order_factor)
    assert mechanism.reactions[1].parameters["A"] == pytest.approx(3.0e-30 * third_order_factor)
    assert mechanism.reactions[2].parameters["k0"]["A"] == pytest.approx(4.0e-30 * third_order_factor)
    assert mechanism.reactions[2].parameters["kinf"]["A"] == pytest.approx(5.0e-12 * second_order_factor)


def test_si_rate_constants_are_unchanged_when_si_conversion_requested():
    mechanism = parse_mechanism_micm(
        "si_units",
        {
            "concentration_unit": "mol m-3",
            "species": [{"name": "A"}, {"name": "B"}],
            "reactions": [{"type": "ARRHENIUS", "reactants": {"A": 1.0}, "products": {"B": 1.0}, "A": 4.0}],
        },
        convert_kinetic_units_to_si=True,
    )

    assert mechanism.concentration_unit == "mol m-3"
    assert not mechanism.kinetic_units_converted_to_si
    assert mechanism.reactions[0].parameters["A"] == 4.0


def test_si_rate_conversion_requires_declared_concentration_units():
    with pytest.raises(CompilationError, match="requires mechanism concentration_unit"):
        parse_mechanism_micm(
            "undeclared_units",
            {"species": [{"name": "A"}], "reactions": []},
            convert_kinetic_units_to_si=True,
        )


def test_openatmos_background_species_are_not_implicitly_fixed():
    mech = parse_mechanism_micm(
        "ts1_role_test",
        {
            "species": [{"name": name} for name in ("O2", "N2", "H2O", "H2", "CH4", "RO2", "M")],
            "reactions": [],
        },
    )
    roles = {species.name: species.role for species in mech.species}
    assert {name for name, role in roles.items() if role == "fixed"} == {"M"}


def test_openatmos_surface_reaction_uses_gas_phase_fields():
    mech = parse_mechanism_micm(
        "surface_test",
        {
            "species": [{"name": name} for name in ("N2O5", "HNO3", "NO3")],
            "reactions": [
                {
                    "type": "SURFACE",
                    "reaction probability": 0.02,
                    "gas-phase species": "N2O5",
                    "gas-phase products": [
                        {"species name": "HNO3", "coefficient": 2.0},
                        {"species name": "NO3", "coefficient": 0.5},
                    ],
                }
            ],
        },
    )

    reaction = mech.reactions[0]
    assert reaction.reactants == {"N2O5": 1.0}
    assert reaction.products == {"HNO3": 2.0, "NO3": 0.5}


def test_detect_config_format(tmp_path):
    json_file = tmp_path / "mech.json"
    json_file.write_text('{"species": [{"name": "O3"}]}')

    yaml_file = tmp_path / "mech.yaml"
    yaml_file.write_text("species:\n  - name: O3")

    yml_file = tmp_path / "mech.yml"
    yml_file.write_text("species:\n  - name: O3")

    no_ext_json = tmp_path / "mech_no_ext_json"
    no_ext_json.write_text('{"species": [{"name": "O3"}]}')

    no_ext_yaml = tmp_path / "mech_no_ext_yaml"
    no_ext_yaml.write_text("species:\n  - name: O3")

    assert detect_config_format(json_file) == "json"
    assert detect_config_format(yaml_file) == "yaml"
    assert detect_config_format(yml_file) == "yaml"
    assert detect_config_format(no_ext_json) == "json"
    assert detect_config_format(no_ext_yaml) == "yaml"


def test_load_mechanism_and_environment_yaml(tmp_path):
    mech_file = tmp_path / "chapman.yaml"
    mech_file.write_text("""
name: chapman
species:
  - name: O3
  - name: O
  - name: O2
reactions:
  - type: ARRHENIUS
    reactants:
      O: 1
      O2: 1
    products:
      O3: 1
    A: 1.2e-12
""")

    env_file = tmp_path / "env.yaml"
    env_file.write_text("""
environment:
  temperature: 298.15
  pressure: 101325.0
  air_density: 2.46e19
  relative_humidity: 0.55
initial_concentrations:
  O3: 1e-6
  O: 1e-10
  O2: 0.21
""")

    mech = load_mechanism(mech_file)
    assert mech.name == "chapman"
    assert len(mech.species) == 3
    assert len(mech.reactions) == 1
    assert mech.reactions[0].reactants == {"O": 1.0, "O2": 1.0}

    env = load_environment(env_file)
    assert env.temperature == 298.15
    assert env.pressure == 101325.0
    assert env.relative_humidity == 0.55
    assert env.initial_concentrations["O3"] == 1e-6


def test_load_mechanism_and_environment_json_parity(tmp_path):
    mech_dict = {
        "species": [{"name": "O3"}, {"name": "NO"}, {"name": "NO2"}],
        "reactions": [
            {
                "type": "ARRHENIUS",
                "reactants": {"NO": 1, "O3": 1},
                "products": {"NO2": 1},
                "A": 1.8e-14,
            }
        ],
    }

    env_dict = {
        "environment": {
            "temperature": 290.0,
            "pressure": 95000.0,
            "air_density": 2.3e19,
            "relative_humidity": 0.4,
        },
        "initial_concentrations": {"O3": 5e-8, "NO": 1e-9, "NO2": 2e-9},
    }

    yaml_mech = tmp_path / "mech.yaml"
    json_mech = tmp_path / "mech.json"
    yaml_env = tmp_path / "env.yaml"
    json_env = tmp_path / "env.json"

    import yaml

    yaml_mech.write_text(yaml.dump(mech_dict))
    json_mech.write_text(json.dumps(mech_dict))
    yaml_env.write_text(yaml.dump(env_dict))
    json_env.write_text(json.dumps(env_dict))

    m_yaml = load_mechanism(yaml_mech)
    m_json = load_mechanism(json_mech)

    assert m_yaml.name == m_json.name
    assert m_yaml.species == m_json.species
    assert m_yaml.reactions == m_json.reactions

    e_yaml = load_environment(yaml_env)
    e_json = load_environment(json_env)

    assert e_yaml == e_json


def test_reject_invalid_or_legacy_formats(tmp_path):
    # Missing species
    invalid_mech = tmp_path / "bad_mech.json"
    invalid_mech.write_text('{"reactions": []}')
    with pytest.raises(CompilationError) as exc_info:
        load_mechanism(invalid_mech)
    assert "species" in exc_info.value.message.lower()

    # Malformed JSON syntax
    bad_json = tmp_path / "syntax_error.json"
    bad_json.write_text('{"species": [{"name": "O3"}')
    with pytest.raises(CompilationError) as exc_info:
        load_mechanism(bad_json)
    assert exc_info.value.stage == "parsing"
    assert "JSON syntax error" in exc_info.value.message

    # Malformed YAML syntax
    bad_yaml = tmp_path / "syntax_error.yaml"
    bad_yaml.write_text("species: [ unclosed_list")
    with pytest.raises(CompilationError) as exc_info:
        load_mechanism(bad_yaml)
    assert exc_info.value.stage == "parsing"

    # Non-existent file
    with pytest.raises(FileNotFoundError):
        load_mechanism(tmp_path / "non_existent.yaml")


def test_parser_performance_benchmark(tmp_path):
    # Construct a 500-reaction mechanism
    species_list = [{"name": f"S{i}"} for i in range(100)]
    reactions = []
    for i in range(500):
        r1 = f"S{i % 100}"
        r2 = f"S{(i + 1) % 100}"
        p1 = f"S{(i + 2) % 100}"
        reactions.append(
            {
                "type": "ARRHENIUS",
                "reactants": {r1: 1, r2: 1},
                "products": {p1: 1},
                "A": 1.0e-12,
            }
        )

    mech_dict = {"species": species_list, "reactions": reactions}
    json_path = tmp_path / "large_mech.json"
    yaml_path = tmp_path / "large_mech.yaml"

    json_path.write_text(json.dumps(mech_dict))
    import yaml

    yaml_path.write_text(yaml.dump(mech_dict))

    t0 = time.perf_counter()
    m1 = load_mechanism(json_path)
    t_json = time.perf_counter() - t0

    t0 = time.perf_counter()
    m2 = load_mechanism(yaml_path)
    t_yaml = time.perf_counter() - t0

    assert len(m1.reactions) == 500
    assert len(m2.reactions) == 500
    assert t_json < 0.5, f"JSON parse took {t_json:.3f}s (target <0.5s)"
    assert t_yaml < 0.5, f"YAML parse took {t_yaml:.3f}s (target <0.5s)"
