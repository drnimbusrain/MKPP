"""Unit test suite for MKPP host model API Jinja2 codegen pipeline."""

import pytest
from mkpp.codegen import generate_headers, generate_host_api_headers
from mkpp.model import (
    AerosolRepresentation,
    MechanismDefinition,
    PhaseMode,
    ReactionDefinition,
    SpeciesDefinition,
)


@pytest.fixture
def dummy_mechanism():
    species = [
        SpeciesDefinition(name="O3", phase=PhaseMode.GAS),
        SpeciesDefinition(name="NO", phase=PhaseMode.GAS),
        SpeciesDefinition(name="NO2", phase=PhaseMode.GAS),
    ]
    reactions = [
        ReactionDefinition(
            reaction_type="ARRHENIUS",
            reactants={"NO": 1.0, "O3": 1.0},
            products={"NO2": 1.0},
            rate_expression="1.0e-12",
            parameters={"A": 1.0e-12, "B": 0.0, "C": 0.0},
        )
    ]
    return MechanismDefinition(
        name="test_chapman",
        description="Test Chapman Mechanism",
        aerosol_representation=AerosolRepresentation.BULK,
        species=species,
        phases=[],
        reactions=reactions,
    )


def test_generate_host_api_headers_standalone(dummy_mechanism, tmp_path):
    out_dir = str(tmp_path)
    results = generate_host_api_headers(dummy_mechanism, out_dir=out_dir)

    assert "c_header" in results
    assert "fortran_module" in results
    assert "cpp_header" in results
    assert "c_api_source" in results

    c_header_content = open(results["c_header"]).read()
    assert "mkpp_create_handle" in c_header_content
    assert "MKPP_NUM_SPECIES   3" in c_header_content

    fortran_content = open(results["fortran_module"]).read()
    assert "module mkpp_mod" in fortran_content
    assert "use, intrinsic :: iso_c_binding" in fortran_content

    cpp_header_content = open(results["cpp_header"]).read()
    assert "class MechanismContext" in cpp_header_content
    assert "getSpeciesIndex" in cpp_header_content

    c_api_content = open(results["c_api_source"]).read()
    assert "mkpp_set_state_ptrs" in c_api_content
    assert '"O3"' in c_api_content


def test_host_api_validates_fixed_layout_view_extents(dummy_mechanism, tmp_path):
    results = generate_host_api_headers(dummy_mechanism, out_dir=str(tmp_path))

    c_api_content = open(results["c_api_source"]).read()
    assert "Kokkos::LayoutLeft" in c_api_content
    assert "species_conc.extent(0)" in c_api_content
    assert "species_conc.extent(1)" in c_api_content
    assert "FATAL ERROR:" in c_api_content


def test_generate_headers_with_host_api_flag(dummy_mechanism, tmp_path):
    out_dir = str(tmp_path)
    results = generate_headers(dummy_mechanism, out_dir=out_dir, generate_host_api=True)

    assert "header" in results
    assert "manifest" in results
    assert "c_header" in results
    assert "fortran_module" in results
    assert "cpp_header" in results
    assert "c_api_source" in results
