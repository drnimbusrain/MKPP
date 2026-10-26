"""Canonical benchmark chemistry must explicitly identify the v1 input format."""

import hashlib
import json
from pathlib import Path

import pytest
from mkpp.benchmark.openatmos import require_openatmos_v1
from mkpp.benchmark.schema import SchemaValidationError

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("version", [None, "0.0.0", "v1", "2.0.0", 1])
def test_rejects_unversioned_or_other_configuration_versions(version):
    document = {
        "name": "test",
        "species": [{"name": "A"}],
        "phases": [{"name": "gas", "species": [{"name": "A"}]}],
        "reactions": [{"type": "ARRHENIUS"}],
    }
    if version is not None:
        document["version"] = version
    with pytest.raises(SchemaValidationError, match="version '1.0.0'"):
        require_openatmos_v1(document)


def test_pinned_ts1_is_explicit_v1_and_restored_csv_matches_original_pin():
    source = ROOT / "benchmarks/solver_comparison/sources/ts1/musica-config"
    require_openatmos_v1(json.loads((source / "ts1.json").read_text()))
    assert hashlib.sha256((source / "initial_conditions.csv").read_bytes()).hexdigest() == (
        "58db29256362fcd7fd60d47fe3d463c566dfbe14c000757202393fa7a28a9253"
    )


def test_directory_name_does_not_certify_unversioned_configuration():
    document = json.loads((ROOT / "mechanisms/openatmos/chapman/mechanism.json").read_text())
    with pytest.raises(SchemaValidationError, match="unversioned"):
        require_openatmos_v1(document)


def test_format_version_does_not_allow_omitting_phase_membership():
    document = {
        "name": "test",
        "version": "1.0.0",
        "species": [{"name": "A"}],
        "phases": [{"name": "gas"}],
        "reactions": [{"type": "ARRHENIUS"}],
    }
    with pytest.raises(SchemaValidationError, match="species membership"):
        require_openatmos_v1(document)
