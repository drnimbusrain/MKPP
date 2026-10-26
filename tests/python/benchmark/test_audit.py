"""An offline audit detects missing or altered scientific assets without admission."""

import hashlib

import yaml
from mkpp.benchmark.audit import audit_campaign_assets


def test_asset_audit_retains_missing_and_changed_sources(tmp_path):
    (tmp_path / "chemistry.json").write_text("changed chemistry")
    manifest = {
        "mechanism_id": "small",
        "version": "1",
        "chemistry_sha256": "a" * 64,
        "canonical_source": [
            {"path": "chemistry.json", "sha256": hashlib.sha256(b"original").hexdigest()},
            {"path": "missing.csv", "sha256": "b" * 64},
        ],
        "required_solvers": ["mkpp"],
        "bindings": {},
        "admission_status": "draft",
    }
    (tmp_path / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    report = audit_campaign_assets({"campaign_id": "test", "targets": [{"manifest": "manifest.yaml", "scenarios": []}]}, tmp_path)
    assert not report["publishable"]
    issues = report["targets"][0]["issues"]
    assert any("sha256 mismatch" in issue for issue in issues)
    assert any("missing.csv" in issue for issue in issues)
    assert any("draft" in issue for issue in issues)
