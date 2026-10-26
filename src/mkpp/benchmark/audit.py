"""Offline asset audit; metadata validation never implies scientific admission."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mkpp.benchmark.openatmos import require_openatmos_v1
from mkpp.benchmark.preflight import PreflightError, resolve_relative_asset, verify_asset_hash
from mkpp.benchmark.schema import SchemaValidationError


def audit_campaign_assets(campaign: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Inventory missing, changed, and unadmitted benchmark inputs without running solvers.

    Parameters
    ----------
    campaign : Mapping
        Structurally validated campaign inputs.
    root : Path
        Repository root containing the declared assets.

    Returns
    -------
    dict
        Findings and an explicit false publication flag. Asset integrity alone
        cannot qualify chemistry, reference, calibration, or resources.
    """
    input_issues: list[str] = []
    for comparison_path in campaign.get("input_policy", {}).get("source_comparisons", ()):
        try:
            comparison_file = resolve_relative_asset(root, comparison_path)
            comparison = yaml.safe_load(comparison_file.read_text(encoding="utf-8"))
            if comparison.get("upstream_equivalence_verified") is not True:
                input_issues.append(f"{comparison_path}: upstream source equivalence is unresolved")
        except PreflightError as error:
            input_issues.append(str(error))
    targets = []
    for target in campaign["targets"]:
        manifest = yaml.safe_load((root / target["manifest"]).read_text(encoding="utf-8"))
        issues: list[str] = []
        verified: list[str] = []
        for source in manifest["canonical_source"]:
            try:
                asset = resolve_relative_asset(root, source["path"])
                verify_asset_hash(asset, source["sha256"])
                verified.append(source["path"])
                if asset.suffix.lower() in {".json", ".yaml", ".yml"}:
                    require_openatmos_v1(yaml.safe_load(asset.read_text(encoding="utf-8")))
            except (PreflightError, SchemaValidationError) as error:
                issues.append(str(error))
        for solver, binding in manifest["bindings"].items():
            for asset in binding["assets"]:
                name = asset if isinstance(asset, str) else asset["path"]
                try:
                    resolved = resolve_relative_asset(root, name)
                    if isinstance(asset, Mapping):
                        verify_asset_hash(resolved, asset["sha256"])
                except PreflightError as error:
                    issues.append(f"{solver}: {error}")
        for scenario_path in target["scenarios"]:
            scenario = yaml.safe_load((root / scenario_path).read_text(encoding="utf-8"))
            expected = {"id": manifest["mechanism_id"], "version": manifest["version"], "sha256": manifest["chemistry_sha256"]}
            if scenario["mechanism"] != expected:
                issues.append(f"{scenario['scenario_id']}: mechanism identity differs from manifest chemistry identity")
        if manifest["admission_status"] != "admitted":
            issues.append(f"mechanism admission is {manifest['admission_status']}; native scientific evidence is required")
        targets.append(
            {
                "mechanism_id": manifest["mechanism_id"],
                "required_solvers": manifest["required_solvers"],
                "verified_canonical_assets": verified,
                "issues": issues,
            }
        )
    return {
        "campaign_id": campaign["campaign_id"],
        "publishable": False,
        "input_issues": input_issues,
        "targets": targets,
        "reason": "Offline audit only; no chemistry, reference, calibration, accuracy, or resource gates were executed.",
    }


__all__ = ["audit_campaign_assets"]
