#!/usr/bin/env python3
"""Verify benchmark TS1 inputs match the project's pinned MUSICA distribution."""

from __future__ import annotations

import hashlib
import json
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


def main() -> int:
    """Check package and configuration versions and byte-identical input assets."""
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / "benchmarks/solver_comparison/sources/ts1/musica-v1.lock.json").read_text())
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependency = f"{lock['package']}=={lock['package_version']}"
    if dependency not in project["project"]["optional-dependencies"]["solver-comparison"]:
        raise ValueError("MUSICA source lock and solver-comparison dependency pin differ")
    package = distribution(lock["package"])
    if package.version != lock["package_version"]:
        raise ValueError(f"installed MUSICA {package.version} differs from pinned {lock['package_version']}")
    for asset in lock["files"]:
        upstream = Path(package.locate_file(asset["package_path"]))
        local = root / asset["vendored_path"]
        upstream_bytes = upstream.read_bytes()
        if hashlib.sha256(upstream_bytes).hexdigest() != asset["sha256"]:
            raise ValueError(f"distributed input differs from source lock: {asset['package_path']}")
        if local.read_bytes() != upstream_bytes:
            raise ValueError(f"benchmark input differs from MUSICA: {asset['vendored_path']}")
        if local.suffix == ".json" and json.loads(upstream_bytes).get("version") != lock["configuration_version"]:
            raise ValueError("configuration version differs from MUSICA source lock")
    print(f"Verified exact {dependency} v{lock['configuration_version']} TS1 mechanism and initial conditions")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, LookupError, PackageNotFoundError) as error:
        print(f"FATAL ERROR: {error}", file=sys.stderr)
        raise SystemExit(4) from error
