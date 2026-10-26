"""Explicit OpenAtmos v1 envelope gate for canonical benchmark chemistry.

This checks the version and core container structure. It is not a replacement
for the upstream mechanism validator or numerical chemistry-equivalence audits.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mkpp.benchmark.schema import SchemaValidationError


def require_openatmos_v1(document: Mapping[str, Any]) -> None:
    """Require an explicitly versioned v1 chemistry document before benchmarking.

    Parameters
    ----------
    document : Mapping
        Canonical mechanism JSON/YAML, not the benchmark scenario or manifest.

    Returns
    -------
    None
        Raises ``SchemaValidationError`` for missing/unsupported versions or
        absent core v1 structures. No defaults or chemistry conversion occur.
    """
    if not isinstance(document, Mapping) or document.get("version") != "1.0.0":
        raise SchemaValidationError(
            "OpenAtmos configuration must explicitly declare version '1.0.0'; unversioned/v0 inputs are forbidden"
        )
    if not isinstance(document.get("name"), str) or not document["name"].strip():
        raise SchemaValidationError("OpenAtmos v1 name must be a non-empty string")
    for field in ("species", "phases", "reactions"):
        entries = document.get(field)
        if not isinstance(entries, list) or not entries or any(not isinstance(entry, Mapping) for entry in entries):
            raise SchemaValidationError(f"OpenAtmos v1 {field} must be a non-empty list of objects")
    for phase in document["phases"]:
        if not isinstance(phase.get("species"), list) or not phase["species"]:
            raise SchemaValidationError("OpenAtmos v1 phase must explicitly declare its species membership")


__all__ = ["require_openatmos_v1"]
