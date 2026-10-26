"""Strict loading and validation for solver-comparison contracts."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

_CONTRACT_DIRECTORY = Path(__file__).resolve().parent / "contracts"


class SchemaValidationError(ValueError):
    """Raised when a benchmark contract document is structurally invalid."""


@cache
def _validator(schema_name: str) -> Draft202012Validator:
    schema_path = _CONTRACT_DIRECTORY / schema_name
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
    except (OSError, json.JSONDecodeError, SchemaError) as exc:
        raise RuntimeError(f"FATAL ERROR: cannot load benchmark schema {schema_path}: {exc}") from exc
    return Draft202012Validator(schema)


def _location(error: Any) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return path or "document"


def _validate(document: Mapping[str, Any], schema_name: str) -> None:
    errors = sorted(
        _validator(schema_name).iter_errors(document),
        key=lambda error: (tuple(str(part) for part in error.absolute_path), error.message),
    )
    if errors:
        error = errors[0]
        raise SchemaValidationError(f"{_location(error)}: {error.message}")


def _require_finite(value: Any, location: str = "document") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise SchemaValidationError(f"{location}: number must be finite")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _require_finite(child, f"{location}.{key}")
    elif isinstance(value, list | tuple):
        for index, child in enumerate(value):
            _require_finite(child, f"{location}.{index}")


def validate_runner_result(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and return one native runner result without mutating it.

    JSON Schema supplies the structural contract. The additional checks enforce
    scientific and operational invariants that draft 2020-12 cannot express.
    """

    _validate(document, "runner-result.schema.json")
    _require_finite(document)

    timing = document["timing"]
    if timing["boundary"] == "steady_state_solve" and not timing["synchronized"]:
        raise SchemaValidationError("timing.synchronized: steady-state solve timing must include completion synchronization")

    for name, counter in document["work"].items():
        if not counter["available"] and not counter.get("reason", "").strip():
            raise SchemaValidationError(f"work.{name}.reason: unavailable counter must explain why")

    if document["status"] != "success" and not document["diagnostics"]:
        raise SchemaValidationError("diagnostics: non-success result requires actionable diagnostics")

    return document


def validate_manifest(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a canonical mechanism manifest."""

    _validate(document, "manifest.schema.json")
    _require_finite(document)
    return document


def validate_scenario(document: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate a canonical benchmark scenario."""

    _validate(document, "scenario.schema.json")
    _require_finite(document)
    return document
