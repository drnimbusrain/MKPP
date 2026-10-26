"""Content-addressable cache for MKPP lowering results."""

import hashlib
import importlib.metadata
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sympy


@dataclass
class CacheKey:
    """Identifies a unique compilation input."""

    yaml_hash: str  # SHA-256 hex digest of file contents
    mkpp_version: str  # package version string


@dataclass
class CacheEntry:
    """Stores all expensive lowering results for a given input."""

    key: CacheKey
    species_map: list[str]
    jacobian_matrix: Any  # sp.Matrix (pickled)
    lu_plan: Any  # SymbolicLUPlan
    f_implicit: Any  # sp.Matrix
    f_explicit: Any  # sp.Matrix
    cse_replacements: list[tuple[Any, Any]] | None = None
    cse_reduced: list[Any] | None = None


class CacheManager:
    """Content-addressable disk cache for lowering artifacts.

    Cache files live under ``cache_dir`` (default ``.mkpp_cache/``) and are
    keyed by the first 16 hex characters of the SHA-256 of the input YAML
    concatenated with the MKPP version string.
    """

    def __init__(self, cache_dir: Path = Path(".mkpp_cache")):
        self.cache_dir = cache_dir

    def compute_key(self, yaml_path: Path, *, convert_kinetic_units_to_si: bool = False) -> CacheKey:
        """SHA-256 of file contents + MKPP version + SymPy version + compiler source.

        The lowering pipeline source (``lowering.py``, ``partitioning.py``) is
        hashed so that any change to the compiler invalidates cached symbolic
        results.  Without this, edits to MKPP itself would silently reuse stale
        Jacobians and desynchronize regenerated artifacts from CI.
        """
        file_bytes = yaml_path.read_bytes()
        try:
            version = importlib.metadata.version("mkpp")
        except importlib.metadata.PackageNotFoundError:
            version = "dev"
        lowering_source = Path(__file__).parent / "lowering.py"
        parser_source = Path(__file__).parent / "parser.py"
        codegen_source = Path(__file__).parent / "codegen.py"
        source_hash = b""
        for src in (lowering_source, parser_source, codegen_source):
            if src.exists():
                source_hash += src.read_bytes()
        conversion_option = b"kinetic_units_to_si=1" if convert_kinetic_units_to_si else b"kinetic_units_to_si=0"
        hash_input = file_bytes + version.encode() + sympy.__version__.encode() + source_hash + conversion_option
        yaml_hash = hashlib.sha256(hash_input).hexdigest()
        return CacheKey(yaml_hash=yaml_hash, mkpp_version=version)

    def lookup(self, key: CacheKey) -> CacheEntry | None:
        """Return deserialized entry or None. Handle corruption gracefully."""
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                entry = pickle.load(f)
            if not isinstance(entry, CacheEntry):
                path.unlink()
                return None
            return entry
        except (
            pickle.UnpicklingError,
            EOFError,
            AttributeError,
            ModuleNotFoundError,
        ):
            path.unlink(missing_ok=True)
            return None

    def store(self, key: CacheKey, entry: CacheEntry) -> Path:
        """Serialize with pickle, return cache file path."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._key_to_path(key)
        with open(path, "wb") as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        return path

    def _key_to_path(self, key: CacheKey) -> Path:
        return self.cache_dir / f"{key.yaml_hash[:16]}.pkl"
