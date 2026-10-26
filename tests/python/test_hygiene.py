"""Static hygiene checks for the Kokkos host layer (feature 024, US4).

Asserts the audit findings are structurally closed:
  * FR-009: no dead integrator functor, no team scratch that is charged but
    never read.
  * FR-011: compute kernels never stream diagnostics from device code; error
    conditions travel through the status buffer and surface at the host
    boundary instead.
  * FR-013: executable entry points own the runtime lifetime through a
    ``Kokkos::ScopeGuard`` rather than a raw ``initialize``/``finalize`` pair
    that leaks on an exception path.

These are deliberately text scans over the checked-in sources: they encode
"never reintroduce" invariants and fail fast and loudly in CI without
requiring a build.
"""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MKPP_HOST_DIR = PROJECT_ROOT / "src" / "mkpp_host"
DISPATCHER = MKPP_HOST_DIR / "include" / "mkpp_host" / "dispatcher.hpp"

# Entry points that must own the Kokkos lifetime via a scope guard.  Scoped
# list: the files feature 024 audits for FR-013 (the solver-comparison runner
# and the Ex-Aero lifecycle/driver units).  Broadening this set is a policy
# change, not a hygiene fix.
SCOPE_GUARD_ENTRY_POINTS = (
    PROJECT_ROOT / "benchmarks" / "solver_comparison" / "runners" / "mkpp" / "main.cpp",
    PROJECT_ROOT / "src" / "exaero" / "src" / "src_impl" / "gocart" / "GocartSolver.cpp",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"expected source file is missing: {path}"
    return path.read_text(encoding="utf-8")


def test_dispatcher_exists() -> None:
    assert DISPATCHER.is_file(), f"dispatcher header not found at {DISPATCHER}"


def test_dead_tiled_cell_integrator_removed() -> None:
    """FR-009: the unused ``TiledCellIntegrator`` functor must stay deleted.

    The live single-step path is ``execute_mechanism_serial``/
    ``TiledCellTimeIntegrator``; the bare ``TiledCellIntegrator`` had no call
    sites.  Match the exact struct name so the similarly-named live functors
    (``TiledCellTimeIntegrator``, ``ParamsTiledCellIntegrator``) do not trip it.
    """
    source = _read(DISPATCHER)
    assert not re.search(
        r"\bTiledCellIntegrator\b", source
    ), "dead functor TiledCellIntegrator reintroduced in dispatcher.hpp (FR-009)"


def test_unread_team_scratch_removed() -> None:
    """FR-009: team scratch that is charged but never read must not return.

    ``scratch_species`` was bound from ``member.team_scratch(0)`` inside the
    batched integrator and never indexed, while the launch still paid the
    ``set_scratch_size`` charge.  Both the binding and the charge must stay gone.
    """
    source = _read(DISPATCHER)
    assert "scratch_species" not in source, "never-read scratch_species binding reintroduced in dispatcher.hpp (FR-009)"
    assert "set_scratch_size" not in source, "team scratch charge reintroduced without a consumer in dispatcher.hpp (FR-009)"


def _kernel_regions(source: str) -> list[str]:
    """Yield the text of every functor ``operator()`` body in ``source``.

    A device region starts at ``void operator()(...) const {`` and ends at the
    matching closing brace of that function body.  Brace counting is sufficient
    here because the kernels use balanced braces; this is a hygiene heuristic,
    not a parser.
    """
    regions: list[str] = []
    pattern = re.compile(r"KOKKOS_INLINE_FUNCTION\s+void operator\(\)")
    for match in pattern.finditer(source):
        open_idx = source.find("{", match.end())
        if open_idx == -1:
            continue
        depth = 0
        for idx in range(open_idx, len(source)):
            if source[idx] == "{":
                depth += 1
            elif source[idx] == "}":
                depth -= 1
                if depth == 0:
                    regions.append(source[open_idx : idx + 1])
                    break
    return regions


def test_no_device_side_streaming_in_kernels() -> None:
    """FR-011: no per-cell diagnostics may stream from inside a device functor.

    Scans every ``operator()`` body for ``Kokkos::printf`` (the audit finding)
    and the C/C++ stream primitives the C++ guidelines forbid in kernels
    (``cpp.instructions.md`` section 3: compute-only, no device I/O).
    """
    source = _read(DISPATCHER)
    forbidden = (
        "Kokkos::printf",
        "printf(",
        "std::cout",
        "std::cerr",
    )
    for region in _kernel_regions(source):
        for token in forbidden:
            assert token not in region, f"device-side I/O token {token!r} found inside a kernel body (FR-011)"


def test_no_kokkos_printf_anywhere_in_host_layer() -> None:
    """FR-011: the whole host layer must be free of in-kernel printfs."""
    offenders: list[str] = []
    for path in sorted(MKPP_HOST_DIR.rglob("*.hpp")):
        if "Kokkos::printf" in _read(path):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"Kokkos::printf present in host headers: {offenders}"


def test_entry_points_use_scope_guard() -> None:
    """FR-013: audited entry points must guard the runtime, not pair init/finalize.

    A raw ``Kokkos::initialize()``/``Kokkos::finalize()`` pair leaks the runtime
    (or double-finalizes) when a body throws between them; ``Kokkos::ScopeGuard``
    is exception-safe.  ``Kokkos::initialize(argc, argv)`` is only acceptable as
    the ScopeGuard's own forwarding argument, which the guard check below covers.
    """
    for path in SCOPE_GUARD_ENTRY_POINTS:
        source = _read(path)
        assert "Kokkos::ScopeGuard" in source, (
            f"{path.relative_to(PROJECT_ROOT)} does not own the Kokkos lifetime " "through a ScopeGuard (FR-013)"
        )
        assert "Kokkos::finalize()" not in source, (
            f"{path.relative_to(PROJECT_ROOT)} still calls Kokkos::finalize() " "manually instead of relying on ScopeGuard (FR-013)"
        )
