"""MKPP Code Generation Orchestrator.

This module provides the top-level `generate_headers` function that emits
Kokkos C++ solver headers from a parsed mechanism definition. It delegates
to the Jinja2 template engine for C++ emission and to focused submodules
for tableau definitions and expression formatting.

Public API (backward-compatible):
    - generate_headers
    - generate_host_api_headers
    - SOLVER_COEFFICIENTS
    - RosenbrockTableau
    - format_eqn
    - get_A, get_C
"""

import hashlib
import json
import warnings
from pathlib import Path

from .format_eqn import format_eqn
from .model import MechanismDefinition
from .rosenbrock import SOLVER_COEFFICIENTS, RosenbrockTableau, get_A, get_C
from .template_context import build_template_context
from .template_engine import TemplateEngine

# Documented per-thread budget for the dense micro-matrix state (data-model E4, FR-007).
# The reference integrator allocates three N x N double buffers (J_values, W_values,
# LU_values). A mechanism whose footprint exceeds this budget must NOT keep them on the
# per-thread stack for a device instantiation; the plan relocates them to team-scratch and
# moves the symbolic L/U masks to device-accessible plan data. The budget is sized for the
# largest mechanism that fits a single host thread stack comfortably (N <= 104).
BYTES_PER_DOUBLE = 8
DENSE_MATRIX_BUFFERS = 3  # {J,W,LU}_values
MAX_MATRIX_BYTES_PER_THREAD = 256 * 1024  # 256 KiB per thread


def dense_matrix_bytes(num_species: int) -> int:
    """Per-thread bytes for the three dense N x N micro-matrix buffers."""
    return DENSE_MATRIX_BUFFERS * num_species * num_species * BYTES_PER_DOUBLE


def per_thread_matrix_budget(num_species: int, compact_bytes: int | None = None) -> dict:
    """Evaluate the documented per-thread memory budget for a mechanism (data-model E4).

    Returns a manifest-ready record describing the selected numeric workspace and whether
    it exceeds the budget. Exceeding the budget emits a generation-time warning: the device
    instantiation must relocate {J,W,LU}_values to team-scratch and the L/U masks to a
    device-accessible View rather than per-thread stack arrays / per-TU inline masks.
    """
    dense_footprint = dense_matrix_bytes(num_species)
    footprint = dense_footprint if compact_bytes is None else compact_bytes
    exceeds = footprint > MAX_MATRIX_BYTES_PER_THREAD
    record = {
        "max_bytes_per_thread": MAX_MATRIX_BYTES_PER_THREAD,
        "dense_matrix_bytes": dense_footprint,
        "numeric_matrix_bytes": footprint,
        "storage": "dense-reference" if compact_bytes is None else "compact-sparse",
        "num_species": num_species,
        "buffers": DENSE_MATRIX_BUFFERS,
        "exceeds_budget": exceeds,
        "plan_mask_placement": "static-constexpr-inline" if not exceeds else "device-view-relocation-pending",
    }
    if exceeds:
        warnings.warn(
            f"Per-thread dense micro-matrix footprint for {num_species}-species mechanism is "
            f"{footprint} bytes, exceeding the documented budget of "
            f"{MAX_MATRIX_BYTES_PER_THREAD} bytes (FR-007). The device instantiation must "
            f"relocate {{J,W,LU}}_values to team-scratch and the L/U masks to device-accessible "
            f"plan data instead of per-thread stack arrays. The host supernodal default path is "
            f"unchanged (research R5).",
            stacklevel=2,
        )
    return record


# Re-export public API for backward compatibility
__all__ = [
    "generate_headers",
    "generate_host_api_headers",
    "SOLVER_COEFFICIENTS",
    "RosenbrockTableau",
    "format_eqn",
    "get_A",
    "get_C",
]


def generate_host_api_headers(
    mech: MechanismDefinition,
    out_dir: str = "src/solvers",
    solver_name: str = "ros3",
) -> dict[str, str]:
    """Emit the C, C++, and Fortran host API headers and wrappers for a mechanism."""
    if not mech or not mech.species:
        raise ValueError("Cannot generate host API headers for empty mechanism")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    context = build_template_context(mech, solver_name=solver_name)
    engine = TemplateEngine()

    rendered = {
        "c_header": ("mkpp.h", engine.render("host_api/mkpp.h.j2", context)),
        "fortran_module": ("mkpp_mod.f90", engine.render("host_api/mkpp_mod.f90.j2", context)),
        "cpp_header": ("mkpp.hpp", engine.render("host_api/mkpp.hpp.j2", context)),
        "c_api_source": ("mkpp_c_api.cpp", engine.render("host_api/mkpp_c_api.cpp.j2", context)),
    }

    results = {}
    for key, (filename, content) in rendered.items():
        file_path = out_path / filename
        with open(file_path, "w") as f:
            f.write(content)
        results[key] = str(file_path)

    return results


def generate_headers(
    mech: MechanismDefinition,
    out_dir: str = "src/solvers",
    suffix: str = "",
    solver_name: str = "ros3",
    adjoint: bool = False,
    generate_host_api: bool = False,
    simd_backend: str = "native",
    emit_reference_backend: bool = False,
) -> dict[str, str]:
    """Emit the Kokkos headers and manifest artifact."""
    if not mech or not mech.species:
        raise ValueError("Cannot generate headers for empty mechanism")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Sensitivity code is always emitted into the generated header, guarded by
    # MKPP_ENABLE_ADJOINT.  That keeps a mechanism's chemistry artifact stable
    # while allowing CMake to preprocess adjoint/TLM code away in forward-only
    # builds.  ``adjoint`` remains accepted for CLI/API compatibility.
    context = build_template_context(mech, solver_name, adjoint=True, simd_backend=simd_backend)
    # Add suffix to context for filename generation
    context["suffix"] = suffix
    # Documented per-thread numeric matrix budget (data-model E4, FR-007). Evaluated at
    # generation time so a mechanism exceeding the budget surfaces a warning and a manifest
    # record driving the device team-scratch relocation plan (research R5).
    budget_record = per_thread_matrix_budget(context["num_species"], context["compact"]["numeric_bytes"])
    # The public header includes the reference factorization fragment only when
    # the unrolled-reference backend is emitted, keeping the default artifact
    # set free of the benchmark-only algebra.
    context["emit_reference_backend"] = emit_reference_backend

    # 2. Render the header via Jinja2 template engine
    engine = TemplateEngine()
    header_text = engine.render("header.j2", context)

    # 3. Write rendered output to the same file path as before
    header_path = out_path / f"{mech.name}{suffix}.hpp"
    with open(header_path, "w") as f:
        f.write(header_text)

    compiled_sources = []
    # Keep the public include at the output root while grouping its compiled
    # implementation units under a mechanism-specific directory.
    compiled_path = out_path / f"{mech.name}{suffix}"
    compiled_path.mkdir(parents=True, exist_ok=True)
    # Emit one algebra definition shared by two build modes: CPU libraries own
    # out-of-line helpers; device/header-only consumers see inline bodies. The
    # CPU public header includes only declarations to bound optimizer work.
    # The layout-aware accessor is the shared vocabulary every chunk uses to
    # address caller-owned buffers and the integrator's local scratch arrays
    # with the correct strides (LayoutLeft cell rows are strided, not
    # contiguous). It is emitted first because the rate/Jacobian fragments and
    # the public header all depend on the type it defines.
    accessor_path = compiled_path / "accessor.hpp"
    with open(accessor_path, "w") as f:
        f.write(engine.render("device_impl/accessor.hpp.j2", context))

    declarations_path = compiled_path / "host_declarations.hpp"
    declarations_path.write_text(engine.render("host_declarations.hpp.j2", context), encoding="utf-8")

    for kernel in ("rates", "jacobian"):
        fragment_path = compiled_path / f"{kernel}.hpp"
        with open(fragment_path, "w") as f:
            f.write(engine.render(f"device_impl/{kernel}.hpp.j2", context))
        source_path = compiled_path / f"{kernel}.cpp"
        with open(source_path, "w") as f:
            f.write(engine.render(f"compiled_{kernel}.cpp.j2", context))
        for stale_path in compiled_path.glob(f"{kernel}_*.cpp"):
            stale_path.unlink()
        compiled_sources.append(str(source_path))

    if emit_reference_backend:
        fragment_path = compiled_path / "factorize.hpp"
        with open(fragment_path, "w") as f:
            f.write(engine.render("device_impl/factorize.hpp.j2", context))
        source_path = compiled_path / "factorize.cpp"
        with open(source_path, "w") as f:
            f.write(engine.render("compiled_factorize.cpp.j2", context))
        compiled_sources.append(str(source_path))
    else:
        for source_path in compiled_path.glob("factorize*.*"):
            source_path.unlink()

    fragment_path = compiled_path / "solve.hpp"
    with open(fragment_path, "w") as f:
        f.write(engine.render("device_impl/solve.hpp.j2", context))
    source_path = compiled_path / "solve.cpp"
    with open(source_path, "w") as f:
        f.write(engine.render("compiled_solve.cpp.j2", context))
    compiled_sources.append(str(source_path))

    for kernel in ("supernodal_factorize", "supernodal_solve"):
        fragment_path = compiled_path / f"{kernel}.hpp"
        with open(fragment_path, "w") as f:
            f.write(engine.render(f"device_impl/{kernel}.hpp.j2", context))
        source_path = compiled_path / f"{kernel}.cpp"
        with open(source_path, "w") as f:
            f.write(engine.render(f"{kernel}.cpp.j2", context))
        compiled_sources.append(str(source_path))

    results = {"header": str(header_path), "compiled_sources": compiled_sources}

    if generate_host_api:
        api_results = generate_host_api_headers(mech, out_dir=out_dir, solver_name=solver_name)
        results.update(api_results)

    # 4. Manifest metadata emission (unchanged logic)
    manifest = {
        "mechanism": mech.name,
        "aerosol_representation": mech.aerosol_representation.value,
        "simd_backend": simd_backend,
        "checksum": hashlib.sha256(mech.name.encode()).hexdigest(),
        "per_thread_budget": budget_record,
        "artifacts": [
            {"kind": "header", "file": header_path.name},
            *({"kind": "compiled_kernel_source", "file": str(Path(source).relative_to(out_path))} for source in compiled_sources),
            {"kind": "adjoint_tlm_record", "differentiable": True},
        ],
    }

    if getattr(mech, "host_interface", None) and mech.host_interface.arrays:
        manifest["host_interface"] = {
            arr.name: {
                "rank": arr.rank,
                "layout": arr.layout,
                "lifetime": "unmanaged_borrowed_from_host" if arr.ownership == "host" else "device_owned",
            }
            for arr in mech.host_interface.arrays
        }

    partition_meta = getattr(mech, "partition_metadata", None)
    if partition_meta:
        manifest["solver_partition"] = partition_meta

    manifest_path = out_path / f"{mech.name}_manifest.json"
    plan_json_path = compiled_path / "factorization_plan.json"
    plan_markdown_path = compiled_path / "factorization_plan.md"
    plan_json_path.write_text(engine.render("factorization_plan.json.j2", context), encoding="utf-8")
    plan_markdown_path.write_text(engine.render("factorization_plan.md.j2", context), encoding="utf-8")
    manifest["factorization_plan"] = {
        "json": str(plan_json_path.relative_to(out_path)),
        "markdown": str(plan_markdown_path.relative_to(out_path)),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    results["manifest"] = str(manifest_path)
    results["factorization_plan"] = str(plan_json_path)
    results["factorization_plan_markdown"] = str(plan_markdown_path)
    return results
