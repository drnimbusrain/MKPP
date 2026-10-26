"""Layout-aware accessor tests for the generated solver kernels.

Feature 024 (Kokkos & Kokkos Kernels compliance hardening), FR-001/FR-002/FR-003.

The production C-ABI binds caller buffers as ``(num_cells, num_species)``
``Kokkos::LayoutLeft`` views (the Fortran zero-copy mandate, Constitution IV).
For such a view the per-cell species row is *strided by* ``num_cells`` — it is
NOT contiguous — so a generated chunk that dereferences ``state.data()`` with
flat ``state[k]`` / ``jacobian[row*N + col]`` indexing reads the wrong memory
(a neighbour cell's species) whenever a caller binds ``LayoutLeft``.

These tests pin the *structural* guarantee that closes the bug: every generated
mechanism reads and writes caller buffers through a layout-aware accessor
(``StrideView`` per contracts/codegen-accessor.md) instead of bare flat pointer
indexing.  They are written RED-first: they assert the accessor emission that
the current flat codegen does not yet produce.

**Validates: Requirements FR-001, FR-002 (contract codegen-accessor.md).**
"""

import re
from pathlib import Path

import pytest
from mkpp.codegen import generate_headers
from mkpp.parser import load_mechanism

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHAPMAN = PROJECT_ROOT / "mechanisms/openatmos/chapman/mechanism.json"


@pytest.fixture(scope="module")
def chapman_artifacts(tmp_path_factory):
    if not CHAPMAN.exists():
        pytest.skip(f"chapman mechanism file not found at {CHAPMAN}")
    mech = load_mechanism(str(CHAPMAN))
    out_dir = tmp_path_factory.mktemp("chapman_layout_gen")
    results = generate_headers(mech, out_dir=str(out_dir), solver_name="ros3", emit_reference_backend=True)
    header = Path(results["header"]).read_text()
    impl_dir = Path(results["header"]).parent / "chapman"
    fragments = {p.name: p.read_text() for p in sorted(impl_dir.glob("*.hpp"))}
    return {"header": header, "fragments": fragments, "results": results, "out_dir": out_dir, "impl_dir": impl_dir}


# ---------------------------------------------------------------------------
# The accessor fragment must exist and be the layout-aware mechanism
# ---------------------------------------------------------------------------


def test_accessor_fragment_is_emitted(chapman_artifacts):
    """A dedicated accessor fragment defines the layout-aware StrideView type."""
    accessor_path = chapman_artifacts["impl_dir"] / "accessor.hpp"
    assert accessor_path.exists(), "codegen must emit an accessor.hpp fragment for the layout-aware view"
    text = accessor_path.read_text()
    assert re.search(r"struct\s+StrideView", text), "accessor.hpp must define struct StrideView"
    # The accessor must be device-portable: every operator is KOKKOS_INLINE_FUNCTION.
    assert "KOKKOS_INLINE_FUNCTION" in text, "StrideView accessors must be KOKKOS_INLINE_FUNCTION"


def test_header_includes_accessor_fragment(chapman_artifacts):
    """The public header includes the accessor so a device instantiation sees the type."""
    header = chapman_artifacts["header"]
    assert re.search(r'#include\s+"[^"]*accessor\.hpp"', header), "header.j2 must include the accessor fragment"


# ---------------------------------------------------------------------------
# Chunks must index through the accessor, not bare flat pointer arithmetic
# ---------------------------------------------------------------------------


def test_rates_chunk_uses_accessor_not_flat_state(chapman_artifacts):
    """compute_rates chunk reads state via accessor() and writes rates via accessor()."""
    rates = chapman_artifacts["fragments"].get("rates.hpp", "")
    assert rates, "expected a rates.hpp fragment"
    # A flat read of the input state (state[i] with square brackets) is the bug.
    flat_state_read = re.search(r"\bstate\[\s*\d+\s*\]", rates)
    assert flat_state_read is None, f"rates chunk still dereferences state with flat indexing: {flat_state_read}"
    # The accessor form is state(i).
    assert re.search(r"\bstate\(\s*\d+\s*\)", rates), "rates chunk must read state through the accessor state(i)"
    # Output writes go through the accessor too.
    assert re.search(r"\brates\(\s*\d+\s*\)\s*=", rates), "rates chunk must write through rates(i) accessor"


def test_jacobian_chunk_uses_accessor_not_flat_index(chapman_artifacts):
    """compute_jacobian chunk writes jacobian(row, col) instead of jacobian[row*N + col]."""
    jac = chapman_artifacts["fragments"].get("jacobian.hpp", "")
    assert jac, "expected a jacobian.hpp fragment"
    flat_write = re.search(r"\bjacobian\[\s*\d+\s*\*\s*\d+\s*\+\s*\d+\s*\]", jac)
    assert flat_write is None, f"jacobian chunk still uses flat row*N+col indexing: {flat_write}"
    assert re.search(
        r"\bjacobian\.at_slot\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)\s*=", jac
    ), "jacobian chunk must write through the layout-aware accessor jacobian.at_slot(slot, row, col)"


# ---------------------------------------------------------------------------
# The caller-facing handoff must pass a layout-aware view, not .data()
# ---------------------------------------------------------------------------


def test_compute_kernel_wires_accessor_not_state_data(chapman_artifacts):
    """compute_rates/compute_jacobian pass a stride-aware view to the chunk, not state.data()."""
    header = chapman_artifacts["header"]
    # Inside the generated compute_rates/compute_jacobian bodies the chunk call must
    # not hand the raw contiguous pointer for the caller-owned state buffer.
    # Capture the full argument list (no ';' inside a single chunk call) so the
    # whole handoff is inspected, not just the first nested 'state.data()'.
    calls = list(re.finditer(r"detail::compute_(?:rates|jacobian)_chunk_\d+\s*\(([^;]*)\);", header))
    assert calls, "expected generated compute_rates/compute_jacobian to call the detail chunks"
    for call in calls:
        args = call.group(1)
        assert "state.data()" not in args, f"chunk call still passes state.data(): {args[:80]!r}"


def test_stride_view_carries_strides(chapman_artifacts):
    """StrideView records strides so operator() reproduces the view's layout (static_assert)."""
    accessor_path = chapman_artifacts["impl_dir"] / "accessor.hpp"
    if not accessor_path.exists():
        pytest.skip("accessor.hpp not yet emitted")
    text = accessor_path.read_text()
    assert re.search(r"static_assert", text), "accessor must static_assert it reproduces the view strides"
