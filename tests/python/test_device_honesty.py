"""Static checks for truthful Kokkos device-callability claims."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ISORROPIA = ROOT / "src" / "isorropialite"
EXAERO = ROOT / "src" / "exaero"


def _source_files(root: Path) -> list[Path]:
    # Audit the auxiliary implementations, not their vendored dependencies
    # (e.g. EX-aero's extern/mam4xx submodule). Those have independent
    # portability policies and are not part of this device-callability gate.
    return [
        path
        for pattern in ("*.cpp", "*.hpp", "*.h")
        for path in root.rglob(pattern)
        if "extern" not in path.relative_to(root).parts
    ]


def test_source_files_exclude_vendored_dependencies(tmp_path: Path) -> None:
    owned = [tmp_path / "src" / "driver.cpp", tmp_path / "include" / "driver.hpp", tmp_path / "include" / "api.h"]
    vendored = tmp_path / "extern" / "dependency" / "src" / "kernel.hpp"
    for path in [*owned, vendored]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("KOKKOS_INLINE_FUNCTION void kernel() { std::log(1.0); }")

    assert set(_source_files(tmp_path)) == set(owned)


def _annotated_bodies(source: str) -> list[str]:
    bodies = []
    for match in re.finditer(r"KOKKOS_INLINE_FUNCTION\b", source):
        opening = source.find("{", match.end())
        if opening == -1:
            continue
        depth = 1
        cursor = opening + 1
        while depth and cursor < len(source):
            depth += (source[cursor] == "{") - (source[cursor] == "}")
            cursor += 1
        if depth == 0:
            bodies.append(source[opening + 1 : cursor - 1])
    return bodies


def test_no_fake_kokkos_inline_definition() -> None:
    definitions = []
    for path in _source_files(ISORROPIA) + _source_files(EXAERO):
        source = path.read_text()
        if re.search(r"^\s*#\s*define\s+KOKKOS_INLINE_FUNCTION\s+inline\b", source, re.MULTILINE):
            definitions.append(str(path.relative_to(ROOT)))

    assert definitions == []


def test_device_annotated_bodies_use_portable_math() -> None:
    forbidden = re.compile(r"\bstd::(?:log|exp|max|min|abs|pow|sqrt|fmax|fmin|cbrt)\s*\(")
    violations = []
    for path in _source_files(ISORROPIA) + _source_files(EXAERO):
        for body in _annotated_bodies(path.read_text()):
            if forbidden.search(body):
                violations.append(str(path.relative_to(ROOT)))

    assert violations == []


def test_auxiliary_call_chain_has_no_undefined_device_symbols() -> None:
    declarations = []
    for path in _source_files(EXAERO):
        source = path.read_text()
        if re.search(r"extern\s+\"C\"[^;]*\b\w+_device\s*\(", source, re.DOTALL):
            declarations.append(str(path.relative_to(ROOT)))

    assert declarations == []


def test_auxiliary_engines_are_honestly_host_confined() -> None:
    # The AuxiliaryEngines drivers exist but are host-confined: they wrap host
    # bind(c) Fortran kernels that cannot run inside a device launch. The
    # module must therefore never advertise device callability, and it must
    # fail loud (never a silent no-op) when an engine was not compiled in.
    header = (EXAERO / "include" / "auxiliary_engines.hpp").read_text()
    impl = (EXAERO / "src" / "auxiliary_engines.cpp").read_text()

    # 1. The team functors must not claim device portability (code lines only;
    # the file's explanatory comments legitimately discuss the annotation).
    annotated = [
        line
        for line in impl.splitlines()
        if re.search(r"\bKOKKOS_INLINE_FUNCTION\b", line) and not line.lstrip().startswith(("//", "///"))
    ]
    assert annotated == [], "auxiliary functor bodies must not be KOKKOS_INLINE_FUNCTION"

    # 2. Dispatch is structurally pinned to the host execution space.
    assert "DefaultHostExecutionSpace" in header
    assert "DefaultHostExecutionSpace" in impl
    assert "Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>" not in header
    assert "TeamPolicy<ExecSpace>" not in impl

    # 3. A device backend combined with an engine option is a compile error.
    assert impl.count("is host-confined: the") >= 2

    # 4. With an engine option OFF the entry points throw instead of launching
    #    an empty team (no silent no-op compute path).
    assert "not compiled in" in impl
    assert "silent no-op" in impl

    # 5. The engine symbols carry no aspirational device suffix (code lines;
    # comments may explain why they must not).
    device_named = [
        line for line in impl.splitlines() if re.search(r"\w+_device\b", line) and not line.lstrip().startswith(("//", "///"))
    ]
    assert device_named == []
