#!/bin/bash
# ==============================================================================
# MKPP Mechanism Compilation Script
#
# Compiles all atmospheric chemical mechanisms into optimized C++ Kokkos headers
# with full feature support (Adjoint/TLM, Manifest, Analysis Report, SymPy CSE).
# ==============================================================================

set -euo pipefail

main() {
    # Ensure execution from repository root
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local repo_root
    repo_root="$(cd "${script_dir}/.." && pwd)"
    cd "${repo_root}" || exit 1

    # Prefer a known-good interpreter instead of silently selecting a local
    # virtualenv.  A broken optional reporting dependency in that virtualenv
    # must not make chemistry compilation crash.  Callers can pin an
    # interpreter with MKPP_PYTHON when needed.
    local python_bin="${MKPP_PYTHON:-python3}"
    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "FATAL ERROR: Python interpreter '${python_bin}' was not found."
        exit 1
    fi

    local test_env="tests/integration/e2e_validation/data/env.yaml"
    if [[ ! -f "${test_env}" ]]; then
        test_env="example_env.yaml"
    fi

    local out_dir="mkpp-generated/"
    mkdir -p "${out_dir}"

    # Reporting imports Matplotlib.  Keep its cache in a writable, disposable
    # location so this reproducibility check works in containers and CI.
    export MPLCONFIGDIR="${TMPDIR:-/tmp}/mkpp-matplotlib"
    mkdir -p "${MPLCONFIGDIR}"

    local mechanisms=(
        "mechanisms/openatmos/chapman/mechanism.json"
        "mechanisms/openatmos/small_strato/mechanism.json"
        "mechanisms/openatmos/carbon/mechanism.json"
        "mechanisms/openatmos/gocart/mechanism.json"
        "mechanisms/openatmos/saprc99/mechanism.json"
        "mechanisms/openatmos/saprcnov/mechanism.json"
        "mechanisms/openatmos/saprc99_mini/mechanism.json"
        "mechanisms/openatmos/ts1/mechanism.json"
        "mechanisms/openatmos/cracmm2/mechanism.json"
    )

    local mech
    for mech in "${mechanisms[@]}"; do
        if [[ ! -f "${mech}" ]]; then
            echo "WARNING: Mechanism file ${mech} not found, skipping."
            continue
        fi

        echo "========================================"
        echo "Compiling ${mech}..."
        echo "========================================"

        # --no-cache is required for reproducible artifact generation: the
        # content-addressable cache keys on mechanism input only, not on MKPP
        # compiler source, so a cached lowering result could mask code changes
        # and desynchronize committed headers from CI (which has no cache).
        if ! "${python_bin}" -m mkpp.cli compile "${mech}" \
            --test-env "${test_env}" \
            --out "${out_dir}" \
            --adjoint \
            --emit-manifest \
            --report \
            --no-cache \
            --verbose; then
            echo "FATAL ERROR: Failed to compile ${mech}"
            exit 1
        fi
    done

    echo "========================================"
    echo "All mechanisms compiled successfully."
    echo "========================================"
}

main "$@"
