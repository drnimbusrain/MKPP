# Multiphase Kinetic PreProcessor (MKPP) Engine

MKPP (Multiphase Kinetic PreProcessor) is a highly optimized, Ahead-Of-Time (AOT) Python compiler that translates atmospheric chemistry mechanisms (defined via OpenAtmos YAML) into Exascale-ready block-sparse Kokkos C++ headers for a Unified Jacobian using SymPy.

## Mechanisms Included
- **Chapman Cycle** (`chapman.yaml`)
- **Small Stratospheric** (`small_strato.yaml`)
- **Carbon** (`carbon.yaml`)
- **GOCART** (`gocart.yaml`)
- **SAPRC** mechanisms (`saprc99.yaml`, `saprcnov.yaml`)

## Building and Testing

Kokkos **5.1.1 or newer** is required. The C++ host interface does not require a
Fortran compiler; Fortran is needed only for targets that compile Fortran code.
CPU builds compile generated chemistry helpers separately by default; see
[build-time guidance](docs/how-to/reduce-native-build-times.md).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
mkdir build && cd build
cmake .. -DMKPP_ENABLE_KOKKOS_KERNELS=ON
ninja
ctest
```

### Compiler CLI Options
- `--simd-backend {native,kokkos_batched}`: Select SIMD vector engine backend for `Wide<W>` (default: `native`). `kokkos_batched` uses KokkosKernels 5.0+ hardware-tuned SIMD vector wrappers.
- `--batch-width <W>`: Set SIMD vector lane width (e.g. 4, 8, 16).

## Governance

Development on this project is governed by the project constitution. All contributors must adhere to its core principles including:
- Clarity Over Cleverness
- Defensive Programming
- Fail Fast, Fail Loudly
- High-Performance Computing & Message Passing (MPI) considerations
- GPU Acceleration & Kokkos constraints
- Zero-Copy Data Interoperability
- Scientific Hygiene & Determinism

See the Constitution and `.github/copilot-instructions.md` for complete guidelines.

## License

This project is part of NOAA-EMC Ecosystem.
See LICENSE and DISCLAIMER for details.

---

#### **Artificial Intelligence (AI) Generation & Transparency Notice**

> Pursuant to Federal guidelines and NOAA policies on the responsible deployment of Artificial Intelligence:
> 1. **AI-Assisted Code Generation:** Portions of the code, scripts, documentation, or unit tests contained within this repository may have been generated, drafted, or refactored using Artificial Intelligence (AI) and Generative AI tools (e.g., Large Language Models, AI coding assistants).
> 2. **Human Oversight & Verification:** In accordance with federal standards for scientific integrity and software quality, all AI-generated content in this repository has been subjected to human review, testing, and validation by the project maintainers prior to publication to ensure correctness, security, and adherence to NOAA standards.
