"""Shared implementation for isolated native solver adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mkpp.benchmark.adapters.base import (
    AdapterProbe,
    AdapterRunRequest,
    ExecutionError,
    PreparationError,
    PreparedAdapterRun,
    ProbeError,
    SupportError,
)
from mkpp.benchmark.models import (
    BuildProvenance,
    HashIdentity,
    ResourcePolicy,
    RunStatus,
    SolverId,
    SolverRun,
    StateCheckpoint,
    TimingBoundary,
    WorkCounter,
)
from mkpp.benchmark.orchestrator import NativeRunnerError, invoke_json_runner


@dataclass(frozen=True, slots=True)
class NativeExecutableAdapter:
    """Bind one solver identity to one immutable native executable.

    The runner, rather than the Python layer, performs chemistry and integration.
    This class deliberately has no alternate numerical implementation.
    """

    solver_id: SolverId
    executable: Path
    version: str
    revision: str
    supported_manifests: tuple[str, ...]
    timeout_seconds: float = 300.0

    def probe(self) -> AdapterProbe:
        path = self.executable.resolve()
        if not path.is_file():
            raise ProbeError(self.solver_id, f"runner does not exist: {path}")
        if not path.stat().st_mode & 0o111:
            raise ProbeError(self.solver_id, f"runner is not executable: {path}")
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        return AdapterProbe(
            solver_id=self.solver_id,
            version=self.version,
            revision=self.revision,
            executable_fingerprint=fingerprint,
            backend="native",
            capabilities=("persistent_runner", "named_state", "json_result"),
        )

    def validate_support(self, request: AdapterRunRequest) -> None:
        if request.manifest_id not in self.supported_manifests:
            raise SupportError(
                self.solver_id,
                f"manifest {request.manifest_id!r} is not admitted",
            )

    def prepare(self, request: AdapterRunRequest) -> PreparedAdapterRun:
        self.validate_support(request)
        probe = self.probe()
        try:
            working_directory = str(self.executable.resolve().parent)
        except OSError as exc:
            raise PreparationError(self.solver_id, str(exc)) from exc
        return PreparedAdapterRun(
            request=request,
            command=(
                str(self.executable.resolve()),
                "--manifest",
                request.manifest_id,
                "--scenario",
                request.scenario_id,
                "--configuration",
                request.configuration_id,
            ),
            working_directory=working_directory,
            input_sha256=probe.executable_fingerprint,
        )

    def run(self, prepared: PreparedAdapterRun) -> SolverRun:
        """Run the native subject and decode its already schema-validated evidence."""

        try:
            result = invoke_json_runner(prepared, self.solver_id, timeout_seconds=self.timeout_seconds)
        except NativeRunnerError as exc:
            raise ExecutionError(self.solver_id, str(exc)) from exc
        return _decode_solver_run(dict(result), prepared.request)


def _hash_identity(value: dict[str, Any]) -> HashIdentity:
    return HashIdentity(id=value["id"], version=value.get("version", ""), sha256=value["sha256"])


def _decode_solver_run(result: dict[str, Any], request: AdapterRunRequest) -> SolverRun:
    """Translate the frozen wire contract without changing any numeric evidence."""

    build = result["build"]
    resources = result["resources"]
    timing = result["timing"]
    checkpoints = tuple(
        StateCheckpoint(
            time_seconds=item["time_seconds"],
            values=tuple(sorted(item["values"].items())),
        )
        for item in result["state"]["checkpoints"]
    )
    counters = tuple(
        WorkCounter(
            name=name,
            available=value["available"],
            value=value.get("value"),
            unavailable_reason=value.get("reason"),
        )
        for name, value in sorted(result["work"].items())
    )
    matching_material = "|".join(
        (
            result["manifest"]["sha256"],
            result["scenario"]["sha256"],
            timing["boundary"],
        )
    ).encode()
    return SolverRun(
        campaign_id=result["campaign_id"],
        run_id=result["run_id"],
        block_id=result["block"],
        repetition=request.repetition,
        order_position=result["order_position"],
        matching_key_sha256=hashlib.sha256(matching_material).hexdigest(),
        solver_id=SolverId(result["solver"]["id"]),
        manifest=_hash_identity(result["manifest"]),
        scenario=_hash_identity(result["scenario"]),
        configuration=_hash_identity(result["configuration"]),
        build=BuildProvenance(
            binary_sha256=build["binary_sha256"],
            asset_sha256=tuple(build["asset_sha256"]),
            compiler=build["compiler"],
            flags=tuple(build["flags"]),
            precision=build["precision"],
        ),
        resources=ResourcePolicy(
            hardware_id=resources["hardware_id"],
            backend=result["solver"]["backend"],
            physical_cores=resources["physical_cores"],
            threads=resources["threads"],
            affinity=resources["affinity"],
            numa_policy=resources["numa_policy"],
        ),
        timing_boundary=TimingBoundary(timing["boundary"]),
        common_work_count=timing["cell_steps"],
        elapsed_seconds=timing["elapsed_ms"] / 1000.0,
        clock=timing["clock"],
        synchronized=timing["synchronized"],
        synchronization_method=timing["synchronization"],
        lifecycle_seconds=tuple((name, milliseconds / 1000.0) for name, milliseconds in sorted(timing["lifecycle_ms"].items())),
        checkpoints=checkpoints,
        work_counters=counters,
        status=RunStatus(result["status"]),
        diagnostics=tuple(result["diagnostics"]),
    )
