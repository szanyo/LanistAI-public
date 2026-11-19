# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Freeze and validate one GUI-selected grouped evaluation before worker start."""

from __future__ import annotations

import json
import platform
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Literal, Protocol, cast

from pydantic import BaseModel

from lanista.foundation.configuration import NestedDictWrapper
from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationSweepProgress,
    LibraryEvaluationScope,
    RunConfigurationSnapshot,
)
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EvaluationSweepRequest,
    checkpoint_digest,
)
from lanista.reinforcengine.rl_model.evaluation.library_application_service import (
    EvaluationLibraryApplicationService,
)
from lanista.reinforcengine.rl_model.evaluation.runtime_execution import (
    LibraryEvaluationExecution,
)
from lanista.reinforcengine.rl_model.execution_contracts import EvaluationPacing
from lanista.reinforcengine.rl_model.model import BaseModelRuntime


class _CheckpointArtifact(Protocol):
    path: Path | None
    exists: bool


class _CheckpointCatalog(Protocol):
    def read_artifact(self, session_name: str, checkpoint_name: str) -> _CheckpointArtifact: ...


def _software_versions(framework: str) -> dict[str, str]:
    packages = ("ray",) if framework == "ray" else ("stable-baselines3",)
    result = {"python": platform.python_version()}
    for package in packages:
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _configuration_json_default(value: object) -> object:
    """Serialize supported configuration values without unstable object reprs."""

    if isinstance(value, NestedDictWrapper):
        return value.get_data()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(f"Unsupported runtime configuration value: {type(value).__name__}.")


def prepare_library_execution(
    *,
    runtime: BaseModelRuntime,
    scope: LibraryEvaluationScope,
    library_service: EvaluationLibraryApplicationService,
    checkpoint_catalog: _CheckpointCatalog,
    session_name: str | None,
    checkpoint_name: str | None,
    pacing: EvaluationPacing,
    progress_callback: Callable[[EvaluationSweepProgress], None] | None = None,
) -> LibraryEvaluationExecution:
    """Reverify all selected identities and freeze the effective headless policy."""

    if not session_name or not checkpoint_name:
        raise ValueError("A selected session and checkpoint are required for library evaluation.")
    if pacing != "maximum-throughput":
        raise ValueError("Library inference currently supports maximum-throughput pacing only.")
    resolved = library_service.resolve_selection(scope)
    configuration = runtime.get_manifest().get_configuration()
    model_root = Path(str(getattr(configuration, "model_folder_path", "")))
    if model_root != resolved.model_root:
        raise ValueError("The selected model changed before library evaluation started.")
    framework = str(getattr(runtime, "framework_uuid", "") or "").lower()
    if framework not in {"ray", "sb3"}:
        raise ValueError("Library evaluation requires a supported active framework.")
    artifact = checkpoint_catalog.read_artifact(session_name, checkpoint_name)
    if not artifact.exists or artifact.path is None:
        raise ValueError("The selected checkpoint is missing or invalid.")
    digest = checkpoint_digest(artifact.path)
    # The digest check is potentially expensive; reject a model/library switch during it.
    if library_service.resolve_selection(scope).model_root != model_root:
        raise ValueError("The selected model changed while preparing library evaluation.")
    effective_configuration = json.loads(
        json.dumps(
            configuration.get_data(),
            default=_configuration_json_default,
            allow_nan=False,
            sort_keys=True,
        )
    )
    if not isinstance(effective_configuration, dict):
        raise ValueError("Effective runtime configuration is not a mapping.")
    model_id = str(getattr(configuration, "model_uuid", "") or "")
    run_id = f"library-{datetime.now(UTC):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
    versions = _software_versions(framework)
    framework_id = cast(Literal["ray", "sb3"], framework)
    snapshot = RunConfigurationSnapshot(
        model_id=model_id,
        framework=framework_id,
        session_id=session_name,
        checkpoint_id=checkpoint_name,
        checkpoint_sha256=digest,
        library_id=scope.library_id,
        library_manifest_sha256=scope.manifest_sha256,
        deterministic_inference=True,
        repetitions_per_scenario=scope.repetitions_per_scenario,
        headless_pacing={"mode": pacing},
        effective_runtime_configuration=effective_configuration,
        software_versions=versions,
    )
    request = EvaluationSweepRequest(
        run_id=run_id,
        framework=framework_id,
        model_id=model_id,
        session_id=session_name,
        checkpoint_id=checkpoint_name,
        checkpoint_path=artifact.path,
        checkpoint_sha256=digest,
        library_id=scope.library_id,
        library_manifest_sha256=scope.manifest_sha256,
        output_root=resolved.run_parent,
        deterministic_inference=True,
        repetitions_per_scenario=scope.repetitions_per_scenario,
        software_versions=versions,
        run_configuration=snapshot,
    )
    return LibraryEvaluationExecution(
        scope=scope,
        library_storage=resolved.library_storage,
        sweep_request=request,
        progress_callback=progress_callback,
        report_publisher=library_service.publish_report,
    )


__all__ = ["prepare_library_execution"]
