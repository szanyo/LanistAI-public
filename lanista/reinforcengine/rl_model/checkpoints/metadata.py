# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Versioned checkpoint metadata persistence.

These low-level helpers do not acquire a storage lock. Admin and runtime
callers must hold their canonical models-root mutation lock across the complete
artifact, sidecar, and session-metadata sequence.
"""

import os
from typing import Any, Protocol

from lanista.reinforcengine.application_errors import PersistenceError
from lanista.reinforcengine.rl_model.metadata import timestamps as metadata_timestamps
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.sessions.configuration_wrapper import (
    SessionConfigurationWrapper,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    CHECKPOINT_MANIFEST_FILE_NAME,
    find_existing_session_manifest_path,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_yaml_file,
    read_yaml_mapping_file,
)

_REQUIRED_RESTORE_FIELDS = frozenset(
    {
        "name",
        "kind",
        "framework_uuid",
        "scenario_index",
        "curriculum_state",
        "step_count",
    }
)


class CheckpointMetadataError(PersistenceError):
    """A selected checkpoint has no usable canonical metadata sidecar."""


class CheckpointStateReadError(PersistenceError):
    """Mandatory runtime state could not be captured for a checkpoint."""


class _WarningLogger(Protocol):
    """Define the typed boundary for warning logger."""

    def warning(self, message: str) -> object:
        """Record a non-fatal metadata projection failure."""


def checkpoint_manifest_path(
    checkpoint_path: str | os.PathLike[str],
) -> str:
    """Checkpoint the manifest path owned by this checkpoint metadata module."""
    normalized_path = os.fspath(checkpoint_path)
    if os.path.isdir(normalized_path):
        return os.path.join(normalized_path, CHECKPOINT_MANIFEST_FILE_NAME)
    return f"{normalized_path}.{CHECKPOINT_MANIFEST_FILE_NAME}"


def _safe_scalar(value: Any) -> Any:
    """Return the safe scalar value owned by this checkpoint metadata module."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict):
        return {str(key): _safe_scalar(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_safe_scalar(item) for item in value]
    return str(value)


def write_checkpoint_manifest(
    checkpoint_path: str,
    *,
    checkpoint_name: str,
    checkpoint_kind: str,
    framework_uuid: str,
    scenario_index: int | None = None,
    curriculum_state: Any = None,
    step_count: int | None = None,
    parent_checkpoint_name: str | None = None,
    saved_at: str | None = None,
    annotation: str | None = None,
) -> str:
    """Write one checkpoint sidecar under a caller-owned storage mutation."""

    manifest_path = checkpoint_manifest_path(checkpoint_path)
    manifest_parent = os.path.dirname(manifest_path)
    if manifest_parent:
        os.makedirs(manifest_parent, exist_ok=True)

    payload = {
        "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
        "checkpoint": {
            "name": checkpoint_name,
            "kind": checkpoint_kind,
            "framework_uuid": framework_uuid,
            "scenario_index": scenario_index,
            "curriculum_state": _safe_scalar(curriculum_state),
            "step_count": step_count,
            "parent_checkpoint_name": parent_checkpoint_name,
            "saved_at": saved_at or metadata_timestamps.utc_now_iso(),
            "annotation": _safe_scalar(annotation),
        },
    }

    validate_versioned_metadata_payload(MetadataKind.CHECKPOINT_MANIFEST, payload)

    atomic_write_yaml_file(
        manifest_path,
        payload,
        verify_reader=read_yaml_mapping_file,
        smoke_check_reader=read_yaml_mapping_file,
    )
    return manifest_path


def _read_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read the checkpoint manifest without changing this checkpoint metadata module's state."""
    manifest_path = checkpoint_manifest_path(checkpoint_path)
    if not os.path.isfile(manifest_path):
        raise CheckpointMetadataError("Checkpoint metadata is missing for the selected checkpoint.")
    try:
        payload = read_yaml_mapping_file(manifest_path)
        validated_payload = validate_versioned_metadata_payload(
            MetadataKind.CHECKPOINT_MANIFEST,
            payload,
        )
    except Exception as exc:
        raise CheckpointMetadataError(
            "Checkpoint metadata is unreadable or uses an unsupported schema."
        ) from exc
    return dict(validated_payload["checkpoint"])


def load_checkpoint_manifest(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read optional presentation metadata without migrating durable state.

    Catalog views may still derive labels from checkpoint filenames when a
    sidecar is unavailable. Restore code must use
    :func:`load_checkpoint_restore_manifest`, which fails closed.
    """

    try:
        return _read_checkpoint_manifest(checkpoint_path)
    except CheckpointMetadataError:
        return {}


def load_checkpoint_restore_manifest(
    checkpoint_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Read canonical metadata required to restore one checkpoint."""

    checkpoint = _read_checkpoint_manifest(checkpoint_path)
    missing_fields = sorted(_REQUIRED_RESTORE_FIELDS.difference(checkpoint))
    if missing_fields:
        raise CheckpointMetadataError(
            "Checkpoint restore metadata is incomplete; missing field(s): "
            f"{', '.join(missing_fields)}."
        )

    for field_name in ("name", "kind", "framework_uuid"):
        value = checkpoint[field_name]
        if not isinstance(value, str) or not value.strip():
            raise CheckpointMetadataError(
                f"Checkpoint restore metadata field '{field_name}' is invalid."
            )

    scenario_index = checkpoint["scenario_index"]
    if type(scenario_index) is not int or scenario_index < 0:
        raise CheckpointMetadataError(
            "Checkpoint restore metadata field 'scenario_index' is invalid."
        )

    step_count = checkpoint["step_count"]
    if type(step_count) is not int or step_count < 0:
        raise CheckpointMetadataError("Checkpoint restore metadata field 'step_count' is invalid.")
    return checkpoint


def update_last_checkpoint_metadata(
    model_package_directory: str,
    checkpoint_name: str,
    checkpoint_kind: str,
    logger: _WarningLogger | None = None,
    *,
    raise_on_error: bool = False,
) -> None:
    """Update session checkpoint metadata under a caller-owned storage mutation."""

    session_manifest_path = find_existing_session_manifest_path(model_package_directory)
    if not session_manifest_path or not os.path.isfile(session_manifest_path):
        return

    wrapper = SessionConfigurationWrapper(session_manifest_path)
    if not wrapper.exists():
        return
    try:
        wrapper.load_configuration()
        wrapper.apply_fields(
            {
                "last_checkpoint_name": checkpoint_name,
                "last_checkpoint_kind": checkpoint_kind,
                "last_checkpoint_saved_at": metadata_timestamps.utc_now_iso(),
            }
        )
        wrapper.save()
    except Exception as exc:
        if logger is not None:
            logger.warning(f"Failed to update session checkpoint metadata: {exc}")
        if raise_on_error:
            raise
