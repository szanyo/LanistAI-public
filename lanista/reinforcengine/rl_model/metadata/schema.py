# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Canonical versioned schemas for persistent model-storage metadata."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from enum import StrEnum
from typing import Any

from lanista.reinforcengine.rl_model.metadata.manifest_schema import (
    validate_manifest_payload,
)
from lanista.reinforcengine.rl_model.metadata.session_manifest_schema import (
    canonicalize_session_manifest_payload,
    validate_session_manifest_payload,
)

CURRENT_METADATA_SCHEMA_VERSION = 1


class MetadataSchemaError(ValueError):
    """Raised when metadata cannot be represented by the current schema."""


class MetadataKind(StrEnum):
    """Known metadata contracts under a model-storage root."""

    MODEL_MANIFEST = "model_manifest"
    MODEL_CONFIGURATION = "model_configuration"
    FRAMEWORK_CONFIGURATION = "framework_configuration"
    SESSION_MANIFEST = "session_manifest"
    CHECKPOINT_MANIFEST = "checkpoint_manifest"
    COMPARISON_PACK_MANIFEST = "comparison_pack_manifest"
    CLONE_MANIFEST = "clone_manifest"
    RUNTIME_STATE = "runtime_state"


def migrate_metadata_payload(kind: MetadataKind, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a legacy payload to its canonical schema-v1 representation."""
    migrated = deepcopy(dict(payload))
    _validate_existing_version(migrated)

    if kind is MetadataKind.CHECKPOINT_MANIFEST and "checkpoint" not in migrated:
        legacy_checkpoint = {
            key: value
            for key, value in migrated.items()
            if key != "schema_version"
        }
        migrated = {"schema_version": CURRENT_METADATA_SCHEMA_VERSION, "checkpoint": legacy_checkpoint}
    elif kind is MetadataKind.COMPARISON_PACK_MANIFEST and "comparison_pack" not in migrated:
        legacy_pack = {
            key: value
            for key, value in migrated.items()
            if key != "schema_version"
        }
        migrated = {"schema_version": CURRENT_METADATA_SCHEMA_VERSION, "comparison_pack": legacy_pack}
    else:
        migrated["schema_version"] = CURRENT_METADATA_SCHEMA_VERSION

    if kind is MetadataKind.SESSION_MANIFEST:
        return canonicalize_session_manifest_payload(migrated)
    validate_versioned_metadata_payload(kind, migrated)
    return migrated


def validate_versioned_metadata_payload(
    kind: MetadataKind,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a canonical v1 payload while preserving its original values."""
    candidate = deepcopy(dict(payload))
    version = candidate.get("schema_version")
    if type(version) is not int or version != CURRENT_METADATA_SCHEMA_VERSION:
        raise MetadataSchemaError(
            f"Unsupported {kind.value} schema version: {version!r}."
        )

    try:
        if kind is MetadataKind.MODEL_MANIFEST:
            validate_manifest_payload(candidate)
        elif kind is MetadataKind.SESSION_MANIFEST:
            validate_session_manifest_payload(candidate)
        elif kind is MetadataKind.CHECKPOINT_MANIFEST:
            _require_mapping_section(candidate, "checkpoint", kind)
        elif kind is MetadataKind.COMPARISON_PACK_MANIFEST:
            _require_mapping_section(candidate, "comparison_pack", kind)
        elif kind is MetadataKind.MODEL_CONFIGURATION:
            _require_mapping_section(candidate, "model_configuration", kind)
        elif kind is MetadataKind.RUNTIME_STATE:
            _validate_runtime_state(candidate)
    except MetadataSchemaError:
        raise
    except Exception as exc:
        raise MetadataSchemaError(f"Invalid {kind.value} metadata: {exc}") from exc
    return candidate


def _validate_existing_version(payload: Mapping[str, Any]) -> None:
    """Validate the existing version required by this metadata schema module."""
    if "schema_version" not in payload:
        return
    version = payload.get("schema_version")
    if type(version) is not int or version != CURRENT_METADATA_SCHEMA_VERSION:
        raise MetadataSchemaError(f"Unsupported metadata schema version: {version!r}.")


def _require_mapping_section(
    payload: Mapping[str, Any],
    section_name: str,
    kind: MetadataKind,
) -> None:
    """Return the required mapping section for this metadata schema module."""
    if not isinstance(payload.get(section_name), Mapping):
        raise MetadataSchemaError(
            f"{kind.value} metadata is missing a '{section_name}' mapping."
        )


def _validate_runtime_state(payload: Mapping[str, Any]) -> None:
    """Validate the runtime state required by this metadata schema module."""
    _require_mapping_section(payload, "runtime_state", MetadataKind.RUNTIME_STATE)
    runtime_state = payload["runtime_state"]
    assert isinstance(runtime_state, Mapping)
    known_fields = (
        "last_successful_model_name",
        "last_successful_framework_uuid",
        "selected_session_name",
        "selected_checkpoint_name",
    )
    for field_name in known_fields:
        value = runtime_state.get(field_name, "")
        if not isinstance(value, str):
            raise MetadataSchemaError(
                f"runtime_state field {field_name!r} must be a string."
            )
