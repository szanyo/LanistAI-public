# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed storage and application contracts for metadata editing."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_model.metadata.schema import MetadataKind
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction


class MetadataEditTransactionFactory(Protocol):
    """Build a transaction rooted at one model directory."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
    ) -> StorageTransaction:
        """Invoke this metadata edit transaction factory as a callable."""
        ...


class MetadataEditLockFactory(Protocol):
    """Build an exclusive lock rooted at the shared models directory."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        timeout_seconds: float,
    ) -> StorageMutationLock:
        """Invoke this metadata edit lock factory as a callable."""
        ...


class MetadataDegradedStateReporter(Protocol):
    """Record restrictions produced by a durable but degraded commit."""

    def __call__(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> None:
        """Invoke this metadata degraded state reporter as a callable."""
        ...


ABSENT_METADATA_DIGEST = "absent"


@dataclass(frozen=True, slots=True)
class MetadataEditSnapshot:
    """Exact durable identities observed when the edit dialog was opened."""

    model_uuid: str
    model_digest: str
    model_name: str = ""
    model_version: str = ""
    session_name: str | None = None
    session_identity_uuid: str | None = None
    session_digest: str | None = None
    checkpoint_name: str | None = None
    checkpoint_digest: str | None = None
    checkpoint_artifact_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataEditContext:
    """Canonical editable values and their exact durable revision snapshot."""

    snapshot: MetadataEditSnapshot
    model_name: str
    model_version: str
    model_description: str
    session_name: str | None
    session_display_name: str
    session_description: str
    checkpoint_name: str | None
    checkpoint_annotation: str


@dataclass(frozen=True, slots=True)
class MetadataEditRequest:
    """Immutable edit values and the model identity captured by the dialog."""

    expected_model_uuid: str
    expected_model_name: str
    expected_model_version: str
    snapshot: MetadataEditSnapshot | None
    model_name: str | None
    model_version: str | None
    model_description: str | None
    session_name: str | None = None
    session_display_name: str | None = None
    session_description: str | None = None
    checkpoint_name: str | None = None
    checkpoint_annotation: str | None = None


@dataclass(frozen=True, slots=True)
class MetadataEditOutcome:
    """Canonical values committed by one metadata edit."""

    model_name: str
    model_version: str
    model_description: str
    session_name: str | None
    checkpoint_name: str | None
    projection_synchronized: bool = True
    storage_cleanup_synchronized: bool = True


@dataclass(frozen=True, slots=True)
class _MetadataReplacement:
    """Carry the data required for metadata replacement."""

    target: Path
    kind: MetadataKind
    payload: dict[str, object]
    source_digest: str
    related_artifact: Path | None = None
    artifact_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedMetadataEdit:
    """Carry the data required for prepared metadata edit."""

    configuration: object
    model_root: Path
    outcome: MetadataEditOutcome
    replacements: tuple[_MetadataReplacement, ...]


@dataclass(frozen=True, slots=True)
class _ModelRevision:
    """Carry the data required for model revision."""

    section: dict[str, object]
    identity_uuid: str
    digest: str


@dataclass(frozen=True, slots=True)
class _SessionRevision:
    """Carry the data required for session revision."""

    identity_uuid: str
    digest: str
    display_name: str
    description: str


@dataclass(frozen=True, slots=True)
class _CheckpointRevision:
    """Carry the data required for checkpoint revision."""

    digest: str
    annotation: str
    artifact_fingerprint: str


__all__ = [
    "ABSENT_METADATA_DIGEST",
    "MetadataDegradedStateReporter",
    "MetadataEditContext",
    "MetadataEditLockFactory",
    "MetadataEditOutcome",
    "MetadataEditRequest",
    "MetadataEditSnapshot",
    "MetadataEditTransactionFactory",
]
