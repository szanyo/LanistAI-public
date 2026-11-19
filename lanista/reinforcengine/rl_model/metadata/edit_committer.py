# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Transactional publish and readback boundary for prepared metadata edits."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml

from lanista.reinforcengine.application_errors import OperationConflict
from lanista.reinforcengine.rl_model.metadata.edit_contracts import (
    ABSENT_METADATA_DIGEST,
    MetadataEditTransactionFactory,
    _MetadataReplacement,
    _PreparedMetadataEdit,
)
from lanista.reinforcengine.rl_model.metadata.revision_store import (
    checkpoint_artifact_fingerprint,
    metadata_digest,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageTransaction,
    StorageTransactionError,
)


class MetadataEditCommitter:
    """Publish one prepared edit while preserving its compare-and-swap proof."""

    _OPERATION = "edit-model-session-checkpoint-metadata"

    def __init__(
        self,
        transaction_factory: MetadataEditTransactionFactory,
    ) -> None:
        """Initialize the metadata edit committer and its owned state."""
        self._transaction_factory = transaction_factory

    def commit(self, prepared: _PreparedMetadataEdit) -> Exception | None:
        """Commit prepared replacements or return a post-marker cleanup error."""

        transaction = self._transaction_factory(
            prepared.model_root,
            operation=self._OPERATION,
        )
        try:
            with transaction:
                staged = tuple(
                    self._stage_replacement(transaction, replacement)
                    for replacement in prepared.replacements
                )
                self._assert_sources_unchanged(
                    prepared.model_root,
                    prepared.replacements,
                )
                for replacement, staging in zip(
                    prepared.replacements,
                    staged,
                    strict=True,
                ):
                    if os.path.lexists(replacement.target):
                        transaction.quarantine(replacement.target)
                    transaction.publish(staging, replacement.target)
                    self._verify_replacement(replacement)
                transaction.commit()
        except Exception as exc:
            if transaction.committed:
                return exc
            if isinstance(exc, ValueError | yaml.YAMLError):
                raise StorageTransactionError(
                    "The metadata transaction could not validate its staged data."
                ) from exc
            raise
        return None

    @staticmethod
    def _assert_sources_unchanged(
        model_root: Path,
        replacements: tuple[_MetadataReplacement, ...],
    ) -> None:
        """Assert the sources unchanged owned by this metadata edit
committer."""
        guard = StoragePathGuard(model_root)
        for replacement in replacements:
            target = replacement.target
            try:
                guarded = guard.require_descendant(
                    target,
                    must_exist=(
                        replacement.source_digest != ABSENT_METADATA_DIGEST
                    ),
                )
                if target.absolute() != guarded or target.is_symlink():
                    raise StoragePathViolation("Metadata aliases are forbidden.")
                if replacement.source_digest == ABSENT_METADATA_DIGEST:
                    unchanged = not os.path.lexists(target)
                else:
                    unchanged = (
                        guarded.is_file()
                        and metadata_digest(guarded.read_bytes())
                        == replacement.source_digest
                    )
                if unchanged and replacement.related_artifact is not None:
                    unchanged = (
                        checkpoint_artifact_fingerprint(
                            replacement.related_artifact
                        )
                        == replacement.artifact_fingerprint
                    )
            except (OSError, StoragePathViolation):
                unchanged = False
            if not unchanged:
                raise OperationConflict(
                    "Metadata changed before the edit could be committed. "
                    "Reopen the edit dialog."
                )

    def _stage_replacement(
        self,
        transaction: StorageTransaction,
        replacement: _MetadataReplacement,
    ) -> Path:
        """Return the stage replacement value owned by this metadata edit committer."""
        staging = transaction.stage_file(replacement.target)
        atomic_write_validated_yaml_file(
            str(staging),
            replacement.payload,
            validator=lambda payload: self._validate_payload(
                replacement.kind,
                payload,
            ),
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )
        return staging

    def _verify_replacement(
        self,
        replacement: _MetadataReplacement,
    ) -> None:
        """Verify the replacement owned by this metadata edit committer."""
        readback = self._validate_payload(
            replacement.kind,
            read_yaml_mapping_file(str(replacement.target)),
        )
        if readback != replacement.payload:
            raise StorageTransactionError("Published metadata readback differs.")

    @staticmethod
    def _validate_payload(
        kind: MetadataKind,
        payload: Mapping[str, object],
    ) -> dict[str, object]:
        """Validate the payload required by this metadata edit committer."""
        return dict(validate_versioned_metadata_payload(kind, payload))


__all__ = ["MetadataEditCommitter"]
