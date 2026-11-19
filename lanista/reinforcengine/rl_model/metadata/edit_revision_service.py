# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Durable revision capture and compare-and-swap checks for metadata edits."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from pathlib import Path

from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    OperationConflict,
    PersistenceError,
    ResourceNotFoundError,
    RuntimeUnavailableError,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_contracts import (
    CheckpointCatalogQueryPort,
)
from lanista.reinforcengine.rl_model.checkpoints.metadata import checkpoint_manifest_path
from lanista.reinforcengine.rl_model.metadata.edit_contracts import (
    ABSENT_METADATA_DIGEST,
    MetadataDegradedStateReporter,
    MetadataEditContext,
    MetadataEditRequest,
    MetadataEditSnapshot,
    _CheckpointRevision,
    _MetadataReplacement,
    _ModelRevision,
    _SessionRevision,
)
from lanista.reinforcengine.rl_model.metadata.revision_store import (
    checkpoint_artifact_fingerprint,
    read_existing_payload,
    require_regular_file,
)
from lanista.reinforcengine.rl_model.metadata.schema import MetadataKind
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_MANIFEST_FILE_NAME,
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction


def _require_mapping_section(
    payload: Mapping[str, object],
    section_name: str,
) -> dict[str, object]:
    """Narrow a schema-validated section for strict revision comparisons."""

    section = payload.get(section_name)
    if not isinstance(section, Mapping):
        raise PersistenceError(
            f"Stored metadata has no valid {section_name!r} section."
        )
    return dict(section)


class MetadataEditRevisionService:
    """Capture exact durable identities and validate prepared edit revisions."""

    def __init__(
        self,
        *,
        models_root_provider: Callable[[], str],
        checkpoint_catalog_provider: Callable[[], CheckpointCatalogQueryPort],
        degraded_state_reporter: MetadataDegradedStateReporter | None = None,
    ) -> None:
        """Initialize the metadata edit revision service and its owned state."""
        self._models_root_provider = models_root_provider
        self._checkpoint_catalog_provider = checkpoint_catalog_provider
        self._degraded_state_reporter = degraded_state_reporter

    def capture_storage_scope(self) -> ModelsStorageMutationScope:
        """Resolve one canonical models root for an entire edit operation."""

        models_root = str(self._models_root_provider() or "").strip()
        if not models_root:
            raise RuntimeUnavailableError("The model storage is unavailable.")
        return ModelsStorageMutationScope.from_models_root(models_root)

    def ensure_recovered(self, model_root: Path) -> None:
        """Require a clean transaction journal before reading or preparing."""

        try:
            recovery = StorageTransaction.recover(model_root)
        except Exception as error:
            self._mark_storage_recovery_required()
            raise PersistenceError(
                "Storage recovery could not be completed. Reload the model "
                "before editing metadata."
            ) from error
        if recovery.failed:
            self._mark_storage_recovery_required()
            raise PersistenceError(
                "Storage recovery failed. Reload the model before editing metadata."
            )

    def _mark_storage_recovery_required(self) -> None:
        """Latch the safety gate before exposing a recovery failure."""

        if self._degraded_state_reporter is None:
            return
        self._degraded_state_reporter(
            reload_required=True,
            storage_recovery_required=True,
        )

    def capture_snapshot(
        self,
        configuration: object,
        model_root: Path,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> MetadataEditSnapshot:
        """Capture only the durable compare-and-swap revision values."""

        return self.capture_context(
            configuration,
            model_root,
            session_name=session_name,
            checkpoint_name=checkpoint_name,
        ).snapshot

    def capture_context(
        self,
        configuration: object,
        model_root: Path,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> MetadataEditContext:
        """Read editable values and their revisions under the caller's lock."""

        self.ensure_recovered(model_root)
        if checkpoint_name and not session_name:
            raise ApplicationValidationError(
                "A checkpoint edit requires its training session."
            )

        model_revision = self._read_model_revision(configuration, model_root)
        session_revision = (
            self._read_session_revision(model_root, session_name)
            if session_name
            else None
        )
        checkpoint_revision = (
            self._read_checkpoint_revision(
                model_root,
                session_name,
                checkpoint_name,
            )
            if checkpoint_name
            else None
        )
        return self._build_edit_context(
            model_revision=model_revision,
            session_name=session_name,
            session_revision=session_revision,
            checkpoint_name=checkpoint_name,
            checkpoint_revision=checkpoint_revision,
        )

    def resolve_model_root(
        self,
        configuration: object,
        *,
        storage_scope: ModelsStorageMutationScope,
    ) -> Path:
        """Resolve the active model below one operation-bound storage root."""

        models_root = storage_scope.models_root
        model_folder = str(
            getattr(configuration, "model_folder_path", "") or ""
        ).strip()
        if not model_folder:
            raise RuntimeUnavailableError("The active model storage is unavailable.")

        guarded_root = StoragePathGuard(models_root).require_descendant(
            model_folder,
            must_exist=True,
        )
        if (
            Path(model_folder).absolute() != guarded_root
            or guarded_root.is_symlink()
        ):
            raise StoragePathViolation("Model storage aliases are forbidden.")
        if not guarded_root.is_dir():
            raise ResourceNotFoundError("The active model storage no longer exists.")
        return guarded_root

    def resolve_checkpoint_manifest_target(
        self,
        model_root: Path,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> tuple[object, Path, Path]:
        """Resolve one checkpoint artifact and sidecar within its session."""

        storage_name = SessionStorageName.parse(session_name or "")
        CheckpointStorageName.parse(checkpoint_name or "")
        detail = self._checkpoint_catalog_provider().build_selection_detail(
            session_name,
            checkpoint_name,
        )
        if not detail.exists or not detail.path:
            raise ResourceNotFoundError(
                "The selected checkpoint no longer exists. Reopen the edit dialog."
            )

        checkpoint_path = Path(detail.path).expanduser().absolute()
        model_guard = StoragePathGuard(model_root)
        guarded_checkpoint = model_guard.require_descendant(
            checkpoint_path,
            must_exist=True,
        )
        if checkpoint_path != guarded_checkpoint or checkpoint_path.is_symlink():
            raise StoragePathViolation("Checkpoint aliases are forbidden.")
        self._assert_checkpoint_session(
            model_root,
            str(storage_name),
            guarded_checkpoint,
        )

        target = Path(checkpoint_manifest_path(str(guarded_checkpoint)))
        guarded_target = model_guard.require_descendant(target)
        if target.absolute() != guarded_target or target.is_symlink():
            raise StoragePathViolation("Checkpoint metadata aliases are forbidden.")
        return detail, guarded_checkpoint, guarded_target

    @staticmethod
    def assert_snapshot_matches(
        snapshot: MetadataEditSnapshot,
        replacements: tuple[_MetadataReplacement, ...],
        request: MetadataEditRequest,
    ) -> None:
        """Validate every selected durable revision before staging writes."""

        MetadataEditRevisionService._assert_selected_targets(snapshot, request)
        replacements_by_kind = {
            replacement.kind: replacement for replacement in replacements
        }
        MetadataEditRevisionService._assert_model_revision(
            snapshot,
            replacements_by_kind[MetadataKind.MODEL_MANIFEST],
        )
        MetadataEditRevisionService._assert_session_revision(
            snapshot,
            replacements_by_kind.get(MetadataKind.SESSION_MANIFEST),
        )
        MetadataEditRevisionService._assert_checkpoint_revision(
            snapshot,
            replacements_by_kind.get(MetadataKind.CHECKPOINT_MANIFEST),
        )

    @staticmethod
    def _build_edit_context(
        *,
        model_revision: _ModelRevision,
        session_name: str | None,
        session_revision: _SessionRevision | None,
        checkpoint_name: str | None,
        checkpoint_revision: _CheckpointRevision | None,
    ) -> MetadataEditContext:
        """Build the edit context required by this metadata edit revision service."""
        model_section = model_revision.section
        snapshot = MetadataEditSnapshot(
            model_uuid=model_revision.identity_uuid,
            model_digest=model_revision.digest,
            model_name=str(model_section.get("name") or ""),
            model_version=str(model_section.get("version") or ""),
            session_name=session_name,
            session_identity_uuid=(
                session_revision.identity_uuid
                if session_revision is not None
                else None
            ),
            session_digest=(
                session_revision.digest
                if session_revision is not None
                else None
            ),
            checkpoint_name=checkpoint_name,
            checkpoint_digest=(
                checkpoint_revision.digest
                if checkpoint_revision is not None
                else None
            ),
            checkpoint_artifact_fingerprint=(
                checkpoint_revision.artifact_fingerprint
                if checkpoint_revision is not None
                else None
            ),
        )
        return MetadataEditContext(
            snapshot=snapshot,
            model_name=str(model_section.get("name") or ""),
            model_version=str(model_section.get("version") or ""),
            model_description=str(model_section.get("description") or ""),
            session_name=session_name,
            session_display_name=(
                session_revision.display_name
                if session_revision is not None
                else ""
            ),
            session_description=(
                session_revision.description
                if session_revision is not None
                else ""
            ),
            checkpoint_name=checkpoint_name,
            checkpoint_annotation=(
                checkpoint_revision.annotation
                if checkpoint_revision is not None
                else ""
            ),
        )

    @staticmethod
    def _read_model_revision(
        configuration: object,
        model_root: Path,
    ) -> _ModelRevision:
        """Read the model revision without changing this metadata edit revision service's state."""
        model_target = require_regular_file(
            StoragePathGuard(model_root),
            model_root / MODEL_MANIFEST_FILE_NAME,
        )
        model_payload, model_digest = read_existing_payload(
            model_target,
            MetadataKind.MODEL_MANIFEST,
        )
        model_section = _require_mapping_section(model_payload, "model")
        model_uuid = str(model_section.get("uuid") or "").strip()
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if not model_uuid:
            raise PersistenceError(
                "Stored model metadata has no stable identity. Run the metadata "
                "migration before editing it."
            )
        if model_uuid != active_uuid:
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog before saving."
            )
        return _ModelRevision(
            section=model_section,
            identity_uuid=model_uuid,
            digest=model_digest,
        )

    def _read_session_revision(
        self,
        model_root: Path,
        session_name: str,
    ) -> _SessionRevision:
        """Read the session revision without changing this metadata edit revision service's state.
        """
        session_target = self._resolve_session_manifest_target(
            model_root,
            session_name,
        )
        session_payload, session_digest = read_existing_payload(
            session_target,
            MetadataKind.SESSION_MANIFEST,
        )
        session_section = _require_mapping_section(session_payload, "session")
        if str(session_section.get("storage_name") or "") != session_name:
            raise OperationConflict(
                "The selected session identity changed. Reopen the edit dialog."
            )
        session_identity_uuid = str(session_section.get("uuid") or "").strip()
        if not session_identity_uuid:
            raise PersistenceError(
                "Stored session metadata has no stable identity. Run the metadata "
                "migration before editing it."
            )
        return _SessionRevision(
            identity_uuid=session_identity_uuid,
            digest=session_digest,
            display_name=str(session_section.get("name") or ""),
            description=str(session_section.get("description") or ""),
        )

    def _read_checkpoint_revision(
        self,
        model_root: Path,
        session_name: str | None,
        checkpoint_name: str,
    ) -> _CheckpointRevision:
        """Read the checkpoint revision without changing this metadata edit revision service's
        state.
        """
        detail, checkpoint_artifact, checkpoint_target = (
            self.resolve_checkpoint_manifest_target(
                model_root,
                session_name,
                checkpoint_name,
            )
        )
        artifact_fingerprint = checkpoint_artifact_fingerprint(
            checkpoint_artifact
        )
        if checkpoint_target.is_file():
            checkpoint_payload, checkpoint_digest = read_existing_payload(
                checkpoint_target,
                MetadataKind.CHECKPOINT_MANIFEST,
            )
            checkpoint_section = _require_mapping_section(
                checkpoint_payload,
                "checkpoint",
            )
            if str(checkpoint_section.get("name") or "") != checkpoint_name:
                raise OperationConflict(
                    "The selected checkpoint identity changed. Reopen the edit dialog."
                )
            annotation = str(checkpoint_section.get("annotation") or "")
        elif os.path.lexists(checkpoint_target):
            raise StoragePathViolation(
                "Checkpoint metadata is not a regular file."
            )
        else:
            checkpoint_digest = ABSENT_METADATA_DIGEST
            annotation = str(getattr(detail, "annotation", None) or "")
        return _CheckpointRevision(
            digest=checkpoint_digest,
            annotation=annotation,
            artifact_fingerprint=artifact_fingerprint,
        )

    @staticmethod
    def _resolve_session_manifest_target(
        model_root: Path,
        session_name: str,
    ) -> Path:
        """Resolve the session manifest target required by this metadata edit revision service."""
        storage_name = SessionStorageName.parse(session_name)
        sessions_root = model_root / TRAINING_SESSIONS_DIR_NAME
        session_root = sessions_root / str(storage_name)
        if sessions_root.is_symlink() or session_root.is_symlink():
            raise StoragePathViolation("Session metadata aliases are forbidden.")
        return require_regular_file(
            StoragePathGuard(model_root),
            session_root / SESSION_MANIFEST_FILE_NAME,
        )

    @staticmethod
    def _assert_checkpoint_session(
        model_root: Path,
        storage_name: str,
        guarded_checkpoint: Path,
    ) -> None:
        """Assert the checkpoint session owned by this metadata edit
revision service."""
        session_root = (
            model_root / TRAINING_SESSIONS_DIR_NAME / storage_name
        )
        if session_root.is_symlink() or not session_root.is_dir():
            raise StoragePathViolation("Session storage aliases are forbidden.")
        guarded_for_session = StoragePathGuard(session_root).require_descendant(
            guarded_checkpoint,
            must_exist=True,
        )
        if guarded_for_session != guarded_checkpoint:
            raise StoragePathViolation(
                "The checkpoint does not belong to the selected session."
            )

    @staticmethod
    def _assert_selected_targets(
        snapshot: MetadataEditSnapshot,
        request: MetadataEditRequest,
    ) -> None:
        """Assert the selected targets owned by this metadata edit revision
service."""
        if snapshot.model_uuid != str(request.expected_model_uuid or "").strip():
            raise OperationConflict(
                "The model edit snapshot is stale. Reopen the edit dialog."
            )
        if snapshot.session_name != request.session_name:
            raise OperationConflict(
                "The selected session changed. Reopen the edit dialog."
            )
        if snapshot.checkpoint_name != request.checkpoint_name:
            raise OperationConflict(
                "The selected checkpoint changed. Reopen the edit dialog."
            )

    @staticmethod
    def _assert_model_revision(
        snapshot: MetadataEditSnapshot,
        replacement: _MetadataReplacement,
    ) -> None:
        """Assert the model revision owned by this metadata edit revision
service."""
        if snapshot.model_digest != replacement.source_digest:
            raise OperationConflict(
                "Model metadata changed. Reopen the edit dialog before saving."
            )

    @staticmethod
    def _assert_session_revision(
        snapshot: MetadataEditSnapshot,
        replacement: _MetadataReplacement | None,
    ) -> None:
        """Assert the session revision owned by this metadata edit revision
service."""
        if replacement is not None:
            session_uuid = str(
                _require_mapping_section(replacement.payload, "session").get(
                    "uuid"
                )
                or ""
            )
            if (
                snapshot.session_digest != replacement.source_digest
                or snapshot.session_identity_uuid != session_uuid
            ):
                raise OperationConflict(
                    "Session metadata changed. Reopen the edit dialog before saving."
                )
        elif snapshot.session_digest is not None:
            raise OperationConflict(
                "The selected session changed. Reopen the edit dialog."
            )

    @staticmethod
    def _assert_checkpoint_revision(
        snapshot: MetadataEditSnapshot,
        replacement: _MetadataReplacement | None,
    ) -> None:
        """Assert the checkpoint revision owned by this metadata edit
revision service."""
        if replacement is not None:
            if (
                snapshot.checkpoint_digest != replacement.source_digest
                or snapshot.checkpoint_artifact_fingerprint
                != replacement.artifact_fingerprint
            ):
                raise OperationConflict(
                    "Checkpoint metadata changed. Reopen the edit dialog before saving."
                )
        elif snapshot.checkpoint_digest is not None:
            raise OperationConflict(
                "The selected checkpoint changed. Reopen the edit dialog."
            )


__all__ = ["MetadataEditRevisionService"]
