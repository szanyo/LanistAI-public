# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Preflight and canonical payload preparation for transactional metadata edits."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from pathlib import Path
from typing import Protocol, cast

from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    OperationConflict,
    PersistenceError,
)
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.metadata.edit_contracts import (
    ABSENT_METADATA_DIGEST,
    MetadataEditOutcome,
    MetadataEditRequest,
    _MetadataReplacement,
    _PreparedMetadataEdit,
)
from lanista.reinforcengine.rl_model.metadata.edit_revision_service import (
    MetadataEditRevisionService,
)
from lanista.reinforcengine.rl_model.metadata.revision_store import (
    checkpoint_artifact_fingerprint,
    read_existing_payload,
    require_regular_file,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_MANIFEST_FILE_NAME,
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import SessionStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


class _CheckpointDetail(Protocol):
    """Describe the checkpoint catalog fields used for manifest creation."""

    checkpoint_kind: str
    framework_uuid: str
    scenario_index: int | None
    step_count: int | None
    parent_checkpoint_name: str | None
    saved_at_raw: str | None


class MetadataEditPreparer:
    """Build validated replacements without changing durable state."""

    _MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*_project$")

    def __init__(
        self,
        *,
        models_root_provider: Callable[[], str],
        revision_service: MetadataEditRevisionService,
    ) -> None:
        """Initialize the metadata edit preparer and its owned state."""
        self._models_root_provider = models_root_provider
        self._revisions = revision_service

    def prepare(
        self,
        configuration: object,
        model_root: Path,
        request: MetadataEditRequest,
    ) -> _PreparedMetadataEdit:
        """Prepare every selected payload and prove its captured revision."""

        if request.snapshot is None:
            raise ApplicationValidationError(
                "The edit snapshot is missing. Reopen the edit dialog."
            )
        self._revisions.ensure_recovered(model_root)

        model_replacement, outcome = self._prepare_model_replacement(
            configuration,
            model_root,
            request,
        )
        replacements = [model_replacement]
        if request.session_name:
            replacements.append(
                self._prepare_session_replacement(model_root, request)
            )
        elif request.checkpoint_name:
            raise ApplicationValidationError(
                "A checkpoint edit requires its training session."
            )
        if request.session_name and request.checkpoint_name:
            replacements.append(
                self._prepare_checkpoint_replacement(model_root, request)
            )

        prepared_replacements = tuple(replacements)
        self._revisions.assert_snapshot_matches(
            request.snapshot,
            prepared_replacements,
            request,
        )
        return _PreparedMetadataEdit(
            configuration=configuration,
            model_root=model_root,
            outcome=outcome,
            replacements=prepared_replacements,
        )

    def _prepare_model_replacement(
        self,
        configuration: object,
        model_root: Path,
        request: MetadataEditRequest,
    ) -> tuple[_MetadataReplacement, MetadataEditOutcome]:
        """Prepare the model replacement required by this metadata edit preparer."""
        target = require_regular_file(
            StoragePathGuard(model_root),
            model_root / MODEL_MANIFEST_FILE_NAME,
        )
        original, source_digest = read_existing_payload(
            target,
            MetadataKind.MODEL_MANIFEST,
        )
        model_section = dict(cast(Mapping[str, object], original["model"]))
        self._guard_model_identity(configuration, model_section, request)
        effective_name, effective_version = self._resolve_model_identity(
            model_section,
            model_root,
            request,
        )
        description = str(request.model_description or "").strip()

        candidate = deepcopy(original)
        candidate_model = dict(cast(Mapping[str, object], candidate["model"]))
        candidate_model.update(
            name=effective_name,
            version=effective_version,
            description=description,
            modified_at=utc_now_iso(),
        )
        candidate["model"] = candidate_model
        canonical = validate_versioned_metadata_payload(
            MetadataKind.MODEL_MANIFEST,
            candidate,
        )
        outcome = MetadataEditOutcome(
            model_name=effective_name,
            model_version=effective_version,
            model_description=description,
            session_name=request.session_name,
            checkpoint_name=request.checkpoint_name,
        )
        return (
            _MetadataReplacement(
                target=target,
                kind=MetadataKind.MODEL_MANIFEST,
                payload=canonical,
                source_digest=source_digest,
            ),
            outcome,
        )

    def _resolve_model_identity(
        self,
        model_section: Mapping[str, object],
        model_root: Path,
        request: MetadataEditRequest,
    ) -> tuple[str, str]:
        """Resolve the model identity required by this metadata edit preparer."""
        current_version = str(model_section.get("version") or "")
        raw_version = str(request.model_version or "").strip()
        resolved_version = EmptyModelGenerator.resolve_model_version(
            raw_version,
            existing_version=current_version,
            models_root=str(self._models_root_provider() or ""),
        )
        if raw_version and not resolved_version:
            raise ApplicationValidationError(
                "The requested model version is invalid."
            )
        effective_version = resolved_version or current_version
        raw_name = str(request.model_name or "").strip()
        effective_name = EmptyModelGenerator.normalize_model_name(
            raw_name or str(model_section.get("name") or ""),
            effective_version,
        )
        self._validate_model_name(effective_name, model_root)
        return effective_name, effective_version

    @staticmethod
    def _guard_model_identity(
        configuration: object,
        model_section: Mapping[str, object],
        request: MetadataEditRequest,
    ) -> None:
        """Guard the model identity owned by this metadata edit preparer."""
        expected_uuid = str(request.expected_model_uuid or "").strip()
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        persisted_uuid = str(model_section.get("uuid") or "").strip()
        if (
            not expected_uuid
            or expected_uuid != active_uuid
            or expected_uuid != persisted_uuid
        ):
            raise OperationConflict(
                "The active model changed. Reopen the edit dialog before saving."
            )

        expected_name = str(request.expected_model_name or "").strip()
        expected_version = str(request.expected_model_version or "").strip()
        if (
            expected_name
            and expected_name != str(model_section.get("name") or "")
        ):
            raise OperationConflict(
                "Model metadata changed. Reopen the edit dialog before saving."
            )
        if (
            expected_version
            and expected_version != str(model_section.get("version") or "")
        ):
            raise OperationConflict(
                "Model metadata changed. Reopen the edit dialog before saving."
            )

    def _validate_model_name(self, model_name: str, model_root: Path) -> None:
        """Validate the model name required by this metadata edit preparer."""
        if not model_name or not self._MODEL_NAME_PATTERN.fullmatch(model_name):
            raise ApplicationValidationError(
                "The model name must resolve to a '<name>_project' value."
            )
        target_folder = Path(self._models_root_provider()) / (
            EmptyModelGenerator.get_model_folder_name(model_name)
        )
        if target_folder.exists() and target_folder.resolve() != model_root:
            raise OperationConflict("Another model already uses the requested name.")

    @staticmethod
    def _prepare_session_replacement(
        model_root: Path,
        request: MetadataEditRequest,
    ) -> _MetadataReplacement:
        """Prepare the session replacement required by this metadata edit preparer."""
        storage_name = SessionStorageName.parse(request.session_name or "")
        guard = StoragePathGuard(model_root)
        target = require_regular_file(
            guard,
            model_root
            / TRAINING_SESSIONS_DIR_NAME
            / str(storage_name)
            / SESSION_MANIFEST_FILE_NAME,
        )
        sessions_root = model_root / TRAINING_SESSIONS_DIR_NAME
        session_root = sessions_root / str(storage_name)
        if sessions_root.is_symlink() or session_root.is_symlink():
            raise StoragePathViolation("Session metadata aliases are forbidden.")

        original, source_digest = read_existing_payload(
            target,
            MetadataKind.SESSION_MANIFEST,
        )
        candidate = deepcopy(original)
        session_section = dict(cast(Mapping[str, object], candidate["session"]))
        if str(session_section.get("storage_name") or "") != str(storage_name):
            raise OperationConflict(
                "The selected session identity changed. Reopen the edit dialog."
            )
        if not str(session_section.get("uuid") or "").strip():
            raise PersistenceError(
                "Stored session metadata has no stable identity. Run the metadata "
                "migration before editing it."
            )
        session_section["name"] = str(request.session_display_name or "").strip()
        session_section["description"] = str(
            request.session_description or ""
        ).strip()
        session_section["modified_at"] = utc_now_iso()
        candidate["session"] = session_section
        return _MetadataReplacement(
            target=target,
            kind=MetadataKind.SESSION_MANIFEST,
            payload=validate_versioned_metadata_payload(
                MetadataKind.SESSION_MANIFEST,
                candidate,
            ),
            source_digest=source_digest,
        )

    def _prepare_checkpoint_replacement(
        self,
        model_root: Path,
        request: MetadataEditRequest,
    ) -> _MetadataReplacement:
        """Prepare the checkpoint replacement required by this metadata edit preparer."""
        detail, checkpoint_artifact, guarded_target = (
            self._revisions.resolve_checkpoint_manifest_target(
                model_root,
                request.session_name,
                request.checkpoint_name,
            )
        )
        checkpoint_detail = cast(_CheckpointDetail, detail)
        artifact_fingerprint = checkpoint_artifact_fingerprint(
            checkpoint_artifact
        )
        candidate, checkpoint_section, source_digest = (
            self._checkpoint_candidate(
                checkpoint_detail,
                guarded_target,
                request,
            )
        )
        checkpoint_section["annotation"] = (
            str(request.checkpoint_annotation or "").strip() or None
        )
        candidate["checkpoint"] = checkpoint_section
        return _MetadataReplacement(
            target=guarded_target,
            kind=MetadataKind.CHECKPOINT_MANIFEST,
            payload=validate_versioned_metadata_payload(
                MetadataKind.CHECKPOINT_MANIFEST,
                candidate,
            ),
            source_digest=source_digest,
            related_artifact=checkpoint_artifact,
            artifact_fingerprint=artifact_fingerprint,
        )

    @staticmethod
    def _checkpoint_candidate(
        detail: _CheckpointDetail,
        guarded_target: Path,
        request: MetadataEditRequest,
    ) -> tuple[dict[str, object], dict[str, object], str]:
        """Checkpoint the candidate owned by this metadata edit preparer."""
        if guarded_target.is_file():
            original, source_digest = read_existing_payload(
                guarded_target,
                MetadataKind.CHECKPOINT_MANIFEST,
            )
            candidate = deepcopy(original)
            checkpoint_section = dict(
                cast(Mapping[str, object], candidate["checkpoint"])
            )
            if str(checkpoint_section.get("name") or "") != str(
                request.checkpoint_name or ""
            ):
                raise OperationConflict(
                    "The selected checkpoint identity changed. Reopen the edit dialog."
                )
            return candidate, checkpoint_section, source_digest
        if os.path.lexists(guarded_target):
            raise StoragePathViolation(
                "Checkpoint metadata is not a regular file."
            )
        checkpoint_section = {
            "name": request.checkpoint_name,
            "kind": detail.checkpoint_kind,
            "framework_uuid": detail.framework_uuid,
            "scenario_index": detail.scenario_index,
            "curriculum_state": None,
            "step_count": detail.step_count,
            "parent_checkpoint_name": detail.parent_checkpoint_name,
            "saved_at": detail.saved_at_raw or utc_now_iso(),
        }
        return (
            {
                "schema_version": 1,
                "checkpoint": checkpoint_section,
            },
            checkpoint_section,
            ABSENT_METADATA_DIGEST,
        )


__all__ = ["MetadataEditPreparer"]
