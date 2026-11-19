# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Strict, framework-neutral training-checkpoint source resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from yaml import YAMLError

from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    PersistenceError,
    ResourceNotFoundError,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.session_manifest_schema import (
    validate_session_manifest_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    InvalidStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file


@dataclass(frozen=True, slots=True)
class TrainingCheckpointSource:
    """Validated session and optional artifact selected for training restore."""

    session_name: str | None
    checkpoint_name: str | None

    def as_pair(self) -> tuple[str | None, str | None]:
        """Project the typed selection onto the established runtime contract."""

        return self.session_name, self.checkpoint_name


class TrainingCheckpointSourceResolver:
    """Resolve one restore source through canonical metadata and guarded paths.

    The selected session remains the save target. A forked session may instead
    name a parent checkpoint in its canonical schema-v1 manifest. Explicit
    runtime selection takes precedence over the session's last checkpoint,
    which in turn takes precedence over its fork source.
    """

    def resolve(self, configuration: object) -> TrainingCheckpointSource:
        """Return the validated restore source for one runtime configuration."""

        target_session, selected_checkpoint = self._configured_selection(
            configuration
        )
        if target_session is None:
            return TrainingCheckpointSource(None, None)

        session_section = self._read_session_manifest(
            configuration,
            target_session,
        )
        if selected_checkpoint is not None:
            return self._validated_source(
                configuration,
                target_session,
                selected_checkpoint,
            )

        last_checkpoint_text = str(
            session_section.get("last_checkpoint_name") or ""
        ).strip()
        if last_checkpoint_text:
            try:
                last_checkpoint = CheckpointStorageName.parse(
                    last_checkpoint_text
                )
            except InvalidStorageName as error:
                raise PersistenceError(
                    "The session manifest contains an invalid last checkpoint "
                    "name. Run the metadata migration or repair the manifest."
                ) from error
            return self._validated_source(
                configuration,
                target_session,
                last_checkpoint,
            )

        source_session_text = str(
            session_section.get("load_checkpoint_from_session") or ""
        ).strip()
        source_checkpoint_text = str(
            session_section.get("load_checkpoint_name") or ""
        ).strip()
        if bool(source_session_text) != bool(source_checkpoint_text):
            raise PersistenceError(
                "The session manifest contains an incomplete fork checkpoint "
                "selection."
            )
        if source_session_text:
            try:
                source_session = SessionStorageName.parse(source_session_text)
                source_checkpoint = CheckpointStorageName.parse(
                    source_checkpoint_text
                )
            except InvalidStorageName as error:
                raise PersistenceError(
                    "The session manifest contains an invalid fork checkpoint "
                    "selection."
                ) from error
            return self._validated_source(
                configuration,
                source_session,
                source_checkpoint,
            )
        return TrainingCheckpointSource(str(target_session), None)

    @classmethod
    def _configured_selection(
        cls,
        configuration: object,
    ) -> tuple[SessionStorageName | None, CheckpointStorageName | None]:
        """Return the configured selection value owned by this training checkpoint source resolver.
        """
        target_session_text = cls._configuration_text(
            configuration,
            "selected_default_model_name",
        )
        if not target_session_text:
            return None, None
        try:
            target_session = SessionStorageName.parse(target_session_text)
            selected_checkpoint_text = cls._configuration_text(
                configuration,
                "selected_default_checkpoint_name",
            )
            selected_checkpoint = (
                CheckpointStorageName.parse(selected_checkpoint_text)
                if selected_checkpoint_text
                else None
            )
        except InvalidStorageName as error:
            raise ApplicationValidationError(
                "The configured training checkpoint selection contains an "
                "invalid storage name."
            ) from error
        return target_session, selected_checkpoint

    @staticmethod
    def _configuration_text(
        configuration: object,
        *attribute_names: str,
    ) -> str:
        """Return the configuration text value owned by this training checkpoint source resolver."""
        for attribute_name in attribute_names:
            value = str(getattr(configuration, attribute_name, "") or "").strip()
            if value:
                return value
        return ""

    @classmethod
    def _validated_source(
        cls,
        configuration: object,
        source_session: SessionStorageName,
        source_checkpoint: CheckpointStorageName,
    ) -> TrainingCheckpointSource:
        """Return the validated source value owned by this training checkpoint source resolver."""
        source_session_path = cls._resolve_session_directory(
            configuration,
            source_session,
        )
        cls._read_session_manifest(configuration, source_session)
        try:
            session_guard = StoragePathGuard(source_session_path)
            checkpoint_root = session_guard.resolve(
                "checkpoints",
                must_exist=True,
            )
            if not checkpoint_root.is_dir():
                raise FileNotFoundError(checkpoint_root)
            candidates = cls._artifact_candidates(
                session_guard,
                StoragePathGuard(checkpoint_root),
                source_checkpoint,
            )
        except FileNotFoundError as error:
            raise ResourceNotFoundError(
                "The selected training checkpoint storage is missing."
            ) from error
        except (InvalidStorageName, StoragePathViolation) as error:
            raise PersistenceError(
                "The selected training checkpoint storage escapes its managed "
                "session root."
            ) from error

        cls._require_artifact(candidates)
        return TrainingCheckpointSource(
            str(source_session),
            str(source_checkpoint),
        )

    @staticmethod
    def _artifact_candidates(
        session_guard: StoragePathGuard,
        checkpoint_guard: StoragePathGuard,
        checkpoint: CheckpointStorageName,
    ) -> tuple[Path, ...]:
        """Return the artifact candidates value owned by this training checkpoint source resolver.
        """
        checkpoint_name = checkpoint.value
        checkpoint_stem = CheckpointStorageName.parse(
            os.path.splitext(checkpoint_name)[0]
        )
        expected_zip = (
            checkpoint
            if checkpoint_name.lower().endswith(".zip")
            else CheckpointStorageName.parse(f"{checkpoint_name}.zip")
        )
        expected_zip_stem = CheckpointStorageName.parse(
            os.path.splitext(expected_zip.value)[0]
        )
        return (
            checkpoint_guard.resolve(checkpoint),
            checkpoint_guard.resolve(checkpoint, expected_zip),
            checkpoint_guard.resolve(
                "stablebaselines",
                checkpoint_stem,
                checkpoint,
            ),
            checkpoint_guard.resolve(
                "stablebaselines",
                expected_zip_stem,
                expected_zip,
            ),
            session_guard.resolve("final_model", checkpoint),
        )

    @staticmethod
    def _require_artifact(candidates: tuple[Path, ...]) -> None:
        """Return the required artifact for this training checkpoint source resolver."""
        for candidate in candidates:
            if candidate.is_file():
                return
            if not candidate.is_dir():
                continue
            try:
                next(candidate.iterdir())
            except StopIteration:
                continue
            except OSError as error:
                raise PersistenceError(
                    "The selected training checkpoint artifact is unreadable."
                ) from error
            return
        raise ResourceNotFoundError(
            "The selected training checkpoint artifact is missing."
        )

    @staticmethod
    def _resolve_session_directory(
        configuration: object,
        session_name: SessionStorageName,
    ) -> Path:
        """Resolve the session directory required by this training checkpoint source resolver."""
        model_folder_path = str(
            getattr(configuration, "model_folder_path", "") or ""
        ).strip()
        if not model_folder_path:
            raise ApplicationValidationError(
                "The active model has no managed storage root."
            )
        try:
            training_sessions_root = StoragePathGuard(
                model_folder_path
            ).resolve(
                TRAINING_SESSIONS_DIR_NAME,
                must_exist=True,
            )
            session_path = StoragePathGuard(training_sessions_root).resolve(
                session_name,
                must_exist=True,
            )
        except FileNotFoundError as error:
            raise ResourceNotFoundError(
                "The selected training session storage is missing."
            ) from error
        except StoragePathViolation as error:
            raise ApplicationValidationError(
                "The selected training session is outside managed model storage."
            ) from error
        if not session_path.is_dir():
            raise ResourceNotFoundError(
                "The selected training session storage is not a directory."
            )
        return session_path

    @classmethod
    def _read_session_manifest(
        cls,
        configuration: object,
        target_session: SessionStorageName,
    ) -> dict[str, object]:
        """Read the session manifest without changing this training checkpoint source resolver's
        state.
        """
        session_path = cls._resolve_session_directory(
            configuration,
            target_session,
        )
        try:
            manifest_path = StoragePathGuard(session_path).resolve(
                SESSION_MANIFEST_FILE_NAME,
                must_exist=True,
            )
        except FileNotFoundError as error:
            raise ResourceNotFoundError(
                "The selected training session manifest is missing."
            ) from error
        except StoragePathViolation as error:
            raise ApplicationValidationError(
                "The selected training session is outside managed model storage."
            ) from error

        try:
            versioned_payload = validate_versioned_metadata_payload(
                MetadataKind.SESSION_MANIFEST,
                read_yaml_mapping_file(str(manifest_path)),
            )
            manifest = validate_session_manifest_payload(versioned_payload)
            session_section = manifest["session"]
            declared_storage_name = SessionStorageName.parse(
                session_section.get("storage_name") or ""
            )
        except FileNotFoundError as error:
            raise ResourceNotFoundError(
                "The selected training session manifest is missing."
            ) from error
        except (InvalidStorageName, OSError, TypeError, ValueError, YAMLError) as error:
            raise PersistenceError(
                "The selected training session manifest is unreadable or uses "
                "an unsupported schema. Run the metadata migration before "
                "training."
            ) from error

        if declared_storage_name != target_session:
            raise PersistenceError(
                "The selected training session manifest has a mismatched "
                "storage identity."
            )
        return dict(session_section)
