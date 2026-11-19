# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stage typed metadata for one crash-recoverable session deletion."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from lanista.reinforcengine.rl_model.metadata.configuration_schema import ConfigurationSchema
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.session_manifest_schema import (
    validate_session_manifest_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_CONFIGURATION_FILE_NAME,
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import SessionStorageName
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


class SessionDeletionMetadataError(StorageTransactionError):
    """Raised when deletion metadata cannot be read, staged, or verified."""


@dataclass(frozen=True, slots=True)
class MetadataReplacement:
    """One validated metadata payload that will replace an existing file."""

    target: Path
    kind: MetadataKind
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class StagedMetadataReplacement:
    """A transaction-owned staged file and its expected canonical payload."""

    target: Path
    staging: Path
    kind: MetadataKind
    expected_payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class PersistedRuntimeSelection:
    """Selection values read from the canonical model configuration."""

    session_name: str | None
    checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class SessionMetadataPatch:
    """Deletion-specific patch with explicit nullable clear operations."""

    modified_at: str
    child_session_uuids: tuple[str, ...] | None = None
    clear_parent_session_uuid: bool = False
    clear_load_checkpoint_source: bool = False

    @property
    def has_changes(self) -> bool:
        """Return the has changes exposed by this session metadata patch."""
        return (
            self.child_session_uuids is not None
            or self.clear_parent_session_uuid
            or self.clear_load_checkpoint_source
        )

    def apply_to(self, session_section: dict[str, object]) -> None:
        """Apply the to transition owned by this session metadata patch."""
        if self.child_session_uuids is not None:
            session_section["child_session_uuids"] = list(
                self.child_session_uuids
            )
        if self.clear_parent_session_uuid:
            session_section["parent_session_uuid"] = None
        if self.clear_load_checkpoint_source:
            session_section["load_checkpoint_from_session"] = None
        session_section["modified_at"] = self.modified_at


class SessionDeletionMetadataRepository:
    """Read and stage only metadata participating in session deletion."""

    def __init__(self, model_root: str | Path) -> None:
        """Initialize the session deletion metadata repository and its owned state."""
        self._guard = StoragePathGuard(model_root)
        self._model_root = self._guard.root
        self._sessions_root = self._guard.resolve(TRAINING_SESSIONS_DIR_NAME)

    @property
    def model_root(self) -> Path:
        """Return the model root exposed by this session deletion metadata repository."""
        return self._model_root

    def build_lineage_replacements(
        self,
        deleted_session: SessionStorageName,
        *,
        modified_at: str,
    ) -> tuple[MetadataReplacement, ...]:
        """Build canonical updates for every manifest referencing the target."""

        records = self._load_session_records()
        deleted_name = str(deleted_session)
        if deleted_name not in records:
            raise SessionDeletionMetadataError(
                f"Session manifest is missing for {deleted_name!r}."
            )

        replacements: list[MetadataReplacement] = []
        for storage_name, (manifest_path, original_payload) in records.items():
            if storage_name == deleted_name:
                continue

            candidate = deepcopy(original_payload)
            session_section = self._session_section(candidate, manifest_path)

            raw_children = session_section.get("child_session_uuids")
            children_source = (
                raw_children if isinstance(raw_children, list | tuple) else ()
            )
            children = [
                str(child).strip()
                for child in children_source
                if str(child).strip() and str(child).strip() != deleted_name
            ]
            original_children = list(children_source)
            parent_name = str(
                session_section.get("parent_session_uuid") or ""
            ).strip()
            load_source = str(
                session_section.get("load_checkpoint_from_session") or ""
            ).strip()
            patch = SessionMetadataPatch(
                modified_at=modified_at,
                child_session_uuids=(
                    tuple(children)
                    if children != original_children
                    else None
                ),
                clear_parent_session_uuid=parent_name == deleted_name,
                clear_load_checkpoint_source=load_source == deleted_name,
            )
            if not patch.has_changes:
                continue
            patch.apply_to(session_section)
            replacements.append(
                MetadataReplacement(
                    target=manifest_path,
                    kind=MetadataKind.SESSION_MANIFEST,
                    payload=self._validate_payload(
                        MetadataKind.SESSION_MANIFEST,
                        candidate,
                    ),
                )
            )
        return tuple(replacements)

    def build_runtime_selection_replacement(
        self,
        *,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> MetadataReplacement:
        """Build a canonical model configuration with the replacement selection."""

        if session_name:
            self._require_session_manifest(SessionStorageName.parse(session_name))
        target, payload, section = self._load_model_configuration()

        updated_section = dict(section)
        updated_section["selected_default_model_name"] = session_name or ""
        updated_section["selected_default_checkpoint_name"] = (
            checkpoint_name if session_name else ""
        ) or ""
        schema = ConfigurationSchema.from_configuration(
            updated_section,
            model_folder_path=str(self._model_root),
        )
        candidate = deepcopy(payload)
        candidate.update(
            schema.to_model_configuration_payload(
                model_folder_path=str(self._model_root),
            )
        )
        return MetadataReplacement(
            target=target,
            kind=MetadataKind.MODEL_CONFIGURATION,
            payload=self._validate_payload(
                MetadataKind.MODEL_CONFIGURATION,
                candidate,
            ),
        )

    def load_runtime_selection(self) -> PersistedRuntimeSelection:
        """Read the durable selection used to detect stale deleted references."""

        _target, _payload, section = self._load_model_configuration()
        session_name = str(
            section.get("selected_default_model_name") or ""
        ).strip() or None
        checkpoint_name = str(
            section.get("selected_default_checkpoint_name") or ""
        ).strip() or None
        if session_name is None:
            checkpoint_name = None
        return PersistedRuntimeSelection(session_name, checkpoint_name)

    def stage(
        self,
        transaction: StorageTransaction,
        replacements: tuple[MetadataReplacement, ...],
    ) -> tuple[StagedMetadataReplacement, ...]:
        """Write and read back canonical replacements without publishing them."""

        staged: list[StagedMetadataReplacement] = []
        seen_targets: set[Path] = set()
        for replacement in replacements:
            target = self._require_regular_metadata_file(replacement.target)
            if target in seen_targets:
                raise SessionDeletionMetadataError(
                    f"Duplicate metadata replacement target: {target!s}."
                )
            seen_targets.add(target)
            expected_payload = self._validate_payload(
                replacement.kind,
                replacement.payload,
            )
            staging = transaction.stage_file(target)
            atomic_write_validated_yaml_file(
                str(staging),
                expected_payload,
                validator=partial(
                    self._validate_payload,
                    replacement.kind,
                ),
                verify_reader=read_yaml_mapping_file,
                smoke_check_reader=read_yaml_mapping_file,
            )
            staged.append(
                StagedMetadataReplacement(
                    target=target,
                    staging=staging,
                    kind=replacement.kind,
                    expected_payload=expected_payload,
                )
            )
        return tuple(staged)

    def publish(
        self,
        transaction: StorageTransaction,
        replacements: tuple[StagedMetadataReplacement, ...],
    ) -> None:
        """Journal replacement of each original and verify the published readback."""

        for replacement in replacements:
            transaction.quarantine(replacement.target)
            transaction.publish(replacement.staging, replacement.target)
            readback = self._validate_payload(
                replacement.kind,
                read_yaml_mapping_file(str(replacement.target)),
            )
            if readback != replacement.expected_payload:
                raise SessionDeletionMetadataError(
                    f"Published metadata readback differs for {replacement.target!s}."
                )

    def _load_session_records(
        self,
    ) -> dict[str, tuple[Path, dict[str, object]]]:
        """Load the session records owned by this session deletion metadata repository."""
        if not self._sessions_root.is_dir():
            raise SessionDeletionMetadataError("Training session storage is missing.")
        if self._sessions_root.is_symlink():
            raise StoragePathViolation(
                "Training session storage cannot be a symbolic link."
            )

        records: dict[str, tuple[Path, dict[str, object]]] = {}
        for entry in sorted(self._sessions_root.iterdir(), key=lambda path: path.name):
            if entry.is_symlink():
                raise StoragePathViolation(
                    f"Session storage cannot be a symbolic link: {entry.name!r}."
                )
            if not entry.is_dir():
                continue
            storage_name = SessionStorageName.parse(entry.name)
            manifest_path = entry / SESSION_MANIFEST_FILE_NAME
            if not manifest_path.exists():
                continue
            manifest_path = self._require_regular_metadata_file(manifest_path)
            payload = self._validate_payload(
                MetadataKind.SESSION_MANIFEST,
                read_yaml_mapping_file(str(manifest_path)),
            )
            session_section = self._session_section(payload, manifest_path)
            declared_storage_name = str(
                session_section.get("storage_name") or ""
            ).strip()
            if declared_storage_name != str(storage_name):
                raise SessionDeletionMetadataError(
                    f"Session manifest storage name mismatch: {entry.name!r}."
                )
            records[str(storage_name)] = (manifest_path, payload)
        return records

    def _load_model_configuration(
        self,
    ) -> tuple[Path, dict[str, object], dict[str, object]]:
        """Load the model configuration owned by this session deletion metadata repository."""
        target = self._guard.resolve(MODEL_CONFIGURATION_FILE_NAME)
        target = self._require_regular_metadata_file(target)
        payload = dict(
            validate_versioned_metadata_payload(
                MetadataKind.MODEL_CONFIGURATION,
                read_yaml_mapping_file(str(target)),
            )
        )
        section = payload.get("model_configuration")
        if not isinstance(section, dict):
            raise SessionDeletionMetadataError(
                "Model configuration is missing its configuration section."
            )
        return target, payload, section

    def _require_session_manifest(
        self,
        storage_name: SessionStorageName,
    ) -> Path:
        """Return the required session manifest for this session deletion metadata repository."""
        manifest_path = self._guard.resolve(
            TRAINING_SESSIONS_DIR_NAME,
            storage_name,
            SESSION_MANIFEST_FILE_NAME,
        )
        manifest_path = self._require_regular_metadata_file(manifest_path)
        payload = self._validate_payload(
            MetadataKind.SESSION_MANIFEST,
            read_yaml_mapping_file(str(manifest_path)),
        )
        section = self._session_section(payload, manifest_path)
        declared_name = str(section.get("storage_name") or "").strip()
        if declared_name != str(storage_name):
            raise SessionDeletionMetadataError(
                f"Session manifest storage name mismatch: {storage_name!s}."
            )
        return manifest_path

    def _require_regular_metadata_file(self, candidate: Path) -> Path:
        """Return the required regular metadata file for this session deletion metadata repository.
        """
        lexical_path = Path(candidate).absolute()
        guarded_path = self._guard.require_descendant(candidate, must_exist=True)
        if lexical_path != guarded_path or candidate.is_symlink():
            raise StoragePathViolation(
                f"Metadata aliases are forbidden: {candidate!s}."
            )
        if not guarded_path.is_file():
            raise SessionDeletionMetadataError(
                f"Metadata file is missing: {guarded_path!s}."
            )
        return guarded_path

    @staticmethod
    def _session_section(
        payload: dict[str, object],
        source: Path,
    ) -> dict[str, object]:
        """Return the session section value owned by this session deletion metadata repository."""
        section = payload.get("session")
        if not isinstance(section, dict):
            raise SessionDeletionMetadataError(
                f"Session manifest is missing its session section: {source!s}."
            )
        return section

    @staticmethod
    def _validate_payload(
        kind: MetadataKind,
        payload: dict[str, object],
    ) -> dict[str, object]:
        """Validate the payload required by this session deletion metadata repository."""
        try:
            if kind is MetadataKind.SESSION_MANIFEST:
                versioned_payload = validate_versioned_metadata_payload(
                    kind,
                    payload,
                )
                return dict(
                    validate_session_manifest_payload(versioned_payload)
                )
            return dict(validate_versioned_metadata_payload(kind, payload))
        except Exception as exc:
            raise SessionDeletionMetadataError(
                f"Invalid {kind.value} metadata: {exc}"
            ) from exc
