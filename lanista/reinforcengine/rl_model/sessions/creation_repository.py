# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Prepare validated metadata for transactional session creation."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from lanista.reinforcengine.rl_model.checkpoints.catalog_repository import (
    CheckpointCatalogRepository,
)
from lanista.reinforcengine.rl_model.checkpoints.storage_resolver import (
    CheckpointStorageResolver,
)
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.metadata.configuration_schema import (
    VALID_FRAMEWORK_UUIDS,
    ConfigurationSchema,
)
from lanista.reinforcengine.rl_model.metadata.manifest_schema import (
    validate_manifest_payload,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.session_manifest_schema import (
    validate_session_manifest_payload,
)
from lanista.reinforcengine.rl_model.name_formatting_service import NameFormattingService
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    SESSION_MANIFEST_FILE_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    SessionStorageName,
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


class SessionCreationMetadataError(StorageTransactionError):
    """Raised when one creation participant cannot be prepared or verified."""


@dataclass(frozen=True, slots=True)
class SessionCreationMetadataRequest:
    """Values required to build all persistent session-creation documents."""

    identity_uuid: str
    label: str
    storage_name: SessionStorageName
    description: str
    framework_uuid: str
    parent_session: SessionStorageName | None
    load_checkpoint_from_session: SessionStorageName | None
    load_checkpoint_name: CheckpointStorageName | None
    base_step_offset: int
    created_at: str


@dataclass(frozen=True, slots=True)
class SessionCreationMetadataReplacement:
    """One existing metadata file and its complete replacement payload."""

    target: Path
    kind: MetadataKind
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class StagedSessionCreationMetadata:
    """Transaction-owned files ready for atomic publication."""

    session_directory: Path
    replacements: tuple[
        tuple[SessionCreationMetadataReplacement, Path], ...
    ]


@dataclass(frozen=True, slots=True)
class PreparedSessionCreationMetadata:
    """Canonical persistent state produced before the first mutation."""

    request: SessionCreationMetadataRequest
    session_path: Path
    session_manifest_payload: dict[str, object]
    replacements: tuple[SessionCreationMetadataReplacement, ...]
    model_version: str
    model_configuration: dict[str, object]
    indexed_sessions: tuple[str, ...]


class SessionCreationMetadataRepository:
    """Build, stage, publish, and read back one session creation commit."""

    def __init__(self, model_root: str | Path) -> None:
        """Initialize the session creation metadata repository and its owned state."""
        self._guard = StoragePathGuard(model_root)
        self._model_root = self._guard.root
        self._sessions_root = self._model_root / TRAINING_SESSIONS_DIR_NAME

    @property
    def model_root(self) -> Path:
        """Return the model root exposed by this session creation metadata repository."""
        return self._model_root

    @property
    def sessions_root(self) -> Path:
        """Return the sessions root exposed by this session creation metadata repository."""
        return self._sessions_root

    def validate_sessions_root(
        self,
        supplied_sessions_root: str | os.PathLike[str],
    ) -> None:
        """Reject an untrusted provider path before it is read or scanned."""

        self._require_exact_path(
            supplied_sessions_root,
            self._sessions_root,
            label="Training sessions root",
        )

    def validate_layout(
        self,
        *,
        supplied_sessions_root: str | os.PathLike[str],
        supplied_session_manifest: str | os.PathLike[str],
        storage_name: SessionStorageName,
    ) -> Path:
        """Require provider paths to equal the canonical managed layout."""

        expected_session_path = self._guard.resolve(
            TRAINING_SESSIONS_DIR_NAME,
            storage_name,
        )
        expected_manifest = self._guard.resolve(
            TRAINING_SESSIONS_DIR_NAME,
            storage_name,
            SESSION_MANIFEST_FILE_NAME,
        )
        self.validate_sessions_root(supplied_sessions_root)
        self._require_exact_path(
            supplied_session_manifest,
            expected_manifest,
            label="Session manifest",
        )
        if os.path.lexists(expected_session_path):
            raise FileExistsError(expected_session_path)
        return expected_session_path

    def session_names(self) -> tuple[str, ...]:
        """Return canonical physical session names without mutating metadata."""

        return tuple(self._load_session_records())

    def session_framework_uuid(self, session_name: str) -> str | None:
        """Read one explicit session framework without inference side effects."""

        record = self._load_session_records().get(str(session_name))
        if record is None:
            return None
        section = self._session_section(record[1], record[0])
        value = str(section.get("framework_uuid") or "").strip().lower()
        return value or None

    def list_checkpoint_names(
        self,
        session_name: str,
        framework_uuid: str,
    ) -> tuple[str, ...]:
        """List checkpoints without framework inference or metadata writes."""

        if self.session_framework_uuid(session_name) != framework_uuid:
            return ()
        catalog = CheckpointCatalogRepository(
            sessions_root_provider=lambda: str(self._sessions_root),
            session_framework_provider=lambda _value: framework_uuid,
            framework_normalizer=lambda value: str(value or "").strip().lower() or None,
        )
        return tuple(catalog.list_for_session(session_name, framework_uuid))

    def prepare(
        self,
        request: SessionCreationMetadataRequest,
        *,
        expected_model_uuid: str,
        session_path: Path,
    ) -> PreparedSessionCreationMetadata:
        """Build every replacement in memory before starting a transaction."""

        if request.framework_uuid not in VALID_FRAMEWORK_UUIDS:
            raise SessionCreationMetadataError(
                "Session creation requires a supported training framework."
            )
        model_target, model_payload = self._load_model_manifest()
        model_section = self._model_section(model_payload, model_target)
        persisted_uuid = str(model_section.get("uuid") or "").strip()
        if not expected_model_uuid or persisted_uuid != expected_model_uuid:
            raise SessionCreationMetadataError(
                "The model manifest owner changed before session creation."
            )

        records = self._load_session_records()
        storage_name = str(request.storage_name)
        if storage_name in records or os.path.lexists(session_path):
            raise FileExistsError(session_path)
        self._validate_checkpoint_source(request, records)

        session_payload = self._build_new_session_payload(request)
        all_payloads = {
            name: deepcopy(payload)
            for name, (_path, payload) in records.items()
        }
        all_payloads[storage_name] = session_payload
        self._apply_parent_link(request, all_payloads)
        model_version = self._apply_identity_versions(
            all_payloads,
            current_model_version=str(model_section.get("version") or ""),
            model_name=str(model_section.get("name") or ""),
        )

        replacements = self._build_session_replacements(records, all_payloads)
        updated_model = deepcopy(model_payload)
        self._model_section(updated_model, model_target)["version"] = model_version
        replacements.append(
            SessionCreationMetadataReplacement(
                model_target,
                MetadataKind.MODEL_MANIFEST,
                self._validate_payload(MetadataKind.MODEL_MANIFEST, updated_model),
            )
        )
        config_target, config_payload = self._build_model_configuration(
            request.storage_name,
            request.framework_uuid,
        )
        replacements.append(
            SessionCreationMetadataReplacement(
                config_target,
                MetadataKind.MODEL_CONFIGURATION,
                config_payload,
            )
        )
        indexed_sessions = list(records)
        if storage_name not in indexed_sessions:
            indexed_sessions.append(storage_name)
        configuration_section = config_payload.get("model_configuration")
        if not isinstance(configuration_section, dict):
            raise SessionCreationMetadataError(
                "Prepared model configuration is missing its configuration section."
            )
        return PreparedSessionCreationMetadata(
            request=request,
            session_path=session_path,
            session_manifest_payload=all_payloads[storage_name],
            replacements=tuple(replacements),
            model_version=model_version,
            model_configuration=dict(configuration_section),
            indexed_sessions=tuple(indexed_sessions),
        )

    def stage(
        self,
        transaction: StorageTransaction,
        prepared: PreparedSessionCreationMetadata,
    ) -> StagedSessionCreationMetadata:
        """Write canonical replacements entirely inside transaction staging."""

        session_directory = transaction.stage_directory(prepared.session_path)
        checkpoints = session_directory / "checkpoints"
        checkpoints.mkdir()
        session_manifest = session_directory / SESSION_MANIFEST_FILE_NAME
        self._write_payload(
            session_manifest,
            MetadataKind.SESSION_MANIFEST,
            prepared.session_manifest_payload,
        )
        transaction.confirm_path(session_directory)

        staged_replacements: list[
            tuple[SessionCreationMetadataReplacement, Path]
        ] = []
        seen_targets: set[Path] = set()
        for replacement in prepared.replacements:
            target = self._require_regular_metadata_file(replacement.target)
            if target in seen_targets:
                raise SessionCreationMetadataError(
                    f"Duplicate session-creation target: {target!s}."
                )
            seen_targets.add(target)
            staging = transaction.stage_file(target)
            self._write_payload(staging, replacement.kind, replacement.payload)
            staged_replacements.append((replacement, staging))
        return StagedSessionCreationMetadata(
            session_directory=session_directory,
            replacements=tuple(staged_replacements),
        )

    def publish(
        self,
        transaction: StorageTransaction,
        prepared: PreparedSessionCreationMetadata,
        staged: StagedSessionCreationMetadata,
    ) -> None:
        """Publish every participant and require exact validated readback."""

        if not self._sessions_root.exists():
            transaction.track_created(self._sessions_root)
            self._sessions_root.mkdir()
            transaction.confirm_path(self._sessions_root)
        self._require_exact_directory(self._sessions_root, "Training sessions root")

        for replacement, staging in staged.replacements:
            transaction.quarantine(replacement.target)
            transaction.publish(staging, replacement.target)
            self._require_expected_payload(replacement)
            transaction.confirm_path(replacement.target)
        transaction.publish(staged.session_directory, prepared.session_path)
        session_manifest = prepared.session_path / SESSION_MANIFEST_FILE_NAME
        session_replacement = SessionCreationMetadataReplacement(
            session_manifest,
            MetadataKind.SESSION_MANIFEST,
            prepared.session_manifest_payload,
        )
        self._require_expected_payload(session_replacement)
        transaction.confirm_path(prepared.session_path)

    def _build_new_session_payload(
        self,
        request: SessionCreationMetadataRequest,
    ) -> dict[str, object]:
        """Build the new session payload required by this session creation metadata repository."""
        return self._validate_payload(
            MetadataKind.SESSION_MANIFEST,
            {
                "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
                "session": {
                    "name": request.label,
                    "uuid": request.identity_uuid,
                    "storage_name": str(request.storage_name),
                    "description": request.description,
                    "tags": [],
                    "framework_uuid": request.framework_uuid,
                    "created_at": request.created_at,
                    "modified_at": request.created_at,
                    "parent_session_uuid": (
                        str(request.parent_session)
                        if request.parent_session is not None
                        else None
                    ),
                    "child_session_uuids": [],
                    "load_checkpoint_from_session": (
                        str(request.load_checkpoint_from_session)
                        if request.load_checkpoint_from_session is not None
                        else None
                    ),
                    "load_checkpoint_name": (
                        str(request.load_checkpoint_name)
                        if request.load_checkpoint_name is not None
                        else None
                    ),
                    "base_step_offset": request.base_step_offset,
                },
            },
        )

    def _apply_parent_link(
        self,
        request: SessionCreationMetadataRequest,
        payloads: dict[str, dict[str, object]],
    ) -> None:
        """Apply the parent link transition owned by this session creation metadata repository."""
        if request.parent_session is None:
            return
        parent_name = str(request.parent_session)
        parent = payloads.get(parent_name)
        if parent is None:
            raise SessionCreationMetadataError(
                f"Parent session is unavailable: {parent_name!r}."
            )
        section = self._session_section(
            parent,
            self._sessions_root / parent_name / SESSION_MANIFEST_FILE_NAME,
        )
        raw_children = section.get("child_session_uuids")
        children_source = raw_children if isinstance(raw_children, list | tuple) else ()
        children = [
            str(value).strip()
            for value in children_source
            if str(value).strip()
        ]
        child_name = str(request.storage_name)
        if child_name not in children:
            children.append(child_name)
        section["child_session_uuids"] = children
        section["modified_at"] = request.created_at

    @staticmethod
    def _session_order(payload: dict[str, object], storage_name: str) -> tuple[float, str]:
        """Return the session order value owned by this session creation metadata repository."""
        section = payload.get("session")
        created_at = str(section.get("created_at") or "") if isinstance(section, dict) else ""
        try:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp()
        except ValueError:
            timestamp = float("inf")
        return timestamp, storage_name

    def _apply_identity_versions(
        self,
        payloads: dict[str, dict[str, object]],
        *,
        current_model_version: str,
        model_name: str,
    ) -> str:
        """Apply the identity versions transition owned by this session creation metadata
        repository.
        """
        base_name = NameFormattingService.sanitize_name(
            model_name,
            fallback="session",
        )
        parsed = EmptyModelGenerator.parse_composite_version(current_model_version)
        highest_version = parsed[2] if parsed is not None else 0
        for payload in payloads.values():
            highest_version = max(
                highest_version,
                self._positive_int(payload.get("session_version")),
            )

        ordered = sorted(
            payloads.items(),
            key=lambda item: self._session_order(item[1], item[0]),
        )
        for _storage_name, payload in ordered:
            section = self._session_section(payload, self._sessions_root)
            if not str(section.get("name") or "").strip():
                section["name"] = base_name
            version = self._positive_int(payload.get("session_version"))
            if version <= 0:
                highest_version += 1
                version = highest_version
                payload["session_version"] = version
            else:
                highest_version = max(highest_version, version)
            if not str(payload.get("model_semantic_version") or "").strip():
                payload["model_semantic_version"] = current_model_version

        if parsed is None:
            return current_model_version
        manual_major, model_number, _session_number = parsed
        return EmptyModelGenerator.build_composite_version(
            manual_major,
            model_number,
            highest_version,
        )

    def _build_session_replacements(
        self,
        records: dict[str, tuple[Path, dict[str, object]]],
        payloads: dict[str, dict[str, object]],
    ) -> list[SessionCreationMetadataReplacement]:
        """Build the session replacements required by this session creation metadata repository."""
        replacements: list[SessionCreationMetadataReplacement] = []
        for name, (target, original) in records.items():
            candidate = self._validate_payload(
                MetadataKind.SESSION_MANIFEST,
                payloads[name],
            )
            if candidate == original:
                continue
            replacements.append(
                SessionCreationMetadataReplacement(
                    target,
                    MetadataKind.SESSION_MANIFEST,
                    candidate,
                )
            )
        return replacements

    def _build_model_configuration(
        self,
        session_name: SessionStorageName,
        framework_uuid: str,
    ) -> tuple[Path, dict[str, object]]:
        """Build the model configuration required by this session creation metadata repository."""
        target = self._model_root / MODEL_CONFIGURATION_FILE_NAME
        target = self._require_regular_metadata_file(target)
        payload = self._validate_payload(
            MetadataKind.MODEL_CONFIGURATION,
            read_yaml_mapping_file(str(target)),
        )
        section = payload.get("model_configuration")
        if not isinstance(section, dict):
            raise SessionCreationMetadataError(
                "Model configuration is missing its configuration section."
            )
        updated = dict(section)
        updated.update(
            {
                "active_runtime_framework_uuid": framework_uuid,
                "selected_default_model_name": str(session_name),
                "selected_default_checkpoint_name": "",
            }
        )
        schema = ConfigurationSchema.from_configuration(
            updated,
            model_folder_path=str(self._model_root),
        )
        return target, self._validate_payload(
            MetadataKind.MODEL_CONFIGURATION,
            schema.to_model_configuration_payload(
                model_folder_path=str(self._model_root),
            ),
        )

    def _validate_checkpoint_source(
        self,
        request: SessionCreationMetadataRequest,
        records: dict[str, tuple[Path, dict[str, object]]],
    ) -> None:
        """Validate the checkpoint source required by this session creation metadata repository."""
        if request.load_checkpoint_name is None:
            return
        source = request.load_checkpoint_from_session
        if source is None or str(source) not in records:
            raise SessionCreationMetadataError(
                "A fork checkpoint requires an existing source session."
            )
        checkpoints_root = (
            self._sessions_root / str(source) / "checkpoints"
        )
        self._require_exact_directory(checkpoints_root, "Checkpoint storage root")
        checkpoint_name = str(request.load_checkpoint_name)
        resolved_raw = CheckpointStorageResolver.try_resolve_candidate_from_root(
            checkpoints_root,
            checkpoint_name,
        )
        if resolved_raw is None:
            raise SessionCreationMetadataError(
                "The selected source checkpoint is unavailable."
            )
        resolved = self._guard.require_descendant(
            resolved_raw,
            must_exist=True,
        )
        if not self._is_exact_checkpoint_candidate(
            checkpoints_root,
            checkpoint_name,
            resolved,
        ):
            raise StoragePathViolation("Checkpoint storage aliases are forbidden.")

    def _is_exact_checkpoint_candidate(
        self,
        checkpoints_root: Path,
        checkpoint_name: str,
        resolved: Path,
    ) -> bool:
        """Return whether exact checkpoint candidate applies to this session creation metadata
        repository.
        """
        stem = Path(checkpoint_name).stem
        candidates = [
            checkpoints_root / checkpoint_name,
            checkpoints_root / "stablebaselines" / stem / checkpoint_name,
        ]
        if not checkpoint_name.lower().endswith(".zip"):
            candidates.append(
                checkpoints_root
                / "stablebaselines"
                / stem
                / f"{checkpoint_name}.zip"
            )
        for candidate in candidates:
            if not os.path.lexists(candidate):
                continue
            guarded = self._guard.require_descendant(candidate, must_exist=True)
            if candidate.absolute() != guarded or candidate.is_symlink():
                raise StoragePathViolation(
                    "Checkpoint storage aliases are forbidden."
                )
            if guarded == resolved:
                return True
        return False

    def _load_model_manifest(self) -> tuple[Path, dict[str, object]]:
        """Load the model manifest owned by this session creation metadata repository."""
        target = self._model_root / MODEL_MANIFEST_FILE_NAME
        target = self._require_regular_metadata_file(target)
        return target, self._validate_payload(
            MetadataKind.MODEL_MANIFEST,
            read_yaml_mapping_file(str(target)),
        )

    def _load_session_records(
        self,
    ) -> dict[str, tuple[Path, dict[str, object]]]:
        """Load the session records owned by this session creation metadata repository."""
        if not self._sessions_root.exists():
            return {}
        self._require_exact_directory(self._sessions_root, "Training sessions root")
        records: dict[str, tuple[Path, dict[str, object]]] = {}
        for entry in sorted(self._sessions_root.iterdir(), key=lambda path: path.name):
            if entry.is_symlink():
                raise StoragePathViolation(
                    f"Session storage cannot be a symbolic link: {entry.name!r}."
                )
            if not entry.is_dir():
                continue
            storage_name = SessionStorageName.parse(entry.name)
            target = entry / SESSION_MANIFEST_FILE_NAME
            if not target.exists():
                continue
            target = self._require_regular_metadata_file(target)
            payload = self._validate_payload(
                MetadataKind.SESSION_MANIFEST,
                read_yaml_mapping_file(str(target)),
            )
            section = self._session_section(payload, target)
            if str(section.get("storage_name") or "").strip() != str(storage_name):
                raise SessionCreationMetadataError(
                    f"Session manifest storage name mismatch: {entry.name!r}."
                )
            records[str(storage_name)] = (target, payload)
        return records

    def _require_expected_payload(
        self,
        replacement: SessionCreationMetadataReplacement,
    ) -> None:
        """Return the required expected payload for this session creation metadata repository."""
        readback = self._validate_payload(
            replacement.kind,
            read_yaml_mapping_file(str(replacement.target)),
        )
        expected = self._validate_payload(replacement.kind, replacement.payload)
        if readback != expected:
            raise SessionCreationMetadataError(
                f"Published session metadata readback differs: {replacement.target!s}."
            )

    def _write_payload(
        self,
        target: Path,
        kind: MetadataKind,
        payload: dict[str, object],
    ) -> None:
        """Persist the payload owned by this session creation metadata repository."""
        atomic_write_validated_yaml_file(
            str(target),
            payload,
            validator=lambda value: self._validate_payload(kind, value),
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )

    def _require_regular_metadata_file(self, candidate: Path) -> Path:
        """Return the required regular metadata file for this session creation metadata repository.
        """
        lexical = candidate.absolute()
        guarded = self._guard.require_descendant(candidate, must_exist=True)
        if lexical != guarded or candidate.is_symlink():
            raise StoragePathViolation(
                f"Metadata aliases are forbidden: {candidate!s}."
            )
        if not guarded.is_file():
            raise SessionCreationMetadataError(
                f"Metadata file is missing: {guarded!s}."
            )
        return guarded

    def _require_exact_path(
        self,
        supplied: str | os.PathLike[str],
        expected: Path,
        *,
        label: str,
    ) -> None:
        """Return the required exact path for this session creation metadata repository."""
        raw = Path(supplied).expanduser()
        if not raw.is_absolute():
            raise StoragePathViolation(f"{label} must be an absolute managed path.")
        raw_text = os.fspath(supplied)
        if ".." in raw.parts or "\\" in raw_text:
            raise StoragePathViolation(
                f"{label} cannot contain relative or ambiguous path components."
            )
        lexical = Path(os.path.abspath(raw))
        guarded = self._guard.require_descendant(raw)
        if lexical != expected or guarded != expected or raw.is_symlink():
            raise StoragePathViolation(
                f"{label} does not match the managed model layout."
            )

    def _require_exact_directory(self, path: Path, label: str) -> None:
        """Return the required exact directory for this session creation metadata repository."""
        lexical = path.absolute()
        guarded = self._guard.require_descendant(path, must_exist=True)
        if lexical != guarded or path.is_symlink() or not guarded.is_dir():
            raise StoragePathViolation(f"{label} is unavailable or aliased.")

    @staticmethod
    def _model_section(
        payload: dict[str, object],
        source: Path,
    ) -> dict[str, object]:
        """Return the model section value owned by this session creation metadata repository."""
        section = payload.get("model")
        if not isinstance(section, dict):
            raise SessionCreationMetadataError(
                f"Model manifest is missing its model section: {source!s}."
            )
        return section

    @staticmethod
    def _session_section(
        payload: dict[str, object],
        source: Path,
    ) -> dict[str, object]:
        """Return the session section value owned by this session creation metadata repository."""
        section = payload.get("session")
        if not isinstance(section, dict):
            raise SessionCreationMetadataError(
                f"Session manifest is missing its session section: {source!s}."
            )
        return section

    @staticmethod
    def _positive_int(value: object) -> int:
        """Return the positive int value owned by this session creation metadata repository."""
        if not isinstance(value, str | int | float):
            return 0
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        return max(0, parsed)

    @staticmethod
    def _validate_payload(
        kind: MetadataKind,
        payload: object,
    ) -> dict[str, object]:
        """Validate the payload required by this session creation metadata repository."""
        try:
            if not isinstance(payload, dict):
                raise ValueError("Metadata payload is not a mapping.")
            if kind is MetadataKind.SESSION_MANIFEST:
                validate_versioned_metadata_payload(kind, payload)
                return dict(validate_session_manifest_payload(payload))
            if kind is MetadataKind.MODEL_MANIFEST:
                validate_versioned_metadata_payload(kind, payload)
                return dict(validate_manifest_payload(payload))
            return dict(validate_versioned_metadata_payload(kind, payload))
        except Exception as exc:
            raise SessionCreationMetadataError(
                f"Invalid {kind.value} metadata: {exc}"
            ) from exc


__all__ = [
    "PreparedSessionCreationMetadata",
    "SessionCreationMetadataError",
    "SessionCreationMetadataRepository",
    "SessionCreationMetadataRequest",
]
