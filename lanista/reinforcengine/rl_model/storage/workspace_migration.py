# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Dry-run-first migration of a legacy external model workspace."""

from __future__ import annotations

import os
import shutil
import stat
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Final

from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.workspace_inventory import (
    WorkspaceDirectoryEntry,
    WorkspaceFileEntry,
    WorkspaceInventory,
    WorkspaceInventoryError,
    WorkspaceInventoryService,
    inventory_from_mapping,
    sha256_regular_file,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    WORKSPACE_DIRECTORY_NAME,
    WorkspaceMigrationError,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    CONTROL_DIRECTORY_NAME as CONTROL_DIRECTORY_NAME,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    JOURNAL_FILE_NAME as JOURNAL_FILE_NAME,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    RECEIPT_DIRECTORY_NAME as RECEIPT_DIRECTORY_NAME,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    MigrationArtifactPaths as _ArtifactPaths,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    artifact_paths as _artifact_paths,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    descendant as _descendant,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    existing_ancestor as _existing_ancestor,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    mkdir_real as _mkdir_real,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    prepare_data_home as _prepare_data_home,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    read_json_mapping as _read_json_mapping,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    relative_to_data_home as _relative_to_data_home,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    remove_tree_no_follow as _remove_tree_no_follow,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    require_artifact_header as _require_artifact_header,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    require_artifact_roots as _require_artifact_roots,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    require_disjoint_roots as _require_disjoint_roots,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    require_operation_id as _require_operation_id,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    resolve_data_home as _resolve_data_home,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    resolve_source_storage as _resolve_source_storage,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    sync_directory as _sync_directory,
)
from lanista.reinforcengine.rl_model.storage.workspace_migration_artifacts import (
    write_json_atomic as _write_json_atomic,
)

_JOURNAL_TYPE: Final = "external_workspace_migration_journal"
_RECEIPT_TYPE: Final = "external_workspace_migration_receipt"
_ROLLBACK_RECEIPT_TYPE: Final = "external_workspace_rollback_receipt"
_CLEANUP_RECEIPT_TYPE: Final = "external_workspace_cleanup_receipt"

MigrationFailureInjector = Callable[[str], None]


class WorkspaceMigrationStatus(StrEnum):
    """Describe an externally visible migration command outcome."""

    DRY_RUN = "dry_run"
    MIGRATED = "migrated"
    RESUMED = "resumed"
    ROLLED_BACK = "rolled_back"
    CLEANED = "cleaned"
    NO_CHANGES = "no_changes"


class _JournalStatus(StrEnum):
    """Describe the durable internal migration state machine."""

    COPYING = "copying"
    READY = "ready"
    PUBLISHED = "published"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True, slots=True)
class WorkspaceMigrationResult:
    """Return inventory, capacity, collision, and recovery evidence."""

    operation_id: str
    status: WorkspaceMigrationStatus
    dry_run: bool
    source_storage: str
    destination_data_home: str
    destination_storage: str
    directory_count: int
    file_count: int
    total_bytes: int
    required_bytes: int
    available_bytes: int
    space_sufficient: bool
    collisions: tuple[str, ...] = ()
    journal_path: str | None = None
    receipt_path: str | None = None
    recovery_path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Return a JSON-compatible command result."""

        return {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "dry_run": self.dry_run,
            "source_storage": self.source_storage,
            "destination_data_home": self.destination_data_home,
            "destination_storage": self.destination_storage,
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "required_bytes": self.required_bytes,
            "available_bytes": self.available_bytes,
            "space_sufficient": self.space_sufficient,
            "collisions": list(self.collisions),
            "journal_path": self.journal_path,
            "receipt_path": self.receipt_path,
            "recovery_path": self.recovery_path,
        }


@dataclass(frozen=True, slots=True)
class _MigrationContext:
    """Hold one validated source, destination, inventory, and capacity view."""

    source_storage: Path
    destination_data_home: Path
    destination_storage: Path
    inventory: WorkspaceInventory
    required_bytes: int
    available_bytes: int
    collisions: tuple[str, ...]


class ExternalWorkspaceMigrationService:
    """Copy, resume, roll back, and clean one explicit legacy workspace."""

    def __init__(
        self,
        inventory_service: WorkspaceInventoryService | None = None,
        *,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the offline service without consulting application state."""

        self._inventory_service = inventory_service or WorkspaceInventoryService()
        self._lock_timeout_seconds = lock_timeout_seconds

    def migrate(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> WorkspaceMigrationResult:
        """Preview or copy a workspace; source data is never moved or deleted."""

        operation_id = uuid.uuid4().hex
        context = self._preflight(source_storage, destination_data_home)
        self._inject(failure_injector, "after_inventory")
        if dry_run:
            return self._result(context, operation_id, WorkspaceMigrationStatus.DRY_RUN, True)
        self._require_write_preflight(context)
        _prepare_data_home(context.destination_data_home)
        with self._lock(context.destination_data_home):
            guarded_context = self._preflight(source_storage, destination_data_home)
            self._require_write_preflight(guarded_context)
            paths = self._create_operation(guarded_context, operation_id)
            self._inject(failure_injector, "after_journal")
            return self._continue_migration(
                guarded_context,
                paths,
                resumed=False,
                failure_injector=failure_injector,
            )

    def resume(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
        journal_path: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> WorkspaceMigrationResult:
        """Resume one journal after revalidating its explicit roots and inventory."""

        source = _resolve_source_storage(source_storage)
        data_home = _resolve_data_home(destination_data_home, must_exist=True)
        payload = _read_json_mapping(Path(journal_path))
        paths, recorded_inventory, status = self._validate_journal(
            payload,
            source,
            data_home,
            Path(journal_path),
        )
        context = self._resume_context(source, data_home, recorded_inventory, status, paths)
        if dry_run:
            return self._result(
                context,
                paths.operation_id,
                WorkspaceMigrationStatus.DRY_RUN,
                True,
                paths=paths,
            )
        with self._lock(data_home):
            payload = _read_json_mapping(paths.journal)
            paths, recorded_inventory, status = self._validate_journal(
                payload,
                source,
                data_home,
                paths.journal,
            )
            context = self._resume_context(source, data_home, recorded_inventory, status, paths)
            return self._continue_migration(
                context,
                paths,
                resumed=True,
                failure_injector=failure_injector,
            )

    def rollback(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
        receipt_path: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> WorkspaceMigrationResult:
        """Back out a published copy by atomically retaining it as recovery data."""

        context, paths, receipt = self._load_receipt_context(
            source_storage,
            destination_data_home,
            receipt_path,
        )
        current_status = _require_receipt_status(receipt)
        self._validate_rollback_state(context, paths, current_status)
        if dry_run:
            return self._result(
                context,
                paths.operation_id,
                WorkspaceMigrationStatus.DRY_RUN,
                True,
                paths=paths,
                recovery_path=paths.recovery_storage,
            )
        with self._lock(context.destination_data_home):
            context, paths, receipt = self._load_receipt_context(
                source_storage,
                destination_data_home,
                receipt_path,
            )
            current_status = _require_receipt_status(receipt)
            if current_status == WorkspaceMigrationStatus.ROLLED_BACK:
                return self._result(
                    context,
                    paths.operation_id,
                    WorkspaceMigrationStatus.NO_CHANGES,
                    False,
                    paths=paths,
                    recovery_path=paths.recovery_storage,
                )
            self._validate_rollback_state(context, paths, current_status)
            return self._apply_rollback(context, paths, receipt, failure_injector)

    def cleanup(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
        receipt_path: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> WorkspaceMigrationResult:
        """Remove only migration control or rollback-backup artifacts after validation."""

        context, paths, receipt = self._load_receipt_context(
            source_storage,
            destination_data_home,
            receipt_path,
        )
        current_status = _require_receipt_status(receipt)
        if current_status == WorkspaceMigrationStatus.CLEANED:
            return self._result(
                context,
                paths.operation_id,
                WorkspaceMigrationStatus.NO_CHANGES,
                dry_run,
                paths=paths,
            )
        self._validate_cleanup_state(context, paths, current_status)
        if dry_run:
            return self._result(
                context,
                paths.operation_id,
                WorkspaceMigrationStatus.DRY_RUN,
                True,
                paths=paths,
                recovery_path=(
                    paths.recovery_storage
                    if current_status == WorkspaceMigrationStatus.ROLLED_BACK
                    else None
                ),
            )
        with self._lock(context.destination_data_home):
            context, paths, receipt = self._load_receipt_context(
                source_storage,
                destination_data_home,
                receipt_path,
            )
            current_status = _require_receipt_status(receipt)
            if current_status == WorkspaceMigrationStatus.CLEANED:
                return self._result(
                    context,
                    paths.operation_id,
                    WorkspaceMigrationStatus.NO_CHANGES,
                    False,
                    paths=paths,
                )
            self._validate_cleanup_state(context, paths, current_status)
            return self._apply_cleanup(
                context,
                paths,
                receipt,
                current_status,
                failure_injector,
            )

    def _preflight(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
    ) -> _MigrationContext:
        """Build a read-only inventory, collision report, and capacity view."""

        source = _resolve_source_storage(source_storage)
        data_home = _resolve_data_home(destination_data_home, must_exist=False)
        target = data_home / WORKSPACE_DIRECTORY_NAME
        _require_disjoint_roots(source, data_home, target)
        inventory = self._inventory(source, "source workspace")
        required_bytes = _required_capacity(inventory)
        available_bytes = shutil.disk_usage(_existing_ancestor(data_home)).free
        collisions = self._collision_paths(target)
        return _MigrationContext(
            source_storage=source,
            destination_data_home=data_home,
            destination_storage=target,
            inventory=inventory,
            required_bytes=required_bytes,
            available_bytes=available_bytes,
            collisions=collisions,
        )

    def _resume_context(
        self,
        source: Path,
        data_home: Path,
        recorded_inventory: WorkspaceInventory,
        status: _JournalStatus,
        paths: _ArtifactPaths,
    ) -> _MigrationContext:
        """Rebuild and compare roots before a journal can resume."""

        current_inventory = self._inventory(source, "source workspace")
        _require_same_inventory(recorded_inventory, current_inventory, "source")
        target = data_home / WORKSPACE_DIRECTORY_NAME
        _require_disjoint_roots(source, data_home, target)
        if status == _JournalStatus.COPYING and os.path.lexists(target):
            raise WorkspaceMigrationError("Destination appeared before migration publication.")
        if status == _JournalStatus.READY and os.path.lexists(target):
            if os.path.lexists(paths.staging_storage):
                raise WorkspaceMigrationError("Both staging and destination workspaces exist.")
            self._require_inventory(target, recorded_inventory, "published destination")
        if status == _JournalStatus.PUBLISHED:
            self._require_inventory(target, recorded_inventory, "published destination")
        if status == _JournalStatus.ROLLED_BACK:
            self._require_inventory(paths.recovery_storage, recorded_inventory, "rollback recovery")
        return _MigrationContext(
            source_storage=source,
            destination_data_home=data_home,
            destination_storage=target,
            inventory=recorded_inventory,
            required_bytes=_required_capacity(recorded_inventory),
            available_bytes=shutil.disk_usage(data_home).free,
            collisions=(".",) if os.path.lexists(target) else (),
        )

    def _create_operation(
        self,
        context: _MigrationContext,
        operation_id: str,
    ) -> _ArtifactPaths:
        """Create a private staging root and its first durable journal."""

        paths = _artifact_paths(context.destination_data_home, operation_id)
        _mkdir_real(paths.operation_root.parent, parents=True)
        _mkdir_real(paths.receipt.parent, parents=True)
        os.chmod(paths.operation_root.parent, 0o700, follow_symlinks=False)
        os.chmod(paths.receipt.parent, 0o700, follow_symlinks=False)
        self._require_no_existing_operation(paths.operation_root.parent)
        if os.path.lexists(paths.operation_root):
            raise WorkspaceMigrationError("Migration operation identifier already exists.")
        _mkdir_real(paths.staging_storage, parents=True)
        os.chmod(paths.operation_root, 0o700, follow_symlinks=False)
        os.chmod(paths.staging_storage.parent, 0o700, follow_symlinks=False)
        _write_json_atomic(
            paths.journal,
            self._journal_payload(context, paths, _JournalStatus.COPYING),
        )
        return paths

    def _continue_migration(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        *,
        resumed: bool,
        failure_injector: MigrationFailureInjector | None,
    ) -> WorkspaceMigrationResult:
        """Advance a validated journal through copy, readback, and publication."""

        payload = _read_json_mapping(paths.journal)
        _, recorded_inventory, status = self._validate_journal(
            payload,
            context.source_storage,
            context.destination_data_home,
            paths.journal,
        )
        if status == _JournalStatus.ROLLED_BACK:
            raise WorkspaceMigrationError("A rolled-back migration cannot be resumed.")
        if status == _JournalStatus.COPYING:
            self._copy_inventory(context, paths, failure_injector)
            self._require_stable_source(context)
            self._require_inventory(paths.staging_storage, recorded_inventory, "staging workspace")
            _write_json_atomic(
                paths.journal,
                self._journal_payload(context, paths, _JournalStatus.READY),
            )
            status = _JournalStatus.READY
        if status == _JournalStatus.READY:
            if os.path.lexists(context.destination_storage):
                if os.path.lexists(paths.staging_storage):
                    raise WorkspaceMigrationError("Both staging and destination workspaces exist.")
                self._require_inventory(
                    context.destination_storage,
                    recorded_inventory,
                    "published destination",
                )
                _write_json_atomic(
                    paths.journal,
                    self._journal_payload(context, paths, _JournalStatus.PUBLISHED),
                )
            else:
                self._publish(context, paths, failure_injector)
            status = _JournalStatus.PUBLISHED
        if status != _JournalStatus.PUBLISHED:
            raise WorkspaceMigrationError("Migration journal has an unsupported state.")
        self._require_inventory(context.destination_storage, recorded_inventory, "published destination")
        self._require_stable_source(context)
        _write_json_atomic(
            paths.receipt,
            self._receipt_payload(context, paths, WorkspaceMigrationStatus.MIGRATED),
        )
        return self._result(
            context,
            paths.operation_id,
            WorkspaceMigrationStatus.RESUMED if resumed else WorkspaceMigrationStatus.MIGRATED,
            False,
            paths=paths,
        )

    def _copy_inventory(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        failure_injector: MigrationFailureInjector | None,
    ) -> None:
        """Materialize directories and regular files into private staging."""

        for directory in context.inventory.directories:
            destination = _descendant(paths.staging_storage, directory.relative_path)
            _mkdir_real(destination, parents=True)
        for entry in context.inventory.files:
            destination = _descendant(paths.staging_storage, entry.relative_path)
            _mkdir_real(destination.parent, parents=True)
            if not _file_matches(destination, entry):
                self._inventory_service.copy_file(context.source_storage, entry, destination)
            os.chmod(destination, entry.mode, follow_symlinks=False)
            os.utime(
                destination,
                ns=(entry.modified_ns, entry.modified_ns),
                follow_symlinks=False,
            )
            self._inject(failure_injector, f"after_copy:{entry.relative_path}")
        self._apply_directory_metadata(paths.staging_storage, context.inventory.directories)

    def _publish(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        failure_injector: MigrationFailureInjector | None,
    ) -> None:
        """Atomically publish verified staging into the fixed destination name."""

        if os.path.lexists(context.destination_storage):
            raise WorkspaceMigrationError("Destination collision appeared before publication.")
        self._inject(failure_injector, "before_publish")
        if os.path.lexists(context.destination_storage):
            raise WorkspaceMigrationError("Destination collision appeared during publication.")
        os.replace(paths.staging_storage, context.destination_storage)
        _sync_directory(context.destination_data_home)
        self._inject(failure_injector, "after_publish")
        _write_json_atomic(
            paths.journal,
            self._journal_payload(context, paths, _JournalStatus.PUBLISHED),
        )

    def _apply_rollback(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        receipt: dict[str, Any],
        failure_injector: MigrationFailureInjector | None,
    ) -> WorkspaceMigrationResult:
        """Retain the published tree in recovery and write rollback evidence."""

        _mkdir_real(paths.recovery_storage.parent, parents=True)
        if os.path.lexists(paths.recovery_storage):
            if os.path.lexists(context.destination_storage):
                raise WorkspaceMigrationError("Both destination and rollback recovery exist.")
            self._require_inventory(paths.recovery_storage, context.inventory, "rollback recovery")
        else:
            os.replace(context.destination_storage, paths.recovery_storage)
            _sync_directory(context.destination_data_home)
        self._inject(failure_injector, "after_rollback_rename")
        if paths.journal.is_file():
            _write_json_atomic(
                paths.journal,
                self._journal_payload(context, paths, _JournalStatus.ROLLED_BACK),
            )
        rollback_payload = self._terminal_receipt_payload(
            context,
            paths,
            _ROLLBACK_RECEIPT_TYPE,
            WorkspaceMigrationStatus.ROLLED_BACK,
        )
        _write_json_atomic(paths.rollback_receipt, rollback_payload)
        updated_receipt = dict(receipt)
        updated_receipt["status"] = WorkspaceMigrationStatus.ROLLED_BACK.value
        updated_receipt["recovery_path"] = _relative_to_data_home(
            paths.recovery_storage,
            context.destination_data_home,
        )
        updated_receipt["rollback_receipt"] = _relative_to_data_home(
            paths.rollback_receipt,
            context.destination_data_home,
        )
        _write_json_atomic(paths.receipt, updated_receipt)
        return self._result(
            context,
            paths.operation_id,
            WorkspaceMigrationStatus.ROLLED_BACK,
            False,
            paths=paths,
            recovery_path=paths.recovery_storage,
        )

    def _apply_cleanup(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        receipt: dict[str, Any],
        previous_status: WorkspaceMigrationStatus,
        failure_injector: MigrationFailureInjector | None,
    ) -> WorkspaceMigrationResult:
        """Delete only verified operation artifacts and retain cleanup evidence."""

        if os.path.lexists(paths.operation_root):
            _remove_tree_no_follow(paths.operation_root)
            _sync_directory(paths.operation_root.parent)
        self._inject(failure_injector, "after_cleanup_remove")
        cleanup_payload = self._terminal_receipt_payload(
            context,
            paths,
            _CLEANUP_RECEIPT_TYPE,
            WorkspaceMigrationStatus.CLEANED,
        )
        cleanup_payload["previous_status"] = previous_status.value
        _write_json_atomic(paths.cleanup_receipt, cleanup_payload)
        updated_receipt = dict(receipt)
        updated_receipt["status"] = WorkspaceMigrationStatus.CLEANED.value
        updated_receipt["cleanup_receipt"] = _relative_to_data_home(
            paths.cleanup_receipt,
            context.destination_data_home,
        )
        _write_json_atomic(paths.receipt, updated_receipt)
        return self._result(
            context,
            paths.operation_id,
            WorkspaceMigrationStatus.CLEANED,
            False,
            paths=paths,
        )

    def _load_receipt_context(
        self,
        source_storage: str | os.PathLike[str],
        destination_data_home: str | os.PathLike[str],
        receipt_path: str | os.PathLike[str],
    ) -> tuple[_MigrationContext, _ArtifactPaths, dict[str, Any]]:
        """Load one confined receipt and rebuild its live source context."""

        source = _resolve_source_storage(source_storage)
        data_home = _resolve_data_home(destination_data_home, must_exist=True)
        receipt = _read_json_mapping(Path(receipt_path))
        operation_id = _require_operation_id(receipt.get("operation_id"))
        paths = _artifact_paths(data_home, operation_id)
        if Path(receipt_path).resolve(strict=True) != paths.receipt:
            raise WorkspaceMigrationError("Receipt path is outside its canonical artifact location.")
        _require_artifact_header(receipt, _RECEIPT_TYPE)
        inventory = self._inventory_artifact(receipt.get("inventory"))
        _require_artifact_roots(receipt, source, data_home)
        current_inventory = self._inventory(source, "source workspace")
        _require_same_inventory(inventory, current_inventory, "source")
        target = data_home / WORKSPACE_DIRECTORY_NAME
        context = _MigrationContext(
            source_storage=source,
            destination_data_home=data_home,
            destination_storage=target,
            inventory=inventory,
            required_bytes=_required_capacity(inventory),
            available_bytes=shutil.disk_usage(data_home).free,
            collisions=(".",) if os.path.lexists(target) else (),
        )
        return context, paths, receipt

    def _validate_journal(
        self,
        payload: Mapping[str, Any],
        source: Path,
        data_home: Path,
        journal_path: Path,
    ) -> tuple[_ArtifactPaths, WorkspaceInventory, _JournalStatus]:
        """Validate a journal's schema, roots, name, inventory, and status."""

        _require_artifact_header(payload, _JOURNAL_TYPE)
        operation_id = _require_operation_id(payload.get("operation_id"))
        paths = _artifact_paths(data_home, operation_id)
        if journal_path.resolve(strict=True) != paths.journal:
            raise WorkspaceMigrationError("Journal path is outside its canonical operation root.")
        _require_artifact_roots(payload, source, data_home)
        inventory = self._inventory_artifact(payload.get("inventory"))
        try:
            status = _JournalStatus(str(payload.get("status") or ""))
        except ValueError as exc:
            raise WorkspaceMigrationError("Migration journal status is invalid.") from exc
        return paths, inventory, status

    def _validate_rollback_state(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        status: WorkspaceMigrationStatus,
    ) -> None:
        """Require either a pristine publication or an intact recovery copy."""

        if status == WorkspaceMigrationStatus.MIGRATED:
            if os.path.lexists(context.destination_storage):
                self._require_inventory(
                    context.destination_storage,
                    context.inventory,
                    "published destination",
                )
                if os.path.lexists(paths.recovery_storage):
                    raise WorkspaceMigrationError("Unexpected rollback recovery collision.")
            elif os.path.lexists(paths.recovery_storage):
                self._require_inventory(paths.recovery_storage, context.inventory, "rollback recovery")
            else:
                raise WorkspaceMigrationError("Published workspace and recovery backup are both absent.")
            return
        if status == WorkspaceMigrationStatus.ROLLED_BACK:
            if os.path.lexists(context.destination_storage):
                raise WorkspaceMigrationError("Rolled-back receipt conflicts with a destination workspace.")
            self._require_inventory(paths.recovery_storage, context.inventory, "rollback recovery")
            return
        raise WorkspaceMigrationError("Only a migrated workspace can be rolled back.")

    def _validate_cleanup_state(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        status: WorkspaceMigrationStatus,
    ) -> None:
        """Require all durable data copies before removing recovery controls."""

        if status == WorkspaceMigrationStatus.MIGRATED:
            self._require_inventory(
                context.destination_storage,
                context.inventory,
                "published destination",
            )
            if os.path.lexists(paths.recovery_storage):
                raise WorkspaceMigrationError("Unexpected rollback recovery collision.")
            return
        if status == WorkspaceMigrationStatus.ROLLED_BACK:
            if os.path.lexists(context.destination_storage):
                raise WorkspaceMigrationError("Rolled-back destination unexpectedly exists.")
            if os.path.lexists(paths.recovery_storage):
                self._require_inventory_subset(
                    paths.recovery_storage,
                    context.inventory,
                    "rollback recovery",
                )
            return
        raise WorkspaceMigrationError("Receipt is not eligible for cleanup.")

    def _collision_paths(self, destination_storage: Path) -> tuple[str, ...]:
        """Return deterministic collision evidence without following aliases."""

        if not os.path.lexists(destination_storage):
            return ()
        try:
            path_stat = destination_storage.lstat()
        except OSError as exc:
            raise WorkspaceMigrationError("Destination collision could not be inspected.") from exc
        if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
            return (".",)
        inventory = self._inventory(destination_storage, "destination collision")
        paths = ["."]
        paths.extend(entry.relative_path for entry in inventory.directories)
        paths.extend(entry.relative_path for entry in inventory.files)
        return tuple(paths)

    def _require_inventory(
        self,
        root: Path,
        expected: WorkspaceInventory,
        label: str,
    ) -> None:
        """Require one complete tree to match the receipt inventory."""

        try:
            observed = self._inventory_service.inventory(root)
        except WorkspaceInventoryError as exc:
            raise WorkspaceMigrationError(f"{label.capitalize()} is unavailable or unsafe.") from exc
        _require_same_inventory(expected, observed, label)

    def _require_stable_source(self, context: _MigrationContext) -> None:
        """Re-inventory the source before any terminal publication."""

        observed = self._inventory(context.source_storage, "source workspace")
        _require_same_inventory(context.inventory, observed, "source")

    def _require_inventory_subset(
        self,
        root: Path,
        expected: WorkspaceInventory,
        label: str,
    ) -> None:
        """Require remaining cleanup data to be an unmodified inventory subset."""

        observed = self._inventory(root, label)
        expected_directories = {entry.relative_path for entry in expected.directories}
        expected_files = {entry.relative_path: entry for entry in expected.files}
        if any(
            entry.relative_path not in expected_directories
            for entry in observed.directories
        ):
            raise WorkspaceMigrationError(f"{label.capitalize()} contains an unknown directory.")
        for entry in observed.files:
            expected_entry = expected_files.get(entry.relative_path)
            if (
                expected_entry is None
                or entry.byte_size != expected_entry.byte_size
                or entry.sha256 != expected_entry.sha256
            ):
                raise WorkspaceMigrationError(f"{label.capitalize()} contains changed data.")

    @staticmethod
    def _require_no_existing_operation(control_root: Path) -> None:
        """Force operators to resume or clean every retained operation first."""

        if not os.path.lexists(control_root):
            return
        if control_root.is_symlink() or not control_root.is_dir():
            raise WorkspaceMigrationError("Migration control root is aliased or unsupported.")
        with os.scandir(control_root) as iterator:
            if next(iterator, None) is not None:
                raise WorkspaceMigrationError(
                    "A retained workspace migration must be resumed, rolled back, or cleaned first."
                )

    def _inventory(self, root: Path, label: str) -> WorkspaceInventory:
        """Translate low-level no-follow inventory failures into command errors."""

        try:
            return self._inventory_service.inventory(root)
        except WorkspaceInventoryError as exc:
            raise WorkspaceMigrationError(f"{label.capitalize()} is unavailable or unsafe.") from exc

    @staticmethod
    def _inventory_artifact(raw_inventory: object) -> WorkspaceInventory:
        """Translate strict artifact-inventory parsing failures."""

        try:
            return inventory_from_mapping(raw_inventory)
        except WorkspaceInventoryError as exc:
            raise WorkspaceMigrationError("Migration artifact inventory is invalid.") from exc

    @staticmethod
    def _apply_directory_metadata(
        staging_root: Path,
        directories: tuple[WorkspaceDirectoryEntry, ...],
    ) -> None:
        """Apply directory metadata after descendants have been materialized."""

        for entry in sorted(
            directories,
            key=lambda candidate: candidate.relative_path.count("/"),
            reverse=True,
        ):
            path = _descendant(staging_root, entry.relative_path)
            if path.is_symlink() or not path.is_dir():
                raise WorkspaceMigrationError("Staging directory became aliased.")
            os.chmod(path, entry.mode, follow_symlinks=False)
            os.utime(path, ns=(entry.modified_ns, entry.modified_ns), follow_symlinks=False)

    @staticmethod
    def _require_write_preflight(context: _MigrationContext) -> None:
        """Reject collisions and inadequate space before any operation artifact."""

        if context.collisions:
            raise WorkspaceMigrationError("Destination rl_model_storage already exists.")
        if context.available_bytes < context.required_bytes:
            raise WorkspaceMigrationError("Destination does not have enough available space.")

    def _journal_payload(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        status: _JournalStatus,
    ) -> dict[str, Any]:
        """Build a complete resumable journal payload."""

        payload = self._base_artifact_payload(context, paths, _JOURNAL_TYPE)
        payload.update(
            {
                "status": status.value,
                "staging_storage": _relative_to_data_home(
                    paths.staging_storage,
                    context.destination_data_home,
                ),
                "receipt": _relative_to_data_home(
                    paths.receipt,
                    context.destination_data_home,
                ),
            }
        )
        return payload

    def _receipt_payload(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        status: WorkspaceMigrationStatus,
    ) -> dict[str, Any]:
        """Build the durable publication receipt."""

        payload = self._base_artifact_payload(context, paths, _RECEIPT_TYPE)
        payload.update(
            {
                "status": status.value,
                "journal": _relative_to_data_home(
                    paths.journal,
                    context.destination_data_home,
                ),
            }
        )
        return payload

    def _terminal_receipt_payload(
        self,
        context: _MigrationContext,
        paths: _ArtifactPaths,
        artifact_type: str,
        status: WorkspaceMigrationStatus,
    ) -> dict[str, Any]:
        """Build a rollback or cleanup audit receipt."""

        payload = self._base_artifact_payload(context, paths, artifact_type)
        payload["status"] = status.value
        payload["source_receipt"] = _relative_to_data_home(
            paths.receipt,
            context.destination_data_home,
        )
        return payload

    @staticmethod
    def _base_artifact_payload(
        context: _MigrationContext,
        paths: _ArtifactPaths,
        artifact_type: str,
    ) -> dict[str, Any]:
        """Build shared identity and inventory fields for an artifact."""

        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": artifact_type,
            "operation_id": paths.operation_id,
            "source_storage": str(context.source_storage),
            "destination_data_home": str(context.destination_data_home),
            "destination_storage": str(context.destination_storage),
            "inventory": context.inventory.to_mapping(),
        }

    @staticmethod
    def _result(
        context: _MigrationContext,
        operation_id: str,
        status: WorkspaceMigrationStatus,
        dry_run: bool,
        *,
        paths: _ArtifactPaths | None = None,
        recovery_path: Path | None = None,
    ) -> WorkspaceMigrationResult:
        """Project internal operation state into the public immutable result."""

        return WorkspaceMigrationResult(
            operation_id=operation_id,
            status=status,
            dry_run=dry_run,
            source_storage=str(context.source_storage),
            destination_data_home=str(context.destination_data_home),
            destination_storage=str(context.destination_storage),
            directory_count=context.inventory.directory_count,
            file_count=context.inventory.file_count,
            total_bytes=context.inventory.total_bytes,
            required_bytes=context.required_bytes,
            available_bytes=context.available_bytes,
            space_sufficient=context.available_bytes >= context.required_bytes,
            collisions=context.collisions,
            journal_path=str(paths.journal) if paths is not None else None,
            receipt_path=str(paths.receipt) if paths is not None else None,
            recovery_path=str(recovery_path) if recovery_path is not None else None,
        )

    def _lock(self, data_home: Path) -> StorageMutationLock:
        """Return the destination-owned cross-process command lock."""

        return StorageMutationLock(data_home, timeout_seconds=self._lock_timeout_seconds)

    @staticmethod
    def _inject(injector: MigrationFailureInjector | None, event: str) -> None:
        """Invoke a test-only failure boundary when supplied."""

        if injector is not None:
            injector(event)


def _required_capacity(inventory: WorkspaceInventory) -> int:
    """Include conservative per-entry filesystem overhead in the space preflight."""

    entry_count = inventory.file_count + inventory.directory_count + 1
    return inventory.total_bytes + entry_count * 4096


def _require_receipt_status(receipt: Mapping[str, Any]) -> WorkspaceMigrationStatus:
    """Return a valid durable receipt status."""

    try:
        status = WorkspaceMigrationStatus(str(receipt.get("status") or ""))
    except ValueError as exc:
        raise WorkspaceMigrationError("Migration receipt status is invalid.") from exc
    if status not in {
        WorkspaceMigrationStatus.MIGRATED,
        WorkspaceMigrationStatus.ROLLED_BACK,
        WorkspaceMigrationStatus.CLEANED,
    }:
        raise WorkspaceMigrationError("Migration receipt status is not durable.")
    return status


def _require_same_inventory(
    expected: WorkspaceInventory,
    observed: WorkspaceInventory,
    label: str,
) -> None:
    """Require exact content, layout, and recorded portable metadata."""

    if expected.to_mapping() != observed.to_mapping():
        raise WorkspaceMigrationError(f"{label.capitalize()} no longer matches the migration inventory.")


def _file_matches(path: Path, entry: WorkspaceFileEntry) -> bool:
    """Return whether one staged regular file already matches for resume."""

    if not os.path.lexists(path):
        return False
    try:
        path_stat = path.lstat()
    except OSError as exc:
        raise WorkspaceMigrationError("Staged file cannot be inspected.") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        raise WorkspaceMigrationError("Staged file is aliased or unsupported.")
    return path_stat.st_size == entry.byte_size and sha256_regular_file(path) == entry.sha256
