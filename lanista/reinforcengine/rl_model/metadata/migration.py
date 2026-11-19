# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Dry-run-first, backup-backed migration of model-storage metadata."""

from __future__ import annotations

import os
import shutil
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from lanista.reinforcengine.rl_model.metadata.inventory import (
    MetadataInventoryEntry,
    MetadataInventoryService,
    sha256_file,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    migrate_metadata_payload,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_yaml_file,
    read_yaml_mapping_file,
)
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction

BACKUP_DIRECTORY_NAME = ".metadata-migration-backups"
RECEIPT_FILE_PREFIX = ".metadata-migration-receipt-"
BACKUP_MANIFEST_FILE_NAME = "backup_manifest.yml"


class MetadataMigrationError(RuntimeError):
    """Raised after a failed migration or rollback has restored prior metadata."""


class MigrationStatus(StrEnum):
    """Terminal state of an explicit migration command."""

    DRY_RUN = "dry_run"
    MIGRATED = "migrated"
    NO_CHANGES = "no_changes"
    ROLLED_BACK = "rolled_back"


MigrationFailureInjector = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class MetadataChange:
    """Hash-backed receipt entry for one migrated metadata file."""

    relative_path: str
    kind: MetadataKind
    before_sha256: str
    after_sha256: str

    def to_mapping(self) -> dict[str, str]:
        """Convert this metadata change's value to mapping."""
        return {
            "relative_path": self.relative_path,
            "kind": self.kind.value,
            "before_sha256": self.before_sha256,
            "after_sha256": self.after_sha256,
        }


@dataclass(frozen=True, slots=True)
class MetadataMigrationResult:
    """In-memory result; write-mode paths point to backup and receipt artifacts."""

    operation_id: str
    status: MigrationStatus
    dry_run: bool
    inventory_count: int
    change_count: int
    changes: tuple[MetadataChange, ...]
    backup_path: str | None = None
    receipt_path: str | None = None

    def to_mapping(self) -> dict[str, Any]:
        """Convert this metadata migration result's value to mapping."""
        return {
            "operation_id": self.operation_id,
            "status": self.status.value,
            "dry_run": self.dry_run,
            "inventory_count": self.inventory_count,
            "change_count": self.change_count,
            "changes": [change.to_mapping() for change in self.changes],
            "backup_path": self.backup_path,
            "receipt_path": self.receipt_path,
        }


@dataclass(frozen=True, slots=True)
class _PreparedMetadata:
    """Describe prepared data."""

    inventory: MetadataInventoryEntry
    payload: dict[str, Any]
    migrated_payload: dict[str, Any]

    @property
    def needs_migration(self) -> bool:
        """Return the needs migration exposed by this prepared metadata."""
        return self.payload != self.migrated_payload


class MetadataSchemaMigrator:
    """Inventory, validate, back up, migrate, verify, receipt, and roll back."""

    def __init__(
        self,
        inventory_service: MetadataInventoryService | None = None,
        *,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the metadata schema migrator and its owned state."""
        self._inventory_service = inventory_service or MetadataInventoryService()
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    def inventory(self, root: str | os.PathLike[str]) -> tuple[MetadataInventoryEntry, ...]:
        """Return the read-only scoped inventory used by migration."""
        return self._inventory_service.inventory(root)

    def migrate(
        self,
        root: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        models_root: str | os.PathLike[str] | None = None,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> MetadataMigrationResult:
        """Migrate known metadata; dry-run is deliberately the default."""

        if dry_run:
            return self._migrate_guarded(
                root,
                dry_run=True,
                failure_injector=failure_injector,
            )
        scope = ModelsStorageMutationScope.for_migration_root(
            root,
            models_root=models_root,
        )
        with scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            return self._migrate_guarded(
                root,
                dry_run=False,
                failure_injector=failure_injector,
            )

    def _migrate_guarded(
        self,
        root: str | os.PathLike[str],
        *,
        dry_run: bool,
        failure_injector: MigrationFailureInjector | None,
    ) -> MetadataMigrationResult:
        """Prepare and apply one migration under the caller's write lock."""

        guard = StoragePathGuard(root)
        operation_id = uuid.uuid4().hex
        inventory = self.inventory(guard.root)
        self._inject(failure_injector, "after_inventory")
        prepared = tuple(self._prepare(entry) for entry in inventory)
        self._inject(failure_injector, "after_validate")
        pending = tuple(item for item in prepared if item.needs_migration)
        preview_changes = tuple(
            MetadataChange(
                relative_path=item.inventory.relative_path,
                kind=item.inventory.kind,
                before_sha256=item.inventory.sha256,
                after_sha256="",
            )
            for item in pending
        )
        if dry_run:
            return self._result(
                operation_id,
                MigrationStatus.DRY_RUN,
                True,
                inventory,
                preview_changes,
            )
        if not pending:
            return self._result(
                operation_id,
                MigrationStatus.NO_CHANGES,
                False,
                inventory,
                (),
            )

        backup_root: Path | None = None
        try:
            backup_root = self._create_backup(guard, operation_id, pending)
            self._inject(failure_injector, "after_backup")
        except Exception as exc:
            if backup_root is not None:
                with suppress(Exception):
                    self._remove_backup(backup_root)
            raise MetadataMigrationError(
                "Metadata migration failed before metadata mutation."
            ) from exc

        assert backup_root is not None
        try:
            return self._apply_migration(
                guard,
                operation_id,
                inventory,
                pending,
                backup_root,
                failure_injector,
            )
        except Exception as exc:
            raise MetadataMigrationError(
                "Metadata migration did not complete cleanly; recovery audit is required "
                "and the durable backup was retained."
            ) from exc

    def rollback(
        self,
        root: str | os.PathLike[str],
        migration_receipt_path: str | os.PathLike[str],
        *,
        dry_run: bool = True,
        models_root: str | os.PathLike[str] | None = None,
        failure_injector: MigrationFailureInjector | None = None,
    ) -> MetadataMigrationResult:
        """Restore exact backup bytes when current files still match the receipt."""

        if dry_run:
            return self._rollback_guarded(
                root,
                migration_receipt_path,
                dry_run=True,
                failure_injector=failure_injector,
            )
        scope = ModelsStorageMutationScope.for_migration_root(
            root,
            models_root=models_root,
        )
        with scope.lock(
            lock_factory=self._lock_factory,
            timeout_seconds=self._lock_timeout_seconds,
        ):
            return self._rollback_guarded(
                root,
                migration_receipt_path,
                dry_run=False,
                failure_injector=failure_injector,
            )

    def _rollback_guarded(
        self,
        root: str | os.PathLike[str],
        migration_receipt_path: str | os.PathLike[str],
        *,
        dry_run: bool,
        failure_injector: MigrationFailureInjector | None,
    ) -> MetadataMigrationResult:
        """Validate and restore a receipt under the caller's write lock."""

        guard = StoragePathGuard(root)
        operation_id = uuid.uuid4().hex
        receipt_path = guard.require_descendant(migration_receipt_path, must_exist=True)
        rollback_entries, backup_root = self._load_rollback_entries(guard, receipt_path)
        current_inventory = {entry.relative_path: entry.kind for entry in self.inventory(guard.root)}
        if any(current_inventory.get(entry.relative_path) is not entry.kind for entry in rollback_entries):
            raise MetadataMigrationError("Migration receipt references metadata outside the current inventory.")
        states = self._validate_rollback_hashes(rollback_entries, backup_root)
        if all(state == "before" for state in states):
            return self._result(
                operation_id,
                MigrationStatus.NO_CHANGES,
                dry_run,
                rollback_entries,
                (),
                backup_path=str(backup_root),
            )
        if dry_run:
            return self._result(
                operation_id,
                MigrationStatus.DRY_RUN,
                True,
                rollback_entries,
                tuple(self._rollback_change(entry) for entry in rollback_entries),
                backup_path=str(backup_root),
            )

        try:
            return self._apply_rollback(
                guard,
                operation_id,
                receipt_path,
                rollback_entries,
                backup_root,
                failure_injector,
            )
        except Exception as exc:
            raise MetadataMigrationError(
                "Metadata rollback did not complete cleanly; recovery audit is required."
            ) from exc

    def _apply_migration(
        self,
        guard: StoragePathGuard,
        operation_id: str,
        inventory: tuple[MetadataInventoryEntry, ...],
        pending: tuple[_PreparedMetadata, ...],
        backup_root: Path,
        failure_injector: MigrationFailureInjector | None,
    ) -> MetadataMigrationResult:
        """Apply the migration transition owned by this metadata schema migrator."""
        changes: list[MetadataChange] = []
        receipt_path = guard.resolve(f"{RECEIPT_FILE_PREFIX}{operation_id}.yml")
        with StorageTransaction(guard.root, operation="metadata-schema-migration") as transaction:
            for item in pending:
                change = self._migrate_one(transaction, item, failure_injector)
                changes.append(change)
            receipt_payload = self._migration_receipt_payload(
                operation_id,
                inventory_count=len(inventory),
                backup_relative_path=guard.relative_path(backup_root),
                changes=tuple(changes),
            )
            self._stage_yaml(transaction, receipt_path, receipt_payload)
            self._inject(failure_injector, "after_receipt")
            transaction.commit()
        return self._result(
            operation_id,
            MigrationStatus.MIGRATED,
            False,
            inventory,
            tuple(changes),
            backup_path=str(backup_root),
            receipt_path=str(receipt_path),
        )

    def _migrate_one(
        self,
        transaction: StorageTransaction,
        item: _PreparedMetadata,
        failure_injector: MigrationFailureInjector | None,
    ) -> MetadataChange:
        """Return the migrate one value owned by this metadata schema migrator."""
        target = item.inventory.path
        if sha256_file(target) != item.inventory.sha256:
            raise MetadataMigrationError("Metadata changed after validation; migration was refused.")
        transaction.quarantine(target)
        self._stage_yaml(transaction, target, item.migrated_payload)
        readback = read_yaml_mapping_file(str(target))
        validate_versioned_metadata_payload(item.inventory.kind, readback)
        if readback != item.migrated_payload:
            raise MetadataMigrationError("Metadata readback differs from the staged payload.")
        self._inject(failure_injector, f"after_readback:{item.inventory.relative_path}")
        return MetadataChange(
            relative_path=item.inventory.relative_path,
            kind=item.inventory.kind,
            before_sha256=item.inventory.sha256,
            after_sha256=sha256_file(target),
        )

    def _apply_rollback(
        self,
        guard: StoragePathGuard,
        operation_id: str,
        source_receipt_path: Path,
        entries: tuple[MetadataInventoryEntry, ...],
        backup_root: Path,
        failure_injector: MigrationFailureInjector | None,
    ) -> MetadataMigrationResult:
        """Apply the rollback transition owned by this metadata schema migrator."""
        changes = tuple(self._rollback_change(entry) for entry in entries)
        rollback_receipt_path = guard.resolve(f"{RECEIPT_FILE_PREFIX}{operation_id}-rollback.yml")
        with StorageTransaction(guard.root, operation="metadata-schema-rollback") as transaction:
            for entry in entries:
                target = entry.path
                backup_path = guard.resolve(guard.relative_path(backup_root), entry.relative_path, must_exist=True)
                if sha256_file(target) != entry.migrated_sha256:
                    raise MetadataMigrationError("Metadata changed during rollback; overwrite was refused.")
                transaction.quarantine(target)
                staged_path = transaction.stage_file(target)
                shutil.copy2(backup_path, staged_path)
                transaction.publish(staged_path, target)
                if sha256_file(target) != entry.sha256:
                    raise MetadataMigrationError("Restored metadata hash differs from its backup receipt.")
                self._inject(failure_injector, f"after_restore:{entry.relative_path}")
            rollback_payload = self._rollback_receipt_payload(
                operation_id,
                guard.relative_path(source_receipt_path),
                changes,
            )
            self._stage_yaml(transaction, rollback_receipt_path, rollback_payload)
            self._inject(failure_injector, "after_rollback_receipt")
            transaction.commit()
        return self._result(
            operation_id,
            MigrationStatus.ROLLED_BACK,
            False,
            entries,
            changes,
            backup_path=str(backup_root),
            receipt_path=str(rollback_receipt_path),
        )

    @staticmethod
    def _prepare(entry: MetadataInventoryEntry) -> _PreparedMetadata:
        """Return the prepare value owned by this metadata schema migrator."""
        try:
            payload = read_yaml_mapping_file(str(entry.path))
            migrated_payload = migrate_metadata_payload(entry.kind, payload)
        except Exception as exc:
            raise MetadataMigrationError(
                f"Metadata validation failed for {entry.relative_path!r}."
            ) from exc
        return _PreparedMetadata(entry, payload, migrated_payload)

    def _create_backup(
        self,
        guard: StoragePathGuard,
        operation_id: str,
        pending: tuple[_PreparedMetadata, ...],
    ) -> Path:
        """Create the backup owned by this metadata schema migrator."""
        backup_root = guard.resolve(BACKUP_DIRECTORY_NAME, operation_id)
        try:
            with StorageTransaction(
                guard.root,
                operation="metadata-schema-backup",
            ) as transaction:
                staged_backup_root = transaction.stage_directory(backup_root)
                for item in pending:
                    backup_path = staged_backup_root / item.inventory.relative_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item.inventory.path, backup_path)
                    if sha256_file(backup_path) != item.inventory.sha256:
                        raise MetadataMigrationError("Backup hash verification failed.")
                atomic_write_yaml_file(
                    str(staged_backup_root / BACKUP_MANIFEST_FILE_NAME),
                    self._backup_manifest_payload(operation_id, pending),
                    verify_reader=read_yaml_mapping_file,
                    smoke_check_reader=read_yaml_mapping_file,
                )
                transaction.confirm_path(staged_backup_root)
                transaction.publish(staged_backup_root, backup_root)
                transaction.confirm_path(backup_root)
                transaction.commit()
        except Exception:
            self._remove_backup(backup_root)
            raise
        return backup_root

    def _load_rollback_entries(
        self,
        guard: StoragePathGuard,
        receipt_path: Path,
    ) -> tuple[tuple[MetadataInventoryEntry, ...], Path]:
        """Load the rollback entries owned by this metadata schema migrator."""
        if not receipt_path.name.startswith(RECEIPT_FILE_PREFIX):
            raise MetadataMigrationError("Rollback receipt name is invalid.")
        receipt = read_yaml_mapping_file(str(receipt_path))
        if receipt.get("schema_version") != CURRENT_METADATA_SCHEMA_VERSION:
            raise MetadataMigrationError("Rollback receipt schema is unsupported.")
        if receipt.get("receipt_type") != "metadata_migration":
            raise MetadataMigrationError("Selected receipt is not a migration receipt.")
        migration_operation_id = str(receipt.get("operation_id") or "")
        if len(migration_operation_id) != 32 or any(
            character not in "0123456789abcdef" for character in migration_operation_id
        ):
            raise MetadataMigrationError("Migration receipt operation identifier is invalid.")
        if receipt_path.name != f"{RECEIPT_FILE_PREFIX}{migration_operation_id}.yml":
            raise MetadataMigrationError("Migration receipt name does not match its operation identifier.")
        backup_relative = str(receipt.get("backup_path") or "")
        if Path(backup_relative).parts != (BACKUP_DIRECTORY_NAME, migration_operation_id):
            raise MetadataMigrationError("Migration receipt backup path is invalid.")
        backup_root = guard.resolve(backup_relative, must_exist=True)
        if not backup_root.is_dir():
            raise MetadataMigrationError("Migration receipt backup path is not a directory.")
        raw_entries = receipt.get("changes")
        if not isinstance(raw_entries, list):
            raise MetadataMigrationError("Migration receipt changes are invalid.")
        entries = tuple(
            self._rollback_inventory_entry(guard, raw_entry)
            for raw_entry in raw_entries
            if isinstance(raw_entry, Mapping)
        )
        if len(entries) != len(raw_entries):
            raise MetadataMigrationError("Migration receipt contains an invalid change entry.")
        self._validate_backup_manifest(backup_root, migration_operation_id, entries)
        return entries, backup_root

    @staticmethod
    def _validate_backup_manifest(
        backup_root: Path,
        operation_id: str,
        entries: tuple[MetadataInventoryEntry, ...],
    ) -> None:
        """Validate the backup manifest required by this metadata schema migrator."""
        manifest = read_yaml_mapping_file(str(backup_root / BACKUP_MANIFEST_FILE_NAME))
        if manifest.get("schema_version") != CURRENT_METADATA_SCHEMA_VERSION:
            raise MetadataMigrationError("Migration backup manifest schema is unsupported.")
        if manifest.get("backup_type") != "metadata_migration":
            raise MetadataMigrationError("Migration backup manifest type is invalid.")
        if manifest.get("operation_id") != operation_id:
            raise MetadataMigrationError("Migration backup operation identifier is invalid.")
        raw_entries = manifest.get("entries")
        if not isinstance(raw_entries, list):
            raise MetadataMigrationError("Migration backup manifest entries are invalid.")
        expected_entries = [
            {
                "relative_path": entry.relative_path,
                "kind": entry.kind.value,
                "sha256": entry.sha256,
            }
            for entry in entries
        ]
        if raw_entries != expected_entries:
            raise MetadataMigrationError("Migration backup manifest does not match its receipt.")

    def _rollback_inventory_entry(
        self,
        guard: StoragePathGuard,
        raw_entry: Mapping[str, Any],
    ) -> MetadataInventoryEntry:
        """Roll back the inventory entry owned by this metadata schema migrator."""
        relative_path = str(raw_entry.get("relative_path") or "")
        target = guard.resolve(relative_path, must_exist=True)
        kind = MetadataKind(str(raw_entry.get("kind") or ""))
        if self._inventory_service.classify(target) is not kind:
            raise MetadataMigrationError("Migration receipt metadata kind does not match its path.")
        before_sha256 = self._validated_hash(raw_entry.get("before_sha256"))
        after_sha256 = self._validated_hash(raw_entry.get("after_sha256"))
        return MetadataInventoryEntry(
            path=target,
            relative_path=relative_path,
            kind=kind,
            byte_size=target.stat().st_size,
            sha256=before_sha256,
            migrated_sha256=after_sha256,
        )

    def _validate_rollback_hashes(
        self,
        entries: tuple[MetadataInventoryEntry, ...],
        backup_root: Path,
    ) -> tuple[str, ...]:
        """Validate the rollback hashes required by this metadata schema migrator."""
        guard = StoragePathGuard(backup_root.parent.parent)
        states: list[str] = []
        for entry in entries:
            backup_path = guard.resolve(guard.relative_path(backup_root), entry.relative_path, must_exist=True)
            if sha256_file(backup_path) != entry.sha256:
                raise MetadataMigrationError("Migration backup no longer matches its receipt.")
            current_hash = sha256_file(entry.path)
            if current_hash == entry.sha256:
                states.append("before")
            elif current_hash == entry.migrated_sha256:
                states.append("after")
            else:
                raise MetadataMigrationError("Metadata changed after migration; rollback was refused.")
        if states and len(set(states)) > 1:
            raise MetadataMigrationError("Metadata is only partially rolled back; automatic overwrite was refused.")
        return tuple(states)

    @staticmethod
    def _stage_yaml(
        transaction: StorageTransaction,
        target: Path,
        payload: Mapping[str, Any],
    ) -> None:
        """Stage the yaml owned by this metadata schema migrator."""
        staged_path = transaction.stage_file(target)
        atomic_write_yaml_file(
            str(staged_path),
            dict(payload),
            allow_unicode=True,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )
        transaction.publish(staged_path, target)

    @staticmethod
    def _result(
        operation_id: str,
        status: MigrationStatus,
        dry_run: bool,
        inventory: tuple[object, ...],
        changes: tuple[MetadataChange, ...],
        *,
        backup_path: str | None = None,
        receipt_path: str | None = None,
    ) -> MetadataMigrationResult:
        """Return the result value owned by this metadata schema migrator."""
        return MetadataMigrationResult(
            operation_id=operation_id,
            status=status,
            dry_run=dry_run,
            inventory_count=len(inventory),
            change_count=len(changes),
            changes=changes,
            backup_path=backup_path,
            receipt_path=receipt_path,
        )

    @staticmethod
    def _rollback_change(entry: MetadataInventoryEntry) -> MetadataChange:
        """Roll back the change owned by this metadata schema migrator."""
        return MetadataChange(
            relative_path=entry.relative_path,
            kind=entry.kind,
            before_sha256=entry.migrated_sha256,
            after_sha256=entry.sha256,
        )

    @staticmethod
    def _migration_receipt_payload(
        operation_id: str,
        *,
        inventory_count: int,
        backup_relative_path: str,
        changes: tuple[MetadataChange, ...],
    ) -> dict[str, Any]:
        """Return the migration receipt payload value owned by this metadata schema migrator."""
        return {
            "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
            "receipt_type": "metadata_migration",
            "operation_id": operation_id,
            "status": MigrationStatus.MIGRATED.value,
            "completed_at": utc_now_iso(),
            "inventory_count": inventory_count,
            "backup_path": backup_relative_path,
            "changes": [change.to_mapping() for change in changes],
        }

    @staticmethod
    def _rollback_receipt_payload(
        operation_id: str,
        source_receipt: str,
        changes: tuple[MetadataChange, ...],
    ) -> dict[str, Any]:
        """Roll back the receipt payload owned by this metadata schema migrator."""
        return {
            "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
            "receipt_type": "metadata_rollback",
            "operation_id": operation_id,
            "status": MigrationStatus.ROLLED_BACK.value,
            "completed_at": utc_now_iso(),
            "source_receipt": source_receipt,
            "changes": [change.to_mapping() for change in changes],
        }

    @staticmethod
    def _backup_manifest_payload(
        operation_id: str,
        pending: tuple[_PreparedMetadata, ...],
    ) -> dict[str, Any]:
        """Return the backup manifest payload value owned by this metadata schema migrator."""
        return {
            "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
            "backup_type": "metadata_migration",
            "operation_id": operation_id,
            "created_at": utc_now_iso(),
            "entries": [
                {
                    "relative_path": item.inventory.relative_path,
                    "kind": item.inventory.kind.value,
                    "sha256": item.inventory.sha256,
                }
                for item in pending
            ],
        }

    @staticmethod
    def _validated_hash(value: object) -> str:
        """Return the validated hash value owned by this metadata schema migrator."""
        candidate = str(value or "")
        if len(candidate) != 64 or any(character not in "0123456789abcdef" for character in candidate):
            raise MetadataMigrationError("Migration receipt contains an invalid SHA-256 hash.")
        return candidate

    @staticmethod
    def _remove_backup(backup_root: Path) -> None:
        """Remove the backup owned by this metadata schema migrator."""
        shutil.rmtree(backup_root, ignore_errors=True)
        with suppress(OSError):
            backup_root.parent.rmdir()

    @staticmethod
    def _inject(injector: MigrationFailureInjector | None, event: str) -> None:
        """Inject the operation owned by this metadata schema migrator."""
        if injector is not None:
            injector(event)
