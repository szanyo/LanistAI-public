# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Recoverable same-filesystem transactions for storage mutations."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import uuid
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_yaml_file,
    read_yaml_mapping_file,
)

TRANSACTION_DIRECTORY_NAME = ".storage-transactions"
JOURNAL_FILE_NAME = "journal.yml"


class StorageTransactionError(RuntimeError):
    """Raised when a recoverable storage mutation cannot be completed."""


class CrossDeviceStorageError(StorageTransactionError):
    """Raised before an atomic move that would cross filesystem boundaries."""


class StorageDurabilityError(StorageTransactionError):
    """Raised when a filesystem mutation cannot be made crash-durable."""


FailureInjector = Callable[[str], None]
DirectorySync = Callable[[Path], None]


def _fsync_directory(path: Path) -> None:
    """Flush one directory entry table, failing closed when unsupported."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    flags |= directory_flag
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise StorageDurabilityError(
            f"Cannot open directory for durable storage update: {path!s}."
        ) from exc

    sync_error: StorageDurabilityError | None = None
    sync_cause: OSError | None = None
    try:
        # O_DIRECTORY is not universal across POSIX implementations.  The
        # descriptor check keeps the fallback from flushing a regular file.
        if not directory_flag and not stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise StorageDurabilityError(f"Durability target is not a directory: {path!s}.")
        os.fsync(descriptor)
    except StorageDurabilityError as exc:
        sync_error = exc
    except OSError as exc:
        sync_error = StorageDurabilityError(f"Cannot make directory update durable: {path!s}.")
        sync_cause = exc

    try:
        os.close(descriptor)
    except OSError as exc:
        if sync_error is None:
            raise StorageDurabilityError(
                f"Cannot close durability descriptor for directory: {path!s}."
            ) from exc
    if sync_error is not None:
        raise sync_error from sync_cause


@dataclass(frozen=True, slots=True)
class StorageRecoveryReport:
    """Summary returned after replaying abandoned storage transaction journals."""

    rolled_back: tuple[str, ...]
    finalized: tuple[str, ...]
    failed: tuple[str, ...]


class StorageTransaction:
    """Journal reversible moves and make deletion a quarantine-then-commit flow."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
        transaction_id: str | None = None,
        failure_injector: FailureInjector | None = None,
        directory_sync: DirectorySync | None = None,
    ) -> None:
        """Initialize the storage transaction and its owned state."""
        self._guard = StoragePathGuard(root)
        self._guard.root.mkdir(parents=True, exist_ok=True)
        self._operation = str(operation or "storage-mutation")
        self._transaction_id = transaction_id or uuid.uuid4().hex
        if not self._transaction_id.isalnum():
            raise StorageTransactionError("Transaction identifiers must be alphanumeric.")
        self._failure_injector = failure_injector
        self._directory_sync = directory_sync or _fsync_directory
        self._actions: list[dict[str, str]] = []
        self._status = "active"
        self._transactions_root = self._guard.resolve(TRANSACTION_DIRECTORY_NAME)
        self._ensure_directory(self._transactions_root)
        self._workspace = self._guard.resolve(TRANSACTION_DIRECTORY_NAME, self._transaction_id)
        self._workspace.mkdir(parents=False, exist_ok=False)
        self._journal_path = self._workspace / JOURNAL_FILE_NAME
        self._write_journal()
        self._sync_directory(self._transactions_root)
        self._inject("after_begin")

    @property
    def transaction_id(self) -> str:
        """Return the transaction id exposed by this storage transaction."""
        return self._transaction_id

    @property
    def committed(self) -> bool:
        """Return the committed exposed by this storage transaction."""
        return self._status == "committed"

    @property
    def rolled_back(self) -> bool:
        """Return whether all journaled mutations were reversed successfully."""

        return self._status == "rolled_back"

    def __enter__(self) -> StorageTransaction:
        """Enter this storage transaction's managed context."""
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> Literal[False]:
        """Release owned resources when leaving this storage transaction's managed context."""
        if self._status == "active":
            self.rollback()
        elif self._status in {"committed", "rolled_back"}:
            self._cleanup_workspace()
        return False

    def quarantine(self, candidate: str | os.PathLike[str]) -> Path:
        """Atomically hide a path so rollback can restore it before commit."""
        self._require_active()
        source = self._guard.require_descendant(candidate, must_exist=True)
        self._assert_same_device(source, self._workspace)
        quarantine_root = self._workspace / "quarantine"
        self._ensure_directory(quarantine_root)
        destination = quarantine_root / f"{len(self._actions):04d}-{uuid.uuid4().hex}"
        self._record("quarantine", source=source, destination=destination)
        self._inject("before_quarantine")
        os.replace(source, destination)
        self._sync_rename_parents(source, destination)
        self._inject("after_quarantine")
        return destination

    def stage_directory(self, target: str | os.PathLike[str]) -> Path:
        """Create an empty same-filesystem directory for building a replacement."""
        self._require_active()
        validated_target = self._guard.require_descendant(target)
        target_parent = self._nearest_existing_parent(validated_target.parent)
        self._assert_same_device(target_parent, self._workspace)
        staging_root = self._workspace / "staging"
        self._ensure_directory(staging_root)
        staging_path = staging_root / f"{len(self._actions):04d}-{uuid.uuid4().hex}"
        self._record("staging", path=staging_path)
        self._inject("before_stage")
        staging_path.mkdir()
        self._sync_directory(staging_root)
        self._inject("after_stage")
        return staging_path

    def stage_file(self, target: str | os.PathLike[str]) -> Path:
        """Create an empty same-filesystem file for an atomic replacement."""
        self._require_active()
        validated_target = self._guard.require_descendant(target)
        target_parent = self._nearest_existing_parent(validated_target.parent)
        self._assert_same_device(target_parent, self._workspace)
        staging_root = self._workspace / "staging"
        self._ensure_directory(staging_root)
        staging_path = staging_root / f"{len(self._actions):04d}-{uuid.uuid4().hex}.file"
        self._record("staging", path=staging_path)
        self._inject("before_stage")
        staging_path.touch(exist_ok=False)
        self._sync_directory(staging_root)
        self._inject("after_stage")
        return staging_path

    def reserve_staging_path(
        self,
        target: str | os.PathLike[str],
    ) -> Path:
        """Reserve an absent staging path with the target's original basename."""

        self._require_active()
        validated_target = self._guard.require_descendant(target)
        target_parent = self._nearest_existing_parent(validated_target.parent)
        self._assert_same_device(target_parent, self._workspace)
        staging_root = self._workspace / "staging"
        self._ensure_directory(staging_root)
        container = staging_root / f"{len(self._actions):04d}-{uuid.uuid4().hex}"
        self._record("staging", path=container)
        self._inject("before_stage")
        container.mkdir()
        self._sync_directory(staging_root)
        staging_path = container / validated_target.name
        self._inject("after_stage")
        return staging_path

    def publish(self, staging_path: str | os.PathLike[str], target: str | os.PathLike[str]) -> Path:
        """Atomically publish a transaction-owned staged path at an absent target."""
        self._require_active()
        staging = self._guard.require_descendant(staging_path, must_exist=True)
        target_path = self._guard.require_descendant(target)
        try:
            staging.relative_to(self._workspace)
        except ValueError as exc:
            raise StorageTransactionError(
                "Only transaction-owned staging paths may be published."
            ) from exc
        if os.path.lexists(target_path):
            raise FileExistsError(target_path)
        self._ensure_directory(target_path.parent)
        self._assert_same_device(staging, target_path.parent)
        self._record("publish", source=staging, target=target_path)
        self._inject("before_publish")
        os.replace(staging, target_path)
        self._sync_rename_parents(staging, target_path)
        self._inject("after_publish")
        return target_path

    def move(self, source: str | os.PathLike[str], target: str | os.PathLike[str]) -> Path:
        """Atomically rename a descendant and journal the inverse move."""
        self._require_active()
        source_path = self._guard.require_descendant(source, must_exist=True)
        target_path = self._guard.require_descendant(target)
        if os.path.lexists(target_path):
            raise FileExistsError(target_path)
        self._ensure_directory(target_path.parent)
        self._assert_same_device(source_path, target_path.parent)
        self._record("move", source=source_path, target=target_path)
        self._inject("before_move")
        os.replace(source_path, target_path)
        self._sync_rename_parents(source_path, target_path)
        self._inject("after_move")
        return target_path

    def track_created(self, candidate: str | os.PathLike[str]) -> Path:
        """Register a not-yet-created path that rollback must remove."""
        self._require_active()
        path = self._guard.require_descendant(candidate)
        if os.path.lexists(path):
            raise FileExistsError(path)
        self._record("created", path=path)
        return path

    def snapshot_file(self, candidate: str | os.PathLike[str]) -> Path:
        """Preserve a regular file so rollback can restore its exact contents."""

        self._require_active()
        lexical_source = self._lexical_candidate(candidate)
        source = self._guard.require_descendant(
            lexical_source,
            must_exist=True,
        )
        if lexical_source != source or lexical_source.is_symlink() or not source.is_file():
            raise StorageTransactionError(
                "Only regular, non-aliased files can be transaction snapshots."
            )
        self._assert_same_device(source, self._workspace)
        snapshot_root = self._workspace / "snapshots"
        self._ensure_directory(snapshot_root)
        destination = snapshot_root / f"{len(self._actions):04d}-{uuid.uuid4().hex}.file"
        self._inject("before_snapshot_file")
        shutil.copy2(source, destination, follow_symlinks=False)
        self._sync_file(destination)
        self._sync_directory(snapshot_root)
        self._record("snapshot_file", source=source, destination=destination)
        self._inject("after_snapshot_file")
        return destination

    def confirm_path(self, candidate: str | os.PathLike[str]) -> Path:
        """Flush a transaction output before its commit marker becomes durable."""

        self._require_active()
        lexical_path = self._lexical_candidate(candidate)
        path = self._guard.require_descendant(
            lexical_path,
            must_exist=True,
        )
        if lexical_path != path or lexical_path.is_symlink():
            raise StorageTransactionError("Transaction outputs cannot use storage aliases.")
        if path.is_file():
            self._sync_file(path)
        elif path.is_dir():
            self._sync_tree(path)
        else:
            raise StorageTransactionError(
                f"Transaction output has an unsupported file type: {path!s}."
            )
        self._sync_directory(path.parent)
        return path

    def commit(self) -> None:
        """Persist the commit marker, then permanently purge quarantined data."""
        if self._status == "committed":
            self._cleanup_workspace()
            return
        self._require_active()
        self._inject("before_commit")
        self._status = "committed"
        try:
            self._write_journal()
        except Exception:
            # The mutation is not committed until its terminal marker is durable.
            self._status = "active"
            raise
        self._inject("after_commit_marker")
        self._cleanup_workspace()

    def rollback(self) -> None:
        """Replay inverse actions; repeated rollback and crash recovery are safe."""
        if self._status == "rolled_back":
            self._cleanup_workspace()
            return
        if self._status == "committed":
            self._cleanup_workspace()
            return
        self._inject("before_rollback")
        for action in reversed(self._actions):
            self._undo_action(action)
        self._status = "rolled_back"
        try:
            self._write_journal()
        except Exception:
            # Recovery may safely replay the already-idempotent inverse actions.
            self._status = "active"
            raise
        self._inject("after_rollback_marker")
        self._cleanup_workspace()

    @classmethod
    def recover(
        cls,
        root: str | os.PathLike[str],
        *,
        directory_sync: DirectorySync | None = None,
    ) -> StorageRecoveryReport:
        """Rollback active journals and finish cleanup for terminal journals."""
        guard = StoragePathGuard(root)
        sync_directory = directory_sync or _fsync_directory
        try:
            transactions_root = guard.resolve(TRANSACTION_DIRECTORY_NAME)
        except StoragePathViolation:
            return StorageRecoveryReport((), (), ())
        if not transactions_root.is_dir():
            # Re-flushing an existing model root is the safe retry when a prior
            # transaction-root removal completed before its parent fsync.
            if guard.root.is_dir() and not os.path.lexists(transactions_root):
                try:
                    sync_directory(guard.root)
                except StorageDurabilityError:
                    raise
                except OSError as exc:
                    raise StorageDurabilityError(
                        f"Cannot confirm transaction cleanup durability: {guard.root!s}."
                    ) from exc
            return StorageRecoveryReport((), (), ())

        rolled_back: list[str] = []
        finalized: list[str] = []
        failed: list[str] = []
        for workspace in sorted(transactions_root.iterdir()):
            if not workspace.is_dir():
                continue
            transaction_id = workspace.name
            try:
                outcome = cls._recover_workspace(
                    guard,
                    workspace,
                    transactions_root,
                    directory_sync=sync_directory,
                )
                if outcome == "rolled_back":
                    rolled_back.append(transaction_id)
                else:
                    finalized.append(transaction_id)
            except Exception:
                failed.append(transaction_id)
        cls._remove_empty_transactions_root(
            guard,
            transactions_root,
            sync_directory=sync_directory,
        )
        return StorageRecoveryReport(tuple(rolled_back), tuple(finalized), tuple(failed))

    @classmethod
    def _recover_workspace(
        cls,
        guard: StoragePathGuard,
        workspace: Path,
        transactions_root: Path,
        *,
        directory_sync: DirectorySync,
    ) -> Literal["rolled_back", "finalized"]:
        """Recover one validated transaction workspace without following aliases."""

        if workspace.is_symlink() or workspace.parent != transactions_root:
            raise StorageTransactionError(
                "Storage transaction workspace is unavailable or aliased."
            )
        journal_path = workspace / JOURNAL_FILE_NAME
        if not os.path.lexists(journal_path):
            if not cls._is_pre_journal_orphan(workspace):
                raise StorageTransactionError(
                    "Journal-less storage transaction contains unclassifiable data."
                )
            cls._remove_orphan_workspace(
                workspace,
                transactions_root,
                sync_directory=directory_sync,
            )
            return "finalized"
        if journal_path.is_symlink() or not journal_path.is_file():
            raise StorageTransactionError("Storage transaction journal is unavailable or aliased.")
        transaction = cls._from_journal(
            guard,
            workspace,
            directory_sync=directory_sync,
        )
        if transaction._status == "active":
            transaction.rollback()
            return "rolled_back"
        transaction._cleanup_workspace()
        return "finalized"

    @classmethod
    def _from_journal(
        cls,
        guard: StoragePathGuard,
        workspace: Path,
        *,
        directory_sync: DirectorySync,
    ) -> StorageTransaction:
        """Create this storage transaction's value from journal."""
        transactions_root = guard.resolve(TRANSACTION_DIRECTORY_NAME)
        guarded_workspace = guard.require_descendant(workspace, must_exist=True)
        if (
            guarded_workspace != workspace
            or workspace.is_symlink()
            or not workspace.is_dir()
            or workspace.parent != transactions_root
        ):
            raise StorageTransactionError(
                "Storage transaction workspace is unavailable or aliased."
            )
        payload = read_yaml_mapping_file(str(guarded_workspace / JOURNAL_FILE_NAME))
        if payload.get("schema_version") != 1:
            raise StorageTransactionError("Storage transaction journal schema is unsupported.")
        transaction_id = str(payload.get("transaction_id") or "")
        if transaction_id != workspace.name or not transaction_id.isalnum():
            raise StorageTransactionError(
                "Storage transaction journal identity does not match its workspace."
            )
        transaction = cls.__new__(cls)
        transaction._guard = guard
        transaction._operation = str(payload.get("operation") or "recovery")
        transaction._transaction_id = transaction_id
        transaction._failure_injector = None
        transaction._directory_sync = directory_sync
        raw_actions = payload.get("actions") or []
        if not isinstance(raw_actions, list) or not all(
            isinstance(action, dict) for action in raw_actions
        ):
            raise StorageTransactionError("Storage transaction journal actions are invalid.")
        transaction._actions = [dict(action) for action in raw_actions]
        transaction._status = str(payload.get("status") or "active")
        if transaction._status not in {"active", "committed", "rolled_back"}:
            raise StorageTransactionError("Storage transaction journal status is invalid.")
        transaction._transactions_root = transactions_root
        transaction._workspace = guarded_workspace
        transaction._journal_path = transaction._workspace / JOURNAL_FILE_NAME
        return transaction

    def _record(self, action_type: str, **paths: Path) -> None:
        """Record the operation owned by this storage transaction."""
        action = {"type": action_type}
        action.update({name: self._guard.relative_path(path) for name, path in paths.items()})
        self._actions.append(action)
        self._write_journal()

    def _undo_action(self, action: dict[str, str]) -> None:
        """Undo the action owned by this storage transaction."""
        action_type = action.get("type")
        if action_type == "created":
            self._remove_path(self._journal_path_for(action, "path"))
            return
        if action_type == "publish":
            self._remove_path(self._journal_path_for(action, "target"))
            return
        if action_type == "move":
            source = self._journal_path_for(action, "source")
            target = self._journal_path_for(action, "target")
            if os.path.lexists(target) and not os.path.lexists(source):
                self._ensure_directory(source.parent)
                os.replace(target, source)
                self._sync_rename_parents(target, source)
            return
        if action_type == "quarantine":
            source = self._journal_path_for(action, "source")
            destination = self._journal_path_for(action, "destination")
            if os.path.lexists(destination) and not os.path.lexists(source):
                self._ensure_directory(source.parent)
                os.replace(destination, source)
                self._sync_rename_parents(destination, source)
            return
        if action_type == "staging":
            self._remove_path(self._journal_path_for(action, "path"))
            return
        if action_type == "snapshot_file":
            source = self._journal_path_for(action, "source")
            destination = self._journal_path_for(action, "destination")
            if os.path.lexists(destination):
                self._remove_path(source)
                self._ensure_directory(source.parent)
                os.replace(destination, source)
                self._sync_rename_parents(destination, source)
            return
        raise StorageTransactionError(f"Unknown storage journal action: {action_type!r}")

    def _journal_path_for(self, action: dict[str, str], key: str) -> Path:
        """Return the journal path for value owned by this storage transaction."""
        relative_path = action.get(key)
        if not relative_path:
            raise StorageTransactionError(f"Storage journal action is missing {key!r}.")
        return self._guard.resolve(relative_path)

    def _write_journal(self) -> None:
        """Persist the journal owned by this storage transaction."""
        atomic_write_yaml_file(
            str(self._journal_path),
            {
                "schema_version": 1,
                "transaction_id": self._transaction_id,
                "operation": self._operation,
                "status": self._status,
                "actions": self._actions,
            },
            verify_reader=read_yaml_mapping_file,
        )
        self._sync_directory(self._journal_path.parent)

    def _cleanup_workspace(self) -> None:
        """Clean up the workspace owned by this storage transaction."""
        if self._workspace.is_symlink():
            raise StorageDurabilityError("Storage transaction workspace cannot be a storage alias.")
        if self._workspace.is_dir():
            for child in sorted(self._workspace.iterdir()):
                if child == self._journal_path:
                    continue
                self._remove_path(child)
            self._remove_path(self._journal_path)
            try:
                self._workspace.rmdir()
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise StorageDurabilityError(
                    f"Cannot remove empty storage transaction workspace: {self._workspace!s}."
                ) from exc
            self._sync_directory(self._transactions_root)
        elif os.path.lexists(self._workspace):
            raise StorageDurabilityError(
                "Storage transaction workspace has an unsupported file type."
            )
        try:
            self._transactions_root.rmdir()
        except FileNotFoundError:
            self._sync_directory(self._guard.root)
        except OSError as exc:
            if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                raise StorageDurabilityError(
                    f"Cannot remove storage transaction directory: {self._transactions_root!s}."
                ) from exc
        else:
            self._sync_directory(self._guard.root)

    def _require_active(self) -> None:
        """Return the required active for this storage transaction."""
        if self._status != "active":
            raise StorageTransactionError(f"Storage transaction is already {self._status}.")

    def _inject(self, event: str) -> None:
        """Inject the operation owned by this storage transaction."""
        if self._failure_injector is not None:
            self._failure_injector(event)

    def _ensure_directory(self, path: Path) -> None:
        """Create missing directory components and persist every parent entry."""

        missing: list[Path] = []
        current = path
        while not current.exists():
            missing.append(current)
            current = current.parent
        if not current.is_dir():
            raise NotADirectoryError(current)
        for directory in reversed(missing):
            try:
                directory.mkdir()
            except FileExistsError:
                if not directory.is_dir():
                    raise
            self._sync_directory(directory.parent)

    def _sync_directory(self, path: Path) -> None:
        """Synchronize the directory owned by this storage transaction."""
        try:
            self._directory_sync(path)
        except StorageDurabilityError:
            raise
        except OSError as exc:
            raise StorageDurabilityError(
                f"Cannot make directory update durable: {path!s}."
            ) from exc

    @staticmethod
    def _sync_file(path: Path) -> None:
        """Synchronize the file owned by this storage transaction."""
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(path, flags)
        except OSError as exc:
            raise StorageDurabilityError(
                f"Cannot open file for durable storage update: {path!s}."
            ) from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise StorageDurabilityError(f"Cannot make file update durable: {path!s}.") from exc
        finally:
            try:
                os.close(descriptor)
            except OSError as exc:
                raise StorageDurabilityError(
                    f"Cannot close durability descriptor for file: {path!s}."
                ) from exc

    def _sync_tree(self, directory: Path) -> None:
        """Synchronize the tree owned by this storage transaction."""
        with os.scandir(directory) as entries:
            sorted_entries = sorted(entries, key=lambda item: item.name)
        for entry in sorted_entries:
            entry_path = Path(entry.path)
            if entry.is_symlink():
                raise StorageTransactionError(
                    "Transaction output trees cannot contain storage aliases."
                )
            if entry.is_dir(follow_symlinks=False):
                self._sync_tree(entry_path)
                continue
            if entry.is_file(follow_symlinks=False):
                self._sync_file(entry_path)
                continue
            raise StorageTransactionError("Transaction output trees cannot contain special files.")
        self._sync_directory(directory)

    def _sync_rename_parents(self, source: Path, target: Path) -> None:
        """Persist a rename destination first, then the distinct source parent."""

        target_parent = target.parent
        self._sync_directory(target_parent)
        if source.parent != target_parent:
            self._sync_directory(source.parent)

    def _lexical_candidate(
        self,
        candidate: str | os.PathLike[str],
    ) -> Path:
        """Return the lexical candidate value owned by this storage transaction."""
        raw_candidate = Path(candidate).expanduser()
        if not raw_candidate.is_absolute():
            raw_candidate = self._guard.root / raw_candidate
        return Path(os.path.abspath(raw_candidate))

    @staticmethod
    def _nearest_existing_parent(path: Path) -> Path:
        """Return the nearest existing parent value owned by this storage transaction."""
        current = path
        while not current.exists() and current != current.parent:
            current = current.parent
        return current

    @staticmethod
    def _assert_same_device(first: Path, second: Path) -> None:
        """Assert the same device owned by this storage transaction."""
        first_device = os.stat(first, follow_symlinks=False).st_dev
        second_device = os.stat(second, follow_symlinks=False).st_dev
        if first_device != second_device:
            raise CrossDeviceStorageError(
                f"Atomic storage mutation would cross filesystems: {first!s} -> {second!s}."
            )

    def _remove_path(self, path: Path) -> None:
        """Remove the path owned by this storage transaction."""
        if path.is_symlink() or path.is_file():
            with suppress(FileNotFoundError):
                path.unlink()
        elif path.is_dir():
            try:
                shutil.rmtree(path)
            except OSError as exc:
                raise StorageDurabilityError(
                    f"Cannot remove transaction-owned storage path: {path!s}."
                ) from exc
        self._sync_directory(self._nearest_existing_parent(path.parent))

    @staticmethod
    def _remove_empty_transactions_root(
        guard: StoragePathGuard,
        transactions_root: Path,
        *,
        sync_directory: DirectorySync,
    ) -> None:
        """Remove the empty transactions root owned by this storage transaction."""
        try:
            transactions_root.rmdir()
        except FileNotFoundError:
            return
        except OSError as exc:
            if exc.errno in {errno.ENOTEMPTY, errno.EEXIST}:
                return
            raise StorageDurabilityError(
                f"Cannot remove empty storage transaction directory: {transactions_root!s}."
            ) from exc
        try:
            sync_directory(guard.root)
        except StorageDurabilityError:
            raise
        except OSError as exc:
            raise StorageDurabilityError(
                f"Cannot make transaction-directory cleanup durable: {guard.root!s}."
            ) from exc

    @staticmethod
    def _is_pre_journal_orphan(workspace: Path) -> bool:
        """Return whether pre journal orphan applies to this storage transaction."""
        try:
            entries = tuple(workspace.iterdir())
        except OSError:
            return False
        return all(
            entry.is_file()
            and not entry.is_symlink()
            and entry.name.startswith(f".{JOURNAL_FILE_NAME}.")
            and entry.name.endswith(".tmp")
            for entry in entries
        )

    @staticmethod
    def _remove_orphan_workspace(
        workspace: Path,
        transactions_root: Path,
        *,
        sync_directory: DirectorySync,
    ) -> None:
        """Remove the orphan workspace owned by this storage transaction."""
        try:
            shutil.rmtree(workspace)
        except OSError as exc:
            raise StorageDurabilityError(
                f"Cannot remove pre-journal transaction orphan: {workspace!s}."
            ) from exc
        try:
            sync_directory(transactions_root)
        except StorageDurabilityError:
            raise
        except OSError as exc:
            raise StorageDurabilityError(
                f"Cannot make pre-journal transaction cleanup durable: {transactions_root!s}."
            ) from exc
