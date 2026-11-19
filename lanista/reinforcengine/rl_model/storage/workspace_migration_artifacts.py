# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Confine and durably persist external-workspace migration artifacts."""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from lanista.reinforcengine.rl_model.storage.workspace_inventory import (
    validate_relative_workspace_path,
)

WORKSPACE_DIRECTORY_NAME: Final = "rl_model_storage"
CONTROL_DIRECTORY_NAME: Final = ".workspace-migrations"
RECEIPT_DIRECTORY_NAME: Final = ".workspace-migration-receipts"
JOURNAL_FILE_NAME: Final = "journal.json"
ARTIFACT_SCHEMA_VERSION: Final = 1
_HEX_CHARACTERS: Final = frozenset("0123456789abcdef")


class WorkspaceMigrationError(RuntimeError):
    """Report a failed-closed workspace migration operation."""


@dataclass(frozen=True, slots=True)
class MigrationArtifactPaths:
    """Own deterministic journal, staging, receipt, and recovery paths."""

    operation_id: str
    operation_root: Path
    journal: Path
    staging_storage: Path
    receipt: Path
    rollback_receipt: Path
    cleanup_receipt: Path
    recovery_storage: Path


@dataclass(frozen=True, slots=True)
class _EntryIdentity:
    """Fence one validated filesystem entry against replacement."""

    device: int
    inode: int
    file_type: int
    mode: int
    byte_size: int
    modified_ns: int


@dataclass(frozen=True, slots=True)
class _ValidatedEntry:
    """Retain one validated entry identity and its directory children."""

    identity: _EntryIdentity
    children: Mapping[str, _ValidatedEntry]


@dataclass(frozen=True, slots=True)
class _DirectoryAnchor:
    """Bind an open directory descriptor to its parent directory entry."""

    parent_descriptor: int
    name: str
    descriptor: int
    identity: _EntryIdentity


def resolve_source_storage(raw_source: str | os.PathLike[str]) -> Path:
    """Resolve one explicit absolute legacy workspace without accepting aliases."""

    source = Path(raw_source).expanduser()
    if not source.is_absolute():
        raise WorkspaceMigrationError("Source storage must be an absolute path.")
    require_no_alias_components(source)
    try:
        source_stat = source.lstat()
    except OSError as exc:
        raise WorkspaceMigrationError("Source workspace is unavailable.") from exc
    if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
        raise WorkspaceMigrationError("Source workspace must be a real directory.")
    canonical = source.resolve(strict=True)
    if canonical.name != WORKSPACE_DIRECTORY_NAME:
        raise WorkspaceMigrationError("Source must name the legacy rl_model_storage workspace.")
    return canonical


def resolve_data_home(
    raw_data_home: str | os.PathLike[str],
    *,
    must_exist: bool,
) -> Path:
    """Resolve an explicit absolute data home without reading global configuration."""

    data_home = Path(raw_data_home).expanduser()
    if not data_home.is_absolute():
        raise WorkspaceMigrationError("Destination data home must be an absolute path.")
    require_no_alias_components(data_home)
    if os.path.lexists(data_home):
        canonical = _resolve_existing_data_home(data_home)
    else:
        if must_exist:
            raise WorkspaceMigrationError("Destination data home is unavailable.")
        canonical = data_home.resolve(strict=False)
    if canonical == Path(canonical.anchor):
        raise WorkspaceMigrationError("A filesystem root is too broad for destination data.")
    return canonical


def prepare_data_home(data_home: Path) -> None:
    """Create only the explicitly selected canonical data-home directory."""

    mkdir_real(data_home, parents=True)
    if data_home.resolve(strict=True) != data_home:
        raise WorkspaceMigrationError("Destination data home changed identity during creation.")


def require_disjoint_roots(source: Path, data_home: Path, target: Path) -> None:
    """Prevent migration controls or publication from entering the source tree."""

    if data_home == source or data_home.is_relative_to(source):
        raise WorkspaceMigrationError("Destination data home must stay outside the source workspace.")
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise WorkspaceMigrationError("Source and destination workspaces overlap.")


def existing_ancestor(path: Path) -> Path:
    """Return the closest existing real directory for a disk-space query."""

    candidate = path
    while not os.path.lexists(candidate):
        if candidate == candidate.parent:
            raise WorkspaceMigrationError("Destination has no inspectable filesystem ancestor.")
        candidate = candidate.parent
    try:
        path_stat = candidate.lstat()
    except OSError as exc:
        raise WorkspaceMigrationError("Destination filesystem cannot be inspected.") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise WorkspaceMigrationError("Destination ancestor is aliased or is not a directory.")
    return candidate


def artifact_paths(data_home: Path, operation_id: str) -> MigrationArtifactPaths:
    """Return fixed artifact paths for one validated operation identifier."""

    operation_id = require_operation_id(operation_id)
    operation_root = data_home / CONTROL_DIRECTORY_NAME / operation_id
    receipt_root = data_home / RECEIPT_DIRECTORY_NAME
    return MigrationArtifactPaths(
        operation_id=operation_id,
        operation_root=operation_root,
        journal=operation_root / JOURNAL_FILE_NAME,
        staging_storage=operation_root / "staging" / WORKSPACE_DIRECTORY_NAME,
        receipt=receipt_root / f"{operation_id}.json",
        rollback_receipt=receipt_root / f"{operation_id}-rollback.json",
        cleanup_receipt=receipt_root / f"{operation_id}-cleanup.json",
        recovery_storage=operation_root / "rollback-backup" / WORKSPACE_DIRECTORY_NAME,
    )


def require_operation_id(raw_operation_id: object) -> str:
    """Return a canonical UUID-hex operation identifier."""

    if (
        not isinstance(raw_operation_id, str)
        or len(raw_operation_id) != 32
        or any(character not in _HEX_CHARACTERS for character in raw_operation_id)
    ):
        raise WorkspaceMigrationError("Migration operation identifier is invalid.")
    return raw_operation_id


def require_artifact_header(payload: Mapping[str, Any], artifact_type: str) -> None:
    """Require the exact supported schema and artifact type."""

    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise WorkspaceMigrationError("Workspace migration artifact schema is unsupported.")
    if payload.get("artifact_type") != artifact_type:
        raise WorkspaceMigrationError("Selected workspace migration artifact has the wrong type.")


def require_artifact_roots(payload: Mapping[str, Any], source: Path, data_home: Path) -> None:
    """Require artifact roots to match the operator's explicit canonical arguments."""

    expected = {
        "source_storage": str(source),
        "destination_data_home": str(data_home),
        "destination_storage": str(data_home / WORKSPACE_DIRECTORY_NAME),
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise WorkspaceMigrationError("Migration artifact belongs to different filesystem roots.")


def descendant(root: Path, relative_path: str) -> Path:
    """Resolve one validated and non-aliased artifact path under its fixed root."""

    parts = validate_relative_workspace_path(relative_path)
    candidate = root.joinpath(*parts)
    try:
        candidate.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defended by the parser above
        raise WorkspaceMigrationError("Workspace path escaped its fixed root.") from exc
    require_no_alias_descendant(root, candidate)
    return candidate


def require_no_alias_components(path: Path) -> None:
    """Reject symbolic links in every existing component of an explicit path."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if not os.path.lexists(current):
            return
        try:
            component_stat = current.lstat()
        except OSError as exc:
            raise WorkspaceMigrationError("Explicit path component cannot be inspected.") from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise WorkspaceMigrationError("Explicit filesystem paths must not contain aliases.")


def require_no_alias_descendant(root: Path, candidate: Path) -> None:
    """Reject aliases in each existing descendant component below a private root."""

    try:
        relative_parts = candidate.relative_to(root).parts
    except ValueError as exc:
        raise WorkspaceMigrationError("Migration descendant escaped its private root.") from exc
    current = root
    for component in relative_parts:
        current /= component
        if not os.path.lexists(current):
            return
        try:
            component_stat = current.lstat()
        except OSError as exc:
            raise WorkspaceMigrationError("Migration descendant cannot be inspected.") from exc
        if stat.S_ISLNK(component_stat.st_mode):
            raise WorkspaceMigrationError("Migration descendant contains a symbolic link.")


def mkdir_real(path: Path, *, parents: bool) -> None:
    """Create a directory and reject any existing final-path alias."""

    try:
        path.mkdir(parents=parents, exist_ok=True)
        path_stat = path.lstat()
    except OSError as exc:
        raise WorkspaceMigrationError(f"Migration directory is unavailable: {path!s}.") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise WorkspaceMigrationError("Migration directory is aliased or unsupported.")


def relative_to_data_home(path: Path, data_home: Path) -> str:
    """Return a canonical data-home-relative artifact path."""

    try:
        relative = path.relative_to(data_home).as_posix()
    except ValueError as exc:
        raise WorkspaceMigrationError("Migration artifact escaped its data home.") from exc
    validate_relative_workspace_path(relative)
    return relative


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist and read back one fsynced same-directory JSON replacement."""

    mkdir_real(path.parent, parents=True)
    if os.path.lexists(path) and path.is_symlink():
        raise WorkspaceMigrationError("Migration artifact target is a symbolic link.")
    temporary_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary_path, flags, 0o600)
    try:
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        read_json_mapping(temporary_path)
        os.replace(temporary_path, path)
        sync_directory(path.parent)
        read_json_mapping(path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if os.path.lexists(temporary_path):
            temporary_path.unlink()


def read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a regular no-follow JSON artifact and require a mapping."""

    if not path.is_absolute():
        raise WorkspaceMigrationError("Migration artifact path must be absolute.")
    if not hasattr(os, "O_NOFOLLOW"):
        raise WorkspaceMigrationError("Safe no-follow artifact access is unsupported.")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise WorkspaceMigrationError("Migration artifact is unavailable or aliased.") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorkspaceMigrationError("Migration artifact is not a regular file.")
        stream = os.fdopen(descriptor, encoding="utf-8")
        descriptor = -1
        with stream:
            payload = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise WorkspaceMigrationError("Migration artifact is unreadable or corrupt.") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise WorkspaceMigrationError("Migration artifact payload is not a string-keyed mapping.")
    return payload


def sync_directory(path: Path) -> None:
    """Fsync a real directory after an artifact or tree publication."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise WorkspaceMigrationError("Safe directory synchronization is unsupported.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def remove_tree_no_follow(root: Path) -> None:
    """Delete a tree through inode-fenced no-follow directory descriptors."""

    anchors, descriptors = _open_directory_chain(root)
    try:
        root_descriptor = anchors[-1].descriptor
        validated_root = _snapshot_anchored_tree(root_descriptor, anchors)
        _require_anchor_chain(anchors)
        _require_snapshot_identity(os.fstat(root_descriptor), validated_root.identity)
        _delete_anchored_tree(root_descriptor, anchors, validated_root)
        _require_anchor_chain(anchors)
        root_anchor = anchors[-1]
        os.rmdir(root_anchor.name, dir_fd=root_anchor.parent_descriptor)
        os.fsync(root_anchor.parent_descriptor)
    except WorkspaceMigrationError:
        raise
    except OSError as exc:
        raise WorkspaceMigrationError("Descriptor-anchored cleanup failed closed.") from exc
    finally:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)


def _open_directory_chain(root: Path) -> tuple[tuple[_DirectoryAnchor, ...], tuple[int, ...]]:
    """Open every absolute directory component and retain its parent binding."""

    _require_descriptor_cleanup_support()
    if not root.is_absolute() or root == Path(root.anchor) or ".." in root.parts:
        raise WorkspaceMigrationError("Cleanup root must be an absolute non-root directory.")
    require_no_alias_components(root)
    descriptors: list[int] = []
    anchors: list[_DirectoryAnchor] = []
    try:
        parent_descriptor = _open_directory_descriptor(root.anchor)
        descriptors.append(parent_descriptor)
        for component in root.parts[1:]:
            anchor = _open_child_anchor(
                parent_descriptor,
                component,
                require_same_device=False,
            )
            anchors.append(anchor)
            descriptors.append(anchor.descriptor)
            parent_descriptor = anchor.descriptor
    except BaseException:
        for descriptor in reversed(descriptors):
            with suppress(OSError):
                os.close(descriptor)
        raise
    if not anchors:
        raise WorkspaceMigrationError("Cleanup root has no removable directory component.")
    return tuple(anchors), tuple(descriptors)


def _snapshot_anchored_tree(
    directory_descriptor: int,
    anchors: tuple[_DirectoryAnchor, ...],
) -> _ValidatedEntry:
    """Create the single pre-deletion tree snapshot used by cleanup."""

    return _validate_anchored_tree(directory_descriptor, anchors)


def _validate_anchored_tree(
    directory_descriptor: int,
    anchors: tuple[_DirectoryAnchor, ...],
) -> _ValidatedEntry:
    """Snapshot a complete regular-file tree through anchored descriptors."""

    _require_anchor_chain(anchors)
    root_stat = os.fstat(directory_descriptor)
    children: dict[str, _ValidatedEntry] = {}
    for name in _directory_entry_names(directory_descriptor):
        _require_anchor_chain(anchors)
        entry_stat = _entry_stat(directory_descriptor, name)
        if stat.S_ISLNK(entry_stat.st_mode):
            raise WorkspaceMigrationError("Cleanup refused a symbolic-link artifact.")
        if stat.S_ISDIR(entry_stat.st_mode):
            child_anchor = _open_child_anchor(
                directory_descriptor,
                name,
                require_same_device=True,
            )
            try:
                children[name] = _validate_anchored_tree(
                    child_anchor.descriptor,
                    (*anchors, child_anchor),
                )
            finally:
                os.close(child_anchor.descriptor)
        elif stat.S_ISREG(entry_stat.st_mode):
            children[name] = _ValidatedEntry(_entry_identity(entry_stat), {})
        else:
            raise WorkspaceMigrationError("Cleanup found an unsupported artifact type.")
    _require_anchor_chain(anchors)
    return _ValidatedEntry(_entry_identity(root_stat), children)


def _delete_anchored_tree(
    directory_descriptor: int,
    anchors: tuple[_DirectoryAnchor, ...],
    validated_directory: _ValidatedEntry,
) -> None:
    """Delete only entries that retain their validated identities and names."""

    _require_anchor_chain(anchors)
    names = _directory_entry_names(directory_descriptor)
    if set(names) != set(validated_directory.children):
        raise WorkspaceMigrationError("Cleanup directory contents changed after validation.")
    for name in names:
        expected = validated_directory.children[name]
        _require_anchor_chain(anchors)
        entry_stat = _entry_stat(directory_descriptor, name)
        _require_snapshot_identity(entry_stat, expected.identity)
        if stat.S_ISDIR(entry_stat.st_mode):
            _delete_child_directory(directory_descriptor, anchors, name, expected)
        elif stat.S_ISREG(entry_stat.st_mode):
            _require_anchor_chain(anchors)
            _require_snapshot_identity(_entry_stat(directory_descriptor, name), expected.identity)
            os.unlink(name, dir_fd=directory_descriptor)
        else:  # pragma: no cover - snapshot validation already restricts the type
            raise WorkspaceMigrationError("Cleanup entry type changed after validation.")


def _delete_child_directory(
    parent_descriptor: int,
    anchors: tuple[_DirectoryAnchor, ...],
    name: str,
    expected: _ValidatedEntry,
) -> None:
    """Delete one validated directory through its still-bound parent entry."""

    child_anchor = _open_child_anchor(
        parent_descriptor,
        name,
        require_same_device=True,
    )
    child_anchors = (*anchors, child_anchor)
    try:
        _require_snapshot_identity(os.fstat(child_anchor.descriptor), expected.identity)
        _delete_anchored_tree(child_anchor.descriptor, child_anchors, expected)
        _require_anchor_chain(child_anchors)
        os.rmdir(name, dir_fd=parent_descriptor)
    finally:
        os.close(child_anchor.descriptor)


def _open_child_anchor(
    parent_descriptor: int,
    name: str,
    *,
    require_same_device: bool,
) -> _DirectoryAnchor:
    """Open one child directory and bind it to its observed parent entry."""

    expected_stat = _entry_stat(parent_descriptor, name)
    if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISDIR(expected_stat.st_mode):
        raise WorkspaceMigrationError("Cleanup directory entry is aliased or unsupported.")
    parent_stat = os.fstat(parent_descriptor)
    if require_same_device and expected_stat.st_dev != parent_stat.st_dev:
        raise WorkspaceMigrationError("Cleanup refuses cross-device directory traversal.")
    descriptor = _open_directory_descriptor(name, parent_descriptor=parent_descriptor)
    identity = _entry_identity(expected_stat)
    try:
        _require_anchor_identity(os.fstat(descriptor), identity)
    except BaseException:
        os.close(descriptor)
        raise
    return _DirectoryAnchor(parent_descriptor, name, descriptor, identity)


def _open_directory_descriptor(
    path: str | os.PathLike[str],
    *,
    parent_descriptor: int | None = None,
) -> int:
    """Open one directory with mandatory no-follow flags."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        raise WorkspaceMigrationError("Cleanup directory could not be anchored safely.") from exc


def _require_anchor_chain(anchors: tuple[_DirectoryAnchor, ...]) -> None:
    """Revalidate every retained parent/name/descriptor identity binding."""

    for anchor in anchors:
        try:
            linked_stat = os.stat(
                anchor.name,
                dir_fd=anchor.parent_descriptor,
                follow_symlinks=False,
            )
            descriptor_stat = os.fstat(anchor.descriptor)
        except OSError as exc:
            raise WorkspaceMigrationError("Cleanup directory binding changed.") from exc
        _require_anchor_identity(linked_stat, anchor.identity)
        _require_anchor_identity(descriptor_stat, anchor.identity)


def _require_anchor_identity(observed: os.stat_result, expected: _EntryIdentity) -> None:
    """Require an observed directory to retain its inode and file type."""

    if (
        observed.st_dev != expected.device
        or observed.st_ino != expected.inode
        or stat.S_IFMT(observed.st_mode) != expected.file_type
        or not stat.S_ISDIR(observed.st_mode)
    ):
        raise WorkspaceMigrationError("Cleanup directory identity changed.")


def _require_snapshot_identity(observed: os.stat_result, expected: _EntryIdentity) -> None:
    """Require an entry to retain its full validation snapshot."""

    observed_identity = _entry_identity(observed)
    if observed_identity != expected:
        raise WorkspaceMigrationError("Cleanup entry changed after validation.")


def _entry_identity(entry_stat: os.stat_result) -> _EntryIdentity:
    """Project stable and mutation-sensitive stat fields into one fence."""

    return _EntryIdentity(
        device=entry_stat.st_dev,
        inode=entry_stat.st_ino,
        file_type=stat.S_IFMT(entry_stat.st_mode),
        mode=stat.S_IMODE(entry_stat.st_mode),
        byte_size=entry_stat.st_size,
        modified_ns=entry_stat.st_mtime_ns,
    )


def _entry_stat(directory_descriptor: int, name: str) -> os.stat_result:
    """Stat one directory entry without following it."""

    try:
        return os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
    except OSError as exc:
        raise WorkspaceMigrationError("Cleanup entry could not be inspected safely.") from exc


def _directory_entry_names(directory_descriptor: int) -> tuple[str, ...]:
    """Return deterministic names from one anchored directory descriptor."""

    try:
        with os.scandir(directory_descriptor) as iterator:
            return tuple(sorted(entry.name for entry in iterator))
    except OSError as exc:
        raise WorkspaceMigrationError("Cleanup directory could not be enumerated safely.") from exc


def _require_descriptor_cleanup_support() -> None:
    """Fail closed when required POSIX descriptor operations are unavailable."""

    supported = (
        hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and os.open in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.rmdir in os.supports_dir_fd
        and os.scandir in os.supports_fd
    )
    if not supported:
        raise WorkspaceMigrationError("Descriptor-anchored cleanup is unsupported on this platform.")


def _resolve_existing_data_home(data_home: Path) -> Path:
    """Validate and canonicalize one existing explicit data home."""

    try:
        path_stat = data_home.lstat()
    except OSError as exc:
        raise WorkspaceMigrationError("Destination data home cannot be inspected.") from exc
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISDIR(path_stat.st_mode):
        raise WorkspaceMigrationError("Destination data home must be a real directory.")
    return data_home.resolve(strict=True)
