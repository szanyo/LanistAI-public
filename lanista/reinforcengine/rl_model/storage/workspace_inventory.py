# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Inventory and copy external workspaces without following symbolic links."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_COPY_BUFFER_SIZE = 1024 * 1024


class WorkspaceInventoryError(RuntimeError):
    """Reject an aliased, unstable, or unsupported workspace entry."""


@dataclass(frozen=True, slots=True)
class WorkspaceDirectoryEntry:
    """Describe one root-relative directory and its portable metadata."""

    relative_path: str
    mode: int
    modified_ns: int

    def to_mapping(self) -> dict[str, int | str]:
        """Return the durable representation used by migration artifacts."""

        return {
            "relative_path": self.relative_path,
            "mode": self.mode,
            "modified_ns": self.modified_ns,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceFileEntry:
    """Describe one immutable inventory observation of a regular file."""

    relative_path: str
    byte_size: int
    sha256: str
    mode: int
    modified_ns: int

    def to_mapping(self) -> dict[str, int | str]:
        """Return the durable representation used by migration artifacts."""

        return {
            "relative_path": self.relative_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "mode": self.mode,
            "modified_ns": self.modified_ns,
        }


@dataclass(frozen=True, slots=True)
class WorkspaceInventory:
    """Hold a deterministic, content-addressed external-workspace inventory."""

    directories: tuple[WorkspaceDirectoryEntry, ...]
    files: tuple[WorkspaceFileEntry, ...]
    total_bytes: int
    fingerprint: str

    @property
    def directory_count(self) -> int:
        """Return the number of descendant directories, excluding the root."""

        return len(self.directories)

    @property
    def file_count(self) -> int:
        """Return the number of regular files in the workspace."""

        return len(self.files)

    def to_mapping(self) -> dict[str, Any]:
        """Return the strict serializable representation of this inventory."""

        return {
            "directory_count": self.directory_count,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "fingerprint": self.fingerprint,
            "directories": [entry.to_mapping() for entry in self.directories],
            "files": [entry.to_mapping() for entry in self.files],
        }


def validate_relative_workspace_path(raw_path: str) -> tuple[str, ...]:
    """Validate one artifact path and return its POSIX components."""

    if not raw_path or "\\" in raw_path:
        raise WorkspaceInventoryError("Workspace artifact path is invalid.")
    candidate = PurePosixPath(raw_path)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise WorkspaceInventoryError("Workspace artifact path is not root-confined.")
    if candidate.as_posix() != raw_path:
        raise WorkspaceInventoryError("Workspace artifact path is not canonical.")
    return candidate.parts


def inventory_from_mapping(raw_inventory: object) -> WorkspaceInventory:
    """Parse and verify an inventory loaded from a journal or receipt."""

    if not isinstance(raw_inventory, dict):
        raise WorkspaceInventoryError("Workspace inventory payload is invalid.")
    directories = _directories_from_mapping(raw_inventory.get("directories"))
    files = _files_from_mapping(raw_inventory.get("files"))
    inventory = build_workspace_inventory(directories, files)
    expected_scalars = {
        "directory_count": inventory.directory_count,
        "file_count": inventory.file_count,
        "total_bytes": inventory.total_bytes,
        "fingerprint": inventory.fingerprint,
    }
    if any(raw_inventory.get(key) != value for key, value in expected_scalars.items()):
        raise WorkspaceInventoryError("Workspace inventory summary is inconsistent.")
    return inventory


def build_workspace_inventory(
    directories: tuple[WorkspaceDirectoryEntry, ...],
    files: tuple[WorkspaceFileEntry, ...],
) -> WorkspaceInventory:
    """Canonicalize entries and calculate their content fingerprint."""

    ordered_directories = tuple(sorted(directories, key=lambda entry: entry.relative_path))
    ordered_files = tuple(sorted(files, key=lambda entry: entry.relative_path))
    _require_unique_paths(ordered_directories, ordered_files)
    total_bytes = sum(entry.byte_size for entry in ordered_files)
    return WorkspaceInventory(
        directories=ordered_directories,
        files=ordered_files,
        total_bytes=total_bytes,
        fingerprint=_inventory_fingerprint(ordered_directories, ordered_files),
    )


class WorkspaceInventoryService:
    """Read and copy a regular-file tree through no-follow descriptors."""

    def inventory(self, root: str | os.PathLike[str]) -> WorkspaceInventory:
        """Return a stable inventory or fail closed on aliases and races."""

        canonical_root = _require_real_directory(Path(root))
        root_descriptor = _open_directory(canonical_root)
        directories: list[WorkspaceDirectoryEntry] = []
        files: list[WorkspaceFileEntry] = []
        try:
            self._walk(root_descriptor, (), directories, files)
        finally:
            os.close(root_descriptor)
        return build_workspace_inventory(tuple(directories), tuple(files))

    def copy_file(
        self,
        source_root: str | os.PathLike[str],
        entry: WorkspaceFileEntry,
        destination: str | os.PathLike[str],
    ) -> None:
        """Copy one inventoried source file and verify its exact bytes."""

        source_descriptor = self._open_relative_file(Path(source_root), entry.relative_path)
        destination_path = Path(destination)
        temporary_path = destination_path.with_name(
            f".{destination_path.name}.workspace-copy.part"
        )
        try:
            source_stat = os.fstat(source_descriptor)
            _require_regular_stat(source_stat, entry)
            _remove_regular_temporary(temporary_path)
            self._copy_descriptor(source_descriptor, temporary_path, entry)
            os.replace(temporary_path, destination_path)
            _sync_directory(destination_path.parent)
            _apply_file_metadata(destination_path, entry)
            if sha256_regular_file(destination_path) != entry.sha256:
                raise WorkspaceInventoryError("Copied workspace file failed readback validation.")
            _require_regular_stat(os.fstat(source_descriptor), entry)
        finally:
            os.close(source_descriptor)
            if os.path.lexists(temporary_path):
                _remove_regular_temporary(temporary_path)

    def _walk(
        self,
        directory_descriptor: int,
        parent_parts: tuple[str, ...],
        directories: list[WorkspaceDirectoryEntry],
        files: list[WorkspaceFileEntry],
    ) -> None:
        """Traverse one already anchored directory descriptor."""

        with os.scandir(directory_descriptor) as iterator:
            entries = sorted(iterator, key=lambda candidate: candidate.name)
        for entry in entries:
            entry_stat = entry.stat(follow_symlinks=False)
            relative_parts = (*parent_parts, entry.name)
            relative_path = PurePosixPath(*relative_parts).as_posix()
            validate_relative_workspace_path(relative_path)
            if stat.S_ISLNK(entry_stat.st_mode):
                raise WorkspaceInventoryError("Symbolic links are forbidden in a workspace migration.")
            if stat.S_ISDIR(entry_stat.st_mode):
                self._walk_directory(
                    directory_descriptor,
                    entry.name,
                    entry_stat,
                    relative_parts,
                    directories,
                    files,
                )
            elif stat.S_ISREG(entry_stat.st_mode):
                files.append(self._inventory_file(directory_descriptor, entry.name, entry_stat, relative_path))
            else:
                raise WorkspaceInventoryError("Only regular files and directories may be migrated.")

    def _walk_directory(
        self,
        parent_descriptor: int,
        name: str,
        expected_stat: os.stat_result,
        relative_parts: tuple[str, ...],
        directories: list[WorkspaceDirectoryEntry],
        files: list[WorkspaceFileEntry],
    ) -> None:
        """Anchor, record, and recursively traverse one directory."""

        descriptor = _open_directory(name, directory_descriptor=parent_descriptor)
        try:
            current_stat = os.fstat(descriptor)
            _require_same_identity(expected_stat, current_stat)
            directories.append(
                WorkspaceDirectoryEntry(
                    relative_path=PurePosixPath(*relative_parts).as_posix(),
                    mode=stat.S_IMODE(current_stat.st_mode),
                    modified_ns=current_stat.st_mtime_ns,
                )
            )
            self._walk(descriptor, relative_parts, directories, files)
            _require_same_identity(current_stat, os.fstat(descriptor))
        finally:
            os.close(descriptor)

    def _inventory_file(
        self,
        parent_descriptor: int,
        name: str,
        expected_stat: os.stat_result,
        relative_path: str,
    ) -> WorkspaceFileEntry:
        """Hash one anchored regular file and revalidate its identity."""

        descriptor = _open_file(name, directory_descriptor=parent_descriptor)
        try:
            initial_stat = os.fstat(descriptor)
            _require_same_identity(expected_stat, initial_stat)
            if not stat.S_ISREG(initial_stat.st_mode):
                raise WorkspaceInventoryError("Workspace entry stopped being a regular file.")
            digest = _sha256_descriptor(descriptor)
            final_stat = os.fstat(descriptor)
            _require_unchanged_file(initial_stat, final_stat)
            return WorkspaceFileEntry(
                relative_path=relative_path,
                byte_size=initial_stat.st_size,
                sha256=digest,
                mode=stat.S_IMODE(initial_stat.st_mode),
                modified_ns=initial_stat.st_mtime_ns,
            )
        finally:
            os.close(descriptor)

    def _open_relative_file(self, source_root: Path, relative_path: str) -> int:
        """Open a source descendant without following any directory alias."""

        parts = validate_relative_workspace_path(relative_path)
        descriptor = _open_directory(_require_real_directory(source_root))
        try:
            for part in parts[:-1]:
                child_descriptor = _open_directory(part, directory_descriptor=descriptor)
                os.close(descriptor)
                descriptor = child_descriptor
            file_descriptor = _open_file(parts[-1], directory_descriptor=descriptor)
        finally:
            os.close(descriptor)
        return file_descriptor

    @staticmethod
    def _copy_descriptor(
        source_descriptor: int,
        temporary_path: Path,
        entry: WorkspaceFileEntry,
    ) -> None:
        """Stream one source descriptor into an exclusive temporary file."""

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        destination_descriptor = os.open(temporary_path, flags, 0o600)
        try:
            os.lseek(source_descriptor, 0, os.SEEK_SET)
            while chunk := os.read(source_descriptor, _COPY_BUFFER_SIZE):
                _write_all(destination_descriptor, chunk)
            os.fsync(destination_descriptor)
            destination_stat = os.fstat(destination_descriptor)
            if destination_stat.st_size != entry.byte_size:
                raise WorkspaceInventoryError("Copied workspace file size differs from inventory.")
        finally:
            os.close(destination_descriptor)


def sha256_regular_file(path: str | os.PathLike[str]) -> str:
    """Hash one regular file through a no-follow descriptor."""

    descriptor = _open_file(Path(path))
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise WorkspaceInventoryError("Workspace artifact is not a regular file.")
        return _sha256_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path) -> Path:
    """Return a canonical directory while rejecting a root-level alias."""

    expanded = path.expanduser()
    try:
        raw_stat = expanded.lstat()
    except OSError as exc:
        raise WorkspaceInventoryError("Workspace directory is unavailable.") from exc
    if stat.S_ISLNK(raw_stat.st_mode) or not stat.S_ISDIR(raw_stat.st_mode):
        raise WorkspaceInventoryError("Workspace root must be a real directory.")
    return expanded.resolve(strict=True)


def _open_directory(
    path: str | os.PathLike[str],
    *,
    directory_descriptor: int | None = None,
) -> int:
    """Open one directory with mandatory no-follow platform support."""

    if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise WorkspaceInventoryError("Safe no-follow directory access is unsupported.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise WorkspaceInventoryError("Workspace directory could not be anchored safely.") from exc


def _open_file(
    path: str | os.PathLike[str],
    *,
    directory_descriptor: int | None = None,
) -> int:
    """Open one file with mandatory no-follow platform support."""

    if not hasattr(os, "O_NOFOLLOW"):
        raise WorkspaceInventoryError("Safe no-follow file access is unsupported.")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(path, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise WorkspaceInventoryError("Workspace file could not be anchored safely.") from exc


def _sha256_descriptor(descriptor: int) -> str:
    """Hash an open descriptor from the beginning without closing it."""

    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, _COPY_BUFFER_SIZE):
        digest.update(chunk)
    return digest.hexdigest()


def _write_all(descriptor: int, payload: bytes) -> None:
    """Write an entire buffer despite short operating-system writes."""

    view = memoryview(payload)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise WorkspaceInventoryError("Workspace copy stopped before completing a buffer.")
        written += count


def _require_same_identity(expected: os.stat_result, observed: os.stat_result) -> None:
    """Reject a directory or file replaced during anchored traversal."""

    if (expected.st_dev, expected.st_ino, expected.st_mode) != (
        observed.st_dev,
        observed.st_ino,
        observed.st_mode,
    ):
        raise WorkspaceInventoryError("Workspace entry changed identity during traversal.")


def _require_unchanged_file(initial: os.stat_result, final: os.stat_result) -> None:
    """Reject a regular file mutated while it was being hashed."""

    _require_same_identity(initial, final)
    if (initial.st_size, initial.st_mtime_ns, initial.st_ctime_ns) != (
        final.st_size,
        final.st_mtime_ns,
        final.st_ctime_ns,
    ):
        raise WorkspaceInventoryError("Workspace file changed while it was being read.")


def _require_regular_stat(observed: os.stat_result, entry: WorkspaceFileEntry) -> None:
    """Require an open source file to still match its inventory entry."""

    if not stat.S_ISREG(observed.st_mode):
        raise WorkspaceInventoryError("Workspace source is no longer a regular file.")
    if observed.st_size != entry.byte_size or stat.S_IMODE(observed.st_mode) != entry.mode:
        raise WorkspaceInventoryError("Workspace source metadata changed after inventory.")


def _remove_regular_temporary(path: Path) -> None:
    """Remove only a regular migration-owned temporary file."""

    try:
        path_stat = path.lstat()
    except FileNotFoundError:
        return
    if not stat.S_ISREG(path_stat.st_mode):
        raise WorkspaceInventoryError("Migration temporary path is aliased or unsupported.")
    path.unlink()


def _apply_file_metadata(path: Path, entry: WorkspaceFileEntry) -> None:
    """Restore portable file mode and timestamp without following aliases."""

    os.chmod(path, entry.mode, follow_symlinks=False)
    os.utime(path, ns=(entry.modified_ns, entry.modified_ns), follow_symlinks=False)


def _sync_directory(path: Path) -> None:
    """Make a same-directory publication durable where supported."""

    descriptor = _open_directory(path)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directories_from_mapping(raw_entries: object) -> tuple[WorkspaceDirectoryEntry, ...]:
    """Parse strict directory entries from a durable artifact."""

    if not isinstance(raw_entries, list):
        raise WorkspaceInventoryError("Workspace directory inventory is invalid.")
    entries: list[WorkspaceDirectoryEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise WorkspaceInventoryError("Workspace directory entry is invalid.")
        relative_path = _require_string(raw_entry, "relative_path")
        validate_relative_workspace_path(relative_path)
        entries.append(
            WorkspaceDirectoryEntry(
                relative_path=relative_path,
                mode=_require_nonnegative_integer(raw_entry, "mode"),
                modified_ns=_require_nonnegative_integer(raw_entry, "modified_ns"),
            )
        )
    return tuple(entries)


def _files_from_mapping(raw_entries: object) -> tuple[WorkspaceFileEntry, ...]:
    """Parse strict regular-file entries from a durable artifact."""

    if not isinstance(raw_entries, list):
        raise WorkspaceInventoryError("Workspace file inventory is invalid.")
    entries: list[WorkspaceFileEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise WorkspaceInventoryError("Workspace file entry is invalid.")
        entries.append(_file_from_mapping(raw_entry))
    return tuple(entries)


def _file_from_mapping(raw_entry: dict[object, object]) -> WorkspaceFileEntry:
    """Parse one strict regular-file inventory entry."""

    relative_path = _require_string(raw_entry, "relative_path")
    validate_relative_workspace_path(relative_path)
    digest = _require_string(raw_entry, "sha256")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise WorkspaceInventoryError("Workspace file digest is invalid.")
    return WorkspaceFileEntry(
        relative_path=relative_path,
        byte_size=_require_nonnegative_integer(raw_entry, "byte_size"),
        sha256=digest,
        mode=_require_nonnegative_integer(raw_entry, "mode"),
        modified_ns=_require_nonnegative_integer(raw_entry, "modified_ns"),
    )


def _require_string(mapping: dict[object, object], key: str) -> str:
    """Return one required non-empty string field."""

    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise WorkspaceInventoryError(f"Workspace inventory field {key!r} is invalid.")
    return value


def _require_nonnegative_integer(mapping: dict[object, object], key: str) -> int:
    """Return one required nonnegative integer field."""

    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise WorkspaceInventoryError(f"Workspace inventory field {key!r} is invalid.")
    return value


def _require_unique_paths(
    directories: tuple[WorkspaceDirectoryEntry, ...],
    files: tuple[WorkspaceFileEntry, ...],
) -> None:
    """Reject duplicates and file/directory path collisions."""

    paths = [entry.relative_path for entry in directories]
    paths.extend(entry.relative_path for entry in files)
    if len(paths) != len(set(paths)):
        raise WorkspaceInventoryError("Workspace inventory contains duplicate paths.")


def _inventory_fingerprint(
    directories: tuple[WorkspaceDirectoryEntry, ...],
    files: tuple[WorkspaceFileEntry, ...],
) -> str:
    """Return a deterministic content-and-layout fingerprint."""

    digest = hashlib.sha256()
    for directory_entry in directories:
        digest.update(f"D\0{directory_entry.relative_path}\0".encode())
    for file_entry in files:
        digest.update(
            f"F\0{file_entry.relative_path}\0{file_entry.byte_size}\0{file_entry.sha256}\0".encode()
        )
    return digest.hexdigest()
