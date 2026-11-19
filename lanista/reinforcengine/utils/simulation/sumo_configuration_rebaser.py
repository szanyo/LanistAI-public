# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Relocation-safe materialization of managed SUMO configurations."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path

from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.utils.simulation.ports import (
    SimulationLifecycleError,
)

_REQUIRED_RESOURCE_TAGS = frozenset(
    {
        "additional-files",
        "gui-settings-file",
        "net-file",
        "route-files",
    }
)
_OPTIONAL_OUTPUT_TAGS = frozenset({"netstate-dump"})
_PATH_TAGS = _REQUIRED_RESOURCE_TAGS | _OPTIONAL_OUTPUT_TAGS
_OUTPUT_SENTINELS = frozenset({"-", "NUL", "stderr", "stdout"})
_SECURE_OPEN_SUPPORTED = all(
    hasattr(os, flag_name) for flag_name in ("O_DIRECTORY", "O_CLOEXEC", "O_NOFOLLOW")
)
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)


class ManagedSumoConfigurationError(SimulationLifecycleError):
    """A managed SUMO configuration failed its confinement or integrity checks."""


def _require_secure_open_support() -> None:
    """Return the required secure open support for this sumo configuration rebaser module."""
    if not _SECURE_OPEN_SUPPORTED:
        raise ManagedSumoConfigurationError(
            "Managed SUMO configurations require no-follow descriptor support."
        )


@dataclass(frozen=True, slots=True)
class _FileSignature:
    """Carry the data required for file signature."""

    device: int
    inode: int
    size: int
    modified_ns: int

    @classmethod
    def capture(cls, metadata: os.stat_result) -> _FileSignature:
        """Return the capture value owned by this file signature."""
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            modified_ns=metadata.st_mtime_ns,
        )


@dataclass(frozen=True, slots=True)
class _ManagedFileIdentity:
    """Carry the data required for managed file identity."""

    resource_root: Path
    relative_path: Path
    signature: _FileSignature
    role: str

    def revalidate(self) -> None:
        """Revalidate the operation owned by this managed file identity."""
        try:
            current = _capture_anchored_signature(
                self.resource_root,
                self.relative_path,
            )
        except ManagedSumoConfigurationError:
            raise
        except OSError as exc:
            raise ManagedSumoConfigurationError(
                f"Managed SUMO {self.role} became unsafe before native start."
            ) from exc
        if current != self.signature:
            raise ManagedSumoConfigurationError(
                f"Managed SUMO {self.role} changed before native start."
            )


@dataclass(frozen=True, slots=True)
class _PrivateFileIdentity:
    """Carry the data required for private file identity."""

    path: Path
    signature: _FileSignature

    def revalidate(self) -> None:
        """Revalidate the operation owned by this private file identity."""
        try:
            current = _capture_private_signature(self.path)
        except ManagedSumoConfigurationError:
            raise
        except OSError as exc:
            raise ManagedSumoConfigurationError(
                "Private SUMO snapshot became unsafe before native start."
            ) from exc
        if current != self.signature:
            raise ManagedSumoConfigurationError(
                "Private SUMO snapshot changed before native start."
            )


@dataclass(frozen=True, slots=True)
class _ManagedOutput:
    """Carry the data required for managed output."""

    resource_root: Path
    relative_path: Path
    staging_root: Path
    staged_path: Path

    def commit(self) -> None:
        """Commit the operation owned by this managed output."""
        _commit_private_output(self)

    def validate_for_start(self) -> None:
        """Validate the for start required by this managed output."""
        try:
            _require_unoccupied_private_output(self)
        except ManagedSumoConfigurationError:
            raise
        except OSError as exc:
            raise ManagedSumoConfigurationError(
                "Private SUMO output destination became unsafe before native start."
            ) from exc


@dataclass(slots=True)
class SumoConfigurationMaterialization:
    """Own and validate one optional private configuration snapshot.

    libsumo accepts paths rather than already-open file descriptors. Managed
    configuration bytes and the direct net, route, additional, and GUI inputs
    are therefore copied from no-follow descriptors into a private tree.
    Outputs are written there too, then atomically committed through an
    anchored directory descriptor after successful native shutdown. Private
    snapshots are signature checked immediately before hand-off.
    """

    source_path: str
    runtime_path: str
    rebased: bool
    _temporary_directory: tempfile.TemporaryDirectory[str] | None = field(
        default=None,
        repr=False,
    )
    _managed_identities: tuple[_ManagedFileIdentity, ...] = field(
        default=(),
        repr=False,
    )
    _private_identities: tuple[_PrivateFileIdentity, ...] = field(
        default=(),
        repr=False,
    )
    _managed_outputs: tuple[_ManagedOutput, ...] = field(default=(), repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def validate_for_start(self) -> None:
        """Fail closed if a managed source, input, or private snapshot changed."""

        for managed_identity in self._managed_identities:
            managed_identity.revalidate()
        for private_identity in self._private_identities:
            private_identity.revalidate()
        for output in self._managed_outputs:
            output.validate_for_start()

    def close(self) -> None:
        """Remove temporary state exactly once, reporting cleanup failures."""

        if self._closed:
            return
        if self._temporary_directory is not None:
            try:
                self._temporary_directory.cleanup()
            except OSError as exc:
                raise ManagedSumoConfigurationError(
                    "Private SUMO configuration cleanup failed."
                ) from exc
        self._closed = True

    def commit_outputs(self) -> None:
        """Atomically publish simulator outputs to their managed destinations."""

        for output in self._managed_outputs:
            output.commit()


def _local_name(tag: str) -> str:
    """Return the local name value owned by this sumo configuration rebaser module."""
    return tag.rsplit("}", maxsplit=1)[-1]


def _lexical_resource_root(source: Path) -> Path | None:
    """Return the lexical resource root value owned by this sumo configuration rebaser module."""
    for candidate in source.parents:
        if (
            candidate.name == "resources"
            and candidate.parent.parent.name == "models"
            and candidate.parent.parent.parent.name == "rl_model_storage"
        ):
            return candidate
    return None


def _canonical_resource_guard(lexical_resource_root: Path) -> StoragePathGuard:
    """Return the canonical resource guard value owned by this sumo configuration rebaser module."""
    storage_root = lexical_resource_root.parent.parent.parent
    package_guard = StoragePathGuard(storage_root.parent)
    canonical_storage_root = package_guard.require_descendant(storage_root, must_exist=True)
    storage_guard = StoragePathGuard(canonical_storage_root)
    canonical_model_root = storage_guard.require_descendant(
        lexical_resource_root.parent,
        must_exist=True,
    )
    model_guard = StoragePathGuard(canonical_model_root)
    canonical_resource_root = model_guard.require_descendant(
        lexical_resource_root,
        must_exist=True,
    )
    return StoragePathGuard(canonical_resource_root)


def _relative_parts(relative_path: Path) -> tuple[str, ...]:
    """Return the relative parts value owned by this sumo configuration rebaser module."""
    if relative_path.is_absolute() or not relative_path.parts:
        raise ManagedSumoConfigurationError("Managed SUMO path is not root-relative.")
    if any(part in {"", ".", ".."} for part in relative_path.parts):
        raise ManagedSumoConfigurationError("Managed SUMO path contains traversal markers.")
    return relative_path.parts


@contextmanager
def _open_resource_root(resource_root: Path) -> Iterator[int]:
    """Open the resource root workflow owned by this sumo configuration rebaser module."""
    model_root = resource_root.parent
    storage_root = model_root.parent.parent
    package_root = storage_root.parent
    if (
        resource_root.name != "resources"
        or model_root.parent.name != "models"
        or storage_root.name != "rl_model_storage"
    ):
        raise ManagedSumoConfigurationError("Managed SUMO resource layout is invalid.")
    file_descriptors: list[int] = []
    try:
        file_descriptors.append(os.open(package_root, _DIRECTORY_OPEN_FLAGS))
        for component in ("rl_model_storage", "models", model_root.name, "resources"):
            descriptor = os.open(
                component,
                _DIRECTORY_OPEN_FLAGS,
                dir_fd=file_descriptors[-1],
            )
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                os.close(descriptor)
                raise ManagedSumoConfigurationError(
                    "Managed SUMO path component is not a directory."
                )
            file_descriptors.append(descriptor)
        yield file_descriptors[-1]
    finally:
        for descriptor in reversed(file_descriptors):
            os.close(descriptor)


@contextmanager
def _open_anchored_directory(
    resource_root: Path,
    relative_directory: Path,
) -> Iterator[int]:
    """Open the anchored directory workflow owned by this sumo configuration rebaser module."""
    parts = () if relative_directory == Path() else _relative_parts(relative_directory)
    opened_directories: list[int] = []
    with _open_resource_root(resource_root) as root_descriptor:
        directory_descriptor = root_descriptor
        try:
            for component in parts:
                descriptor = os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=directory_descriptor,
                )
                opened_directories.append(descriptor)
                directory_descriptor = descriptor
            yield directory_descriptor
        finally:
            for descriptor in reversed(opened_directories):
                os.close(descriptor)


@contextmanager
def _open_anchored_regular_file(
    resource_root: Path,
    relative_path: Path,
) -> Iterator[int]:
    """Open the anchored regular file workflow owned by this sumo configuration rebaser module."""
    parts = _relative_parts(relative_path)
    file_descriptor: int | None = None
    with _open_anchored_directory(
        resource_root,
        Path(*parts[:-1]),
    ) as parent_descriptor:
        try:
            file_descriptor = os.open(
                parts[-1],
                _FILE_OPEN_FLAGS,
                dir_fd=parent_descriptor,
            )
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise ManagedSumoConfigurationError("Managed SUMO input is not a regular file.")
            yield file_descriptor
        finally:
            if file_descriptor is not None:
                os.close(file_descriptor)


@contextmanager
def _open_private_directory(root: Path, relative_directory: Path) -> Iterator[int]:
    """Open the private directory workflow owned by this sumo configuration rebaser module."""
    parts = () if relative_directory == Path() else _relative_parts(relative_directory)
    file_descriptors: list[int] = []
    try:
        file_descriptors.append(os.open(root, _DIRECTORY_OPEN_FLAGS))
        for component in parts:
            file_descriptors.append(
                os.open(
                    component,
                    _DIRECTORY_OPEN_FLAGS,
                    dir_fd=file_descriptors[-1],
                )
            )
        yield file_descriptors[-1]
    finally:
        for descriptor in reversed(file_descriptors):
            os.close(descriptor)


def _require_unoccupied_private_output(output: _ManagedOutput) -> None:
    """Return the required unoccupied private output for this sumo configuration rebaser module."""
    relative_path = output.staged_path.relative_to(output.staging_root)
    with _open_private_directory(output.staging_root, relative_path.parent) as parent_descriptor:
        try:
            os.stat(
                relative_path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
    raise ManagedSumoConfigurationError(
        "Private SUMO output destination was occupied before native start."
    )


def _read_anchored_snapshot(
    resource_root: Path,
    relative_path: Path,
) -> tuple[bytes, _FileSignature]:
    """Read the anchored snapshot without changing this sumo configuration rebaser module's state.
    """
    with _open_anchored_regular_file(resource_root, relative_path) as descriptor:
        before = _FileSignature.capture(os.fstat(descriptor))
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        snapshot = b"".join(chunks)
        after = _FileSignature.capture(os.fstat(descriptor))
    if before != after or len(snapshot) != before.size:
        raise ManagedSumoConfigurationError("Managed SUMO file changed while it was snapshotted.")
    return snapshot, before


def _capture_anchored_signature(
    resource_root: Path,
    relative_path: Path,
) -> _FileSignature:
    """Capture the anchored signature required by this sumo configuration rebaser module."""
    with _open_anchored_regular_file(resource_root, relative_path) as descriptor:
        return _FileSignature.capture(os.fstat(descriptor))


def _capture_private_signature(path: Path) -> _FileSignature:
    """Capture the private signature required by this sumo configuration rebaser module."""
    descriptor = os.open(path, _FILE_OPEN_FLAGS)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ManagedSumoConfigurationError(
                "Private SUMO configuration snapshot is not a regular file."
            )
        return _FileSignature.capture(metadata)
    finally:
        os.close(descriptor)


def _legacy_resource_suffix(
    path: Path,
    *,
    expected_model_name: str,
) -> tuple[str, ...] | None:
    """Return the legacy resource suffix value owned by this sumo configuration rebaser module."""
    parts = path.parts
    matches = [
        index
        for index in range(len(parts) - 3)
        if parts[index : index + 4]
        == ("rl_model_storage", "models", expected_model_name, "resources")
    ]
    if len(matches) != 1:
        return None
    return tuple(parts[matches[0] + 4 :])


def _resolve_resource_path(
    value: str,
    *,
    source_directory: Path,
    resource_guard: StoragePathGuard,
    must_exist: bool,
) -> Path:
    """Resolve the resource path required by this sumo configuration rebaser module."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        return resource_guard.require_descendant(
            source_directory / path,
            must_exist=must_exist,
        )
    try:
        return resource_guard.require_descendant(path, must_exist=must_exist)
    except StoragePathViolation:
        suffix = _legacy_resource_suffix(
            path,
            expected_model_name=resource_guard.root.parent.name,
        )
        if not suffix:
            raise
        return resource_guard.resolve(*suffix, must_exist=must_exist)


def _rebase_path_list(
    value: str,
    *,
    source_directory: Path,
    resource_guard: StoragePathGuard,
    must_exist: bool,
) -> tuple[str, tuple[Path, ...]]:
    """Return the rebase path list value owned by this sumo configuration rebaser module."""
    rebased: list[str] = []
    resolved_paths: list[Path] = []
    for raw_part in value.split(","):
        stripped = raw_part.strip()
        if not stripped:
            raise ManagedSumoConfigurationError(
                "Managed SUMO configuration contains an empty resource path."
            )
        if stripped in _OUTPUT_SENTINELS and not must_exist:
            rebased.append(raw_part)
            continue
        prefix = raw_part[: len(raw_part) - len(raw_part.lstrip())]
        suffix = raw_part[len(raw_part.rstrip()) :]
        resolved = _resolve_resource_path(
            stripped,
            source_directory=source_directory,
            resource_guard=resource_guard,
            must_exist=must_exist,
        )
        rebased.append(prefix + os.fspath(resolved) + suffix)
        resolved_paths.append(resolved)
    return ",".join(rebased), tuple(resolved_paths)


def _rebase_tree_paths(
    tree: ET.ElementTree[ET.Element[str]],
    *,
    source_directory: Path,
    resource_guard: StoragePathGuard,
) -> tuple[bool, tuple[Path, ...], tuple[Path, ...]]:
    """Return the rebase tree paths value owned by this sumo configuration rebaser module."""
    changed = False
    inputs: list[Path] = []
    outputs: list[Path] = []
    for element in tree.getroot().iter():
        tag = _local_name(element.tag)
        if tag not in _PATH_TAGS:
            continue
        value = element.get("value")
        if value is None:
            raise ManagedSumoConfigurationError("Managed SUMO resource entry has no value.")
        rebased, element_inputs = _rebase_path_list(
            value,
            source_directory=source_directory,
            resource_guard=resource_guard,
            must_exist=tag in _REQUIRED_RESOURCE_TAGS,
        )
        if tag in _REQUIRED_RESOURCE_TAGS:
            inputs.extend(element_inputs)
        else:
            outputs.extend(element_inputs)
        if rebased != value:
            element.set("value", rebased)
            changed = True
    return changed, tuple(inputs), tuple(outputs)


def _write_private_bytes(path: Path, content: bytes) -> None:
    """Persist the private bytes owned by this sumo configuration rebaser module."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(content)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_private_tree(
    runtime_path: Path,
    tree: ET.ElementTree[ET.Element[str]],
) -> None:
    """Persist the private tree owned by this sumo configuration rebaser module."""
    descriptor = os.open(
        runtime_path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            tree.write(stream, encoding="utf-8", xml_declaration=True)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _stage_managed_inputs(
    resource_root: Path,
    inputs: tuple[Path, ...],
    staging_root: Path,
) -> tuple[dict[str, str], tuple[_PrivateFileIdentity, ...]]:
    """Return the stage managed inputs value owned by this sumo configuration rebaser module."""
    replacements: dict[str, str] = {}
    identities: list[_PrivateFileIdentity] = []
    for source in dict.fromkeys(inputs):
        relative_path = source.relative_to(resource_root)
        snapshot, _signature = _read_anchored_snapshot(resource_root, relative_path)
        staged_path = staging_root / "inputs" / relative_path
        _write_private_bytes(staged_path, snapshot)
        replacements[os.fspath(source)] = os.fspath(staged_path)
        identities.append(
            _PrivateFileIdentity(staged_path, _capture_private_signature(staged_path))
        )
    return replacements, tuple(identities)


def _redirect_tree_paths(
    tree: ET.ElementTree[ET.Element[str]],
    path_tags: frozenset[str],
    replacements: dict[str, str],
) -> None:
    """Redirect the tree paths owned by this sumo configuration rebaser
module."""
    for element in tree.getroot().iter():
        if _local_name(element.tag) not in path_tags:
            continue
        value = element.get("value")
        if value is None:
            raise ManagedSumoConfigurationError("Managed SUMO resource entry has no value.")
        staged_parts: list[str] = []
        for raw_part in value.split(","):
            stripped = raw_part.strip()
            if path_tags == _OPTIONAL_OUTPUT_TAGS and stripped in _OUTPUT_SENTINELS:
                staged_parts.append(raw_part)
                continue
            replacement = replacements.get(stripped)
            if replacement is None:
                raise ManagedSumoConfigurationError(
                    "Managed SUMO private-path mapping is incomplete."
                )
            prefix = raw_part[: len(raw_part) - len(raw_part.lstrip())]
            suffix = raw_part[len(raw_part.rstrip()) :]
            staged_parts.append(prefix + replacement + suffix)
        element.set("value", ",".join(staged_parts))


def _prepare_managed_outputs(
    resource_root: Path,
    outputs: tuple[Path, ...],
    staging_root: Path,
) -> tuple[dict[str, str], tuple[_ManagedOutput, ...]]:
    """Prepare the managed outputs required by this sumo configuration rebaser module."""
    replacements: dict[str, str] = {}
    managed_outputs: list[_ManagedOutput] = []
    for destination in dict.fromkeys(outputs):
        relative_path = destination.relative_to(resource_root)
        with _open_anchored_directory(resource_root, relative_path.parent):
            pass
        staged_path = staging_root / "outputs" / relative_path
        staged_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        replacements[os.fspath(destination)] = os.fspath(staged_path)
        managed_outputs.append(
            _ManagedOutput(
                resource_root=resource_root,
                relative_path=relative_path,
                staging_root=staging_root,
                staged_path=staged_path,
            )
        )
    return replacements, tuple(managed_outputs)


def _read_private_output(path: Path) -> bytes | None:
    """Read the private output without changing this sumo configuration rebaser module's state."""
    try:
        descriptor = os.open(path, _FILE_OPEN_FLAGS)
    except FileNotFoundError:
        return None
    try:
        before = _FileSignature.capture(os.fstat(descriptor))
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ManagedSumoConfigurationError("Private SUMO output is not a regular file.")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        content = b"".join(chunks)
        after = _FileSignature.capture(os.fstat(descriptor))
    finally:
        os.close(descriptor)
    if before != after or len(content) != before.size:
        raise ManagedSumoConfigurationError("Private SUMO output changed while it was finalized.")
    return content


def _atomic_replace_anchored(
    resource_root: Path,
    relative_path: Path,
    content: bytes,
) -> None:
    """Handle the atomic replace anchored operation owned by this sumo configuration rebaser
    module.
    """
    temporary_name = f".{relative_path.name}.lanistai-{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    with _open_anchored_directory(resource_root, relative_path.parent) as parent_descriptor:
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=parent_descriptor,
            )
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary_name,
                relative_path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            os.fsync(parent_descriptor)
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            raise


def _commit_private_output(output: _ManagedOutput) -> None:
    """Commit the private output owned by this sumo configuration rebaser module."""
    try:
        content = _read_private_output(output.staged_path)
        if content is None:
            return
        _atomic_replace_anchored(
            output.resource_root,
            output.relative_path,
            content,
        )
    except ManagedSumoConfigurationError:
        raise
    except OSError as exc:
        raise ManagedSumoConfigurationError(
            "Private SUMO output could not be committed safely."
        ) from exc


def _cleanup_with_primary_error(
    temporary_directory: tempfile.TemporaryDirectory[str],
    primary_error: ManagedSumoConfigurationError,
) -> None:
    """Clean up the with primary error owned by this sumo configuration rebaser module."""
    try:
        temporary_directory.cleanup()
    except OSError as cleanup_error:
        primary_error.add_note(
            "Private SUMO staging cleanup also failed; the primary materialization "
            f"error was preserved ({cleanup_error!r})."
        )


def _create_private_snapshot(
    source: Path,
    tree: ET.ElementTree[ET.Element[str]],
    resource_root: Path,
    inputs: tuple[Path, ...],
    outputs: tuple[Path, ...],
) -> tuple[
    tempfile.TemporaryDirectory[str],
    Path,
    tuple[_PrivateFileIdentity, ...],
    tuple[_ManagedOutput, ...],
]:
    """Create the private snapshot owned by this sumo configuration rebaser module."""
    try:
        temporary_directory = tempfile.TemporaryDirectory(prefix="lanistai-sumo-config-")
    except OSError as exc:
        raise ManagedSumoConfigurationError(
            "Private SUMO configuration staging could not be created."
        ) from exc
    runtime_path = Path(temporary_directory.name) / source.name
    try:
        staging_root = Path(temporary_directory.name)
        replacements, input_identities = _stage_managed_inputs(
            resource_root,
            inputs,
            staging_root,
        )
        output_replacements, managed_outputs = _prepare_managed_outputs(
            resource_root,
            outputs,
            staging_root,
        )
        _redirect_tree_paths(tree, _REQUIRED_RESOURCE_TAGS, replacements)
        _redirect_tree_paths(tree, _OPTIONAL_OUTPUT_TAGS, output_replacements)
        _write_private_tree(runtime_path, tree)
        runtime_identity = _PrivateFileIdentity(
            runtime_path,
            _capture_private_signature(runtime_path),
        )
    except (OSError, ManagedSumoConfigurationError) as exc:
        primary = ManagedSumoConfigurationError(
            "Managed SUMO configuration could not be materialized."
        )
        _cleanup_with_primary_error(temporary_directory, primary)
        raise primary from exc
    return (
        temporary_directory,
        runtime_path,
        (runtime_identity, *input_identities),
        managed_outputs,
    )


def _managed_identity(
    resource_root: Path,
    path: Path,
    *,
    signature: _FileSignature | None = None,
    role: str,
) -> _ManagedFileIdentity:
    """Return the managed identity value owned by this sumo configuration rebaser module."""
    try:
        relative_path = path.relative_to(resource_root)
    except ValueError as exc:
        raise ManagedSumoConfigurationError("Managed SUMO file escaped its resource root.") from exc
    return _ManagedFileIdentity(
        resource_root=resource_root,
        relative_path=relative_path,
        signature=signature or _capture_anchored_signature(resource_root, relative_path),
        role=role,
    )


def _close_preserving_primary(
    materialization: SumoConfigurationMaterialization,
    primary_error: ManagedSumoConfigurationError,
) -> None:
    """Close the preserving primary owned by this sumo configuration rebaser module."""
    try:
        materialization.close()
    except ManagedSumoConfigurationError as cleanup_error:
        primary_error.add_note(
            "Private SUMO snapshot cleanup also failed; validation remains the "
            f"primary error ({cleanup_error!r})."
        )


def _materialize_managed_configuration(
    lexical_source: Path,
    lexical_resource_root: Path,
) -> SumoConfigurationMaterialization:
    """Return the materialize managed configuration value owned by this sumo configuration rebaser
    module.
    """
    _require_secure_open_support()
    resource_guard = _canonical_resource_guard(lexical_resource_root)
    try:
        source_relative = lexical_source.relative_to(lexical_resource_root)
    except ValueError as exc:
        raise ManagedSumoConfigurationError(
            "Managed SUMO configuration escaped its resource root."
        ) from exc
    source = resource_guard.root / source_relative
    source_snapshot, source_signature = _read_anchored_snapshot(
        resource_guard.root,
        source_relative,
    )
    parsed_root = ET.fromstring(source_snapshot)
    if parsed_root is None:
        raise ManagedSumoConfigurationError("Managed SUMO configuration has no XML root.")
    root = parsed_root
    tree: ET.ElementTree[ET.Element[str]] = ET.ElementTree(root)
    changed, inputs, outputs = _rebase_tree_paths(
        tree,
        source_directory=source.parent,
        resource_guard=resource_guard,
    )
    identities = (
        _managed_identity(
            resource_guard.root,
            source,
            signature=source_signature,
            role="configuration source",
        ),
    )
    (
        temporary_directory,
        runtime_path,
        private_identities,
        managed_outputs,
    ) = _create_private_snapshot(
        source,
        tree,
        resource_guard.root,
        inputs,
        outputs,
    )
    materialization = SumoConfigurationMaterialization(
        source_path=os.fspath(source),
        runtime_path=os.fspath(runtime_path),
        rebased=changed,
        _temporary_directory=temporary_directory,
        _managed_identities=identities,
        _private_identities=private_identities,
        _managed_outputs=managed_outputs,
    )
    try:
        materialization.validate_for_start()
    except ManagedSumoConfigurationError as exc:
        _close_preserving_primary(materialization, exc)
        raise
    return materialization


def materialize_sumo_configuration(
    configuration_path: str,
) -> SumoConfigurationMaterialization:
    """Validate and privately snapshot one managed configuration."""

    lexical_source = Path(configuration_path).expanduser()
    if not lexical_source.is_absolute():
        lexical_source = Path.cwd() / lexical_source
    lexical_resource_root = _lexical_resource_root(lexical_source)
    if lexical_resource_root is None:
        return SumoConfigurationMaterialization(
            source_path=configuration_path,
            runtime_path=configuration_path,
            rebased=False,
        )
    try:
        return _materialize_managed_configuration(
            lexical_source,
            lexical_resource_root,
        )
    except ManagedSumoConfigurationError:
        raise
    except (ET.ParseError, FileNotFoundError, OSError, StoragePathViolation) as exc:
        raise ManagedSumoConfigurationError(
            "Managed SUMO configuration could not be validated safely."
        ) from exc
