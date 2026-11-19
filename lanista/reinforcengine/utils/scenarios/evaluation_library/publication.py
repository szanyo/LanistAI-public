# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validate and publish one complete-only immutable evaluation library."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path
from types import TracebackType
from typing import Self

from lanista.reinforcengine.utils.scenarios.evaluation_library.atomic_publish import (
    AtomicPublishUnavailable,
    rename_directory_no_replace,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    AnyEvaluationLibraryManifest,
    canonical_manifest_bytes,
    manifest_digest,
    validate_manifest_payload,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryPathViolation,
    EvaluationLibraryStorageConfiguration,
)

EVALUATION_LIBRARY_MANIFEST_FILENAME = "evaluation-library.json"


class EvaluationLibraryPublicationError(RuntimeError):
    """Raised when a library cannot be safely published."""


class EvaluationLibraryIntegrityError(EvaluationLibraryPublicationError):
    """Raised when staged files do not match their manifest evidence."""


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one already-confined regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular_file_inventory(root: Path) -> set[str]:
    """Return a no-symlink inventory relative to the staged library root."""

    inventory: set[str] = set()
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for directory_name in directory_names:
            directory = current / directory_name
            if directory.is_symlink():
                raise EvaluationLibraryIntegrityError(
                    f"Symbolic links are forbidden in evaluation libraries: {directory!s}."
                )
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink() or not path.is_file():
                raise EvaluationLibraryIntegrityError(
                    f"Only regular files are allowed in evaluation libraries: {path!s}."
                )
            inventory.add(path.relative_to(root).as_posix())
    return inventory


def verify_library_contents(root: Path, manifest: AnyEvaluationLibraryManifest) -> None:
    """Verify exact staged-file coverage, sizes, and SHA-256 identities."""

    expected: dict[str, tuple[int, str]] = {}
    for scenario in manifest.scenarios:
        for file_digest in scenario.files:
            relative_path = (
                Path(scenario.relative_path) / file_digest.relative_path
            ).as_posix()
            if relative_path in expected:
                raise EvaluationLibraryIntegrityError(
                    f"Duplicate expanded artifact path: {relative_path}."
                )
            expected[relative_path] = (file_digest.byte_size, file_digest.sha256)

    actual = _regular_file_inventory(root)
    actual.discard(EVALUATION_LIBRARY_MANIFEST_FILENAME)
    if actual != set(expected):
        missing = sorted(set(expected) - actual)
        unexpected = sorted(actual - set(expected))
        raise EvaluationLibraryIntegrityError(
            f"Staged library inventory differs from the manifest; missing={missing}, unexpected={unexpected}."
        )

    for relative_path, (expected_size, expected_digest) in expected.items():
        path = root / relative_path
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_digest:
            raise EvaluationLibraryIntegrityError(
                f"Staged library file does not match its manifest digest: {relative_path}."
            )


def read_verified_library(
    storage: EvaluationLibraryStorageConfiguration,
    library_id: str,
    *,
    expected_manifest_sha256: str | None = None,
) -> AnyEvaluationLibraryManifest:
    """Load one complete library only after rechecking its immutable evidence."""

    root_descriptor = storage.open_library_root(create=False)
    library_descriptor: int | None = None
    try:
        flags = (
            os.O_RDONLY
            | os.O_DIRECTORY
            | os.O_NOFOLLOW
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            library_descriptor = os.open(
                library_id,
                flags,
                dir_fd=root_descriptor,
            )
        except OSError as exc:
            raise EvaluationLibraryIntegrityError(
                f"Evaluation library is missing or unsafe: {library_id}."
            ) from exc
        library_root = Path(f"/proc/self/fd/{library_descriptor}")
        manifest_path = library_root / EVALUATION_LIBRARY_MANIFEST_FILENAME
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise EvaluationLibraryIntegrityError(
                "Evaluation library manifest is missing or unsafe."
            )
        raw_manifest = manifest_path.read_bytes()
        try:
            manifest = validate_manifest_payload(json.loads(raw_manifest))
        except Exception as exc:
            raise EvaluationLibraryIntegrityError("Evaluation library manifest is invalid.") from exc
        if raw_manifest != canonical_manifest_bytes(manifest):
            raise EvaluationLibraryIntegrityError("Evaluation library manifest is not canonical.")
        if manifest.library_id != library_id or not manifest.complete:
            raise EvaluationLibraryIntegrityError(
                "Evaluation library identity is mismatched or incomplete."
            )
        actual_digest = manifest_digest(manifest)
        if expected_manifest_sha256 is not None and actual_digest != expected_manifest_sha256:
            raise EvaluationLibraryIntegrityError("Evaluation library manifest digest changed.")
        verify_library_contents(library_root, manifest)
        storage.require_current_root(root_descriptor)
        return manifest
    finally:
        if library_descriptor is not None:
            os.close(library_descriptor)
        os.close(root_descriptor)


class EvaluationLibraryPublication:
    """Own a private staging directory and publish it exactly once."""

    def __init__(
        self,
        storage: EvaluationLibraryStorageConfiguration,
        library_id: str,
    ) -> None:
        """Resolve the immutable target without touching training storage."""

        self._storage = storage
        self._target = storage.library_path(library_id)
        self._library_id = library_id
        self._root_descriptor: int | None = None
        self._workspace_descriptor: int | None = None
        self._staging_descriptor: int | None = None
        self._workspace_name: str | None = None
        self._published = False

    @property
    def staging_root(self) -> Path:
        """Return the private root in which generation may materialise files."""

        if self._staging_descriptor is None:
            raise EvaluationLibraryPublicationError("Publication staging is not active.")
        return Path(f"/proc/self/fd/{self._staging_descriptor}")

    def __enter__(self) -> Self:
        """Create private same-filesystem staging for an absent library ID."""

        if os.path.lexists(self._target):
            raise FileExistsError(self._target)
        try:
            self._root_descriptor = self._storage.open_library_root(create=True)
            self._workspace_name = f".{self._target.name}.staging-{uuid.uuid4().hex}"
            os.mkdir(self._workspace_name, mode=0o700, dir_fd=self._root_descriptor)
            flags = (
                os.O_RDONLY
                | os.O_DIRECTORY
                | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0)
            )
            self._workspace_descriptor = os.open(
                self._workspace_name,
                flags,
                dir_fd=self._root_descriptor,
            )
            os.mkdir(self._library_id, mode=0o700, dir_fd=self._workspace_descriptor)
            self._staging_descriptor = os.open(
                self._library_id,
                flags,
                dir_fd=self._workspace_descriptor,
            )
            return self
        except BaseException:
            self._cleanup()
            raise

    def publish(self, manifest: AnyEvaluationLibraryManifest) -> Path:
        """Validate staged evidence and publish a manifest-committed library."""

        if self._published:
            raise EvaluationLibraryPublicationError("Library was already published.")
        validated = validate_manifest_payload(manifest.model_dump(mode="json"))
        if validated.library_id != self._library_id:
            raise EvaluationLibraryPublicationError(
                "Manifest library_id does not match the publication target."
            )
        if not validated.complete:
            raise EvaluationLibraryPublicationError("An incomplete library cannot be published.")
        root_descriptor = self._require_descriptor(self._root_descriptor)
        workspace_descriptor = self._require_descriptor(self._workspace_descriptor)
        self._require_unchanged_root(root_descriptor)
        staging = self.staging_root
        verify_library_contents(staging, validated)
        manifest_path = staging / EVALUATION_LIBRARY_MANIFEST_FILENAME
        with manifest_path.open("xb") as handle:
            handle.write(canonical_manifest_bytes(validated))
            handle.flush()
            os.fsync(handle.fileno())
        validate_manifest_payload(json.loads(manifest_path.read_bytes()))
        if os.path.lexists(self._target):
            raise FileExistsError(self._target)
        self._require_unchanged_root(root_descriptor)
        try:
            rename_directory_no_replace(
                workspace_descriptor,
                self._library_id,
                root_descriptor,
                self._target.name,
            )
        except AtomicPublishUnavailable:
            self._publish_with_manifest_commit(root_descriptor, workspace_descriptor)
        self._published = True
        return self._target

    def _publish_with_manifest_commit(
        self, root_descriptor: int, workspace_descriptor: int
    ) -> None:
        """Reserve a name on limited filesystems and expose the manifest last.

        Readers reject a directory without a complete verified manifest. The
        no-replace mkdir prevents overwriting a concurrent publisher.
        """

        staging_descriptor = self._require_descriptor(self._staging_descriptor)
        os.mkdir(self._library_id, mode=0o700, dir_fd=root_descriptor)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
        try:
            target_descriptor = os.open(self._library_id, flags, dir_fd=root_descriptor)
        except BaseException:
            os.rmdir(self._library_id, dir_fd=root_descriptor)
            raise
        moved: list[str] = []
        try:
            self._require_unchanged_root(root_descriptor)
            entries = sorted(path.name for path in self.staging_root.iterdir())
            for name in entries:
                if name == EVALUATION_LIBRARY_MANIFEST_FILENAME:
                    continue
                os.rename(name, name, src_dir_fd=staging_descriptor, dst_dir_fd=target_descriptor)
                moved.append(name)
            os.fsync(target_descriptor)
            os.rename(
                EVALUATION_LIBRARY_MANIFEST_FILENAME,
                EVALUATION_LIBRARY_MANIFEST_FILENAME,
                src_dir_fd=staging_descriptor,
                dst_dir_fd=target_descriptor,
            )
            # After the manifest move the target is complete and selectable.
        except BaseException:
            for name in reversed(moved):
                os.rename(
                    name, name,
                    src_dir_fd=target_descriptor,
                    dst_dir_fd=staging_descriptor,
                )
            os.rmdir(self._library_id, dir_fd=root_descriptor)
            raise
        finally:
            os.close(target_descriptor)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove private staging while leaving a published library untouched."""

        self._cleanup()

    @staticmethod
    def _require_descriptor(descriptor: int | None) -> int:
        """Return an active descriptor or fail a misuse of the lifecycle."""

        if descriptor is None:
            raise EvaluationLibraryPublicationError("Publication staging is not active.")
        return descriptor

    def _require_unchanged_root(self, descriptor: int) -> None:
        """Translate root-identity violations to the publication boundary."""

        try:
            self._storage.require_current_root(descriptor)
        except EvaluationLibraryPathViolation as exc:
            raise EvaluationLibraryPublicationError(
                f"Evaluation-library root changed: {exc}"
            ) from exc

    def _cleanup(self) -> None:
        """Remove only the handle-confined staging workspace and close handles."""

        staging_descriptor = self._staging_descriptor
        self._staging_descriptor = None
        if staging_descriptor is not None:
            os.close(staging_descriptor)

        workspace_descriptor = self._workspace_descriptor
        self._workspace_descriptor = None
        if workspace_descriptor is not None:
            staging_path = Path(f"/proc/self/fd/{workspace_descriptor}") / self._library_id
            if staging_path.exists():
                shutil.rmtree(staging_path)
            os.close(workspace_descriptor)

        root_descriptor = self._root_descriptor
        workspace_name = self._workspace_name
        self._root_descriptor = None
        self._workspace_name = None
        if root_descriptor is not None:
            if workspace_name is not None:
                try:
                    os.rmdir(workspace_name, dir_fd=root_descriptor)
                except FileNotFoundError:
                    pass
            os.close(root_descriptor)


__all__ = [
    "EVALUATION_LIBRARY_MANIFEST_FILENAME",
    "EvaluationLibraryIntegrityError",
    "EvaluationLibraryPublication",
    "EvaluationLibraryPublicationError",
    "read_verified_library",
    "verify_library_contents",
]
