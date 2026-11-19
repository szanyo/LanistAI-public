# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Strict durable-revision reads and identity fingerprints for metadata edits."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path

import yaml

from lanista.reinforcengine.application_errors import (
    OperationConflict,
    PersistenceError,
    ResourceNotFoundError,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


def metadata_digest(payload: bytes) -> str:
    """Return the stable digest used by metadata compare-and-swap checks."""

    return hashlib.sha256(payload).hexdigest()


def read_existing_payload(
    target: Path,
    kind: MetadataKind,
) -> tuple[dict[str, object], str]:
    """Read one strict versioned payload and prove it did not change mid-read."""

    try:
        raw_payload = target.read_bytes()
        decoded = raw_payload.decode("utf-8")
        payload = yaml.safe_load(decoded) or {}
        if not isinstance(payload, Mapping):
            raise ValueError("Metadata payload is not a mapping.")
        canonical = validate_versioned_metadata_payload(kind, payload)
        if target.read_bytes() != raw_payload:
            raise OperationConflict(
                "Metadata changed while it was being read. Reopen the edit dialog."
            )
    except OperationConflict:
        raise
    except FileNotFoundError as exc:
        raise ResourceNotFoundError(
            "A metadata target no longer exists. Reopen the edit dialog."
        ) from exc
    except (OSError, UnicodeError, ValueError, yaml.YAMLError) as exc:
        raise PersistenceError(
            f"Stored {kind.value} metadata is unreadable or unsupported. "
            "Run the metadata migration before editing it."
        ) from exc
    return dict(canonical), metadata_digest(raw_payload)


def checkpoint_artifact_fingerprint(artifact: Path) -> str:
    """Fingerprint artifact identity and metadata without reading large models."""

    records: list[str] = []
    candidates: tuple[Path, ...] = (artifact,)
    if artifact.is_dir():
        candidates += tuple(sorted(artifact.rglob("*")))
    for candidate in candidates:
        artifact_stat = candidate.lstat()
        if stat.S_ISLNK(artifact_stat.st_mode):
            raise StoragePathViolation(
                "Checkpoint artifacts cannot contain symbolic links."
            )
        if not (
            stat.S_ISREG(artifact_stat.st_mode)
            or stat.S_ISDIR(artifact_stat.st_mode)
        ):
            raise StoragePathViolation(
                "Checkpoint artifacts must contain only files and directories."
            )
        relative = (
            "."
            if candidate == artifact
            else candidate.relative_to(artifact).as_posix()
        )
        records.append(
            ":".join(
                (
                    relative,
                    str(stat.S_IFMT(artifact_stat.st_mode)),
                    str(artifact_stat.st_dev),
                    str(artifact_stat.st_ino),
                    str(artifact_stat.st_size),
                    str(artifact_stat.st_mtime_ns),
                )
            )
        )
    return metadata_digest("\n".join(records).encode("utf-8"))


def require_regular_file(
    guard: StoragePathGuard,
    candidate: Path,
) -> Path:
    """Resolve one required metadata file without accepting aliases."""

    lexical = candidate.absolute()
    guarded = guard.require_descendant(candidate, must_exist=True)
    if lexical != guarded or candidate.is_symlink():
        raise StoragePathViolation("Metadata aliases are forbidden.")
    if not guarded.is_file():
        raise ResourceNotFoundError("A metadata target no longer exists.")
    return guarded


__all__ = [
    "checkpoint_artifact_fingerprint",
    "metadata_digest",
    "read_existing_payload",
    "require_regular_file",
]
