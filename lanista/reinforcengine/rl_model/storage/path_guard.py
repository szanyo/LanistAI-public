# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Root-confined path resolution for model and runtime storage."""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.rl_model.storage.names import StorageName


class StoragePathViolation(ValueError):
    """Raised when a path would escape or erase its declared storage root."""


@dataclass(frozen=True, slots=True)
class StoragePathGuard:
    """Resolve paths only when their canonical location stays below ``root``."""

    root: Path

    def __init__(self, root: str | os.PathLike[str]) -> None:
        """Initialize the storage path guard and its owned state."""
        raw_root = Path(root).expanduser()
        canonical_root = raw_root.resolve(strict=False)
        if canonical_root == Path(canonical_root.anchor):
            raise StoragePathViolation("A filesystem root is too broad for managed storage.")
        object.__setattr__(self, "root", canonical_root)

    def resolve(
        self,
        *relative_parts: str | os.PathLike[str] | StorageName,
        must_exist: bool = False,
    ) -> Path:
        """Resolve a relative path below the root and reject ambiguous input."""
        if not relative_parts:
            raise StoragePathViolation("Storage-root operations are forbidden.")

        raw_parts: list[str] = []
        for part in relative_parts:
            raw_part = os.fspath(part)
            if not raw_part:
                raise StoragePathViolation("Empty storage path components are forbidden.")
            candidate_part = Path(raw_part)
            if candidate_part.is_absolute() or os.path.isabs(raw_part):
                raise StoragePathViolation(f"Absolute storage paths are forbidden: {raw_part!r}")
            if "\\" in raw_part:
                raise StoragePathViolation(f"Backslash-separated storage paths are forbidden: {raw_part!r}")
            path_components = candidate_part.parts
            if any(component in {"", ".", ".."} for component in path_components):
                raise StoragePathViolation(f"Relative path markers are forbidden: {raw_part!r}")
            raw_parts.extend(path_components)

        return self._validate_candidate(self.root.joinpath(*raw_parts), must_exist=must_exist)

    def require_descendant(
        self,
        candidate: str | os.PathLike[str],
        *,
        must_exist: bool = False,
    ) -> Path:
        """Validate an already assembled absolute or relative candidate path."""
        raw_candidate = Path(candidate).expanduser()
        if raw_candidate.is_absolute():
            return self._validate_candidate(raw_candidate, must_exist=must_exist)
        return self.resolve(raw_candidate, must_exist=must_exist)

    def relative_path(self, candidate: str | os.PathLike[str]) -> str:
        """Return a validated descendant as a root-relative POSIX journal path."""
        validated = self.require_descendant(candidate)
        return validated.relative_to(self.root).as_posix()

    def validate_all(
        self,
        candidates: Iterable[str | os.PathLike[str]],
        *,
        must_exist: bool = False,
    ) -> tuple[Path, ...]:
        """Validate several paths before the caller starts a multi-path operation."""
        return tuple(self.require_descendant(candidate, must_exist=must_exist) for candidate in candidates)

    def _validate_candidate(self, candidate: Path, *, must_exist: bool) -> Path:
        """Validate the candidate required by this storage path guard."""
        canonical_candidate = candidate.resolve(strict=False)
        if canonical_candidate == self.root:
            raise StoragePathViolation("Storage-root operations are forbidden.")
        try:
            canonical_candidate.relative_to(self.root)
        except ValueError as exc:
            raise StoragePathViolation(
                f"Storage path escapes its root: {candidate!s} is outside {self.root!s}."
            ) from exc
        if must_exist and not os.path.lexists(canonical_candidate):
            raise FileNotFoundError(canonical_candidate)
        return canonical_candidate
