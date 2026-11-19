# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Confine held-out libraries away from the fixed training inventory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    validate_library_id,
)


class EvaluationLibraryPathViolation(ValueError):
    """Raised when evaluation storage could overlap protected training data."""


def _contains(parent: Path, child: Path) -> bool:
    """Return whether ``child`` is equal to or below canonical ``parent``."""

    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _open_directory_chain(path: Path, *, create: bool) -> int:
    """Open an absolute directory through no-follow component traversal."""

    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if os.name != "posix" or not no_follow or not directory_flag:
        raise EvaluationLibraryPathViolation(
            "Secure evaluation-library storage requires POSIX no-follow directory handles."
        )
    flags = os.O_RDONLY | directory_flag | no_follow | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                if not create:
                    raise
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


@dataclass(frozen=True, slots=True)
class EvaluationLibraryStorageConfiguration:
    """Resolve disjoint training and held-out evaluation storage roots."""

    training_root: Path
    library_root: Path
    _canonical_training_root: Path
    _canonical_library_root: Path

    def __init__(
        self,
        *,
        training_root: str | os.PathLike[str],
        library_root: str | os.PathLike[str],
    ) -> None:
        """Validate canonical roots before any evaluation-library mutation."""

        configured_training = Path(training_root).expanduser().absolute()
        configured_library = Path(library_root).expanduser().absolute()
        canonical_training = configured_training.resolve(strict=False)
        canonical_library = configured_library.resolve(strict=False)
        if canonical_training == Path(canonical_training.anchor):
            raise EvaluationLibraryPathViolation("The training root is too broad.")
        if canonical_library == Path(canonical_library.anchor):
            raise EvaluationLibraryPathViolation("The evaluation-library root is too broad.")
        if _contains(canonical_training, canonical_library) or _contains(
            canonical_library,
            canonical_training,
        ):
            raise EvaluationLibraryPathViolation(
                "Evaluation-library storage must be disjoint from the training scenario root."
            )
        object.__setattr__(self, "training_root", configured_training)
        object.__setattr__(self, "library_root", configured_library)
        object.__setattr__(self, "_canonical_training_root", canonical_training)
        object.__setattr__(self, "_canonical_library_root", canonical_library)

    def library_path(self, library_id: str) -> Path:
        """Return one validated immutable-library target below ``library_root``."""

        try:
            normalized_id = validate_library_id(library_id)
        except ValueError as exc:
            raise EvaluationLibraryPathViolation(str(exc)) from exc
        candidate = self.library_root / normalized_id
        if candidate.parent != self.library_root:
            raise EvaluationLibraryPathViolation("Library path escapes its configured root.")
        return candidate

    def open_library_root(self, *, create: bool) -> int:
        """Open and pin the configured root without following path components."""

        try:
            descriptor = _open_directory_chain(self.library_root, create=create)
        except OSError as exc:
            raise EvaluationLibraryPathViolation(
                f"Cannot securely open evaluation-library root: {self.library_root!s}."
            ) from exc
        try:
            self.require_current_root(descriptor)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    def require_current_root(self, descriptor: int) -> None:
        """Fail when a configured root no longer names the pinned directory."""

        try:
            path_stat = self.library_root.lstat()
            descriptor_stat = os.fstat(descriptor)
            current_library = self.library_root.resolve(strict=True)
            current_training = self.training_root.resolve(strict=True)
        except OSError as exc:
            raise EvaluationLibraryPathViolation("Evaluation-library root changed.") from exc
        if self.library_root.is_symlink() or (
            path_stat.st_dev,
            path_stat.st_ino,
        ) != (descriptor_stat.st_dev, descriptor_stat.st_ino):
            raise EvaluationLibraryPathViolation("Evaluation-library root changed.")
        if current_library != self._canonical_library_root:
            raise EvaluationLibraryPathViolation("Evaluation-library root changed.")
        if current_training != self._canonical_training_root:
            raise EvaluationLibraryPathViolation("Training scenario root changed.")
        if _contains(current_training, current_library) or _contains(
            current_library,
            current_training,
        ):
            raise EvaluationLibraryPathViolation(
                "Evaluation-library storage now overlaps the training scenario root."
            )


__all__ = [
    "EvaluationLibraryPathViolation",
    "EvaluationLibraryStorageConfiguration",
]
