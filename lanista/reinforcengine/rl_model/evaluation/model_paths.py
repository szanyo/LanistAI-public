# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Resolve held-out library and run roots beneath one selected model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryPathViolation,
    EvaluationLibraryStorageConfiguration,
)


def _absolute_unaliased(path: Path, *, label: str) -> Path:
    """Reject relative, traversing, or symlink-aliased storage identities."""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise EvaluationLibraryPathViolation(f"{label} must be an absolute, direct path.")
    if candidate.resolve(strict=False) != candidate:
        raise EvaluationLibraryPathViolation(f"{label} contains a symbolic-link alias.")
    return candidate


def _require_no_existing_symlinks(root: Path, child: Path) -> None:
    """Check every existing child component without creating directories."""

    try:
        relative = child.relative_to(root)
    except ValueError as exc:
        raise EvaluationLibraryPathViolation("Evaluation path escapes its model root.") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise EvaluationLibraryPathViolation(
                f"Evaluation path contains a symbolic link: {current!s}."
            )


@dataclass(frozen=True, slots=True)
class ModelEvaluationPaths:
    """Carry safe, model-local held-out library and run locations."""

    model_root: Path
    library_storage: EvaluationLibraryStorageConfiguration
    run_parent: Path

    @classmethod
    def resolve(
        cls,
        *,
        model_root: Path,
        models_root: Path,
        training_root: Path,
    ) -> ModelEvaluationPaths:
        """Validate one real model child and its disjoint scenario roots."""

        models = _absolute_unaliased(models_root, label="Models root")
        model = _absolute_unaliased(model_root, label="Model root")
        training = _absolute_unaliased(training_root, label="Training root")
        if model.parent != models or not model.is_dir():
            raise EvaluationLibraryPathViolation(
                "The selected model must be an existing direct child of the models root."
            )
        if not training.is_relative_to(model) or not training.is_dir():
            raise EvaluationLibraryPathViolation(
                "The training scenario root must exist beneath the selected model."
            )
        library_root = model / "resources" / "evaluation_libraries"
        run_parent = model / "tensorboard" / "evaluation" / "libraries"
        for path in (training, library_root, run_parent):
            _require_no_existing_symlinks(model, path)
        storage = EvaluationLibraryStorageConfiguration(
            training_root=training,
            library_root=library_root,
        )
        if training.is_relative_to(run_parent) or run_parent.is_relative_to(training):
            raise EvaluationLibraryPathViolation(
                "Evaluation runs must be disjoint from training scenarios."
            )
        return cls(model_root=model, library_storage=storage, run_parent=run_parent)

    def require_current_roots(self) -> None:
        """Recheck all configured parents before a later read or write."""

        if not self.model_root.is_dir():
            raise EvaluationLibraryPathViolation("Selected model root changed.")
        for path in (
            self.library_storage.training_root,
            self.library_storage.library_root,
            self.run_parent,
        ):
            _require_no_existing_symlinks(self.model_root, path)
        if self.model_root.resolve(strict=False) != self.model_root:
            raise EvaluationLibraryPathViolation("Selected model root changed.")


__all__ = ["ModelEvaluationPaths"]
