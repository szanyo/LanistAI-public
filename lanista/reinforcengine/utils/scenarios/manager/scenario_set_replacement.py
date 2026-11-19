# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Staged replacement of one generated scenario inventory."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self


class ScenarioSetReplacement:
    """Build a complete scenario set before replacing the published inventory."""

    def __init__(self, target_root: str | os.PathLike[str]) -> None:
        """Initialize the scenario set replacement and its owned state."""
        self._target_root = Path(target_root).absolute()
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._workspace: Path | None = None
        self._staging_root: Path | None = None
        self._backup_root: Path | None = None
        self._published = False

    @property
    def staging_root(self) -> Path:
        """Return the staging root exposed by this scenario set replacement."""
        if self._staging_root is None:
            raise RuntimeError("Scenario replacement workspace is not active.")
        return self._staging_root

    def __enter__(self) -> Self:
        """Enter this scenario set replacement's managed context."""
        target = self._target_root
        if target.is_symlink():
            raise RuntimeError("Scenario inventory root must not be a symbolic link.")
        if os.path.lexists(target) and not target.is_dir():
            raise NotADirectoryError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_directory = tempfile.TemporaryDirectory(
            prefix=f".{target.name}.generation-",
            dir=target.parent,
        )
        self._workspace = Path(self._temporary_directory.name)
        self._staging_root = self._workspace / "replacement"
        self._staging_root.mkdir()
        self._backup_root = self._workspace / "previous"
        return self

    def publish(self) -> None:
        """Replace the target only after the complete staged set exists."""

        staging = self.staging_root
        workspace = self._workspace
        backup = self._backup_root
        if workspace is None or backup is None:
            raise RuntimeError("Scenario replacement workspace is not active.")
        if self._published:
            raise RuntimeError("Scenario replacement was already published.")

        previous_inventory = os.path.lexists(self._target_root)
        if previous_inventory:
            os.replace(self._target_root, backup)
        try:
            os.replace(staging, self._target_root)
        except BaseException:
            if previous_inventory and os.path.lexists(backup):
                os.replace(backup, self._target_root)
            raise
        self._published = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release owned resources when leaving this scenario set replacement's managed context."""
        backup = self._backup_root
        if (
            not self._published
            and backup is not None
            and os.path.lexists(backup)
            and not os.path.lexists(self._target_root)
        ):
            os.replace(backup, self._target_root)
        temporary_directory = self._temporary_directory
        if temporary_directory is not None:
            temporary_directory.cleanup()
