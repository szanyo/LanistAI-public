# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Root-confined Ray checkpoint discovery and selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from lanista.reinforcengine.rl_model.storage.names import CheckpointStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard

_CHECKPOINT_STEP_PATTERN = re.compile(
    r"_checkpoint_([0-9]+(?:\.[0-9]+)?)_steps$"
)


class RayCheckpointSource(StrEnum):
    """Describe how a Ray checkpoint was selected."""

    EXPLICIT_FINAL = "explicit_final"
    EXPLICIT_INTERMEDIATE = "explicit_intermediate"
    LATEST_FINAL = "latest_final"
    LATEST_INTERMEDIATE = "latest_intermediate"


@dataclass(frozen=True, slots=True)
class RayCheckpointSelection:
    """One validated Ray checkpoint directory below a framework root."""

    path: Path
    source: RayCheckpointSource
    checkpoint_name: CheckpointStorageName

    @property
    def is_intermediate(self) -> bool:
        """Return whether continuation metadata should name this checkpoint."""

        return self.source in {
            RayCheckpointSource.EXPLICIT_INTERMEDIATE,
            RayCheckpointSource.LATEST_INTERMEDIATE,
        }


class RayCheckpointResolver:
    """Select Ray checkpoints without accepting paths outside managed roots."""

    def __init__(
        self,
        *,
        final_directory: str | Path,
        checkpoint_directory: str | Path,
    ) -> None:
        """Initialize the ray checkpoint resolver and its owned state."""
        self._final_guard = StoragePathGuard(final_directory)
        self._checkpoint_guard = StoragePathGuard(checkpoint_directory)

    def resolve(
        self,
        checkpoint_name: str | CheckpointStorageName | None = None,
    ) -> RayCheckpointSelection | None:
        """Resolve an explicit name or auto-detect the newest valid directory."""

        if checkpoint_name is not None:
            validated_name = CheckpointStorageName.parse(checkpoint_name)
            return self._resolve_explicit(validated_name)

        latest_final = self._latest_final()
        if latest_final is not None:
            return latest_final
        return self._latest_intermediate()

    def _resolve_explicit(
        self,
        checkpoint_name: CheckpointStorageName,
    ) -> RayCheckpointSelection | None:
        """Resolve the explicit required by this ray checkpoint resolver."""
        final_candidate = self._resolve_nonempty_directory(
            self._final_guard,
            checkpoint_name,
        )
        if final_candidate is not None:
            return RayCheckpointSelection(
                path=final_candidate,
                source=RayCheckpointSource.EXPLICIT_FINAL,
                checkpoint_name=checkpoint_name,
            )

        checkpoint_candidate = self._resolve_nonempty_directory(
            self._checkpoint_guard,
            checkpoint_name,
        )
        if checkpoint_candidate is None:
            return None
        return RayCheckpointSelection(
            path=checkpoint_candidate,
            source=RayCheckpointSource.EXPLICIT_INTERMEDIATE,
            checkpoint_name=checkpoint_name,
        )

    def _latest_final(self) -> RayCheckpointSelection | None:
        """Return the latest final value owned by this ray checkpoint resolver."""
        candidates = self._nonempty_children(self._final_guard)
        if not candidates:
            return None
        name, path = max(
            candidates,
            key=lambda item: (item[1].stat().st_mtime_ns, item[0].value),
        )
        return RayCheckpointSelection(
            path=path,
            source=RayCheckpointSource.LATEST_FINAL,
            checkpoint_name=name,
        )

    def _latest_intermediate(self) -> RayCheckpointSelection | None:
        """Return the latest intermediate value owned by this ray checkpoint resolver."""
        candidates = self._nonempty_children(self._checkpoint_guard)
        if not candidates:
            return None
        name, path = max(
            candidates,
            key=lambda item: (
                self._step_from_name(item[0].value),
                item[1].stat().st_mtime_ns,
                item[0].value,
            ),
        )
        return RayCheckpointSelection(
            path=path,
            source=RayCheckpointSource.LATEST_INTERMEDIATE,
            checkpoint_name=name,
        )

    @staticmethod
    def _resolve_nonempty_directory(
        guard: StoragePathGuard,
        checkpoint_name: CheckpointStorageName,
    ) -> Path | None:
        """Resolve the nonempty directory required by this ray checkpoint resolver."""
        candidate = guard.resolve(checkpoint_name)
        if not candidate.is_dir():
            return None
        try:
            next(candidate.iterdir())
        except StopIteration:
            return None
        return candidate

    def _nonempty_children(
        self,
        guard: StoragePathGuard,
    ) -> list[tuple[CheckpointStorageName, Path]]:
        """Return the nonempty children value owned by this ray checkpoint resolver."""
        if not guard.root.is_dir():
            return []

        candidates: list[tuple[CheckpointStorageName, Path]] = []
        for raw_name in sorted(entry.name for entry in guard.root.iterdir()):
            checkpoint_name = CheckpointStorageName.parse(raw_name)
            candidate = self._resolve_nonempty_directory(
                guard,
                checkpoint_name,
            )
            if candidate is not None:
                candidates.append((checkpoint_name, candidate))
        return candidates

    @staticmethod
    def _step_from_name(checkpoint_name: str) -> int:
        """Return the step from name value owned by this ray checkpoint resolver."""
        match = _CHECKPOINT_STEP_PATTERN.search(checkpoint_name)
        if match is None:
            return -1
        return int(float(match.group(1)))
