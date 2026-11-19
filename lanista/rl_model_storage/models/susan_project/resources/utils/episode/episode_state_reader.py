# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral reads of state shared across Susan episodes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, cast

CurriculumState = Mapping[str, object]


class EpisodeStateReader(Protocol):
    """Read the state values needed by the shared episode lifecycle."""

    def read_scenario_index(self, state: object | None) -> int | None:
        """Return a valid scenario index, or ``None`` for absent/malformed state."""

    def read_curriculum_state(
        self,
        state: object | None,
    ) -> CurriculumState | None:
        """Return valid curriculum metadata, or ``None`` when unavailable."""


def validate_scenario_index(value: object | None) -> int | None:
    """Accept only an integer scenario index; booleans are not indices."""

    return value if type(value) is int else None


def validate_curriculum_state(
    value: object | None,
) -> CurriculumState | None:
    """Accept mappings with string keys as curriculum state."""

    if not isinstance(value, Mapping):
        return None
    if not all(isinstance(key, str) for key in value):
        return None
    return cast(CurriculumState, value)


class LocalEpisodeStateReader:
    """Read in-process SB3/direct state objects without hiding read failures."""

    @staticmethod
    def _read_value(state: object | None) -> object | None:
        """Read the value without changing this local episode state reader's state."""
        if state is None:
            return None
        getter = getattr(state, "get", None)
        if not callable(getter):
            return None
        return cast(object, getter())

    def read_scenario_index(self, state: object | None) -> int | None:
        """Read and validate an in-process scenario index."""

        return validate_scenario_index(self._read_value(state))

    def read_curriculum_state(
        self,
        state: object | None,
    ) -> CurriculumState | None:
        """Read and validate in-process curriculum metadata."""

        return validate_curriculum_state(self._read_value(state))
