# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Ray adapter for Susan's framework-neutral episode-state contract."""

from __future__ import annotations

from typing import cast

import ray

from ..utils.episode.episode_state_reader import (
    CurriculumState,
    validate_curriculum_state,
    validate_scenario_index,
)


class RayEpisodeStateReader:
    """Resolve Ray actor state while preserving mandatory read failures."""

    @staticmethod
    def _read_value(state: object | None) -> object | None:
        """Read the value without changing this ray episode state reader's state."""
        if state is None:
            return None
        getter = getattr(state, "get", None)
        remote_getter = getattr(getter, "remote", None)
        if callable(remote_getter):
            return cast(object, ray.get(remote_getter()))
        if callable(getter):
            return cast(object, getter())
        return None

    def read_scenario_index(self, state: object | None) -> int | None:
        """Resolve and validate a remote scenario index."""

        return validate_scenario_index(self._read_value(state))

    def read_curriculum_state(
        self,
        state: object | None,
    ) -> CurriculumState | None:
        """Resolve and validate remote curriculum metadata."""

        return validate_curriculum_state(self._read_value(state))
