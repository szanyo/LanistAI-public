# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Deterministic scenario-scaled warmup jitter policy."""

from __future__ import annotations

from typing import Protocol

from ..episode.episode_state_reader import (
    EpisodeStateReader,
    LocalEpisodeStateReader,
)

MIN_SCENARIO_SCALED_WARMUP_JITTER_SECONDS = 10.0


class WarmupScenarioManager(Protocol):
    """Scenario information needed to scale reset warmup jitter."""

    def get_total_number_of_scenarios(self) -> int:
        """Return the number of configured scenarios."""

    def get_index(self) -> int:
        """Return the locally active scenario index."""


def scenario_scaled_warmup_jitter_seconds(
    max_jitter_seconds: float,
    scenario_manager: WarmupScenarioManager,
    scenario_index_state: object | None = None,
    *,
    state_reader: EpisodeStateReader | None = None,
) -> float:
    """Scale warmup jitter from a useful starting floor to the configured maximum."""
    try:
        max_jitter_seconds = max(0.0, float(max_jitter_seconds))
    except (TypeError, ValueError):
        max_jitter_seconds = 0.0
    if max_jitter_seconds <= 0:
        return 0.0

    try:
        scenario_count = int(scenario_manager.get_total_number_of_scenarios())
    except Exception:
        scenario_count = 1
    scenario_count = max(1, scenario_count)
    if scenario_count <= 1:
        return max_jitter_seconds

    reader = (
        state_reader
        if state_reader is not None
        else LocalEpisodeStateReader()
    )
    scenario_index = reader.read_scenario_index(scenario_index_state)
    if scenario_index is None:
        try:
            scenario_index = scenario_manager.get_index()
        except Exception:
            scenario_index = 0

    try:
        scenario_index = int(scenario_index)
    except (TypeError, ValueError):
        scenario_index = 0
    scenario_index = min(max(0, scenario_index), scenario_count - 1)

    min_jitter_seconds = min(MIN_SCENARIO_SCALED_WARMUP_JITTER_SECONDS, max_jitter_seconds)
    progress = scenario_index / max(1, scenario_count - 1)
    return min_jitter_seconds + (max_jitter_seconds - min_jitter_seconds) * progress
