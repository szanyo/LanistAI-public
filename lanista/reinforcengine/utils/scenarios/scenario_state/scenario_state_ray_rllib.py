# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track scenario state for RLlib environments."""

import ray
from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_scenario_state import AbstractScenarioState


@ray.remote
class RayScenarioState(AbstractScenarioState):
    """Ray Actor backend multi-worker, multi-agent környezethez."""

    def __init__(self):
        """Initialize the ray scenario state and its owned state."""
        self._current = 0

    def get(self) -> int:
        """Return the get value owned by this ray scenario state."""
        return self._current

    def set(self, value: int):
        """Set the operation owned by this ray scenario state."""
        self._current = value


