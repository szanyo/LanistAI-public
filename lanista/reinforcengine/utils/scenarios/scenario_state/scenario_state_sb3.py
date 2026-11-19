# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track scenario state for Stable-Baselines3 environments."""

from multiprocessing import Lock, Value
from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_scenario_state import AbstractScenarioState

class StableBaselinesScenarioState(AbstractScenarioState):
    """Represent runtime state for stable baselines scenario."""

    def __init__(self):
        """Initialize the stable baselines scenario state and its owned state."""
        self._current = Value("i", 0)
        self._lock: Lock = Lock()

    def get(self) -> int:
        """Return the get value owned by this stable baselines scenario state."""
        return self._current.value

    def set(self, value: int):
        """Set the operation owned by this stable baselines scenario state."""
        with self._lock:
            self._current.value = value
