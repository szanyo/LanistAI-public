# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track curriculum state for RLlib training."""

import ray

from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_curriculum_state import AbstractCurriculumState


@ray.remote
class RayCurriculumIndexState(AbstractCurriculumState):
    """Ray Actor backend a curriculum cél (pl. intersection target) megosztására."""

    def __init__(self):
        """Initialize the ray curriculum index state and its owned state."""
        self._target_intersection = None

    def get(self):
        """Return the get value owned by this ray curriculum index state."""
        return self._target_intersection

    def set(self, value):
        """Set the operation owned by this ray curriculum index state."""
        self._target_intersection = value
