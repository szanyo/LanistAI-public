# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the RLlib curriculum lifecycle."""

from typing import Any, Dict, List, Optional

from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    CurriculumStateMachine,
    published_curriculum_state,
)


class RayCurriculumManager(LoggerInterface):
    """Publish the shared curriculum state machine through a Ray actor."""

    def __init__(self, states: List[Dict[str, Any]], curriculum_index_state, **kwargs):
        """Initialize the ray curriculum manager and its owned state."""
        base_min_terminated_episodes = kwargs.pop("base_min_terminated_episodes", 0)
        super().__init__(**kwargs)
        self._state_machine = CurriculumStateMachine(
            states or [],
            base_min_terminated_episodes=base_min_terminated_episodes,
        )
        self._curriculum_index_state = curriculum_index_state

    def current_state(self) -> Optional[Dict[str, Any]]:
        """Return the current state value owned by this ray curriculum manager."""
        return self._state_machine.current_state()

    def _publish_state(self, state: Optional[Dict[str, Any]]):
        """Publish the state owned by this ray curriculum manager."""
        self._curriculum_index_state.set.remote(
            published_curriculum_state(state)
        )

    def initialize(self):
        """Initialize the operation owned by this ray curriculum manager."""
        self._publish_state(self.current_state())

    def reset(self) -> None:
        """Reset this ray curriculum manager to its initial operational state."""
        self._publish_state(self._state_machine.reset())
        self._logger.info("[Curriculum] Reset to initial state.")

    def is_unlocked(self) -> bool:
        """Return whether unlocked applies to this ray curriculum manager."""
        return self._state_machine.is_unlocked()

    def reset_for_scenario_switch(self) -> bool:
        """Reset the for scenario switch owned by this ray curriculum manager."""
        if self.is_unlocked():
            self._publish_state(self.current_state())
            self._logger.info("[Curriculum] Scenario switch keeps unlocked/full curriculum state.")
            return False
        self.reset()
        return True

    def restore_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore the state owned by this ray curriculum manager."""
        result = self._state_machine.restore(state)
        if result.recognized:
            self._publish_state(result.publication_state)
        else:
            # Preserve the exact unknown payload for checkpoint compatibility.
            self._curriculum_index_state.set.remote(state)
            self._logger.warning(
                f"[Curriculum] Restored unknown checkpoint state: {state}"
            )

    def evaluate_and_update(self, ready: bool) -> None:
        """Evaluate the and update owned by this ray curriculum manager."""
        transition = self._state_machine.advance(ready)
        self._publish_state(transition.current_state)
