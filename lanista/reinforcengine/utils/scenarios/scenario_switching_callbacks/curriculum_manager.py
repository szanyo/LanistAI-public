# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the lifecycle of curriculum."""

from typing import Any, Dict, List, Optional

from lanista.foundation.logging import LoggerInterface

from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    CurriculumStateMachine,
    CurriculumDefinition,
    curriculum_state_label,
    decorate_curriculum_states,
    ensure_full_curriculum_state,
    is_full_curriculum_state,
    published_curriculum_state,
    resolve_curriculum_definition,
)

__all__ = [
    "CurriculumManager",
    "CurriculumDefinition",
    "CurriculumStateMachine",
    "curriculum_state_label",
    "decorate_curriculum_states",
    "ensure_full_curriculum_state",
    "is_full_curriculum_state",
    "published_curriculum_state",
    "resolve_curriculum_definition",
]


class CurriculumManager(LoggerInterface):
    """Framework-neutral curriculum manager for local state backends."""

    def __init__(self, states: List[Dict[str, Any]], curriculum_index_state, **kwargs):
        """Initialize the curriculum manager and its owned state."""
        base_min_terminated_episodes = kwargs.pop("base_min_terminated_episodes", 0)
        super().__init__(**kwargs)
        self._state_machine = CurriculumStateMachine(
            states or [],
            base_min_terminated_episodes=base_min_terminated_episodes,
        )
        self._curriculum_index_state = curriculum_index_state

    def current_state(self) -> Optional[Dict[str, Any]]:
        """Return the current state value owned by this curriculum manager."""
        return self._state_machine.current_state()

    def _publish_state(self, state: Optional[Dict[str, Any]]):
        """Publish the state owned by this curriculum manager."""
        self._curriculum_index_state.set(published_curriculum_state(state))

    def initialize(self):
        """Initialize the operation owned by this curriculum manager."""
        self._publish_state(self.current_state())
        self._logger.info(f"[Curriculum] Initial state: {curriculum_state_label(self.current_state())}")

    def reset(self) -> None:
        """Reset this curriculum manager to its initial operational state."""
        self._publish_state(self._state_machine.reset())
        self._logger.info("[Curriculum] Reset to initial state.")

    def is_unlocked(self) -> bool:
        """Return whether unlocked applies to this curriculum manager."""
        return self._state_machine.is_unlocked()

    def reset_for_scenario_switch(self) -> bool:
        """Reset the for scenario switch owned by this curriculum manager."""
        if self.is_unlocked():
            self._publish_state(self.current_state())
            self._logger.info("[Curriculum] Scenario switch keeps unlocked/full curriculum state.")
            return False
        self.reset()
        return True

    def restore_state(self, state: Optional[Dict[str, Any]]) -> None:
        """Restore the state owned by this curriculum manager."""
        result = self._state_machine.restore(state)
        self._publish_state(result.publication_state)
        if not result.recognized:
            self._logger.warning(
                f"[Curriculum] Restored unknown checkpoint state: {state}"
            )

    def evaluate_and_update(self, ready: bool) -> None:
        """Evaluate the and update owned by this curriculum manager."""
        transition = self._state_machine.advance(ready)
        self._publish_state(transition.current_state)
        if transition.changed:
            self._logger.info(
                "[Curriculum] Advanced from "
                f"{curriculum_state_label(transition.previous_state)} to "
                f"{curriculum_state_label(transition.current_state)}."
            )
