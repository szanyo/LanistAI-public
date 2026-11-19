# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral facade for Susan episode lifecycle decisions."""

from __future__ import annotations

import logging

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import ScenarioManager
from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_curriculum_state import (
    AbstractCurriculumState,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_scenario_state import (
    AbstractScenarioState,
)

from .episode_service import EpisodeLifecycleService
from .episode_state_reader import (
    EpisodeStateReader,
    LocalEpisodeStateReader,
)


class EpisodeController:
    """Coordinate episode operations."""

    def __init__(
        self,
        *,
        episode_state_reader: EpisodeStateReader | None = None,
        **kwargs,
    ):
        """Initialize the episode controller and its owned state."""
        self._logger = logging.getLogger()
        self._scenario_index_state: AbstractScenarioState = kwargs.get("scenario_index_state")
        self._curriculum_index_state: AbstractCurriculumState = kwargs.get("curriculum_index_state")
        self.scenario_manager: ScenarioManager = kwargs.get("scenario_manager")

        if self.scenario_manager is not None:
            self._scenario_manager_configuration: Configuration = self.scenario_manager.get_configuration()
        else:
            self._scenario_manager_configuration: Configuration = None

        self.episode_counter = 0
        self.episode_reward_counter = 0
        self.cumulative_step_counter_over_episodes = 0
        self.step_counter = 0
        self.episode_terminated = False
        self.episode_truncated = False
        self.last_termination_reason = None
        self.last_truncation_reason = None
        self.episode_end_conflict = False
        self.episode_end_conflict_target_time_limit = False
        self.episode_end_conflict_target_red_light_overdrive = False
        self.curriculum_target_detected = False
        self.curriculum_target_accepted = False
        self.curriculum_target_rejected_by_red_light_overdrive = False
        self.curriculum_target_reached = False
        self._episode_start_simulation_time_seconds = 0.0
        self._episode_active_step_limit = None
        self._episode_state_reader = (
            episode_state_reader
            if episode_state_reader is not None
            else LocalEpisodeStateReader()
        )
        self._episode_lifecycle_service = EpisodeLifecycleService(
            self,
            self._episode_state_reader,
        )

    def reset(self):
        """Reset this episode controller to its initial operational state."""
        self._episode_lifecycle_service.reset()

    def mark_episode_start_after_warmup(self, sumo_simulation_controller, active_step_limit=None):
        """
        Mark the active policy episode start after SUMO warmup.

        SUMO simulation time includes reset warmup steps, while the RL episode
        time limit should only count steps controlled by the policy.
        """
        self._episode_lifecycle_service.mark_episode_start_after_warmup(
            sumo_simulation_controller,
            active_step_limit=active_step_limit,
        )

    def step(
        self,
        sumo_simulation_controller,
        scenario_context_preprocessor,
        ego_vehicle,
        cumulative_reward_counter_over_episodes,
    ):
        """Advance the workflow owned by this episode controller by one step."""
        return self._episode_lifecycle_service.step(
            sumo_simulation_controller,
            scenario_context_preprocessor,
            ego_vehicle,
            cumulative_reward_counter_over_episodes,
        )

    def _termination_check(
        self,
        sumo_simulation_controller,
        scenario_context_preprocessor,
        ego_vehicle,
    ):
        """Return the termination check value owned by this episode controller."""
        return self._episode_lifecycle_service.termination_check(
            sumo_simulation_controller,
            scenario_context_preprocessor,
            ego_vehicle,
        )

    def _truncation_check(self, sumo_simulation_controller):
        """Return the truncation check value owned by this episode controller."""
        return self._episode_lifecycle_service.truncation_check(sumo_simulation_controller)
