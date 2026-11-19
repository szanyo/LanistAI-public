# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model-neutral episode counters, outcomes, and terminal decisions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import ray
from lanista.reinforcengine.running_mode import RunningMode
from lanista.reinforcengine.utils.simulation.episode_end import (
    EgoVehiclePresence,
    observe_ego_vehicle,
)
from lanista.reinforcengine.utils.simulation.ports import VehiclePort

EGO_AGENT_ID = "EGO_agent_0"
ALL_AGENTS_ID = "__all__"


class EpisodeScenarioConfiguration(Protocol):
    """Scenario configuration values consumed by episode lifecycle."""

    sumo_configuration: object | None

    def get_max_timesteps(self) -> float:
        """Return the max timesteps exposed by this episode scenario configuration."""
        ...


class EpisodeScenarioManager(Protocol):
    """Scenario manager operations used by the lifecycle controller."""

    def get_configuration(self) -> EpisodeScenarioConfiguration:
        """Return the configuration exposed by this episode scenario manager."""
        ...

    def get_index(self) -> object:
        """Return the index exposed by this episode scenario manager."""
        ...

    def set_index(self, index: object) -> None:
        """Set the index owned by this episode scenario manager."""
        ...

    def get_cumulative_training_counter(self) -> int:
        """Return the cumulative training counter exposed by this episode scenario manager."""
        ...


class EpisodeSimulator(VehiclePort, Protocol):
    """Simulator counters and clock needed when an episode ends."""

    def is_initialized(self) -> bool:
        """Return whether initialized applies to this episode simulator."""
        ...

    def get_simulation_time(self) -> float:
        """Return the simulation time exposed by this episode simulator."""
        ...


class ScenarioContextView(Protocol):
    """Scenario values required by terminal decisions and logging."""

    intersection_number: int
    ego_road_id: str | int


class EpisodeEgoVehicle(Protocol):
    """EGO state retained between episode lifecycle checks."""

    last_ut: str | int
    MaxSpeed: float
    AverageSpeed: float


@dataclass(frozen=True, slots=True)
class EpisodeLifecycleConfiguration:
    """Immutable route and simulator-time semantics."""

    route_end_road_id: str = "6321.56.41"
    ego_vehicle_id: str = "EGO"
    simulation_time_scale: float = 20.0


@dataclass(frozen=True, slots=True)
class EpisodeAgentOutcome:
    """Per-agent and aggregate terminal flags for one framework step."""

    terminated: bool
    truncated: bool

    def as_framework_mappings(
        self,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Return RLlib-compatible terminated and truncated mappings."""

        return (
            {
                EGO_AGENT_ID: self.terminated,
                ALL_AGENTS_ID: self.terminated,
            },
            {
                EGO_AGENT_ID: self.truncated,
                ALL_AGENTS_ID: self.truncated and not self.terminated,
            },
        )


class EpisodeLifecycleController:
    """Own episode counters and model-independent end-of-episode decisions."""

    _mode: RunningMode

    def __init__(
        self,
        *,
        configuration: EpisodeLifecycleConfiguration | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize the episode lifecycle controller and its owned state."""
        self._logger = logging.getLogger()
        self._scenario_index_state = kwargs.get("scenario_index_state")
        self._curriculum_index_state = kwargs.get("curriculum_index_state")
        self.scenario_manager = cast(
            EpisodeScenarioManager | None,
            kwargs.get("scenario_manager"),
        )
        self._scenario_manager_configuration = (
            self.scenario_manager.get_configuration() if self.scenario_manager is not None else None
        )
        self._episode_configuration = configuration or EpisodeLifecycleConfiguration()
        self.episode_counter = 0
        self.episode_reward_counter = 0.0
        self.cumulative_step_counter_over_episodes = 0
        self.step_counter = 0
        self.episode_terminated = False
        self.episode_truncated = False

    @staticmethod
    def _state_value(state: object | None) -> object | None:
        """Return the state value value owned by this episode lifecycle controller."""
        if state is None:
            return None
        getter = getattr(state, "get", None)
        if not callable(getter):
            return None
        remote_getter = getattr(getter, "remote", None)
        if callable(remote_getter):
            return cast(object, ray.get(remote_getter()))
        return cast(object, getter())

    def reset(self) -> None:
        """Validate scenario configuration and start fresh episode counters."""

        manager = self.scenario_manager
        configuration = self._scenario_manager_configuration
        if manager is None:
            raise ValueError("Scenario manager is not set.")
        if configuration is None or configuration.sumo_configuration is None:
            raise ValueError("SUMO initial configuration is not set.")
        state_index = self._state_value(self._scenario_index_state)
        if state_index is not None and manager.get_index() != state_index:
            manager.set_index(state_index)

        self.episode_counter += 1
        self.cumulative_step_counter_over_episodes += self.step_counter
        self.step_counter = 0
        self.episode_reward_counter = 0
        self.episode_terminated = False
        self.episode_truncated = False
        self._reset_optional_violation_fields()

    def _reset_optional_violation_fields(self) -> None:
        """Reset the optional violation fields owned by this episode lifecycle controller."""
        for field_name in (
            "red_light_crossing_truncation",
            "red_light_crossing",
            "red_light_violation",
        ):
            if hasattr(self, field_name):
                setattr(self, field_name, 0.0)

    def step(
        self,
        sumo_simulation_controller: EpisodeSimulator,
        scenario_context_preprocessor: ScenarioContextView,
        ego_vehicle: EpisodeEgoVehicle,
        cumulative_reward_counter_over_episodes: float,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Advance counters, report a finished episode, and expose flags."""

        self.step_counter += 1
        if self.episode_terminated or self.episode_truncated:
            self._log_finished_episode(
                sumo_simulation_controller,
                scenario_context_preprocessor,
                ego_vehicle,
                cumulative_reward_counter_over_episodes,
            )
        return EpisodeAgentOutcome(
            terminated=self.episode_terminated,
            truncated=self.episode_truncated,
        ).as_framework_mappings()

    def _log_finished_episode(
        self,
        simulator: EpisodeSimulator,
        scenario_context: ScenarioContextView,
        ego_vehicle: EpisodeEgoVehicle,
        cumulative_reward: float,
    ) -> None:
        """Log the finished episode owned by this episode lifecycle controller."""
        manager = cast(EpisodeScenarioManager, self.scenario_manager)
        self._logger.info(
            "RUNNING SCENARIO INDEX:     %s\n"
            "TRAINING ITERATION COUNTER: %s\n"
            "EPISODE COUNTER:            %s\n"
            "CUMULATIVE STEP COUNTER:    %s\n"
            "VEHICLE COUNTER (EGO TOO):  %s\n"
            "PEDESTRIAN COUNTER:         %s\n"
            "REACHED INTERSECTION NUMBER:%s\n"
            "REACHED ROAD ID:            %s\n"
            "CUMULATIVE REWARD:          %s\n"
            "EPISODE REWARD:             %s\n"
            "MAX SPEED:                  %s m/s\n"
            "AVERAGE SPEED:              %s m/s\n"
            "TERMINATED:                 %s\n"
            "TRUNCATED:                  %s",
            manager.get_index(),
            manager.get_cumulative_training_counter(),
            self.episode_counter,
            self.cumulative_step_counter_over_episodes,
            simulator.get_vehicle_id_count(),
            simulator.get_pedestrian_id_count(),
            scenario_context.intersection_number,
            scenario_context.ego_road_id,
            cumulative_reward,
            self.episode_reward_counter,
            ego_vehicle.MaxSpeed,
            ego_vehicle.AverageSpeed,
            self.episode_terminated,
            self.episode_truncated,
        )
        self._logger.info("---Episode finished---")

    def _termination_check(
        self,
        sumo_simulation_controller: EpisodeSimulator,
        scenario_context_preprocessor: ScenarioContextView,
        ego_vehicle: EpisodeEgoVehicle,
    ) -> bool:
        """Return whether curriculum, route end, or confirmed absence ended the episode."""

        if self._curriculum_target_reached(scenario_context_preprocessor):
            return True
        if not sumo_simulation_controller.is_initialized():
            self._logger.warning("Sumo is not initialized. Termination check skipped.")
            return False
        observation = observe_ego_vehicle(
            sumo_simulation_controller,
            previous_road_id=cast(str | None, ego_vehicle.last_ut),
            route_end_road_id=self._episode_configuration.route_end_road_id,
            vehicle_id=self._episode_configuration.ego_vehicle_id,
        )
        if observation.presence is EgoVehiclePresence.ROUTE_END:
            self._logger.info(
                "TERM_COND4 - Training episode terminated properly. "
                "EGO vehicle reached the end of its route. =)"
            )
            return True
        if observation.presence is EgoVehiclePresence.UNAVAILABLE:
            self._logger.warning("EGO vehicle absence/collision confirmed. Episode will terminate.")
            return True
        ego_vehicle.last_ut = cast(str, observation.road_id)
        return False

    def _curriculum_target_reached(
        self,
        scenario_context: ScenarioContextView,
    ) -> bool:
        """Return the curriculum target reached value owned by this episode lifecycle controller."""
        if self._mode is not RunningMode.TRAINING or self._curriculum_index_state is None:
            return False
        target_value = self._state_value(self._curriculum_index_state)
        if target_value is None:
            return False
        target = cast(Mapping[str, object], target_value)
        if target.get("target_type") != "intersection":
            return False
        intersection_target = cast(int, target.get("target_value"))
        if scenario_context.intersection_number < intersection_target:
            return False
        target_name = target.get("state_name")
        self._logger.info(
            "TERM_CURRICULUM (%s) - Target intersection reached: %s >= %s",
            target_name,
            scenario_context.intersection_number,
            intersection_target,
        )
        return True

    def _truncation_check(
        self,
        sumo_simulation_controller: EpisodeSimulator,
    ) -> bool:
        """Return whether time or a red-light crossing truncated the episode."""

        manager = cast(EpisodeScenarioManager, self.scenario_manager)
        max_timesteps = manager.get_configuration().get_max_timesteps()
        simulation_time = (
            sumo_simulation_controller.get_simulation_time()
            * self._episode_configuration.simulation_time_scale
        )
        if simulation_time >= max_timesteps:
            self._logger.warning(
                "TRUNC_COND - Truncated because simulation time is up. (%s >= limit(%s) s)",
                simulation_time,
                max_timesteps,
            )
            return True
        if getattr(self, "red_light_crossing_truncation", 0.0) == 1.0:
            self._logger.warning(
                "TRUNC_COND_RED_CROSS - EGO crossed a red light. Episode will be truncated."
            )
            self.red_light_crossing_truncation = 0.0
            return True
        return False
