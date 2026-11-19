# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Episode reset, truncation, and termination workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, cast

from lanista.reinforcengine.running_mode import RunningMode
from lanista.reinforcengine.utils.simulation.episode_end import (
    EgoVehiclePresence,
    observe_ego_vehicle,
)
from lanista.reinforcengine.utils.simulation.ports import (
    SimulationLifecyclePort,
    VehiclePort,
)

from ..simulation.simulation_contracts import ScenarioContextPreprocessorPort
from .episode_state_reader import EpisodeStateReader


class _ScenarioConfigurationPort(Protocol):
    """Define the typed boundary for scenario configuration."""

    sumo_configuration: object | None

    def get_max_timesteps(self) -> float:
        """Return the max timesteps exposed by this scenario configuration port."""
        ...

    def get_step_length(self) -> float:
        """Return the step length exposed by this scenario configuration port."""
        ...


class _ScenarioManagerPort(Protocol):
    """Define the typed boundary for scenario manager."""

    def get_configuration(self) -> _ScenarioConfigurationPort:
        """Return the configuration exposed by this scenario manager port."""
        ...

    def get_index(self) -> int:
        """Return the index exposed by this scenario manager port."""
        ...

    def set_index(self, index: int) -> None:
        """Set the index owned by this scenario manager port."""
        ...

    def get_cumulative_training_counter(self) -> int:
        """Return the cumulative training counter exposed by this scenario manager port."""
        ...


class _EpisodeEgoVehiclePort(Protocol):
    """Define the typed boundary for episode ego vehicle."""

    last_ut: str | None
    MaxSpeed: float
    AverageSpeed: float


class _TerminationSimulatorPort(VehiclePort, Protocol):
    """Define the typed boundary for termination simulator."""

    def is_initialized(self) -> bool:
        """Return whether initialized applies to this termination simulator port."""
        ...


@dataclass
class EpisodeLifecycleService:
    """Coordinate episode lifecycle operations."""

    controller: Any
    state_reader: EpisodeStateReader

    @property
    def logger(self) -> logging.Logger:
        """Return the logger exposed by this episode lifecycle service."""
        return cast(logging.Logger, getattr(self.controller, "_logger", None))

    @property
    def scenario_manager(self) -> _ScenarioManagerPort | None:
        """Return the scenario manager exposed by this episode lifecycle service."""
        return cast(
            _ScenarioManagerPort | None,
            getattr(self.controller, "scenario_manager", None),
        )

    @property
    def scenario_manager_configuration(self) -> _ScenarioConfigurationPort | None:
        """Return the scenario manager configuration exposed by this episode lifecycle service."""
        return cast(
            _ScenarioConfigurationPort | None,
            getattr(self.controller, "_scenario_manager_configuration", None),
        )

    def reset(self) -> None:
        """Reset this episode lifecycle service to its initial operational state."""
        if self.scenario_manager is None:
            raise ValueError("Scenario manager is not set.")
        if (
            self.scenario_manager_configuration is None
            or self.scenario_manager_configuration.sumo_configuration is None
        ):
            raise ValueError("SUMO initial configuration is not set.")

        scenario_index_state = getattr(self.controller, "_scenario_index_state", None)
        state_index = self.state_reader.read_scenario_index(
            scenario_index_state
        )
        if state_index is not None and self.scenario_manager.get_index() != state_index:
            self.scenario_manager.set_index(state_index)

        self.controller.episode_counter += 1
        self.controller.cumulative_step_counter_over_episodes += self.controller.step_counter
        self.controller.step_counter = 0
        self.controller.episode_reward_counter = 0
        self.controller.episode_terminated = False
        self.controller.episode_truncated = False
        self.controller.last_termination_reason = None
        self.controller.last_truncation_reason = None
        self.controller.episode_end_conflict = False
        self.controller.episode_end_conflict_target_time_limit = False
        self.controller.episode_end_conflict_target_red_light_overdrive = False
        self.controller.curriculum_target_detected = False
        self.controller.curriculum_target_accepted = False
        self.controller.curriculum_target_rejected_by_red_light_overdrive = False
        self.controller.curriculum_target_reached = False
        self.controller._episode_start_simulation_time_seconds = 0.0
        self.controller._episode_active_step_limit = None

    def mark_episode_start_after_warmup(
        self,
        sumo_simulation_controller: SimulationLifecyclePort,
        active_step_limit: float | None = None,
    ) -> None:
        """Mark the episode start after warmup owned by this episode
lifecycle service."""
        self.controller._episode_start_simulation_time_seconds = (
            sumo_simulation_controller.get_simulation_time()
        )
        self.controller._episode_active_step_limit = active_step_limit

    def step(
        self,
        sumo_simulation_controller: VehiclePort,
        scenario_context_preprocessor: ScenarioContextPreprocessorPort,
        ego_vehicle: _EpisodeEgoVehiclePort,
        cumulative_reward_counter_over_episodes: float,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Advance the workflow owned by this episode lifecycle service by one step."""
        self.controller.step_counter += 1
        self._reset_per_step_flags()
        self._resolve_episode_end_conflict()
        self._project_curriculum_outcome()

        if self.controller.episode_terminated or self.controller.episode_truncated:
            self._log_terminal_summary(
                sumo_simulation_controller,
                scenario_context_preprocessor,
                ego_vehicle,
                cumulative_reward_counter_over_episodes,
            )

        return self._build_multi_agent_done_dictionaries()

    def _reset_per_step_flags(self) -> None:
        """Reset the per step flags owned by this episode lifecycle service."""
        self.controller.episode_end_conflict = False
        self.controller.episode_end_conflict_target_time_limit = False
        self.controller.episode_end_conflict_target_red_light_overdrive = False
        self.controller.curriculum_target_accepted = False
        self.controller.curriculum_target_rejected_by_red_light_overdrive = False

    def _resolve_episode_end_conflict(self) -> None:
        """Resolve the episode end conflict required by this episode lifecycle service."""
        if not (
            self.controller.episode_terminated
            and self.controller.episode_truncated
        ):
            return

        self.controller.episode_end_conflict = True
        termination_reason = getattr(
            self.controller,
            "last_termination_reason",
            None,
        )
        truncation_reason = getattr(
            self.controller,
            "last_truncation_reason",
            None,
        )
        target_reached = termination_reason in {"curriculum_target", "route_end"}
        self.controller.episode_end_conflict_target_time_limit = (
            target_reached and truncation_reason == "time_limit"
        )
        self.controller.episode_end_conflict_target_red_light_overdrive = (
            target_reached and truncation_reason == "red_light_overdrive"
        )
        if self.controller.episode_end_conflict_target_time_limit:
            self.logger.warning(
                "EPISODE_END_CONFLICT - Both termination and time-limit truncation "
                "were detected; treating the episode as terminated because the "
                "target was reached on the final allowed step."
            )
            self.controller.episode_truncated = False
        else:
            self.logger.warning(
                "EPISODE_END_CONFLICT - Both termination and truncation were detected; "
                "treating the episode as truncated."
            )
            self.controller.episode_terminated = False

    def _project_curriculum_outcome(self) -> None:
        """Project the curriculum outcome owned by this episode lifecycle service."""
        self.controller.curriculum_target_accepted = bool(
            self.controller.episode_terminated
            and getattr(self.controller, "last_termination_reason", None) == "curriculum_target"
        )
        self.controller.curriculum_target_rejected_by_red_light_overdrive = bool(
            getattr(self.controller, "curriculum_target_detected", False)
            and self.controller.episode_end_conflict_target_red_light_overdrive
        )
        self.controller.curriculum_target_reached = self.controller.curriculum_target_accepted

    def _log_terminal_summary(
        self,
        sumo_simulation_controller: VehiclePort,
        scenario_context_preprocessor: ScenarioContextPreprocessorPort,
        ego_vehicle: _EpisodeEgoVehiclePort,
        cumulative_reward_counter_over_episodes: float,
    ) -> None:
        """Log the terminal summary owned by this episode lifecycle service."""
        scenario_context = scenario_context_preprocessor.current_context
        vehicle_count = sumo_simulation_controller.get_vehicle_id_count()
        pedestrian_count = (
            sumo_simulation_controller.get_pedestrian_id_count()
        )
        scenario_manager = cast(_ScenarioManagerPort, self.scenario_manager)
        self.logger.info(
            f"RUNNING SCENARIO INDEX:     {scenario_manager.get_index()}\n"
            f"TRAINING ITERATION COUNTER: {scenario_manager.get_cumulative_training_counter()}\n"
            f"EPISODE COUNTER:            {self.controller.episode_counter}\n"
            f"CUMULATIVE STEP COUNTER:    {self.controller.cumulative_step_counter_over_episodes}\n"
            f"VEHICLE COUNTER (EGO TOO):  {vehicle_count}\n"
            f"PEDESTRIAN COUNTER:         {pedestrian_count}\n"
            f"REACHED INTERSECTION NUMBER:{scenario_context.intersection_number}\n"
            f"REACHED ROAD ID:            {scenario_context.ego_road_id}\n"
            f"CUMULATIVE REWARD:          {cumulative_reward_counter_over_episodes}\n"
            f"EPISODE REWARD:             {self.controller.episode_reward_counter}\n"
            f"MAX SPEED:                  {ego_vehicle.MaxSpeed} m/s\n"
            f"AVERAGE SPEED:              {ego_vehicle.AverageSpeed} m/s\n"
            f"TERMINATED:                 {self.controller.episode_terminated}\n"
            f"TRUNCATED:                  {self.controller.episode_truncated}"
        )
        self.logger.info("---Episode finished---")

    def _build_multi_agent_done_dictionaries(
        self,
    ) -> tuple[dict[str, bool], dict[str, bool]]:
        """Build the multi agent done dictionaries required by this episode lifecycle service."""
        terminateds = {"EGO_agent_0": self.controller.episode_terminated}
        truncateds = {"EGO_agent_0": self.controller.episode_truncated}

        if self.controller.episode_terminated:
            terminateds["__all__"] = True
            truncateds["__all__"] = False
        elif self.controller.episode_truncated:
            terminateds["__all__"] = False
            truncateds["__all__"] = True
        else:
            terminateds["__all__"] = False
            truncateds["__all__"] = False

        return terminateds, truncateds

    def termination_check(
        self,
        sumo_simulation_controller: _TerminationSimulatorPort,
        scenario_context_preprocessor: ScenarioContextPreprocessorPort,
        ego_vehicle: _EpisodeEgoVehiclePort,
    ) -> bool:
        """Return the termination check value owned by this episode lifecycle service."""
        scenario_context = scenario_context_preprocessor.current_context
        if (
            getattr(self.controller, "_mode", None) == RunningMode.TRAINING
            and getattr(self.controller, "_curriculum_index_state", None) is not None
        ):
            target = self.state_reader.read_curriculum_state(
                getattr(self.controller, "_curriculum_index_state", None)
            )
            if target is not None:
                target_type = target.get("target_type")
                target_value = target.get("target_value")
                target_name = target.get("state_name")
                if (
                    target_type == "intersection"
                    and scenario_context.intersection_number
                    >= cast(int, target_value)
                ):
                    self.controller.last_termination_reason = "curriculum_target"
                    self.controller.curriculum_target_detected = True
                    self.logger.info(
                        f"TERM_CURRICULUM ({target_name}) - Target intersection reached: "
                        f"{scenario_context.intersection_number} >= {target_value}"
                    )
                    return True
        if not sumo_simulation_controller.is_initialized():
            self.logger.warning("Sumo is not initialized. Termination check skipped.")
            return False
        observation = observe_ego_vehicle(
            sumo_simulation_controller,
            previous_road_id=ego_vehicle.last_ut,
            route_end_road_id="6321.56.41",
        )
        if observation.presence is EgoVehiclePresence.ROUTE_END:
            self.controller.last_termination_reason = "route_end"
            self.logger.info(
                "TERM_COND4 - Training episode terminated properly. "
                "EGO vehicle reached the end of its route. =)"
            )
            return True
        if observation.presence is EgoVehiclePresence.UNAVAILABLE:
            self.controller.last_termination_reason = "ego_unavailable"
            self.logger.warning(
                "EGO vehicle absence/collision confirmed. Episode will terminate."
            )
            return True
        ego_vehicle.last_ut = observation.road_id
        self.controller.last_termination_reason = None
        return False

    def truncation_check(
        self,
        sumo_simulation_controller: SimulationLifecyclePort,
    ) -> bool:
        """Return the truncation check value owned by this episode lifecycle service."""
        current_simulation_time = sumo_simulation_controller.get_simulation_time()
        episode_start_time = getattr(
            self.controller,
            "_episode_start_simulation_time_seconds",
            0.0,
        )
        elapsed_simulation_time = max(0.0, current_simulation_time - episode_start_time)
        scenario_manager = cast(_ScenarioManagerPort, self.scenario_manager)
        step_length = max(
            float(scenario_manager.get_configuration().get_step_length()),
            1e-9,
        )
        elapsed_episode_steps = elapsed_simulation_time / step_length
        step_limit = getattr(self.controller, "_episode_active_step_limit", None)
        if step_limit is None:
            step_limit = (
                scenario_manager.get_configuration().get_max_timesteps()
            )
        if elapsed_episode_steps >= step_limit:
            self.controller.last_truncation_reason = "time_limit"
            self.logger.warning(
                "TRUNC_COND - Truncated because active episode time is up. "
                f"({elapsed_episode_steps} >= limit({step_limit}) steps; "
                f"elapsed={elapsed_simulation_time}s, start={episode_start_time}s, current={current_simulation_time}s)"
            )
            return True

        current_step_overdrive = getattr(
            self.controller,
            "red_light_overdrive_this_step",
            None,
        )
        if not isinstance(current_step_overdrive, bool):
            restricted_area_check = getattr(
                self.controller,
                "is_running_into_restricted_area",
                None,
            )
            current_step_overdrive = bool(
                callable(restricted_area_check) and restricted_area_check()
            )
        if current_step_overdrive:
            self.controller.last_truncation_reason = "red_light_overdrive"
            self.logger.warning(
                "TRUNC_RED_LIGHT_OVERDRIVE - Truncated because of red light overdrive (piros lámpa túlfutás detected)."
            )
            return True

        self.controller.last_truncation_reason = None
        return False
