# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model-neutral lifecycle shared by the Eve and Jasmine environments."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar, Protocol, cast

from lanista.reinforcengine.running_mode import (
    RunningMode,
    require_supported_running_mode,
)
from lanista.reinforcengine.utils.simulation.episode_lifecycle import (
    EpisodeEgoVehicle,
    EpisodeSimulator,
    ScenarioContextView,
)
from lanista.reinforcengine.utils.simulation.ports import SimulatorPort
from lanista.reinforcengine.utils.simulation.simulation_controller import (
    SUMOSimulationController,
)

EGO_AGENT_ID = "EGO_agent_0"
EGO_POLICY_ID = "EGO_policy"

SUPPORTED_RUNNING_MODES = (
    RunningMode.TRAINING,
    RunningMode.TESTING,
    RunningMode.DEPLOYMENT,
)


class ScalarAction(Protocol):
    """Scalar policy action accepted by the longitudinal controller."""

    def item(self) -> float:
        """Return the item value provided by this scalar action."""
        ...


class EnvironmentSimulatorPort(SimulatorPort, Protocol):
    """Simulator lifecycle plus the deployment-only GUI commands."""

    def track_vehicle(self, view_id: str, vehicle_id: str) -> None:
        """Track the vehicle owned by this environment simulator port."""
        ...

    def set_zoom(self, view_id: str, zoom: float) -> None:
        """Set the zoom owned by this environment simulator port."""
        ...


class ScenarioConfigurationPort(Protocol):
    """Scenario path used to start SUMO."""

    sumo_cfg: str


class ScenarioPort(Protocol):
    """Selected scenario exposed by a scenario manager."""

    def get_configuration(self) -> ScenarioConfigurationPort:
        """Return the configuration exposed by this scenario port."""
        ...


class ScenarioManagerPort(Protocol):
    """Scenario selection surface consumed during environment reset."""

    def get_scenario(self) -> ScenarioPort:
        """Return the scenario exposed by this scenario manager port."""
        ...


class ScenarioManagerConfigurationPort(Protocol):
    """SUMO process options resolved before controller creation."""

    def get_step_length(self) -> float:
        """Return the step length exposed by this scenario manager configuration port."""
        ...

    def get_no_warnings(self) -> bool:
        """Return the no warnings exposed by this scenario manager configuration port."""
        ...


@dataclass(frozen=True, slots=True)
class SimulationEnvironmentConfiguration:
    """Immutable model variation for the shared environment adapters."""

    observation_size: int
    warmup_step_count: int = 110 * 20
    ego_vehicle_id: str = "EGO"
    deployment_view_id: str = "View #0"
    deployment_zoom: float = 800.0
    speed_mode: int = 39


@dataclass(frozen=True, slots=True)
class SimulationStepResult:
    """Framework-neutral result produced by one simulator step."""

    observation: list[float]
    reward: float
    terminated: dict[str, bool]
    truncated: dict[str, bool]


class SharedSimulationRuntime:
    """Coordinate common reset, step, and state-access behavior.

    Concrete model adapters supply only their episode and vehicle controller
    classes plus the observation-vector size. The explicit class bindings keep
    model-specific imports outside the shared engine.
    """

    episode_controller_type: ClassVar[type[object]]
    vehicle_controller_type: ClassVar[type[object]]
    environment_configuration: ClassVar[SimulationEnvironmentConfiguration]

    _logger: logging.Logger
    _mode: RunningMode
    sumo_simulation_controller: EnvironmentSimulatorPort
    episode_counter: int
    episode_reward_counter: float
    cumulative_step_counter_over_episodes: int
    cumulative_reward_counter_over_episodes: float
    step_counter: int
    episode_terminated: bool
    episode_truncated: bool

    @staticmethod
    def _initializer(controller_type: type[object]) -> Callable[..., None]:
        """Return the initializer value owned by this shared simulation runtime."""
        return cast(Callable[..., None], controller_type.__init__)

    def _initialize_episode_controller(
        self,
        constructor_arguments: Mapping[str, object],
    ) -> None:
        """Initialize model-neutral episode counters and scenario state."""

        self._initializer(self.episode_controller_type)(
            self,
            **constructor_arguments,
        )

    def _initialize_simulator_and_vehicle_controller(
        self,
        constructor_arguments: Mapping[str, object],
    ) -> None:
        """Create the SUMO owner before constructing model control state."""

        configuration = cast(
            ScenarioManagerConfigurationPort,
            self.__dict__["_scenario_manager_configuration"],
        )
        if constructor_arguments.get("defer_gui_shutdown_to_process_restart"):
            simulation_controller = SUMOSimulationController(
                step_length=configuration.get_step_length(),
                no_warnings=configuration.get_no_warnings(),
                defer_gui_shutdown_to_process_restart=True,
            )
        else:
            simulation_controller = SUMOSimulationController(
                step_length=configuration.get_step_length(),
                no_warnings=configuration.get_no_warnings(),
            )
        self.sumo_simulation_controller = cast(
            EnvironmentSimulatorPort,
            simulation_controller,
        )
        self._initializer(self.vehicle_controller_type)(
            self,
            simulator_port=self.sumo_simulation_controller,
        )

    @staticmethod
    def _controller_method[ReturnValue](
        controller_type: type[object],
        method_name: str,
    ) -> Callable[..., ReturnValue]:
        """Return the controller method value owned by this shared simulation runtime."""
        return cast(
            Callable[..., ReturnValue],
            getattr(controller_type, method_name),
        )

    def _reset_runtime(self) -> RunningMode:
        """Reset controller state, then dispatch the selected SUMO lifecycle."""

        mode = require_supported_running_mode(
            self._mode,
            SUPPORTED_RUNNING_MODES,
        )
        episode_reset = cast(
            Callable[[object], None],
            self._controller_method(
                self.episode_controller_type,
                "reset",
            ),
        )
        vehicle_reset = cast(
            Callable[[object], None],
            self._controller_method(
                self.vehicle_controller_type,
                "reset",
            ),
        )
        episode_reset(self)
        vehicle_reset(self)
        reset_handler = {
            RunningMode.TRAINING: self._reset_training,
            RunningMode.TESTING: self._reset_testing,
            RunningMode.DEPLOYMENT: self._reset_deployment,
        }[mode]
        self._logger.info(f"Running mode detected: {mode.name}")
        reset_handler()
        return mode

    def _scenario_configuration_path(self) -> str:
        """Return the scenario configuration path value owned by this shared simulation runtime."""
        scenario_manager = cast(
            ScenarioManagerPort,
            self.__dict__["scenario_manager"],
        )
        return scenario_manager.get_scenario().get_configuration().sumo_cfg

    def _prepare_ego_vehicle(self) -> None:
        """Prepare the ego vehicle required by this shared simulation runtime."""
        configuration = self.environment_configuration
        simulator = self.sumo_simulation_controller
        simulator.sim_step()
        simulator.set_vehicle_speed_mode(
            configuration.ego_vehicle_id,
            configuration.speed_mode,
        )
        simulator.set_vehicle_speed(configuration.ego_vehicle_id, 0.0)

    def _warm_up_traffic(self) -> None:
        """Warm the up traffic owned by this shared simulation runtime."""
        for _ in range(self.environment_configuration.warmup_step_count):
            self.sumo_simulation_controller.sim_step()

    def _reset_training(self) -> None:
        """Restart headless SUMO and warm its traffic flow for training."""

        simulator = self.sumo_simulation_controller
        simulator.shutdown_simulation()
        simulator.start_simulation(False, self._scenario_configuration_path())
        self._prepare_ego_vehicle()
        self._warm_up_traffic()
        self._logger.info("Traffic flow started")

    def _reset_testing(self) -> None:
        """Restart headless SUMO without enabling training-only truncation."""

        self._reset_training()

    def _reset_deployment(self) -> None:
        """Start GUI SUMO and warm its traffic flow for deployment."""

        configuration = self.environment_configuration
        simulator = self.sumo_simulation_controller
        simulator.start_simulation(True, self._scenario_configuration_path())
        simulator.sim_step()
        simulator.track_vehicle(
            configuration.deployment_view_id,
            configuration.ego_vehicle_id,
        )
        simulator.set_zoom(
            configuration.deployment_view_id,
            configuration.deployment_zoom,
        )
        simulator.set_vehicle_speed_mode(
            configuration.ego_vehicle_id,
            configuration.speed_mode,
        )
        simulator.set_vehicle_speed(configuration.ego_vehicle_id, 0.0)
        self._logger.info("Speeding up initial traffic flow...")
        self._warm_up_traffic()
        self._logger.info("Traffic flow initialized")

    def _execute_step(
        self,
        actions: Mapping[str, ScalarAction],
    ) -> SimulationStepResult:
        """Run control, simulator, terminal, and episode-counter stages."""

        vehicle_step = cast(
            Callable[..., tuple[list[float], float]],
            self._controller_method(
                self.vehicle_controller_type,
                "step",
            ),
        )
        next_observation, reward = vehicle_step(
            self,
            actions,
            self.step_counter + 1,
        )
        simulator = self.sumo_simulation_controller
        simulator.sim_step()

        scenario_context = cast(
            ScenarioContextView,
            self.__dict__["scenario_context_preprocessor"],
        )
        ego_vehicle = cast(
            EpisodeEgoVehicle,
            self.__dict__["ego_vehicle"],
        )
        self.episode_terminated = self._termination_check(
            simulator,
            scenario_context,
            ego_vehicle,
        )
        if self._mode is RunningMode.TRAINING:
            self.episode_truncated = self._truncation_check(simulator)
        else:
            self.episode_truncated = False

        self.episode_reward_counter += reward
        finish_episode = cast(
            Callable[..., tuple[dict[str, bool], dict[str, bool]]],
            self._controller_method(
                self.episode_controller_type,
                "step",
            ),
        )
        terminated, truncated = finish_episode(
            self,
            simulator,
            scenario_context,
            ego_vehicle,
            self.cumulative_reward_counter_over_episodes,
        )
        return SimulationStepResult(
            observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )

    def _termination_check(
        self,
        simulator: EpisodeSimulator,
        scenario_context: ScenarioContextView,
        ego_vehicle: EpisodeEgoVehicle,
    ) -> bool:
        """Return the termination check value owned by this shared simulation runtime."""
        handler = cast(
            Callable[..., bool],
            self._controller_method(
                self.episode_controller_type,
                "_termination_check",
            ),
        )
        return handler(self, simulator, scenario_context, ego_vehicle)

    def _truncation_check(
        self,
        simulator: EpisodeSimulator,
    ) -> bool:
        """Return the truncation check value owned by this shared simulation runtime."""
        handler = cast(
            Callable[..., bool],
            self._controller_method(
                self.episode_controller_type,
                "_truncation_check",
            ),
        )
        return handler(self, simulator)

    def get_mode(self) -> RunningMode:
        """Return the active validated running mode."""

        return self._mode

    def set_mode(self, value: object) -> None:
        """Validate and select a mode without partially mutating state."""

        self._mode = require_supported_running_mode(
            value,
            SUPPORTED_RUNNING_MODES,
        )

    def close(self) -> None:
        """Synchronously close the owned libsumo session and GUI window."""

        self.sumo_simulation_controller.shutdown_simulation()

    def get_episode_cnt(self) -> int:
        """Return the number of environment resets."""

        return self.episode_counter

    def get_cumulative_step_cnt(self) -> int:
        """Return the number of completed steps before the active episode."""

        return self.cumulative_step_counter_over_episodes

    def get_step_cnt(self) -> int:
        """Return the active episode's step count."""

        return self.step_counter
