# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Ray environment adapter for Susan's simulation and episode pipeline."""

from lanista.reinforcengine.rl_technologies.ray.abstract_multi_agent_ray_env import (
    AbstractMultiAgentRayEnv,
)
from lanista.reinforcengine.running_mode import RunningMode, require_supported_running_mode
from lanista.reinforcengine.utils.simulation.simulation_controller import SUMOSimulationController

from ..utils.control.vehicle_sim_controller import VehicleSimController
from ..utils.episode.episode_controller import EpisodeController
from ..utils.simulation.environment_lifecycle import (
    SUSAN_ENVIRONMENT_LIFECYCLE,
    EnvironmentResetRequest,
    TrainingBudgetInput,
    WarmupInput,
    resolve_curriculum_time_limit_policy,
)
from ..utils.training.training_warmup import scenario_scaled_warmup_jitter_seconds
from .episode_state_reader import RayEpisodeStateReader

_SUPPORTED_RUNNING_MODES = (
    RunningMode.TRAINING,
    RunningMode.TESTING,
    RunningMode.DEPLOYMENT,
)


class RaySimEnv(AbstractMultiAgentRayEnv, EpisodeController, VehicleSimController):
    """Multi-agent RLlib facade over Susan's simulation controllers."""

    _TRAINING_WARMUP_JITTER_SECONDS = 30

    def __init__(self, **kwargs):
        """Initialize the RLlib adapter and its SUMO controller."""

        if "rl_model_mode" in kwargs:
            kwargs["rl_model_mode"] = require_supported_running_mode(
                kwargs["rl_model_mode"],
                _SUPPORTED_RUNNING_MODES,
            )
        # MultiAgentEnv init is handled in abstract_multi_agent_ray_env
        AbstractMultiAgentRayEnv.__init__(self, **kwargs)
        EpisodeController.__init__(
            self,
            episode_state_reader=RayEpisodeStateReader(),
            **kwargs,
        )
        self._diagnostic_trace_capture_gate = kwargs.get(
            "diagnostic_trace_capture_gate"
        )
        lifecycle_metrics = SUSAN_ENVIRONMENT_LIFECYCLE.initialize_metrics(
            kwargs.get(
                "training_warmup_jitter_seconds",
                self._TRAINING_WARMUP_JITTER_SECONDS,
            ),
        )
        # Keep framework-visible lifecycle state at the adapter boundary; the shared
        # policy remains immutable and unaware of environment attributes.
        self._training_warmup_jitter_seconds = lifecycle_metrics.configured_jitter_seconds
        self.current_training_warmup_jitter_seconds = lifecycle_metrics.current_jitter_seconds
        self.training_warmup_seconds = lifecycle_metrics.warmup_seconds
        self.training_warmup_steps = lifecycle_metrics.warmup_steps
        self.training_warmup_max_seconds = lifecycle_metrics.maximum_warmup_seconds
        self.training_warmup_max_steps = lifecycle_metrics.maximum_warmup_steps
        self.training_startup_steps = lifecycle_metrics.startup_steps
        self.active_policy_step_limit = lifecycle_metrics.active_policy_step_limit
        self.active_policy_seconds_limit = lifecycle_metrics.active_policy_seconds_limit

        controller_kwargs = {
            "step_length": self._scenario_manager_configuration.get_step_length(),
            "no_warnings": self._scenario_manager_configuration.get_no_warnings(),
        }
        if kwargs.get("defer_gui_shutdown_to_process_restart"):
            controller_kwargs["defer_gui_shutdown_to_process_restart"] = True
        self.sumo_simulation_controller = SUMOSimulationController(**controller_kwargs)
        VehicleSimController.__init__(
            self,
            simulator_port=self.sumo_simulation_controller,
        )

    def _training_warmup_step_count(self) -> int:
        """Sample and publish the warmup window for this reset."""

        result = SUSAN_ENVIRONMENT_LIFECYCLE.sample_warmup(
            WarmupInput(
                step_length_seconds=float(self._scenario_manager_configuration.get_step_length()),
                scaled_jitter_seconds=(self._scaled_training_warmup_jitter_seconds()),
            )
        )
        self.current_training_warmup_jitter_seconds = result.current_jitter_seconds
        self.training_warmup_seconds = result.warmup_seconds
        self.training_warmup_steps = result.warmup_steps
        self.training_warmup_max_seconds = result.maximum_warmup_seconds
        self.training_warmup_max_steps = result.maximum_warmup_steps
        return result.warmup_steps

    def _active_training_step_limit(self) -> int:
        """Return the guaranteed policy budget after maximum warmup."""

        curriculum_state = self._episode_state_reader.read_curriculum_state(
            self._curriculum_index_state
        )
        result = SUSAN_ENVIRONMENT_LIFECYCLE.calculate_training_budget(
            TrainingBudgetInput(
                step_length_seconds=float(self._scenario_manager_configuration.get_step_length()),
                scaled_jitter_seconds=(self._scaled_training_warmup_jitter_seconds()),
                maximum_timesteps=int(
                    self.scenario_manager.get_configuration().get_max_timesteps()
                ),
                startup_steps=self.training_startup_steps,
                simulation_begin_time_seconds=float(
                    self._scenario_manager_configuration.get_begin_time()
                ),
                simulation_end_time_seconds=float(
                    self._scenario_manager_configuration.get_end_time()
                ),
                curriculum_time_limit=resolve_curriculum_time_limit_policy(
                    self._rl_model_manifest_configuration,
                    curriculum_state,
                ),
            )
        )
        self.current_training_warmup_jitter_seconds = result.current_jitter_seconds
        self.training_warmup_max_seconds = result.maximum_warmup_seconds
        self.training_warmup_max_steps = result.maximum_warmup_steps
        self.active_policy_step_limit = result.active_policy_step_limit
        self.active_policy_seconds_limit = result.active_policy_seconds_limit
        return result.active_policy_step_limit

    def _scaled_training_warmup_jitter_seconds(self) -> float:
        """Return the scaled training warmup jitter seconds value owned by this ray sim env."""
        return float(
            scenario_scaled_warmup_jitter_seconds(
                getattr(
                    self,
                    "_training_warmup_jitter_seconds",
                    self._TRAINING_WARMUP_JITTER_SECONDS,
                ),
                self.__dict__.get("scenario_manager") or self.__dict__.get("_scenario_manager"),
                getattr(self, "_scenario_index_state", None),
                state_reader=getattr(
                    self,
                    "_episode_state_reader",
                    None,
                ),
            )
        )

    def _environment_reset_request(self) -> EnvironmentResetRequest:
        """Return the environment reset request value owned by this ray sim env."""
        return EnvironmentResetRequest(
            simulator=self.sumo_simulation_controller,
            resolve_sumo_configuration_path=lambda: (
                self.scenario_manager.get_scenario().get_configuration().sumo_cfg
            ),
            sample_warmup_steps=self._training_warmup_step_count,
            calculate_active_step_limit=(self._active_training_step_limit),
            mark_episode_start=lambda active_limit: (
                self.mark_episode_start_after_warmup(
                    self.sumo_simulation_controller,
                    active_step_limit=active_limit,
                )
            ),
            log_info=self._logger.info,
        )

    def _compose_agent_ids(self):
        """Compose the agent ids owned by this ray sim env."""
        self._EGO_agent_id = [f"EGO_agent_{i}" for i in range(1)]

        self._agent_ids = list(self._EGO_agent_id)
        self.agents = self.possible_agents = self._agent_ids

    def _compose_observation_space(self):
        """Compose the observation space owned by this ray sim env."""
        _EGO_observation_space = self._policies["EGO_policy"].observation_space

        self._observation_space_dict = {}
        self._observation_space_dict.update(
            {agent_id: _EGO_observation_space for agent_id in self._EGO_agent_id}
        )

    def _compose_action_space(self):
        """Compose the action space owned by this ray sim env."""
        _EGO_action_space = self._policies["EGO_policy"].action_space

        self._action_space_dict = {}
        self._action_space_dict.update(
            {agent_id: _EGO_action_space for agent_id in self._EGO_agent_id}
        )

    def _configure_observations(self):
        """Configure the observations owned by this ray sim env."""
        def _handle_missing_declaration():
            """Handle the missing declaration workflow for this ray sim env."""
            self._logger.error("Observation configuration failed because of missing declaration.")
            raise Exception("Observation configuration failed because of missing declaration.")

        self._observations.EGO_policy = self._policies["EGO_policy"].config.get("observation", None)

        # check if observation is not valid
        for observation in self._observations.items():
            if observation is None:
                _handle_missing_declaration()

    def get_observation_sample(self, policy_name):
        """Return the observation sample exposed by this ray sim env."""
        if policy_name == "EGO_policy":
            return self._observations.EGO_policy(self, integrity_check=True)
        else:
            self._logger.error(f"Policy name is not valid: {policy_name}")
            raise Exception(f"Policy name is not valid: {policy_name}")

    def reset(self, **kwargs):
        """
        Reset environment.

        :param kwargs: keyword arguments
        """

        mode = require_supported_running_mode(
            self._mode,
            _SUPPORTED_RUNNING_MODES,
        )
        # Gym requires this compatibility parameter even though the adapter ignores it.
        SUSAN_ENVIRONMENT_LIFECYCLE.prepare_reset(
            lambda: EpisodeController.reset(self),
            lambda: VehicleSimController.reset(self),
        )

        reset_handler = {
            RunningMode.TRAINING: self._reset_training,
            RunningMode.TESTING: self._reset_testing,
            RunningMode.DEPLOYMENT: self._reset_deployment,
        }[mode]
        self._logger.info(f"Running mode detected: {mode.name}")
        reset_handler()

        next_observations = {"EGO_agent_0": self._observations.EGO_policy()}

        infos = {}

        return next_observations, infos

    def _reset_training(self) -> None:
        """Reset SUMO for headless training."""

        SUSAN_ENVIRONMENT_LIFECYCLE.reset_training(self._environment_reset_request())

    def _reset_testing(self) -> None:
        """Reset headless libsumo without training-only truncation."""

        self._reset_training()

    def _reset_deployment(self) -> None:
        """Reset SUMO for GUI deployment."""

        SUSAN_ENVIRONMENT_LIFECYCLE.reset_deployment(self._environment_reset_request())

    def step(self, actions):
        """
        Step function.

        1. Pre-process the input data
        2. Calculate the output of the original Simulink environment model
        3. Perform a SUMO simulation step
        4. Check for termination
        5. Check for truncation
        6. Update the real-time scheduler

        :param actions: actions for the agents
        :return: observation, reward, terminated, truncated, info
        """

        next_observations, rewards, terminateds, truncateds, infos = {}, {}, {}, {}, {}

        next_observation, reward = VehicleSimController.step(self, actions, self.step_counter + 1)

        self.sumo_simulation_controller.sim_step()

        self.episode_terminated = self._termination_check(
            self.sumo_simulation_controller,
            self.scenario_context_preprocessor,
            self.ego_vehicle,
        )
        if self._mode == RunningMode.TRAINING:
            self.episode_truncated = self._truncation_check(self.sumo_simulation_controller)

        next_observations = {"EGO_agent_0": self._observations.EGO_policy(next_observation)}
        rewards = {"EGO_agent_0": reward}

        self.episode_reward_counter += reward

        terminateds, truncateds = EpisodeController.step(
            self,
            self.sumo_simulation_controller,
            self.scenario_context_preprocessor,
            self.ego_vehicle,
            self.cumulative_reward_counter_over_episodes,
        )

        return next_observations, rewards, terminateds, truncateds, infos

    def get_mode(self):
        """Return the mode exposed by this ray sim env."""
        return self._mode

    def set_mode(self, value):
        """Set the mode owned by this ray sim env."""
        self._mode = require_supported_running_mode(
            value,
            _SUPPORTED_RUNNING_MODES,
        )

    def get_episode_cnt(self):
        """Return the episode cnt exposed by this ray sim env."""
        return self.episode_counter

    def get_cumulative_step_cnt(self):
        """Return the cumulative step cnt exposed by this ray sim env."""
        return self.cumulative_step_counter_over_episodes

    def get_step_cnt(self):
        """Return the step cnt exposed by this ray sim env."""
        return self.step_counter

    def get_observation_component_series(self) -> dict[str, float]:
        """Expose Susan observation metrics at the RLlib adapter boundary."""

        return self._observation_service.component_series()

    def get_reward_component_series(self) -> dict[str, float]:
        """Expose Susan reward metrics at the RLlib adapter boundary."""

        return self._reward_service.component_series()

    def get_training_diagnostic_component_series(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, float]:
        """Expose semantic diagnostics only to the training aggregation path."""

        if self._mode is not RunningMode.TRAINING:
            return {}
        return VehicleSimController.get_training_diagnostic_component_series(
            self,
            terminated=terminated,
            truncated=truncated,
        )

    def get_policy_mapping_fn(self):
        """Expose the policy-mapping callback required by the RLlib adapter."""
        return self._policy_mapping_fn

    def close(self) -> None:
        """Synchronously release Susan's libsumo session and GUI window."""

        self.sumo_simulation_controller.shutdown_simulation()
