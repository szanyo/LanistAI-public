# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable-Baselines3 environment adapter for Susan's simulation pipeline."""

import gymnasium as gym
import numpy as np

from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_metrics.snapshot_projection import build_env_lanistai_snapshot
from lanista.reinforcengine.running_mode import RunningMode, require_supported_running_mode
from lanista.reinforcengine.utils.simulation.simulation_controller import SUMOSimulationController
from lanista.reinforcengine.utils.simulation.telemetry import read_episode_time_milliseconds

from ..utils.control.vehicle_sim_controller import VehicleSimController
from ..utils.episode.episode_controller import EpisodeController
from ..utils.episode.episode_state_reader import LocalEpisodeStateReader
from ..utils.observation.observation import observation_dict, observation_space_dict
from ..utils.simulation.environment_lifecycle import (
    SUSAN_ENVIRONMENT_LIFECYCLE,
    EnvironmentResetRequest,
    TrainingBudgetInput,
    WarmupInput,
    resolve_curriculum_time_limit_policy,
)
from ..utils.training.training_warmup import scenario_scaled_warmup_jitter_seconds

TRAINING = RunningMode.TRAINING
"""RunningMode: Training mode constant."""

TESTING = RunningMode.TESTING
"""RunningMode: Testing/evaluation mode constant."""

DEPLOYMENT = RunningMode.DEPLOYMENT
"""RunningMode: Deployment mode constant."""

_SUPPORTED_RUNNING_MODES = (
    TRAINING,
    TESTING,
    DEPLOYMENT,
)


def _capture_training_diagnostic_trace(
    environment,
    *,
    terminated: bool,
    truncated: bool,
) -> dict[str, object] | None:
    """Capture an armed training trace without expanding the SB3 adapter contract."""

    if environment._mode is not RunningMode.TRAINING:
        return None
    capture = getattr(environment, "capture_training_diagnostic_trace_row", None)
    scenario_index_state = getattr(environment, "_scenario_index_state", None)
    if not callable(capture) or scenario_index_state is None:
        return None
    return capture(
        scenario_index_provider=scenario_index_state.get,
        terminated=terminated,
        truncated=truncated,
    )


class StableBaselinesSimEnv(gym.Env, LoggerInterface, EpisodeController, VehicleSimController):
    """Single-agent Gymnasium facade over Susan's simulation controllers."""

    _TRAINING_WARMUP_JITTER_SECONDS = 30

    def __init__(self, **kwargs):
        """Initialize the stable baselines sim env and its owned state."""
        mode = require_supported_running_mode(
            kwargs.get("rl_model_mode"),
            _SUPPORTED_RUNNING_MODES,
        )
        gym.Env.__init__(self)
        EpisodeController.__init__(
            self,
            episode_state_reader=LocalEpisodeStateReader(),
            **kwargs,
        )
        LoggerInterface.__init__(self, **kwargs)

        self._mode = mode
        self._scenario_manager = self.scenario_manager
        self._scenario_index_state = kwargs["scenario_index_state"]
        self._rl_model_manifest = kwargs["rl_model_manifest"]
        self._rl_model_manifest_configuration = self._rl_model_manifest.get_configuration()
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

        self._verbose = kwargs.get("verbose", False)
        self._detailed_verbose = kwargs.get("detailed_verbose", False)

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

        self.observation_space = observation_space_dict()
        self.action_space = gym.spaces.Box(
            low=np.array([-4.0], dtype=np.float32),
            high=np.array([3.0], dtype=np.float32),
            dtype=np.float32,
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
        """Return the scaled training warmup jitter seconds value owned by this stable baselines
        sim env.
        """
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
        """Return the environment reset request value owned by this stable baselines sim env."""
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

    def reset(self, **kwargs):
        """
        Reset environment.

        :param kwargs: keyword arguments
        """
        mode = require_supported_running_mode(
            self._mode,
            _SUPPORTED_RUNNING_MODES,
        )
        self._logger.debug("Resetting StableBaselinesSimEnv environment...")

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

        next_observation = observation_dict()
        info = {
            "term": False,
            "trunc": False,
            "ep_rew": 0,
        }

        return next_observation, info

    def _reset_training(self) -> None:
        """Reset SUMO for headless training."""

        SUSAN_ENVIRONMENT_LIFECYCLE.reset_training(self._environment_reset_request())

    def _reset_testing(self) -> None:
        """
        Reset function for non-GUI evaluation/testing mode.

        This mirrors training startup, but the step loop does not apply training
        truncation checks, so diagnostic runs can pass restricted areas and
        record the full route telemetry.
        """
        self._reset_training()

    def _reset_deployment(self) -> None:
        """Reset SUMO for GUI deployment."""

        SUSAN_ENVIRONMENT_LIFECYCLE.reset_deployment(self._environment_reset_request())

    def step(self, action):
        """
        Step function.

        :param action: SB3 single-agent action
        :return: observation, reward, terminated, truncated, info
        """
        actions = {"EGO_agent_0": action}

        next_observation, reward = VehicleSimController.step(
            self,
            actions,
            self.step_counter + 1,
        )

        self.sumo_simulation_controller.sim_step()

        self.episode_terminated = self._termination_check(
            self.sumo_simulation_controller,
            self.scenario_context_preprocessor,
            self.ego_vehicle,
        )
        if self._mode == RunningMode.TRAINING:
            self.episode_truncated = self._truncation_check(self.sumo_simulation_controller)
        else:
            self.episode_truncated = False

        self.episode_reward_counter += reward

        terminateds, truncateds = EpisodeController.step(
            self,
            self.sumo_simulation_controller,
            self.scenario_context_preprocessor,
            self.ego_vehicle,
            self.cumulative_reward_counter_over_episodes,
        )

        terminated = terminateds["EGO_agent_0"]
        truncated = truncateds["EGO_agent_0"]
        episode_time = read_episode_time_milliseconds(self.sumo_simulation_controller)

        observation_out = observation_dict(next_observation)
        terminal_observation = {
            key: np.array(value, dtype=np.float64, copy=True)
            for key, value in observation_out.items()
        }
        diagnostic_trace = _capture_training_diagnostic_trace(
            self,
            terminated=terminated,
            truncated=truncated,
        )

        info = {
            "term": terminated,
            "trunc": truncated,
            "ep_rew": self.episode_reward_counter,
            "episode": {
                "r": self.episode_reward_counter,
                "l": self.step_counter,
                "t": episode_time.value,
                "t_quality": episode_time.quality.value,
            },
            "terminal_observation": terminal_observation,
            "lanistai": build_env_lanistai_snapshot(
                self, next_observation, reward, terminated, truncated
            ),
        }
        if diagnostic_trace is not None:
            info["lanistai_diagnostic_trace"] = diagnostic_trace

        return observation_out, reward, terminated, truncated, info

    def render(self, mode="human") -> None:
        """Return no frame because the external SUMO process owns rendering."""
        return None

    def close(self) -> None:
        """Synchronously release Susan's libsumo session and GUI window."""

        self.sumo_simulation_controller.shutdown_simulation()

    def get_mode(self):
        """Return the mode exposed by this stable baselines sim env."""
        return self._mode

    def set_mode(self, value):
        """Set the mode owned by this stable baselines sim env."""
        self._mode = require_supported_running_mode(
            value,
            _SUPPORTED_RUNNING_MODES,
        )

    def get_episode_cnt(self):
        """Return the episode cnt exposed by this stable baselines sim env."""
        return self.episode_counter

    def get_cumulative_step_cnt(self):
        """Return the cumulative step cnt exposed by this stable baselines sim env."""
        return self.cumulative_step_counter_over_episodes

    def get_step_cnt(self):
        """Return the step cnt exposed by this stable baselines sim env."""
        return self.step_counter

    def get_observation_component_series(self) -> dict[str, float]:
        """Expose Susan observation metrics at the SB3 adapter boundary."""

        return self._observation_service.component_series()

    def get_reward_component_series(self) -> dict[str, float]:
        """Expose Susan reward metrics at the SB3 adapter boundary."""

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
