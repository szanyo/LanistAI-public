# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable-Baselines3 projection of the shared Eve/Jasmine lifecycle."""

from __future__ import annotations

from typing import Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_environment.simulation_runtime import (
    EGO_AGENT_ID,
    SUPPORTED_RUNNING_MODES,
    ScalarAction,
    SharedSimulationRuntime,
)
from lanista.reinforcengine.rl_metrics.snapshot_projection import build_env_lanistai_snapshot
from lanista.reinforcengine.running_mode import require_supported_running_mode
from lanista.reinforcengine.utils.simulation.telemetry import (
    read_episode_time_milliseconds,
)


class _ManifestPort(Protocol):
    """Define the typed boundary for manifest."""

    def get_configuration(self) -> object:
        """Return the configuration exposed by this manifest port."""
        ...


class SharedSB3SimulationEnvironment(
    SharedSimulationRuntime,
    gym.Env[NDArray[np.float64], NDArray[np.float64]],
    LoggerInterface,
):
    """Map the common lifecycle to Gymnasium's single-agent API."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the shared sb3 simulation environment and its owned state."""
        mode = require_supported_running_mode(
            kwargs.get("rl_model_mode"),
            SUPPORTED_RUNNING_MODES,
        )
        gym.Env.__init__(self)
        self._initialize_episode_controller(kwargs)
        LoggerInterface.__init__(self, **kwargs)

        self._mode = mode
        self._scenario_manager = self.__dict__["scenario_manager"]
        self._scenario_index_state = kwargs["scenario_index_state"]
        manifest = cast(_ManifestPort, kwargs["rl_model_manifest"])
        self._rl_model_manifest = manifest
        self._rl_model_manifest_configuration = manifest.get_configuration()
        self._verbose = bool(kwargs.get("verbose", False))
        self._detailed_verbose = bool(kwargs.get("detailed_verbose", False))

        self._initialize_simulator_and_vehicle_controller(kwargs)
        size = self.environment_configuration.observation_size
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(size,),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            low=-4.0,
            high=3.0,
            shape=(1,),
            dtype=np.float64,
        )

    def reset(
        self,
        **kwargs: object,
    ) -> tuple[NDArray[np.float64], dict[str, object]]:
        """Reset the shared lifecycle and expose a zeroed SB3 observation."""

        del kwargs
        require_supported_running_mode(
            self._mode,
            SUPPORTED_RUNNING_MODES,
        )
        self._logger.debug("Resetting StableBaselinesSimEnv environment...")
        self._reset_runtime()
        observation: NDArray[np.float64] = np.zeros(
            (self.environment_configuration.observation_size,),
            dtype=np.float64,
        )
        return observation, {"term": False, "trunc": False, "ep_rew": 0}

    def step(
        self,
        action: ScalarAction,
    ) -> tuple[
        NDArray[np.float64],
        float,
        bool,
        bool,
        dict[str, object],
    ]:
        """Run one shared step and project it to Gymnasium's five values."""

        result = self._execute_step({EGO_AGENT_ID: action})
        terminated = result.terminated[EGO_AGENT_ID]
        truncated = result.truncated[EGO_AGENT_ID]
        observation = np.asarray(result.observation, dtype=np.float64)
        episode_time = read_episode_time_milliseconds(self.sumo_simulation_controller)
        info: dict[str, object] = {
            "term": terminated,
            "trunc": truncated,
            "ep_rew": self.episode_reward_counter,
            "episode": {
                "r": self.episode_reward_counter,
                "l": self.step_counter,
                "t": episode_time.value,
                "t_quality": episode_time.quality.value,
            },
            "terminal_observation": observation.copy(),
            "lanistai": build_env_lanistai_snapshot(
                self,
                result.observation,
                result.reward,
                terminated,
                truncated,
            ),
        }
        return observation, result.reward, terminated, truncated, info

    def render(self) -> None:
        """Return ``None`` because visualization is owned by SUMO's GUI."""

        return None
