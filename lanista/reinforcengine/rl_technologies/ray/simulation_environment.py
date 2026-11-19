# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""RLlib projection of the shared Eve/Jasmine simulation lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, cast

from lanista.reinforcengine.rl_environment.simulation_runtime import (
    EGO_AGENT_ID,
    EGO_POLICY_ID,
    SUPPORTED_RUNNING_MODES,
    ScalarAction,
    SharedSimulationRuntime,
)
from lanista.reinforcengine.rl_technologies.ray.abstract_multi_agent_ray_env import (
    AbstractMultiAgentRayEnv,
)
from lanista.reinforcengine.running_mode import require_supported_running_mode


class _PolicyConfiguration(Protocol):
    """Define the typed boundary for policy configuration."""

    observation_space: object
    action_space: object
    config: Mapping[str, object]


class SharedRaySimulationEnvironment(  # type: ignore[misc]
    SharedSimulationRuntime,
    AbstractMultiAgentRayEnv,
):
    """Map the common lifecycle to RLlib's multi-agent dictionaries."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the shared ray simulation environment and its owned state."""
        if "rl_model_mode" in kwargs:
            kwargs["rl_model_mode"] = require_supported_running_mode(
                kwargs["rl_model_mode"],
                SUPPORTED_RUNNING_MODES,
            )
        AbstractMultiAgentRayEnv.__init__(self, **kwargs)  # type: ignore[no-untyped-call]
        self._initialize_episode_controller(kwargs)
        self._initialize_simulator_and_vehicle_controller(kwargs)

    def _compose_agent_ids(self) -> None:
        """Compose the agent ids owned by this shared ray simulation environment."""
        self._EGO_agent_id = [EGO_AGENT_ID]
        self._agent_ids = set(self._EGO_agent_id)
        self.agents = self.possible_agents = list(self._EGO_agent_id)

    def _policy(self) -> _PolicyConfiguration:
        """Return the policy value owned by this shared ray simulation environment."""
        return cast(_PolicyConfiguration, self._policies[EGO_POLICY_ID])

    def _compose_observation_space(self) -> None:
        """Compose the observation space owned by this shared ray simulation environment."""
        observation_space = self._policy().observation_space
        self._observation_space_dict = {
            agent_id: observation_space for agent_id in self._EGO_agent_id
        }

    def _compose_action_space(self) -> None:
        """Compose the action space owned by this shared ray simulation environment."""
        action_space = self._policy().action_space
        self._action_space_dict = {agent_id: action_space for agent_id in self._EGO_agent_id}

    def _configure_observations(self) -> None:
        """Configure the observations owned by this shared ray simulation environment."""
        observation = self._policy().config.get("observation")
        if not callable(observation):
            self._logger.error("Observation configuration failed because of missing declaration.")
            raise ValueError("Observation configuration failed because of missing declaration.")
        self._observations.EGO_policy = observation

    def _observation_callback(self) -> Callable[..., object]:
        """Return the observation callback value owned by this shared ray simulation environment."""
        callback = getattr(self._observations, EGO_POLICY_ID)
        if not callable(callback):
            raise ValueError("EGO policy observation callback is unavailable.")
        return cast(Callable[..., object], callback)

    def get_observation_sample(self, policy_name: str) -> object:
        """Return an integrity-check sample for the configured EGO policy."""

        if policy_name != EGO_POLICY_ID:
            self._logger.error(f"Policy name is not valid: {policy_name}")
            raise ValueError(f"Policy name is not valid: {policy_name}")
        return self._observation_callback()(self, integrity_check=True)

    def reset(
        self,
        **kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Reset the shared lifecycle and expose RLlib's agent mapping."""

        del kwargs
        self._reset_runtime()
        observation = self._observation_callback()()
        return {EGO_AGENT_ID: observation}, {}

    def step(  # type: ignore[override]
        self,
        actions: Mapping[str, ScalarAction],
    ) -> tuple[
        dict[str, object],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, object],
    ]:
        """Run one shared step and project it to RLlib's five mappings."""

        result = self._execute_step(actions)
        observation = self._observation_callback()(result.observation)
        return (
            {EGO_AGENT_ID: observation},
            {EGO_AGENT_ID: result.reward},
            result.terminated,
            result.truncated,
            {},
        )

    def get_policy_mapping_fn(self) -> Callable[..., object]:
        """Return the mapping function loaded from the model configurator."""

        return cast(Callable[..., object], self._policy_mapping_fn)
