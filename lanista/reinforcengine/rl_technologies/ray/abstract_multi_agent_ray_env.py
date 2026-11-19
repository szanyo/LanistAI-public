# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the environment adapter for abstract multi agent ray."""

import logging
from abc import ABC, abstractmethod

from lanista.foundation.configuration import Configuration
from gymnasium import spaces
from ray.rllib import MultiAgentEnv
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.utils.tools.model_configurator_loader import load_algorithm_configurator_module


class AbstractMultiAgentRayEnv(MultiAgentEnv, ABC):
    """Define the environment boundary for abstract multi agent ray."""

    def __init__(self, **kwargs):
        """Initialize the abstract multi agent ray env and its owned state."""
        MultiAgentEnv.__init__(self)
        self._logger = logging.getLogger()

        self._mode: RLModelRunningModes = kwargs.get("rl_model_mode")
        self._rl_model_manifest = kwargs.get("rl_model_manifest")
        self._rl_model_manifest_configuration: Configuration = (
            self._rl_model_manifest.get_configuration()
        )

        self._logger.info("Environment initialization started.")

        self._logger.debug("Configuration extraction started.")

        algorithm_configuration = self._rl_model_manifest_configuration.algorithm
        configurator_module_ref, configurator_module = load_algorithm_configurator_module(
            self._rl_model_manifest,
            self._logger,
        )
        algorithm_configuration.configurator_module = configurator_module_ref

        self._logger.debug("Configuration extraction finished.")
        self._logger.debug("Loading policies configurator function started.")

        _policies_configurator_fn = getattr(
            configurator_module, algorithm_configuration.components.policies
        )
        self._policies = _policies_configurator_fn()

        self._logger.debug("Loading policies configurator function finished.")

        self._logger.debug("Loading policy mapping function started.")

        self._policy_mapping_fn = getattr(
            configurator_module, algorithm_configuration.components.policy_mapping_fn
        )

        self._logger.debug("Loading policy mapping function finished.")

        self._logger.debug("Agent IDs configuration started.")
        self._compose_agent_ids()

        self._logger.debug("Observation space (dict) configuration started.")

        self._observation_space_dict = {}
        self._compose_observation_space()
        self.observation_space = spaces.Dict(self._observation_space_dict)

        self._logger.debug("Action space (dict) configuration started.")

        self._action_space_dict = {}
        self._compose_action_space()
        self.action_space = spaces.Dict(self._action_space_dict)

        self._logger.debug(
            "Observations configuration started. (Based on the honeycomb's data provider)"
        )

        self._observations = Configuration()
        self._configure_observations()

    @abstractmethod
    def _compose_agent_ids(self):
        """Compose the agent ids owned by this abstract multi agent ray env."""
        raise NotImplementedError("Subclasses must implement this method to compose agent IDs")

    @abstractmethod
    def _compose_observation_space(self):
        """Compose the observation space owned by this abstract multi agent ray env."""
        raise NotImplementedError(
            "Subclasses must implement this method to compose observation space dict"
        )

    @abstractmethod
    def _compose_action_space(self):
        """Compose the action space owned by this abstract multi agent ray env."""
        raise NotImplementedError(
            "Subclasses must implement this method to compose action space dict"
        )

    @abstractmethod
    def _configure_observations(self):
        """Configure the observations owned by this abstract multi agent ray env."""
        raise NotImplementedError(
            "Subclasses must implement this method to configure observations based on the honeycomb's data provider"
        )

    def reset(self, **kwargs):
        """Reset this abstract multi agent ray env to its initial operational state."""
        raise NotImplementedError("Subclasses must implement this method")

    def step(self, action):
        """Advance the workflow owned by this abstract multi agent ray env by one step."""
        raise NotImplementedError("Subclasses must implement this method")
