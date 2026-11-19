# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt configuration access for scenario manifest configuration."""

from lanista.foundation.configuration import Configuration


class ScenarioManifestConfigurationWrapper(Configuration):
    """
    Base class for scenario manifest configuration in reinforcement learning.
    This class should be inherited by specific scenario manifest implementations.
    """

    def __init__(self):
        """
        Initialize the ScenarioManifestConfigurationWrapper with optional keyword arguments.
        """
        Configuration.__init__(self)

    def get_scenario_folder_path(self) -> str:
        """Return the scenario folder path exposed by this scenario manifest configuration wrapper.
        """
        return self.scenario_folder_path
