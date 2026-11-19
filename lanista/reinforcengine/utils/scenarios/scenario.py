# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define the simulation scenario abstraction."""

from lanista.foundation.logging import LoggerInterface

from lanista.reinforcengine.utils.scenarios.manifest.scenario_manifest import ScenarioManifest


class Scenario(LoggerInterface):
    """Define the shared lifecycle of a simulation scenario."""

    def __init__(self, scenario_manifest: ScenarioManifest, **kwargs):
        """Initialize the scenario with the path to the scenario folder."""
        LoggerInterface.__init__(self, **kwargs)
        self._scenario_manifest: ScenarioManifest = scenario_manifest

    def get_manifest(self) -> ScenarioManifest:
        """Get the scenario manifest."""
        return self._scenario_manifest

    def get_configuration(self):
        """Get the scenario configuration."""
        return self._scenario_manifest.get_configuration()
