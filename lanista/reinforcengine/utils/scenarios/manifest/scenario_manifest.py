# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define manifest data for scenario."""

import os

from lanista.foundation import filesystem as IO
from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.filesystem import only_files_from_dir
from lanista.foundation.logging import LoggerInterface

from lanista.reinforcengine.utils.scenarios.manifest.scenario_manifest_configuration_wrapper import ScenarioManifestConfigurationWrapper

APP_CONFIGURATION = AppConfiguration

class ScenarioManifest(LoggerInterface):
    """Describe scenario data."""

    def __init__(self, scenario_folder_path: str, **kwargs):
        """Initialize the scenario manifest with the path to the scenario folder."""
        self._scenario_folder_path = scenario_folder_path
        super().__init__(**kwargs)
        self._scenario_manifest_configuration = ScenarioManifestConfigurationWrapper()
        path = os.path.join(scenario_folder_path, "scenario_manifest_configuration.yaml")
        self._scenario_manifest_configuration.set_config_file_path(path)

    def load(self) -> bool:
        """Load the scenario manifest configuration from the scenario folder."""

        if not IO.existf(self._scenario_manifest_configuration.get_config_file_path()):
            return False
        try:
            self._scenario_manifest_configuration.load()
            self._scenario_manifest_configuration.scenario_folder_path = self._scenario_folder_path
            files = only_files_from_dir(self._scenario_folder_path)
            for file in files:
                if file.endswith(".sumocfg"):
                    self._scenario_manifest_configuration.sumo_cfg = file
                elif file.endswith(".rou.xml"):
                    self._scenario_manifest_configuration.random_route_xml = file
            return True
        except Exception as e:
            self._logger.debug(f"{self._scenario_folder_path} is not a valid scenario folder.")
            self._scenario_manifest_configuration = ScenarioManifestConfigurationWrapper()
            return False

    def save(self) -> bool:
        """Persist the current state owned by this scenario manifest."""
        try:
            self._scenario_manifest_configuration.save()
            return True
        except Exception as e:
            self._logger.error(f"Failed to save scenario manifest configuration: {e}")
            return False

    def get_configuration(self):
        """Get the scenario manifest configuration."""
        if self._scenario_manifest_configuration is None:
            self._logger.error("Scenario manifest configuration is not loaded or invalid.")
            return None
        return self._scenario_manifest_configuration
