# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Scenario generation, validation, and publication orchestration."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from lanista.foundation.filesystem import existdir, makedirs, only_directories_from_dir
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager_configuration_wrapper import (
    ScenarioManagerConfigurationWrapper,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_set_replacement import (
    ScenarioSetReplacement,
)
from lanista.reinforcengine.utils.scenarios.manifest.scenario_manifest import ScenarioManifest
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.scenarios.scenario import Scenario
from lanista.reinforcengine.utils.tools.steppable import Steppable
from lanista.reinforcengine.utils.trip_and_route_generator import TripAndRouteGenerator
from lanista.reinforcengine.utils.xml_manipulators.pedestrian_route_xml_manipulator import (
    PedestrianRouteXMLManipulator,
)
from lanista.reinforcengine.utils.xml_manipulators.xml_manipulator import XMLManipulator

SCENARIO_ARTIFACT_FORMAT_VERSION = 2


class _AdaptiveVehicleFlow(Protocol):
    """Define the typed boundary for adaptive vehicle flow."""

    initial_number_of_vehicles: int
    target_number_of_vehicles: int


class _AdaptivePedestrianFlow(Protocol):
    """Define the typed boundary for adaptive pedestrian flow."""

    initial_number_of_pedestrians: int
    target_number_of_pedestrians: int


class _VehicleGenerationConfiguration(Protocol):
    """Define the typed boundary for vehicle generation configuration."""

    period: float


@dataclass(frozen=True)
class _GenerationInputs:
    """Carry the data required for generation inputs."""

    scenario_root: str
    network_file: str
    vehicle_flow: _AdaptiveVehicleFlow
    pedestrian_flow: _AdaptivePedestrianFlow


@dataclass(frozen=True)
class _ScenarioCounts:
    """Carry the data required for scenario counts."""

    vehicles: int
    pedestrians: int


@dataclass(frozen=True)
class _VehicleRouteArtifacts:
    """Carry the data required for vehicle route artifacts."""

    scenario_directory: str
    trip_file: str
    route_file: str
    generation_configuration: _VehicleGenerationConfiguration


def _calculate_adaptive_count(
    initial_count: int,
    target_count: int,
    progress_ratio: float,
) -> int:
    """
    Calculate the current count based on progress ratio.

    :param initial_count: Starting count
    :param target_count: Target count
    :param progress_ratio: Current progress (0.0 to 1.0)
    :return: Current count
    """

    return int(initial_count + (target_count - initial_count) * progress_ratio)


def _adjust_vehicle_config_for_count(
    config: _VehicleGenerationConfiguration,
    simulation_time: float,
    target_count: int,
) -> _VehicleGenerationConfiguration:
    """
    Adjust vehicle config to produce approximately the desired vehicle count.
    SUMO doesn't allow direct specification of vehicle count, so we adjust the
    period parameter to influence the generation rate.

    :param config: Original vehicle configuration
    :param target_count: Target vehicle count
    :return: Adjusted configuration
    """

    from copy import deepcopy

    adjusted_config = deepcopy(config)

    # Adjust period to approximate target count
    # Formula: count ≈ end_time / period
    if target_count > 0:
        # Use a minimum period to avoid too high vehicle density
        adjusted_config.period = max(0.1, simulation_time / target_count)
    else:
        adjusted_config.period = simulation_time

    return adjusted_config


def extract_number(path):
    """Return the extract number value owned by this scenario manager module."""
    match = re.search(r"(\d+)$", path)  # get number at the end
    return int(match.group(1)) if match else -1


class ScenarioManager(Steppable, LoggerInterface):
    """Definition of the ScenarioManager class for managing randomly generated scenarios as data source for training the
    reinforcement learning model. Generate random scenarios, regenerate them and store them.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.

    """

    def __init__(
        self,
        *,
        route_entities_provider: RouteEntitiesProvider | None = None,
        **kwargs: object,
    ) -> None:
        """
        Constructor for Scenarios class.
        """

        Steppable.__init__(self)
        LoggerInterface.__init__(self, **kwargs)

        kwargs = {
            "parent_logger_name": self._logger.name,
            "child_logger_name": "manager",
            "child_logger_level": self._logger.level,
        }

        self._configuration = ScenarioManagerConfigurationWrapper()
        self._scenarios: list[Scenario] = []
        self._index = 0
        self._finished = False
        self._cumulative_training_counter = 0
        self._scenario_index_state = None
        self._route_entities_provider = route_entities_provider

    def __getstate__(self):
        """Return serializable state for this scenario manager."""
        state = self.__dict__.copy()
        state["_scenario_index_state"] = None
        return state

    def __setstate__(self, state):
        """Restore this scenario manager from serialized state."""
        self.__dict__.update(state)
        if "_scenario_index_state" not in self.__dict__:
            self._scenario_index_state = None
        if "_route_entities_provider" not in self.__dict__:
            self._route_entities_provider = None

    def load(self):
        """
        Load the scenarios from the preserved directory.
        """

        if self._configuration.get_scenarios_folder_path() is None:
            self._logger.warning("Directory is not set. Cannot load scenarios.")
            return
        elif not existdir(self._configuration.get_scenarios_folder_path()):
            self._logger.warning(
                f"Directory {self._configuration.get_scenarios_folder_path()} does not exist. Cannot load scenarios."
            )
            makedirs(self._configuration.get_scenarios_folder_path(), exist_ok=True)
            self._logger.info(
                f"Directory {self._configuration.get_scenarios_folder_path()} created."
            )

        scenarios_paths = only_directories_from_dir(self._configuration.get_scenarios_folder_path())
        scenarios_paths = sorted(scenarios_paths, key=extract_number)

        loaded_scenarios: list[Scenario] = []
        for current_scenario_path in scenarios_paths:
            current_scenario_path = Path(current_scenario_path)
            scenario_name = current_scenario_path.name
            kwargs = {
                "parent_logger_name": self._logger.name,
                "child_logger_name": "manifest",
                "child_logger_level": self._logger.level,
            }
            _manifest = ScenarioManifest(str(current_scenario_path), **kwargs)
            if not _manifest.load():
                self._logger.warning(
                    f"Directory {current_scenario_path} is not a valid scenario folder. Skipping."
                )
            else:
                kwargs = {
                    "parent_logger_name": self._logger.name,
                    "child_logger_name": scenario_name,
                    "child_logger_level": self._logger.level,
                }
                _scenario = Scenario(_manifest, **kwargs)
                loaded_scenarios.append(_scenario)

        self._scenarios = loaded_scenarios

        self._logger.info(
            f"Scenarios loaded from {self._configuration.get_scenarios_folder_path()}. Found {len(self._scenarios)} scenarios."
        )

    def generate(self):
        """
        Generate random scenarios.
        Generate random vehicle trips and routes and manipulate pedestrians into route files.

        This method is a wrapper for the following steps:

        1) Generate random vehicle trips and routes
        2) Extend vehicle routes with pedestrians
        3) Generate SUMO configuration files
        """

        scenario_count = self._configuration.get_number_of_scenarios() + 1
        generation_inputs = self._resolve_generation_inputs()
        if self._has_current_artifact_inventory(
            scenario_count,
            generation_inputs,
        ):
            self._logger.info("We have enough training scenarios. Not needed for generating more.")
            return

        self._logger.info(
            f"Existing inventory has {len(self._scenarios)} training scenarios; "
            f"generating a fresh {scenario_count}-scenario distribution."
        )
        scenario_root = self._configuration.get_scenarios_folder_path()
        with ScenarioSetReplacement(scenario_root) as replacement:
            inputs = self._resolve_generation_inputs(
                scenario_root=str(replacement.staging_root),
            )
            for scenario_index in range(scenario_count):
                self._generate_scenario(scenario_index, inputs)
            replacement.publish()
        self.load()

    def _has_current_artifact_inventory(
        self,
        scenario_count: int,
        inputs: _GenerationInputs,
    ) -> bool:
        """Return whether artifacts match the active format and traffic distribution."""
        if len(self._scenarios) != scenario_count:
            return False
        for scenario_index, scenario in enumerate(self._scenarios):
            configuration = scenario.get_configuration()
            expected = self._scenario_counts(scenario_index, inputs)
            if (
                getattr(configuration, "artifact_format_version", None)
                != SCENARIO_ARTIFACT_FORMAT_VERSION
                or getattr(configuration, "number_of_vehicles", None) != expected.vehicles
                or getattr(configuration, "number_of_pedestrians", None) != expected.pedestrians
            ):
                return False
        return True

    def _resolve_generation_inputs(
        self,
        *,
        scenario_root: str | None = None,
    ) -> _GenerationInputs:
        """Resolve the generation inputs required by this scenario manager."""
        network_file = os.path.join(
            self._configuration.get_model_folder_path(),
            self._configuration.get_network_filename(),
        )
        return _GenerationInputs(
            scenario_root=(
                scenario_root
                if scenario_root is not None
                else self._configuration.get_scenarios_folder_path()
            ),
            network_file=network_file,
            vehicle_flow=(self._configuration.get_adaptive_vehicle_flow_configuration()),
            pedestrian_flow=(self._configuration.get_adaptive_pedestrian_flow_configuration()),
        )

    def _generate_scenario(
        self,
        scenario_index: int,
        inputs: _GenerationInputs,
    ) -> None:
        """Generate the scenario owned by this scenario manager."""
        counts = self._scenario_counts(scenario_index, inputs)
        self._logger.info(
            f"Generating scenario {scenario_index} with {counts.vehicles} "
            f"vehicles and {counts.pedestrians} pedestrians"
        )
        artifacts = self._prepare_vehicle_route_artifacts(
            inputs.scenario_root,
            scenario_index,
            counts.vehicles,
        )
        self._generate_vehicle_route(inputs.network_file, artifacts)
        route_file = self._extend_route_with_pedestrians(
            scenario_index,
            artifacts,
            counts.pedestrians,
        )
        self._generate_sumo_configuration(
            scenario_index,
            artifacts.scenario_directory,
            inputs.network_file,
            route_file,
        )
        self._publish_scenario_manifest(
            artifacts.scenario_directory,
            counts,
        )

    def _scenario_counts(
        self,
        scenario_index: int,
        inputs: _GenerationInputs,
    ) -> _ScenarioCounts:
        """Return the scenario counts value owned by this scenario manager."""
        progress_ratio = scenario_index / self._configuration.get_number_of_scenarios()
        return _ScenarioCounts(
            vehicles=_calculate_adaptive_count(
                inputs.vehicle_flow.initial_number_of_vehicles,
                inputs.vehicle_flow.target_number_of_vehicles,
                progress_ratio,
            ),
            pedestrians=_calculate_adaptive_count(
                inputs.pedestrian_flow.initial_number_of_pedestrians,
                inputs.pedestrian_flow.target_number_of_pedestrians,
                progress_ratio,
            ),
        )

    def _prepare_vehicle_route_artifacts(
        self,
        scenario_root: str,
        scenario_index: int,
        vehicle_count: int,
    ) -> _VehicleRouteArtifacts:
        """Prepare the vehicle route artifacts required by this scenario manager."""
        scenario_directory = os.path.join(
            scenario_root,
            f"{self._configuration.get_scenario_wrapper_directory_prefix()}_{scenario_index}",
        )
        makedirs(scenario_directory)
        trip_file = os.path.join(
            scenario_directory,
            f"{self._configuration.get_vehicle_trip_filename_prefix()}_{scenario_index}.trips.xml",
        )
        route_file = os.path.join(
            scenario_directory,
            f"{self._configuration.get_vehicle_route_filename_prefix()}_{scenario_index}.rou.xml",
        )
        vehicle_configuration = self._configuration.get_general_vehicles_configuration()
        simulation_time = self._configuration.get_vehicle_simulation_time()
        return _VehicleRouteArtifacts(
            scenario_directory=scenario_directory,
            trip_file=trip_file,
            route_file=route_file,
            generation_configuration=_adjust_vehicle_config_for_count(
                vehicle_configuration,
                simulation_time,
                vehicle_count,
            ),
        )

    def _generate_vehicle_route(
        self,
        network_file: str,
        artifacts: _VehicleRouteArtifacts,
    ) -> None:
        """Generate the vehicle route owned by this scenario manager."""
        route_entities_provider = self._route_entities_provider
        if route_entities_provider is None:
            raise RuntimeError("Scenario generation requires an injected route-entities provider.")
        generator = TripAndRouteGenerator(
            route_entities_provider=route_entities_provider,
            **self._child_logger_kwargs("random_vehicle_trip_and_route_generator"),
        )
        generated = generator.generate_random_vehicle_trip_and_route(
            artifacts.generation_configuration,
            network_file,
            artifacts.trip_file,
            artifacts.route_file,
        )
        if not generated:
            raise RuntimeError(
                "Vehicle route generation failed for scenario "
                f"'{os.path.basename(artifacts.scenario_directory)}'."
            )

    def _extend_route_with_pedestrians(
        self,
        scenario_index: int,
        artifacts: _VehicleRouteArtifacts,
        pedestrian_count: int,
    ) -> str:
        """Return the extend route with pedestrians value owned by this scenario manager."""
        route_file = os.path.join(
            artifacts.scenario_directory,
            f"{self._configuration.get_route_filename_prefix()}_{scenario_index}.rou.xml",
        )
        logger_kwargs = self._child_logger_kwargs(
            "pedestrian_route_xml_manipulator_extend_vehicle_route_with_pedestrians"
        )
        if not os.path.isfile(artifacts.route_file):
            raise RuntimeError(
                f"Vehicle route file missing after generation: '{artifacts.route_file}'."
            )
        PedestrianRouteXMLManipulator(**logger_kwargs).extend_vehicle_route_with_pedestrians(
            route_file,
            artifacts.route_file,
            pedestrian_count,
        )
        return route_file

    def _generate_sumo_configuration(
        self,
        scenario_index: int,
        scenario_directory: str,
        network_file: str,
        route_file: str,
    ) -> None:
        """Generate the sumo configuration owned by this scenario manager."""
        output_file = os.path.join(
            scenario_directory,
            f"smartcity_{scenario_index}.sumocfg",
        )
        netstate_dump = os.path.join(
            scenario_directory,
            f"virtualtraffic_{scenario_index}.xml",
        )
        gui_settings_file = os.path.join(
            self._configuration.get_model_folder_path(),
            self._configuration.get_gui_settings_filename(),
        )
        XMLManipulator(
            **self._child_logger_kwargs("xml_manipulator_generate_sumo_config")
        ).generate_smartcity_sumo_config(
            network_file,
            os.path.basename(route_file),
            output_file,
            os.path.basename(netstate_dump),
            gui_settings_file,
            self._configuration.get_begin_time(),
            self._configuration.get_end_time(),
            self._configuration.get_time_to_teleport(),
        )

    def _publish_scenario_manifest(
        self,
        scenario_directory: str,
        counts: _ScenarioCounts,
    ) -> None:
        """Publish the scenario manifest owned by this scenario manager."""
        manifest = ScenarioManifest(
            scenario_directory,
            **self._child_logger_kwargs("scenario_manifest_creation"),
        )
        configuration = manifest.get_configuration()
        configuration.name = Path(scenario_directory).name
        configuration.timestamp = datetime.now()
        configuration.number_of_vehicles = counts.vehicles
        configuration.number_of_pedestrians = counts.pedestrians
        configuration.artifact_format_version = SCENARIO_ARTIFACT_FORMAT_VERSION
        manifest.save()

    def _child_logger_kwargs(self, child_logger_name: str) -> dict[str, object]:
        """Return the child logger kwargs value owned by this scenario manager."""
        return {
            "parent_logger_name": self._logger.name,
            "child_logger_name": child_logger_name,
            "child_logger_level": self._logger.level,
        }

    def next(self):
        """
        Go to the next scenario if the training branch limit reached on the current scenario.

        :return: True if the training branch limit reached on the current scenario, False otherwise.
        """
        self._cumulative_training_counter += 1
        if self._index + 1 < len(self._scenarios):
            self._index += 1
        else:
            self._finished = True
        self._publish_index_state()

    def reset(self):
        """
        Reset the scenario manager.
        """
        self._index = 0
        self._finished = False
        self._publish_index_state()

    def finished(self) -> bool:
        """
        Check if the scenario manager finished the scenarios.

        :return: True if the scenario manager finished the scenarios, False otherwise.
        """
        return self._finished

    def get_configuration(self) -> ScenarioManagerConfigurationWrapper:
        """Return the configuration exposed by this scenario manager."""
        return self._configuration

    def get_scenario(self) -> Scenario:
        """Return the scenario exposed by this scenario manager."""
        return self._scenarios[self._index]

    def get_index(self) -> int:
        """Return the index exposed by this scenario manager."""
        return self._index

    def set_index(self, index: int):
        """Set the index owned by this scenario manager."""
        if 0 <= index < len(self._scenarios):
            self._index = index
            self._finished = False
            self._publish_index_state()
        else:
            self._logger.warning(f"Index {index} is out of range. Cannot set index.")

    def set_scenario_index_state(self, scenario_index_state):
        """Set the scenario index state owned by this scenario manager."""
        self._scenario_index_state = scenario_index_state
        self._publish_index_state()

    def get_scenario_index_state(self):
        """Return the scenario index state exposed by this scenario manager."""
        return self._scenario_index_state

    def get_cumulative_training_counter(self) -> int:
        """Return the cumulative training counter exposed by this scenario manager."""
        return self._cumulative_training_counter

    def get_total_number_of_scenarios(self) -> int:
        """Return the total number of scenarios exposed by this scenario manager."""
        return len(self._scenarios)

    def clone(self):
        """Return an independent copy of this scenario manager."""
        return deepcopy(self)

    def clone_for_evaluation_scenario(
        self,
        scenario_directory: str | os.PathLike[str],
    ) -> ScenarioManager:
        """Return an isolated manager containing one validated held-out scenario."""

        scenario_path = Path(scenario_directory)
        manifest = ScenarioManifest(
            str(scenario_path),
            **self._child_logger_kwargs("evaluation_scenario_manifest"),
        )
        if not manifest.load():
            raise RuntimeError(
                f"Evaluation scenario is missing a valid manifest: {scenario_path!s}."
            )
        cloned = self.clone()
        cloned._scenarios = [
            Scenario(
                manifest,
                **self._child_logger_kwargs("evaluation_scenario"),
            )
        ]
        cloned._index = 0
        cloned._finished = False
        cloned._cumulative_training_counter = 0
        cloned._scenario_index_state = None
        return cloned

    def _publish_index_state(self):
        """Publish the index state owned by this scenario manager."""
        if self._scenario_index_state is None:
            return
        set_fn = getattr(self._scenario_index_state, "set", None)
        if set_fn is None:
            return
        try:
            if hasattr(set_fn, "remote"):
                set_fn.remote(self._index)
            else:
                set_fn(self._index)
        except Exception as exc:
            self._logger.warning(f"Failed to publish scenario index state: {exc}")
