# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt the existing SUMO scenario builders to evaluation-library generation."""

from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.utils.scenarios.evaluation_library.generation import (
    MaterializedScenario,
    ScenarioMaterializationRequest,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    _adjust_vehicle_config_for_count,
)
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.trip_and_route_generator import TripAndRouteGenerator
from lanista.reinforcengine.utils.xml_manipulators.pedestrian_route_xml_manipulator import (
    PedestrianRouteXMLManipulator,
)
from lanista.reinforcengine.utils.xml_manipulators.xml_manipulator import XMLManipulator


class ScenarioMaterializerConfiguration(Protocol):
    """Expose the existing generation settings consumed by the adapter."""

    def get_general_vehicles_configuration(self) -> object: ...

    def get_vehicle_simulation_time(self) -> float: ...

    def get_model_folder_path(self) -> str: ...

    def get_gui_settings_filename(self) -> str: ...

    def get_begin_time(self) -> float: ...

    def get_end_time(self) -> float: ...

    def get_time_to_teleport(self) -> float: ...


@dataclass(frozen=True, slots=True)
class _ScenarioArtifacts:
    network: Path
    gui_settings: Path
    trips: Path
    vehicle_route: Path
    final_route: Path
    sumo_configuration: Path


def _route_entity_vehicle_ids(provider: RouteEntitiesProvider) -> set[str]:
    fragment = "<routes>" + "\n".join(provider.route_entity_lines()) + "</routes>"
    return {
        element.attrib["id"]
        for element in ET.fromstring(fragment).iter("vehicle")
        if "id" in element.attrib
    }


def _normalize_route(path: Path) -> ET.Element:
    tree = ET.parse(path)
    ET.indent(tree, space="    ")
    ET.register_namespace("xsi", "http://www.w3.org/2001/XMLSchema-instance")
    tree.write(path, encoding="utf-8", xml_declaration=True, short_empty_elements=True)
    with path.open("ab") as handle:
        handle.write(b"\n")
    return tree.getroot()


def _scenario_artifacts(output: Path) -> _ScenarioArtifacts:
    return _ScenarioArtifacts(
        network=output / "network.net.xml",
        gui_settings=output / "gui-settings.xml",
        trips=output / "vehicle.trips.xml",
        vehicle_route=output / "vehicle.rou.xml",
        final_route=output / "scenario.rou.xml",
        sumo_configuration=output / "scenario.sumocfg",
    )


def _population_note(
    request: ScenarioMaterializationRequest,
    *,
    realised_vehicles: int,
    realised_pedestrians: int,
) -> str | None:
    if (
        realised_vehicles == request.requested_vehicle_count
        and realised_pedestrians == request.requested_pedestrian_count
    ):
        return None
    return (
        f"Generator realised {realised_vehicles} vehicles and "
        f"{realised_pedestrians} pedestrians from requested "
        f"{request.requested_vehicle_count} and "
        f"{request.requested_pedestrian_count}."
    )


def _write_scenario_manifest(
    request: ScenarioMaterializationRequest,
    *,
    realised_vehicles: int,
    realised_pedestrians: int,
) -> None:
    (request.output_directory / "scenario_manifest_configuration.yaml").write_text(
        "\n".join(
            (
                f"name: {request.scenario_id}",
                f"timestamp: {request.library_created_at}",
                f"number_of_vehicles: {realised_vehicles}",
                f"number_of_pedestrians: {realised_pedestrians}",
                "artifact_format_version: 2",
                "",
            )
        ),
        encoding="utf-8",
    )


class ConfiguredScenarioMaterializer:
    """Create portable scenario artifacts through the current SUMO builders."""

    def __init__(
        self,
        *,
        configuration: ScenarioMaterializerConfiguration,
        route_entities_provider: RouteEntitiesProvider,
    ) -> None:
        self._configuration = configuration
        self._route_entities_provider = route_entities_provider

    def materialize(self, request: ScenarioMaterializationRequest) -> MaterializedScenario:
        """Generate, normalize, measure, and retain only runtime-required artifacts."""

        # SUMO's child process cannot reopen a close-on-exec /proc/self/fd handle.
        output = request.output_directory.resolve(strict=True)
        if not output.is_dir() or any(output.iterdir()):
            raise RuntimeError("Scenario output directory must exist and be empty.")
        gui_source = Path(self._configuration.get_model_folder_path()) / (
            self._configuration.get_gui_settings_filename()
        )
        recorded_inputs = {path.resolve() for path in request.generator_input_paths}
        if gui_source.resolve() not in recorded_inputs:
            raise RuntimeError("GUI settings must be recorded as a generation input.")

        artifacts = _scenario_artifacts(output)
        shutil.copyfile(request.network_path, artifacts.network)
        shutil.copyfile(gui_source, artifacts.gui_settings)

        vehicle_configuration = _adjust_vehicle_config_for_count(
            self._configuration.get_general_vehicles_configuration(),
            self._configuration.get_vehicle_simulation_time(),
            request.requested_vehicle_count,
        )
        generated = TripAndRouteGenerator(
            route_entities_provider=self._route_entities_provider,
        ).generate_random_vehicle_trip_and_route(
            vehicle_configuration,
            str(artifacts.network),
            str(artifacts.trips),
            str(artifacts.vehicle_route),
            seed=request.seed,
        )
        if not generated or not artifacts.vehicle_route.is_file():
            raise RuntimeError(f"Vehicle route generation failed for {request.scenario_id}.")
        PedestrianRouteXMLManipulator().extend_vehicle_route_with_pedestrians(
            str(artifacts.final_route),
            str(artifacts.vehicle_route),
            request.requested_pedestrian_count,
            seed=request.seed,
        )
        route_root = _normalize_route(artifacts.final_route)
        XMLManipulator().generate_smartcity_sumo_config(
            artifacts.network.name,
            artifacts.final_route.name,
            str(artifacts.sumo_configuration),
            "virtualtraffic.xml",
            artifacts.gui_settings.name,
            self._configuration.get_begin_time(),
            self._configuration.get_end_time(),
            self._configuration.get_time_to_teleport(),
        )
        artifacts.trips.unlink(missing_ok=True)
        artifacts.vehicle_route.unlink(missing_ok=True)

        reserved_vehicle_ids = _route_entity_vehicle_ids(self._route_entities_provider)
        realised_vehicles = sum(
            element.attrib.get("id") not in reserved_vehicle_ids
            for element in route_root.iter("vehicle")
        )
        realised_pedestrians = sum(1 for _ in route_root.iter("person"))
        note = _population_note(
            request,
            realised_vehicles=realised_vehicles,
            realised_pedestrians=realised_pedestrians,
        )
        _write_scenario_manifest(
            request,
            realised_vehicles=realised_vehicles,
            realised_pedestrians=realised_pedestrians,
        )
        return MaterializedScenario(
            realised_vehicle_count=realised_vehicles,
            realised_pedestrian_count=realised_pedestrians,
            realisation_note=note,
        )


__all__ = ["ConfiguredScenarioMaterializer", "ScenarioMaterializerConfiguration"]
