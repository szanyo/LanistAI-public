# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""SUMO trip and route generation utilities."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

from lanista.foundation.filesystem import existf, readf, writef
from lanista.foundation.logging import LoggerInterface
from lanista.locations import get_random_route_generator_script_file_path
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.xml_manipulators.pedestrian_route_xml_manipulator import (
    PedestrianRouteXMLManipulator,
)

EGO_LINES_POSITION_IN_ROUTE_XML = 47
"""Position where would be the EGO vehicle lines in the route XML file."""

FIRST_VEHICLE_TRIP_DEFINITION_LINE_IN_TRIP_XML = 48
"""Position where would be the first vehicle trip definition line in the trip XML file."""

class TripAndRouteGenerator(LoggerInterface):
    """Definition of the Trip and Route Generator class.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.
    """

    def __init__(
        self,
        *,
        route_entities_provider: RouteEntitiesProvider,
        **kwargs: object,
    ) -> None:
        """
        Constructor for trip_and_route_generator class.
        """
        LoggerInterface.__init__(self, **kwargs)
        self._route_entities_provider = route_entities_provider
        kwargs = {
            "parent_logger_name": self._logger.name,
            "child_logger_name": "pedestrian_route_xml_manipulator",
            "child_logger_level": self._logger.level
        }
        self._pedestrian_route_xml_manipulator = PedestrianRouteXMLManipulator(**kwargs)

    @staticmethod
    def _resolve_configured_sumo_bin_dir() -> str:
        """Resolve the configured sumo bin dir required by this trip and route generator."""
        script_path = get_random_route_generator_script_file_path()
        return os.path.join(os.path.dirname(os.path.dirname(script_path)), "bin")

    @classmethod
    def _candidate_sumo_bin_dirs(cls) -> list[str]:
        """Return the candidate sumo bin dirs value owned by this trip and route generator."""
        candidates: list[str] = []

        configured_bin = cls._resolve_configured_sumo_bin_dir()
        candidates.append(configured_bin)

        python_bin = os.path.dirname(sys.executable)
        candidates.append(python_bin)

        for site_path in sys.path:
            if not site_path or "site-packages" not in site_path:
                continue
            candidates.append(os.path.join(site_path, "sumo", "bin"))

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = str(Path(candidate).resolve()) if candidate else ""
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            unique_candidates.append(normalized)
        return unique_candidates

    def _resolve_duarouter_path(self) -> str | None:
        """Resolve the duarouter path required by this trip and route generator."""
        discovered = shutil.which("duarouter")
        if discovered:
            return discovered

        for bin_dir in self._candidate_sumo_bin_dirs():
            candidate = os.path.join(bin_dir, "duarouter")
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
        return None

    def _build_route_generation_env(self) -> dict[str, str]:
        """Build the route generation env required by this trip and route generator."""
        env = dict(os.environ)
        current_path = str(env.get("PATH", "") or "")
        path_entries = [entry for entry in current_path.split(os.pathsep) if entry]
        for sumo_bin_dir in reversed(self._candidate_sumo_bin_dirs()):
            if sumo_bin_dir not in path_entries:
                path_entries.insert(0, sumo_bin_dir)
        env["PATH"] = os.pathsep.join(path_entries)
        return env

    def _generate_random_trip_and_route(
        self,
        command_configuration,
        network_file,
        trip_file,
        route_file,
        vehicle_mode=True,
        *,
        seed: int | None = None,
    ):
        """
        Generate a random scenario with vehicles or pedestrians.
        """

        duarouter_path = self._resolve_duarouter_path()
        if duarouter_path is None:
            self._logger.error(
                f"Cannot generate random trips because 'duarouter' is not available. "
                f"Checked PATH and candidate directories: {', '.join(self._candidate_sumo_bin_dirs())}."
            )
            return False

        # Combine the command for generating random trips
        route_generation_command = [
            sys.executable, get_random_route_generator_script_file_path(),
            "--net-file", network_file,
            "--end", str(command_configuration.end_time),
            "--route-file", route_file,
            "--output-trip-file", trip_file,
            "--validate",
            "--period", str(command_configuration.period),
            "--min-distance", str(command_configuration.min_distance),
            "--fringe-factor", str(command_configuration.fringe_factor),
            "--trip-attributes", command_configuration.trip_attributes,
        ]
        if seed is None:
            route_generation_command.append("--random")
        else:
            if type(seed) is not int or seed <= 0:
                raise ValueError("randomTrips seed must be a positive integer.")
            route_generation_command += ["--seed", str(seed)]

        if vehicle_mode:
            route_generation_command += [
                "--vehicle-class", command_configuration.vehicle_class,
            ]
        else:
            # Add vehicle-specific parameters
            route_generation_command += [
                "--pedestrian",
                "--max-distance", str(command_configuration.max_distance),
                "--prefix", command_configuration.prefix,
            ]

        # Run the command
        try:
            subprocess.run(
                route_generation_command,
                check=True,
                capture_output=True,
                text=True,
                env=self._build_route_generation_env(),
            )
            return True
        except subprocess.CalledProcessError as e:
            self._logger.error("Error executing route generation command:")
            self._logger.error(f"Command: {' '.join(route_generation_command)}")
            self._logger.error(f"Return code: {e.returncode}")
            if e.stdout:
                self._logger.error(f"Output: {e.stdout.strip()}")
            if e.stderr:
                self._logger.error(f"Errors: {e.stderr.strip()}")
            return False

    @staticmethod
    def _insert_route_entity_lines_into_route_xml(
        lines: list[str],
        route_entities_provider: RouteEntitiesProvider,
    ) -> list[str]:
        """Insert injected domain entities immediately before the routes closing tag."""
        closing_routes_index = None
        for index in range(len(lines) - 1, -1, -1):
            if lines[index].strip() == "</routes>":
                closing_routes_index = index
                break

        if closing_routes_index is None:
            closing_routes_index = len(lines)

        return (
            lines[:closing_routes_index]
            + route_entities_provider.route_entity_lines()
            + lines[closing_routes_index:]
        )

    def generate_random_vehicle_trip_and_route(
        self,
        command_configuration,
        network_file,
        trip_file,
        route_file,
        *,
        seed: int | None = None,
    ):
        """
        Generate a random scenario with vehicles.
        """

        if not self._generate_random_trip_and_route(
            command_configuration,
            network_file,
            trip_file,
            route_file,
            seed=seed,
        ):
            self._logger.error("Failed to generate random trip and route.")
            return False

        if not existf(route_file):
            self._logger.error(f"Route file {route_file} not found.")
            return False

        lines = readf(route_file)

        # Add EGO vehicle inside the routes root. randomTrips may generate an
        # empty <routes> file, where the legacy fixed insertion line points
        # after the closing root tag.
        tmp_lines = self._insert_route_entity_lines_into_route_xml(
            lines,
            self._route_entities_provider,
        )

        if command_configuration.end_time == command_configuration.period:
            self._logger.debug(f"End time {command_configuration.end_time} is equal to vehicle generation period {command_configuration.period}. Zero automatically simulated vehicle lines will be added to {route_file}.")


        buffer_line = ""
        for line in tmp_lines:
            buffer_line += line + "\n"

        writef(route_file, buffer_line)
        self._logger.info(f"Route file {route_file} generated successfully with EGO vehicle lines.")

        if command_configuration.end_time == command_configuration.period:
            self._logger.debug(f"Zero automatically simulated vehicle lines will be added to {route_file}.")
            lines = readf(trip_file)

            tmp_lines = lines[:FIRST_VEHICLE_TRIP_DEFINITION_LINE_IN_TRIP_XML]
            tmp_lines += lines[len(lines) - 1:]

            buffer_line = ""
            for line in tmp_lines:
                buffer_line += line + "\n"

            writef(trip_file, buffer_line)
            self._logger.info(f"Trip file {trip_file} generated successfully without any vehicle trip definitions.")

        return True


    def generate_random_pedestrian_trip_and_route(
        self,
        command_configuration,
        network_file,
        trip_file,
        route_file,
        *,
        seed: int | None = None,
    ):
        """
        Generate a random scenario with pedestrians.
        """

        if not self._generate_random_trip_and_route(
            command_configuration,
            network_file,
            trip_file,
            route_file,
            vehicle_mode=False,
            seed=seed,
        ):
            self._logger.error("Failed to generate random pedestrian trip and route.")
            return False

        if not existf(route_file):
            self._logger.error(f"Route file {route_file} not found.")
            return False

        # Set desiredMaxSpeed for pedestrians
        self._pedestrian_route_xml_manipulator.set_desired_max_speed_for_pedestrians(command_configuration.pedestrians,
                                                                            route_file)
        return True
