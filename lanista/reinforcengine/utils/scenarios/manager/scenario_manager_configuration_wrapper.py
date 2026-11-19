# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt configuration access for scenario manager configuration."""

from lanista.foundation.configuration import Configuration


class ScenarioManagerConfigurationWrapper(Configuration):
    """
    Base class for scenarios in reinforcement learning.
    This class should be inherited by specific scenario implementations.
    """

    def __init__(self):
        """
        Initialize the ScenarioManagerConfigurationWrapper with optional keyword arguments.
        """
        Configuration.__init__(self)

    def get_model_folder_path(self) -> str:
        """Return the model folder path exposed by this scenario manager configuration wrapper."""
        return self.model_folder_path

    def get_scenarios_folder_path(self) -> str:
        """Return the scenarios folder path exposed by this scenario manager configuration wrapper.
        """
        return self.scenarios_folder_path

    def get_number_of_scenarios(self) -> int:
        """Return the number of scenarios exposed by this scenario manager configuration wrapper."""
        return self.random_scenarios_configuration.number_of_scenarios

    def get_network_filename(self) -> str:
        """Return the network filename exposed by this scenario manager configuration wrapper."""
        return self.sumo_configuration.network_filename

    def get_scenario_wrapper_directory_prefix(self) -> str:
        """Return the scenario wrapper directory prefix exposed by this scenario manager
        configuration wrapper.
        """
        return self.random_scenarios_configuration.scenario_wrapper_directory_prefix

    def get_trip_filename_prefix(self) -> str:
        """Return the trip filename prefix exposed by this scenario manager configuration wrapper.
        """
        return self.random_scenarios_configuration.trip_filename_prefix

    def get_route_filename_prefix(self) -> str:
        """Return the route filename prefix exposed by this scenario manager configuration wrapper.
        """
        return self.random_scenarios_configuration.route_filename_prefix

    def get_general_vehicles_configuration(self):
        """Return the general vehicles configuration exposed by this scenario manager configuration
        wrapper.
        """
        return self.traffic_definition.general_vehicles_configuration

    def get_general_pedestrians_configuration(self):
        """Return the general pedestrians configuration exposed by this scenario manager
        configuration wrapper.
        """
        return self.traffic_definition.general_pedestrians_configuration

    def get_adaptive_vehicle_flow_configuration(self):
        """Return the adaptive vehicle flow configuration exposed by this scenario manager
        configuration wrapper.
        """
        return self.traffic_flow_definition.adaptive_vehicle_flow_configuration

    def get_adaptive_pedestrian_flow_configuration(self):
        """Return the adaptive pedestrian flow configuration exposed by this scenario manager
        configuration wrapper.
        """
        return self.traffic_flow_definition.adaptive_pedestrian_flow_configuration

    def get_vehicle_trip_filename_prefix(self):
        """Return the vehicle trip filename prefix exposed by this scenario manager configuration
        wrapper.
        """
        return self.traffic_definition.general_vehicles_configuration.trip_filename_prefix

    def get_pedestrian_trip_filename_prefix(self):
        """Return the pedestrian trip filename prefix exposed by this scenario manager
        configuration wrapper.
        """
        return self.traffic_definition.general_pedestrians_configuration.trip_filename_prefix

    def get_vehicle_route_filename_prefix(self):
        """Return the vehicle route filename prefix exposed by this scenario manager configuration
        wrapper.
        """
        return self.traffic_definition.general_vehicles_configuration.route_filename_prefix

    def get_pedestrian_route_filename_prefix(self):
        """Return the pedestrian route filename prefix exposed by this scenario manager
        configuration wrapper.
        """
        return self.traffic_definition.general_pedestrians_configuration.route_filename_prefix

    def get_gui_settings_filename(self):
        """Return the gui settings filename exposed by this scenario manager configuration wrapper.
        """
        return self.sumo_configuration.gui_settings_filename

    def get_begin_time(self):
        """Return the begin time exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.begin_time

    def get_end_time(self):
        """Return the end time exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.end_time

    def get_time_to_teleport(self):
        """Return the time to teleport exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.time_to_teleport

    def get_step_length(self) -> float:
        """Return the step length exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.step_length

    def get_no_warnings(self) -> bool:
        """Return the no warnings exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.no_warnings

    def get_vehicle_simulation_time(self):
        """Return the vehicle simulation time exposed by this scenario manager configuration
        wrapper.
        """
        return self.traffic_definition.general_vehicles_configuration.end_time - self.traffic_definition.general_vehicles_configuration.begin_time

    def get_pedestrian_simulation_time(self):
        """Return the pedestrian simulation time exposed by this scenario manager configuration
        wrapper.
        """
        return self.traffic_definition.general_pedestrians_configuration.end_time - self.traffic_definition.general_pedestrians_configuration.begin_time

    def get_max_timesteps(self):
        """Return the max timesteps exposed by this scenario manager configuration wrapper."""
        return self.simulation_configuration.max_timesteps
