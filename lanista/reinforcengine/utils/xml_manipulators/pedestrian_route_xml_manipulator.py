# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pedestrian route XML mutation utilities."""
import random
from xml.etree import ElementTree as xml_ET

from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.utils.xml_manipulators.xml_manipulator import XMLManipulator


class PedestrianRouteXMLManipulator(LoggerInterface):
    """
    pedestrian_route_xml_manipulator class for manipulating pedestrian route XML files.

    This class is responsible for setting desiredMaxSpeed for pedestrians and extending vehicle routes with pedestrian routes.
    """

    def __init__(self, **kwargs):
        """
        Constructor for pedestrian_route_xml_manipulator class.
        """
        super().__init__(**kwargs)

    def set_desired_max_speed_for_pedestrians(self, pedestrian_configuration, route_file):
        """
        Set desiredMaxSpeed for pedestrians in the route file
        """

        # Preserve comments and metadata from the XML file
        comments, xml_lines = XMLManipulator.preserve_comments_and_metadata_from_xml(route_file)

        tree = xml_ET.parse(route_file)
        root = tree.getroot()

        for person in root.iter('person'):
            person.set('desiredMaxSpeed', str(pedestrian_configuration.desired_max_speed))
            person.attrib.pop('type')

        xml_str = xml_ET.tostring(root, encoding='utf-8', method='xml').decode()

        lines = xml_str.splitlines()

        lines = [""] + comments + lines

        # Write the prettified XML to a file
        with open(route_file, 'w') as f:
            f.write(xml_lines[0] + "\n")
            for line in lines:
                f.write(line + "\n")

        self._logger.info(f"Updated {route_file} with desiredMaxSpeed for pedestrians set to {pedestrian_configuration.desired_max_speed} m/s.")


    def extend_vehicle_route_with_pedestrians(
        self,
        route_file,
        vehicle_route_file,
        desired_number_of_pedestrians,
        *,
        seed: int | None = None,
    ):
        """
        Extend vehicle route file with pedestrian routes.

        :param route_file:
        :param vehicle_route_file:
        :param desired_number_of_pedestrians:
        """

        tree = xml_ET.parse(vehicle_route_file)
        root = tree.getroot()
        comments, xml_lines = self._preserve_route_prologue(vehicle_route_file)
        vtypes, vehicle_elements = self._partition_vehicle_elements(list(root))
        pedestrian_elements = self._build_pedestrian_elements(vehicle_elements)
        if seed is not None and (type(seed) is not int or seed <= 0):
            raise ValueError("Pedestrian selection seed must be a positive integer.")
        random_source = random if seed is None else random.Random(seed)
        self._retain_desired_pedestrian_count(
            pedestrian_elements,
            desired_number_of_pedestrians,
            random_source=random_source,
        )
        merged_elements = XMLManipulator.sort_elements_by_departure_time(
            vehicle_elements + pedestrian_elements
        )
        lines = self._render_route_lines(vtypes, merged_elements, comments)
        self._write_route(route_file, xml_lines, lines)

        self._logger.info(
            f"Extended {vehicle_route_file} with "
            f"{len(pedestrian_elements)} pedestrians."
        )
        self._logger.info(f"Updated {route_file} with pedestrian routes.")

    def _preserve_route_prologue(
        self,
        vehicle_route_file,
    ) -> tuple[list[str], list[str]]:
        """Return the preserve route prologue value owned by this pedestrian route xml manipulator.
        """
        kwargs = {
            'parent_logger_name': self._logger.name,
            'child_logger_name': f"{str(vehicle_route_file).lower()}_pedestrian_route_xml_manipulator",
            'child_logger_level': self._logger.level
        }
        return XMLManipulator(
            **kwargs
        ).preserve_comments_and_metadata_from_xml(vehicle_route_file)

    @staticmethod
    def _partition_vehicle_elements(
        vehicle_elements: list[xml_ET.Element],
    ) -> tuple[list[xml_ET.Element], list[xml_ET.Element]]:
        """Return the partition vehicle elements value owned by this pedestrian route xml
        manipulator.
        """
        vtypes = []
        for element in vehicle_elements:
            if element.tag == "vType":
                vtypes.append(element)
        vehicle_elements = [x for x in vehicle_elements if x.tag != "vType"]
        return vtypes, vehicle_elements

    @staticmethod
    def _build_pedestrian_elements(
        vehicle_elements: list[xml_ET.Element],
    ) -> list[xml_ET.Element]:
        """Build the pedestrian elements required by this pedestrian route xml manipulator."""
        pedestrian_elements = []
        for vehicle in vehicle_elements:
            new_element = xml_ET.Element('person', attrib={'id': f"ped_{vehicle.attrib['id']}",
                                                           'depart': vehicle.attrib['depart'],
                                                           'desiredMaxSpeed': '0.6',
                                                           'departPos': '20', })
            new_element.tail = '\n    '
            new_element.text = '\n        '
            sub_element = xml_ET.Element('walk', attrib={'from': '-5018.1.00', 'to': '-5020.0.00'})
            sub_element.tail = '\n    '
            new_element.append(sub_element)
            pedestrian_elements.append(new_element)
        return pedestrian_elements

    @staticmethod
    def _retain_desired_pedestrian_count(
        pedestrian_elements: list[xml_ET.Element],
        desired_number_of_pedestrians,
        *,
        random_source=random,
    ) -> None:
        """Retain the desired pedestrian count owned by this pedestrian
route xml manipulator."""
        while len(pedestrian_elements) > desired_number_of_pedestrians:
            random_index = random_source.randint(0, len(pedestrian_elements) - 1)
            pedestrian_elements.pop(random_index)

    @staticmethod
    def _render_route_lines(
        vtypes: list[xml_ET.Element],
        merged_elements: list[xml_ET.Element],
        comments: list[str],
    ) -> list[str]:
        """Return the render route lines value owned by this pedestrian route xml manipulator."""
        root = xml_ET.Element('routes')
        root.tail = '\n    '
        root.text = '\n    '
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        root.extend(vtypes)
        root.extend(merged_elements)

        xml_str = xml_ET.tostring(root, encoding='utf-8', method='xml').decode()
        lines = xml_str.splitlines()
        lines = [""] + comments + lines

        # Preserve the historic SUMO artifact formatting.
        lines[len(lines) - 5] = "    " + lines[len(lines) - 5]
        if lines[len(lines) - 2].startswith("    </routes>"):
            lines[len(lines)-2] = lines[len(lines) - 2][4:]
        return lines

    @staticmethod
    def _write_route(
        route_file,
        xml_lines: list[str],
        lines: list[str],
    ) -> None:
        """Persist the route owned by this pedestrian route xml manipulator."""
        with open(route_file, 'w') as f:
            f.write(xml_lines[0] + "\n")
            for line in lines:
                f.write(line + "\n")
