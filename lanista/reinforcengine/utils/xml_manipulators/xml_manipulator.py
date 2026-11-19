# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""XML parsing and comment/metadata preservation helpers."""
import xml.etree.ElementTree as xml_ET
from xml.dom import minidom

from lanista.foundation.filesystem import readf
from lanista.foundation.logging import LoggerInterface


class XMLManipulator(LoggerInterface):
    """Apply controlled mutations for xml."""

    def __init__(self, **kwargs):
        """
        Constructor for xml_manipulator class.
        """
        super().__init__(**kwargs)

    @staticmethod
    def preserve_comments_and_metadata_from_xml(route_file):
        """
        Preserve comments and metadata from the XML file.

        :param route_file:
        """

        lines = readf(route_file)

        # Separate comments and XML content
        comments = []
        xml_lines = []

        for i in range(len(lines)):
            line = lines[i]
            if line.startswith("<!--"):
                while not line.endswith("-->"):
                    comments.append(line)
                    i += 1
                    line = lines[i]
                comments.append(line)
                comments.append("")
            elif line.startswith("<?xml"):
                xml_lines.append(line)

        return comments, xml_lines

    @staticmethod
    def sort_elements_by_departure_time(elements):
        """
        Sort elements by departure time.

        :param elements:
        """

        # Eliminate everything that is not a vehicle or person element
        elements = [x for x in elements if x.tag == 'vehicle' or x.tag == 'person']

        # Sort the elements by departure time
        elements.sort(key=lambda x: float(x.get('depart')))

        return elements


    def merge_routes(self, vehicle_route_file, pedestrian_route_file, route_file):
        """
        Merge vehicle and pedestrian routes by departure time.
        """

        # Parse the vehicle route file
        vehicle_tree = xml_ET.parse(vehicle_route_file)
        vehicle_root = vehicle_tree.getroot()
        vehicle_comments, xml_lines = self.preserve_comments_and_metadata_from_xml(vehicle_route_file)
        # xml_lines is same as in both vehicle and pedestrian route files

        # Parse the pedestrian route file
        pedestrian_tree = xml_ET.parse(pedestrian_route_file)
        pedestrian_root = pedestrian_tree.getroot()
        pedestrian_comments, xml_lines = self.preserve_comments_and_metadata_from_xml(pedestrian_route_file)
        # xml_lines is same as in both vehicle and pedestrian route files

        # Vehicle list of elements
        vehicle_elements = list(vehicle_root)

        # Pedestrian list of elements
        pedestrian_elements = list(pedestrian_root)

        # Merge the two lists
        elements = vehicle_elements + pedestrian_elements

        # Preserve and remove vTypes from vehicles
        vTypes = []
        for element in elements:
            if element.tag == "vType":
                vTypes.append(element)
        elements = [x for x in elements if x.tag != "vType"]

        # Sort the elements by departure time
        elements = self.sort_elements_by_departure_time(elements)

        # Create the new root element
        root = xml_ET.Element('routes')
        root.tail = '\n    '
        root.text = '\n    '
        root.set('xmlns:xsi', 'http://www.w3.org/2001/XMLSchema-instance')
        root.set('xsi:noNamespaceSchemaLocation', 'http://sumo.dlr.de/xsd/routes_file.xsd')
        root.extend(vTypes)
        root.extend(elements)

        # Convert to string and pretty print using minidom
        xml_str = xml_ET.tostring(root, encoding='utf-8', method='xml').decode()
        lines = xml_str.splitlines()
        lines = [""] + vehicle_comments + pedestrian_comments + lines

        # Because of a bug in formatting, we need to add manually an indent to th 5th line backwards in the list.
        lines[len(lines) - 5] = "    " + lines[len(lines) - 5]

        # Write the prettified XML to a file
        with open(route_file, 'w') as f:
            f.write(xml_lines[0] + "\n")
            for line in lines:
                f.write(line + "\n")

        self._logger.debug(f"Vehicle route file: {vehicle_route_file}")
        self._logger.debug(f"Pedestrian route file: {pedestrian_route_file}")
        self._logger.debug(f"Route file: {route_file}")
        # Log the merge operation
        self._logger.info(f"Merged {vehicle_route_file} and {pedestrian_route_file} into {route_file}.")
        self._logger.info(f"Route file '{route_file}' generated successfully with {len(elements)} elements.")


    def generate_smartcity_sumo_config(self, network_file, route_file, output_file, netstate_dump, gui_settings_file,
                                       begin=0, end=3600, time_to_teleport=300):
        """
        Generate the SUMO configuration file for each scenario.
            (smartcity.sumocfg)
        """

        # Create the root element
        config = xml_ET.Element('configuration', {
            'xmlns:xsi': 'http://www.w3.org/2001/XMLSchema-instance',
            'xsi:noNamespaceSchemaLocation': 'http://sumo.dlr.de/xsd/sumoConfiguration.xsd'
        })

        # Create input element
        input_element = xml_ET.SubElement(config, 'input')
        xml_ET.SubElement(input_element, 'net-file', {'value': network_file})
        xml_ET.SubElement(input_element, 'route-files', {'value': route_file})

        # Create output element
        output_element = xml_ET.SubElement(config, 'output')
        xml_ET.SubElement(output_element, 'netstate-dump', {'value': netstate_dump})

        # Create time element
        time_element = xml_ET.SubElement(config, 'time')
        xml_ET.SubElement(time_element, 'begin', {'value': str(begin)})
        xml_ET.SubElement(time_element, 'end', {'value': str(end)})

        # Create processing element
        processing_element = xml_ET.SubElement(config, 'processing')
        xml_ET.SubElement(processing_element, 'time-to-teleport', {'value': str(time_to_teleport)})

        # Create report element
        report_element = xml_ET.SubElement(config, 'report')
        xml_ET.SubElement(report_element, 'xml-validation', {'value': 'never'})
        xml_ET.SubElement(report_element, 'xml-validation.routes', {'value': 'never'})
        xml_ET.SubElement(report_element, 'duration-log.disable', {'value': 'true'})
        xml_ET.SubElement(report_element, 'no-step-log', {'value': 'true'})

        # Create gui_only element
        gui_only_element = xml_ET.SubElement(config, 'gui_only')
        xml_ET.SubElement(gui_only_element, 'gui-settings-file', {'value': gui_settings_file})

        # Convert to string and pretty print using minidom
        xml_str = xml_ET.tostring(config, encoding='unicode')
        pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="    ")

        # Write the prettified XML to a file
        with open(output_file, 'w') as f:
            f.write(pretty_xml_str)

        self._logger.debug(f"Network file: {network_file}")
        self._logger.debug(f"Route file: {route_file}")
        self._logger.debug(f"Output file: {output_file}")
        self._logger.debug(f"Netstate dump: {netstate_dump}")
        self._logger.debug(f"GUI settings file: {gui_settings_file}")
        self._logger.info(f"Generated SUMO configuration file '{output_file}' with the following parameters:")
        self._logger.info(f"  - Begin time: {begin}")
        self._logger.info(f"  - End time: {end}")
        self._logger.info(f"  - Time to teleport: {time_to_teleport}")
        # Log the successful generation of the configuration file
        self._logger.info(f"SUMO configuration file '{output_file}' generated successfully.")
