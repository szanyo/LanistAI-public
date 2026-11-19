# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared SUMO intersection metadata and route classification."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationOperationError,
    SimulationRead,
    SimulationReadQuality,
)

TRAFFIC_LIGHT_STATE_NOT_APPLICABLE = -1
TRAFFIC_LIGHT_STATE_UNKNOWN = 0
TRAFFIC_LIGHT_STATE_RED = 1
TRAFFIC_LIGHT_STATE_GREEN = 2
TRAFFIC_LIGHT_STATE_YELLOW = 3
TRAFFIC_LIGHT_STATE_RED_YELLOW = 4

TRAFFIC_LIGHT_STOP_PROFILE_STATE_CODES = frozenset(
    {
        TRAFFIC_LIGHT_STATE_RED,
        TRAFFIC_LIGHT_STATE_YELLOW,
        TRAFFIC_LIGHT_STATE_RED_YELLOW,
    }
)
TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES = frozenset(
    {
        TRAFFIC_LIGHT_STATE_RED,
        TRAFFIC_LIGHT_STATE_RED_YELLOW,
    }
)

INTERSECTION_TYPE_UNKNOWN = 0
INTERSECTION_TYPE_TRAFFIC_LIGHT = 1
INTERSECTION_TYPE_PEDESTRIAN_CROSSING = 2
INTERSECTION_TYPE_ROUNDABOUT = 3
INTERSECTION_TYPE_T_JUNCTION = 4
INTERSECTION_TYPE_PRIORITY_INTERSECTION = 5
INTERSECTION_TYPE_TERMINAL = 6

INTERSECTION_TYPE_KEYS = {
    INTERSECTION_TYPE_UNKNOWN: "unknown",
    INTERSECTION_TYPE_TRAFFIC_LIGHT: "traffic_light",
    INTERSECTION_TYPE_PEDESTRIAN_CROSSING: "pedestrian_crossing",
    INTERSECTION_TYPE_ROUNDABOUT: "roundabout",
    INTERSECTION_TYPE_T_JUNCTION: "t_junction",
    INTERSECTION_TYPE_PRIORITY_INTERSECTION: "priority_intersection",
    INTERSECTION_TYPE_TERMINAL: "terminal",
}

INTERSECTION_TYPE_LABELS = {
    INTERSECTION_TYPE_UNKNOWN: "Unknown",
    INTERSECTION_TYPE_TRAFFIC_LIGHT: "Traffic light",
    INTERSECTION_TYPE_PEDESTRIAN_CROSSING: "Pedestrian crossing",
    INTERSECTION_TYPE_ROUNDABOUT: "Roundabout",
    INTERSECTION_TYPE_T_JUNCTION: "T-junction",
    INTERSECTION_TYPE_PRIORITY_INTERSECTION: "Priority intersection",
    INTERSECTION_TYPE_TERMINAL: "Terminal",
}

_INTERNAL_EDGE_RE = re.compile(r"^:([^_]+)")
_TOPOLOGICAL_JUNCTION_TYPES = {
    "priority",
    "right_before_left",
    "allway_stop",
    "zipper",
    "unregulated",
}
_TERMINAL_JUNCTION_TYPES = {"dead_end"}


def get_intersection_type_key(type_code: int) -> str:
    """Return the intersection type key exposed by this intersection metadata module."""
    return INTERSECTION_TYPE_KEYS.get(int(type_code), "unknown")


def get_intersection_type_label(type_code: int) -> str:
    """Return the intersection type label exposed by this intersection metadata module."""
    return INTERSECTION_TYPE_LABELS.get(int(type_code), "Unknown")


def get_registered_intersection_types() -> list[dict[str, str | int]]:
    """Return the registered intersection types exposed by this intersection metadata module."""
    return [
        {
            "key": key,
            "code": code,
            "label": get_intersection_type_label(code),
        }
        for code, key in INTERSECTION_TYPE_KEYS.items()
    ]


def _junction_id_from_internal_edge(edge_id: str | None) -> str | None:
    """Return the junction id from internal edge value owned by this intersection metadata module.
    """
    if not edge_id:
        return None
    match = _INTERNAL_EDGE_RE.match(str(edge_id))
    if match is None:
        return None
    return match.group(1)


def _split_ids(value: str | None) -> set[str]:
    """Return the split ids value owned by this intersection metadata module."""
    if not value:
        return set()
    return {item for item in str(value).split() if item}


@dataclass
class SumoNetworkTopology:
    """Parsed SUMO topology used by model-neutral intersection classifiers."""

    edge_from_junction: dict[str, str] = field(default_factory=dict)
    edge_to_junction: dict[str, str] = field(default_factory=dict)
    edge_functions: dict[str, str] = field(default_factory=dict)
    junction_types: dict[str, str] = field(default_factory=dict)
    junction_incoming_edges: dict[str, set[str]] = field(default_factory=dict)
    junction_outgoing_edges: dict[str, set[str]] = field(default_factory=dict)
    traffic_light_junctions: set[str] = field(default_factory=set)
    pedestrian_crossing_junctions: set[str] = field(default_factory=set)
    roundabout_junctions: set[str] = field(default_factory=set)
    roundabout_edges: set[str] = field(default_factory=set)

    @classmethod
    def from_file(cls, network_file_path: str | Path) -> SumoNetworkTopology:
        """Parse one SUMO network without changing source ordering semantics."""

        topology = cls()
        root = ET.parse(network_file_path).getroot()
        topology._record_edges(root)
        topology._record_junctions(root)
        topology._record_connections(root)
        topology._record_roundabouts(root)
        return topology

    def _record_edges(self, root: ET.Element) -> None:
        """Record the edges owned by this sumo network topology."""
        for edge in root.findall("edge"):
            edge_id = edge.get("id")
            if not edge_id:
                continue

            edge_function = edge.get("function", "")
            self.edge_functions[edge_id] = edge_function
            if edge_function == "crossing":
                junction_id = _junction_id_from_internal_edge(edge_id)
                if junction_id:
                    self.pedestrian_crossing_junctions.add(junction_id)
                continue

            self._record_edge_junctions(edge_id, edge)

    def _record_edge_junctions(
        self,
        edge_id: str,
        edge: ET.Element,
    ) -> None:
        """Record the edge junctions owned by this sumo network topology."""
        from_junction = edge.get("from")
        to_junction = edge.get("to")
        if from_junction:
            self.edge_from_junction[edge_id] = from_junction
            self.junction_outgoing_edges.setdefault(
                from_junction,
                set(),
            ).add(edge_id)
        if to_junction:
            self.edge_to_junction[edge_id] = to_junction
            self.junction_incoming_edges.setdefault(
                to_junction,
                set(),
            ).add(edge_id)

    def _record_junctions(self, root: ET.Element) -> None:
        """Record the junctions owned by this sumo network topology."""
        for junction in root.findall("junction"):
            junction_id = junction.get("id")
            if not junction_id:
                continue

            junction_type = junction.get("type", "")
            self.junction_types[junction_id] = junction_type
            if junction_type == "traffic_light":
                self.traffic_light_junctions.add(junction_id)

            internal_lanes = _split_ids(junction.get("intLanes"))
            if any(
                "_w" in lane_id or "_c" in lane_id
                for lane_id in internal_lanes
            ):
                self.pedestrian_crossing_junctions.add(junction_id)

    def _record_connections(self, root: ET.Element) -> None:
        """Record the connections owned by this sumo network topology."""
        for connection in root.findall("connection"):
            traffic_light_id = connection.get("tl")
            if traffic_light_id:
                self.traffic_light_junctions.add(traffic_light_id)

            for attr_name in ("from", "to", "via"):
                attr_value = connection.get(attr_name)
                junction_id = _junction_id_from_internal_edge(attr_value)
                if junction_id and (
                    "_w" in str(attr_value) or "_c" in str(attr_value)
                ):
                    self.pedestrian_crossing_junctions.add(junction_id)

    def _record_roundabouts(self, root: ET.Element) -> None:
        """Record the roundabouts owned by this sumo network topology."""
        for roundabout in root.findall("roundabout"):
            self.roundabout_junctions.update(
                _split_ids(roundabout.get("nodes"))
            )
            self.roundabout_edges.update(
                _split_ids(roundabout.get("edges"))
            )

    def classify_junction(self, junction_id: str | None) -> int:
        """Classify the junction owned by this sumo network topology."""
        if not junction_id:
            return INTERSECTION_TYPE_UNKNOWN

        if junction_id in self.traffic_light_junctions:
            return INTERSECTION_TYPE_TRAFFIC_LIGHT
        if junction_id in self.pedestrian_crossing_junctions:
            return INTERSECTION_TYPE_PEDESTRIAN_CROSSING
        if junction_id in self.roundabout_junctions:
            return INTERSECTION_TYPE_ROUNDABOUT

        junction_type = self.junction_types.get(junction_id, "")
        incoming_edges = self.junction_incoming_edges.get(junction_id, set())
        outgoing_edges = self.junction_outgoing_edges.get(junction_id, set())

        if junction_type in _TERMINAL_JUNCTION_TYPES or not outgoing_edges:
            return INTERSECTION_TYPE_TERMINAL
        if junction_type not in _TOPOLOGICAL_JUNCTION_TYPES:
            return INTERSECTION_TYPE_UNKNOWN

        neighbor_count = self._neighbor_count(
            junction_id,
            incoming_edges,
            outgoing_edges,
        )
        if neighbor_count == 3:
            return INTERSECTION_TYPE_T_JUNCTION
        if neighbor_count >= 4:
            return INTERSECTION_TYPE_PRIORITY_INTERSECTION
        return INTERSECTION_TYPE_UNKNOWN

    def classify_edge_target(self, edge_id: str | None) -> int:
        """Classify the edge target owned by this sumo network topology."""
        if not edge_id:
            return INTERSECTION_TYPE_UNKNOWN
        if edge_id in self.roundabout_edges:
            return INTERSECTION_TYPE_ROUNDABOUT

        edge_function = self.edge_functions.get(edge_id, "")
        if edge_function == "crossing":
            return INTERSECTION_TYPE_PEDESTRIAN_CROSSING
        if edge_function == "internal":
            return self.classify_junction(
                _junction_id_from_internal_edge(edge_id)
            )
        return self.classify_junction(self.edge_to_junction.get(edge_id))

    def _neighbor_count(
        self,
        junction_id: str,
        incoming_edges: Iterable[str],
        outgoing_edges: Iterable[str],
    ) -> int:
        """Return the neighbor count value owned by this sumo network topology."""
        neighbors = set()
        for edge_id in incoming_edges:
            neighbor = self.edge_from_junction.get(edge_id)
            if neighbor and neighbor != junction_id:
                neighbors.add(neighbor)
        for edge_id in outgoing_edges:
            neighbor = self.edge_to_junction.get(edge_id)
            if neighbor and neighbor != junction_id:
                neighbors.add(neighbor)
        return len(neighbors)


class DynamicIntersectionTypeClassifier:
    """Classify the upcoming route feature from the active SUMO network."""

    def __init__(
        self,
        network_file_path: str | Path | None = None,
        *,
        lookahead_edges: int = 24,
    ) -> None:
        """Initialize the dynamic intersection type classifier and its owned state."""
        self._network_file_path = (
            Path(network_file_path) if network_file_path else None
        )
        self._lookahead_edges = lookahead_edges
        self._topology = (
            SumoNetworkTopology.from_file(self._network_file_path)
            if self._network_file_path is not None
            else None
        )

    def classify(
        self,
        sumo_module: EntitySimulationPort,
        ego_vehicle_id: str = "EGO",
        current_road_id: str | None = None,
    ) -> SimulationRead[int]:
        """Classify the operation owned by this dynamic intersection type classifier."""
        if self._topology is None:
            return self._classify_from_live_sumo(
                sumo_module,
                ego_vehicle_id,
            )

        empty_route: tuple[str, ...] = ()
        route_read = sumo_module.optional_read(
            "read intersection-classifier EGO route",
            lambda: sumo_module.vehicle.getRoute(ego_vehicle_id),
            empty_route,
        )
        route_index_read = sumo_module.optional_read(
            "read intersection-classifier EGO route index",
            lambda: sumo_module.vehicle.getRouteIndex(ego_vehicle_id),
            None,
        )
        road_read = self._read_current_road(
            sumo_module,
            ego_vehicle_id,
            current_road_id,
        )
        code = self._classify_from_topology(
            list(route_read.value),
            route_index_read.value,
            road_read.value,
        )
        return self._quality_result(
            code,
            (route_read.quality, route_read.error),
            (route_index_read.quality, route_index_read.error),
            (road_read.quality, road_read.error),
        )

    def _classify_from_topology(
        self,
        route_edges: list[str],
        route_index: object,
        current_edge: str | None,
    ) -> int:
        """Classify already-read route context without simulator side effects."""

        topology = self._topology
        if topology is None:
            return INTERSECTION_TYPE_UNKNOWN
        if current_edge:
            current_code = topology.classify_edge_target(current_edge)
            if current_code != INTERSECTION_TYPE_UNKNOWN:
                return current_code
        if not route_edges:
            return INTERSECTION_TYPE_UNKNOWN

        start_index = self._normalise_route_index(
            route_edges,
            route_index,
            current_edge,
        )
        route_window = route_edges[
            start_index : start_index + self._lookahead_edges
        ]
        for edge_id in route_window:
            code = topology.classify_edge_target(edge_id)
            if code != INTERSECTION_TYPE_UNKNOWN:
                return code

        if len(route_edges) - start_index <= 2:
            return INTERSECTION_TYPE_TERMINAL
        return INTERSECTION_TYPE_UNKNOWN

    @staticmethod
    def _read_current_road(
        sumo_module: EntitySimulationPort,
        ego_vehicle_id: str,
        current_road_id: str | None,
    ) -> SimulationRead[str | None]:
        """Read the current road without changing this dynamic intersection type classifier's
        state.
        """
        if current_road_id:
            return SimulationRead.live(current_road_id)
        return sumo_module.optional_read(
            "read intersection-classifier EGO road",
            lambda: sumo_module.vehicle.getRoadID(ego_vehicle_id),
            None,
        )

    @staticmethod
    def _quality_result(
        value: int,
        *evidence: tuple[
            SimulationReadQuality,
            SimulationOperationError | None,
        ],
    ) -> SimulationRead[int]:
        """Return the quality result value owned by this dynamic intersection type classifier."""
        for quality, error in evidence:
            if quality is SimulationReadQuality.LIVE:
                continue
            if error is None:
                raise SimulationOperationError(
                    "A fallback simulator read omitted its failure."
                )
            return SimulationRead.fallback(value, error)
        return SimulationRead.live(value)

    @staticmethod
    def _normalise_route_index(
        route_edges: list[str],
        route_index: object,
        current_edge: str | None,
    ) -> int:
        """Return the normalise route index value owned by this dynamic intersection type
        classifier.
        """
        if isinstance(route_index, int) and 0 <= route_index < len(route_edges):
            return route_index
        if current_edge in route_edges:
            return route_edges.index(current_edge)
        return 0

    @staticmethod
    def _classify_from_live_sumo(
        sumo_module: EntitySimulationPort,
        ego_vehicle_id: str,
    ) -> SimulationRead[int]:
        """Classify the from live sumo owned by this dynamic intersection type classifier."""
        empty_next_tls: tuple[tuple[object, ...], ...] = ()
        next_tls_read = sumo_module.optional_read(
            "read intersection-classifier EGO next traffic light",
            lambda: sumo_module.vehicle.getNextTLS(ego_vehicle_id),
            empty_next_tls,
        )
        code = (
            INTERSECTION_TYPE_TRAFFIC_LIGHT
            if next_tls_read.value
            else INTERSECTION_TYPE_UNKNOWN
        )
        return DynamicIntersectionTypeClassifier._quality_result(
            code,
            (next_tls_read.quality, next_tls_read.error),
        )
