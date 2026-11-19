# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""SUMO reads used to build scenario-context snapshots."""

import math
from dataclasses import dataclass

from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationOperationError,
    SimulationRead,
    SimulationReadQuality,
)


@dataclass(frozen=True)
class EgoLeaderContext:
    """Carry the runtime context for ego leader."""

    present: bool = False
    distance: float = 0.0
    speed_mps: float = 0.0
    quality: SimulationReadQuality = SimulationReadQuality.LIVE
    error: SimulationOperationError | None = None
    vehicle_id: str | None = None


@dataclass
class PreprocessorSumoContextService:
    """Coordinate preprocessor sumo context operations."""

    backend: EntitySimulationPort

    def read_ego_leader(self, lookahead_meters: float) -> EgoLeaderContext:
        """Read the ego leader without changing this preprocessor sumo context service's state."""
        read = self.backend.optional_read(
            "read Susan ego leader",
            lambda: self.backend.vehicle.getLeader("EGO", lookahead_meters),
            None,
        )
        if not read.value:
            return EgoLeaderContext(quality=read.quality, error=read.error)
        vehicle_id, distance = read.value
        try:
            distance = float(distance)
        except (TypeError, ValueError):
            return EgoLeaderContext()
        if not math.isfinite(distance) or distance < 0.0:
            return EgoLeaderContext(quality=read.quality, error=read.error)

        speed_read = self.backend.optional_read(
            "read Susan ego leader speed",
            lambda: self.backend.vehicle.getSpeed(str(vehicle_id)),
            0.0,
        )
        try:
            speed_mps = float(speed_read.value)
        except (TypeError, ValueError):
            speed_mps = 0.0
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            speed_mps = 0.0
        return EgoLeaderContext(
            distance=distance,
            speed_mps=speed_mps,
            present=True,
            vehicle_id=str(vehicle_id),
            quality=(
                SimulationReadQuality.FALLBACK
                if read.degraded or speed_read.degraded
                else SimulationReadQuality.LIVE
            ),
            error=read.error or speed_read.error,
        )

    def route_link_indices(self, traffic_light_id) -> SimulationRead[tuple[int, ...]]:
        """Return the route link indices value owned by this preprocessor sumo context service."""
        read = self.backend.optional_read(
            "read Susan traffic-light route links",
            lambda: (
                self.backend.vehicle.getLaneID("EGO"),
                self.backend.trafficlight.getControlledLinks(traffic_light_id),
            ),
            None,
        )
        if read.value is None:
            return SimulationRead((), read.quality, read.error)
        ego_lane_id, controlled_links = read.value
        indices = []
        for link_index, link_group in enumerate(controlled_links or []):
            if any(link and link[0] == ego_lane_id for link in (link_group or [])):
                indices.append(link_index)
        unique = tuple(sorted(set(indices)))
        return SimulationRead(
            unique if len(unique) == 1 else (),
            read.quality,
            read.error,
        )

    def signal_character(
        self,
        traffic_light_id,
        link_index: int,
    ) -> SimulationRead[str | None]:
        """Return the signal character value owned by this preprocessor sumo context service."""
        read = self.backend.optional_read(
            "read Susan traffic-light link state",
            lambda: self.backend.trafficlight.getRedYellowGreenState(traffic_light_id),
            None,
        )
        if read.value is None:
            return SimulationRead(None, read.quality, read.error)
        try:
            index = int(link_index)
        except (TypeError, ValueError):
            return SimulationRead.live(None)
        value = read.value[index] if 0 <= index < len(read.value) else None
        return SimulationRead(value, read.quality, read.error)
