# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Narrow simulator contracts consumed by engine and model components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable


class SimulationLifecycleError(RuntimeError):
    """Base error for mandatory simulator lifecycle operations."""


class SimulationOperationError(SimulationLifecycleError):
    """A mandatory simulator operation failed at the native adapter boundary."""


class SimulationNotStartedError(SimulationOperationError):
    """Raised when a simulator command has no active session."""


class SimulationAlreadyStartedError(SimulationLifecycleError):
    """Raised when a second session is requested on one adapter."""


class SimulationReadQuality(StrEnum):
    """Whether an optional read contains live simulator data or a fallback."""

    LIVE = "live"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class SimulationRead[ReadValue]:
    """Value and provenance returned by an explicitly optional simulator read."""

    value: ReadValue
    quality: SimulationReadQuality = SimulationReadQuality.LIVE
    error: SimulationOperationError | None = None

    @property
    def degraded(self) -> bool:
        """Return whether the value is a declared fallback."""

        return self.quality is SimulationReadQuality.FALLBACK

    @classmethod
    def live(cls, value: ReadValue) -> SimulationRead[ReadValue]:
        """Build a successful live read."""

        return cls(value=value)

    @classmethod
    def fallback(
        cls,
        value: ReadValue,
        error: SimulationOperationError,
    ) -> SimulationRead[ReadValue]:
        """Build an observable fallback read."""

        return cls(
            value=value,
            quality=SimulationReadQuality.FALLBACK,
            error=error,
        )


class OptionalSimulationReadPort(Protocol):
    """Convert an explicitly optional operation into a quality-bearing result."""

    def optional_read[ReadValue](
        self,
        operation: str,
        reader: Callable[[], ReadValue],
        fallback: ReadValue,
    ) -> SimulationRead[ReadValue]:
        """Run a read and expose any fallback instead of hiding its failure."""


class VehicleEntityPort(Protocol):
    """Unit-explicit vehicle entity API used by model preprocessors."""

    def setSpeedMode(self, vehicle_id: str, speed_mode: int) -> None:
        """Set the speed mode owned by this vehicle entity port."""
        ...

    def setSpeed(self, vehicle_id: str, speed_mps: float) -> None:
        """Set the speed owned by this vehicle entity port."""
        ...

    def getIDCount(self) -> int:
        """Return the id count exposed by this vehicle entity port."""
        ...

    def getIDList(self) -> tuple[str, ...]:
        """Return the id list exposed by this vehicle entity port."""
        ...

    def getPosition(self, vehicle_id: str) -> tuple[float, float]:
        """Return the position exposed by this vehicle entity port."""
        ...

    def getSpeed(self, vehicle_id: str) -> float:
        """Return the speed exposed by this vehicle entity port."""
        ...

    def getRoadID(self, vehicle_id: str) -> str:
        """Return the road id exposed by this vehicle entity port."""
        ...

    def getLaneID(self, vehicle_id: str) -> str:
        """Return the lane id exposed by this vehicle entity port."""
        ...

    def getLanePosition(self, vehicle_id: str) -> float:
        """Return the longitudinal position on the current lane in metres."""
        ...

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]:
        """Return the route exposed by this vehicle entity port."""
        ...

    def getRouteIndex(self, vehicle_id: str) -> int:
        """Return the route index exposed by this vehicle entity port."""
        ...

    def getNextTLS(self, vehicle_id: str) -> tuple[tuple[object, ...], ...]:
        """Return the next tls exposed by this vehicle entity port."""
        ...

    def getLeader(
        self,
        vehicle_id: str,
        lookahead_m: float,
    ) -> tuple[str, float] | None:
        """Return the leader exposed by this vehicle entity port."""
        ...


class LaneEntityPort(Protocol):
    """Lane membership reads required by object sampling."""

    def getLastStepVehicleIDs(self, lane_id: str) -> tuple[str, ...]:
        """Return the last step vehicle i ds exposed by this lane entity port."""
        ...


class PersonEntityPort(Protocol):
    """Pedestrian observations used by crossing strategies."""

    def getIDCount(self) -> int:
        """Return the id count exposed by this person entity port."""
        ...

    def getIDList(self) -> tuple[str, ...]:
        """Return the id list exposed by this person entity port."""
        ...

    def getPosition(self, person_id: str) -> tuple[float, float]:
        """Return the position exposed by this person entity port."""
        ...

    def getSpeed(self, person_id: str) -> float:
        """Return the speed exposed by this person entity port."""
        ...


class TrafficLightEntityPort(Protocol):
    """Traffic-light observations and commands exposed to model code."""

    def getRedYellowGreenState(self, traffic_light_id: str) -> str:
        """Return the red yellow green state exposed by this traffic light entity port."""
        ...

    def setRedYellowGreenState(self, traffic_light_id: str, state: str) -> None:
        """Set the red yellow green state owned by this traffic light entity port."""
        ...

    def getPhase(self, traffic_light_id: str) -> int:
        """Return the phase exposed by this traffic light entity port."""
        ...

    def setPhase(self, traffic_light_id: str, phase: int) -> None:
        """Set the phase owned by this traffic light entity port."""
        ...

    def getControlledLinks(
        self,
        traffic_light_id: str,
    ) -> tuple[tuple[tuple[object, ...], ...], ...]:
        """Return the controlled links exposed by this traffic light entity port."""
        ...

    def getProgram(self, traffic_light_id: str) -> str:
        """Return the program exposed by this traffic light entity port."""
        ...

    def getAllProgramLogics(self, traffic_light_id: str) -> tuple[object, ...]:
        """Return the all program logics exposed by this traffic light entity port."""
        ...

    def getSpentDuration(self, traffic_light_id: str) -> float:
        """Return the spent duration exposed by this traffic light entity port."""
        ...

    def getPhaseDuration(self, traffic_light_id: str) -> float:
        """Return the phase duration exposed by this traffic light entity port."""
        ...

    def getNextSwitch(self, traffic_light_id: str) -> float:
        """Return the next switch exposed by this traffic light entity port."""
        ...


class SimulationEntityPort(Protocol):
    """Simulation clock values used by telemetry calculations."""

    def getTime(self) -> float:
        """Return the time exposed by this simulation entity port."""
        ...

    def getDeltaT(self) -> float:
        """Return the delta t exposed by this simulation entity port."""
        ...


@runtime_checkable
class EntitySimulationPort(OptionalSimulationReadPort, Protocol):
    """Grouped typed entity ports backed by one simulator session owner."""

    vehicle: VehicleEntityPort
    lane: LaneEntityPort
    person: PersonEntityPort
    trafficlight: TrafficLightEntityPort
    simulation: SimulationEntityPort


@runtime_checkable
class SimulationLifecyclePort(Protocol):
    """Start, advance, inspect, and close one simulator session."""

    def start_simulation(self, gui: bool, sumo_config_path: str) -> None:
        """Start one configured session."""

    def sim_step(self) -> None:
        """Advance the active simulation or propagate a fatal failure."""

    def shutdown_simulation(self) -> bool:
        """Close the session, returning whether one was active."""

    def is_initialized(self) -> bool:
        """Return whether this adapter owns an active session."""

    def get_simulation_time(self) -> float:
        """Return simulation time in seconds."""


@runtime_checkable
class VehiclePort(Protocol):
    """Commands and observations for simulator vehicle entities."""

    def set_vehicle_speed_mode(self, vehicle_id: str, speed_mode: int) -> None:
        """Set the SUMO bitset controlling vehicle speed checks."""

    def set_vehicle_speed(self, vehicle_id: str, speed_mps: float) -> None:
        """Set vehicle speed in metres per second."""

    def get_vehicle_id_count(self) -> int:
        """Return the number of active vehicle identifiers."""

    def get_vehicle_ids(self) -> tuple[str, ...]:
        """Return the active vehicle identifiers."""

    def has_vehicle(self, vehicle_id: str) -> bool:
        """Return whether the named vehicle is currently observable."""

    def get_pedestrian_id_count(self) -> int:
        """Return the number of active pedestrian identifiers."""

    def get_vehicle_road_id(self, vehicle_id: str) -> str:
        """Return the current SUMO road identifier for a vehicle."""

    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Return vehicle speed in metres per second."""

    def get_simulation_step_length(self) -> float:
        """Return the configured simulation step length in seconds."""


@runtime_checkable
class TrafficLightPort(Protocol):
    """Small traffic-light command/query boundary."""

    def get_traffic_light_state(self, traffic_light_id: str) -> str:
        """Return the encoded red/yellow/green state."""

    def set_traffic_light_state(self, traffic_light_id: str, state: str) -> None:
        """Apply an encoded red/yellow/green state."""

    def get_traffic_light_phase(self, traffic_light_id: str) -> int:
        """Return the active phase index."""

    def set_traffic_light_phase(self, traffic_light_id: str, phase: int) -> None:
        """Select a phase by index."""


@runtime_checkable
class TrafficLightTimingPort(Protocol):
    """Timing observations used by traffic-light telemetry."""

    def get_program(self, traffic_light_id: str) -> str:
        """Return the program exposed by this traffic light timing port."""
        ...

    def get_program_logics(self, traffic_light_id: str) -> tuple[object, ...]:
        """Return the program logics exposed by this traffic light timing port."""
        ...

    def get_phase(self, traffic_light_id: str) -> int:
        """Return the phase exposed by this traffic light timing port."""
        ...

    def get_spent_duration(self, traffic_light_id: str) -> float:
        """Return the spent duration exposed by this traffic light timing port."""
        ...

    def get_phase_duration(self, traffic_light_id: str) -> float:
        """Return the phase duration exposed by this traffic light timing port."""
        ...

    def get_next_switch(self, traffic_light_id: str) -> float:
        """Return the next switch exposed by this traffic light timing port."""
        ...


@runtime_checkable
class SimulatorPort(
    SimulationLifecyclePort,
    VehiclePort,
    TrafficLightPort,
    TrafficLightTimingPort,
    EntitySimulationPort,
    Protocol,
):
    """Complete typed surface implemented by the shared native adapter."""
