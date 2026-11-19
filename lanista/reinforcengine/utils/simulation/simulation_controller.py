# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""libsumo adapter implementing the engine's narrow simulation ports."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar, cast

import libsumo as sumo

from lanista.reinforcengine.utils.simulation.native_gui_usage import (
    PROCESS_NATIVE_GUI_USAGE,
    NativeGuiUsageTracker,
)
from lanista.reinforcengine.utils.simulation.ports import (
    LaneEntityPort,
    PersonEntityPort,
    SimulationAlreadyStartedError,
    SimulationEntityPort,
    SimulationLifecycleError,
    SimulationNotStartedError,
    SimulationOperationError,
    SimulationRead,
    TrafficLightEntityPort,
    VehicleEntityPort,
)
from lanista.reinforcengine.utils.simulation.sumo_configuration_rebaser import (
    ManagedSumoConfigurationError,
    SumoConfigurationMaterialization,
    materialize_sumo_configuration,
)

OperationValue = TypeVar("OperationValue")


class _VehicleApi(Protocol):
    """Define the typed boundary for vehicle api."""

    def setSpeedMode(self, vehicle_id: str, speed_mode: int) -> None:
        """Set the speed mode owned by this vehicle api."""
        ...
    def setSpeed(self, vehicle_id: str, speed_mps: float) -> None:
        """Set the speed owned by this vehicle api."""
        ...
    def getIDCount(self) -> int:
        """Return the id count exposed by this vehicle api."""
        ...
    def getIDList(self) -> Sequence[str]:
        """Return the id list exposed by this vehicle api."""
        ...
    def getPosition(self, vehicle_id: str) -> Sequence[float]:
        """Return the position exposed by this vehicle api."""
        ...
    def getSpeed(self, vehicle_id: str) -> float:
        """Return the speed exposed by this vehicle api."""
        ...
    def getRoadID(self, vehicle_id: str) -> str:
        """Return the road id exposed by this vehicle api."""
        ...
    def getLaneID(self, vehicle_id: str) -> str:
        """Return the lane id exposed by this vehicle api."""
        ...
    def getLanePosition(self, vehicle_id: str) -> float:
        """Return the longitudinal lane position exposed by this vehicle api."""
        ...
    def getRoute(self, vehicle_id: str) -> Sequence[str]:
        """Return the route exposed by this vehicle api."""
        ...
    def getRouteIndex(self, vehicle_id: str) -> int:
        """Return the route index exposed by this vehicle api."""
        ...
    def getNextTLS(self, vehicle_id: str) -> Sequence[Sequence[object]]:
        """Return the next tls exposed by this vehicle api."""
        ...
    def getLeader(
        self,
        vehicle_id: str,
        lookahead_m: float,
    ) -> tuple[str, float] | None:
        """Return the leader exposed by this vehicle api."""
        ...


class _LaneApi(Protocol):
    """Define the typed boundary for lane api."""

    def getLastStepVehicleIDs(self, lane_id: str) -> Sequence[str]:
        """Return the last step vehicle i ds exposed by this lane api."""
        ...


class _PersonApi(Protocol):
    """Define the typed boundary for person api."""

    def getIDCount(self) -> int:
        """Return the id count exposed by this person api."""
        ...
    def getIDList(self) -> Sequence[str]:
        """Return the id list exposed by this person api."""
        ...
    def getPosition(self, person_id: str) -> Sequence[float]:
        """Return the position exposed by this person api."""
        ...
    def getSpeed(self, person_id: str) -> float:
        """Return the speed exposed by this person api."""
        ...


class _GuiApi(Protocol):
    """Define the typed boundary for gui api."""

    def trackVehicle(self, view_id: str, vehicle_id: str) -> None:
        """Track the vehicle owned by this gui api."""
        ...
    def setZoom(self, view_id: str, zoom: float) -> None:
        """Set the zoom owned by this gui api."""
        ...


class _SimulationApi(Protocol):
    """Define the typed boundary for simulation api."""

    def getTime(self) -> float:
        """Return the time exposed by this simulation api."""
        ...
    def getDeltaT(self) -> float:
        """Return the delta t exposed by this simulation api."""
        ...


class _TrafficLightApi(Protocol):
    """Define the typed boundary for traffic light api."""

    def getRedYellowGreenState(self, traffic_light_id: str) -> str:
        """Return the red yellow green state exposed by this traffic light api."""
        ...
    def setRedYellowGreenState(self, traffic_light_id: str, state: str) -> None:
        """Set the red yellow green state owned by this traffic light api."""
        ...
    def getPhase(self, traffic_light_id: str) -> int:
        """Return the phase exposed by this traffic light api."""
        ...
    def setPhase(self, traffic_light_id: str, phase: int) -> None:
        """Set the phase owned by this traffic light api."""
        ...
    def getControlledLinks(
        self,
        traffic_light_id: str,
    ) -> Sequence[Sequence[Sequence[object]]]:
        """Return the controlled links exposed by this traffic light api."""
        ...
    def getProgram(self, traffic_light_id: str) -> str:
        """Return the program exposed by this traffic light api."""
        ...
    def getAllProgramLogics(self, traffic_light_id: str) -> Sequence[object]:
        """Return the all program logics exposed by this traffic light api."""
        ...
    def getSpentDuration(self, traffic_light_id: str) -> float:
        """Return the spent duration exposed by this traffic light api."""
        ...
    def getPhaseDuration(self, traffic_light_id: str) -> float:
        """Return the phase duration exposed by this traffic light api."""
        ...
    def getNextSwitch(self, traffic_light_id: str) -> float:
        """Return the next switch exposed by this traffic light api."""
        ...


class _SumoBinding(Protocol):
    """Define the typed boundary for sumo binding."""

    vehicle: _VehicleApi
    lane: _LaneApi
    person: _PersonApi
    gui: _GuiApi
    simulation: _SimulationApi
    trafficlight: _TrafficLightApi

    def start(self, command: list[str]) -> None:
        """Start the workflow owned by this sumo binding."""
        ...
    def close(self) -> None:
        """Close resources owned by this sumo binding."""
        ...
    def simulationStep(self) -> None:
        """Handle the simulation step operation owned by this sumo binding."""
        ...


class _VehicleEntityAdapter(VehicleEntityPort):
    """Session-checked vehicle view over the native binding."""

    def __init__(self, owner: SUMOSimulationController) -> None:
        """Initialize the vehicle entity adapter and its owned state."""
        self._owner = owner

    def setSpeedMode(self, vehicle_id: str, speed_mode: int) -> None:
        """Set the speed mode owned by this vehicle entity adapter."""
        self._owner._invoke(
            "set a vehicle speed mode",
            lambda: self._owner._binding.vehicle.setSpeedMode(vehicle_id, speed_mode),
        )

    def setSpeed(self, vehicle_id: str, speed_mps: float) -> None:
        """Set the speed owned by this vehicle entity adapter."""
        self._owner._invoke(
            "set a vehicle speed",
            lambda: self._owner._binding.vehicle.setSpeed(vehicle_id, float(speed_mps)),
        )

    def getIDCount(self) -> int:
        """Return the id count exposed by this vehicle entity adapter."""
        return int(
            self._owner._invoke(
                "read the vehicle count",
                self._owner._binding.vehicle.getIDCount,
            )
        )

    def getIDList(self) -> tuple[str, ...]:
        """Return the id list exposed by this vehicle entity adapter."""
        vehicle_ids = self._owner._invoke(
            "read vehicle identifiers",
            self._owner._binding.vehicle.getIDList,
        )
        return tuple(str(vehicle_id) for vehicle_id in vehicle_ids)

    def getPosition(self, vehicle_id: str) -> tuple[float, float]:
        """Return the position exposed by this vehicle entity adapter."""
        position = self._owner._invoke(
            "read a vehicle position",
            lambda: self._owner._binding.vehicle.getPosition(vehicle_id),
        )
        return float(position[0]), float(position[1])

    def getSpeed(self, vehicle_id: str) -> float:
        """Return the speed exposed by this vehicle entity adapter."""
        return float(
            self._owner._invoke(
                "read a vehicle speed",
                lambda: self._owner._binding.vehicle.getSpeed(vehicle_id),
            )
        )

    def getRoadID(self, vehicle_id: str) -> str:
        """Return the road id exposed by this vehicle entity adapter."""
        return str(
            self._owner._invoke(
                "read a vehicle road identifier",
                lambda: self._owner._binding.vehicle.getRoadID(vehicle_id),
            )
        )

    def getLaneID(self, vehicle_id: str) -> str:
        """Return the lane id exposed by this vehicle entity adapter."""
        return str(
            self._owner._invoke(
                "read a vehicle lane identifier",
                lambda: self._owner._binding.vehicle.getLaneID(vehicle_id),
            )
        )

    def getLanePosition(self, vehicle_id: str) -> float:
        """Return the longitudinal lane position exposed by this vehicle adapter."""
        return float(
            self._owner._invoke(
                "read a vehicle lane position",
                lambda: self._owner._binding.vehicle.getLanePosition(vehicle_id),
            )
        )

    def getRoute(self, vehicle_id: str) -> tuple[str, ...]:
        """Return the route exposed by this vehicle entity adapter."""
        route = self._owner._invoke(
            "read a vehicle route",
            lambda: self._owner._binding.vehicle.getRoute(vehicle_id),
        )
        return tuple(str(edge_id) for edge_id in route)

    def getRouteIndex(self, vehicle_id: str) -> int:
        """Return the route index exposed by this vehicle entity adapter."""
        return int(
            self._owner._invoke(
                "read a vehicle route index",
                lambda: self._owner._binding.vehicle.getRouteIndex(vehicle_id),
            )
        )

    def getNextTLS(self, vehicle_id: str) -> tuple[tuple[object, ...], ...]:
        """Return the next tls exposed by this vehicle entity adapter."""
        entries = self._owner._invoke(
            "read a vehicle's upcoming traffic lights",
            lambda: self._owner._binding.vehicle.getNextTLS(vehicle_id),
        )
        return tuple(tuple(entry) for entry in entries or ())

    def getLeader(
        self,
        vehicle_id: str,
        lookahead_m: float,
    ) -> tuple[str, float] | None:
        """Return the leader exposed by this vehicle entity adapter."""
        leader = self._owner._invoke(
            "read a vehicle leader",
            lambda: self._owner._binding.vehicle.getLeader(vehicle_id, lookahead_m),
        )
        if not leader:
            return None
        return str(leader[0]), float(leader[1])


class _LaneEntityAdapter(LaneEntityPort):
    """Session-checked lane view over the native binding."""

    def __init__(self, owner: SUMOSimulationController) -> None:
        """Initialize the lane entity adapter and its owned state."""
        self._owner = owner

    def getLastStepVehicleIDs(self, lane_id: str) -> tuple[str, ...]:
        """Return the last step vehicle i ds exposed by this lane entity adapter."""
        vehicle_ids = self._owner._invoke(
            "read lane vehicle membership",
            lambda: self._owner._binding.lane.getLastStepVehicleIDs(lane_id),
        )
        return tuple(str(vehicle_id) for vehicle_id in vehicle_ids)


class _PersonEntityAdapter(PersonEntityPort):
    """Session-checked pedestrian view over the native binding."""

    def __init__(self, owner: SUMOSimulationController) -> None:
        """Initialize the person entity adapter and its owned state."""
        self._owner = owner

    def getIDCount(self) -> int:
        """Return the id count exposed by this person entity adapter."""
        return int(
            self._owner._invoke(
                "read the pedestrian count",
                self._owner._binding.person.getIDCount,
            )
        )

    def getIDList(self) -> tuple[str, ...]:
        """Return the id list exposed by this person entity adapter."""
        person_ids = self._owner._invoke(
            "read pedestrian identifiers",
            self._owner._binding.person.getIDList,
        )
        return tuple(str(person_id) for person_id in person_ids)

    def getPosition(self, person_id: str) -> tuple[float, float]:
        """Return the position exposed by this person entity adapter."""
        position = self._owner._invoke(
            "read a pedestrian position",
            lambda: self._owner._binding.person.getPosition(person_id),
        )
        return float(position[0]), float(position[1])

    def getSpeed(self, person_id: str) -> float:
        """Return the speed exposed by this person entity adapter."""
        return float(
            self._owner._invoke(
                "read a pedestrian speed",
                lambda: self._owner._binding.person.getSpeed(person_id),
            )
        )


class _TrafficLightEntityAdapter(TrafficLightEntityPort):
    """Session-checked traffic-light view over the native binding."""

    def __init__(self, owner: SUMOSimulationController) -> None:
        """Initialize the traffic light entity adapter and its owned state."""
        self._owner = owner

    def getRedYellowGreenState(self, traffic_light_id: str) -> str:
        """Return the red yellow green state exposed by this traffic light entity adapter."""
        return str(
            self._owner._invoke(
                "read a traffic-light state",
                lambda: self._owner._binding.trafficlight.getRedYellowGreenState(traffic_light_id),
            )
        )

    def setRedYellowGreenState(self, traffic_light_id: str, state: str) -> None:
        """Set the red yellow green state owned by this traffic light entity adapter."""
        self._owner._invoke(
            "set a traffic-light state",
            lambda: self._owner._binding.trafficlight.setRedYellowGreenState(
                traffic_light_id,
                state,
            ),
        )

    def getPhase(self, traffic_light_id: str) -> int:
        """Return the phase exposed by this traffic light entity adapter."""
        return int(
            self._owner._invoke(
                "read a traffic-light phase",
                lambda: self._owner._binding.trafficlight.getPhase(traffic_light_id),
            )
        )

    def setPhase(self, traffic_light_id: str, phase: int) -> None:
        """Set the phase owned by this traffic light entity adapter."""
        self._owner._invoke(
            "set a traffic-light phase",
            lambda: self._owner._binding.trafficlight.setPhase(traffic_light_id, phase),
        )

    def getControlledLinks(
        self,
        traffic_light_id: str,
    ) -> tuple[tuple[tuple[object, ...], ...], ...]:
        """Return the controlled links exposed by this traffic light entity adapter."""
        groups = self._owner._invoke(
            "read traffic-light controlled links",
            lambda: self._owner._binding.trafficlight.getControlledLinks(traffic_light_id),
        )
        return tuple(tuple(tuple(link) for link in group or ()) for group in groups or ())

    def getProgram(self, traffic_light_id: str) -> str:
        """Return the program exposed by this traffic light entity adapter."""
        return str(
            self._owner._invoke(
                "read a traffic-light program",
                lambda: self._owner._binding.trafficlight.getProgram(traffic_light_id),
            )
        )

    def getAllProgramLogics(self, traffic_light_id: str) -> tuple[object, ...]:
        """Return the all program logics exposed by this traffic light entity adapter."""
        def read_logics() -> Sequence[object]:
            """Read the logics without changing this traffic light entity adapter's state."""
            traffic_light = self._owner._binding.trafficlight
            getter = getattr(traffic_light, "getAllProgramLogics", None)
            if getter is None:
                getter = getattr(
                    traffic_light,
                    "getCompleteRedYellowGreenDefinition",
                    None,
                )
            if getter is None:
                return ()
            typed_getter = cast(Callable[[str], Sequence[object]], getter)
            return typed_getter(traffic_light_id)

        logics = self._owner._invoke(
            "read traffic-light program definitions",
            read_logics,
        )
        return tuple(logics or ())

    def getSpentDuration(self, traffic_light_id: str) -> float:
        """Return the spent duration exposed by this traffic light entity adapter."""
        return float(
            self._owner._invoke(
                "read traffic-light spent duration",
                lambda: self._owner._binding.trafficlight.getSpentDuration(traffic_light_id),
            )
        )

    def getPhaseDuration(self, traffic_light_id: str) -> float:
        """Return the phase duration exposed by this traffic light entity adapter."""
        return float(
            self._owner._invoke(
                "read traffic-light phase duration",
                lambda: self._owner._binding.trafficlight.getPhaseDuration(traffic_light_id),
            )
        )

    def getNextSwitch(self, traffic_light_id: str) -> float:
        """Return the next switch exposed by this traffic light entity adapter."""
        return float(
            self._owner._invoke(
                "read traffic-light next switch time",
                lambda: self._owner._binding.trafficlight.getNextSwitch(traffic_light_id),
            )
        )


class _SimulationEntityAdapter(SimulationEntityPort):
    """Session-checked simulation-clock view over the native binding."""

    def __init__(self, owner: SUMOSimulationController) -> None:
        """Initialize the simulation entity adapter and its owned state."""
        self._owner = owner

    def getTime(self) -> float:
        """Return the time exposed by this simulation entity adapter."""
        return float(
            self._owner._invoke(
                "read simulation time",
                self._owner._binding.simulation.getTime,
            )
        )

    def getDeltaT(self) -> float:
        """Return the delta t exposed by this simulation entity adapter."""
        return float(
            self._owner._invoke(
                "read simulation step length",
                self._owner._binding.simulation.getDeltaT,
            )
        )


class SUMOSimulationController:
    """Own exactly one libsumo session and its optional embedded GUI window."""

    def __init__(
        self,
        *,
        step_length: float = 0.2,
        no_warnings: bool = True,
        defer_gui_shutdown_to_process_restart: bool = False,
        logger: logging.Logger | None = None,
        binding: _SumoBinding | None = None,
        native_gui_usage: NativeGuiUsageTracker = PROCESS_NATIVE_GUI_USAGE,
        configuration_materializer: Callable[
            [str], SumoConfigurationMaterialization
        ] = materialize_sumo_configuration,
    ) -> None:
        """Initialize the sumo simulation controller and its owned state."""
        self._logger = logger or logging.getLogger(__name__)
        self._binding: _SumoBinding = binding or cast(_SumoBinding, sumo)
        self._native_gui_usage = native_gui_usage
        self._step_length = step_length
        self._no_warnings = no_warnings
        self._defer_gui_shutdown_to_process_restart = bool(
            defer_gui_shutdown_to_process_restart
        )
        self._configuration_materializer = configuration_materializer
        self._active_configuration: SumoConfigurationMaterialization | None = None
        self._active_configuration_requires_commit = False
        self._sumo_initialized = False
        self._active_gui = False
        self._process_restart_required = False
        self.vehicle: VehicleEntityPort = _VehicleEntityAdapter(self)
        self.lane: LaneEntityPort = _LaneEntityAdapter(self)
        self.person: PersonEntityPort = _PersonEntityAdapter(self)
        self.trafficlight: TrafficLightEntityPort = _TrafficLightEntityAdapter(self)
        self.simulation: SimulationEntityPort = _SimulationEntityAdapter(self)

    def shutdown_simulation(self) -> bool:
        """Close a session once and retain ownership until finalization succeeds."""
        if not self._sumo_initialized:
            self._release_active_configuration()
            return False
        if self._active_gui and self._defer_gui_shutdown_to_process_restart:
            self._sumo_initialized = False
            self._active_configuration_requires_commit = True
            self._process_restart_required = True
            self._release_active_configuration()
            self._logger.info(
                "SUMO GUI native cleanup deferred to the pending process-image restart."
            )
            return True
        native_error: Exception | None = None
        try:
            self._binding.close()
            self._logger.info("SUMO closed successfully.")
        except Exception as exc:
            self._logger.exception("SUMO shutdown failed for an adapter-owned session.")
            native_error = exc
        if native_error is not None:
            raise SimulationLifecycleError(
                "SUMO shutdown failed; the adapter retained ownership for retry."
            ) from native_error
        self._sumo_initialized = False
        self._active_gui = False
        self._active_configuration_requires_commit = True
        try:
            self._release_active_configuration()
        except ManagedSumoConfigurationError as cleanup_error:
            raise cleanup_error
        return True

    def _release_active_configuration(self) -> None:
        """Release the active configuration owned by this sumo simulation controller."""
        materialization = self._active_configuration
        if materialization is not None:
            if self._active_configuration_requires_commit:
                materialization.commit_outputs()
                self._active_configuration_requires_commit = False
            materialization.close()
            self._active_configuration = None

    def _cleanup_failed_start(
        self,
        materialization: SumoConfigurationMaterialization,
    ) -> None:
        """Clean up the failed start owned by this sumo simulation controller."""
        try:
            materialization.close()
        except ManagedSumoConfigurationError:
            self._active_configuration = materialization
            self._active_configuration_requires_commit = False
            self._logger.exception(
                "Private SUMO configuration cleanup failed; ownership was retained for retry."
            )

    def start_simulation(self, gui: bool, sumo_config_path: str) -> None:
        """Start a native session and claim ownership only after success."""
        if self._process_restart_required:
            raise SimulationLifecycleError(
                "A deferred SUMO GUI session requires an application restart "
                "before another simulation can start."
            )
        if self._sumo_initialized:
            raise SimulationAlreadyStartedError("This SUMO adapter already owns an active session.")
        self._release_active_configuration()
        materialization = self._configuration_materializer(str(sumo_config_path))
        executable = "sumo-gui" if gui else "sumo"
        command = [
            executable,
            "-c",
            materialization.runtime_path,
            "--start",
            "--step-length",
            str(self._step_length),
            "--no-warnings",
            str(self._no_warnings),
        ]
        if gui:
            command.extend(("--quit-on-end", "false"))
        try:
            materialization.validate_for_start()
            self._binding.start(command)
        except ManagedSumoConfigurationError:
            self._cleanup_failed_start(materialization)
            raise
        except Exception as exc:
            self._cleanup_failed_start(materialization)
            self._logger.exception(
                "SUMO start failed.",
                extra={"gui": gui, "configuration": str(sumo_config_path)},
            )
            raise SimulationLifecycleError("SUMO start failed.") from exc
        self._active_configuration = materialization
        self._active_configuration_requires_commit = False
        self._sumo_initialized = True
        self._active_gui = bool(gui)
        if gui:
            self._native_gui_usage.mark_started()

    def sim_step(self) -> None:
        """Advance one mandatory step through the typed error boundary."""

        self._invoke("advance the simulation", self._binding.simulationStep)

    def set_vehicle_speed_mode(self, vehicle_id: str, speed_mode: int) -> None:
        """Set the vehicle speed mode owned by this sumo simulation controller."""
        self.vehicle.setSpeedMode(vehicle_id, speed_mode)

    def set_vehicle_speed(self, vehicle_id: str, speed_mps: float) -> None:
        """Set the vehicle speed owned by this sumo simulation controller."""
        self.vehicle.setSpeed(vehicle_id, speed_mps)

    def track_vehicle(self, view_id: str, vehicle_id: str) -> None:
        """Track the vehicle owned by this sumo simulation controller."""
        self._invoke(
            "track a vehicle",
            lambda: self._binding.gui.trackVehicle(view_id, vehicle_id),
        )

    def set_zoom(self, view_id: str, zoom: float) -> None:
        """Set the zoom owned by this sumo simulation controller."""
        self._invoke(
            "set a GUI zoom level",
            lambda: self._binding.gui.setZoom(view_id, zoom),
        )

    def get_vehicle_id_count(self) -> int:
        """Return the vehicle id count exposed by this sumo simulation controller."""
        return self.vehicle.getIDCount()

    def get_vehicle_ids(self) -> tuple[str, ...]:
        """Return the vehicle ids exposed by this sumo simulation controller."""
        return self.vehicle.getIDList()

    def has_vehicle(self, vehicle_id: str) -> bool:
        """Return whether vehicle applies to this sumo simulation controller."""
        return str(vehicle_id) in self.get_vehicle_ids()

    def get_pedestrian_id_count(self) -> int:
        """Return the pedestrian id count exposed by this sumo simulation controller."""
        return self.person.getIDCount()

    def is_initialized(self) -> bool:
        """Return whether initialized applies to this sumo simulation controller."""
        return self._sumo_initialized

    def requires_process_restart(self) -> bool:
        """Return whether process replacement owns the deferred GUI teardown."""

        return self._process_restart_required

    def get_vehicle_road_id(self, vehicle_id: str) -> str:
        """Return the vehicle road id exposed by this sumo simulation controller."""
        return self.vehicle.getRoadID(vehicle_id)

    def get_vehicle_speed(self, vehicle_id: str) -> float:
        """Return the vehicle speed exposed by this sumo simulation controller."""
        return self.vehicle.getSpeed(vehicle_id)

    def get_simulation_step_length(self) -> float:
        """Return the simulation step length exposed by this sumo simulation controller."""
        return self.simulation.getDeltaT()

    def get_simulation_time(self) -> float:
        """Return the simulation time exposed by this sumo simulation controller."""
        return self.simulation.getTime()

    def get_traffic_light_state(self, traffic_light_id: str) -> str:
        """Return the traffic light state exposed by this sumo simulation controller."""
        return self.trafficlight.getRedYellowGreenState(traffic_light_id)

    def set_traffic_light_state(self, traffic_light_id: str, state: str) -> None:
        """Set the traffic light state owned by this sumo simulation controller."""
        self.trafficlight.setRedYellowGreenState(traffic_light_id, state)

    def get_traffic_light_phase(self, traffic_light_id: str) -> int:
        """Return the traffic light phase exposed by this sumo simulation controller."""
        return self.trafficlight.getPhase(traffic_light_id)

    def set_traffic_light_phase(self, traffic_light_id: str, phase: int) -> None:
        """Set the traffic light phase owned by this sumo simulation controller."""
        self.trafficlight.setPhase(traffic_light_id, phase)

    def get_program(self, traffic_light_id: str) -> str:
        """Return the program exposed by this sumo simulation controller."""
        return self.trafficlight.getProgram(traffic_light_id)

    def get_program_logics(self, traffic_light_id: str) -> tuple[object, ...]:
        """Return the program logics exposed by this sumo simulation controller."""
        return self.trafficlight.getAllProgramLogics(traffic_light_id)

    def get_phase(self, traffic_light_id: str) -> int:
        """Return the phase exposed by this sumo simulation controller."""
        return self.trafficlight.getPhase(traffic_light_id)

    def get_spent_duration(self, traffic_light_id: str) -> float:
        """Return the spent duration exposed by this sumo simulation controller."""
        return self.trafficlight.getSpentDuration(traffic_light_id)

    def get_phase_duration(self, traffic_light_id: str) -> float:
        """Return the phase duration exposed by this sumo simulation controller."""
        return self.trafficlight.getPhaseDuration(traffic_light_id)

    def get_next_switch(self, traffic_light_id: str) -> float:
        """Return the next switch exposed by this sumo simulation controller."""
        return self.trafficlight.getNextSwitch(traffic_light_id)

    def optional_read(
        self,
        operation: str,
        reader: Callable[[], OperationValue],
        fallback: OperationValue,
    ) -> SimulationRead[OperationValue]:
        """Return a declared fallback while preserving failure provenance."""

        try:
            return SimulationRead.live(reader())
        except SimulationOperationError as exc:
            self._logger.warning(
                "Optional SUMO read used a fallback.",
                extra={"operation": operation, "quality": "fallback"},
            )
            return SimulationRead.fallback(fallback, exc)
        except Exception as exc:
            wrapped = SimulationOperationError(f"Optional SUMO read failed: {operation}.")
            wrapped.__cause__ = exc
            self._logger.exception(
                "Optional SUMO read used a fallback.",
                extra={"operation": operation, "quality": "fallback"},
            )
            return SimulationRead.fallback(fallback, wrapped)

    def _invoke(
        self,
        operation: str,
        callback: Callable[[], OperationValue],
    ) -> OperationValue:
        """Execute a mandatory native operation through one typed boundary."""

        self._require_initialized(operation)
        try:
            return callback()
        except SimulationOperationError:
            raise
        except Exception as exc:
            self._logger.exception(
                "Mandatory SUMO operation failed.",
                extra={"operation": operation},
            )
            raise SimulationOperationError(f"SUMO operation failed: {operation}.") from exc

    def _require_initialized(self, operation: str) -> None:
        """Return the required initialized for this sumo simulation controller."""
        if not self._sumo_initialized:
            raise SimulationNotStartedError(
                f"Cannot {operation}: this adapter has no active SUMO session."
            )
