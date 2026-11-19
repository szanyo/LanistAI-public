# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Mandatory SUMO vehicle-command adapter for the Susan control pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field

from lanista.reinforcengine.utils.simulation.ports import (
    SimulationOperationError,
    SimulationReadQuality,
    SimulatorPort,
)

from ..observation.vehicle_telemetry import VehicleControllerTelemetryState


@dataclass
class SumoCommandService:
    """SUMO command/read boundary for the Susan controller pipeline."""

    port: SimulatorPort
    controller_state: VehicleControllerTelemetryState | None = None
    vehicle_id: str = "EGO"
    default_step_length_seconds: float = 0.2
    last_read_quality: SimulationReadQuality = field(
        default=SimulationReadQuality.LIVE,
        init=False,
    )
    last_read_error: SimulationOperationError | None = field(default=None, init=False)

    def read_ego_speed(self) -> float | None:
        """Read the ego speed without changing this sumo command service's state."""
        read = self.port.optional_read(
            "read Susan EGO speed telemetry",
            lambda: self.port.get_vehicle_speed(self.vehicle_id),
            None,
        )
        self.last_read_quality = read.quality
        self.last_read_error = read.error
        return None if read.value is None else float(read.value)

    def set_ego_speed(self, speed_mps: float) -> None:
        """Set the ego speed owned by this sumo command service."""
        self.port.set_vehicle_speed(self.vehicle_id, float(speed_mps))

    def step_length_seconds(self) -> float:
        """Return the step length seconds value owned by this sumo command service."""
        read = self.port.optional_read(
            "read Susan simulation step length telemetry",
            self.port.get_simulation_step_length,
            self.default_step_length_seconds,
        )
        self.last_read_quality = read.quality
        self.last_read_error = read.error
        step_length = float(read.value)
        if step_length <= 0:
            self.last_read_quality = SimulationReadQuality.FALLBACK
            return self.default_step_length_seconds
        if step_length > 10:
            return step_length / 1000.0
        return step_length

    def acceleration_since_last_step(
        self,
        current_speed_mps: float | None,
        previous_speed_mps: float | None,
    ) -> float | None:
        """Return the acceleration since last step value owned by this sumo command service."""
        if current_speed_mps is None or previous_speed_mps is None:
            return None
        step_length = self.step_length_seconds()
        return (float(current_speed_mps) - float(previous_speed_mps)) / step_length

    def read_acceleration_since_last_controller_step(
        self,
    ) -> tuple[float | None, float | None]:
        """Read the acceleration since last controller step without changing this sumo command
        service's state.
        """
        current_sumo_speed = self.read_ego_speed()
        last_sumo_speed = (
            self.controller_state.last_sumo_speed
            if self.controller_state is not None
            else None
        )
        return current_sumo_speed, self.acceleration_since_last_step(
            current_sumo_speed,
            last_sumo_speed,
        )

    def remember_step_speed(self, current_speed_mps: float | None) -> None:
        """Remember the step speed owned by this sumo command service."""
        if self.controller_state is not None:
            self.controller_state.last_sumo_speed = current_speed_mps
