# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Quality-bearing optional simulator telemetry helpers."""

from __future__ import annotations

from typing import Protocol

from lanista.reinforcengine.utils.simulation.ports import (
    OptionalSimulationReadPort,
    SimulationRead,
)


class SimulationTimeTelemetryPort(OptionalSimulationReadPort, Protocol):
    """Optional-read adapter with access to the mandatory simulation clock."""

    def get_simulation_time(self) -> float:
        """Return current simulation time in seconds."""


def read_episode_time_milliseconds(
    simulator: SimulationTimeTelemetryPort,
) -> SimulationRead[float]:
    """Read optional episode timing without discarding fallback provenance."""

    seconds = simulator.optional_read(
        "read SB3 episode timing telemetry",
        simulator.get_simulation_time,
        0.0,
    )
    return SimulationRead(
        value=float(seconds.value) * 1000.0,
        quality=seconds.quality,
        error=seconds.error,
    )
