# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model-facing adapter for the Zala scenario-context service."""

from typing import cast

from lanista.domains.automotive.control.scenario_context_contracts import (
    ScenarioContextSnapshot,
)
from lanista.domains.automotive.zala_zone.scenario_context_service import (
    ScenarioContextConfiguration,
    ScenarioContextLogger,
    ScenarioContextService,
)
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.utils.simulation.ports import (
    EntitySimulationPort,
    SimulationReadQuality,
)


class ScenarioContextPreprocessorAdapter(LoggerInterface):
    """Expose the stable model-preprocessor property surface over the Zala service."""

    def __init__(
        self,
        *,
        simulator_port: EntitySimulationPort,
        configuration: ScenarioContextConfiguration,
        **kwargs: object,
    ) -> None:
        """Initialize the adapter with one owned scenario-context service."""
        LoggerInterface.__init__(self, **kwargs)
        self._context_service = ScenarioContextService(
            simulator=simulator_port,
            logger=cast(ScenarioContextLogger, self._logger),
            configuration=configuration,
        )

    def update(self, ego_speed_mps: float) -> ScenarioContextSnapshot:
        """Refresh and return the immutable scenario-context projection."""
        return self._context_service.update(ego_speed_mps)

    @property
    def snapshot(self) -> ScenarioContextSnapshot:
        """Return the latest immutable scenario-context projection."""
        return self._context_service.snapshot

    @property
    def reference_speed_mps(self) -> float:
        """Return the current reference speed in metres per second."""
        return self.snapshot.reference_speed_mps

    @property
    def object_stations_m(self) -> list[float]:
        """Return a mutable copy of the observed object stations in metres."""
        return list(self.snapshot.object_stations_m)

    @property
    def object_speeds_mps(self) -> list[float]:
        """Return a mutable copy of the observed object speeds in metres per second."""
        return list(self.snapshot.object_speeds_mps)

    @property
    def ego_station_m(self) -> float:
        """Return the current ego station in metres."""
        return self.snapshot.ego_station_m

    @property
    def pedestrian_parameters(self) -> tuple[float, float, int]:
        """Return pedestrian station, speed, and encoded state."""
        return self.snapshot.pedestrian_parameters

    @property
    def intersection_number(self) -> int:
        """Return the current Zala route intersection number."""
        return self.snapshot.intersection_number

    @property
    def traffic_light_state_code(self) -> int | None:
        """Return the configured traffic-light state code when available."""
        return self.snapshot.traffic_light_state_code

    @property
    def upcoming_intersection_type_code(self) -> int | None:
        """Return the classified upcoming-intersection code when available."""
        return self.snapshot.upcoming_intersection_type_code

    @property
    def ego_position_xy_m(self) -> tuple[float, float]:
        """Return the current ego position in metres."""
        return self.snapshot.ego_position_xy_m

    @property
    def ego_road_id(self) -> str | int:
        """Return the current ego road identifier."""
        return self.snapshot.ego_road_id

    @property
    def simulator_read_quality(self) -> SimulationReadQuality:
        """Return the aggregate simulator-read quality for the step."""
        return self.snapshot.simulator_read_quality

    @property
    def _blinker_left(self) -> int:
        """Return the legacy left-blinker projection."""
        return self.snapshot.blinkers.left

    @property
    def _blinker_right(self) -> int:
        """Return the legacy right-blinker projection."""
        return self.snapshot.blinkers.right

    @property
    def _blinker_out(self) -> int:
        """Return the legacy encoded-blinker projection."""
        return self.snapshot.blinkers.encoded
