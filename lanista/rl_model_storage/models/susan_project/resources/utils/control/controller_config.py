# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Static controller and traffic-light route configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from lanista.domains.automotive.control.longitudinal_control_optimizer import (
    nominal_vehicle_avoidance_approach_station_m,
)

LEGACY_VEHICLE_CONFLICT_WINDOW_START_M = -25.0
LEGACY_VEHICLE_CONFLICT_RELEASE_M = -6.0
INACTIVE_SUPERVISOR_EGO_STATION_M = 100.0


@dataclass(frozen=True, slots=True)
class LongitudinalLeaderQpProjection:
    """One direct leader expressed in the original QP coordinate system."""

    ego_station_m: float
    object_station_m: float
    constraint_clearance_m: float


@dataclass(frozen=True, slots=True)
class LongitudinalLeaderConstraint:
    """Calibrate SUMO bumper gaps to Susan's original leader constraint."""

    object_station_m: float = -15.0
    ego_distance_scale: float = 1.1
    bumper_gap_to_center_distance_m: float = 5.0

    def project(self, bumper_gap_m: float) -> LongitudinalLeaderQpProjection:
        """Return the historical QP projection and its fixed safe clearance."""

        gap_m = float(bumper_gap_m)
        if not math.isfinite(gap_m) or gap_m < 0.0:
            raise ValueError("longitudinal leader gap must be finite and non-negative")
        center_distance_m = gap_m + self.bumper_gap_to_center_distance_m
        ego_station_m = -(self.ego_distance_scale * center_distance_m)
        boundary_station_m = nominal_vehicle_avoidance_approach_station_m(
            self.object_station_m,
        )
        return LongitudinalLeaderQpProjection(
            ego_station_m=ego_station_m,
            object_station_m=self.object_station_m,
            constraint_clearance_m=max(0.0, boundary_station_m - ego_station_m),
        )


@dataclass(frozen=True)
class TrafficLightRouteContext:
    """Carry the runtime context for traffic light route."""

    tls_id: str
    green_phase: int
    link_index: int
    route_internal_edge: str
    signal_boundary_raw_station_m: float
    qp_restricted_object_station_m: float

    def qp_nominal_approach_raw_station_m(
        self,
        ego_speed_mps: float = 0.0,
    ) -> float:
        """Return the calibrated QP approach station in Bence coordinates."""

        return nominal_vehicle_avoidance_approach_station_m(
            self.qp_restricted_object_station_m,
            ego_speed_mps,
        )

    @property
    def qp_stationary_stop_raw_station_m(self) -> float:
        """Return the QP-calibrated real stop optimum at zero speed."""

        return self.qp_nominal_approach_raw_station_m(0.0)

    def as_dict(self) -> dict[str, object]:
        """Return this traffic light route context's value as dict."""
        return {
            "tls_id": self.tls_id,
            "green_phase": self.green_phase,
            "link_index": self.link_index,
            "route_internal_edge": self.route_internal_edge,
            "signal_boundary_raw_station_m": self.signal_boundary_raw_station_m,
            "qp_restricted_object_station_m": self.qp_restricted_object_station_m,
        }


@dataclass(frozen=True)
class ControllerConfig:
    """Represent configuration for controller."""

    ego_lead_vehicle_lookahead_meters: float = 60.0
    longitudinal_leader_constraint: LongitudinalLeaderConstraint = field(
        default_factory=LongitudinalLeaderConstraint
    )
    ego_station_observation_offsets: dict[int, float] = field(
        default_factory=lambda: {
            # Legacy route-section handover metadata retained for configuration
            # compatibility. Susan's non-signal policy station now preserves
            # the raw conflict origin shared with the supervisor object axes;
            # signal stops use traffic_light_route_contexts below.
            1: 4.46,  # :18_9_0 endpoint y=400.46; raw station = y - 396.
            2: 0.93,  # :23_0_0 endpoint x=77.07; raw station = 78 - x.
            3: 6.85,  # :43_4_0 endpoint y=349.15; raw station = 356 - y.
            4: 3.38,  # :27_2_0 endpoint x=119.38; no-ped station = x - 116.
            5: -9.75,  # :19_7_1 endpoint x=158.25; raw station = x - 168.
            6: 9.42,  # :31_4_0 endpoint y=300.58; raw station = 310 - y.
            7: 5.77,  # :35_9_0 endpoint y=200.23; raw station = 206 - y.
            8: -1.78,  # :2_5_0 endpoint x=176.78; raw station = 175 - x.
        }
    )
    traffic_light_route_contexts: dict[int, TrafficLightRouteContext] = field(
        default_factory=lambda: {
            1: TrafficLightRouteContext(
                tls_id="18",
                green_phase=2,
                link_index=9,
                route_internal_edge=":18_9",
                # Raw Bence station at the configured SUMO internal-edge entry.
                signal_boundary_raw_station_m=-7.91,
                # Original non-green virtual object consumed by the QP.
                qp_restricted_object_station_m=-5.0,
            ),
            6: TrafficLightRouteContext(
                tls_id="31",
                green_phase=11,
                link_index=4,
                route_internal_edge=":31_4",
                # Raw Bence station at the configured SUMO internal-edge entry.
                signal_boundary_raw_station_m=0.56,
                # Original non-green virtual object consumed by the QP.
                qp_restricted_object_station_m=-1.0,
            ),
        }
    )

    def get_ego_station_observation_offset(self, intersection_number: int, default: float = 0.0) -> float:
        """Return the ego station observation offset exposed by this controller config."""
        try:
            return float(self.ego_station_observation_offsets.get(int(intersection_number), default))
        except (TypeError, ValueError):
            return float(default)

    def get_ego_lead_vehicle_lookahead_meters(self, default: float = 60.0) -> float:
        """Return the ego lead vehicle lookahead meters exposed by this controller config."""
        try:
            return float(self.ego_lead_vehicle_lookahead_meters)
        except (TypeError, ValueError):
            return float(default)

    def project_longitudinal_leader_constraint(
        self,
        bumper_gap_m: float,
    ) -> LongitudinalLeaderQpProjection:
        """Project one already-read SUMO leader gap into the legacy QP space."""

        return self.longitudinal_leader_constraint.project(bumper_gap_m)

    def get_traffic_light_route_context(
        self,
        intersection_number: int,
    ) -> TrafficLightRouteContext | None:
        """Return the traffic light route context exposed by this controller config."""
        try:
            return self.traffic_light_route_contexts.get(int(intersection_number))
        except (TypeError, ValueError):
            return None

    def get_traffic_light_route_context_dict(
        self,
        intersection_number: int,
    ) -> dict[str, object] | None:
        """Return the traffic light route context dict exposed by this controller config."""
        context = self.get_traffic_light_route_context(intersection_number)
        return context.as_dict() if context is not None else None


CONTROLLER_CONFIG = ControllerConfig()


def get_ego_station_observation_offset(intersection_number: int, default: float = 0.0) -> float:
    """Return the ego station observation offset exposed by this controller config module."""
    return CONTROLLER_CONFIG.get_ego_station_observation_offset(intersection_number, default)


def get_ego_lead_vehicle_lookahead_meters(default: float = 60.0) -> float:
    """Return the ego lead vehicle lookahead meters exposed by this controller config module."""
    return CONTROLLER_CONFIG.get_ego_lead_vehicle_lookahead_meters(default)


def get_traffic_light_route_context(intersection_number: int) -> TrafficLightRouteContext | None:
    """Return the traffic light route context exposed by this controller config module."""
    return CONTROLLER_CONFIG.get_traffic_light_route_context(intersection_number)


def get_traffic_light_route_context_dict(intersection_number: int) -> dict[str, object] | None:
    """Return the traffic light route context dict exposed by this controller config module."""
    return CONTROLLER_CONFIG.get_traffic_light_route_context_dict(intersection_number)
