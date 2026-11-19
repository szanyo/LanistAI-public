# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project heterogeneous physical actors into Susan's generic object slots."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import StrEnum


class PolicyObstacleKind(StrEnum):
    """Actor kinds that share the policy's existing object representation."""

    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"


class PolicyObstacleGeometry(StrEnum):
    """Coordinate meaning carried beside a fixed-width policy object slot."""

    CONFLICT_POINT = "conflict_point"
    LONGITUDINAL_GAP = "longitudinal_gap"


@dataclass(frozen=True, slots=True)
class PolicyObstacleCandidate:
    """One actor record kept intact while assigning nearest-first policy slots."""

    actor_id: str
    kind: PolicyObstacleKind
    station_m: float
    speed_mps: float | None
    distance_to_ego_m: float | None = None
    geometry: PolicyObstacleGeometry = PolicyObstacleGeometry.CONFLICT_POINT
    constraint_clearance_m: float | None = None


@dataclass(frozen=True, slots=True)
class PolicyObstacleProjection:
    """Fixed-width policy projection without changing the observation shape."""

    candidates: tuple[PolicyObstacleCandidate, ...]
    stations_m: tuple[float, ...]
    speeds_mps: tuple[float, ...]
    constraint_clearances_m: tuple[float, ...]


def select_nearest_policy_obstacles(
    candidates: tuple[PolicyObstacleCandidate, ...],
    *,
    ego_station_m: float,
    slot_count: int = 3,
) -> tuple[PolicyObstacleCandidate, ...]:
    """Return whole actor records in deterministic physical-distance order."""

    if slot_count < 1:
        raise ValueError("policy obstacle slot count must be positive")
    ego_station = float(ego_station_m)
    if not math.isfinite(ego_station):
        return ()

    valid = tuple(
        candidate
        for candidate in candidates
        if candidate.actor_id
        and math.isfinite(float(candidate.station_m))
    )
    ranked = sorted(
        valid,
        key=lambda candidate: (
            _candidate_distance_m(candidate, ego_station),
            candidate.actor_id,
            candidate.kind.value,
        ),
    )[:slot_count]
    return tuple(
        replace(
            candidate,
            distance_to_ego_m=_candidate_distance_m(candidate, ego_station),
        )
        for candidate in ranked
    )


def build_policy_obstacle_projection(
    candidates: tuple[PolicyObstacleCandidate, ...],
    *,
    slot_count: int = 3,
    empty_station_m: float = -200.0,
) -> PolicyObstacleProjection:
    """Pad a resolved nearest-first actor list to the existing slot width."""

    if slot_count < 1:
        raise ValueError("policy obstacle slot count must be positive")
    selected = tuple(candidates[:slot_count])
    missing = slot_count - len(selected)
    stations = tuple(float(candidate.station_m) for candidate in selected) + (
        (float(empty_station_m),) * missing
    )
    speeds = tuple(
        _finite_non_negative_speed(candidate.speed_mps) for candidate in selected
    ) + ((0.0,) * missing)
    constraint_clearances = tuple(
        _finite_non_negative_clearance(candidate.constraint_clearance_m)
        for candidate in selected
    ) + ((math.nan,) * missing)
    return PolicyObstacleProjection(
        candidates=selected,
        stations_m=stations,
        speeds_mps=speeds,
        constraint_clearances_m=constraint_clearances,
    )


def _finite_non_negative_speed(value: float | None) -> float:
    """Return a safe physical speed for one already-selected actor."""

    if value is None:
        return 0.0
    speed = float(value)
    return max(0.0, speed) if math.isfinite(speed) else 0.0


def _finite_non_negative_clearance(value: float | None) -> float:
    """Return an aligned clearance or NaN when this actor has no QP projection."""

    if value is None:
        return math.nan
    clearance = float(value)
    return max(0.0, clearance) if math.isfinite(clearance) else math.nan


def _candidate_distance_m(
    candidate: PolicyObstacleCandidate,
    ego_station_m: float,
) -> float:
    """Return exact actor distance when available, otherwise conflict distance."""

    if candidate.distance_to_ego_m is not None:
        distance = float(candidate.distance_to_ego_m)
        if math.isfinite(distance) and distance >= 0.0:
            return distance
    return math.hypot(ego_station_m, float(candidate.station_m))
