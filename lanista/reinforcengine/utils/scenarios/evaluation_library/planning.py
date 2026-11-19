# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Build deterministic Cartesian generation plans for held-out libraries."""

from __future__ import annotations

import hashlib
import unicodedata
from dataclasses import dataclass
from itertools import product

from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    EvaluationGroupDefinition,
)

SEED_DERIVATION_VERSION = "sha256-v1"
GROUPED_SEED_DERIVATION_VERSION = "sha256-group-v1"
_MAX_RANDOM_TRIPS_SEED = 2**31 - 1


@dataclass(frozen=True, slots=True)
class PlannedEvaluationScenario:
    """Describe one scenario before filesystem materialisation."""

    scenario_id: str
    requested_vehicle_count: int
    requested_pedestrian_count: int
    replicate_index: int
    seed: int


def _validated_levels(name: str, levels: tuple[int, ...]) -> tuple[int, ...]:
    """Require a nonempty strictly increasing sequence of nonnegative counts."""

    if not levels or any(type(value) is not int or value < 0 for value in levels):
        raise ValueError(f"{name} must contain nonnegative integers.")
    if len(levels) != len(set(levels)) or levels != tuple(sorted(levels)):
        raise ValueError(f"{name} must be unique and sorted.")
    return levels


def derive_scenario_seed(
    *,
    master_seed: int,
    requested_vehicle_count: int,
    requested_pedestrian_count: int,
    replicate_index: int,
) -> int:
    """Derive one stable nonzero randomTrips seed from experiment factors."""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer.")
    factors = (
        f"{SEED_DERIVATION_VERSION}\0{master_seed}\0{requested_vehicle_count}"
        f"\0{requested_pedestrian_count}\0{replicate_index}"
    ).encode("ascii")
    derived = int.from_bytes(hashlib.sha256(factors).digest()[:8], "big")
    return derived % _MAX_RANDOM_TRIPS_SEED + 1


def build_generation_plan(
    *,
    vehicle_levels: tuple[int, ...],
    pedestrian_levels: tuple[int, ...],
    replicate_count: int,
    master_seed: int,
) -> tuple[PlannedEvaluationScenario, ...]:
    """Enumerate the complete factor grid in canonical manifest order."""

    vehicles = _validated_levels("vehicle_levels", vehicle_levels)
    pedestrians = _validated_levels("pedestrian_levels", pedestrian_levels)
    if type(replicate_count) is not int or replicate_count < 1:
        raise ValueError("replicate_count must be a positive integer.")
    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer.")

    entries = tuple(
        PlannedEvaluationScenario(
            scenario_id=f"veh-{vehicle_count}_ped-{pedestrian_count}_rep-{replicate_index}",
            requested_vehicle_count=vehicle_count,
            requested_pedestrian_count=pedestrian_count,
            replicate_index=replicate_index,
            seed=derive_scenario_seed(
                master_seed=master_seed,
                requested_vehicle_count=vehicle_count,
                requested_pedestrian_count=pedestrian_count,
                replicate_index=replicate_index,
            ),
        )
        for vehicle_count, pedestrian_count, replicate_index in product(
            vehicles,
            pedestrians,
            range(replicate_count),
        )
    )
    seeds = tuple(entry.seed for entry in entries)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Derived scenario seed collision; choose a different master seed.")
    return entries


@dataclass(frozen=True, slots=True)
class PlannedGroupedScenario(PlannedEvaluationScenario):
    """Bind one deterministic scenario to a stable group and local index."""

    group_id: str
    group_scenario_index: int


def derive_grouped_scenario_seed(
    *, master_seed: int, group_id: str, group_scenario_index: int
) -> int:
    """Derive a positive SUMO seed without depending on the editable group name."""

    if type(master_seed) is not int or master_seed < 0:
        raise ValueError("master_seed must be a nonnegative integer.")
    if type(group_scenario_index) is not int or group_scenario_index < 0:
        raise ValueError("group_scenario_index must be a nonnegative integer.")
    EvaluationGroupDefinition._validate_group_id(group_id)
    factors = (
        f"{GROUPED_SEED_DERIVATION_VERSION}\0{master_seed}\0"
        f"{group_id}\0{group_scenario_index}"
    ).encode("ascii")
    derived = int.from_bytes(hashlib.sha256(factors).digest()[:8], "big")
    return derived % _MAX_RANDOM_TRIPS_SEED + 1


def build_grouped_generation_plan(
    *, groups: tuple[EvaluationGroupDefinition, ...], master_seed: int
) -> tuple[PlannedGroupedScenario, ...]:
    """Enumerate ordered groups and their complete local scenario sequences."""

    if not groups or any(not isinstance(group, EvaluationGroupDefinition) for group in groups):
        raise ValueError("groups must contain validated group definitions.")
    ids = tuple(group.group_id for group in groups)
    if ids != tuple(sorted(set(ids), key=lambda value: int(value[6:]))):
        raise ValueError("Group IDs must be unique and ordered by creation index.")
    names = tuple(unicodedata.normalize("NFKC", group.name).casefold() for group in groups)
    if len(names) != len(set(names)):
        raise ValueError("Group names must be unique after case-insensitive normalization.")
    plan = tuple(
        PlannedGroupedScenario(
            scenario_id=f"{group.group_id}-scenario-{index + 1:04d}",
            requested_vehicle_count=group.vehicle_count,
            requested_pedestrian_count=group.pedestrian_count,
            replicate_index=index,
            seed=derive_grouped_scenario_seed(
                master_seed=master_seed,
                group_id=group.group_id,
                group_scenario_index=index,
            ),
            group_id=group.group_id,
            group_scenario_index=index,
        )
        for group in groups
        for index in range(group.scenario_count)
    )
    seeds = tuple(item.seed for item in plan)
    if len(seeds) != len(set(seeds)):
        raise ValueError("Derived grouped scenario seed collision; choose a different master seed.")
    return plan


__all__ = [
    "SEED_DERIVATION_VERSION",
    "GROUPED_SEED_DERIVATION_VERSION",
    "PlannedGroupedScenario",
    "build_grouped_generation_plan",
    "derive_grouped_scenario_seed",
    "PlannedEvaluationScenario",
    "build_generation_plan",
    "derive_scenario_seed",
]
