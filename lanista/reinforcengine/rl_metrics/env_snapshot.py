# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed construction stages for one environment metrics snapshot."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import cast

MetricSnapshot = dict[str, object]
ObservationChannels = Mapping[str, Mapping[str, float]]


def read_live_vehicle_count(env: object) -> float:
    """Return the current SUMO vehicle population, including the ego vehicle."""

    simulator = getattr(env, "sumo_simulation_controller", None)
    getter = getattr(simulator, "get_vehicle_id_count", None)
    if not callable(getter):
        return 0.0
    try:
        return float(max(0, int(getter())))
    except (TypeError, ValueError, RuntimeError):
        return 0.0


@dataclass(frozen=True)
class EnvSnapshotDependencies:
    """Schema-owned adapters used by the engine-neutral snapshot workflow."""

    reward_snapshot: Callable[[object, object], MetricSnapshot]
    observation_snapshot: Callable[[object], MetricSnapshot]
    observation_channels: Callable[[object], ObservationChannels]
    reward_channels: Callable[[object, object], dict[str, float]]
    is_valid_number: Callable[[object], bool]
    diagnostic_observation_names: frozenset[str]


@dataclass(frozen=True)
class _EpisodeSnapshotContext:
    """Values shared by the episode, vehicle, and terminal metric groups."""

    terminated: bool
    truncated: bool
    termination_reason: object
    truncation_reason: object
    max_speed: float
    average_speed: float
    reached_intersection: float
    curriculum_target_detected: bool
    curriculum_target_accepted: bool
    curriculum_target_rejected_by_red_light_overdrive: bool


def _build_episode_context(
    env: object,
    *,
    terminated: bool,
    truncated: bool,
) -> _EpisodeSnapshotContext:
    """Build the episode context required by this env snapshot module."""
    termination_reason = getattr(env, "last_termination_reason", None)
    truncation_reason = getattr(env, "last_truncation_reason", None)
    ego_vehicle = getattr(env, "ego_vehicle", None)
    preprocessor = getattr(env, "scenario_context_preprocessor", None)
    max_speed = float(getattr(ego_vehicle, "MaxSpeed", 0.0))
    average_speed = float(getattr(ego_vehicle, "AverageSpeed", 0.0))
    reached_intersection = float(
        getattr(preprocessor, "intersection_number", 0.0)
    )
    curriculum_target_detected = bool(
        getattr(env, "curriculum_target_detected", False)
    )
    curriculum_target_accepted = bool(
        getattr(
            env,
            "curriculum_target_accepted",
            terminated and termination_reason == "curriculum_target",
        )
    )
    rejected_by_red_light_overdrive = bool(
        getattr(
            env,
            "curriculum_target_rejected_by_red_light_overdrive",
            curriculum_target_detected
            and bool(
                getattr(
                    env,
                    "episode_end_conflict_target_red_light_overdrive",
                    False,
                )
            ),
        )
    )
    return _EpisodeSnapshotContext(
        terminated=terminated,
        truncated=truncated,
        termination_reason=termination_reason,
        truncation_reason=truncation_reason,
        max_speed=max_speed,
        average_speed=average_speed,
        reached_intersection=reached_intersection,
        curriculum_target_detected=curriculum_target_detected,
        curriculum_target_accepted=curriculum_target_accepted,
        curriculum_target_rejected_by_red_light_overdrive=(
            rejected_by_red_light_overdrive
        ),
    )


def _count_mean(value: bool) -> dict[str, int | float]:
    """Return the count mean value owned by this env snapshot module."""
    return {"sum": int(value), "mean": float(value)}


def _episode_info(
    env: object,
    context: _EpisodeSnapshotContext,
) -> dict[str, object]:
    """Return the episode info value owned by this env snapshot module."""
    tracker = getattr(env, "episode_interactions", None)
    interactions = tracker.summary() if tracker is not None else {}
    return {
        "episode_terminated": _count_mean(context.terminated),
        "episode_truncated": _count_mean(context.truncated),
        "episode_end_conflict": _count_mean(
            bool(getattr(env, "episode_end_conflict", False))
        ),
        "episode_end_conflict_target_time_limit": _count_mean(
            bool(
                getattr(
                    env,
                    "episode_end_conflict_target_time_limit",
                    False,
                )
            )
        ),
        "episode_end_conflict_target_red_light_overdrive": _count_mean(
            bool(
                getattr(
                    env,
                    "episode_end_conflict_target_red_light_overdrive",
                    False,
                )
            )
        ),
        "curriculum_target_detected": _count_mean(
            context.curriculum_target_detected
        ),
        "curriculum_target_accepted": _count_mean(
            context.curriculum_target_accepted
        ),
        "curriculum_target_reached": _count_mean(
            context.curriculum_target_accepted
        ),
        "curriculum_target_rejected_by_red_light_overdrive": _count_mean(
            context.curriculum_target_rejected_by_red_light_overdrive
        ),
        "termination_curriculum_target": _count_mean(
            context.terminated
            and context.termination_reason == "curriculum_target"
        ),
        "termination_route_end": _count_mean(
            context.terminated and context.termination_reason == "route_end"
        ),
        "truncation_time_limit": _count_mean(
            context.truncated and context.truncation_reason == "time_limit"
        ),
        "truncation_red_light_overdrive": _count_mean(
            context.truncated
            and context.truncation_reason == "red_light_overdrive"
        ),
        **{key: {"mean": value} for key, value in interactions.items()},
    }


def _vehicle_info(
    env: object,
    context: _EpisodeSnapshotContext,
) -> dict[str, object]:
    """Return the vehicle info value owned by this env snapshot module."""
    return {
        "max_speed_per_episode": {"mean": context.max_speed},
        "average_speed_per_episode": {"mean": context.average_speed},
        "reached_intersection_number": {
            "mean": context.reached_intersection
        },
        "number_of_vehicles": {
            "mean": read_live_vehicle_count(env)
        },
        "number_of_pedestrians": {
            "mean": float(getattr(env, "pedestrian_id_counter", 0.0))
        },
    }


def _episode_setup(env: object) -> dict[str, object]:
    """Return the episode setup value owned by this env snapshot module."""
    return {
        "warmup_seconds": {
            "value": float(getattr(env, "training_warmup_seconds", 0.0))
        },
        "warmup_steps": {
            "value": float(getattr(env, "training_warmup_steps", 0.0))
        },
        "warmup_max_seconds": {
            "value": float(
                getattr(env, "training_warmup_max_seconds", 0.0)
            )
        },
        "warmup_max_steps": {
            "value": float(getattr(env, "training_warmup_max_steps", 0.0))
        },
        "startup_steps": {
            "value": float(getattr(env, "training_startup_steps", 0.0))
        },
        "active_policy_step_limit": {
            "value": float(getattr(env, "active_policy_step_limit", 0.0))
        },
        "active_policy_seconds_limit": {
            "value": float(
                getattr(env, "active_policy_seconds_limit", 0.0)
            )
        },
    }


def _append_terminal_details(
    snapshot: MetricSnapshot,
    context: _EpisodeSnapshotContext,
) -> None:
    """Append the terminal details owned by this env snapshot module."""
    episode_info = cast(dict[str, object], snapshot["episode_info"])
    if context.terminated:
        episode_info["terminated_reached_intersection"] = {
            "mean": context.reached_intersection
        }
        episode_info["terminated_max_speed"] = {"mean": context.max_speed}
        episode_info["terminated_average_speed"] = {
            "mean": context.average_speed
        }
    if context.truncated:
        episode_info["truncated_reached_intersection"] = {
            "mean": context.reached_intersection
        }
        episode_info["truncated_max_speed"] = {"mean": context.max_speed}
        episode_info["truncated_average_speed"] = {
            "mean": context.average_speed
        }


def _set_nested_metric(
    root: MetricSnapshot,
    path_parts: tuple[str, ...],
    value: float,
) -> None:
    """Set the nested metric owned by this env snapshot module."""
    cursor = root
    for part in path_parts[:-1]:
        cursor = cast(dict[str, object], cursor.setdefault(part, {}))
    cursor[path_parts[-1]] = float(value)


def _overlay_diagnostic_observations(
    snapshot: MetricSnapshot,
    env: object,
    dependencies: EnvSnapshotDependencies,
) -> None:
    """Overlay the diagnostic observations owned by this env snapshot
module."""
    components_getter = getattr(
        env,
        "get_observation_component_series",
        None,
    )
    if not callable(components_getter):
        return

    getter = cast(Callable[[], Mapping[str, object]], components_getter)
    for key, value in getter().items():
        if key not in dependencies.diagnostic_observation_names:
            continue
        if dependencies.is_valid_number(value):
            _set_nested_metric(
                snapshot,
                ("observations", key, "stats", "mean"),
                float(cast(float, value)),
            )


def build_env_lanistai_snapshot(
    env: object,
    observation: object,
    reward: object,
    terminated: bool,
    truncated: bool,
    *,
    dependencies: EnvSnapshotDependencies,
) -> MetricSnapshot:
    """Build one immutable-by-publication environment metrics snapshot."""

    context = _build_episode_context(
        env,
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    snapshot: MetricSnapshot = {
        "episode_info": _episode_info(env, context),
        "vehicle_info": _vehicle_info(env, context),
        "episode_setup": _episode_setup(env),
    }
    _append_terminal_details(snapshot, context)
    snapshot.update(dependencies.reward_snapshot(env, reward))
    snapshot.update(dependencies.observation_snapshot(observation))
    observation_channels = dependencies.observation_channels(observation)
    snapshot["channels"] = {
        "observation": dict(observation_channels["observation"]),
        "telemetry": dict(observation_channels["telemetry"]),
        "reward": dependencies.reward_channels(env, reward),
    }
    _overlay_diagnostic_observations(snapshot, env, dependencies)
    return snapshot
