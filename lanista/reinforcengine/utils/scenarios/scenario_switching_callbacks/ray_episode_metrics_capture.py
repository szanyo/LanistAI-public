# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral capture stages for Ray episode metrics.

This component samples episode and environment state without importing Ray or
RLlib. Curriculum state is observed for metrics only; curriculum decisions and
transitions remain owned by the curriculum manager and state machine.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import cast

import numpy as np

from lanista.reinforcengine.rl_metrics.aggregation import (
    extract_current_observation_from_dict,
    is_valid_number,
)
from lanista.reinforcengine.rl_metrics.channel_schema import (
    DIAGNOSTIC_OBSERVATION_PARAM_NAMES,
    INTERACTION_EPISODE_METRIC_NAMES,
    POLICY_OBSERVATION_PARAM_LENGTHS,
    POLICY_OBSERVATION_PARAM_NAMES,
    TRAINING_DIAGNOSTIC_PARAM_NAMES,
)
from lanista.reinforcengine.rl_metrics.env_snapshot import read_live_vehicle_count
from lanista.reinforcengine.rl_metrics.snapshot_projection import reward_series_values_from_env
from lanista.reinforcengine.rl_metrics.training_state_projection import (
    SCENARIO_ACTIVE_METRIC,
    active_scenario_number,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    is_full_curriculum_state,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_contracts import (
    EpisodeDataPort,
    EpisodeMetricStore,
    EpisodeOutcome,
    EpisodeStatus,
    EpisodeStepPort,
    MetricScalar,
    OptionalStateReader,
    OptionalTelemetryQuality,
    OptionalTelemetryRead,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_store import (
    CURRICULUM_METRICS,
    EPISODE_SETUP_METRICS,
    LEGACY_REWARD_METRIC_TO_SERIES_KEY,
    build_episode_metric_store,
    episode_metric_store,
    metric_series,
    record_optional_quality,
)


def _attribute(source: object, name: str, default: object = None) -> object:
    """Return the attribute value owned by this ray episode metrics capture module."""
    return cast(object, getattr(source, name, default))


def _zero_argument_getter(source: object, name: str) -> Callable[[], object] | None:
    """Return the zero argument getter value owned by this ray episode metrics capture module."""
    candidate = _attribute(source, name)
    return cast(Callable[[], object], candidate) if callable(candidate) else None


def _append_series_values(
    series_map: dict[str, list[float]],
    values: Mapping[str, float],
) -> None:
    """Append the series values owned by this ray episode metrics capture module."""
    for key, value in values.items():
        if key not in series_map or not is_valid_number(value):
            continue
        series_map[key].append(float(value))


def _diagnostic_observation_values(values: Mapping[str, object]) -> dict[str, float]:
    """Return the diagnostic observation values value owned by this ray episode metrics capture
    module.
    """
    return {
        key: float(value)
        for key, value in values.items()
        if key in DIAGNOSTIC_OBSERVATION_PARAM_NAMES and is_valid_number(value)
    }


def extract_latest_observation_values(
    observation: object,
    names: Sequence[str],
) -> dict[str, float]:
    """Decode current values from dictionary, mixed, or legacy flat layouts."""

    if isinstance(observation, dict):
        current_observation = extract_current_observation_from_dict(observation)
        return {
            name: float(value)
            for name in names
            if is_valid_number(value := current_observation.get(name))
        }

    observation_array = np.asarray(observation, dtype=float).reshape(-1)
    name_count = len(names)
    if name_count == 0 or observation_array.size == 0:
        return {}
    if observation_array.size == name_count:
        return {
            name: float(observation_array[index])
            for index, name in enumerate(names)
            if is_valid_number(observation_array[index])
        }

    shared_lengths = [POLICY_OBSERVATION_PARAM_LENGTHS.get(name, 1) for name in names]
    if observation_array.size == sum(shared_lengths):
        return _latest_values_from_mixed_layout(observation_array, names, shared_lengths)
    if observation_array.size % name_count == 0:
        return _latest_values_from_uniform_history(observation_array, names)
    return {
        name: float(observation_array[index])
        for index, name in enumerate(names)
        if index < observation_array.size and is_valid_number(observation_array[index])
    }


def _latest_values_from_mixed_layout(
    observation: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    names: Sequence[str],
    lengths: Sequence[int],
) -> dict[str, float]:
    """Return the latest values from mixed layout value owned by this ray episode metrics capture
    module.
    """
    values: dict[str, float] = {}
    offset = 0
    for name, length in zip(names, lengths, strict=True):
        value = observation[offset + length - 1]
        if is_valid_number(value):
            values[name] = float(value)
        offset += length
    return values


def _latest_values_from_uniform_history(
    observation: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    names: Sequence[str],
) -> dict[str, float]:
    """Return the latest values from uniform history value owned by this ray episode metrics
    capture module.
    """
    history_length = observation.size // len(names)
    return {
        name: float(observation[index * history_length + history_length - 1])
        for index, name in enumerate(names)
        if is_valid_number(observation[index * history_length + history_length - 1])
    }


def get_unwrapped_sub_env(env: object | None, env_index: int = 0) -> object | None:
    """Return the selected unwrapped sub-environment with legacy index fallback."""

    if env is None:
        return None
    if hasattr(env, "envs"):
        environments = cast(Sequence[object], getattr(env, "envs", []))
        if environments:
            index = int(env_index or 0)
            if index < 0 or index >= len(environments):
                index = 0
            selected = environments[index]
            return getattr(selected, "unwrapped", selected)
    return getattr(env, "unwrapped", env)


def _curriculum_metrics_from_state(state: object) -> dict[str, float]:
    """Return the curriculum metrics from state value owned by this ray episode metrics capture
    module.
    """
    if not isinstance(state, dict):
        return {key: 0.0 for key in CURRICULUM_METRICS}
    curriculum_index = state.get("curriculum_index")
    active_level = (
        float(int(cast(float | int | str, curriculum_index)) + 1)
        if is_valid_number(curriculum_index)
        else 0.0
    )
    target_value = state.get("target_value")
    required = state.get("cumulative_min_terminated_episodes")
    return {
        "active_level": active_level,
        "target_intersection": float(target_value) if is_valid_number(target_value) else 0.0,
        "required_terminated_episodes": float(required) if is_valid_number(required) else 0.0,
        "horizon_is_full": float(
            bool(
                state.get(
                    "horizon_is_full",
                    is_full_curriculum_state(state),
                )
            )
        ),
    }


class EpisodeMetricsCapture:
    """Capture one ordered episode-metric stream without framework imports."""

    def __init__(self, state_reader: OptionalStateReader) -> None:
        """Initialize the episode metrics capture and its owned state."""
        self._state_reader = state_reader

    @staticmethod
    def initialize_episode(episode: EpisodeDataPort) -> None:
        """Replace stale custom data with a fresh, stable metric store."""

        episode.custom_data["lanistai"] = build_episode_metric_store()

    @staticmethod
    def capture_status(episode: EpisodeStepPort) -> EpisodeStatus:
        """Read terminal flags once so later sampling cannot change the gate."""

        terminated = episode.get_terminateds()
        truncated = episode.get_truncateds()
        return EpisodeStatus(
            overall_terminated=bool(terminated.get("__all__", False)),
            overall_truncated=bool(truncated.get("__all__", False)),
            is_terminated=bool(episode.is_terminated),
            is_truncated=bool(episode.is_truncated),
            is_done=bool(episode.is_done),
        )

    @staticmethod
    def capture_latest_observation(episode: EpisodeStepPort) -> None:
        """Append the latest finite policy values and their source quality."""

        try:
            observations = episode.get_observations(indices=-1, agent_ids="EGO_agent_0")
            observation = (
                observations.get("EGO_agent_0")
                if isinstance(observations, dict)
                else None
            )
        except Exception as error:
            record_optional_quality(
                episode,
                source="observation",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="observation_read_failed",
                error=error,
            )
            return
        if observation is None:
            record_optional_quality(
                episode,
                source="observation",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="observation_missing",
            )
            return
        values = extract_latest_observation_values(observation, POLICY_OBSERVATION_PARAM_NAMES)
        if not values:
            record_optional_quality(
                episode,
                source="observation",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="observation_values_unavailable",
            )
            return
        record_optional_quality(
            episode,
            source="observation",
            quality=OptionalTelemetryQuality.COMPLETE,
        )
        _append_series_values(episode_metric_store(episode)["observation_series"], values)

    @staticmethod
    def select_environment(env: object | None, env_index: int) -> object | None:
        """Select the callback's indexed unwrapped sub-environment."""

        return get_unwrapped_sub_env(env, env_index)

    @staticmethod
    def _latest_telemetry_values(environment: object) -> OptionalTelemetryRead:
        """Return the latest telemetry values value owned by this episode metrics capture."""
        latest_telemetry: object | None = None
        primary_cause = "telemetry_primary_missing"
        primary_error: BaseException | None = None
        telemetry = _attribute(environment, "telemetry")
        latest_getter = _zero_argument_getter(telemetry, "latest")
        if latest_getter is not None:
            try:
                latest_telemetry = latest_getter()
            except Exception as error:
                primary_cause = "telemetry_primary_read_failed"
                primary_error = error

        if latest_telemetry is not None and hasattr(latest_telemetry, "to_channels"):
            try:
                converter = cast(
                    Callable[[], Mapping[str, object]],
                    _attribute(latest_telemetry, "to_channels"),
                )
                channels = converter()
                primary_values = _diagnostic_observation_values(
                    cast(Mapping[str, object], channels.get("telemetry", {}))
                )
            except Exception as error:
                primary_cause = "telemetry_channel_conversion_failed"
                primary_error = error
            else:
                if primary_values:
                    return OptionalTelemetryRead(
                        primary_values,
                        OptionalTelemetryQuality.COMPLETE,
                    )
                primary_cause = "telemetry_primary_values_unavailable"

        fallback_getter = _zero_argument_getter(
            environment,
            "get_observation_component_series",
        )
        if fallback_getter is not None:
            try:
                fallback_values = _diagnostic_observation_values(
                    cast(Mapping[str, object], fallback_getter())
                )
            except Exception as error:
                return OptionalTelemetryRead(
                    {},
                    OptionalTelemetryQuality.UNAVAILABLE,
                    "telemetry_fallback_read_failed",
                    error,
                )
            if fallback_values:
                return OptionalTelemetryRead(
                    fallback_values,
                    OptionalTelemetryQuality.FALLBACK,
                    primary_cause,
                    primary_error,
                )
        return OptionalTelemetryRead(
            {},
            OptionalTelemetryQuality.UNAVAILABLE,
            primary_cause,
            primary_error,
        )

    @classmethod
    def capture_step_series(
        cls,
        episode: EpisodeDataPort,
        environment: object,
    ) -> None:
        """Append reward and diagnostic telemetry for one environment step."""

        store = episode_metric_store(episode)
        component_getter = _zero_argument_getter(
            environment,
            "get_reward_component_series",
        )
        reward_value = None if component_getter is not None else _attribute(
            environment,
            "step_reward",
        )
        _append_series_values(
            store["reward_series"],
            reward_series_values_from_env(environment, reward_value),
        )
        telemetry_read = cls._latest_telemetry_values(environment)
        record_optional_quality(
            episode,
            source="environment",
            quality=telemetry_read.quality,
            cause=telemetry_read.cause,
            error=telemetry_read.error,
        )
        _append_series_values(store["telemetry_series"], telemetry_read.values)

    @staticmethod
    def capture_diagnostic_series(
        episode: EpisodeDataPort,
        environment: object,
        status: EpisodeStatus,
    ) -> None:
        """Append finite semantic diagnostics from the existing step snapshot."""

        getter = _attribute(
            environment,
            "get_training_diagnostic_component_series",
        )
        if not callable(getter):
            return
        try:
            values = getter(
                terminated=status.terminated,
                truncated=status.truncated,
            )
        except Exception as error:
            record_optional_quality(
                episode,
                source="diagnostic",
                quality=OptionalTelemetryQuality.UNAVAILABLE,
                cause="diagnostic_projection_failed",
                error=error,
            )
            return
        if not isinstance(values, Mapping):
            return
        selected = {
            name: float(value)
            for name in TRAINING_DIAGNOSTIC_PARAM_NAMES
            if is_valid_number(value := values.get(name))
        }
        _append_series_values(
            episode_metric_store(episode)["diagnostic_series"],
            selected,
        )

    @staticmethod
    def _episode_outcome(environment: object) -> EpisodeOutcome:
        """Return the episode outcome value owned by this episode metrics capture."""
        termination_reason = _attribute(environment, "last_termination_reason")
        truncation_reason = _attribute(environment, "last_truncation_reason")
        terminated = bool(_attribute(environment, "episode_terminated"))
        truncated = bool(_attribute(environment, "episode_truncated"))
        if truncated:
            terminated = False
        detected = bool(_attribute(environment, "curriculum_target_detected", False))
        conflict = bool(
            _attribute(
                environment,
                "episode_end_conflict_target_red_light_overdrive",
                False,
            )
        )
        return EpisodeOutcome(
            terminated=terminated,
            truncated=truncated,
            termination_reason=termination_reason,
            truncation_reason=truncation_reason,
            curriculum_target_detected=detected,
            curriculum_target_accepted=bool(
                _attribute(
                    environment,
                    "curriculum_target_accepted",
                    terminated and termination_reason == "curriculum_target",
                )
            ),
            curriculum_target_rejected_by_red_light_overdrive=bool(
                _attribute(
                    environment,
                    "curriculum_target_rejected_by_red_light_overdrive",
                    detected and conflict,
                )
            ),
        )

    @staticmethod
    def _capture_outcome_metrics(
        store: EpisodeMetricStore,
        environment: object,
        outcome: EpisodeOutcome,
    ) -> None:
        """Capture the outcome metrics required by this episode metrics capture."""
        store["episode_terminated"].append(outcome.terminated)
        store["episode_truncated"].append(outcome.truncated)
        for key in (
            "episode_end_conflict",
            "episode_end_conflict_target_time_limit",
            "episode_end_conflict_target_red_light_overdrive",
        ):
            metric_series(store, key).append(float(bool(_attribute(environment, key, False))))
        store["curriculum_target_detected"].append(float(outcome.curriculum_target_detected))
        store["curriculum_target_accepted"].append(float(outcome.curriculum_target_accepted))
        store["curriculum_target_reached"].append(float(outcome.curriculum_target_accepted))
        store["curriculum_target_rejected_by_red_light_overdrive"].append(
            float(outcome.curriculum_target_rejected_by_red_light_overdrive)
        )
        store["termination_curriculum_target"].append(
            float(outcome.terminated and outcome.termination_reason == "curriculum_target")
        )
        store["termination_route_end"].append(
            float(outcome.terminated and outcome.termination_reason == "route_end")
        )
        store["truncation_time_limit"].append(
            float(outcome.truncated and outcome.truncation_reason == "time_limit")
        )
        store["truncation_red_light_overdrive"].append(
            float(outcome.truncated and outcome.truncation_reason == "red_light_overdrive")
        )

    @staticmethod
    def _capture_vehicle_metrics(
        store: EpisodeMetricStore,
        environment: object,
        outcome: EpisodeOutcome,
    ) -> None:
        """Capture the vehicle metrics required by this episode metrics capture."""
        vehicle = _attribute(environment, "ego_vehicle")
        scenario = _attribute(environment, "scenario_context_preprocessor")
        max_speed = float(cast(float | int | str, _attribute(vehicle, "MaxSpeed")))
        average_speed = float(
            cast(float | int | str, _attribute(vehicle, "AverageSpeed"))
        )
        intersection = float(
            cast(float | int | str, _attribute(scenario, "intersection_number"))
        )
        store["terminated_reached_intersection"].append(
            intersection if outcome.terminated else float("nan")
        )
        store["terminated_max_speed"].append(
            max_speed if outcome.terminated else float("nan")
        )
        store["terminated_average_speed"].append(
            average_speed if outcome.terminated else float("nan")
        )
        store["truncated_reached_intersection"].append(
            intersection if outcome.truncated else float("nan")
        )
        store["truncated_max_speed"].append(
            max_speed if outcome.truncated else float("nan")
        )
        store["truncated_average_speed"].append(
            average_speed if outcome.truncated else float("nan")
        )
        store["max_speed_per_episode"].append(max_speed)
        store["average_speed_per_episode"].append(average_speed)
        store["reached_intersection_number"].append(intersection)
        store["number_of_vehicles"].append(read_live_vehicle_count(environment))
        store["number_of_pedestrians"].append(
            cast(MetricScalar, _attribute(environment, "pedestrian_id_counter"))
        )
        tracker = _attribute(environment, "episode_interactions")
        summary = getattr(tracker, "summary", None)
        interaction_metrics = summary() if callable(summary) else {}
        for name in INTERACTION_EPISODE_METRIC_NAMES:
            metric_series(store, name).append(float(interaction_metrics.get(name, 0.0)))

    def _capture_training_state_and_setup(
        self,
        episode: EpisodeDataPort,
        environment: object,
    ) -> None:
        """Capture curriculum, scenario, and setup state for one episode."""
        store = episode_metric_store(episode)
        state_read = self._state_reader(
            _attribute(environment, "_curriculum_index_state")
        )
        record_optional_quality(
            episode,
            source="curriculum_state",
            quality=state_read.quality,
            cause=state_read.cause,
            error=state_read.error,
        )
        for key, value in _curriculum_metrics_from_state(state_read.value).items():
            store["curriculum_metrics"][key].append(value)
        scenario_read = self._state_reader(
            _attribute(environment, "_scenario_index_state")
        )
        record_optional_quality(
            episode,
            source="scenario_state",
            quality=scenario_read.quality,
            cause=scenario_read.cause,
            error=scenario_read.error,
        )
        store["scenario_metrics"][SCENARIO_ACTIVE_METRIC].append(
            active_scenario_number(scenario_read.value)
        )
        for key, attribute_name in EPISODE_SETUP_METRICS.items():
            setup_value = _attribute(environment, attribute_name)
            if is_valid_number(setup_value):
                store["episode_setup_series"][key].append(float(setup_value))

    @staticmethod
    def _capture_legacy_reward_metrics(
        store: EpisodeMetricStore,
        environment: object,
    ) -> None:
        """Capture the legacy reward metrics required by this episode metrics capture."""
        component_getter = _zero_argument_getter(
            environment,
            "get_reward_component_series",
        )
        if component_getter is not None:
            component_series = cast(Mapping[str, object], component_getter())
            for metric_name, series_key in LEGACY_REWARD_METRIC_TO_SERIES_KEY.items():
                metric_series(store, metric_name).append(
                    cast(MetricScalar, component_series[series_key])
                )
            return
        for metric_name in LEGACY_REWARD_METRIC_TO_SERIES_KEY:
            metric_series(store, metric_name).append(
                cast(MetricScalar, _attribute(environment, metric_name))
            )

    def capture_terminal_snapshot(
        self,
        episode: EpisodeDataPort,
        environment: object,
    ) -> None:
        """Append terminal values without changing curriculum ownership or state."""

        store = episode_metric_store(episode)
        outcome = self._episode_outcome(environment)
        self._capture_outcome_metrics(store, environment, outcome)
        self._capture_vehicle_metrics(store, environment, outcome)
        self._capture_training_state_and_setup(episode, environment)
        self._capture_legacy_reward_metrics(store, environment)
