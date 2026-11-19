# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Thin RLlib hook adapter for ordered project episode metrics."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import cast

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from lanista.reinforcengine.rl_metrics.channel_schema import (
    PER_EPISODE_METRICS,
    POLICY_OBSERVATION_PARAM_NAMES,
    REWARD_SERIES_KEYS,
)
from lanista.reinforcengine.rl_metrics.checkpoint_observation_csv import (
    write_checkpoint_observation_csv,
)
from lanista.reinforcengine.rl_technologies.ray.episode_trace_capture import (
    EpisodeTraceClaim,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_capture import (
    EpisodeMetricsCapture,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_contracts import (
    EpisodeDataPort,
    EpisodeMetricsCapturePort,
    EpisodeMetricsPublisherPort,
    EpisodeStatus,
    EpisodeStepPort,
    MetricsLoggerPort,
    OptionalStateRead,
    OptionalTelemetryQuality,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_publisher import (
    EpisodeMetricsPublisher,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.ray_episode_metrics_store import (
    CURRICULUM_METRICS,
    EPISODE_SETUP_METRICS,
    SCENARIO_METRICS,
    episode_metric_store,
)

_TRACE_CAPTURE_KEY = "_lanistai_episode_trace_capture"
_LOGGER = logging.getLogger(__name__)


def _trace_capture_state(episode: EpisodeDataPort) -> dict[str, object]:
    """Return the mutable optional trace state installed on one episode."""

    state = episode.custom_data.get(_TRACE_CAPTURE_KEY)
    if isinstance(state, dict):
        return state
    state = {
        "token": uuid.uuid4().hex,
        "claim_attempted": False,
        "gate": None,
        "checkpoint_paths": (),
        "rows": [],
        "projection_failed": False,
    }
    episode.custom_data[_TRACE_CAPTURE_KEY] = state
    return state


def _read_optional_state(state: object | None) -> OptionalStateRead:
    """Read the optional state without changing this scenario switching callback ray rllib module's
    state.
    """
    if state is None:
        return OptionalStateRead(
            None,
            OptionalTelemetryQuality.UNAVAILABLE,
            "state_missing",
        )
    getter_candidate = cast(object, getattr(state, "get", None))
    remote_candidate = cast(object, getattr(getter_candidate, "remote", None))
    if (
        not callable(remote_candidate)
        and not callable(getter_candidate)
    ):
        return OptionalStateRead(
            None,
            OptionalTelemetryQuality.UNAVAILABLE,
            "state_getter_missing",
        )
    try:
        if callable(remote_candidate):
            value = ray.get(
                cast(Callable[[], object], remote_candidate)()
            )
        else:
            value = cast(Callable[[], object], getter_candidate)()
    except Exception as error:
        return OptionalStateRead(
            None,
            OptionalTelemetryQuality.UNAVAILABLE,
            "state_read_failed",
            error,
        )
    return OptionalStateRead(value, OptionalTelemetryQuality.COMPLETE)


class RayScenarioSwitchingCallback(DefaultCallbacks):
    """Translate RLlib lifecycle hooks into ordered typed metric stages.

    Capture and publication stay framework-neutral and pickle-safe. This
    adapter alone owns Ray actor reads and the RLlib callback inheritance.
    """

    PER_EPISODE_METRICS = PER_EPISODE_METRICS
    OBSERVATION_PARAM_NAMES = POLICY_OBSERVATION_PARAM_NAMES
    REWARD_SERIES_KEYS = REWARD_SERIES_KEYS
    EPISODE_SETUP_METRICS = EPISODE_SETUP_METRICS
    CURRICULUM_METRICS = CURRICULUM_METRICS
    SCENARIO_METRICS = SCENARIO_METRICS

    def __init__(
        self,
        capture: EpisodeMetricsCapturePort | None = None,
        publisher: EpisodeMetricsPublisherPort | None = None,
    ) -> None:
        """Initialize the ray scenario switching callback and its owned state."""
        super().__init__()
        self._capture = (
            capture if capture is not None else EpisodeMetricsCapture(_read_optional_state)
        )
        self._publisher = publisher if publisher is not None else EpisodeMetricsPublisher()

    def on_environment_created(
        self,
        *,
        metrics_logger: MetricsLoggerPort | None = None,
        **ignored: object,
    ) -> None:
        """Initialize stable logger keys before RLlib completes an episode."""

        _ = ignored
        self._publisher.initialize_environment_metrics(
            cast(MetricsLoggerPort, metrics_logger)
        )

    def on_episode_created(
        self,
        *,
        episode: object,
        metrics_logger: object,
        **ignored: object,
    ) -> None:
        """Install a fresh stable metric store for one RLlib episode."""

        _ = metrics_logger, ignored
        self._capture.initialize_episode(cast(EpisodeDataPort, episode))
        _trace_capture_state(cast(EpisodeDataPort, episode))

    def on_episode_step(
        self,
        *,
        episode: object,
        env: object | None = None,
        env_index: int = 0,
        **ignored: object,
    ) -> None:
        """Capture one step in the historical observation-to-terminal order."""

        _ = ignored
        step_episode = cast(EpisodeStepPort, episode)
        status = self._capture.capture_status(step_episode)
        self._capture.capture_latest_observation(step_episode)
        environment = self._capture.select_environment(env, env_index)
        if environment is None:
            return
        self._capture.capture_step_series(step_episode, environment)
        self._capture.capture_diagnostic_series(
            step_episode,
            environment,
            status,
        )
        if status.requires_terminal_snapshot:
            self._capture.capture_terminal_snapshot(step_episode, environment)
        self._capture_diagnostic_trace_step(
            step_episode,
            environment,
            status,
        )

    def on_episode_end(
        self,
        *,
        episode: object,
        metrics_logger: MetricsLoggerPort,
        **ignored: object,
    ) -> None:
        """Publish captured values in the stable downstream-consumer order."""

        _ = ignored
        episode_port = cast(EpisodeDataPort, episode)
        try:
            store = episode_metric_store(episode_port)
            self._publisher.publish_episode_scalars(store, metrics_logger)
            self._publisher.publish_curriculum_metrics(store, metrics_logger)
            self._publisher.publish_scenario_metrics(store, metrics_logger)
            self._publisher.publish_intersection_metrics(store, metrics_logger)
            self._publisher.publish_observation_statistics(
                store["observation_series"],
                metrics_logger,
                diagnostic_only=False,
            )
            self._publisher.publish_observation_statistics(
                store["telemetry_series"],
                metrics_logger,
                diagnostic_only=True,
            )
            self._publisher.publish_episode_setup_metrics(store, metrics_logger)
            self._publisher.publish_reward_statistics(store, metrics_logger)
            self._publisher.publish_diagnostic_statistics(store, metrics_logger)
            self._publisher.record_episode_histograms(episode_port)
            self._publisher.publish_optional_telemetry_quality(store, metrics_logger)
        finally:
            self._finalize_diagnostic_trace(episode_port)

    @staticmethod
    def _capture_diagnostic_trace_step(
        episode: EpisodeDataPort,
        environment: object,
        status: EpisodeStatus,
    ) -> None:
        """Project raw fields only for the episode holding the atomic claim."""

        state = _trace_capture_state(episode)
        if not state["claim_attempted"]:
            state["claim_attempted"] = True
            gate = getattr(environment, "_diagnostic_trace_capture_gate", None)
            claim_method = getattr(gate, "claim", None)
            remote = getattr(claim_method, "remote", None)
            if not callable(remote):
                return
            try:
                claim = ray.get(remote(state["token"]))
            except Exception:
                return
            if not isinstance(claim, EpisodeTraceClaim):
                return
            state["token"] = claim.token
            state["gate"] = gate
            state["checkpoint_paths"] = claim.checkpoint_paths

        if not state["checkpoint_paths"]:
            return
        row_provider = getattr(environment, "latest_diagnostic_trace_row", None)
        if not callable(row_provider):
            return
        scenario_read = _read_optional_state(
            getattr(environment, "_scenario_index_state", None)
        )
        scenario_index = (
            int(cast(int | str, scenario_read.value))
            if scenario_read.value is not None
            else None
        )
        try:
            row = row_provider(
                scenario_index=scenario_index,
                terminated=status.terminated,
                truncated=status.truncated,
            )
        except Exception:
            state["projection_failed"] = True
            return
        if not isinstance(row, Mapping):
            state["projection_failed"] = True
            return
        output_row = {
            "trace_run_id": str(state["token"]),
            "framework": "ray",
            "mode": "training",
        }
        output_row.update({str(key): value for key, value in row.items()})
        output_row.update(
            {
                "trace_run_id": str(state["token"]),
                "framework": "ray",
                "mode": "training",
            }
        )
        output_row["_diagnostic_trace_fields"] = tuple(output_row)
        rows = cast(list[dict[str, object]], state["rows"])
        rows.append(output_row)

    @staticmethod
    def _finalize_diagnostic_trace(episode: EpisodeDataPort) -> None:
        """Write every claimed checkpoint or atomically requeue the claim."""

        state = _trace_capture_state(episode)
        checkpoint_paths = cast(tuple[str, ...], state["checkpoint_paths"])
        gate = state["gate"]
        if not checkpoint_paths or gate is None:
            return
        rows = cast(list[dict[str, object]], state["rows"])
        succeeded = bool(rows) and not bool(state["projection_failed"])
        if succeeded:
            try:
                for checkpoint_path in checkpoint_paths:
                    write_checkpoint_observation_csv(checkpoint_path, rows)
            except Exception as error:
                _LOGGER.warning(
                    "ray_checkpoint_trace_write_failed: %s: %s",
                    type(error).__name__,
                    error,
                )
                succeeded = False
        method_name = "complete" if succeeded else "abandon"
        operation = getattr(getattr(gate, method_name, None), "remote", None)
        if not callable(operation):
            return
        with suppress(Exception):
            ray.get(operation(state["token"]))
