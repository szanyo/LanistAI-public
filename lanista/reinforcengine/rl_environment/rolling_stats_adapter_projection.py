# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral projections shared by rolling-statistics adapters."""

from __future__ import annotations

from collections import deque
from typing import Any, NotRequired, Protocol, TypedDict, cast

from lanista.foundation.logging import LoggerCondition
from lanista.reinforcengine.rl_environment.rolling_stats_conditions import (
    build_conditions,
)
from lanista.reinforcengine.rl_environment.rolling_stats_configuration import (
    CurriculumConfiguration,
    CurriculumProjector,
    coerce_number,
    config_get,
    threshold_lower_bounds,
    threshold_upper_bounds,
)
from lanista.reinforcengine.rl_environment.rolling_stats_decision_core import (
    CurriculumTargetState,
    EpisodeCounts,
    FullCurriculumEpisodeTracker,
    LowerBoundPolicy,
    NormalizedWindowMetrics,
    OptionalUpperBoundPolicy,
    PreferredOverrideWindow,
    ScenarioSwitchPolicy,
    TrainingStabilityPolicy,
    evaluate_curriculum_readiness,
    evaluate_scenario_readiness,
    evaluate_training_stability,
)
from lanista.reinforcengine.rl_environment.training_progress import (
    TrainingProgressEvidence,
)


class EpisodeCountSummary(TypedDict):
    """Stable training-global terminated/truncated count projection."""

    terminated: int
    truncated: int
    total: int


class WindowStats(TypedDict):
    """Framework-normalized active-window statistics."""

    terminated_in_window: int | float
    truncated_in_window: int | float
    termination_rate: float
    red_light_overdrive_rate: NotRequired[float]
    time_limit_truncation_rate: NotRequired[float]


class EpisodeCountPort(Protocol):
    """Framework-specific training-global episode counters."""

    def _all_terminated_episodes(self) -> int:
        """Return all terminated episodes retained by the adapter."""

    def _all_truncated_episodes(self) -> int:
        """Return all truncated episodes retained by the adapter."""


class ScenarioEpisodeCountPort(Protocol):
    """Framework-specific active-scenario episode counters."""

    def _overall_terminated_episodes(self) -> int:
        """Return the adapter's active-scenario terminated count."""

    def _overall_truncated_episodes(self) -> int:
        """Return the adapter's active-scenario truncated count."""


class CurriculumReadinessPort(ScenarioEpisodeCountPort, Protocol):
    """Adapter operations needed for shared curriculum readiness."""

    def _get_window_stats(self) -> WindowStats | None:
        """Return framework-normalized window statistics."""

    def _get_preferred_override_window_stats(
        self,
        required_episodes: int,
    ) -> WindowStats | None:
        """Return recent statistics spanning the requested episode evidence."""

    def log_decision(
        self,
        *,
        method: str,
        conditions: list[LoggerCondition],
        decision: bool,
    ) -> None:
        """Publish the stable condition projection."""


class ScenarioReadinessPort(CurriculumReadinessPort, Protocol):
    """Framework-normalized scenario progress and reward samples."""

    def _overall_observed_episodes(self) -> int:
        """Count observations using the backend's episode record format."""

    def _scenario_rewards(self) -> tuple[float, ...]:
        """Return native per-episode or per-result reward samples."""


def episode_count_summary(port: EpisodeCountPort) -> EpisodeCountSummary:
    """Project training-global counts without scenario-baseline rebasing."""

    terminated = port._all_terminated_episodes()
    truncated = port._all_truncated_episodes()
    return {
        "terminated": terminated,
        "truncated": truncated,
        "total": terminated + truncated,
    }


def training_progress_evidence(
    port: ScenarioReadinessPort,
    curriculum_state: object | None = None,
) -> TrainingProgressEvidence:
    """Expose the same active-scenario milestone counters for both adapters."""

    terminated = port._overall_terminated_episodes()
    truncated = port._overall_truncated_episodes()
    episode_counts = EpisodeCounts(
        terminated=terminated,
        truncated=truncated,
        total=terminated + truncated,
    )
    curriculum = curriculum_target(
        port,
        curriculum_state,
        include_progress=False,
    )
    full_curriculum_episodes = _observe_full_curriculum_episodes(
        port,
        curriculum_full=curriculum.full,
        episode_counts=episode_counts,
    )
    return TrainingProgressEvidence(
        terminated_episodes=terminated,
        total_episodes=port._overall_observed_episodes(),
        window_episodes=len(getattr(port, "_windowed_values", ())),
        full_curriculum_episodes=full_curriculum_episodes,
    )


def scenario_ready(
    port: ScenarioReadinessPort,
    curriculum_state: object | None = None,
) -> bool:
    """Evaluate one shared full-curriculum dwell and safety window."""

    terminated = port._overall_terminated_episodes()
    truncated = port._overall_truncated_episodes()
    episode_counts = EpisodeCounts(
        terminated=terminated,
        truncated=truncated,
        total=terminated + truncated,
    )
    curriculum = curriculum_target(
        port,
        curriculum_state,
        include_progress=False,
    )
    full_curriculum_episodes = _observe_full_curriculum_episodes(
        port,
        curriculum_full=curriculum.full,
        episode_counts=episode_counts,
    )

    decision = evaluate_scenario_readiness(
        window_episode_count=len(getattr(port, "_windowed_values", ())),
        episode_counts=EpisodeCounts(
            terminated=terminated,
            truncated=truncated,
            total=port._overall_observed_episodes(),
        ),
        policy=_scenario_switch_policy(port),
        curriculum=curriculum,
        full_curriculum_episode_count=full_curriculum_episodes,
        window_metrics=_normalized_window_metrics(port._get_window_stats()),
        preferred_override_window=_preferred_override_window(
            port,
            _scenario_switch_policy(port).preferred_override_window_size,
        ),
    )
    projected_conditions = build_conditions(decision.conditions)
    port.log_decision(
        method="RollingStats.scenario_ready",
        conditions=projected_conditions,
        decision=decision.decision,
    )
    return decision.decision


def configure_scenario_readiness(port: object, configuration: object) -> None:
    """Read the same optional dwell and readiness settings for both adapters."""

    mutable_port = cast(Any, port)
    mutable_port._min_full_curriculum_episodes = config_get(
        configuration, "min_full_curriculum_episodes", 0,
    )
    mutable_port._scenario_preferred_override_window_size = config_get(
        configuration, "preferred_override_window_size", 0,
    )
    for key in (
        "termination_ratio_threshold",
        "red_light_overdrive_rate_threshold",
        "time_limit_truncation_rate_threshold",
    ):
        setattr(port, f"_scenario_{key}", config_get(configuration, key, None))


def _observe_full_curriculum_episodes(
    port: object,
    *,
    curriculum_full: bool,
    episode_counts: EpisodeCounts,
) -> int:
    """Share the stateful full-curriculum dwell counter across projections."""

    tracker = getattr(port, "_full_curriculum_episode_tracker", None)
    if not isinstance(tracker, FullCurriculumEpisodeTracker):
        tracker = FullCurriculumEpisodeTracker()
        cast(Any, port)._full_curriculum_episode_tracker = tracker
    transition_started = curriculum_full and tracker.baseline_counts is None
    observed = tracker.observe(
        curriculum_full=curriculum_full,
        episode_counts=episode_counts,
    )
    if transition_started:
        preferred_override_records(port).clear()
    return int(observed)


def _scenario_switch_policy(port: object) -> ScenarioSwitchPolicy:
    """Normalize adapter configuration into the shared scenario policy."""

    termination = _attribute(port, "_scenario_termination_ratio_threshold", None)
    red_light = _attribute(port, "_scenario_red_light_overdrive_rate_threshold", None)
    time_limit = _attribute(port, "_scenario_time_limit_truncation_rate_threshold", None)
    return ScenarioSwitchPolicy(
        lookback_window_size=coerce_number(_attribute(port, "_lookback_window_size", 0), 0, int),
        minimum_episodes=coerce_number(
            _attribute(port, "_min_episodes_for_switch_per_scenario", 0), 0, int,
        ),
        minimum_terminated_episodes=coerce_number(
            _attribute(port, "_min_terminated_episodes_for_switch_per_scenario", 0), 0, int,
        ),
        minimum_full_curriculum_episodes=coerce_number(
            _attribute(port, "_min_full_curriculum_episodes", 0), 0, int,
        ),
        termination_ratio=(
            LowerBoundPolicy(*threshold_lower_bounds(termination))
            if termination is not None else None
        ),
        red_light_overdrive_rate=(
            OptionalUpperBoundPolicy(*threshold_upper_bounds(red_light))
            if red_light is not None else None
        ),
        time_limit_truncation_rate=(
            OptionalUpperBoundPolicy(*threshold_upper_bounds(time_limit))
            if time_limit is not None else None
        ),
        preferred_override_window_size=max(0, coerce_number(
            _attribute(port, "_scenario_preferred_override_window_size", 0),
            0,
            int,
        )),
    )


def _normalized_window_metrics(stats: WindowStats | None) -> NormalizedWindowMetrics | None:
    """Convert native window counts and rates once for both readiness decisions."""

    if stats is None:
        return None
    terminated = int(stats["terminated_in_window"])
    truncated = int(stats["truncated_in_window"])
    return NormalizedWindowMetrics(
        episodes=EpisodeCounts(terminated, truncated, terminated + truncated),
        termination_ratio=float(stats["termination_rate"]),
        red_light_overdrive_rate=float(stats.get("red_light_overdrive_rate", 0.0)),
        time_limit_truncation_rate=float(stats.get("time_limit_truncation_rate", 0.0)),
    )


def metrics_ready(port: ScenarioReadinessPort) -> bool:
    """Apply the shared positive-reward gate to native framework samples."""

    stability = _attribute(port, "_episode_reward_stability", None)
    decision = evaluate_training_stability(
        rewards=port._scenario_rewards(),
        policy=TrainingStabilityPolicy(
            minimum_mean_reward=coerce_number(config_get(
                stability, "target_mean_by_lookback_window", 0.0,
            ), 0.0, float),
        ),
    )
    port.log_decision(
        method="RollingStats.metrics_ready",
        conditions=build_conditions(decision.conditions),
        decision=decision.decision,
    )
    return decision.decision


def reset_scenario_readiness(port: object) -> None:
    """Reset full-curriculum dwell tracking at a scenario boundary."""

    tracker = getattr(port, "_full_curriculum_episode_tracker", None)
    if isinstance(tracker, FullCurriculumEpisodeTracker):
        tracker.reset()


def curriculum_projector(port: object) -> CurriculumProjector:
    """Compose immutable curriculum defaults from one adapter instance."""

    minimum = coerce_number(
        _attribute(port, "_curriculum_min_terminated_episodes", 0),
        0,
        int,
    )
    legacy_ratio = _attribute(
        port,
        "_curriculum_min_termination_rate",
        0.0,
    )
    return CurriculumProjector(
        CurriculumConfiguration(
            minimum_terminated_episodes=minimum,
            termination_ratio_threshold=_attribute(
                port,
                "_curriculum_termination_ratio_threshold",
                legacy_ratio,
            ),
            red_light_overdrive_rate_threshold=_attribute(
                port,
                "_curriculum_red_light_overdrive_rate_threshold",
                None,
            ),
            time_limit_truncation_rate_threshold=_attribute(
                port,
                "_curriculum_time_limit_truncation_rate_threshold",
                None,
            ),
            preferred_override_window_size=max(0, coerce_number(
                _attribute(port, "_curriculum_preferred_override_window_size", 0),
                0,
                int,
            )),
        )
    )


def curriculum_target(
    port: object,
    state_or_conditions: object | None,
    *,
    base_minimum: int | None = None,
    include_progress: bool = True,
) -> CurriculumTargetState:
    """Project one curriculum target through the adapter's defaults."""

    return curriculum_projector(port).target(
        state_or_conditions,
        base_minimum=base_minimum,
        include_progress=include_progress,
    )


def curriculum_ready(
    port: CurriculumReadinessPort,
    conditions: dict[str, object] | None,
) -> bool:
    """Bound public method sharing policy projection and diagnostics."""

    window_stats = port._get_window_stats()
    if window_stats is None:
        return False

    projector = curriculum_projector(port)
    policy = projector.policy(conditions)
    terminated = port._overall_terminated_episodes()
    truncated = port._overall_truncated_episodes()
    core_decision = evaluate_curriculum_readiness(
        overall_episode_counts=EpisodeCounts(
            terminated=terminated,
            truncated=truncated,
            total=terminated + truncated,
        ),
        window_metrics=_normalized_window_metrics(window_stats),
        policy=policy,
        curriculum=projector.target(
            conditions,
            base_minimum=policy.minimum_terminated_episodes,
        ),
        preferred_override_window=_preferred_override_window(
            port,
            policy.preferred_override_window_size,
        ),
    )
    projected_conditions = build_conditions(core_decision.conditions)
    port.log_decision(
        method="RollingStats.curriculum_ready",
        conditions=projected_conditions,
        decision=core_decision.decision,
    )
    return core_decision.decision


def _attribute(target: object, name: str, default: object) -> object:
    """Return the attribute value owned by this rolling stats adapter projection module."""
    return cast(object, getattr(target, name, default))


def _preferred_override_window(
    port: object,
    required_episodes: int,
) -> PreferredOverrideWindow | None:
    """Normalize an adapter-owned recent episode window for shared decisions."""

    if required_episodes <= 0:
        return None
    stats_reader = getattr(port, "_get_preferred_override_window_stats", None)
    if not callable(stats_reader):
        return PreferredOverrideWindow(
            metrics=None,
            observed_episodes=0,
            required_episodes=required_episodes,
        )
    metrics = _normalized_window_metrics(stats_reader(required_episodes))
    return PreferredOverrideWindow(
        metrics=metrics,
        observed_episodes=metrics.episodes.total if metrics is not None else 0,
        required_episodes=required_episodes,
    )


def preferred_override_records(port: object) -> deque[dict]:
    """Return the recent repository, including legacy restored instances."""

    records = getattr(port, "_preferred_override_windowed_values", None)
    if records is None:
        capacity = max(
            coerce_number(
                getattr(port, "_scenario_preferred_override_window_size", 0),
                0,
                int,
            ),
            coerce_number(
                getattr(port, "_curriculum_preferred_override_window_size", 0),
                0,
                int,
            ),
        )
        records = deque(maxlen=capacity)
        cast(Any, port)._preferred_override_windowed_values = records
    return cast(deque[dict], records)
