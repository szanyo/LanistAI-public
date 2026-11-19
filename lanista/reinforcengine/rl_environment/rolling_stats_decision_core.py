# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral scenario and curriculum readiness decisions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal

from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    CurriculumState,
    is_full_curriculum_state,
)

type ConditionValue = str | int | float | bool | None
type ConditionOperator = Literal[">", ">=", "<=", "==", "active", "target", "satisfies"]


@dataclass(frozen=True, slots=True)
class DecisionConditionCheck:
    """One named branch of a compound decision condition."""

    label: str
    lhs_val: ConditionValue
    operator: ConditionOperator
    rhs_val: ConditionValue
    result: bool


@dataclass(frozen=True, slots=True)
class DecisionCondition:
    """One ordered, logger-independent predicate in a switch decision."""

    lhs_name: str
    lhs_val: ConditionValue
    operator: ConditionOperator
    rhs_name: str
    rhs_val: ConditionValue
    result: bool
    checks: tuple[DecisionConditionCheck, ...] = ()


@dataclass(frozen=True, slots=True)
class SwitchDecision:
    """Pure decision plus its stable diagnostic projection."""

    conditions: tuple[DecisionCondition, ...]
    decision: bool
    evaluated: bool = True


@dataclass(frozen=True, slots=True)
class EpisodeCounts:
    """Normalized terminated, truncated, and observed episode counts."""

    terminated: int
    truncated: int
    total: int


@dataclass(frozen=True, slots=True)
class NormalizedWindowMetrics:
    """Framework-neutral metrics derived from one adapter-owned window."""

    episodes: EpisodeCounts
    termination_ratio: float
    red_light_overdrive_rate: float = 0.0
    time_limit_truncation_rate: float = 0.0


@dataclass(frozen=True, slots=True)
class PreferredOverrideWindow:
    """Recent episode evidence allowed to replace only a preferred bound."""

    metrics: NormalizedWindowMetrics | None
    observed_episodes: int
    required_episodes: int

    @property
    def full(self) -> bool:
        """Return whether the configured completed-episode window is populated."""

        return (
            self.required_episodes > 0
            and self.observed_episodes >= self.required_episodes
            and self.metrics is not None
        )


@dataclass(frozen=True, slots=True)
class ScenarioSwitchPolicy:
    """Minimum progress required before changing the active scenario."""

    lookback_window_size: int
    minimum_episodes: int
    minimum_terminated_episodes: int
    minimum_full_curriculum_episodes: int = 0
    termination_ratio: LowerBoundPolicy | None = None
    red_light_overdrive_rate: OptionalUpperBoundPolicy | None = None
    time_limit_truncation_rate: OptionalUpperBoundPolicy | None = None
    preferred_override_window_size: int = 0


@dataclass(frozen=True, slots=True)
class LowerBoundPolicy:
    """Acceptable and preferred lower thresholds for one metric."""

    acceptable: float
    preferred: float


@dataclass(frozen=True, slots=True)
class OptionalUpperBoundPolicy:
    """Optional acceptable and preferred upper thresholds for one metric."""

    acceptable: float | None
    preferred: float | None


@dataclass(frozen=True, slots=True)
class TrainingStabilityPolicy:
    """Shared minimum recent-reward requirement (historic policy name)."""

    minimum_mean_reward: float


@dataclass(frozen=True, slots=True)
class CurriculumSwitchPolicy:
    """Normalized readiness policy for one curriculum transition."""

    minimum_terminated_episodes: int
    termination_ratio: LowerBoundPolicy
    red_light_overdrive_rate: OptionalUpperBoundPolicy
    time_limit_truncation_rate: OptionalUpperBoundPolicy
    preferred_override_window_size: int = 0


@dataclass(frozen=True, slots=True)
class CurriculumTargetState:
    """Immutable curriculum identity and cumulative progress metadata."""

    scenario_state_name: ConditionValue
    diagnostic_state_name: str
    target_type: ConditionValue
    target_value: ConditionValue
    curriculum_index: int | None
    cumulative_minimum_terminated_episodes: int | None
    full: bool

    @property
    def curriculum_level(self) -> int | str:
        """Return the one-based partial level or the historic ``full`` label."""

        if self.curriculum_index is None:
            return "full"
        return self.curriculum_index + 1

    def required_terminated_episodes(self, base_minimum: int) -> int:
        """Resolve explicit cumulative progress before index-derived progress."""

        if self.cumulative_minimum_terminated_episodes is not None:
            return self.cumulative_minimum_terminated_episodes
        if self.curriculum_index is None:
            return base_minimum
        return base_minimum * (self.curriculum_index + 1)


@dataclass(slots=True)
class FullCurriculumEpisodeTracker:
    """Project episode counts after a scenario first reaches full curriculum."""

    baseline_counts: EpisodeCounts | None = None

    def observe(
        self,
        *,
        curriculum_full: bool,
        episode_counts: EpisodeCounts,
    ) -> int:
        """Return completed full-curriculum episodes since the transition boundary."""

        if not curriculum_full:
            self.reset()
            return 0
        normalized_counts = self._normalized(episode_counts)
        if self.baseline_counts is None:
            self.baseline_counts = normalized_counts
            return 0
        projected = self.counts_since_transition(normalized_counts)
        return projected.total if projected is not None else 0

    def counts_since_transition(
        self,
        episode_counts: EpisodeCounts,
    ) -> EpisodeCounts | None:
        """Return terminated and truncated counts after the full transition."""

        if self.baseline_counts is None:
            return None
        current = self._normalized(episode_counts)
        if (
            current.terminated < self.baseline_counts.terminated
            or current.truncated < self.baseline_counts.truncated
        ):
            self.baseline_counts = current
        terminated = max(0, current.terminated - self.baseline_counts.terminated)
        truncated = max(0, current.truncated - self.baseline_counts.truncated)
        return EpisodeCounts(
            terminated=terminated,
            truncated=truncated,
            total=terminated + truncated,
        )

    def reset(self) -> None:
        """Forget the previous scenario's full-curriculum boundary."""

        self.baseline_counts = None

    @staticmethod
    def _normalized(episode_counts: EpisodeCounts) -> EpisodeCounts:
        """Clamp adapter-owned counters before storing a transition baseline."""

        terminated = max(0, int(episode_counts.terminated))
        truncated = max(0, int(episode_counts.truncated))
        return EpisodeCounts(terminated, truncated, terminated + truncated)


def curriculum_target_state(
    *,
    state_present: bool,
    state_name: ConditionValue,
    target_type: ConditionValue,
    target_value: ConditionValue,
    curriculum_index: int | None,
    cumulative_minimum_terminated_episodes: int | None,
) -> CurriculumTargetState:
    """Normalize adapter values while sharing the canonical full-state rule."""

    state: CurriculumState | None = None
    if state_present:
        state = {
            "name": state_name,
            "target_type": target_type,
            "target_value": target_value,
        }

    scenario_state_name = state_name if state_present else "full"
    diagnostic_default = "curriculum" if state_present else "full"
    diagnostic_state_name = str(state_name or diagnostic_default)
    return CurriculumTargetState(
        scenario_state_name=scenario_state_name,
        diagnostic_state_name=diagnostic_state_name,
        target_type=target_type,
        target_value=target_value,
        curriculum_index=curriculum_index,
        cumulative_minimum_terminated_episodes=(cumulative_minimum_terminated_episodes),
        full=is_full_curriculum_state(state),
    )


def evaluate_scenario_readiness(
    *,
    window_episode_count: int,
    episode_counts: EpisodeCounts,
    policy: ScenarioSwitchPolicy,
    curriculum: CurriculumTargetState,
    full_curriculum_episode_count: int = 0,
    window_metrics: NormalizedWindowMetrics | None = None,
    preferred_override_window: PreferredOverrideWindow | None = None,
) -> SwitchDecision:
    """Evaluate scenario readiness without owning framework statistics."""

    conditions = [
        _lower_bound(
            "scenario_ready.window_samples",
            window_episode_count,
            "scenario_switch.lookback_window_size",
            policy.lookback_window_size,
        ),
    ]
    if policy.minimum_episodes > 0:
        conditions.append(
            _lower_bound(
                "scenario_ready.observed_episodes",
                episode_counts.total,
                "scenario_switch.min_episodes_for_switch_per_scenario",
                policy.minimum_episodes,
            )
        )
    conditions.extend(
        (
            _lower_bound(
                "scenario_ready.terminated_episodes",
                episode_counts.terminated,
                "scenario_switch.min_terminated_episodes_for_switch_per_scenario",
                policy.minimum_terminated_episodes,
            ),
            DecisionCondition(
                lhs_name="scenario_ready.curriculum_state",
                lhs_val=curriculum.scenario_state_name,
                operator="==",
                rhs_name="full",
                rhs_val="full",
                result=curriculum.full,
            ),
        )
    )
    if policy.minimum_full_curriculum_episodes > 0:
        conditions.append(
            _lower_bound(
                "scenario_ready.full_curriculum_episodes",
                full_curriculum_episode_count,
                "scenario_switch.min_full_curriculum_episodes",
                policy.minimum_full_curriculum_episodes,
            )
        )

    readiness_policy_present = any(
        threshold is not None
        for threshold in (
            policy.termination_ratio,
            policy.red_light_overdrive_rate,
            policy.time_limit_truncation_rate,
        )
    )
    if readiness_policy_present and window_metrics is None:
        return SwitchDecision(
            conditions=tuple(conditions),
            decision=False,
            evaluated=False,
        )
    conditions.extend(
        _scenario_readiness_conditions(
            policy,
            window_metrics,
            preferred_override_window,
        )
    )
    return _switch_decision(tuple(conditions))


def _scenario_readiness_conditions(
    policy: ScenarioSwitchPolicy,
    window_metrics: NormalizedWindowMetrics | None,
    preferred_override_window: PreferredOverrideWindow | None,
) -> list[DecisionCondition]:
    """Project configured recent-window bounds for scenario promotion."""

    conditions: list[DecisionCondition] = []
    if window_metrics is not None:
        if policy.termination_ratio is not None:
            _append_lower_bounds_with_preferred_override(
                conditions,
                lhs_name="scenario_ready.window.termination_ratio",
                lhs_value=window_metrics.termination_ratio,
                rhs_name="scenario_termination_ratio_threshold",
                bounds=policy.termination_ratio,
                override_window=preferred_override_window,
                override_value=_window_metric(
                    preferred_override_window,
                    "termination_ratio",
                ),
                override_name="scenario_ready.override.termination_ratio",
            )
        if policy.red_light_overdrive_rate is not None:
            _append_optional_upper_bounds(
                conditions,
                lhs_name="scenario_ready.window.red_light_overdrive_rate",
                lhs_value=window_metrics.red_light_overdrive_rate,
                rhs_name="scenario_red_light_overdrive_rate_threshold",
                bounds=policy.red_light_overdrive_rate,
                override_window=preferred_override_window,
                override_value=_window_metric(
                    preferred_override_window,
                    "red_light_overdrive_rate",
                ),
                override_name="scenario_ready.override.red_light_overdrive_rate",
            )
        if policy.time_limit_truncation_rate is not None:
            _append_optional_upper_bounds(
                conditions,
                lhs_name="scenario_ready.window.time_limit_truncation_rate",
                lhs_value=window_metrics.time_limit_truncation_rate,
                rhs_name="scenario_time_limit_truncation_rate_threshold",
                bounds=policy.time_limit_truncation_rate,
                override_window=preferred_override_window,
                override_value=_window_metric(
                    preferred_override_window,
                    "time_limit_truncation_rate",
                ),
                override_name="scenario_ready.override.time_limit_truncation_rate",
            )
    return conditions


def evaluate_training_stability(
    *,
    rewards: tuple[float, ...],
    policy: TrainingStabilityPolicy,
) -> SwitchDecision:
    """Require a positive recent reward mean without a variance gate."""

    if not rewards or not all(isfinite(reward) for reward in rewards):
        return SwitchDecision(conditions=(), decision=False, evaluated=False)
    mean_reward = sum(rewards) / len(rewards)
    conditions = (
        DecisionCondition(
            "metrics_ready.reward.mean",
            mean_reward,
            ">",
            "_episode_reward_stability.target_mean_by_lookback_window",
            policy.minimum_mean_reward,
            mean_reward > policy.minimum_mean_reward,
        ),
    )
    return _switch_decision(conditions)


def evaluate_curriculum_readiness(
    *,
    overall_episode_counts: EpisodeCounts,
    window_metrics: NormalizedWindowMetrics | None,
    policy: CurriculumSwitchPolicy,
    curriculum: CurriculumTargetState,
    preferred_override_window: PreferredOverrideWindow | None = None,
) -> SwitchDecision:
    """Evaluate one curriculum transition, preserving empty-window semantics."""

    if window_metrics is None:
        return SwitchDecision(conditions=(), decision=False, evaluated=False)

    terminated_episodes = overall_episode_counts.terminated
    if terminated_episodes == 0:
        terminated_episodes = window_metrics.episodes.terminated

    conditions = [
        DecisionCondition(
            lhs_name="curriculum_ready.state",
            lhs_val=curriculum.diagnostic_state_name,
            operator="active",
            rhs_name="curriculum_level",
            rhs_val=curriculum.curriculum_level,
            result=True,
        ),
        DecisionCondition(
            lhs_name="curriculum_ready.target_type",
            lhs_val=curriculum.target_type,
            operator="target",
            rhs_name="target_value",
            rhs_val=curriculum.target_value,
            result=True,
        ),
        _lower_bound(
            "curriculum_ready.terminated_episodes",
            terminated_episodes,
            "required_curriculum_terminated_episodes",
            curriculum.required_terminated_episodes(policy.minimum_terminated_episodes),
        ),
    ]
    _append_lower_bounds_with_preferred_override(
        conditions,
        lhs_name="curriculum_ready.window.termination_ratio",
        lhs_value=window_metrics.termination_ratio,
        rhs_name="curriculum_termination_ratio_threshold",
        bounds=policy.termination_ratio,
        override_window=preferred_override_window,
        override_value=_window_metric(
            preferred_override_window,
            "termination_ratio",
        ),
        override_name="curriculum_ready.override.termination_ratio",
    )
    _append_optional_upper_bounds(
        conditions,
        lhs_name="curriculum_ready.window.red_light_overdrive_rate",
        lhs_value=window_metrics.red_light_overdrive_rate,
        rhs_name="curriculum_red_light_overdrive_rate_threshold",
        bounds=policy.red_light_overdrive_rate,
        override_window=preferred_override_window,
        override_value=_window_metric(
            preferred_override_window,
            "red_light_overdrive_rate",
        ),
        override_name="curriculum_ready.override.red_light_overdrive_rate",
    )
    _append_optional_upper_bounds(
        conditions,
        lhs_name="curriculum_ready.window.time_limit_truncation_rate",
        lhs_value=window_metrics.time_limit_truncation_rate,
        rhs_name="curriculum_time_limit_truncation_rate_threshold",
        bounds=policy.time_limit_truncation_rate,
        override_window=preferred_override_window,
        override_value=_window_metric(
            preferred_override_window,
            "time_limit_truncation_rate",
        ),
        override_name="curriculum_ready.override.time_limit_truncation_rate",
    )
    return _switch_decision(tuple(conditions))


def _append_optional_upper_bounds(
    conditions: list[DecisionCondition],
    *,
    lhs_name: str,
    lhs_value: float,
    rhs_name: str,
    bounds: OptionalUpperBoundPolicy,
    override_window: PreferredOverrideWindow | None = None,
    override_value: float | None = None,
    override_name: str = "preferred_override_window",
) -> None:
    """Append the optional upper bounds owned by this rolling stats decision core module."""
    if bounds.acceptable is not None:
        conditions.append(
            _upper_bound(
                lhs_name,
                lhs_value,
                f"{rhs_name}.acceptable_range.upper",
                bounds.acceptable,
            )
        )
    if bounds.preferred is not None:
        conditions.append(
            _preferred_override_gate(
                lhs_name=lhs_name,
                lhs_value=lhs_value,
                preferred_bound=bounds.preferred,
                acceptable_bound=bounds.acceptable,
                rhs_name=rhs_name,
                override_window=override_window,
                override_value=override_value,
                override_name=override_name,
                lower_bound=False,
            )
        )


def _append_lower_bounds_with_preferred_override(
    conditions: list[DecisionCondition],
    *,
    lhs_name: str,
    lhs_value: float,
    rhs_name: str,
    bounds: LowerBoundPolicy,
    override_window: PreferredOverrideWindow | None,
    override_value: float | None,
    override_name: str,
) -> None:
    """Keep acceptable hard while permitting recent evidence for preferred."""

    conditions.append(
        _lower_bound(
            lhs_name,
            lhs_value,
            f"{rhs_name}.acceptable_range.lower",
            bounds.acceptable,
        )
    )
    conditions.append(
        _preferred_override_gate(
            lhs_name=lhs_name,
            lhs_value=lhs_value,
            preferred_bound=bounds.preferred,
            acceptable_bound=bounds.acceptable,
            rhs_name=rhs_name,
            override_window=override_window,
            override_value=override_value,
            override_name=override_name,
            lower_bound=True,
        )
    )


def _preferred_override_gate(
    *,
    lhs_name: str,
    lhs_value: float,
    preferred_bound: float,
    acceptable_bound: float | None,
    rhs_name: str,
    override_window: PreferredOverrideWindow | None,
    override_value: float | None,
    override_name: str,
    lower_bound: bool,
) -> DecisionCondition:
    """Pass preferred directly or through a full, acceptable recent window."""

    direct_preferred = lhs_value >= preferred_bound if lower_bound else lhs_value <= preferred_bound
    recent_acceptable = False
    if acceptable_bound is not None and override_window is not None and override_value is not None:
        recent_acceptable = (
            override_value >= acceptable_bound
            if lower_bound
            else override_value <= acceptable_bound
        )
    observed = override_window.observed_episodes if override_window else 0
    required = override_window.required_episodes if override_window else 0
    window_full = override_window.full if override_window is not None else False
    override_acceptable = recent_acceptable and window_full
    override_text = (
        f"{override_name}={override_value}; episodes={observed}/{required}"
        if override_window is not None
        else f"{override_name}=disabled"
    )
    direction: ConditionOperator = ">=" if lower_bound else "<="
    checks = [
        DecisionConditionCheck(
            label="base",
            lhs_val=lhs_value,
            operator=direction,
            rhs_val=preferred_bound,
            result=direct_preferred,
        )
    ]
    if override_window is not None:
        checks.extend(
            (
                DecisionConditionCheck(
                    label="recent",
                    lhs_val=override_value,
                    operator=direction,
                    rhs_val=acceptable_bound,
                    result=recent_acceptable,
                ),
                DecisionConditionCheck(
                    label="window",
                    lhs_val=observed,
                    operator=">=",
                    rhs_val=required,
                    result=window_full,
                ),
            )
        )
    return DecisionCondition(
        lhs_name=f"{lhs_name}.preferred_gate",
        lhs_val=f"base={lhs_value}; {override_text}",
        operator="satisfies",
        rhs_name=f"{rhs_name}.preferred_or_full_override_acceptable",
        rhs_val=(f"base {direction} {preferred_bound} OR override {direction} {acceptable_bound}"),
        result=direct_preferred or override_acceptable,
        checks=tuple(checks),
    )


def _window_metric(
    window: PreferredOverrideWindow | None,
    metric_name: Literal[
        "termination_ratio",
        "red_light_overdrive_rate",
        "time_limit_truncation_rate",
    ],
) -> float | None:
    """Read one normalized metric without weakening an absent window."""

    if window is None or window.metrics is None:
        return None
    return float(getattr(window.metrics, metric_name))


def _lower_bound(
    lhs_name: str,
    lhs_value: int | float,
    rhs_name: str,
    rhs_value: int | float,
) -> DecisionCondition:
    """Return the lower bound value owned by this rolling stats decision core module."""
    return DecisionCondition(
        lhs_name=lhs_name,
        lhs_val=lhs_value,
        operator=">=",
        rhs_name=rhs_name,
        rhs_val=rhs_value,
        result=lhs_value >= rhs_value,
    )


def _upper_bound(
    lhs_name: str,
    lhs_value: int | float,
    rhs_name: str,
    rhs_value: int | float,
) -> DecisionCondition:
    """Return the upper bound value owned by this rolling stats decision core module."""
    return DecisionCondition(
        lhs_name=lhs_name,
        lhs_val=lhs_value,
        operator="<=",
        rhs_name=rhs_name,
        rhs_val=rhs_value,
        result=lhs_value <= rhs_value,
    )


def _switch_decision(
    conditions: tuple[DecisionCondition, ...],
) -> SwitchDecision:
    """Return the switch decision value owned by this rolling stats decision core module."""
    return SwitchDecision(
        conditions=conditions,
        decision=all(condition.result for condition in conditions),
    )
