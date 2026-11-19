# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project metric-owned curriculum and scenario milestones onto UI progress."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from lanista.reinforcengine.rl_environment.rolling_stats_configuration import (
    coerce_number,
    config_get,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    decorate_curriculum_states,
    ensure_full_curriculum_state,
    is_full_curriculum_state,
)

_PHASE_READINESS_CEILING = 0.99
_PHASE_ACTIVITY_RESERVE = 0.009
_PHASE_PROGRESS_CEILING = (
    _PHASE_READINESS_CEILING + _PHASE_ACTIVITY_RESERVE
)


@dataclass(frozen=True, slots=True)
class TrainingProgressEvidence:
    """Monotonic active-scenario counters used for fractional progress."""

    terminated_episodes: int = 0
    total_episodes: int = 0
    window_episodes: int = 0
    full_curriculum_episodes: int = 0


@dataclass(frozen=True, slots=True)
class TrainingProgressProjection:
    """One UI-ready percentage and its metric evidence summary."""

    percentage: float
    completed_milestones: float
    total_milestones: int
    message: str


@dataclass(frozen=True, slots=True)
class TrainingProgressPlan:
    """Treat the initial curriculum phase as one scenario-sized milestone."""

    total_scenarios: int
    curriculum_thresholds: tuple[int, ...]
    lookback_window_size: int
    minimum_scenario_episodes: int
    minimum_scenario_terminated_episodes: int
    minimum_full_curriculum_episodes: int

    @classmethod
    def from_training_conditions(
        cls,
        training_conditions: object,
        *,
        total_scenarios: int,
    ) -> TrainingProgressPlan:
        """Normalize the milestone counts shared by SB3 and Ray."""

        curriculum_switch = config_get(
            training_conditions,
            "curriculum_switch",
        )
        raw_states = config_get(curriculum_switch, "states", [])
        states = raw_states if isinstance(raw_states, list) else []
        base_threshold = coerce_number(
            config_get(curriculum_switch, "min_terminated_episodes", 0),
            0,
            int,
        )
        decorated_states = decorate_curriculum_states(
            ensure_full_curriculum_state(states),
            base_min_terminated_episodes=base_threshold,
        )
        curriculum_thresholds = tuple(
            max(
                0,
                coerce_number(
                    state.get("cumulative_min_terminated_episodes"),
                    0,
                    int,
                ),
            )
            for state in decorated_states
            if not is_full_curriculum_state(state)
        )

        scenario_switch = config_get(
            training_conditions,
            "scenario_switch",
        )
        return cls(
            total_scenarios=max(0, int(total_scenarios)),
            curriculum_thresholds=curriculum_thresholds,
            lookback_window_size=_nonnegative_int(
                config_get(scenario_switch, "lookback_window_size", 0)
            ),
            minimum_scenario_episodes=_nonnegative_int(
                config_get(
                    scenario_switch,
                    "min_episodes_for_switch_per_scenario",
                    0,
                )
            ),
            minimum_scenario_terminated_episodes=_nonnegative_int(
                config_get(
                    scenario_switch,
                    "min_terminated_episodes_for_switch_per_scenario",
                    0,
                )
            ),
            minimum_full_curriculum_episodes=_nonnegative_int(
                config_get(
                    scenario_switch,
                    "min_full_curriculum_episodes",
                    0,
                )
            ),
        )

    @property
    def total_milestones(self) -> int:
        """Return one initial curriculum phase plus all scenario phases."""

        return self.total_scenarios + int(bool(self.curriculum_thresholds))

    @property
    def curriculum_transition_weight(self) -> float:
        """Return one restricted curriculum transition's phase share."""

        if not self.curriculum_thresholds:
            return 0.0
        return 1.0 / len(self.curriculum_thresholds)

    def project(
        self,
        *,
        scenario_index: int,
        curriculum_state: object | None,
        evidence: TrainingProgressEvidence,
        finished: bool,
    ) -> TrainingProgressProjection:
        """Project current metric evidence without claiming readiness itself."""

        total_milestones = self.total_milestones
        if finished or total_milestones == 0:
            return TrainingProgressProjection(
                percentage=100.0,
                completed_milestones=float(total_milestones),
                total_milestones=total_milestones,
                message=(
                    "Training milestones completed "
                    f"({total_milestones}/{total_milestones})"
                ),
            )

        completed_scenarios = min(
            max(0, int(scenario_index)),
            self.total_scenarios,
        )
        if self._is_full(curriculum_state) or not self.curriculum_thresholds:
            fraction = self._scenario_fraction(
                evidence,
                activity_observations=(
                    evidence.full_curriculum_episodes
                    if self.curriculum_thresholds and completed_scenarios == 0
                    else evidence.total_episodes
                ),
            )
            completed = (
                int(bool(self.curriculum_thresholds))
                + completed_scenarios
                + fraction
            )
            detail = self._scenario_detail(evidence)
            curriculum_label = "full"
        else:
            curriculum_index = min(
                max(0, self._curriculum_index(curriculum_state)),
                max(0, len(self.curriculum_thresholds) - 1),
            )
            fraction = self._curriculum_fraction(
                curriculum_index,
                evidence,
            )
            completed = self.curriculum_transition_weight * (
                curriculum_index + fraction
            )
            target = self.curriculum_thresholds[curriculum_index]
            detail = (
                f"curriculum terminated {evidence.terminated_episodes}/{target}"
            )
            curriculum_label = (
                f"{curriculum_index + 1}/"
                f"{len(self.curriculum_thresholds) + 1}"
            )

        completed = min(max(0.0, completed), float(total_milestones))
        percentage = 100.0 * completed / total_milestones
        scenario_label = (
            f"{min(completed_scenarios + 1, self.total_scenarios)}/"
            f"{self.total_scenarios}"
            if self.total_scenarios
            else "0/0"
        )
        return TrainingProgressProjection(
            percentage=percentage,
            completed_milestones=completed,
            total_milestones=total_milestones,
            message=(
                f"Training {percentage:.1f}% — scenario {scenario_label}, "
                f"curriculum {curriculum_label}; {detail}; milestones "
                f"{completed:.2f}/{total_milestones}"
            ),
        )

    def _curriculum_fraction(
        self,
        curriculum_index: int,
        evidence: TrainingProgressEvidence,
    ) -> float:
        """Return readiness plus bounded per-episode curriculum progress."""

        previous = (
            self.curriculum_thresholds[curriculum_index - 1]
            if curriculum_index > 0
            else 0
        )
        target = self.curriculum_thresholds[curriculum_index]
        if target <= previous:
            return 0.0
        readiness_progress = (
            max(0, evidence.terminated_episodes) - previous
        ) / (
            target - previous
        )
        readiness_fraction = min(
            _PHASE_READINESS_CEILING,
            max(0.0, readiness_progress),
        )
        return _with_activity_progress(
            readiness_fraction,
            observations=max(0, evidence.total_episodes - previous),
            expected_observations=target - previous,
        )

    def _scenario_fraction(
        self,
        evidence: TrainingProgressEvidence,
        *,
        activity_observations: int,
    ) -> float:
        """Use only monotonic count gates; volatile quality gates retain authority."""

        ratios = [
            _quota_ratio(evidence.window_episodes, self.lookback_window_size),
            _quota_ratio(
                evidence.total_episodes,
                self.minimum_scenario_episodes,
            ),
            _quota_ratio(
                evidence.terminated_episodes,
                self.minimum_scenario_terminated_episodes,
            ),
            _quota_ratio(
                evidence.full_curriculum_episodes,
                self.minimum_full_curriculum_episodes,
            ),
        ]
        active_ratios = [ratio for ratio in ratios if ratio is not None]
        if not active_ratios:
            readiness_fraction = 0.0
        else:
            readiness_fraction = min(
                _PHASE_READINESS_CEILING,
                min(active_ratios),
            )
        expected_observations = max(
            1,
            self.lookback_window_size,
            self.minimum_scenario_episodes,
            self.minimum_scenario_terminated_episodes,
            self.minimum_full_curriculum_episodes,
        )
        return _with_activity_progress(
            readiness_fraction,
            observations=activity_observations,
            expected_observations=expected_observations,
        )

    def _scenario_detail(self, evidence: TrainingProgressEvidence) -> str:
        """Format the count evidence without implying volatile-gate readiness."""

        details: list[str] = []
        for label, current, target in (
            (
                "scenario_ready.window",
                evidence.window_episodes,
                self.lookback_window_size,
            ),
            ("episodes", evidence.total_episodes, self.minimum_scenario_episodes),
            (
                "terminated",
                evidence.terminated_episodes,
                self.minimum_scenario_terminated_episodes,
            ),
            (
                "full episodes",
                evidence.full_curriculum_episodes,
                self.minimum_full_curriculum_episodes,
            ),
        ):
            if target > 0:
                details.append(f"{label} {current}/{target}")
        return ", ".join(details) or "waiting for metric readiness"

    @staticmethod
    def _is_full(curriculum_state: object | None) -> bool:
        """Apply the canonical full-state rule to mapping-like state values."""

        if curriculum_state is None:
            return True
        if not hasattr(curriculum_state, "get"):
            return False
        return bool(
            is_full_curriculum_state(
                cast(dict[str, Any], curriculum_state)
            )
        )

    @staticmethod
    def _curriculum_index(curriculum_state: object | None) -> int:
        """Read a decorated curriculum index with a legacy-safe fallback."""

        if curriculum_state is None or not hasattr(curriculum_state, "get"):
            return 0
        return _nonnegative_int(
            cast(dict[str, Any], curriculum_state).get("curriculum_index", 0)
        )


def _nonnegative_int(value: object) -> int:
    """Normalize one configuration or state counter."""

    return max(0, int(coerce_number(value, 0, int)))


def _quota_ratio(current: int, target: int) -> float | None:
    """Return a clamped ratio or omit a disabled quota."""

    if target <= 0:
        return None
    return min(1.0, max(0, current) / target)


def _with_activity_progress(
    readiness_fraction: float,
    *,
    observations: int,
    expected_observations: int,
) -> float:
    """Keep progress moving without allowing activity to complete a phase."""

    observed = max(0, int(observations))
    scale = max(1, int(expected_observations))
    activity_fraction = observed / (observed + scale)
    return min(
        _PHASE_PROGRESS_CEILING,
        readiness_fraction + _PHASE_ACTIVITY_RESERVE * activity_fraction,
    )
