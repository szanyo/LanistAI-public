# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed configuration normalization for rolling-statistics adapters."""

from __future__ import annotations

import numbers
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, SupportsFloat, SupportsInt, cast, overload

from lanista.reinforcengine.rl_environment.rolling_stats_decision_core import (
    ConditionValue,
    CurriculumSwitchPolicy,
    CurriculumTargetState,
    LowerBoundPolicy,
    OptionalUpperBoundPolicy,
    curriculum_target_state,
)

type NumericCast = type[int] | type[float]
type NumericValue = int | float
type NumericDefault = NumericValue | None
type _IntInput = str | bytes | bytearray | SupportsInt
type _FloatInput = str | bytes | bytearray | SupportsFloat


class _ConfigSource(Protocol):
    """Define the typed boundary for config source."""

    def get(self, key: str, default: object = None) -> object:
        """Return one configuration value or ``default``."""


def config_get(
    configuration: object | None,
    key: str,
    default: object = None,
) -> object:
    """Read mappings, wrapper ``get`` APIs, and attribute configurations."""

    if configuration is None:
        return default
    if hasattr(configuration, "get"):
        return cast(_ConfigSource, configuration).get(key, default)
    return cast(object, getattr(configuration, key, default))


def scenario_minimum_episode_floor(
    training_conditions: object,
    scenario_switch: object,
) -> object:
    """Resolve the optional scenario total-episode floor with legacy fallback."""

    legacy_floor = config_get(
        training_conditions,
        "min_episodes_for_switch_per_scenario",
        0,
    )
    return config_get(
        scenario_switch,
        "min_episodes_for_switch_per_scenario",
        legacy_floor,
    )


@overload
def coerce_number(
    value: object,
    default: int,
    cast_type: type[int],
) -> int:
    """Coerce the number owned by this rolling stats configuration module."""
    ...


@overload
def coerce_number(
    value: object,
    default: float,
    cast_type: type[float] = float,
) -> float:
    """Coerce the number owned by this rolling stats configuration module."""
    ...


@overload
def coerce_number(
    value: object,
    default: None,
    cast_type: type[int],
) -> int | None:
    """Coerce the number owned by this rolling stats configuration module."""
    ...


@overload
def coerce_number(
    value: object,
    default: None = None,
    cast_type: type[float] = float,
) -> float | None:
    """Coerce the number owned by this rolling stats configuration module."""
    ...


def coerce_number(
    value: object,
    default: NumericDefault = None,
    cast_type: NumericCast = float,
) -> NumericDefault:
    """Coerce a legacy scalar while retaining its configured fallback."""

    if value is None:
        return default
    if isinstance(value, numbers.Number) and not _is_bool(value):
        return _apply_numeric_cast(value, cast_type)
    try:
        return _apply_numeric_cast(value, cast_type)
    except (TypeError, ValueError):
        return default


def normalize_range(
    range_value: object | None,
    fallback_lower: float,
    fallback_upper: float,
) -> tuple[float, float]:
    """Normalize lower/upper objects and legacy two-element iterables."""

    lower = fallback_lower
    upper = fallback_upper
    if range_value is None:
        return lower, upper

    value_lower = config_get(range_value, "lower")
    value_upper = config_get(range_value, "upper")
    if value_lower is not None or value_upper is not None:
        lower = coerce_number(value_lower, lower, float)
        upper = coerce_number(value_upper, upper, float)
        return lower, upper

    try:
        values = list(cast(Iterable[object], range_value))
    except TypeError:
        return lower, upper
    if len(values) >= 2:
        lower = coerce_number(values[0], lower, float)
        upper = coerce_number(values[1], upper, float)
    return lower, upper


def threshold_lower_bounds(
    threshold: object | None,
    fallback: float = 0.0,
) -> tuple[float, float]:
    """Normalize scalar or acceptable/preferred lower-bound thresholds."""

    scalar = coerce_number(threshold, None, float)
    if scalar is not None:
        return scalar, scalar

    acceptable_range = config_get(threshold, "acceptable_range")
    preferred_range = config_get(threshold, "preferred_range")
    acceptable_lower = coerce_number(
        config_get(acceptable_range, "lower"),
        fallback,
        float,
    )
    preferred_lower = coerce_number(
        config_get(preferred_range, "lower"),
        acceptable_lower,
        float,
    )
    return acceptable_lower, preferred_lower


def threshold_upper_bounds(
    threshold: object | None,
    fallback: float = 1.0,
) -> tuple[float | None, float | None]:
    """Normalize optional scalar or acceptable/preferred upper bounds."""

    if threshold is None:
        return None, None
    scalar = coerce_number(threshold, None, float)
    if scalar is not None:
        return scalar, scalar

    acceptable_range = config_get(threshold, "acceptable_range")
    preferred_range = config_get(threshold, "preferred_range")
    acceptable_upper = coerce_number(
        config_get(acceptable_range, "upper"),
        fallback,
        float,
    )
    preferred_upper = coerce_number(
        config_get(preferred_range, "upper"),
        acceptable_upper,
        float,
    )
    return acceptable_upper, preferred_upper


@dataclass(frozen=True, slots=True)
class CurriculumConfiguration:
    """Adapter defaults consumed by one curriculum projection."""

    minimum_terminated_episodes: int
    termination_ratio_threshold: object
    red_light_overdrive_rate_threshold: object | None
    time_limit_truncation_rate_threshold: object | None
    preferred_override_window_size: int = 0


@dataclass(frozen=True, slots=True)
class CurriculumProjector:
    """Pure curriculum target and policy projection over legacy configs."""

    defaults: CurriculumConfiguration

    def switch_conditions(
        self,
        state_or_conditions: object | None,
    ) -> object | None:
        """Return the switch conditions value owned by this curriculum projector."""
        if state_or_conditions is None:
            return None
        transition_conditions = config_get(
            state_or_conditions,
            "transition_conditions",
        )
        if transition_conditions is not None:
            return transition_conditions
        return state_or_conditions

    def minimum_terminated_episodes(
        self,
        state_or_conditions: object | None,
    ) -> int:
        """Return the minimum terminated episodes value owned by this curriculum projector."""
        conditions = self.switch_conditions(state_or_conditions)
        raw_minimum = (
            config_get(conditions, "min_terminated_episodes")
            if conditions is not None
            else self.defaults.minimum_terminated_episodes
        )
        return coerce_number(
            raw_minimum,
            self.defaults.minimum_terminated_episodes,
            int,
        )

    def target(
        self,
        state_or_conditions: object | None,
        *,
        base_minimum: int | None = None,
        include_progress: bool = True,
    ) -> CurriculumTargetState:
        """Return the target value owned by this curriculum projector."""
        if base_minimum is None:
            base_minimum = self.defaults.minimum_terminated_episodes
        curriculum_index = self._curriculum_index(
            state_or_conditions,
            include_progress=include_progress,
        )
        cumulative_minimum = self._cumulative_minimum(
            state_or_conditions,
            base_minimum=base_minimum,
            include_progress=include_progress,
        )
        state_name = config_get(
            state_or_conditions,
            "name",
        ) or config_get(state_or_conditions, "state_name")
        return curriculum_target_state(
            state_present=state_or_conditions is not None,
            state_name=_condition_value(state_name),
            target_type=_condition_value(
                config_get(state_or_conditions, "target_type", "full")
            ),
            target_value=_condition_value(
                config_get(state_or_conditions, "target_value", "full")
            ),
            curriculum_index=curriculum_index,
            cumulative_minimum_terminated_episodes=cumulative_minimum,
        )

    def termination_ratio_bounds(
        self,
        state_or_conditions: object | None,
    ) -> tuple[float, float]:
        """Return the termination ratio bounds value owned by this curriculum projector."""
        conditions = self.switch_conditions(state_or_conditions)
        threshold = config_get(
            conditions,
            "termination_ratio_threshold",
            config_get(conditions, "min_termination_rate"),
        )
        if threshold is None:
            threshold = self.defaults.termination_ratio_threshold
        return threshold_lower_bounds(threshold)

    def upper_rate_bounds(
        self,
        state_or_conditions: object | None,
        key: str,
        default_threshold: object | None,
    ) -> tuple[float | None, float | None]:
        """Return the upper rate bounds value owned by this curriculum projector."""
        conditions = self.switch_conditions(state_or_conditions)
        threshold = config_get(conditions, key)
        if threshold is None:
            threshold = default_threshold
        return threshold_upper_bounds(threshold)

    def preferred_override_window_size(
        self,
        state_or_conditions: object | None,
    ) -> int:
        """Resolve the curriculum-level recent evidence size in episodes."""

        conditions = self.switch_conditions(state_or_conditions)
        raw_size = config_get(
            conditions,
            "preferred_override_window_size",
            self.defaults.preferred_override_window_size,
        )
        return max(0, coerce_number(
            raw_size,
            self.defaults.preferred_override_window_size,
            int,
        ))

    def policy(
        self,
        state_or_conditions: object | None,
    ) -> CurriculumSwitchPolicy:
        """Return the policy value owned by this curriculum projector."""
        minimum = self.minimum_terminated_episodes(state_or_conditions)
        acceptable_ratio, preferred_ratio = self.termination_ratio_bounds(
            state_or_conditions
        )
        red_acceptable, red_preferred = self.upper_rate_bounds(
            state_or_conditions,
            "red_light_overdrive_rate_threshold",
            self.defaults.red_light_overdrive_rate_threshold,
        )
        time_acceptable, time_preferred = self.upper_rate_bounds(
            state_or_conditions,
            "time_limit_truncation_rate_threshold",
            self.defaults.time_limit_truncation_rate_threshold,
        )
        return CurriculumSwitchPolicy(
            minimum_terminated_episodes=minimum,
            termination_ratio=LowerBoundPolicy(
                acceptable=acceptable_ratio,
                preferred=preferred_ratio,
            ),
            red_light_overdrive_rate=OptionalUpperBoundPolicy(
                acceptable=red_acceptable,
                preferred=red_preferred,
            ),
            time_limit_truncation_rate=OptionalUpperBoundPolicy(
                acceptable=time_acceptable,
                preferred=time_preferred,
            ),
            preferred_override_window_size=(
                self.preferred_override_window_size(state_or_conditions)
            ),
        )

    @staticmethod
    def _curriculum_index(
        state_or_conditions: object | None,
        *,
        include_progress: bool,
    ) -> int | None:
        """Return the curriculum index value owned by this curriculum projector."""
        if not include_progress:
            return None
        raw_index = config_get(state_or_conditions, "curriculum_index")
        if raw_index is None:
            return None
        return coerce_number(raw_index, 0, int)

    @staticmethod
    def _cumulative_minimum(
        state_or_conditions: object | None,
        *,
        base_minimum: int,
        include_progress: bool,
    ) -> int | None:
        """Return the cumulative minimum value owned by this curriculum projector."""
        if not include_progress:
            return None
        raw_cumulative = config_get(
            state_or_conditions,
            "cumulative_min_terminated_episodes",
        )
        if raw_cumulative is None:
            return None
        return coerce_number(raw_cumulative, base_minimum, int)


def _condition_value(value: object) -> ConditionValue:
    """Return the condition value value owned by this rolling stats configuration module."""
    return cast(ConditionValue, value)


def _apply_numeric_cast(
    value: object,
    cast_type: NumericCast,
) -> NumericValue:
    """Apply the numeric cast transition owned by this rolling stats configuration module."""
    if cast_type is int:
        return int(cast(_IntInput, value))
    return float(cast(_FloatInput, value))


def _is_bool(value: object) -> bool:
    """Return whether bool applies to this rolling stats configuration module."""
    return isinstance(value, bool)
