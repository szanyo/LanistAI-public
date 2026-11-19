# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define runtime conditions for rolling stats."""

from collections.abc import Iterable
from dataclasses import dataclass

from lanista.foundation.logging import LoggerCondition, LoggerConditionCheck
from lanista.reinforcengine.rl_environment.rolling_stats_decision_core import (
    ConditionOperator,
    ConditionValue,
    DecisionCondition,
)


@dataclass(frozen=True, slots=True)
class ConditionDefinition(DecisionCondition):
    """Adapter-owned condition with its historic logger projection API."""

    def build(self) -> LoggerCondition:
        """Build the configured result owned by this condition definition."""
        return _build_logger_condition(self)


def condition(
    lhs_name: str,
    lhs_val: ConditionValue,
    operator: ConditionOperator,
    rhs_name: str,
    rhs_val: ConditionValue,
    result: bool,
) -> ConditionDefinition:
    """Return the condition value owned by this rolling stats conditions module."""
    return ConditionDefinition(
        lhs_name=lhs_name,
        lhs_val=lhs_val,
        operator=operator,
        rhs_name=rhs_name,
        rhs_val=rhs_val,
        result=result,
    )


def lower_bound_condition(
    lhs_name: str,
    lhs_val: int | float,
    rhs_name: str,
    rhs_val: int | float,
) -> ConditionDefinition:
    """Return the lower bound condition value owned by this rolling stats conditions module."""
    return condition(lhs_name, lhs_val, ">=", rhs_name, rhs_val, lhs_val >= rhs_val)


def upper_bound_condition(
    lhs_name: str,
    lhs_val: int | float,
    rhs_name: str,
    rhs_val: int | float,
) -> ConditionDefinition:
    """Return the upper bound condition value owned by this rolling stats conditions module."""
    return condition(lhs_name, lhs_val, "<=", rhs_name, rhs_val, lhs_val <= rhs_val)


def bounded_conditions(
    lhs_name: str,
    lhs_val: int | float,
    rhs_name: str,
    *,
    lower: int | float | None = None,
    upper: int | float | None = None,
) -> list[ConditionDefinition]:
    """Return the bounded conditions value owned by this rolling stats conditions module."""
    definitions = []
    if lower is not None:
        definitions.append(lower_bound_condition(lhs_name, lhs_val, f"{rhs_name}.lower", lower))
    if upper is not None:
        definitions.append(upper_bound_condition(lhs_name, lhs_val, f"{rhs_name}.upper", upper))
    return definitions


def build_conditions(
    definitions: Iterable[DecisionCondition],
) -> list[LoggerCondition]:
    """Build the conditions required by this rolling stats conditions module."""
    return [_build_logger_condition(definition) for definition in definitions]


def _build_logger_condition(definition: DecisionCondition) -> LoggerCondition:
    """Project one core condition through the single logger representation."""

    return LoggerCondition(
        lhs_name=definition.lhs_name,
        lhs_val=definition.lhs_val,
        operator=definition.operator,
        rhs_name=definition.rhs_name,
        rhs_val=definition.rhs_val,
        result=bool(definition.result),
        checks=tuple(
            LoggerConditionCheck(
                label=check.label,
                lhs_val=check.lhs_val,
                operator=check.operator,
                rhs_val=check.rhs_val,
                result=check.result,
            )
            for check in definition.checks
        ),
    )


def conditions_pass(conditions: Iterable[LoggerCondition]) -> bool:
    """Return the conditions pass value owned by this rolling stats conditions module."""
    return all(condition.result for condition in conditions)
