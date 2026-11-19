# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Pure curriculum transitions shared by local and distributed adapters."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Optional

CurriculumState = dict[str, Any]


def ensure_full_curriculum_state(states: list[CurriculumState]) -> list[CurriculumState]:
    """Append an explicit unrestricted state when a curriculum omits one."""

    if not states:
        return states

    last_state = states[-1]
    is_full = (
        last_state.get("target_type") in (None, "full")
        and last_state.get("target_value") in (None, "full")
    )
    if is_full:
        return states

    return states + [
        {
            "name": "full",
            "target_type": None,
            "target_value": None,
        }
    ]


def is_full_curriculum_state(state: Optional[CurriculumState]) -> bool:
    """Return whether a state permits unrestricted scenario operation."""

    if state is None:
        return True

    state_name = state.get("name") or state.get("state_name")
    target_type = state.get("target_type")
    target_value = state.get("target_value")
    return (
        state_name == "full"
        or target_type in (None, "full")
        or target_value in (None, "full")
    )


def decorate_curriculum_states(
    states: list[CurriculumState],
    base_min_terminated_episodes: int = 0,
) -> list[CurriculumState]:
    """Copy states and add deterministic cumulative transition thresholds."""

    try:
        normalized_threshold = int(base_min_terminated_episodes)
    except (TypeError, ValueError):
        normalized_threshold = 0

    decorated: list[CurriculumState] = []
    curriculum_index = 0
    cumulative_threshold = 0
    for state in states or []:
        copied_state = deepcopy(state)
        if is_full_curriculum_state(copied_state):
            copied_state["curriculum_index"] = curriculum_index
            copied_state["cumulative_min_terminated_episodes"] = max(
                cumulative_threshold,
                normalized_threshold * curriculum_index,
            )
            copied_state["horizon_is_full"] = True
            decorated.append(copied_state)
            continue

        copied_state["curriculum_index"] = curriculum_index
        configured_threshold = copied_state.get(
            "cumulative_min_terminated_episodes"
        )
        try:
            configured_threshold = int(configured_threshold)
        except (TypeError, ValueError):
            configured_threshold = 0
        cumulative_threshold = max(
            cumulative_threshold,
            configured_threshold,
            normalized_threshold * (curriculum_index + 1),
        )
        copied_state["cumulative_min_terminated_episodes"] = (
            cumulative_threshold
        )
        copied_state["horizon_is_full"] = False
        curriculum_index += 1
        decorated.append(copied_state)
    return decorated


def published_curriculum_state(
    state: Optional[CurriculumState],
) -> CurriculumState:
    """Convert an internal state into the framework-neutral public payload."""

    if state is None:
        return {
            "target_type": None,
            "target_value": None,
            "state_name": "full",
            "horizon_is_full": True,
        }
    return {
        "target_type": state.get("target_type"),
        "target_value": state.get("target_value"),
        "state_name": state.get("name") or state.get("state_name"),
        "curriculum_index": state.get("curriculum_index"),
        "cumulative_min_terminated_episodes": state.get(
            "cumulative_min_terminated_episodes"
        ),
        "horizon_is_full": is_full_curriculum_state(state),
    }


def curriculum_state_label(state: Optional[CurriculumState]) -> str:
    """Format one state for human-facing diagnostic logs."""

    published = published_curriculum_state(state)
    return (
        f"{published.get('state_name') or 'full'} "
        f"(target_type={published.get('target_type')}, "
        f"target_value={published.get('target_value')})"
    )


@dataclass(frozen=True, slots=True)
class CurriculumDefinition:
    """Normalized curriculum inputs consumed by framework adapters."""

    states: list[CurriculumState]
    base_min_terminated_episodes: int


def resolve_curriculum_definition(
    training_conditions: object,
    *,
    state_completer: Callable[
        [list[CurriculumState]],
        list[CurriculumState],
    ] = ensure_full_curriculum_state,
) -> CurriculumDefinition:
    """Extract one typed definition from a nested training configuration."""

    curriculum_switch = getattr(training_conditions, "curriculum_switch", None)
    raw_states = (
        getattr(curriculum_switch, "states", [])
        if curriculum_switch is not None
        else []
    )
    states = raw_states if isinstance(raw_states, list) else []
    raw_threshold = (
        getattr(curriculum_switch, "min_terminated_episodes", 0)
        if curriculum_switch is not None
        else 0
    )
    try:
        threshold = int(raw_threshold)
    except (TypeError, ValueError):
        threshold = 0
    return CurriculumDefinition(
        states=state_completer(states),
        base_min_terminated_episodes=threshold,
    )


@dataclass(frozen=True, slots=True)
class CurriculumTransition:
    """One deterministic state transition, independent of its publisher."""

    previous_state: Optional[CurriculumState]
    current_state: Optional[CurriculumState]
    changed: bool


@dataclass(frozen=True, slots=True)
class CurriculumRestoreResult:
    """Restore outcome including an unknown state that must still be published."""

    active_state: Optional[CurriculumState]
    publication_state: Optional[CurriculumState]
    recognized: bool


class CurriculumStateMachine:
    """Own curriculum position while framework adapters own state transport."""

    def __init__(
        self,
        states: list[CurriculumState],
        *,
        base_min_terminated_episodes: int = 0,
    ) -> None:
        """Initialize the curriculum state machine and its owned state."""
        self._states = decorate_curriculum_states(
            states or [],
            base_min_terminated_episodes,
        )
        self._state_index = 0
        self._active = bool(self._states)

    def current_state(self) -> Optional[CurriculumState]:
        """Return the current decorated state, or unrestricted completion."""

        if not self._active or self._state_index >= len(self._states):
            return None
        return self._states[self._state_index]

    def reset(self) -> Optional[CurriculumState]:
        """Return to the first configured state."""

        self._state_index = 0
        self._active = bool(self._states)
        return self.current_state()

    def is_unlocked(self) -> bool:
        """Return whether the current state permits a scenario transition."""

        return is_full_curriculum_state(self.current_state())

    def advance(self, ready: bool) -> CurriculumTransition:
        """Advance at most one state when rolling statistics permit it."""

        previous_state = self.current_state()
        if (
            previous_state is not None
            and ready
            and not is_full_curriculum_state(previous_state)
        ):
            self._state_index += 1
        current_state = self.current_state()
        return CurriculumTransition(
            previous_state=previous_state,
            current_state=current_state,
            changed=previous_state is not current_state,
        )

    def restore(
        self,
        state: Optional[CurriculumState],
    ) -> CurriculumRestoreResult:
        """Restore a checkpoint state without coupling to its transport."""

        if state is None:
            current_state = self.current_state()
            return CurriculumRestoreResult(current_state, current_state, True)

        if is_full_curriculum_state(state):
            for index, candidate in enumerate(self._states):
                if is_full_curriculum_state(candidate):
                    self._state_index = index
                    self._active = True
                    return CurriculumRestoreResult(
                        candidate,
                        candidate,
                        True,
                    )
            self._state_index = len(self._states)
            self._active = False
            return CurriculumRestoreResult(None, None, True)

        state_name = state.get("name") or state.get("state_name")
        target_type = state.get("target_type")
        target_value = state.get("target_value")
        for index, candidate in enumerate(self._states):
            candidate_name = candidate.get("name") or candidate.get("state_name")
            if state_name == candidate_name or (
                target_type == candidate.get("target_type")
                and target_value == candidate.get("target_value")
            ):
                self._state_index = index
                self._active = True
                return CurriculumRestoreResult(candidate, candidate, True)

        return CurriculumRestoreResult(self.current_state(), state, False)
