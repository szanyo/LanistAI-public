# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Lossless Susan step snapshots and human-readable diagnostic states."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import cast

STOP_ENTRY_SPEED_MPS = 0.05
STOP_EXIT_SPEED_MPS = 0.20
MOTION_ACCELERATION_EPSILON_MPS2 = 0.05
OPTIMIZER_INTERVENTION_EPSILON_MPS2 = 0.05


class MotionState(StrEnum):
    """Human-readable physical motion state for one EGO step."""

    UNKNOWN = "UNKNOWN"
    MOVING = "MOVING"
    DECELERATING = "DECELERATING"
    STOPPED = "STOPPED"
    ACCELERATING = "ACCELERATING"


class StopLifecycleEvent(StrEnum):
    """Boundary event emitted by the diagnostic stop state machine."""

    NONE = "NONE"
    STOP_BEGIN = "STOP_BEGIN"
    STOP_ACTIVE = "STOP_ACTIVE"
    STOP_END = "STOP_END"


class StopCause(StrEnum):
    """Constraint context visible while EGO is physically stopped."""

    NONE = "NONE"
    TRAFFIC_SIGNAL = "TRAFFIC_SIGNAL"
    LEAD_VEHICLE = "LEAD_VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    MULTIPLE = "MULTIPLE"
    UNKNOWN = "UNKNOWN"


class ControlUseCase(StrEnum):
    """Non-exclusive control situations active during one simulator step."""

    FREE_DRIVE = "FREE_DRIVE"
    SPEED_TRACKING = "SPEED_TRACKING"
    TRAFFIC_SIGNAL_APPROACH = "TRAFFIC_SIGNAL_APPROACH"
    TRAFFIC_SIGNAL_STOP = "TRAFFIC_SIGNAL_STOP"
    TRAFFIC_SIGNAL_RELEASE = "TRAFFIC_SIGNAL_RELEASE"
    LEAD_FOLLOWING = "LEAD_FOLLOWING"
    PEDESTRIAN_AVOIDANCE = "PEDESTRIAN_AVOIDANCE"
    OPTIMIZER_INTERVENTION = "OPTIMIZER_INTERVENTION"
    SUMO_INTERVENTION = "SUMO_INTERVENTION"
    TERMINAL_STEP = "TERMINAL_STEP"


@dataclass(frozen=True, slots=True)
class SusanDiagnosticStepSnapshot:
    """Preserve already-read Susan control values for lazy diagnostics."""

    step_counter: int = 0
    policy_action: float | None = None
    actual_speed_mps: float | None = None
    requested_speed_mps: float | None = None
    reference_speed_mps: float | None = None
    road_speed_limit_mps: float | None = None
    previous_speed_mps: float | None = None
    base_command_mps2: float | None = None
    applied_command_mps2: float | None = None
    supervised_command_mps2: float | None = None
    actual_acceleration_mps2: float | None = None
    optimizer_delta_mps2: float | None = None
    optimizer_correction_mps2: float | None = None
    optimizer_cost: float | None = None
    distance_to_stopline_m: float | None = None
    ego_station_m: float | None = None
    raw_ego_station_m: float | None = None
    target_ego_station_m: float | None = None
    supervisor_ego_station_m: float | None = None
    ego_position_x_m: float | None = None
    ego_position_y_m: float | None = None
    ego_road_id: str | int | None = None
    intersection_number: int | None = None
    traffic_light_state_code: int | None = None
    traffic_light_tls_id: str | None = None
    traffic_light_link_index: int | None = None
    traffic_signal_constraint_active: bool = False
    traffic_signal_constraint_cleared: bool = False
    red_light_overdrive: bool = False
    ego_lead_vehicle_present: bool = False
    ego_lead_vehicle_id: str | None = None
    ego_lead_distance_m: float | None = None
    ego_lead_speed_mps: float | None = None
    supervisor_vehicle_conflict: bool = False
    policy_object_ids: tuple[str, ...] = ()
    policy_object_kinds: tuple[str, ...] = ()
    policy_object_distances_m: tuple[float, ...] = ()
    policy_vehicle_slot_count: int = 0
    policy_pedestrian_slot_count: int = 0
    active_pedestrian_count: int = 0
    pedestrian_constraint_active: bool = False
    pedestrian_id: str | None = None
    pedestrian_station_m: float | None = None
    pedestrian_speed_mps: float | None = None
    pedestrian_crossing_state: int | None = None
    simulator_read_quality: str | None = None
    observation_values: Mapping[str, object] = field(default_factory=dict)
    reward_values: Mapping[str, object] = field(default_factory=dict)


_BASE_FIELDNAMES = (
    "scenario_index",
    "step_counter",
    "motion_state",
    "stop_event",
    "stop_cause",
    "active_use_cases",
    "termination_reason",
    "truncation_reason",
    "policy_action",
    "actual_speed_mps",
    "requested_speed_mps",
    "reference_speed_mps",
    "road_speed_limit_mps",
    "previous_speed_mps",
    "base_command_mps2",
    "applied_command_mps2",
    "supervised_command_mps2",
    "actual_acceleration_mps2",
    "optimizer_delta_mps2",
    "optimizer_correction_mps2",
    "optimizer_cost",
    "distance_to_stopline_m",
    "ego_station_m",
    "raw_ego_station_m",
    "target_ego_station_m",
    "supervisor_ego_station_m",
    "ego_position_x_m",
    "ego_position_y_m",
    "ego_road_id",
    "intersection_number",
    "traffic_light_state_code",
    "traffic_light_tls_id",
    "traffic_light_link_index",
    "red_light_overdrive",
    "ego_lead_vehicle_present",
    "ego_lead_vehicle_id",
    "ego_lead_distance_m",
    "ego_lead_speed_mps",
    "supervisor_vehicle_conflict",
    "policy_vehicle_slot_count",
    "policy_pedestrian_slot_count",
    "active_pedestrian_count",
    "policy_object_1_id",
    "policy_object_1_kind",
    "policy_object_1_distance_to_ego_m",
    "policy_object_2_id",
    "policy_object_2_kind",
    "policy_object_2_distance_to_ego_m",
    "policy_object_3_id",
    "policy_object_3_kind",
    "policy_object_3_distance_to_ego_m",
    "supervisor_pedestrian_conflict",
    "pedestrian_id",
    "pedestrian_station_m",
    "pedestrian_speed_mps",
    "pedestrian_crossing_state",
    "simulator_read_quality",
)


def _finite(value: object) -> bool:
    """Return whether a value is a finite non-boolean number."""

    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(cast(float | int | str, value)))
    except (TypeError, ValueError):
        return False


def susan_diagnostic_trace_fieldnames(
    *,
    observation_fields: tuple[str, ...] = (),
    reward_fields: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return the stable Susan trace columns for recorder construction."""

    motion_flags = tuple(
        f"motion_state_{state.value.lower()}" for state in MotionState
    )
    stop_event_flags = tuple(
        f"stop_event_{event.value.lower()}" for event in StopLifecycleEvent
    )
    cause_flags = tuple(
        f"stop_cause_{cause.value.lower()}" for cause in StopCause
    )
    use_case_flags = tuple(
        f"use_case_{use_case.value.lower()}" for use_case in ControlUseCase
    )
    return tuple(
        dict.fromkeys(
            (
                *_BASE_FIELDNAMES,
                *motion_flags,
                *stop_event_flags,
                *cause_flags,
                *use_case_flags,
                *(f"observation_{name}" for name in observation_fields),
                *(f"reward_{name}" for name in reward_fields),
            )
        )
    )


class SusanDiagnosticTraceProjector:
    """Project raw snapshots and own diagnostic-only stop hysteresis."""

    def __init__(self) -> None:
        """Initialize one episode-local diagnostic state machine."""

        self._stopped = False
        self._active_stop_cause = StopCause.NONE
        self._cached_step_counter: int | None = None
        self._cached_base_row: dict[str, object] | None = None

    def project(
        self,
        snapshot: SusanDiagnosticStepSnapshot,
        *,
        scenario_index: int | None = None,
        terminated: bool = False,
        truncated: bool = False,
        termination_reason: str | None = None,
        truncation_reason: str | None = None,
    ) -> dict[str, object]:
        """Return one flat lossless row with diagnostic state labels."""

        row = self._row_with_terminal_state(
            snapshot,
            terminated=terminated,
            truncated=truncated,
        )
        row.update(
            {
                "scenario_index": scenario_index,
                "termination_reason": termination_reason,
                "truncation_reason": truncation_reason,
            }
        )
        row.update(
            {
                f"observation_{name}": value
                for name, value in snapshot.observation_values.items()
            }
        )
        row.update(
            {
                f"reward_{name}": value
                for name, value in snapshot.reward_values.items()
            }
        )
        return row

    def project_training_metrics(
        self,
        snapshot: SusanDiagnosticStepSnapshot,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, object]:
        """Project core numeric fields without detailed observation/reward rows."""

        return self._row_with_terminal_state(
            snapshot,
            terminated=terminated,
            truncated=truncated,
        )

    def _row_with_terminal_state(
        self,
        snapshot: SusanDiagnosticStepSnapshot,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, object]:
        """Return one cached classification with current terminal indicators."""

        cache_miss = (
            snapshot.step_counter != self._cached_step_counter
            or self._cached_base_row is None
        )
        if cache_miss:
            self._cached_step_counter = snapshot.step_counter
            self._cached_base_row = self._project_snapshot(snapshot)
        row = dict(cast(dict[str, object], self._cached_base_row))
        if not cache_miss:
            row.update(
                self._raw_row(
                    snapshot,
                    scenario_index=None,
                    terminated=False,
                    truncated=False,
                    termination_reason=None,
                    truncation_reason=None,
                )
            )
        row["terminated"] = int(terminated)
        row["truncated"] = int(truncated)
        terminal = bool(terminated or truncated)
        row[f"use_case_{ControlUseCase.TERMINAL_STEP.value.lower()}"] = int(
            terminal
        )
        row["active_use_cases"] = "|".join(
            use_case.value
            for use_case in ControlUseCase
            if bool(row.get(f"use_case_{use_case.value.lower()}", 0))
        )
        return row

    def _project_snapshot(
        self,
        snapshot: SusanDiagnosticStepSnapshot,
    ) -> dict[str, object]:
        """Advance state once and project the terminal-independent base row."""

        stopped, stop_event = self._stop_state(snapshot.actual_speed_mps)
        cause_candidates = self._cause_candidates(snapshot)
        stop_cause = self._stop_cause(
            stopped=stopped,
            event=stop_event,
            candidates=cause_candidates,
        )
        use_cases = self._use_cases(
            snapshot,
            stopped=stopped,
            terminated=False,
            truncated=False,
            cause_candidates=cause_candidates,
        )
        row = self._raw_row(
            snapshot,
            scenario_index=None,
            terminated=False,
            truncated=False,
            termination_reason=None,
            truncation_reason=None,
        )
        motion_state = self._motion_state(snapshot, stopped)
        row.update(
            {
                "motion_state": motion_state.value,
                "stop_event": stop_event.value,
                "stop_cause": stop_cause.value,
                "active_use_cases": "|".join(
                    use_case.value
                    for use_case in ControlUseCase
                    if use_case in use_cases
                ),
            }
        )
        for state in MotionState:
            row[f"motion_state_{state.value.lower()}"] = int(
                motion_state is state
            )
        for event in StopLifecycleEvent:
            row[f"stop_event_{event.value.lower()}"] = int(
                stop_event is event
            )
        for cause in StopCause:
            candidate_active = cause in cause_candidates
            resolved_active = cause is stop_cause
            row[f"stop_cause_{cause.value.lower()}"] = int(
                candidate_active or resolved_active
            )
        for use_case in ControlUseCase:
            row[f"use_case_{use_case.value.lower()}"] = int(
                use_case in use_cases
            )
        return row

    def _stop_state(
        self,
        actual_speed_mps: float | None,
    ) -> tuple[bool, StopLifecycleEvent]:
        """Advance the diagnostic stop state with entry/exit hysteresis."""

        previous = self._stopped
        if not _finite(actual_speed_mps):
            return previous, StopLifecycleEvent.NONE
        speed = abs(float(cast(float, actual_speed_mps)))
        stopped = previous
        if speed <= STOP_ENTRY_SPEED_MPS:
            stopped = True
        elif speed >= STOP_EXIT_SPEED_MPS:
            stopped = False
        self._stopped = stopped
        if stopped and not previous:
            return stopped, StopLifecycleEvent.STOP_BEGIN
        if stopped:
            return stopped, StopLifecycleEvent.STOP_ACTIVE
        if previous:
            return stopped, StopLifecycleEvent.STOP_END
        return stopped, StopLifecycleEvent.NONE

    @staticmethod
    def _cause_candidates(
        snapshot: SusanDiagnosticStepSnapshot,
    ) -> frozenset[StopCause]:
        """Return every already-observed blocking context for this step."""

        candidates: set[StopCause] = set()
        if snapshot.traffic_signal_constraint_active:
            candidates.add(StopCause.TRAFFIC_SIGNAL)
        if snapshot.supervisor_vehicle_conflict:
            candidates.add(StopCause.LEAD_VEHICLE)
        if snapshot.pedestrian_constraint_active:
            candidates.add(StopCause.PEDESTRIAN)
        return frozenset(candidates)

    def _stop_cause(
        self,
        *,
        stopped: bool,
        event: StopLifecycleEvent,
        candidates: frozenset[StopCause],
    ) -> StopCause:
        """Resolve the readable cause without discarding candidate flags."""

        if event is StopLifecycleEvent.STOP_END:
            cause = self._active_stop_cause
            self._active_stop_cause = StopCause.NONE
            return cause
        if not stopped:
            return StopCause.NONE
        if not candidates:
            cause = StopCause.UNKNOWN
        elif len(candidates) > 1:
            cause = StopCause.MULTIPLE
        else:
            cause = next(iter(candidates))
        self._active_stop_cause = cause
        return cause

    @staticmethod
    def _motion_state(
        snapshot: SusanDiagnosticStepSnapshot,
        stopped: bool,
    ) -> MotionState:
        """Return the diagnostic motion state for the current raw values."""

        if not _finite(snapshot.actual_speed_mps):
            return MotionState.UNKNOWN
        if stopped:
            return MotionState.STOPPED
        if _finite(snapshot.actual_acceleration_mps2):
            acceleration = float(
                cast(float, snapshot.actual_acceleration_mps2)
            )
            if acceleration < -MOTION_ACCELERATION_EPSILON_MPS2:
                return MotionState.DECELERATING
            if acceleration > MOTION_ACCELERATION_EPSILON_MPS2:
                return MotionState.ACCELERATING
        return MotionState.MOVING

    @staticmethod
    def _use_cases(
        snapshot: SusanDiagnosticStepSnapshot,
        *,
        stopped: bool,
        terminated: bool,
        truncated: bool,
        cause_candidates: frozenset[StopCause],
    ) -> frozenset[ControlUseCase]:
        """Return all non-exclusive use cases visible on this step."""

        use_cases: set[ControlUseCase] = set()
        if _finite(snapshot.reference_speed_mps):
            use_cases.add(ControlUseCase.SPEED_TRACKING)
        if StopCause.TRAFFIC_SIGNAL in cause_candidates:
            use_cases.add(ControlUseCase.TRAFFIC_SIGNAL_APPROACH)
            if stopped:
                use_cases.add(ControlUseCase.TRAFFIC_SIGNAL_STOP)
        if snapshot.traffic_signal_constraint_cleared:
            use_cases.add(ControlUseCase.TRAFFIC_SIGNAL_RELEASE)
        if StopCause.LEAD_VEHICLE in cause_candidates:
            use_cases.add(ControlUseCase.LEAD_FOLLOWING)
        if StopCause.PEDESTRIAN in cause_candidates:
            use_cases.add(ControlUseCase.PEDESTRIAN_AVOIDANCE)
        if (
            _finite(snapshot.optimizer_correction_mps2)
            and abs(float(cast(float, snapshot.optimizer_correction_mps2)))
            > OPTIMIZER_INTERVENTION_EPSILON_MPS2
        ):
            use_cases.add(ControlUseCase.OPTIMIZER_INTERVENTION)
        if bool(snapshot.reward_values.get("sync_sumo_intervention", 0.0)):
            use_cases.add(ControlUseCase.SUMO_INTERVENTION)
        if terminated or truncated:
            use_cases.add(ControlUseCase.TERMINAL_STEP)
        if not cause_candidates:
            use_cases.add(ControlUseCase.FREE_DRIVE)
        return frozenset(use_cases)

    @staticmethod
    def _raw_row(
        snapshot: SusanDiagnosticStepSnapshot,
        *,
        scenario_index: int | None,
        terminated: bool,
        truncated: bool,
        termination_reason: str | None,
        truncation_reason: str | None,
    ) -> dict[str, object]:
        """Project direct values before adding derived diagnostic states."""

        return {
            "scenario_index": scenario_index,
            "step_counter": snapshot.step_counter,
            "terminated": int(terminated),
            "truncated": int(truncated),
            "termination_reason": termination_reason,
            "truncation_reason": truncation_reason,
            "policy_action": snapshot.policy_action,
            "actual_speed_mps": snapshot.actual_speed_mps,
            "requested_speed_mps": snapshot.requested_speed_mps,
            "reference_speed_mps": snapshot.reference_speed_mps,
            "road_speed_limit_mps": snapshot.road_speed_limit_mps,
            "previous_speed_mps": snapshot.previous_speed_mps,
            "base_command_mps2": snapshot.base_command_mps2,
            "applied_command_mps2": snapshot.applied_command_mps2,
            "supervised_command_mps2": snapshot.supervised_command_mps2,
            "actual_acceleration_mps2": snapshot.actual_acceleration_mps2,
            "optimizer_delta_mps2": snapshot.optimizer_delta_mps2,
            "optimizer_correction_mps2": snapshot.optimizer_correction_mps2,
            "optimizer_cost": snapshot.optimizer_cost,
            "distance_to_stopline_m": snapshot.distance_to_stopline_m,
            "ego_station_m": snapshot.ego_station_m,
            "raw_ego_station_m": snapshot.raw_ego_station_m,
            "target_ego_station_m": snapshot.target_ego_station_m,
            "supervisor_ego_station_m": snapshot.supervisor_ego_station_m,
            "ego_position_x_m": snapshot.ego_position_x_m,
            "ego_position_y_m": snapshot.ego_position_y_m,
            "ego_road_id": snapshot.ego_road_id,
            "intersection_number": snapshot.intersection_number,
            "traffic_light_state_code": snapshot.traffic_light_state_code,
            "traffic_light_tls_id": snapshot.traffic_light_tls_id,
            "traffic_light_link_index": snapshot.traffic_light_link_index,
            "red_light_overdrive": int(snapshot.red_light_overdrive),
            "ego_lead_vehicle_present": int(
                snapshot.ego_lead_vehicle_present
            ),
            "ego_lead_vehicle_id": snapshot.ego_lead_vehicle_id,
            "ego_lead_distance_m": snapshot.ego_lead_distance_m,
            "ego_lead_speed_mps": snapshot.ego_lead_speed_mps,
            "supervisor_vehicle_conflict": int(
                snapshot.supervisor_vehicle_conflict
            ),
            **_policy_object_row(snapshot),
            "supervisor_pedestrian_conflict": int(
                snapshot.pedestrian_constraint_active
            ),
            "pedestrian_id": snapshot.pedestrian_id,
            "pedestrian_station_m": snapshot.pedestrian_station_m,
            "pedestrian_speed_mps": snapshot.pedestrian_speed_mps,
            "pedestrian_crossing_state": snapshot.pedestrian_crossing_state,
            "simulator_read_quality": snapshot.simulator_read_quality,
        }


def _policy_object_row(
    snapshot: SusanDiagnosticStepSnapshot,
) -> dict[str, object]:
    """Project fixed policy-slot metadata without expanding the main row."""

    row: dict[str, object] = {
        "policy_vehicle_slot_count": snapshot.policy_vehicle_slot_count,
        "policy_pedestrian_slot_count": snapshot.policy_pedestrian_slot_count,
        "active_pedestrian_count": snapshot.active_pedestrian_count,
    }
    for index in range(1, 4):
        offset = index - 1
        row[f"policy_object_{index}_id"] = _tuple_item(
            snapshot.policy_object_ids,
            offset,
        )
        row[f"policy_object_{index}_kind"] = _tuple_item(
            snapshot.policy_object_kinds,
            offset,
        )
        row[f"policy_object_{index}_distance_to_ego_m"] = _tuple_item(
            snapshot.policy_object_distances_m,
            offset,
        )
    return row


def _tuple_item(values: tuple[object, ...], index: int) -> object | None:
    """Return one optional fixed-slot diagnostic value."""

    return values[index] if index < len(values) else None


__all__ = [
    "ControlUseCase",
    "MotionState",
    "StopCause",
    "StopLifecycleEvent",
    "SusanDiagnosticStepSnapshot",
    "SusanDiagnosticTraceProjector",
    "susan_diagnostic_trace_fieldnames",
]
