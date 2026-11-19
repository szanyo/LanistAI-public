# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Traffic-light phase control and timing diagnostics."""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TypeVar

from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_NOT_APPLICABLE,
    TRAFFIC_LIGHT_STATE_RED,
    TRAFFIC_LIGHT_STATE_RED_YELLOW,
    TRAFFIC_LIGHT_STATE_UNKNOWN,
    TRAFFIC_LIGHT_STATE_YELLOW,
)
from lanista.reinforcengine.utils.simulation.ports import (
    SimulationOperationError,
    SimulationReadQuality,
    SimulatorPort,
)

from ..simulation.simulation_contracts import TrafficLightControllerPort
from .traffic_light_timing_state import (
    TrafficLightTimingDiagnostics,
    TrafficLightTimingState,
)

ReadValue = TypeVar("ReadValue")


def _as_int_code(value: object) -> int | None:
    """Return this traffic light service module's value as int code."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clip(value: float, lower: float, upper: float) -> float:
    """Return the clip value owned by this traffic light service module."""
    return max(lower, min(upper, float(value)))


@dataclass
class TrafficLightService:
    """Calculate signal timing through a narrow, injectable SUMO adapter."""

    controller: TrafficLightControllerPort
    simulation: SimulatorPort
    last_read_quality: SimulationReadQuality = field(
        default=SimulationReadQuality.LIVE,
        init=False,
    )
    last_read_error: SimulationOperationError | None = field(default=None, init=False)
    timing_state: TrafficLightTimingState = field(
        default_factory=TrafficLightTimingState,
        init=False,
    )

    def _optional_read(
        self,
        operation: str,
        reader: Callable[[], ReadValue],
        fallback: ReadValue,
    ) -> ReadValue:
        """Return the optional read value owned by this traffic light service."""
        read = self.simulation.optional_read(operation, reader, fallback)
        self.last_read_quality = read.quality
        self.last_read_error = read.error
        return read.value

    @staticmethod
    def _as_int_code(value):
        """Return this traffic light service's value as int code."""
        return _as_int_code(value)

    @staticmethod
    def _sumo_signal_char_to_traffic_light_state_code(signal_char):
        """Return the sumo signal char to traffic light state code value owned by this traffic
        light service.
        """
        if signal_char is None:
            return TRAFFIC_LIGHT_STATE_UNKNOWN

        signal_char = str(signal_char)[:1]
        if signal_char in {"G", "g"}:
            return TRAFFIC_LIGHT_STATE_GREEN
        if signal_char in {"y", "Y"}:
            return TRAFFIC_LIGHT_STATE_YELLOW
        if signal_char in {"u", "U"}:
            return TRAFFIC_LIGHT_STATE_RED_YELLOW
        if signal_char in {"r", "R", "s"}:
            return TRAFFIC_LIGHT_STATE_RED
        return TRAFFIC_LIGHT_STATE_UNKNOWN

    @staticmethod
    def _traffic_light_state_priority(state_code):
        """Return the traffic light state priority value owned by this traffic light service."""
        return {
            TRAFFIC_LIGHT_STATE_YELLOW: 0,
            TRAFFIC_LIGHT_STATE_RED_YELLOW: 1,
            TRAFFIC_LIGHT_STATE_RED: 2,
            TRAFFIC_LIGHT_STATE_GREEN: 3,
            TRAFFIC_LIGHT_STATE_UNKNOWN: 4,
            TRAFFIC_LIGHT_STATE_NOT_APPLICABLE: 5,
        }.get(state_code, 4)

    @classmethod
    def _select_safest_traffic_light_state_code(cls, state_codes):
        """Select the safest traffic light state code owned by this traffic light service."""
        valid_codes = [
            cls._as_int_code(code)
            for code in state_codes
            if cls._as_int_code(code) is not None
        ]
        if not valid_codes:
            return TRAFFIC_LIGHT_STATE_UNKNOWN
        return min(valid_codes, key=cls._traffic_light_state_priority)

    @staticmethod
    def _sumo_phase_duration_seconds(phase, default=0.0):
        """Return the sumo phase duration seconds value owned by this traffic light service."""
        for attr_name in ("duration", "minDur", "maxDur"):
            try:
                value = float(getattr(phase, attr_name))
            except (TypeError, ValueError, AttributeError):
                continue
            if math.isfinite(value) and value > 0.0:
                return value
        return max(0.0, float(default))

    @staticmethod
    def _sumo_phase_state_string(phase):
        """Return the sumo phase state string value owned by this traffic light service."""
        return str(getattr(phase, "state", "") or "")

    def _get_tls_program_phases(self, tls_id, current_program_id=None):
        """Return the tls program phases exposed by this traffic light service."""
        if current_program_id is None:
            current_program_id = self._optional_read(
                "read Susan traffic-light program",
                lambda: self.simulation.get_program(tls_id),
                None,
            )

        logics = list(
            self._optional_read(
                "read Susan traffic-light program logics",
                lambda: self.simulation.get_program_logics(tls_id),
                (),
            )
        )
        if not logics:
            return []

        selected_logic = None
        if current_program_id is not None:
            for logic in logics:
                logic_program_id = getattr(
                    logic,
                    "programID",
                    getattr(logic, "program_id", None),
                )
                if str(logic_program_id) == str(current_program_id):
                    selected_logic = logic
                    break

        if selected_logic is None:
            selected_logic = logics[0]

        return list(getattr(selected_logic, "phases", []) or [])

    def _phase_aggregate_state_for_link_indices(self, phase, tls_link_indices):
        """Return the phase aggregate state for link indices value owned by this traffic light
        service.
        """
        state_string = self._sumo_phase_state_string(phase)
        codes = []
        for link_index in tls_link_indices:
            if link_index < 0 or link_index >= len(state_string):
                codes.append(TRAFFIC_LIGHT_STATE_UNKNOWN)
            else:
                codes.append(self._sumo_signal_char_to_traffic_light_state_code(state_string[link_index]))
        return self._select_safest_traffic_light_state_code(codes)

    def _build_ego_tls_state_schedule(self, tls_id, tls_link_indices, current_program_id=None):
        """Build the ego tls state schedule required by this traffic light service."""
        phases = self._get_tls_program_phases(tls_id, current_program_id)
        if not phases:
            return None

        try:
            link_indices = tuple(sorted({int(index) for index in tls_link_indices}))
        except (TypeError, ValueError):
            return None
        if not link_indices:
            return None

        phase_starts = []
        raw_blocks = []
        cycle_time = 0.0

        for phase_index, phase in enumerate(phases):
            phase_starts.append(cycle_time)
            duration = self._sumo_phase_duration_seconds(phase)
            if duration <= 0.0:
                continue

            state_code = self._phase_aggregate_state_for_link_indices(phase, link_indices)
            raw_blocks.append({
                "state_code": state_code,
                "start": cycle_time,
                "end": cycle_time + duration,
                "duration": duration,
                "phase_indices": [phase_index],
            })
            cycle_time += duration

        if cycle_time <= 1e-6 or not raw_blocks:
            return None

        merged_blocks = []
        for block in raw_blocks:
            if merged_blocks and merged_blocks[-1]["state_code"] == block["state_code"]:
                merged_blocks[-1]["end"] = block["end"]
                merged_blocks[-1]["duration"] += block["duration"]
                merged_blocks[-1]["phase_indices"].extend(block["phase_indices"])
            else:
                merged_blocks.append(dict(block))

        if len(merged_blocks) > 1 and merged_blocks[0]["state_code"] == merged_blocks[-1]["state_code"]:
            first = merged_blocks.pop(0)
            last = merged_blocks[-1]
            last["end"] = first["end"] + cycle_time
            last["duration"] += first["duration"]
            last["phase_indices"].extend(first["phase_indices"])

        return {
            "tls_id": tls_id,
            "tls_link_indices": link_indices,
            "program_id": current_program_id,
            "cycle_time": cycle_time,
            "phase_starts": phase_starts,
            "blocks": merged_blocks,
        }

    def _get_ego_tls_state_schedule(self, tls_id, tls_link_indices):
        """Return the ego tls state schedule exposed by this traffic light service."""
        current_program_id = self._optional_read(
            "read Susan traffic-light program for schedule",
            lambda: self.simulation.get_program(tls_id),
            None,
        )

        try:
            link_indices = tuple(sorted({int(index) for index in tls_link_indices}))
        except (TypeError, ValueError):
            return None
        if not link_indices:
            return None

        cache_key = (str(tls_id), link_indices, str(current_program_id))
        schedule = self.timing_state.schedule_cache.get(cache_key)
        if schedule is None:
            schedule = self._build_ego_tls_state_schedule(tls_id, link_indices, current_program_id)
            if schedule is not None:
                self.timing_state.schedule_cache[cache_key] = schedule

        return schedule

    def _find_ego_tls_state_block(self, schedule, cycle_position):
        """Find the ego tls state block required by this traffic light service."""
        cycle_time = schedule["cycle_time"]
        position = float(cycle_position) % cycle_time

        for block in schedule["blocks"]:
            start = float(block["start"])
            end = float(block["end"])

            if end <= cycle_time:
                if start <= position < end:
                    return block, position
            else:
                wrapped_end = end - cycle_time
                if position >= start:
                    return block, position
                if position < wrapped_end:
                    return block, position + cycle_time

        return None, position

    def _ego_tls_state_remaining_norm(self, tls_id, tls_link_indices, expected_state_code=None):
        """Return the ego tls state remaining norm value owned by this traffic light service."""
        schedule = self._get_ego_tls_state_schedule(tls_id, tls_link_indices)
        if schedule is None:
            return None

        current_phase_index = self._optional_read(
            "read Susan traffic-light phase telemetry",
            lambda: self.simulation.get_phase(tls_id),
            None,
        )
        if current_phase_index is None:
            return None

        phase_starts = schedule["phase_starts"]
        if current_phase_index < 0 or current_phase_index >= len(phase_starts):
            return None

        spent_duration = self._optional_read(
            "read Susan traffic-light spent duration",
            lambda: self.simulation.get_spent_duration(tls_id),
            None,
        )
        if spent_duration is None:
            timing = self._optional_read(
                "read Susan traffic-light switch timing fallback",
                lambda: (
                    self.simulation.get_phase_duration(tls_id),
                    self.simulation.get_simulation_time(),
                    self.simulation.get_next_switch(tls_id),
                ),
                None,
            )
            if timing is None:
                return None
            phase_duration, current_time, next_switch_time = timing
            spent_duration = phase_duration - max(
                0.0,
                next_switch_time - current_time,
            )

        cycle_position = float(phase_starts[current_phase_index]) + max(0.0, spent_duration)
        block, block_position = self._find_ego_tls_state_block(schedule, cycle_position)
        if block is None:
            return None

        expected_state = _as_int_code(expected_state_code)
        if (
            expected_state not in (None, TRAFFIC_LIGHT_STATE_UNKNOWN, TRAFFIC_LIGHT_STATE_NOT_APPLICABLE)
            and _as_int_code(block.get("state_code")) != expected_state
        ):
            return None

        remaining_seconds = max(0.0, float(block["end"]) - float(block_position))
        total_seconds = max(float(block["duration"]), 1e-6)
        return _clip(remaining_seconds / total_seconds, 0.0, 1.0)

    def _parse_next_tls_entries_for_timing(self, next_tls):
        """Parse the next tls entries for timing owned by this traffic light service."""
        parsed_entries = []
        for entry in next_tls or []:
            if entry is None or len(entry) < 4:
                continue
            try:
                tls_id = entry[0]
                link_index = int(entry[1])
                distance = float(entry[2])
                signal_char = entry[3]
            except (TypeError, ValueError):
                continue
            parsed_entries.append({
                "tls_id": tls_id,
                "link_index": link_index,
                "distance": distance,
                "state_code": self._sumo_signal_char_to_traffic_light_state_code(signal_char),
                "signal_char": signal_char,
            })
        return parsed_entries

    def _select_next_tls_group_for_timing(self, next_tls, traffic_light_state_code=None, preferred_tls_id=None):
        """Select the next tls group for timing owned by this traffic light service."""
        entries = self._parse_next_tls_entries_for_timing(next_tls)
        if not entries:
            return None, ()

        if preferred_tls_id is not None:
            preferred_entries = [
                entry for entry in entries
                if str(entry["tls_id"]) == str(preferred_tls_id)
            ]
            if preferred_entries:
                entries = preferred_entries

        observed_code = _as_int_code(traffic_light_state_code)
        matching_entries = [
            entry for entry in entries
            if observed_code is not None and entry["state_code"] == observed_code
        ]
        candidates = matching_entries if matching_entries else entries
        candidates.sort(key=lambda entry: entry["distance"])
        selected = candidates[0]
        return selected["tls_id"], (selected["link_index"],)

    def _preferred_scenario_tls_id_for_timing(self):
        """Return the preferred scenario tls id for timing value owned by this traffic light
        service.
        """
        context = self.controller.current_scenario_context.traffic_light_context
        return context.tls_id

    def _scenario_traffic_light_context_for_timing(self, traffic_light_state_code=None):
        """Return the scenario traffic light context for timing value owned by this traffic light
        service.
        """
        context = self.controller.current_scenario_context.traffic_light_context
        if context is None:
            return None

        tls_id = getattr(context, "tls_id", None)
        link_indices = tuple(getattr(context, "link_indices", ()) or ())
        if tls_id is None or not link_indices:
            return None

        compact_state = _as_int_code(getattr(context, "compact_state", None))
        if compact_state is None:
            compact_state = _as_int_code(traffic_light_state_code)

        return {
            "tls_id": tls_id,
            "link_indices": link_indices,
            "state_code": compact_state,
            "canonical": True,
        }

    def _reset_observed_state_remaining_guard(self):
        """Reset the observed state remaining guard owned by this traffic light service."""
        self.timing_state.reset_observation_guard()

    def _record_observed_state_remaining_diagnostics(
        self,
        remaining_norm,
        traffic_light_state_code,
        tls_id=None,
        tls_link_indices=None,
    ):
        """Record the observed state remaining diagnostics owned by this traffic light service."""
        try:
            raw_remaining = _clip(float(remaining_norm), 0.0, 1.0)
        except (TypeError, ValueError):
            raw_remaining = 0.0

        observed_state = _as_int_code(traffic_light_state_code)
        context = (
            str(tls_id) if tls_id is not None else None,
            tuple(tls_link_indices or ()),
            observed_state,
        )

        previous_context = self.timing_state.last_remaining_context
        previous_remaining = self.timing_state.last_remaining_norm

        raw_delta = 0.0
        context_changed = 0.0
        monotonic_violation = 0.0
        if previous_remaining is not None:
            previous_remaining = float(previous_remaining)
            raw_delta = raw_remaining - previous_remaining
            context_changed = 1.0 if previous_context != context else 0.0
            monotonic_violation = 1.0 if context_changed == 0.0 and raw_delta > 1e-6 else 0.0

        if monotonic_violation:
            strict_mode = str(os.environ.get("LANISTAI_STRICT_TL_PHASE_MONOTONIC", "0") or "").strip().lower()
            if strict_mode in {"1", "true", "on", "yes"}:
                raise AssertionError(
                    "Traffic-light phase remaining time increased within the same "
                    f"context: previous={previous_remaining:.6f}, current={raw_remaining:.6f}, "
                    f"delta={raw_delta:.6f}, context={context!r}"
                )

        self.timing_state.last_remaining_context = context
        self.timing_state.last_remaining_norm = raw_remaining
        self.timing_state.phase_remaining_raw_delta_norm = raw_delta
        self.timing_state.phase_remaining_delta_norm = raw_delta
        self.timing_state.context_changed = context_changed
        self.timing_state.phase_remaining_monotonic_violation = monotonic_violation
        return raw_remaining

    def _traffic_light_phase_timing_observation_features(
        self,
        upcoming_intersection_type_code=None,
        traffic_light_state_code=None,
    ):
        """Return the traffic light phase timing observation features value owned by this traffic
        light service.
        """
        defaults = {
            "tl_phase_remaining_norm": 0.0,
            "tl_phase_remaining_delta_norm": 0.0,
            "tl_phase_remaining_raw_delta_norm": 0.0,
            "tl_timing_context_changed": 0.0,
            "tl_phase_remaining_monotonic_violation": 0.0,
        }
        if _as_int_code(upcoming_intersection_type_code) != INTERSECTION_TYPE_TRAFFIC_LIGHT:
            self._reset_observed_state_remaining_guard()
            return defaults

        timing_context = self._scenario_traffic_light_context_for_timing(
            traffic_light_state_code
        )
        if timing_context is None:
            self._reset_observed_state_remaining_guard()
            return defaults

        tls_id = timing_context["tls_id"]
        tls_link_indices = timing_context["link_indices"]
        state_remaining_norm = self._ego_tls_state_remaining_norm(
            tls_id,
            tls_link_indices,
            expected_state_code=timing_context["state_code"],
        )
        if state_remaining_norm is None:
            self._reset_observed_state_remaining_guard()
            return defaults

        remaining_norm = _clip(float(state_remaining_norm), 0.0, 1.0)
        remaining_norm = self._record_observed_state_remaining_diagnostics(
            remaining_norm,
            traffic_light_state_code,
            tls_id=tls_id,
            tls_link_indices=tls_link_indices,
        )
        diagnostics = self.timing_state.diagnostics()
        return {
            "tl_phase_remaining_norm": remaining_norm,
            "tl_phase_remaining_delta_norm": diagnostics.phase_remaining_delta_norm,
            "tl_phase_remaining_raw_delta_norm": (
                diagnostics.phase_remaining_raw_delta_norm
            ),
            "tl_timing_context_changed": diagnostics.context_changed,
            "tl_phase_remaining_monotonic_violation": (
                diagnostics.phase_remaining_monotonic_violation
            ),
        }

    def reset_episode(self) -> None:
        """Clear timing state at the controller episode boundary."""

        self.timing_state.reset()

    def timing_diagnostics(self) -> TrafficLightTimingDiagnostics:
        """Return the current immutable telemetry snapshot."""

        return self.timing_state.diagnostics()

    def timing_observation_features(
        self,
        upcoming_intersection_type_code: object = None,
        traffic_light_state_code: object = None,
    ) -> dict[str, float]:
        """Return the timing observation features value owned by this traffic light service."""
        return self._traffic_light_phase_timing_observation_features(
            upcoming_intersection_type_code,
            traffic_light_state_code,
        )

    def context_diagnostic_features(self) -> dict[str, float]:
        """Return the context diagnostic features value owned by this traffic light service."""
        return self._traffic_light_context_diagnostic_features()

    def _traffic_light_context_diagnostic_features(self):
        """Return the traffic light context diagnostic features value owned by this traffic light
        service.
        """
        context = self.controller.current_scenario_context.traffic_light_context

        tls_id = getattr(context, "tls_id", None) if context is not None else None
        link_indices = tuple(getattr(context, "link_indices", ()) or ()) if context is not None else ()
        intersection_number = getattr(context, "intersection_number", -1) if context is not None else -1

        try:
            tls_id_numeric = float(tls_id)
        except (TypeError, ValueError):
            tls_id_numeric = -1.0

        link_index = float(link_indices[0]) if len(link_indices) == 1 else -1.0

        try:
            intersection_number = float(intersection_number)
        except (TypeError, ValueError):
            intersection_number = -1.0

        return {
            "tl_tls_id_numeric": tls_id_numeric,
            "tl_link_index": link_index,
            "tl_intersection_number": intersection_number,
        }
