# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Mutable phase-timing state and immutable observation diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field

TrafficLightContextKey = tuple[str | None, tuple[int, ...], int | None]
TrafficLightScheduleKey = tuple[str, tuple[int, ...], str]


@dataclass(frozen=True)
class TrafficLightTimingDiagnostics:
    """Immutable timing diagnostics published to observation telemetry."""

    phase_remaining_delta_norm: float
    phase_remaining_raw_delta_norm: float
    context_changed: float
    phase_remaining_monotonic_violation: float


@dataclass
class TrafficLightTimingState:
    """Episode-local cache and monotonicity state owned by the timing service."""

    schedule_cache: dict[TrafficLightScheduleKey, dict[str, object]] = field(
        default_factory=dict
    )
    last_remaining_context: TrafficLightContextKey | None = None
    last_remaining_norm: float | None = None
    phase_remaining_delta_norm: float = 0.0
    phase_remaining_raw_delta_norm: float = 0.0
    context_changed: float = 0.0
    phase_remaining_monotonic_violation: float = 0.0

    def reset(self) -> None:
        """Reset this traffic light timing state to its initial operational state."""
        self.schedule_cache.clear()
        self.reset_observation_guard()

    def reset_observation_guard(self) -> None:
        """Reset the observation guard owned by this traffic light timing state."""
        self.last_remaining_context = None
        self.last_remaining_norm = None
        self.phase_remaining_delta_norm = 0.0
        self.phase_remaining_raw_delta_norm = 0.0
        self.context_changed = 0.0
        self.phase_remaining_monotonic_violation = 0.0

    def diagnostics(self) -> TrafficLightTimingDiagnostics:
        """Return the diagnostics value owned by this traffic light timing state."""
        return TrafficLightTimingDiagnostics(
            phase_remaining_delta_norm=self.phase_remaining_delta_norm,
            phase_remaining_raw_delta_norm=self.phase_remaining_raw_delta_norm,
            context_changed=self.context_changed,
            phase_remaining_monotonic_violation=(
                self.phase_remaining_monotonic_violation
            ),
        )
