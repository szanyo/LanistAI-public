# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed reward configuration, state, component, and channel telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TypedDict, cast

REWARD_CHANNEL_NAMES = ("reward",)
REWARD_COMPONENT_NAMES = ("sync", "accel", "speed", "delta")
SYNC_ERROR_DEADBAND_MPS2 = 0.05


class RewardComponentTelemetryChannelPayload(TypedDict, total=False):
    """Describe the typed mapping for reward component telemetry channel payload."""

    raw: float
    normalized: float
    rolling_average: float
    cumulative_average: float
    current: float
    effective: float
    weighted: float
    weighted_uncapped: float
    diagnostics: dict[str, float]


class RewardTelemetryChannelPayload(TypedDict):
    """Describe the typed mapping for reward telemetry channel payload."""

    total: float
    components: dict[str, RewardComponentTelemetryChannelPayload]


class RewardTelemetryChannels(TypedDict):
    """Describe the typed mapping for reward telemetry channels."""

    reward: RewardTelemetryChannelPayload


@dataclass
class RewardConfigTelemetry:
    """Carry the data required for reward config telemetry."""

    speed_reward_v_max: float = 8.3333
    red_light_a_comfort: float = 1.5
    red_light_brake_horizon_distance: float = 35
    traffic_light_intersection_code: int = 1
    red_light_state_code: int = 1
    red_light_station_threshold: float = 0.0


@dataclass
class SyncRewardStateTelemetry:
    """Carry the data required for sync reward state telemetry."""

    raw: float = 0.0
    normalized: float = 0.0
    rolling_average: float = 0.0
    cumulative_average: float = 0.0
    command: float = 0.0
    actual_acceleration: float = 0.0
    absolute_error: float = 0.0
    effective_error: float = 0.0
    speed_factor: float = 0.0
    direction_penalty: float = 0.0
    magnitude_penalty: float = 0.0
    weighted: float = 0.0
    weighted_uncapped: float = 0.0
    penalty_sum: float = 0.0
    lead_vehicle_present: float = 0.0
    sumo_intervention: float = 0.0


@dataclass
class AccelRewardStateTelemetry:
    """Carry the data required for accel reward state telemetry."""

    raw: float = 0.0
    normalized: float = 0.0
    rolling_average: float = 0.0
    cumulative_average: float = 0.0
    weighted: float = 0.0
    weighted_uncapped: float = 0.0
    penalty_sum: float = 0.0


@dataclass
class SpeedRewardDiagnosticsTelemetry:
    """Carry the data required for speed reward diagnostics telemetry."""

    red_light_speed_reward: float = 0.0
    red_light_profile_reward: float = 0.0
    red_light_stopline_factor: float = 0.0
    red_light_target_speed: float = 0.0
    # Internal curve diagnostic retained outside the stable telemetry payload.
    red_light_preferred_speed: float = 0.0
    red_light_distance_to_stopline: float = float("inf")
    red_light_far_wait_penalty: float = 0.0
    green_light_idle_penalty: float = 0.0
    blocking_profile_distance_m: float = float("inf")
    blocking_profile_obstacle_speed_mps: float = 0.0
    blocking_profile_closing_speed_mps: float = 0.0
    blocking_profile_target_speed_mps: float = 0.0
    blocking_profile_preferred_speed_mps: float = 0.0
    blocking_profile_reward: float = 0.0
    signal_stop_observed: float = 0.0
    signal_stop_quality: float = 0.0
    signal_stop_positive_accel_penalty: float = 0.0
    signal_release_reward: float = 0.0

    def reset(self) -> None:
        """Reset this speed reward diagnostics telemetry to its initial operational state."""
        fresh = type(self)()
        self.__dict__.update(fresh.__dict__)


@dataclass
class SpeedRewardStateTelemetry:
    """Carry the data required for speed reward state telemetry."""

    raw: float = 0.0
    normalized: float = 0.0
    current: float = 0.0
    effective: float = 0.0
    weighted: float = 0.0
    weighted_uncapped: float = 0.0
    blocking_profile_active: bool = False
    diagnostics: SpeedRewardDiagnosticsTelemetry = field(default_factory=SpeedRewardDiagnosticsTelemetry)


@dataclass
class DeltaRewardStateTelemetry:
    """Carry the data required for delta reward state telemetry."""

    raw: float = 0.0
    normalized: float = 0.0
    rolling_average: float = 0.0
    cumulative_average: float = 0.0
    weighted: float = 0.0
    weighted_uncapped: float = 0.0
    penalty_sum: float = 0.0


@dataclass
class SignalStopRewardStateTelemetry:
    """Remember one traffic-signal stop without exposing policy observations."""

    encounter_active: bool = False
    stop_observed: bool = False
    quality_recorded: bool = False
    stop_quality: float = 0.0
    release_motion_seen: bool = False
    approach_motion_seen: bool = False
    motion_qualified_stop_observed: bool = False
    positive_profile_locked: bool = False

    def reset(self) -> None:
        """Reset the current traffic-signal encounter."""
        fresh = type(self)()
        self.__dict__.update(fresh.__dict__)


@dataclass
class RewardStateTelemetry:
    """Carry the data required for reward state telemetry."""

    total: float = 0.0
    sync: SyncRewardStateTelemetry = field(default_factory=SyncRewardStateTelemetry)
    accel: AccelRewardStateTelemetry = field(default_factory=AccelRewardStateTelemetry)
    speed: SpeedRewardStateTelemetry = field(default_factory=SpeedRewardStateTelemetry)
    delta: DeltaRewardStateTelemetry = field(default_factory=DeltaRewardStateTelemetry)
    signal_stop: SignalStopRewardStateTelemetry = field(
        default_factory=SignalStopRewardStateTelemetry
    )

    def reset_step_values(self) -> None:
        """Reset the step values owned by this reward state telemetry."""
        fresh = type(self)()
        self.__dict__.update(fresh.__dict__)


@dataclass
class RewardComponentTelemetry:
    """Carry the data required for reward component telemetry."""

    raw: float | None = None
    normalized: float | None = None
    rolling_average: float | None = None
    cumulative_average: float | None = None
    current: float | None = None
    effective: float | None = None
    weighted: float | None = None
    weighted_uncapped: float | None = None
    diagnostics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> RewardComponentTelemetryChannelPayload:
        """Convert this reward component telemetry's value to dict."""
        data = asdict(self)
        return cast(
            RewardComponentTelemetryChannelPayload,
            {
                key: value
                for key, value in data.items()
                if value is not None and value != {}
            },
        )


@dataclass
class RewardTelemetry:
    """Carry the data required for reward telemetry."""

    total: float = 0.0
    components: dict[str, RewardComponentTelemetry] = field(default_factory=dict)

    @classmethod
    def from_state(cls, state: RewardStateTelemetry) -> RewardTelemetry:
        """Create this reward telemetry's value from state."""
        speed_diagnostics = asdict(state.speed.diagnostics)
        speed_diagnostics["blocking_profile_active"] = float(
            state.speed.blocking_profile_active
        )
        # The historical public payload exposes the brake-limit target only;
        # retaining that exact schema keeps downstream metric ordering stable.
        speed_diagnostics.pop("red_light_preferred_speed")
        return cls(
            total=state.total,
            components={
                "sync": RewardComponentTelemetry(
                    raw=state.sync.raw,
                    normalized=state.sync.normalized,
                    rolling_average=state.sync.rolling_average,
                    cumulative_average=state.sync.cumulative_average,
                    weighted=state.sync.weighted,
                    weighted_uncapped=state.sync.weighted_uncapped,
                    diagnostics={
                        "command": state.sync.command,
                        "actual_acceleration": state.sync.actual_acceleration,
                        "absolute_error": state.sync.absolute_error,
                        "effective_error": state.sync.effective_error,
                        "speed_factor": state.sync.speed_factor,
                        "direction_penalty": state.sync.direction_penalty,
                        "magnitude_penalty": state.sync.magnitude_penalty,
                        "sumo_intervention": state.sync.sumo_intervention,
                    },
                ),
                "accel": RewardComponentTelemetry(
                    raw=state.accel.raw,
                    normalized=state.accel.normalized,
                    rolling_average=state.accel.rolling_average,
                    cumulative_average=state.accel.cumulative_average,
                    weighted=state.accel.weighted,
                    weighted_uncapped=state.accel.weighted_uncapped,
                ),
                "speed": RewardComponentTelemetry(
                    raw=state.speed.raw,
                    normalized=state.speed.normalized,
                    current=state.speed.current,
                    effective=state.speed.effective,
                    weighted=state.speed.weighted,
                    weighted_uncapped=state.speed.weighted_uncapped,
                    diagnostics=speed_diagnostics,
                ),
                "delta": RewardComponentTelemetry(
                    raw=state.delta.raw,
                    normalized=state.delta.normalized,
                    rolling_average=state.delta.rolling_average,
                    cumulative_average=state.delta.cumulative_average,
                    weighted=state.delta.weighted,
                    weighted_uncapped=state.delta.weighted_uncapped,
                ),
            },
        )

    def to_dict(self) -> RewardTelemetryChannelPayload:
        """Convert this reward telemetry's value to dict."""
        return {
            "total": self.total,
            "components": {
                name: component.to_dict()
                for name, component in self.components.items()
            },
        }

    def to_channels(self) -> RewardTelemetryChannels:
        """Convert this reward telemetry's value to channels."""
        return {"reward": self.to_dict()}

    def to_channel_dict(self) -> RewardTelemetryChannelPayload:
        """Convert this reward telemetry's value to channel dict."""
        return self.to_dict()

    def channel_names(self) -> tuple[str, ...]:
        """Return the channel names value owned by this reward telemetry."""
        return REWARD_CHANNEL_NAMES

    def component_names(self) -> tuple[str, ...]:
        """Return the component names value owned by this reward telemetry."""
        return tuple(self.components.keys()) or REWARD_COMPONENT_NAMES

    def to_flat_series(self) -> dict[str, float]:
        """Convert this reward telemetry's value to flat series."""
        values = {"total": float(self.total)}
        for component_name, component in self.components.items():
            component_data = component.to_dict()
            for key, value in component_data.items():
                if key == "diagnostics":
                    continue
                values[f"{component_name}_{key}"] = float(
                    cast(float, value)
                )
            diagnostics = component_data.get("diagnostics", {})
            for key, value in diagnostics.items():
                values[f"{component_name}_{key}"] = float(value)
        return values

    def flat_series_names(self) -> tuple[str, ...]:
        """Return the flat series names value owned by this reward telemetry."""
        return tuple(self.to_flat_series().keys())


__all__ = [
    "REWARD_CHANNEL_NAMES",
    "REWARD_COMPONENT_NAMES",
    "SYNC_ERROR_DEADBAND_MPS2",
    "RewardComponentTelemetry",
    "RewardComponentTelemetryChannelPayload",
    "RewardConfigTelemetry",
    "RewardTelemetryChannelPayload",
    "RewardTelemetryChannels",
    "RewardStateTelemetry",
    "RewardTelemetry",
    "AccelRewardStateTelemetry",
    "DeltaRewardStateTelemetry",
    "SpeedRewardDiagnosticsTelemetry",
    "SpeedRewardStateTelemetry",
    "SyncRewardStateTelemetry",
]
