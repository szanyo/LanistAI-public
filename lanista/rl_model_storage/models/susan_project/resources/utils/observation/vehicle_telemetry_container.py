# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Bounded history containers for policy observations and telemetry."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .vehicle_telemetry import OBSERVATION_KEYS, VehicleObservationTelemetry, VehicleTelemetry
from .observation_config import (
    OBSERVATION_HISTORY_LENGTH,
    OBSERVATION_KEYS as POLICY_OBSERVATION_KEYS,
    observation_key_length,
)


class VehicleObservationTelemetryContainer:
    """Store and expose vehicle observation telemetry values."""

    def __init__(self, maxlen: int = OBSERVATION_HISTORY_LENGTH):
        """Initialize the vehicle observation telemetry container and its owned state."""
        self._items = deque(maxlen=maxlen)
        self.reset()

    def reset(self) -> None:
        """Reset this vehicle observation telemetry container to its initial operational state."""
        self._items.clear()
        for _ in range(self._items.maxlen or OBSERVATION_HISTORY_LENGTH):
            self._items.append(VehicleObservationTelemetry.zero())

    def append(self, observation: VehicleObservationTelemetry | dict[str, float]) -> None:
        """Append the operation owned by this vehicle observation telemetry container."""
        if isinstance(observation, dict):
            observation = VehicleObservationTelemetry.from_dict(observation)
        self._items.append(observation)

    def to_policy_time_window_dict(self) -> dict[str, np.ndarray]:
        """Convert this vehicle observation telemetry container's value to policy time window dict.
        """
        if not self._items:
            return {
                key: np.zeros((observation_key_length(key),), dtype=np.float64)
                for key in POLICY_OBSERVATION_KEYS
            }
        return {
            key: (
                np.array([getattr(observation, key) for observation in self._items], dtype=np.float64)
                if observation_key_length(key) > 1
                else np.array([getattr(self._items[-1], key)], dtype=np.float64)
            )
            for key in POLICY_OBSERVATION_KEYS
        }

    def to_telemetry_time_window_dict(self) -> dict[str, np.ndarray]:
        """Convert this vehicle observation telemetry container's value to telemetry time window
        dict.
        """
        if not self._items:
            return {
                key: np.zeros((observation_key_length(key),), dtype=np.float64)
                for key in OBSERVATION_KEYS
            }
        return {
            key: (
                np.array([getattr(observation, key) for observation in self._items], dtype=np.float64)
                if observation_key_length(key) > 1
                else np.array([getattr(self._items[-1], key)], dtype=np.float64)
            )
            for key in OBSERVATION_KEYS
        }

    def to_channel_time_window_dict(self) -> dict[str, dict[str, np.ndarray]]:
        """Convert this vehicle observation telemetry container's value to channel time window
        dict.
        """
        return {
            "observation": self.to_policy_time_window_dict(),
            "telemetry": self.to_telemetry_time_window_dict(),
        }

    def latest(self) -> VehicleObservationTelemetry | None:
        """Return the latest value owned by this vehicle observation telemetry container."""
        if not self._items:
            return None
        return self._items[-1]

    def to_time_window_dict(self) -> dict[str, np.ndarray]:
        """Convert this vehicle observation telemetry container's value to time window dict."""
        return self.to_telemetry_time_window_dict()

    def __len__(self) -> int:
        """Return the number of values exposed by this vehicle observation telemetry container."""
        return len(self._items)

    def __iter__(self):
        """Iterate over the values exposed by this vehicle observation telemetry container."""
        return iter(self._items)


class VehicleTelemetryContainer:
    """Store and expose vehicle telemetry values."""

    def __init__(self, items: Iterable[VehicleTelemetry] | None = None, maxlen: int | None = None):
        """Initialize the vehicle telemetry container and its owned state."""
        self._items = deque(items or (), maxlen=maxlen)

    def append(self, telemetry: VehicleTelemetry) -> None:
        """Append the operation owned by this vehicle telemetry container."""
        self._items.append(telemetry)

    def clear(self) -> None:
        """Handle the clear operation owned by this vehicle telemetry container."""
        self._items.clear()

    def latest(self) -> VehicleTelemetry | None:
        """Return the latest value owned by this vehicle telemetry container."""
        if not self._items:
            return None
        return self._items[-1]

    def to_rows(self) -> list[dict[str, float]]:
        """Convert this vehicle telemetry container's value to rows."""
        return [telemetry.to_row() for telemetry in self._items]

    def to_channel_rows(self) -> list[dict[str, float]]:
        """Convert this vehicle telemetry container's value to channel rows."""
        return [telemetry.to_channel_row() for telemetry in self._items]

    def latest_dict(self) -> dict[str, float] | None:
        """Return the latest dict value owned by this vehicle telemetry container."""
        latest = self.latest()
        if latest is None:
            return None
        return latest.to_row()

    def latest_channel_dict(self) -> dict[str, float] | None:
        """Return the latest channel dict value owned by this vehicle telemetry container."""
        latest = self.latest()
        if latest is None:
            return None
        return latest.to_channel_row()

    def __len__(self) -> int:
        """Return the number of values exposed by this vehicle telemetry container."""
        return len(self._items)

    def __iter__(self):
        """Iterate over the values exposed by this vehicle telemetry container."""
        return iter(self._items)
