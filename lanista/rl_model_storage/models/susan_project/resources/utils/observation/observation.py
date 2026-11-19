# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan policy observation mappings and Gymnasium spaces."""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from typing import TypedDict

from .observation_config import (
    OBSERVATION_DESCRIPTOR,
    OBSERVATION_KEYS as POLICY_OBSERVATION_KEYS,
    observation_key_bounds,
    observation_key_length,
)

OBSERVATION_KEYS = OBSERVATION_DESCRIPTOR.keys

OBSERVATION_KEY_LENGTHS = OBSERVATION_DESCRIPTOR.key_lengths
OBSERVATION_KEY_BOUNDS = OBSERVATION_DESCRIPTOR.key_bounds
OBSERVATION_SIZE = OBSERVATION_DESCRIPTOR.size
OBSERVATION_DTYPE = np.float32


class SusanObservationChannels(TypedDict):
    """Describe the typed mapping for susan observation channels."""

    observation: dict[str, np.ndarray]
    telemetry: dict[str, np.ndarray]


def observation_dict(sensors=None):
    """Return the observation dict value owned by this observation module."""
    observation = {}
    if sensors is None:
        for key in OBSERVATION_KEYS:
            observation[key] = np.full((observation_key_length(key),), 0.0, dtype=OBSERVATION_DTYPE)
        return observation

    for key in OBSERVATION_KEYS:
        length = observation_key_length(key)
        if hasattr(sensors, "get"):
            value = sensors.get(key)
        else:
            value = getattr(sensors, key, None)
        if value is None:
            observation[key] = np.full((length,), 0.0, dtype=OBSERVATION_DTYPE)
        else:
            arr = np.asarray(value, dtype=OBSERVATION_DTYPE).reshape(-1)
            if arr.size == length:
                observation[key] = arr.reshape((length,))
            elif arr.size == 1:
                observation[key] = np.full((length,), float(arr[-1]), dtype=OBSERVATION_DTYPE)
            else:
                resized = np.zeros((length,), dtype=OBSERVATION_DTYPE)
                copy_len = min(length, arr.size)
                resized[-copy_len:] = arr[-copy_len:]
                observation[key] = resized
    return observation


def flatten_observation(sensors=None):
    """Return the flatten observation value owned by this observation module."""
    observation = observation_dict(sensors)
    flattened_parts = [np.asarray(observation[key], dtype=OBSERVATION_DTYPE).reshape(-1) for key in OBSERVATION_KEYS]
    if not flattened_parts:
        return np.zeros((OBSERVATION_SIZE,), dtype=OBSERVATION_DTYPE)
    return np.concatenate(flattened_parts).astype(OBSERVATION_DTYPE, copy=False)


def observation_channels(sensors=None) -> SusanObservationChannels:
    """Return the observation channels value owned by this observation module."""
    observation = observation_dict(sensors)
    return {
        "observation": {key: observation[key] for key in POLICY_OBSERVATION_KEYS},
        "telemetry": observation,
    }


def observation_space_bounds(keys=OBSERVATION_KEYS) -> tuple[np.ndarray, np.ndarray]:
    """Return the observation space bounds value owned by this observation module."""
    lows = []
    highs = []
    for key in keys:
        low, high = observation_key_bounds(key)
        length = observation_key_length(key)
        lows.append(np.full((length,), low, dtype=OBSERVATION_DTYPE))
        highs.append(np.full((length,), high, dtype=OBSERVATION_DTYPE))
    return np.concatenate(lows), np.concatenate(highs)


def observation_space_dict() -> gym.spaces.Dict:
    """Return the observation space dict value owned by this observation module."""
    return gym.spaces.Dict({
        key: gym.spaces.Box(
            low=np.full((observation_key_length(key),), observation_key_bounds(key)[0], dtype=OBSERVATION_DTYPE),
            high=np.full((observation_key_length(key),), observation_key_bounds(key)[1], dtype=OBSERVATION_DTYPE),
            shape=(observation_key_length(key),),
            dtype=OBSERVATION_DTYPE,
        )
        for key in OBSERVATION_KEYS
    })


def observation_space_box() -> gym.spaces.Box:
    """Return the observation space box value owned by this observation module."""
    low, high = observation_space_bounds()
    return gym.spaces.Box(low=low, high=high, shape=(OBSERVATION_SIZE,), dtype=OBSERVATION_DTYPE)
