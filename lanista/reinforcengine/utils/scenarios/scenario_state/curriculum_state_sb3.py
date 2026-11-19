# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Track curriculum state for Stable-Baselines3 training."""

import json
from multiprocessing import Array, Lock

from lanista.reinforcengine.utils.scenarios.scenario_state.abstract_curriculum_state import AbstractCurriculumState

_MAX_STATE_BYTES = 8192


class StableBaselinesCurriculumState(AbstractCurriculumState):
    """Local curriculum state backend for Stable-Baselines3 environments."""

    def __init__(self):
        """Initialize the stable baselines curriculum state and its owned state."""
        self._current = Array("b", _MAX_STATE_BYTES)
        self._lock: Lock = Lock()

    def get(self):
        """Return the get value owned by this stable baselines curriculum state."""
        with self._lock:
            raw = bytes(self._current[:]).split(b"\0", 1)[0]
        if not raw:
            return None
        value = json.loads(raw.decode("utf-8"))
        return dict(value) if isinstance(value, dict) else value

    def set(self, value):
        """Set the operation owned by this stable baselines curriculum state."""
        payload = b""
        if value is not None:
            payload = json.dumps(value, sort_keys=True).encode("utf-8")
            if len(payload) >= _MAX_STATE_BYTES:
                raise ValueError(f"Curriculum state exceeds {_MAX_STATE_BYTES} bytes.")
        with self._lock:
            self._current[:] = b"\0" * _MAX_STATE_BYTES
            if payload:
                self._current[:len(payload)] = payload
