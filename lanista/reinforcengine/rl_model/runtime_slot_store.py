# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Thread-safe ownership boundary for the active runtime slot."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock

from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime

__all__ = ["RuntimeSlotState", "RuntimeSlotStore"]


@dataclass(frozen=True, slots=True)
class RuntimeSlotState:
    """Complete active-slot state guarded by one monotonically increasing revision."""

    model: BaseModelRuntime | None
    manifest: ModelManifest | None
    framework_uuid: str | None
    session_name: str | None
    checkpoint_name: str | None
    last_successful_model_name: str | None
    last_successful_framework_uuid: str | None
    revision: int
    last_successful_session_name: str | None = None
    last_successful_checkpoint_name: str | None = None


def _empty_runtime_slot_state() -> RuntimeSlotState:
    """Return the empty runtime slot state value owned by this runtime slot store module."""
    return RuntimeSlotState(
        model=None,
        manifest=None,
        framework_uuid=None,
        session_name=None,
        checkpoint_name=None,
        last_successful_model_name=None,
        last_successful_framework_uuid=None,
        last_successful_session_name=None,
        last_successful_checkpoint_name=None,
        revision=0,
    )


class RuntimeSlotStore:
    """Publish immutable runtime-slot snapshots under one re-entrant lock."""

    def __init__(
        self,
        initial_state: RuntimeSlotState | None = None,
    ) -> None:
        """Initialize the runtime slot store and its owned state."""
        self._lock = RLock()
        self._state = (
            initial_state
            if initial_state is not None
            else _empty_runtime_slot_state()
        )

    def snapshot(self) -> RuntimeSlotState:
        """Return the currently published immutable slot snapshot."""

        with self._lock:
            return self._state

    def publish(self, state: RuntimeSlotState) -> None:
        """Atomically replace the complete slot snapshot."""

        with self._lock:
            self._state = state

    def update(
        self,
        transition: Callable[[RuntimeSlotState], RuntimeSlotState],
    ) -> RuntimeSlotState:
        """Apply one in-memory state transition without losing concurrent updates."""

        with self._lock:
            updated = transition(self._state)
            self._state = updated
            return updated

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Hold the slot monitor across a multi-step transaction."""

        with self._lock:
            yield
