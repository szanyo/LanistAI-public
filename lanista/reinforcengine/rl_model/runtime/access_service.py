# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read-only access to the runtime currently published in the active slot."""

from __future__ import annotations

from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime_slot_store import RuntimeSlotStore


class RuntimeAccessService:
    """Keep raw runtime reads behind one typed, read-only application boundary."""

    def __init__(
        self,
        slot_store: RuntimeSlotStore,
    ) -> None:
        """Initialize the runtime access service and its owned state."""
        self._slot_store = slot_store

    def get_active_runtime(self) -> BaseModelRuntime | None:
        """Return the runtime from one immutable active-slot snapshot."""

        return self._slot_store.snapshot().model
