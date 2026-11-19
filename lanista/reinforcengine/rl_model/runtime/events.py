# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral events emitted by model runtime application workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.model import BaseModelRuntime


@dataclass(frozen=True)
class ModelChangedEvent:
    """Describe an active-runtime slot transition without a GUI dependency."""

    model: BaseModelRuntime | None
    state: RLModelState = RLModelState.IDLE


@dataclass(frozen=True)
class ModelLoadProgressEvent:
    """Report one monotonic step of candidate runtime loading."""

    activity: str
    step: str
    progress: float
    total: float = 100.0


type RuntimeEvent = ModelChangedEvent | ModelLoadProgressEvent


class RuntimeEventPublisher(Protocol):
    """Publish runtime events to an adapter owned by the calling layer."""

    def publish(self, event: RuntimeEvent) -> None:
        """Publish the operation owned by this runtime event publisher."""
        ...
