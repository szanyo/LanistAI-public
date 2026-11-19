# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt framework-neutral model runtime events to Textual messages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from textual.message import Message

from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.events import (
    ModelChangedEvent,
    ModelLoadProgressEvent,
    RuntimeEvent,
)


@dataclass
class ModelChangedMessage(Message):
    """Carry the model changed message interface message."""

    model: BaseModelRuntime | None
    state: RLModelState


@dataclass
class ModelLoadProgressMessage(Message):
    """Carry the model load progress message interface message."""

    activity: str
    step: str
    progress: float
    total: float


class _MessageTarget(Protocol):
    """Define the typed boundary for message target."""

    def post_message(self, message: Message) -> object:
        """Return the post message value provided by this message target."""
        ...


class TextualModelRuntimeEventPublisher:
    """Translate domain events at the GUI boundary and post them to Textual."""

    def __init__(self, target: _MessageTarget) -> None:
        """Initialize the textual model runtime event publisher and its owned state."""
        self._target = target

    def publish(self, event: RuntimeEvent) -> None:
        """Publish the operation owned by this textual model runtime event publisher."""
        if isinstance(event, ModelChangedEvent):
            self._target.post_message(
                ModelChangedMessage(model=event.model, state=event.state)
            )
            return
        if isinstance(event, ModelLoadProgressEvent):
            self._target.post_message(
                ModelLoadProgressMessage(
                    activity=event.activity,
                    step=event.step,
                    progress=event.progress,
                    total=event.total,
                )
            )
            return
        raise TypeError(f"Unsupported runtime event: {type(event).__name__}")
