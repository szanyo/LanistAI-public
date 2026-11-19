# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed requests crossing the runtime plugin construction boundary."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from lanista.reinforcengine.rl_model.execution_contracts import ExecutionLeaseFactory


@dataclass(frozen=True, slots=True)
class RuntimeSelectionSnapshot:
    """Immutable session/checkpoint selection for one runtime operation."""

    session_name: str | None
    checkpoint_name: str | None


RuntimeSelectionProvider = Callable[[], RuntimeSelectionSnapshot]


@dataclass(frozen=True)
class RuntimeInitializationRequest:
    """Typed dependencies required to construct a managed model runtime plugin."""

    parent_logger_name: str
    child_logger_name: str
    child_logger_level: int | str
    execution_lease_factory: ExecutionLeaseFactory
    runtime_selection_provider: RuntimeSelectionProvider | None = None

    def as_keyword_arguments(self) -> dict[str, object]:
        """Return the explicit plugin constructor arguments."""

        arguments: dict[str, object] = {
            "parent_logger_name": self.parent_logger_name,
            "child_logger_name": self.child_logger_name,
            "child_logger_level": self.child_logger_level,
            "execution_lease_factory": self.execution_lease_factory,
        }
        if self.runtime_selection_provider is not None:
            arguments["runtime_selection_provider"] = self.runtime_selection_provider
        return arguments
