# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Running modes shared by reinforcement-learning environments."""

from __future__ import annotations

from enum import Enum, auto

from lanista.reinforcengine.application_errors import ApplicationValidationError


class RunningMode(Enum):
    """Select the lifecycle behavior of a reinforcement-learning environment."""

    TRAINING = auto()
    TESTING = auto()
    DEPLOYMENT = auto()


class UnsupportedRunningModeError(ApplicationValidationError):
    """A configured running mode has no implementation at this boundary."""

    def __init__(
        self,
        unsupported_mode: object,
        supported_modes: tuple[RunningMode, ...],
    ) -> None:
        """Initialize the unsupported running mode error and its owned state."""
        self.unsupported_mode = unsupported_mode
        self.supported_modes = supported_modes
        mode_label = (
            unsupported_mode.name
            if isinstance(unsupported_mode, RunningMode)
            else repr(unsupported_mode)
        )
        supported_labels = ", ".join(mode.name for mode in supported_modes)
        super().__init__(
            f"Running mode {mode_label} is unsupported; "
            f"supported modes: {supported_labels}."
        )


def require_supported_running_mode(
    mode: object,
    supported_modes: tuple[RunningMode, ...],
) -> RunningMode:
    """Return a supported mode or fail before environment state is mutated."""

    if isinstance(mode, RunningMode) and mode in supported_modes:
        return mode
    raise UnsupportedRunningModeError(mode, supported_modes)
