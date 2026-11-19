# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Handle lifecycle callbacks for training control."""

from typing import Protocol

from stable_baselines3.common.callbacks import BaseCallback


class _RuntimeControl(Protocol):
    """Pause and stop checks required by the SB3 learn-loop adapter."""

    def check_pause(self) -> None:
        """Block while execution is paused."""

    def check_stop(self) -> None:
        """Raise when execution has been stopped."""


class StableBaselinesTrainingControlCallback(BaseCallback):
    """Bridge LanistAI pause/stop controls into the SB3 learn loop."""

    def __init__(
        self,
        rl_model_wrapper: _RuntimeControl,
        verbose: int = 0,
    ) -> None:
        """Initialize the stable baselines training control callback and its owned state."""
        super().__init__(verbose)
        self._rl_model_wrapper = rl_model_wrapper

    def _on_step(self) -> bool:
        """Handle the step event for this stable baselines training control callback."""
        self._rl_model_wrapper.check_pause()
        self._rl_model_wrapper.check_stop()
        return True
