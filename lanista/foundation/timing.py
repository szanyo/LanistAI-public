# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Monotonic elapsed-time measurement."""

from __future__ import annotations

import time
from typing import Self


class Stopper:
    """Compatibility-sized timer backed by a monotonic clock."""

    def __init__(self) -> None:
        """Initialize the stopper and its owned state."""
        self._start_time = time.monotonic()
        self._all = 0.0

    def __enter__(self) -> Self:
        """Enter this stopper's managed context."""
        self.startTime()
        return self

    def __exit__(self, *_args: object) -> None:
        """Release owned resources when leaving this stopper's managed context."""
        return None

    def startTime(self) -> None:
        """Start the time owned by this stopper."""
        self._start_time = time.monotonic()

    def getElapsedTime(self) -> float:
        """Return the elapsed time exposed by this stopper."""
        return time.monotonic() - self._start_time

    def setAllNum(self, number: float) -> None:
        """Set the all num owned by this stopper."""
        self._all = number

    def getRemainingTime(self, completed: float) -> float:
        """Return the remaining time exposed by this stopper."""
        if completed == 0:
            return 0.0
        elapsed = self.getElapsedTime()
        return elapsed * (self._all / completed) - elapsed

    def getPredictedTotalTime(self, completed: float) -> float:
        """Return the predicted total time exposed by this stopper."""
        if completed == 0:
            return 0.0
        return self.getElapsedTime() * (self._all / completed)

    @staticmethod
    def formatTime(duration: float) -> tuple[int, int, float]:
        """Return the format time value owned by this stopper."""
        hours = int(duration / 3600)
        minutes = int((duration - hours * 3600) / 60)
        seconds = duration - hours * 3600 - minutes * 60
        return hours, minutes, seconds

