# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Interval-based simulation scheduling."""
import time


class Scheduler:
    """Definition of the scheduler class for scheduling the execution of a code block.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.

    """
    def __init__(self, scheduler_interval):
        """
        Constructor for scheduler class.
        """
        self._scheduler_interval = scheduler_interval
        self._time = 0
        self._max_overdue = 0

    def reset(self):
        """
        Reset the scheduler.
        """
        self._time = time.time()

    def update(self):
        """
        Update the scheduler.
        """
        self._time = time.time()

    def delay(self):
        """
        Delay the execution of the code block.
        """
        current_time = time.time()
        remaining_time = self._scheduler_interval - (current_time - self._time)
        if remaining_time > self._max_overdue:
            self._max_overdue = remaining_time
        if remaining_time > 0:
            time.sleep(remaining_time)
        self._time = current_time
