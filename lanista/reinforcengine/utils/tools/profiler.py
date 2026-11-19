# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Execution profiling utilities."""

import cProfile
import io
import os
import pstats
from time import localtime, strftime

DEFAULT_DIR = os.path.sep.join(("data", "profiling"))
"""str: Default directory for saving profiling results."""


class Profiler:
    """Definition of the profiler class for profiling the execution of a code block.
    Shows the cumulative time spent within each function.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.

    Attributes:
    ----------

    """
    def __init__(self, name="default", group=""):
        """
        Constructor for profiler class.
        :param name:
        :param group:
        """
        self._name = name
        self._group = group
        self._pr = None
        self._ps = None
        self._s = None

    def __enter__(self):
        """
        Context manager enter method.
        """
        self._pr = cProfile.Profile()
        self._pr.enable()
        return self

    def __exit__(self, *exc):
        """
        Context manager exit method.
        :param exc:
        """
        self._pr.disable()
        self._s = io.StringIO()

        # Sort descending order of cumulative time.
        self._ps = pstats.Stats(self._pr, stream=self._s).sort_stats("cumulative")

    def display(self):
        """
        Display the profiling results.
        """
        self._ps.print_stats()
        print(self._s.getvalue())

    def save(self):
        """
        Save the profiling results into a file.
        """
        date = strftime("%Y-%m-%d", localtime())
        time = strftime("%H-%M-%S", localtime())

        RELATIVE_DIR = os.path.join(DEFAULT_DIR, self._group, f"profile_{self._name}_{date}_{time}.log")
        os.makedirs(os.path.join(DEFAULT_DIR, self._group), exist_ok=True)
        with open(RELATIVE_DIR, "w") as f:
            f.write(self._s.getvalue())
