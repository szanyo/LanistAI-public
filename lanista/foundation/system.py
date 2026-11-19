# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Host resource queries used by framework configuration."""

from __future__ import annotations

import multiprocessing
import os

import psutil


def get_cpu_count() -> int:
    """Return the cpu count exposed by this system module."""
    return multiprocessing.cpu_count()


def get_core_count() -> int:
    """Return the core count exposed by this system module."""
    return psutil.cpu_count(logical=False) or get_cpu_count()


def get_thread_count() -> int:
    """Return the thread count exposed by this system module."""
    return os.cpu_count() or get_cpu_count()

