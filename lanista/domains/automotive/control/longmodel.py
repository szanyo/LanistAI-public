# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Discrete longitudinal vehicle-model helpers."""

import numpy as np
from numpy.typing import NDArray


def longmodel_np(
    acc: float,
    speed: float,
    station: float,
) -> NDArray[np.float64]:
    """
    Numpy optimized version of the longmodel function.
    This is slower than the tuple version.
    The np.array() creation is costly in this case.

    :param acc: Acceleration
    :param speed: Current speed
    :param station: Current position
    :return: Numpy array of the new speed and station
    """
    speed = max(0, speed)

    Ts = 0.05  # Simulation step in seconds.

    speedn = max(0, speed + Ts * acc)
    stationn = station + Ts * speedn
    kimenet = np.array([speedn, stationn])
    return kimenet


def longmodel_tuple(
    acc: float,
    speed: float,
    station: float,
) -> tuple[float, float]:
    """
    Highly optimized version of the longmodel function.
    This is 1.8222962538247742x time faster than the numpy version.

    :param acc: Acceleration
    :param speed: Current speed
    :param station: Current position
    :return: Tuple of the new speed and station
    """
    speed = max(0, speed)  # Longitudinal speed cannot be negative.

    Ts = 0.05  # Simulation step in seconds.

    speedn = max(0, speed + Ts * acc)
    stationn = Ts * speedn
    return speedn, stationn
