# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Reference-generator implementations for longitudinal control."""
import numpy as np
from numpy.typing import NDArray


def generator_np(r: float, d: float, uK: float, v: float) -> NDArray[np.float64]:
    """
    Numpy optimized version of the generator function.
    This is slower than the tuple version.
    The np.array() creation is costly in this case.
    np.clip() is also costly.

    Numpy is not the best choice for small amount of data.

    :param r: reference
    :param d: delta
    :param uK:
    :param v:
    :return: np.array([ref, delta])
    """
    umax = 1  # 2 volt
    umin = -2.5  # -5 volt
    K = 20  # 1/Ts

    satx = umax / K  # max v_diff
    satn = umin / K  # min v_diff

    ref = np.clip(r - v, satn, satx) + v  # max sebességet ne lépje túl

    delta = np.clip(d, umin - uK, umax - uK)  # eredeti constrain
    delta = np.clip(delta, umin, umax)  # ne lehessen delta túl nagy vagy túl kicsi

    kimenet = np.array([ref, delta])

    return kimenet


def generator_tuple(r: float, d: float, uK: float, v: float) -> tuple[float, float]:
    """
    Highly optimized version of the generator function.
    This is 8.77714367059032x time faster than the numpy version.

    :param r: reference
    :param d: delta
    :param uK:
    :param v:
    :return: tuple(ref, delta)
    """
    umax = 1  # 2 volt
    umin = -2.5  # -5 volt
    K = 20  # 1/Ts

    satx = umax / K  # max v_diff
    satn = umin / K  # min v_diff

    ref = max(satn, min(satx, r - v)) + v  # max sebességet ne lépje túl

    delta = max(umin, min(umax, max(umin - uK, min(umax - uK, d))))  # eredeti constrain

    return ref, delta
