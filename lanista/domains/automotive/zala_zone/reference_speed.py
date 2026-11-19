# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Reference-speed policy for the ZalaZONE route geometry."""

from collections.abc import Callable

_REFERENCE_SPEED_RULES_KMH: tuple[tuple[Callable[[float, float], bool], int], ...] = (
    (lambda x_m, y_m: y_m > 356 and x_m > 153, 15),
    (lambda x_m, y_m: y_m > 356 and x_m > 110, 28),
    (lambda _x_m, y_m: y_m > 380, 15),
    (lambda _x_m, y_m: y_m > 365, 15),
    (lambda x_m, _y_m: x_m < 72, 10),
    (lambda x_m, _y_m: x_m < 120, 30),
    (lambda _x_m, y_m: y_m > 342, 15),
    (lambda _x_m, y_m: y_m > 320, 18),
    (lambda x_m, y_m: y_m > 291 and x_m < 183, 15),
    (lambda x_m, y_m: y_m > 291 and x_m < 210, 20),
    (lambda _x_m, y_m: y_m > 291, 10),
    (lambda x_m, y_m: y_m > 235 and x_m > 195, 20),
    (lambda x_m, y_m: 235 > y_m > 206 and x_m > 195, 10),
    (lambda x_m, y_m: y_m < 235 and x_m > 223, 10),
    (lambda x_m, y_m: y_m < 235 and x_m > 192, 20),
    (lambda _x_m, y_m: y_m < 212, 15),
    (lambda x_m, y_m: y_m > 212 and x_m < 195, 30),
)


def zala_zone_reference_speed_mps(ego_x_m: float, ego_y_m: float) -> float:
    """Return the first matching route-zone speed limit in metres per second.

    The order is an invariant because several geometric regions overlap.
    """

    for matches_zone, speed_kmh in _REFERENCE_SPEED_RULES_KMH:
        if matches_zone(ego_x_m, ego_y_m):
            return speed_kmh / 3.6
    return 10 / 3.6
