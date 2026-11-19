# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""EGO vehicle runtime state used by simulation controllers."""

from lanista.domains.automotive.control.generator import generator_tuple
from lanista.domains.automotive.control.longmodel import longmodel_tuple
from lanista.foundation.logging import LoggerInterface

Y_POS = 100
ORIENTATION = 100


class EGO(LoggerInterface):
    """Definition of the EGO vehicle state machine.

    This class is responsible for the EGO vehicle's states between the simulation steps.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.

    """
    _last_acceleration: float

    def __init__(self, **kwargs: object) -> None:
        """
        Constructor of the EGO vehicle state machine.
        """
        LoggerInterface.__init__(self, **kwargs)
        self.prev_speed = 0.0
        self.prev_station = 0.0
        self.prev_uK = 0.0
        self.last_ut = 0.0

        self._cum_avg_speed = 0.0
        self._cum_avg_num = 0
        self._max_speed = 0.0


    def calculate(
        self,
        delta: float,
        refSpeed: float,
    ) -> tuple[float, float, float, float]:
        """
        Calculate the next step for the EGO vehicle.
        This function is called in every simulation step.
        This was an original function in the SUMO project named 'EGO_Vehicle'.

        :param delta:
        :param refSpeed:
        :return: uK, self.prev_speed, u1, self.prev_station
        """

        uK = self.prev_uK * 20
        ref, delta = generator_tuple(refSpeed, delta, uK, self.prev_speed)
        u1 = delta + uK
        self.prev_uK = ref - self.prev_speed
        self.prev_speed, self.prev_station = longmodel_tuple(u1, self.prev_speed, self.prev_station)

        self._cum_avg_speed += self.prev_speed
        self._cum_avg_num += 1
        self._max_speed = max(self._max_speed, self.prev_speed)

        return uK, self.prev_speed, u1, self.prev_station

    @property
    def AverageSpeed(self) -> float:
        """Return the average speed exposed by this ego."""
        return self._cum_avg_speed / self._cum_avg_num

    @property
    def MaxSpeed(self) -> float:
        """Return the max speed exposed by this ego."""
        return self._max_speed

    @property
    def LastAcceleration(self) -> float:
        """Return the last acceleration exposed by this ego."""
        return self._last_acceleration
