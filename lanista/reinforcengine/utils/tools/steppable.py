# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Abstract contract for simulation-step participants."""
from abc import ABC, abstractmethod

class Steppable(ABC):
    """Definition of the steppable class for defining a step in a simulation. Abstract class.

    This module is part of the sb3 project that set as a goal porting an existing autonomous vehicle control
    system into python environment, scaling it up and making it more efficient.

    """
    def __init__(self):
        """
        Constructor for steppable class.
        """
        pass

    @abstractmethod
    def next(self):
        """
        Abstract method for the next step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        """
        Abstract method for resetting object.
        """
        pass

    @abstractmethod
    def finished(self):
        """
        Abstract method for checking if all steps are done.
        """
        pass
