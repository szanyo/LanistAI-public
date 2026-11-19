# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Represent runtime state for abstract scenario."""

from abc import ABC, abstractmethod

class AbstractScenarioState(ABC):
    """Globális state backend interfész."""

    @abstractmethod
    def get(self) -> int:
        """Return the get value provided by this abstract scenario state."""
        raise NotImplementedError

    @abstractmethod
    def set(self, index: int):
        """Set the operation owned by this abstract scenario state."""
        raise NotImplementedError
