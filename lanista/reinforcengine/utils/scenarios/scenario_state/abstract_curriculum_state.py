# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Represent runtime state for abstract curriculum."""

from abc import ABC, abstractmethod


class AbstractCurriculumState(ABC):
    """Globális curriculum state backend interfész."""

    @abstractmethod
    def get(self):
        """Return the operation owned by this abstract curriculum state."""
        raise NotImplementedError

    @abstractmethod
    def set(self, value):
        """Set the operation owned by this abstract curriculum state."""
        raise NotImplementedError