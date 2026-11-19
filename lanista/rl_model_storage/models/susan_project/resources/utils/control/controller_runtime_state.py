# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Explicit mutable state owned by Susan's simulation controller."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ControlCommandState:
    """Commands at the policy/controller boundary for one interval."""

    policy_action: float = 0.0


@dataclass
class TrafficSignalRuntimeState:
    """Previous signal context needed for restricted-area detection."""

    previous_intersection_type_code: Optional[int] = None
    previous_traffic_light_state_code: Optional[int] = None
    red_light_overdrive_this_step: bool = False


@dataclass
class ControllerRuntimeState:
    """Represent runtime state for controller runtime."""

    command: ControlCommandState = field(default_factory=ControlCommandState)
    traffic_signal: TrafficSignalRuntimeState = field(default_factory=TrafficSignalRuntimeState)

    def reset_episode(self) -> None:
        """Reset the episode owned by this controller runtime state."""
        self.command = ControlCommandState()
        self.traffic_signal = TrafficSignalRuntimeState()
