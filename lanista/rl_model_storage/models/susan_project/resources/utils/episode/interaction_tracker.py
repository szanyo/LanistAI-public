# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Episode-local actor counts for the selected obstacle reward profile."""

from dataclasses import dataclass, field
from typing import Literal

type InteractionKind = Literal["lead", "pedestrian"]


@dataclass
class EpisodeInteractionTracker:
    """Count selected obstacle interactions without exposing actor IDs to policy."""

    _lead_ids: set[str] = field(default_factory=set)
    _pedestrian_ids: set[str] = field(default_factory=set)
    _steps: int = 0
    _lead_steps: int = 0
    _pedestrian_steps: int = 0

    def record_step(self, kind: InteractionKind | None, actor_id: str | None) -> None:
        """Record one policy-controlled step and its selected blocker, if any."""

        self._steps += 1
        if kind == "lead":
            self._lead_steps += 1
            if actor_id:
                self._lead_ids.add(actor_id)
        elif kind == "pedestrian":
            self._pedestrian_steps += 1
            if actor_id:
                self._pedestrian_ids.add(actor_id)

    def reset(self) -> None:
        """Start the next episode without carrying actor identities forward."""

        self._lead_ids.clear()
        self._pedestrian_ids.clear()
        self._steps = 0
        self._lead_steps = 0
        self._pedestrian_steps = 0

    def summary(self) -> dict[str, float]:
        """Return per-episode counts and step fractions, including empty episodes."""

        denominator = max(1, self._steps)
        return {
            "unique_followed_vehicles": float(len(self._lead_ids)),
            "unique_blocking_pedestrians": float(len(self._pedestrian_ids)),
            "lead_interaction_step_fraction": self._lead_steps / denominator,
            "pedestrian_interaction_step_fraction": self._pedestrian_steps / denominator,
            "any_interaction_step_fraction": (
                self._lead_steps + self._pedestrian_steps
            ) / denominator,
        }
