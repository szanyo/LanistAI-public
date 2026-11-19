# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project shared training-state values without framework dependencies."""

from __future__ import annotations

from typing import Final

SCENARIO_ACTIVE_METRIC: Final = "active"


def active_scenario_number(scenario_index: object) -> float:
    """Return a one-based scenario number or zero for invalid state."""

    if type(scenario_index) is not int or scenario_index < 0:
        return 0.0
    return float(scenario_index + 1)
