# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Stable PPO configuration entry points shared by simple SB3 plugins."""

from __future__ import annotations


def training_algorithm_configuration(env_id: str) -> dict[str, object]:
    """Return the Eve/Jasmine SB3 training constructor values."""

    return {
        "env_id": env_id,
        "policy": "MlpPolicy",
        "clip_range_vf": 150.0,
    }


def deployment_algorithm_configuration(env_id: str) -> dict[str, object]:
    """Return the Eve/Jasmine SB3 deployment constructor values."""

    return {"env_id": env_id, "policy": "MlpPolicy"}
