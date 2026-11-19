# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Configure algorithm."""

from __future__ import annotations

from ..utils.training.runtime_config import build_sb3_ppo_constructor_config


def training_algorithm_configurator(env_id: str):
    """Return the training algorithm configurator value owned by this algorithm configurator
    module.
    """
    return build_sb3_ppo_constructor_config(env_id)


def deployment_algorithm_configurator(env_id: str):
    """Return the deployment algorithm configurator value owned by this algorithm configurator
    module.
    """
    return build_sb3_ppo_constructor_config(env_id)
