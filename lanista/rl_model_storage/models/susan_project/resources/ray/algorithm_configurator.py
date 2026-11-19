# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Ray PPO configuration for the Susan model."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime

import torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.torch.heads import TorchFreeLogStdMLPHead
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import UnifiedLogger

from lanista.reinforcengine.rl_technologies.ray.ppo_configuration import (
    wrap_tensorboard_continuity_logger,
)

from ..utils.training.runtime_config import (
    PPO_POLICY_ACTIVATION,
    PPO_POLICY_NET_ARCH,
    FilteredRayTuneTBXLogger,  # noqa: F401 - public metrics integration surface
    build_ray_deployment_algorithm_config,
    build_ray_policy_action_space,
    build_ray_policy_observation,
    build_ray_policy_observation_space,
    build_ray_training_algorithm_config,
    build_ray_tune_logger_classes,
)

from ..utils.training.ppo_config import PPO_POLICY_LOG_STD_INIT


class SusanPPOCatalog(PPOCatalog):
    """Build Ray policy heads with the same initial exploration as SB3."""

    def build_pi_head(self, framework: str) -> Model:
        """Return a free learned log standard deviation initialized to -1."""
        head = super().build_pi_head(framework)
        if framework != "torch":
            return head
        if not isinstance(head, TorchFreeLogStdMLPHead):
            raise TypeError(
                "Susan Ray PPO requires TorchFreeLogStdMLPHead when "
                "free_log_std is enabled."
            )
        with torch.no_grad():
            head.log_std.fill_(PPO_POLICY_LOG_STD_INIT)
        head.clip_log_std = False
        return head


def _EGO_policy_observation_space():
    """Return the ego policy observation space value owned by this algorithm configurator module."""
    return build_ray_policy_observation_space()


def _EGO_policy_action_space():
    """Return the ego policy action space value owned by this algorithm configurator module."""
    return build_ray_policy_action_space()


def _EGO_policy_observation(sensors=None):
    """Return the ego policy observation value owned by this algorithm configurator module."""
    return build_ray_policy_observation(sensors)


def policies():
    """Return the policies value owned by this algorithm configurator module."""
    return {
        "EGO_policy": PolicySpec(
            policy_class=None,
            observation_space=_EGO_policy_observation_space(),
            action_space=_EGO_policy_action_space(),
            config={"observation": _EGO_policy_observation},
        )
    }


def policy_mapping_fn(agent_id, *args, **kwargs):
    """Return the policy mapping fn value owned by this algorithm configurator module."""
    if agent_id.startswith("EGO_agent"):
        return "EGO_policy"
    raise ValueError(f"Unknown agent_id: {agent_id}")


def _ray_model_config():
    """Return the ray model config value owned by this algorithm configurator module."""
    return {
        "fcnet_hiddens": list(PPO_POLICY_NET_ARCH),
        "fcnet_activation": PPO_POLICY_ACTIVATION,
        "vf_share_layers": False,
        "free_log_std": True,
    }


def my_logger_creator(config):
    """Return the my logger creator value owned by this algorithm configurator module."""
    override_logdir = str(os.environ.get("LANISTAI_TENSORBOARD_LOGDIR", "") or "").strip()
    loggers = build_ray_tune_logger_classes()
    if override_logdir:
        os.makedirs(override_logdir, exist_ok=True)
        return wrap_tensorboard_continuity_logger(
            UnifiedLogger(config, override_logdir, loggers=loggers)
        )

    from lanista.locations import get_tensorboard_logs_path

    logdir = str(get_tensorboard_logs_path())
    os.makedirs(logdir, exist_ok=True)
    prefix = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
    full_dir = tempfile.mkdtemp(prefix=prefix, dir=logdir)
    return wrap_tensorboard_continuity_logger(
        UnifiedLogger(config, full_dir, loggers=loggers)
    )


def training_algorithm_configurator(env_id: str):
    """Return the training algorithm configurator value owned by this algorithm configurator
    module.
    """
    policies_dict = policies()
    return build_ray_training_algorithm_config(
        env_id=env_id,
        policies=policies_dict,
        policy_mapping_fn=policy_mapping_fn,
        logger_creator=my_logger_creator,
        model_config_builder=_ray_model_config,
        catalog_class=SusanPPOCatalog,
    )


def deployment_algorithm_configurator(env_id: str):
    """Return the deployment algorithm configurator value owned by this algorithm configurator
    module.
    """
    policies_dict = policies()
    return build_ray_deployment_algorithm_config(
        env_id=env_id,
        policies=policies_dict,
        policy_mapping_fn=policy_mapping_fn,
        model_config_builder=_ray_model_config,
        catalog_class=SusanPPOCatalog,
    )
