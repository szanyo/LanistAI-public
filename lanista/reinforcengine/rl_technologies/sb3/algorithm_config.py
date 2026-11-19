# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define configuration values for algorithm."""

from typing import Any

from lanista.reinforcengine.rl_technologies.training_device import sb3_device


_SUPPORTED_PPO_KEYS = {
    "policy",
    "verbose",
    "learning_rate",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "clip_range_vf",
    "normalize_advantage",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "use_sde",
    "sde_sample_freq",
    "rollout_buffer_class",
    "rollout_buffer_kwargs",
    "target_kl",
    "stats_window_size",
    "tensorboard_log",
    "policy_kwargs",
    "seed",
    "device",
    "_init_setup_model",
}

_IGNORED_CONFIG_KEYS = {"env_id"}


def stable_baselines_configurator_name(model_configuration: Any, default: str = "training_algorithm_configurator") -> str:
    """Return the stable baselines configurator name value owned by this algorithm config module."""
    algorithm = getattr(model_configuration, "algorithm", None)
    components = getattr(algorithm, "components", None)
    configured = getattr(components, "training_configurator", None) if components is not None else None
    return str(configured or default)


def build_ppo_constructor_args(
        *,
        configurator_module,
        configurator_name: str,
        env_id: str,
        env,
        tensorboard_log: str,
        default_verbose=False,
        logger=None,
) -> tuple[str, Any, dict]:
    """Build the ppo constructor args required by this algorithm config module."""
    configurator = getattr(configurator_module, configurator_name, None)
    raw_config = configurator(env_id) if callable(configurator) else {}
    if raw_config is None:
        raw_config = {}
    if not isinstance(raw_config, dict):
        if logger is not None:
            logger.warning(
                f"SB3 algorithm configurator '{configurator_name}' returned {type(raw_config).__name__}; using defaults."
            )
        raw_config = {}

    unsupported_keys = sorted(
        key for key in raw_config
        if key not in _SUPPORTED_PPO_KEYS and key not in _IGNORED_CONFIG_KEYS
    )
    if unsupported_keys and logger is not None:
        logger.warning(
            "Unsupported SB3 PPO configurator keys ignored: "
            + ", ".join(unsupported_keys)
        )

    policy = raw_config.get("policy", "MlpPolicy")
    ppo_kwargs = {
        "verbose": raw_config.get("verbose", default_verbose),
        "clip_range_vf": raw_config.get("clip_range_vf", 150.0),
        "tensorboard_log": str(raw_config.get("tensorboard_log") or tensorboard_log),
    }
    for key, value in raw_config.items():
        if key in ("policy", "verbose", "clip_range_vf", "tensorboard_log") or key in _IGNORED_CONFIG_KEYS:
            continue
        if key in _SUPPORTED_PPO_KEYS:
            ppo_kwargs[key] = value

    ppo_kwargs["device"] = sb3_device()
    return policy, env, ppo_kwargs
