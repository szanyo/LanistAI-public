# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Evaluate trained reinforcement-learning policies."""

from stable_baselines3.common.callbacks import EvalCallback

from lanista.reinforcengine.utils.scenarios.scenario_selection import apply_scenario_selection


def _config_get(configuration, key: str, default=None):
    """Return the config get value owned by this evaluation module."""
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(key, default)
    getter = getattr(configuration, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(configuration, key, default)


def _evaluation_runtime_config(model_configuration):
    """Return the evaluation runtime config value owned by this evaluation module."""
    runtime = _config_get(model_configuration, "runtime")
    return _config_get(runtime, "evaluation")


def training_eval_callback_enabled(model_configuration) -> bool:
    """Return whether training eval callback enabled applies to this evaluation module."""
    evaluation_config = _evaluation_runtime_config(model_configuration)
    return bool(_config_get(evaluation_config, "training_eval_callback_enabled", False))


def build_training_eval_callback(
        *,
        model_configuration,
        sim_env_cls,
        env_kwargs: dict,
        scenario_manager,
        scenario_state_factory,
        logger=None,
):
    """Build the training eval callback required by this evaluation module."""
    if not training_eval_callback_enabled(model_configuration):
        if logger is not None:
            logger.info("SB3 training evaluation callback disabled by configuration.")
        return None

    eval_scenario_manager = scenario_manager.clone()
    apply_scenario_selection(
        model_configuration,
        eval_scenario_manager,
        purpose="evaluation",
        default_index=0,
        logger=logger,
    )

    eval_scenario_index_state = scenario_state_factory()
    eval_scenario_manager.set_scenario_index_state(eval_scenario_index_state)

    eval_env_kwargs = dict(env_kwargs)
    eval_env_kwargs["scenario_manager"] = eval_scenario_manager
    eval_env_kwargs["scenario_index_state"] = eval_scenario_index_state
    eval_env_kwargs["child_logger_name"] = "stable_baselines_eval_env"

    eval_freq = int(_config_get(model_configuration, "evaluation_frequency", 100000) or 100000)
    deterministic = bool(_config_get(model_configuration, "evaluation_deterministic", True))
    evaluation_config = _evaluation_runtime_config(model_configuration)
    n_eval_episodes = int(_config_get(evaluation_config, "n_eval_episodes", 1) or 1)

    eval_env = sim_env_cls(**eval_env_kwargs)
    if logger is not None:
        logger.info(
            "SB3 training evaluation callback enabled: "
            f"scenario={eval_scenario_manager.get_index()}, eval_freq={eval_freq}, "
            f"n_eval_episodes={n_eval_episodes}, deterministic={deterministic}"
        )

    return EvalCallback(
        eval_env,
        best_model_save_path=None,
        eval_freq=eval_freq,
        deterministic=deterministic,
        n_eval_episodes=n_eval_episodes,
    )
