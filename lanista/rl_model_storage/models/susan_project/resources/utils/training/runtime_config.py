# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared Ray/SB3 construction and curriculum learning-rate configuration."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, TypedDict

import gymnasium as gym
import numpy as np
import torch.nn as nn
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.policy.policy import PolicySpec
from ray.tune.logger import CSVLogger, JsonLogger, UnifiedLogger
from ray.tune.logger.tensorboardx import VALID_SUMMARY_TYPES, TBXLogger
from ray.tune.result import TIMESTEPS_TOTAL, TRAINING_ITERATION
from ray.tune.utils import flatten_dict

from lanista.foundation.system import get_core_count, get_cpu_count
from lanista.locations import get_tensorboard_logs_path
from lanista.reinforcengine.rl_technologies.training_device import ray_gpu_count
from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    TRAINING_TENSORBOARD_ROOTS,
)

from ..observation.observation import flatten_observation, observation_space_box
from .ppo_config import (
    PPO_CLIP_RANGE,
    PPO_ENTROPY_COEFF,
    PPO_FULL_CURRICULUM_LEARNING_RATE,
    PPO_GAE_LAMBDA,
    PPO_GAMMA,
    PPO_LEARNING_RATE,
    PPO_MAX_GRAD_NORM,
    PPO_MINIBATCH_SIZE,
    PPO_NUM_EPOCHS,
    PPO_POLICY_ACTIVATION,
    PPO_POLICY_LOG_STD_INIT,
    PPO_POLICY_NET_ARCH,
    PPO_ROLLOUT_STEPS,
    PPO_TARGET_KL,
    PPO_VALUE_CLIP_RANGE,
    PPO_VALUE_LOSS_COEFF,
)

_LOGGER = logging.getLogger(__name__)


def _warn_quality_degradation(
    event: str,
    exc: BaseException | None = None,
    **context: object,
) -> None:
    """Report the quality degradation owned by this runtime config module."""
    fields = [f"event={event}", "quality=degraded"]
    fields.extend(f"{key}={value}" for key, value in sorted(context.items()))
    if exc is not None:
        fields.extend((f"error_type={type(exc).__name__}", f"error={exc}"))
    _LOGGER.warning(" ".join(fields))


class RuntimeConfig:
    """Represent configuration for runtime."""

    @staticmethod
    def _resolve_nested_value(root: Any, path: tuple[str, ...], default=None):
        """Resolve the nested value required by this runtime config."""
        current = root
        for part in path:
            if current is None:
                return default
            if isinstance(current, dict):
                current = current.get(part)
            else:
                current = getattr(current, part, None)
        return default if current is None else current

    def resolve_training_warmup_jitter_seconds(self, configuration: Any, default: int) -> int:
        """Resolve the training warmup jitter seconds required by this runtime config."""
        value = self._resolve_nested_value(
            configuration,
            ("algorithm", "training_conditions", "training_warmup_jitter_seconds"),
            default,
        )
        try:
            return max(0, int(value))
        except (TypeError, ValueError) as exc:
            _warn_quality_degradation(
                "training_warmup_jitter_fallback",
                exc,
                fallback=default,
            )
            return int(default)

    @staticmethod
    def resolve_nested_configuration_value(configuration: Any, name: str, default=None):
        """Resolve the nested configuration value required by this runtime config."""
        runtime = RuntimeConfig._resolve_nested_value(configuration, ("runtime",), None)
        value = RuntimeConfig._resolve_nested_value(runtime, ("evaluation", name), None)
        return default if value is None else value

    def resolve_constant_speed_diagnostic_speed(self, configuration: Any, default: float = 4.2) -> float:
        """Resolve the constant speed diagnostic speed required by this runtime config."""
        value = self.resolve_nested_configuration_value(configuration, "constant_speed_diagnostic_speed_mps", default)
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError) as exc:
            _warn_quality_degradation(
                "constant_speed_diagnostic_speed_fallback",
                exc,
                fallback=default,
            )
            return float(default)

    def resolve_constant_speed_diagnostic_max_steps(self, configuration: Any, default: int = 5000) -> int:
        """Resolve the constant speed diagnostic max steps required by this runtime config."""
        value = self.resolve_nested_configuration_value(configuration, "constant_speed_diagnostic_max_steps", default)
        try:
            return max(1, int(value))
        except (TypeError, ValueError) as exc:
            _warn_quality_degradation(
                "constant_speed_diagnostic_max_steps_fallback",
                exc,
                fallback=default,
            )
            return int(default)


RUNTIME_CONFIG = RuntimeConfig()


@dataclass(frozen=True)
class RayTrainingTopology:
    """Carry the data required for ray training topology."""

    number_of_cpu_cores: int
    number_of_cpu_threads: int
    number_of_rollout_workers: int
    number_of_envs_per_worker: int
    rollout_fragment_length: int | str
    batch_mode: str

    @property
    def train_batch_size_per_learner(self) -> int:
        """Return the train batch size per learner exposed by this ray training topology."""
        return PPO_ROLLOUT_STEPS


def build_ray_training_topology() -> RayTrainingTopology:
    """Build the ray training topology required by this runtime config module."""
    number_of_cpu_cores = get_core_count()
    number_of_cpu_threads = get_cpu_count()
    default_rollout_workers = 8
    configured_rollout_workers = int(os.environ.get("LANISTAI_RAY_ROLLOUT_WORKERS", default_rollout_workers))
    number_of_rollout_workers = max(1, min(configured_rollout_workers, max(1, number_of_cpu_cores - 2)))
    number_of_envs_per_worker = 1
    configured_fragment_length = os.environ.get("LANISTAI_RAY_ROLLOUT_FRAGMENT_LENGTH")
    if configured_fragment_length is None:
        rollout_fragment_length: int | str = "auto"
    else:
        try:
            rollout_fragment_length = max(
                PPO_MINIBATCH_SIZE,
                int(configured_fragment_length),
            )
        except (TypeError, ValueError) as exc:
            _warn_quality_degradation(
                "ray_rollout_fragment_length_fallback",
                exc,
                configured_value=repr(configured_fragment_length),
                fallback="auto",
            )
            rollout_fragment_length = "auto"

    batch_mode = str(os.environ.get("LANISTAI_RAY_BATCH_MODE", "truncate_episodes")).strip()
    if batch_mode not in {"truncate_episodes", "complete_episodes"}:
        _warn_quality_degradation(
            "ray_batch_mode_fallback",
            configured_value=repr(batch_mode),
            fallback="truncate_episodes",
        )
        batch_mode = "truncate_episodes"

    return RayTrainingTopology(
        number_of_cpu_cores=number_of_cpu_cores,
        number_of_cpu_threads=number_of_cpu_threads,
        number_of_rollout_workers=number_of_rollout_workers,
        number_of_envs_per_worker=number_of_envs_per_worker,
        rollout_fragment_length=rollout_fragment_length,
        batch_mode=batch_mode,
    )


def build_ray_checkpoint_saving_frequency(base_save_frequency: int) -> int:
    """Scale Ray checkpoint cadence by the actual parallel rollout topology.

    The manifest value is shared with SB3 as the single-run baseline. Ray samples
    from multiple env runners in parallel, so saving every raw manifest interval
    creates checkpoint artifacts much more often in wall-clock time. Scaling by
    the active runner/env count keeps storage growth comparable to SB3 while
    leaving PPO sampling and training parameters untouched.
    """
    try:
        base_save_frequency = int(base_save_frequency)
    except (TypeError, ValueError) as exc:
        _warn_quality_degradation(
            "ray_checkpoint_frequency_fallback",
            exc,
            fallback=0,
        )
        base_save_frequency = 0
    if base_save_frequency <= 0:
        return base_save_frequency

    topology = build_ray_training_topology()
    parallel_envs = max(1, topology.number_of_rollout_workers * topology.number_of_envs_per_worker)
    return base_save_frequency * parallel_envs


def is_full_curriculum_state_like(curriculum_state: Any) -> bool:
    """Return whether a published/current curriculum state means full route.

    Both SB3 and Ray publish `None` when the curriculum is unlocked. Historical
    checkpoints may still carry dict-like state fields, so keep the check
    tolerant and local to this resource package.
    """
    if curriculum_state is None:
        return True
    getter = curriculum_state.get if hasattr(curriculum_state, "get") else None
    if getter is not None:
        missing = object()
        state_name = getter("name", None) or getter("state_name", None)
        target_type = getter("target_type", missing)
        target_value = getter("target_value", missing)
        targets_explicitly_unlocked = target_type is None and target_value is None
    else:
        missing = object()
        state_name = getattr(curriculum_state, "name", None) or getattr(curriculum_state, "state_name", None)
        target_type = getattr(curriculum_state, "target_type", missing)
        target_value = getattr(curriculum_state, "target_value", missing)
        targets_explicitly_unlocked = target_type is None and target_value is None
    return (
        state_name == "full" or
        target_type == "full" or
        target_value == "full" or
        targets_explicitly_unlocked
    )


def ppo_learning_rate_for_curriculum_state(curriculum_state: Any) -> float:
    """Return the ppo learning rate for curriculum state value owned by this runtime config module.
    """
    return (
        PPO_FULL_CURRICULUM_LEARNING_RATE
        if is_full_curriculum_state_like(curriculum_state)
        else PPO_LEARNING_RATE
    )


def apply_sb3_learning_rate(model: Any, learning_rate: float) -> bool:
    """Apply a new PPO learning rate to SB3 immediately and for future updates."""
    if model is None:
        _warn_quality_degradation("sb3_learning_rate_model_unavailable")
        return False
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as exc:
        _warn_quality_degradation("sb3_learning_rate_invalid", exc)
        return False

    schedule_updated = True
    try:
        model.lr_schedule = lambda _progress_remaining: learning_rate
    except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
        schedule_updated = False
        _warn_quality_degradation("sb3_learning_rate_schedule_update_failed", exc)

    optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
    if optimizer is None:
        _warn_quality_degradation("sb3_learning_rate_optimizer_unavailable")
        return False
    try:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        _warn_quality_degradation("sb3_learning_rate_optimizer_update_failed", exc)
        return False
    return schedule_updated


class _RayLearnerPort(Protocol):
    """Define the typed boundary for ray learner."""

    def get_optimizer(self, *, module_id: str, optimizer_name: str) -> object:
        """Return the optimizer exposed by this ray learner port."""
        ...

    def _set_optimizer_lr(self, optimizer: object, learning_rate: float) -> None:
        """Set the optimizer lr owned by this ray learner port."""
        ...


class _RayLearnerFailure(TypedDict):
    """Describe the typed mapping for ray learner failure."""

    module_id: str
    error_type: str


class _RayLearnerUpdate(TypedDict):
    """Describe the typed mapping for ray learner update."""

    module_id: str | None
    failures: list[_RayLearnerFailure]


class _RayLearnerGroupPort(Protocol):
    """Define the typed boundary for ray learner group."""

    def foreach_learner(
        self,
        operation: Callable[[_RayLearnerPort, float, str], _RayLearnerUpdate],
        *,
        lr: float,
        primary_module_id: str,
    ) -> Iterable[object]:
        """Return the foreach learner value provided by this ray learner group port."""
        ...


class _RayLearningRateAlgorithmPort(Protocol):
    """Define the typed boundary for ray learning rate algorithm."""

    learner_group: _RayLearnerGroupPort


@dataclass(frozen=True, slots=True)
class _RayResultSummary:
    """Summarize ray result results."""

    readable: bool
    applied: bool


def _set_ray_learner_learning_rate(
    learner: _RayLearnerPort,
    lr: float,
    primary_module_id: str,
) -> _RayLearnerUpdate:
    """Set the ray learner learning rate owned by this runtime config module."""
    failures: list[_RayLearnerFailure] = []
    for candidate_module_id in (
        primary_module_id,
        "default_policy",
        "__all_modules__",
    ):
        try:
            optimizer = learner.get_optimizer(
                module_id=candidate_module_id,
                optimizer_name="default_optimizer",
            )
            learner._set_optimizer_lr(optimizer, lr)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            failures.append(
                {
                    "module_id": candidate_module_id,
                    "error_type": type(exc).__name__,
                }
            )
        else:
            return {"module_id": candidate_module_id, "failures": failures}
    return {"module_id": None, "failures": failures}


def _ray_learning_rate_result_applied(item: object) -> bool:
    """Return the ray learning rate result applied value owned by this runtime config module."""
    try:
        result = item.get() if hasattr(item, "get") else item
    except Exception as exc:
        _warn_quality_degradation("ray_learning_rate_result_failed", exc)
        return False

    if hasattr(item, "ok") and not item.ok:
        failure = result if isinstance(result, BaseException) else RuntimeError(str(result))
        _warn_quality_degradation("ray_learning_rate_remote_failure", failure)
        return False
    if not isinstance(result, dict):
        return result is not None

    module_id = result.get("module_id")
    if not module_id:
        _warn_quality_degradation(
            "ray_learning_rate_optimizer_unavailable",
            attempted_modules=len(result.get("failures", [])),
        )
    return bool(module_id)


def _summarize_ray_learning_rate_results(results: object) -> _RayResultSummary:
    """Return the summarize ray learning rate results value owned by this runtime config module."""
    try:
        result_items = list(results)
    except TypeError as exc:
        _warn_quality_degradation("ray_learning_rate_results_unreadable", exc)
        return _RayResultSummary(readable=False, applied=False)

    applied = False
    for item in result_items:
        applied = _ray_learning_rate_result_applied(item) or applied
    return _RayResultSummary(readable=True, applied=applied)


def apply_ray_learning_rate(
    algorithm: _RayLearningRateAlgorithmPort | None,
    learning_rate: float,
    *,
    module_id: str = "EGO_policy",
) -> bool:
    """Apply a PPO learning rate to Ray's learner optimizer when available."""
    if algorithm is None:
        _warn_quality_degradation("ray_learning_rate_algorithm_unavailable")
        return False
    try:
        learning_rate = float(learning_rate)
    except (TypeError, ValueError) as exc:
        _warn_quality_degradation("ray_learning_rate_invalid", exc)
        return False

    learner_group = getattr(algorithm, "learner_group", None)
    if learner_group is None:
        _warn_quality_degradation("ray_learning_rate_learner_group_unavailable")
        return False
    try:
        results = learner_group.foreach_learner(
            _set_ray_learner_learning_rate,
            lr=learning_rate,
            primary_module_id=module_id,
        )
    except Exception as exc:
        _warn_quality_degradation("ray_learning_rate_dispatch_failed", exc)
        return False

    summary = _summarize_ray_learning_rate_results(results)
    if not summary.readable:
        return False
    if not summary.applied:
        _warn_quality_degradation("ray_learning_rate_not_applied")
    return summary.applied


def build_ray_policy_observation_space():
    """Build the ray policy observation space required by this runtime config module."""
    return observation_space_box()


def build_ray_policy_action_space():
    """Build the ray policy action space required by this runtime config module."""
    return gym.spaces.Box(
        low=np.array([-4.0], dtype=np.float32),
        high=np.array([3.0], dtype=np.float32),
        dtype=np.float32,
    )


def build_ray_policy_observation(sensors=None):
    """Build the ray policy observation required by this runtime config module."""
    return flatten_observation(sensors)


def build_ray_policy_spec() -> PolicySpec:
    """Build the ray policy spec required by this runtime config module."""
    return PolicySpec(
        policy_class=None,
        observation_space=build_ray_policy_observation_space(),
        action_space=build_ray_policy_action_space(),
        config={"observation": build_ray_policy_observation},
    )


def build_ray_tune_logger_creator(loggers=None):
    """Build the ray tune logger creator required by this runtime config module."""
    def _logger_creator(config):
        """Perform the logger creator step within the enclosing build ray tune logger creator
        workflow.
        """
        override_logdir = str(os.environ.get("LANISTAI_TENSORBOARD_LOGDIR", "") or "").strip()
        if override_logdir:
            os.makedirs(override_logdir, exist_ok=True)
            return UnifiedLogger(config, override_logdir, loggers=loggers)

        logdir = str(get_tensorboard_logs_path())
        os.makedirs(logdir, exist_ok=True)
        prefix = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        full_dir = tempfile.mkdtemp(prefix=prefix, dir=logdir)
        return UnifiedLogger(config, full_dir, loggers=loggers)

    return _logger_creator


def build_ray_training_algorithm_config(
    *,
    env_id: str,
    policies: dict[str, PolicySpec],
    policy_mapping_fn,
    logger_creator,
    model_config_builder,
    catalog_class=None,
) -> PPOConfig:
    """Build the ray training algorithm config required by this runtime config module."""
    topology = build_ray_training_topology()
    model_config = model_config_builder()
    return (
        PPOConfig()
        .environment(
            env_id,
            disable_env_checking=False,
            normalize_actions=False,
            clip_actions=True,
        )
        .framework(framework="torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),
            model_config=model_config,
        )
        .env_runners(
            num_env_runners=topology.number_of_rollout_workers,
            num_envs_per_env_runner=topology.number_of_envs_per_worker,
            rollout_fragment_length=topology.rollout_fragment_length,
            batch_mode=topology.batch_mode,
            compress_observations=False,
        )
        .training(
            lr=PPO_LEARNING_RATE,
            num_epochs=PPO_NUM_EPOCHS,
            minibatch_size=PPO_MINIBATCH_SIZE,
            gamma=PPO_GAMMA,
            lambda_=PPO_GAE_LAMBDA,
            clip_param=PPO_CLIP_RANGE,
            vf_clip_param=PPO_VALUE_CLIP_RANGE,
            use_kl_loss=True,
            kl_target=PPO_TARGET_KL,
            entropy_coeff=PPO_ENTROPY_COEFF,
            vf_loss_coeff=PPO_VALUE_LOSS_COEFF,
            grad_clip=PPO_MAX_GRAD_NORM,
            train_batch_size_per_learner=topology.train_batch_size_per_learner,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=ray_gpu_count(),
            num_cpus_per_learner=1,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .debugging(
            log_level="INFO",
            logger_creator=logger_creator,
        )
    )


def build_ray_deployment_algorithm_config(
    *,
    env_id: str,
    policies: dict[str, PolicySpec],
    policy_mapping_fn,
    model_config_builder,
    catalog_class=None,
) -> PPOConfig:
    """Build the ray deployment algorithm config required by this runtime config module."""
    model_config = model_config_builder()
    return (
        PPOConfig()
        .environment(
            env_id,
            disable_env_checking=True,
            normalize_actions=False,
            clip_actions=True,
        )
        .env_runners(
            num_env_runners=0,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=list(policies.keys()),
        )
        .framework(framework="torch")
        .rl_module(
            rl_module_spec=RLModuleSpec(catalog_class=catalog_class),
            model_config=model_config,
        )
        .debugging(log_level="DEBUG")
    )


def build_sb3_ppo_constructor_config(env_id: str) -> dict[str, Any]:
    """Build the sb3 ppo constructor config required by this runtime config module."""
    activation_map = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }
    activation_name = str(PPO_POLICY_ACTIVATION).strip().lower()
    activation_fn = activation_map.get(activation_name)
    if activation_fn is None:
        raise ValueError(
            f"Unsupported Susan PPO activation {PPO_POLICY_ACTIVATION!r}; "
            f"expected one of {tuple(activation_map)}."
        )
    return {
        "env_id": env_id,
        "policy": "MultiInputPolicy",
        "learning_rate": PPO_LEARNING_RATE,
        "n_steps": PPO_ROLLOUT_STEPS,
        "batch_size": PPO_MINIBATCH_SIZE,
        "n_epochs": PPO_NUM_EPOCHS,
        "gamma": PPO_GAMMA,
        "gae_lambda": PPO_GAE_LAMBDA,
        "clip_range": PPO_CLIP_RANGE,
        "clip_range_vf": PPO_VALUE_CLIP_RANGE,
        "normalize_advantage": True,
        "ent_coef": PPO_ENTROPY_COEFF,
        "vf_coef": PPO_VALUE_LOSS_COEFF,
        "max_grad_norm": PPO_MAX_GRAD_NORM,
        "target_kl": PPO_TARGET_KL,
        "policy_kwargs": {
            "activation_fn": activation_fn,
            "log_std_init": PPO_POLICY_LOG_STD_INIT,
            "net_arch": {
                "pi": list(PPO_POLICY_NET_ARCH),
                "vf": list(PPO_POLICY_NET_ARCH),
            },
        },
    }


def resolve_training_warmup_jitter_seconds(configuration: Any, default: int) -> int:
    """Resolve the training warmup jitter seconds required by this runtime config module."""
    return RUNTIME_CONFIG.resolve_training_warmup_jitter_seconds(configuration, default)


def resolve_nested_configuration_value(configuration: Any, name: str, default=None):
    """Resolve the nested configuration value required by this runtime config module."""
    return RUNTIME_CONFIG.resolve_nested_configuration_value(configuration, name, default)


def resolve_constant_speed_diagnostic_speed(configuration: Any, default: float = 4.2) -> float:
    """Resolve the constant speed diagnostic speed required by this runtime config module."""
    return RUNTIME_CONFIG.resolve_constant_speed_diagnostic_speed(configuration, default)


def resolve_constant_speed_diagnostic_max_steps(configuration: Any, default: int = 5000) -> int:
    """Resolve the constant speed diagnostic max steps required by this runtime config module."""
    return RUNTIME_CONFIG.resolve_constant_speed_diagnostic_max_steps(configuration, default)


class FilteredRayTuneTBXLogger(TBXLogger):
    """TensorBoard logger for the small Ray/Tune diagnostics worth keeping."""

    _NATIVE_SCALAR_TAGS = {
        "env_runners/episode_len_mean": "ray_sampling/episode_length_mean",
        "env_runners/episode_return_mean": "ray_sampling/episode_return_mean",
        "env_runners/episode_return_min": "ray_sampling/episode_return_min",
        "env_runners/episode_return_max": "ray_sampling/episode_return_max",
        "env_runners/num_env_steps_sampled": "ray_sampling/sampled_steps",
        "env_runners/num_env_steps_sampled_lifetime": "ray_sampling/sampled_steps_lifetime",
        "num_env_steps_sampled_lifetime": "ray_sampling/sampled_steps_lifetime",
        "env_runners/num_env_steps_sampled_lifetime_throughput": "ray_sampling/sampling_throughput",
        "learners/EGO_policy/total_loss": "ray_learner/total_loss",
        "learners/EGO_policy/policy_loss": "ray_learner/policy_loss",
        "learners/EGO_policy/vf_loss": "ray_learner/value_loss",
        "learners/EGO_policy/entropy": "ray_learner/entropy",
        "learners/EGO_policy/mean_kl_loss": "ray_learner/mean_kl_loss",
        "learners/EGO_policy/curr_kl_coeff": "ray_learner/kl_coefficient",
        "learners/EGO_policy/clip_fraction": "ray_learner/clip_fraction",
        "learners/EGO_policy/vf_explained_var": "ray_learner/value_explained_variance",
        "learners/EGO_policy/gradients_default_optimizer_global_norm": "ray_learner/gradient_norm",
        "learners/EGO_policy/default_optimizer_learning_rate": "ray_learner/learning_rate",
        "fault_tolerance/num_healthy_workers": "ray_system/healthy_workers",
        "fault_tolerance/num_remote_worker_restarts": "ray_system/remote_worker_restarts",
        "learners/EGO_policy/diff_num_grad_updates_vs_sampler_policy": "ray_system/learner_sampler_update_lag",
        "learners/EGO_policy/num_module_steps_trained_lifetime": "ray_system/trained_module_steps_lifetime",
    }
    _LANISTAI_ROOTS = frozenset(TRAINING_TENSORBOARD_ROOTS)

    @staticmethod
    def _lanistai_histogram_tag(attr: str) -> str | None:
        """Return the lanistai histogram tag value owned by this filtered ray tune tbx logger."""
        attr = str(attr)
        for prefix in (
            "hist_stats/lanistai_distribution/",
            "env_runners/hist_stats/lanistai_distribution/",
            "lanistai_distribution/",
            "hist_stats/lanistai_hist/",
            "env_runners/hist_stats/lanistai_hist/",
            "lanistai_hist/",
        ):
            if attr.startswith(prefix):
                root = (
                    "lanistai_distribution"
                    if "lanistai_distribution" in prefix
                    else "lanistai_hist"
                )
                return f"{root}/{attr[len(prefix):]}"
        return None

    @classmethod
    def _lanistai_scalar_tag(cls, attr: str) -> str | None:
        """Strip RLlib's result-container prefix from a canonical custom tag."""

        candidate = str(attr)
        if candidate.startswith("env_runners/"):
            candidate = candidate[len("env_runners/") :]
        root = candidate.split("/", 1)[0]
        return candidate if root in cls._LANISTAI_ROOTS else None

    def on_result(self, result):
        """Handle the result event for this filtered ray tune tbx logger."""
        step = result.get(TIMESTEPS_TOTAL) or result.get(TRAINING_ITERATION)
        flat_result = flatten_dict(result, delimiter="/")
        emitted_tags: set[str] = set()
        for attr, value in flat_result.items():
            histogram_tag = self._lanistai_histogram_tag(attr)
            if histogram_tag is not None:
                histogram_values = np.asarray(value, dtype=float).reshape(-1)
                histogram_values = histogram_values[np.isfinite(histogram_values)]
                if histogram_values.size:
                    self._file_writer.add_histogram(
                        histogram_tag,
                        histogram_values,
                        global_step=step,
                    )
                continue

            tag = self._lanistai_scalar_tag(attr) or self._NATIVE_SCALAR_TAGS.get(attr)
            if tag is None or tag in emitted_tags:
                continue
            if isinstance(value, tuple(VALID_SUMMARY_TYPES)) and not np.isnan(value):
                self._file_writer.add_scalar(
                    tag,
                    value,
                    global_step=step,
                )
                emitted_tags.add(tag)


def build_ray_tune_logger_classes() -> list[type] | None:
    """Build the ray tune logger classes required by this runtime config module."""
    mode = str(os.environ.get("LANISTAI_RAY_TUNE_TENSORBOARD_MODE", "compact") or "").strip().lower()
    if mode in {"0", "off", "none", "false"}:
        return []
    if mode in {"1", "full", "all", "true"}:
        return None
    return [JsonLogger, CSVLogger, FilteredRayTuneTBXLogger]
