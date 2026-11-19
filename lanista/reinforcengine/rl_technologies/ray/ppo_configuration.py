# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed PPO configuration shared by the Eve and Jasmine Ray plugins."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray
from ray.rllib.algorithms import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from lanista.foundation.filesystem import makedirs
from lanista.foundation.system import get_core_count, get_cpu_count
from lanista.locations import get_tensorboard_logs_path
from lanista.reinforcengine.rl_technologies.training_device import ray_gpu_count


class RayLoggerFactory(Protocol):
    """Factory signature accepted by RLlib's legacy logger hook."""

    def __call__(
        self,
        config: dict[object, object],
        logdir: str,
        trial: object | None = None,
    ) -> object:
        """Invoke this ray logger factory as a callable."""
        ...


class TensorboardContinuityLogger:
    """Offset Ray result steps above the prior logical-session event floor."""

    def __init__(self, delegate: object, *, step_floor: int | None = None) -> None:
        """Initialize the logger proxy without taking delegate ownership."""

        self._delegate = delegate
        raw_floor = (
            os.environ.pop("LANISTAI_TENSORBOARD_STEP_FLOOR", "0")
            if step_floor is None
            else step_floor
        )
        try:
            self._step_floor = max(0, int(raw_floor or 0))
        except (TypeError, ValueError):
            self._step_floor = 0
        self._first_raw_step: int | None = None
        self._last_emitted_step = self._step_floor

    @property
    def enabled(self) -> bool:
        """Return whether this segment requires step translation."""

        return self._step_floor > 0

    def __getattr__(self, name: str) -> object:
        """Delegate the remaining Ray logger surface."""

        return getattr(self._delegate, name)

    @staticmethod
    def _raw_step(result: dict[object, object]) -> int:
        """Resolve the step used by Ray's TensorBoard logger."""

        candidates = (
            result.get("timesteps_total"),
            result.get("num_env_steps_sampled_lifetime"),
            (
                result.get("env_runners", {}).get("num_env_steps_sampled_lifetime")
                if isinstance(result.get("env_runners"), dict)
                else None
            ),
            result.get("training_iteration"),
        )
        for candidate in candidates:
            try:
                if candidate is not None:
                    return max(0, int(float(candidate)))
            except (TypeError, ValueError):
                continue
        return 0

    def on_result(self, result: dict[object, object]) -> object:
        """Publish a strictly increasing TensorBoard step for this segment."""

        if self._step_floor <= 0:
            return self._delegate.on_result(result)  # type: ignore[attr-defined,no-any-return]
        raw_step = self._raw_step(result)
        if self._first_raw_step is None:
            self._first_raw_step = raw_step
        relative_step = max(1, raw_step - self._first_raw_step + 1)
        emitted_step = max(
            self._last_emitted_step + 1,
            self._step_floor + relative_step,
        )
        self._last_emitted_step = emitted_step
        adjusted = dict(result)
        adjusted["timesteps_total"] = emitted_step
        return self._delegate.on_result(adjusted)  # type: ignore[attr-defined,no-any-return]


def wrap_tensorboard_continuity_logger(logger: object) -> object:
    """Wrap one Ray logger with the session step-floor contract."""

    continuity_logger = TensorboardContinuityLogger(logger)
    return continuity_logger if continuity_logger.enabled else logger


@dataclass(frozen=True, slots=True)
class RayPPOConfiguration:
    """Host topology, spaces, and stable PPO hyperparameters."""

    observation_size: int
    number_of_cpu_cores: int
    number_of_cpu_threads: int
    number_of_rollout_workers: int
    number_of_envs_per_worker: int = 1
    rollout_fragment_length: int = 400
    learning_rate: float = 3e-4
    num_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_param: float = 0.2
    entropy_coeff: float = 0.0
    vf_loss_coeff: float = 0.5
    max_grad_norm: float = 0.5

    @classmethod
    def for_host(cls, *, observation_size: int) -> RayPPOConfiguration:
        """Resolve the historical worker topology for the current host."""

        core_count = int(get_core_count())
        return cls(
            observation_size=observation_size,
            number_of_cpu_cores=core_count,
            number_of_cpu_threads=int(get_cpu_count()),
            number_of_rollout_workers=core_count - 2,
        )

    @property
    def train_batch_size_per_learner(self) -> int:
        """Return one fragment per remote environment runner."""

        return self.number_of_rollout_workers * self.rollout_fragment_length


class RayPPOConfigurator:
    """Build policy spaces and RLlib configurations from immutable settings."""

    def __init__(self, configuration: RayPPOConfiguration) -> None:
        """Initialize the ray ppo configurator and its owned state."""
        self._configuration = configuration

    @property
    def configuration(self) -> RayPPOConfiguration:
        """Expose immutable values for diagnostics and parity tests."""

        return self._configuration

    def observation_space(self) -> gym.spaces.Box:
        """Return the model-sized float64 observation space."""

        return gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._configuration.observation_size,),
            dtype=np.float64,
        )

    @staticmethod
    def action_space() -> gym.spaces.Box:
        """Return the shared longitudinal acceleration-command space."""

        return gym.spaces.Box(
            low=-4.0,
            high=3.0,
            shape=(1,),
            dtype=np.float64,
        )

    def observation(
        self,
        sensors: object | None = None,
        *,
        integrity_check: bool = False,
    ) -> NDArray[np.float64]:
        """Normalize configured sensor values to a float64 array."""

        if sensors is None or integrity_check:
            return np.zeros(
                (self._configuration.observation_size,),
                dtype=np.float64,
            )
        return np.asarray(
            sensors,
            dtype=np.float64,
        )

    def policies(self) -> dict[str, PolicySpec]:
        """Build the single EGO policy consumed by the environment."""

        return {
            "EGO_policy": PolicySpec(  # type: ignore[no-untyped-call]
                policy_class=None,
                observation_space=self.observation_space(),
                action_space=self.action_space(),
                config={"observation": self.observation},
            )
        }

    @staticmethod
    def policy_mapping_fn(
        agent_id: str,
        *args: object,
        **kwargs: object,
    ) -> str:
        """Map the single supported EGO agent to its policy."""

        del args, kwargs
        if agent_id.startswith("EGO_agent"):
            return "EGO_policy"
        raise ValueError(f"Unknown agent_id: {agent_id}")

    @staticmethod
    def create_logger(
        configuration: dict[object, object],
        *,
        logger_factory: RayLoggerFactory,
    ) -> object:
        """Create an RLlib logger under the configured TensorBoard root."""

        override = str(os.environ.get("LANISTAI_TENSORBOARD_LOGDIR", "") or "").strip()
        if override:
            makedirs(override, exist_ok=True)
            return wrap_tensorboard_continuity_logger(
                logger_factory(configuration, override, None)
            )

        log_directory = str(get_tensorboard_logs_path())
        makedirs(log_directory, exist_ok=True)
        prefix = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
        run_directory = tempfile.mkdtemp(prefix=prefix, dir=log_directory)
        return wrap_tensorboard_continuity_logger(
            logger_factory(configuration, run_directory, None)
        )

    def training_configuration(
        self,
        env_id: str,
        *,
        logger_creator: Callable[[dict[object, object]], object],
    ) -> PPOConfig:
        """Build the stable training configuration."""

        settings = self._configuration
        policies = self.policies()
        configured = (
            PPOConfig()  # type: ignore[no-untyped-call,call-arg]
            .environment(env_id, disable_env_checking=False)
            .framework(framework="torch")
            .env_runners(
                num_env_runners=settings.number_of_rollout_workers,
                num_envs_per_env_runner=settings.number_of_envs_per_worker,
                rollout_fragment_length="auto",
                batch_mode="truncate_episodes",
                compress_observations=False,
            )
            .training(
                lr=settings.learning_rate,
                num_epochs=settings.num_epochs,
                minibatch_size=settings.minibatch_size,
                gamma=settings.gamma,
                lambda_=settings.gae_lambda,
                clip_param=settings.clip_param,
                entropy_coeff=settings.entropy_coeff,
                vf_loss_coeff=settings.vf_loss_coeff,
                grad_clip=settings.max_grad_norm,
                train_batch_size_per_learner=(settings.train_batch_size_per_learner),
            )
            .learners(
                num_learners=0,
                num_gpus_per_learner=ray_gpu_count(),
                num_cpus_per_learner=1,
            )
            .multi_agent(
                policies=policies,
                policy_mapping_fn=self.policy_mapping_fn,
                policies_to_train=list(policies),
            )
            .debugging(
                log_level="INFO",
                logger_creator=logger_creator,
            )
        )
        return cast(PPOConfig, configured)

    def deployment_configuration(self, env_id: str) -> PPOConfig:
        """Build the local restored-policy deployment configuration."""

        policies = self.policies()
        configured = (
            PPOConfig()  # type: ignore[no-untyped-call]
            .environment(env_id, disable_env_checking=True)
            .env_runners(num_env_runners=0)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .multi_agent(
                policies=policies,
                policy_mapping_fn=self.policy_mapping_fn,
                policies_to_train=list(policies),
            )
            .framework(framework="torch")
            .debugging(log_level="DEBUG")
        )
        return cast(PPOConfig, configured)
