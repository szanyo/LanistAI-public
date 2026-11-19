# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed model policies for the shared Ray algorithm lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import ray

from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    canonical_training_metrics_from_result,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    is_full_curriculum_state,
)

CurriculumState = dict[str, object] | None


class RayPolicyLogger(Protocol):
    """Logging surface used by framework-neutral model policies."""

    def info(self, message: str) -> object:
        """Record an informational lifecycle event."""

    def warning(self, message: str) -> object:
        """Record a quality degradation."""


class _RemoteGetter(Protocol):
    """Define the typed boundary for remote getter."""

    def remote(self) -> ray.ObjectRef[object]:
        """Schedule a state read."""


class _RemoteSetter(Protocol):
    """Define the typed boundary for remote setter."""

    def remote(self, value: object) -> ray.ObjectRef[object]:
        """Schedule a state update."""


class RayCurriculumStateActor(Protocol):
    """Actor surface needed while restoring curriculum state."""

    @property
    def get(self) -> _RemoteGetter:
        """Return the remote state reader."""

    @property
    def set(self) -> _RemoteSetter:
        """Return the remote state writer."""


class RayScenarioManager(Protocol):
    """Scenario operations used by a completed training-window policy."""

    def get_index(self) -> int:
        """Return the active scenario index."""

    def next(self) -> object:
        """Advance to the next scenario."""


class RayCurriculumWindowManager(Protocol):
    """Curriculum operations used after one rolling training window."""

    def current_state(self) -> CurriculumState:
        """Return the active curriculum state."""

    def evaluate_and_update(self, ready: bool) -> None:
        """Advance the curriculum when its readiness gate passes."""

    def reset(self) -> None:
        """Reset the curriculum after a shared scenario switch."""

    def reset_for_scenario_switch(self) -> object:
        """Apply the model-specific scenario-switch reset policy."""


class RayCurriculumRestoreManager(Protocol):
    """Curriculum operation required while restoring a checkpoint."""

    def restore_state(self, state: dict[str, object]) -> None:
        """Restore one non-empty serialized curriculum state."""


class RayPolicyRollingStats(Protocol):
    """Rolling-statistics decisions consumed by model policies."""

    def curriculum_ready(
        self,
        current_state: dict[str, object],
    ) -> bool:
        """Return whether the current curriculum state can advance."""

    def scenario_ready(
        self,
        current_state: CurriculumState,
    ) -> bool:
        """Return whether the active scenario can advance."""

    def episode_count_summary(self) -> dict[str, int]:
        """Return training-global normalized episode counters."""

    def reset(self) -> None:
        """Reset scenario-local rolling state after a scenario switch."""


@dataclass(frozen=True, slots=True)
class RayTrainingWindowContext:
    """Inputs for one completed rolling-statistics training window."""

    algorithm: object
    runtime: object
    logger: RayPolicyLogger
    scenario_manager: RayScenarioManager
    curriculum_manager: RayCurriculumWindowManager
    rolling_stats: RayPolicyRollingStats
    train_result: dict[str, object]
    env_steps: int
    metrics_ready: bool


def curriculum_ready_for_update(
    rolling_stats: RayPolicyRollingStats,
    current_state: CurriculumState,
) -> bool:
    """Evaluate only curriculum states that still have a transition gate."""

    return current_state is not None and (
        is_full_curriculum_state(current_state)
        or rolling_stats.curriculum_ready(current_state)
    )


class RayModelPolicy(Protocol):
    """Model-specific decisions injected into the shared Ray lifecycle."""

    @property
    def training_scenario_default_index(self) -> int:
        """Return the fallback training scenario index."""

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return stable quality signals observed by optional hooks."""

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return model-specific environment constructor options."""

    def checkpoint_frequency(
        self,
        configured_frequency: int,
        logger: RayPolicyLogger,
    ) -> int:
        """Resolve the periodic checkpoint cadence."""

    def restore_curriculum_state(
        self,
        curriculum_manager: RayCurriculumRestoreManager,
        state_actor: RayCurriculumStateActor,
        restored_state: CurriculumState,
    ) -> CurriculumState:
        """Restore model-specific curriculum state and ordering barriers."""

    def on_algorithm_ready(
        self,
        algorithm: object,
        current_state: CurriculumState,
        logger: RayPolicyLogger,
    ) -> None:
        """Apply model-specific state after build/restore."""

    def record_training_result(
        self,
        runtime: object,
        train_result: dict[str, object],
        logger: RayPolicyLogger,
    ) -> bool:
        """Project optional telemetry and report whether data was written."""

    def checkpoint_step(self, train_result: object) -> int:
        """Extract the raw lifetime step used by checkpoint naming."""

    def process_training_window(
        self,
        context: RayTrainingWindowContext,
    ) -> CurriculumState:
        """Apply curriculum/scenario decisions for one completed window."""


class DefaultRayModelPolicy:
    """Preserve the historical Eve/Jasmine Ray lifecycle decisions."""

    training_scenario_default_index = 1

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this default ray model policy."""
        return frozenset()

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return the training environment options value owned by this default ray model policy."""
        _ = configuration, environment_type
        return {}

    def checkpoint_frequency(
        self,
        configured_frequency: int,
        logger: RayPolicyLogger,
    ) -> int:
        """Checkpoint the frequency owned by this default ray model policy."""
        _ = logger
        return int(configured_frequency)

    def restore_curriculum_state(
        self,
        curriculum_manager: object,
        state_actor: RayCurriculumStateActor,
        restored_state: CurriculumState,
    ) -> CurriculumState:
        """Restore the curriculum state owned by this default ray model policy."""
        _ = curriculum_manager
        state_actor.set.remote(restored_state)
        return restored_state

    def on_algorithm_ready(
        self,
        algorithm: object,
        current_state: CurriculumState,
        logger: RayPolicyLogger,
    ) -> None:
        """Handle the algorithm ready event for this default ray model policy."""
        _ = algorithm, current_state, logger

    def record_training_result(
        self,
        runtime: object,
        train_result: dict[str, object],
        logger: RayPolicyLogger,
    ) -> bool:
        """Record the training result owned by this default ray model policy."""
        _ = runtime, train_result, logger
        return False

    def checkpoint_step(self, train_result: object) -> int:
        """Checkpoint the step owned by this default ray model policy."""
        if not isinstance(train_result, dict):
            return 0
        env_runners = train_result.get("env_runners", {})
        if not isinstance(env_runners, dict):
            return 0
        try:
            return int(float(env_runners.get("num_env_steps_sampled_lifetime", 0)))
        except (TypeError, ValueError):
            return 0

    def process_training_window(
        self,
        context: RayTrainingWindowContext,
    ) -> CurriculumState:
        """Return the process training window value owned by this default ray model policy."""
        current_state = context.curriculum_manager.current_state()
        curriculum_ready = curriculum_ready_for_update(
            context.rolling_stats,
            current_state,
        )
        context.curriculum_manager.evaluate_and_update(curriculum_ready)

        scenario_ready = context.rolling_stats.scenario_ready(
            current_state
        )
        if context.metrics_ready and scenario_ready:
            context.scenario_manager.next()
            context.curriculum_manager.reset()
            context.rolling_stats.reset()

        train_result_episodes = self._train_result_episode_count(
            context.train_result
        )
        episode_counts = context.rolling_stats.episode_count_summary()
        train_iteration = context.train_result.get("training_iteration", "?")
        curriculum_name = (
            current_state.get("name")
            if current_state is not None
            else "full"
        )
        context.logger.info(
            f"[Progress] train_iter={train_iteration} "
            f"env_steps={context.env_steps} "
            f"train_result_episodes={train_result_episodes} "
            f"total_episodes={episode_counts['total']} "
            f"terminated={episode_counts['terminated']} "
            f"truncated={episode_counts['truncated']} "
            f"curriculum={curriculum_name} "
            f"scenario={context.scenario_manager.get_index()}"
        )
        return current_state

    @staticmethod
    def _train_result_episode_count(train_result: dict[str, object]) -> int:
        """Train the result episode count owned by this default ray model policy."""
        env_runners = train_result["env_runners"]
        if not isinstance(env_runners, dict):
            raise TypeError("Ray train result 'env_runners' must be a mapping.")
        metrics = canonical_training_metrics_from_result(env_runners)
        terminated = metrics.get("episode_info.episode_terminated.sum")
        truncated = metrics.get("episode_info.episode_truncated.sum")
        if terminated is None or truncated is None:
            raise TypeError(
                "Ray episode termination counters are missing from the "
                "legacy or schema-v2 metrics."
            )
        terminated_count = int(round(float(terminated)))
        truncated_count = int(round(float(truncated)))
        return terminated_count + truncated_count
