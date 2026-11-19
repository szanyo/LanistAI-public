# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan-specific strategy hooks for the shared Ray lifecycle."""

from __future__ import annotations

from typing import cast

import ray

from lanista.reinforcengine.rl_technologies.ray.model_policy import (
    CurriculumState,
    RayCurriculumRestoreManager,
    RayCurriculumStateActor,
    RayPolicyLogger,
    RayTrainingWindowContext,
    curriculum_ready_for_update,
)

from ..utils.training.runtime_config import (
    RUNTIME_CONFIG,
    _RayLearningRateAlgorithmPort,
    apply_ray_learning_rate,
    build_ray_checkpoint_saving_frequency,
    ppo_learning_rate_for_curriculum_state,
)


class SusanRayModelPolicy:
    """Configure Susan without duplicating Ray algorithm ownership."""

    training_scenario_default_index = 0

    def __init__(self) -> None:
        """Initialize the susan ray model policy and its owned state."""
        self._last_learning_rate: float | None = None
        self._quality_degradations: set[str] = set()

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this susan ray model policy."""
        return frozenset(self._quality_degradations)

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return the training environment options value owned by this susan ray model policy."""
        default_jitter = int(
            getattr(environment_type, "_TRAINING_WARMUP_JITTER_SECONDS", 0)
        )
        return {
            "training_warmup_jitter_seconds": (
                RUNTIME_CONFIG.resolve_training_warmup_jitter_seconds(
                    configuration,
                    default_jitter,
                )
            )
        }

    def checkpoint_frequency(
        self,
        configured_frequency: int,
        logger: RayPolicyLogger,
    ) -> int:
        """Checkpoint the frequency owned by this susan ray model policy."""
        resolved_frequency = build_ray_checkpoint_saving_frequency(
            configured_frequency
        )
        if resolved_frequency != configured_frequency:
            logger.info(
                "Ray checkpoint save frequency scaled by parallel runners: "
                f"{configured_frequency} -> {resolved_frequency} env steps"
            )
        return resolved_frequency

    def restore_curriculum_state(
        self,
        curriculum_manager: RayCurriculumRestoreManager,
        state_actor: RayCurriculumStateActor,
        restored_state: CurriculumState,
    ) -> CurriculumState:
        """Restore the curriculum state owned by this susan ray model policy."""
        if restored_state is None:
            return None
        curriculum_manager.restore_state(restored_state)
        # The read is an ordering barrier for the preceding actor write.
        ray.get(state_actor.get.remote())
        return restored_state

    def on_algorithm_ready(
        self,
        algorithm: object,
        current_state: CurriculumState,
        logger: RayPolicyLogger,
    ) -> None:
        """Handle the algorithm ready event for this susan ray model policy."""
        self._apply_curriculum_learning_rate(
            algorithm,
            current_state,
            logger,
        )

    def record_training_result(
        self,
        runtime: object,
        train_result: dict[str, object],
        logger: RayPolicyLogger,
    ) -> bool:
        """Let the filtered Tune logger own the sole Ray event projection."""

        _ = runtime, train_result, logger
        return True

    def checkpoint_step(self, train_result: object) -> int:
        """Checkpoint the step owned by this susan ray model policy."""
        if not isinstance(train_result, dict):
            return 0
        env_runners = train_result.get("env_runners", {})
        if not isinstance(env_runners, dict):
            raise TypeError("Ray train result 'env_runners' must be a mapping.")
        return int(float(env_runners.get("num_env_steps_sampled_lifetime", 0)))

    def process_training_window(
        self,
        context: RayTrainingWindowContext,
    ) -> CurriculumState:
        """Return the process training window value owned by this susan ray model policy."""
        current_state = context.curriculum_manager.current_state()
        curriculum_ready = curriculum_ready_for_update(
            context.rolling_stats,
            current_state,
        )
        context.curriculum_manager.evaluate_and_update(curriculum_ready)
        current_state = context.curriculum_manager.current_state()
        self._apply_curriculum_learning_rate(
            context.algorithm,
            current_state,
            context.logger,
        )

        scenario_ready = context.rolling_stats.scenario_ready(
            current_state
        )
        episode_counts = context.rolling_stats.episode_count_summary()
        train_iteration = context.train_result.get("training_iteration", "?")
        context.logger.info(
            f"[Progress] train_iter={train_iteration} "
            f"env_steps={context.env_steps} "
            f"total_episodes={episode_counts['total']} "
            f"terminated={episode_counts['terminated']} "
            f"truncated={episode_counts['truncated']} "
            f"curriculum={self._curriculum_name(current_state)} "
            f"scenario={context.scenario_manager.get_index()} "
            f"metrics_ready={context.metrics_ready} "
            f"scenario_ready={scenario_ready}"
        )

        if context.metrics_ready and scenario_ready:
            context.scenario_manager.next()
            context.curriculum_manager.reset_for_scenario_switch()
            context.rolling_stats.reset()
        return current_state

    def _apply_curriculum_learning_rate(
        self,
        algorithm: object,
        current_state: CurriculumState,
        logger: RayPolicyLogger,
    ) -> None:
        """Apply the curriculum learning rate transition owned by this susan ray model policy."""
        learning_rate = ppo_learning_rate_for_curriculum_state(current_state)
        if self._last_learning_rate == learning_rate:
            return
        # The utility validates learner_group before dispatch; this cast only
        # adapts the opaque framework-policy boundary to that checked port.
        learning_rate_algorithm = cast(
            _RayLearningRateAlgorithmPort,
            algorithm,
        )
        if apply_ray_learning_rate(
            learning_rate_algorithm,
            learning_rate,
        ):
            self._last_learning_rate = learning_rate
            logger.info(
                f"[PPO] Learning rate set to {learning_rate:.2e} "
                f"for curriculum={self._curriculum_name(current_state)}"
            )
            return
        self._record_quality_degradation(
            "ray_curriculum_learning_rate_not_applied",
            logger,
        )

    @staticmethod
    def _curriculum_name(current_state: CurriculumState) -> str:
        """Return the curriculum name value owned by this susan ray model policy."""
        if current_state is None:
            return "full"
        return str(
            current_state.get("name")
            or current_state.get("state_name")
            or "curriculum"
        )

    def _record_quality_degradation(
        self,
        event: str,
        logger: RayPolicyLogger,
        exc: BaseException | None = None,
    ) -> None:
        """Record the quality degradation owned by this susan ray model policy."""
        first_occurrence = event not in self._quality_degradations
        self._quality_degradations.add(event)
        if not first_occurrence:
            return
        error_fields = (
            ""
            if exc is None
            else f" error_type={type(exc).__name__} error={exc}"
        )
        logger.warning(f"event={event} quality=degraded{error_fields}")
