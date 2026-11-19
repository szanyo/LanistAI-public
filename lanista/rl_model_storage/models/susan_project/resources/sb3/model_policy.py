# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan-specific strategy hooks for the shared SB3 lifecycle."""

from __future__ import annotations

from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    CheckpointPathResolver,
    SB3CheckpointMetricProvider,
    SB3MetricSnapshot,
    SB3PolicyLogger,
    SB3ScenarioSwitchingOptions,
    SB3TensorboardOptions,
    SB3TrainingRestore,
)

from ..utils.training.runtime_config import (
    RUNTIME_CONFIG,
    apply_sb3_learning_rate,
    ppo_learning_rate_for_curriculum_state,
)
from ..utils.training.training_runtime_helper import TrainingRuntimeHelper


class SusanSB3ModelPolicy:
    """Configure Susan without duplicating SB3 algorithm ownership."""

    def __init__(self) -> None:
        """Initialize the susan sb3 model policy and its owned state."""
        self._quality_degradations: set[str] = set()

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this susan sb3 model policy."""
        return frozenset(self._quality_degradations)

    def resolve_training_restore(
        self,
        runtime: BaseModelRuntime,
        configuration: object,
        *,
        models_directory: str,
        resolve_checkpoint_path: CheckpointPathResolver,
    ) -> SB3TrainingRestore:
        """Resolve the training restore required by this susan sb3 model policy."""
        checkpoint_source = (
            TrainingRuntimeHelper.resolve_training_checkpoint_source(
                runtime,
                configuration,
            )
        )
        checkpoint_metadata = (
            TrainingRuntimeHelper.load_sb3_source_checkpoint_metadata(
                models_directory=models_directory,
                checkpoint_source=checkpoint_source,
                resolve_checkpoint_path=resolve_checkpoint_path,
            )
        )
        restored_state = TrainingRuntimeHelper.restored_training_state(
            checkpoint_metadata
        )
        return SB3TrainingRestore(
            selected_session_name=checkpoint_source.selected_session_name,
            source_session_name=checkpoint_source.source_session_name,
            source_checkpoint_name=checkpoint_source.source_checkpoint_name,
            scenario_index=restored_state.scenario_index,
            curriculum_state=restored_state.curriculum_state,
            restore_curriculum_state=(
                restored_state.curriculum_state is not None
            ),
        )

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return the training environment options value owned by this susan sb3 model policy."""
        default_jitter = int(
            getattr(
                environment_type,
                "_TRAINING_WARMUP_JITTER_SECONDS",
                0,
            )
        )
        return {
            "training_warmup_jitter_seconds": (
                RUNTIME_CONFIG.resolve_training_warmup_jitter_seconds(
                    configuration,
                    default_jitter,
                )
            )
        }

    def scenario_switching_options(self) -> SB3ScenarioSwitchingOptions:
        """Return the scenario switching options value owned by this susan sb3 model policy."""
        return SB3ScenarioSwitchingOptions(
            learning_rate_resolver=ppo_learning_rate_for_curriculum_state,
            learning_rate_applier=apply_sb3_learning_rate,
        )

    def tensorboard_options(self) -> SB3TensorboardOptions:
        """Return the tensorboard options value owned by this susan sb3 model policy."""
        return SB3TensorboardOptions(record_rollout_stat_details=False)

    def checkpoint_metric_snapshots(
        self,
        checkpoint_callback: SB3CheckpointMetricProvider | None,
        logger: SB3PolicyLogger,
    ) -> list[SB3MetricSnapshot]:
        """Checkpoint the metric snapshots owned by this susan sb3 model policy."""
        if checkpoint_callback is None:
            return []
        try:
            return checkpoint_callback.metric_snapshots_for_checkpoint()
        except Exception as exc:
            self._record_quality_degradation(
                "checkpoint_metric_snapshots_unavailable",
                logger,
                exc,
            )
            return []

    def _record_quality_degradation(
        self,
        event: str,
        logger: SB3PolicyLogger,
        exc: BaseException,
    ) -> None:
        """Record the quality degradation owned by this susan sb3 model policy."""
        first_occurrence = event not in self._quality_degradations
        self._quality_degradations.add(event)
        if first_occurrence:
            logger.warning(
                f"event={event} quality=degraded "
                f"error_type={type(exc).__name__} error={exc}"
            )
