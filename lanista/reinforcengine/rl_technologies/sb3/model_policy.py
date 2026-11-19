# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed model and evaluation policies for the shared SB3 lifecycle."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    CheckpointMetadataError,
    load_checkpoint_restore_manifest,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)

CheckpointPathResolver = Callable[[str, str], str | None]
CurriculumLearningRateResolver = Callable[[object], float]
CurriculumLearningRateApplier = Callable[[object, float], bool]
SB3MetricSnapshot = dict[str, float]
SB3CheckpointObservationRow = dict[str, object]


class SB3PolicyLogger(Protocol):
    """Logging surface required by model-local SB3 policies."""

    name: str
    level: int

    def info(self, message: str) -> object:
        """Record an informational lifecycle event."""

    def warning(self, message: str) -> object:
        """Record an optional-quality degradation."""


class SB3CheckpointMetricProvider(Protocol):
    """Expose optional metric snapshots without leaking callback internals."""

    def metric_snapshots_for_checkpoint(
        self,
    ) -> list[SB3MetricSnapshot]:
        """Return snapshots accumulated before the next checkpoint."""


class SB3CheckpointObservationUnavailable(RuntimeError):
    """Report an expected loss of optional checkpoint observation telemetry."""


class SB3CheckpointObservationProvider(Protocol):
    """Expose optional observation rows without leaking callback internals."""

    def episode_observations_for_checkpoint(
        self,
    ) -> list[SB3CheckpointObservationRow]:
        """Return rows in capture order or raise the typed telemetry error."""


@dataclass(frozen=True, slots=True)
class SB3TrainingRestore:
    """Validated source and state used to continue one training session."""

    selected_session_name: str | None
    source_session_name: str | None
    source_checkpoint_name: str | None
    scenario_index: int | None
    curriculum_state: object | None
    restore_curriculum_state: bool

    @property
    def has_source_checkpoint(self) -> bool:
        """Return the has source checkpoint exposed by this sb3 training restore."""
        return bool(self.source_session_name and self.source_checkpoint_name)

    @property
    def source_matches_selected_session(self) -> bool:
        """Return the source matches selected session exposed by this sb3 training restore."""
        return (
            self.has_source_checkpoint
            and self.source_session_name == self.selected_session_name
        )


@dataclass(frozen=True, slots=True)
class SB3ScenarioSwitchingOptions:
    """Optional callbacks applied when SB3 changes curriculum state."""

    learning_rate_resolver: CurriculumLearningRateResolver | None = None
    learning_rate_applier: CurriculumLearningRateApplier | None = None


@dataclass(frozen=True, slots=True)
class SB3TensorboardOptions:
    """Model-specific detail level for the shared TensorBoard callback."""

    record_rollout_stat_details: bool = False


@dataclass(frozen=True, slots=True)
class SB3EvaluationContext:
    """Resources exposed to a model-local diagnostic evaluation."""

    runtime: BaseModelRuntime
    logger: SB3PolicyLogger
    model_configuration: object
    model_manifest: ModelManifest
    scenario_manager: ScenarioManager


class SB3EvaluationPolicy(Protocol):
    """Create and run an evaluation while the shared core owns cleanup."""

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return optional evaluation failures observed by the policy."""

    def create_environment(self, context: SB3EvaluationContext) -> object:
        """Create the environment before the shared core publishes ownership."""

    def evaluate(
        self,
        environment: object,
        context: SB3EvaluationContext,
    ) -> None:
        """Run the diagnostic against an already published environment."""


class SB3ModelPolicy(Protocol):
    """Model-specific decisions injected into the shared SB3 lifecycle."""

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return optional training/checkpoint failures seen by the policy."""

    def resolve_training_restore(
        self,
        runtime: BaseModelRuntime,
        configuration: object,
        *,
        models_directory: str,
        resolve_checkpoint_path: CheckpointPathResolver,
    ) -> SB3TrainingRestore:
        """Resolve and validate the checkpoint source for one training run."""

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return model-specific environment constructor options."""

    def scenario_switching_options(self) -> SB3ScenarioSwitchingOptions:
        """Return optional curriculum transition hooks."""

    def tensorboard_options(self) -> SB3TensorboardOptions:
        """Return the desired shared TensorBoard detail level."""

    def checkpoint_metric_snapshots(
        self,
        checkpoint_callback: SB3CheckpointMetricProvider | None,
        logger: SB3PolicyLogger,
    ) -> list[SB3MetricSnapshot]:
        """Return optional model-specific checkpoint metric snapshots."""


class DefaultSB3ModelPolicy:
    """Preserve the Eve/Jasmine SB3 lifecycle without model-local logic."""

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this default sb3 model policy."""
        return frozenset()

    def resolve_training_restore(
        self,
        runtime: BaseModelRuntime,
        configuration: object,
        *,
        models_directory: str,
        resolve_checkpoint_path: CheckpointPathResolver,
    ) -> SB3TrainingRestore:
        """Resolve the training restore required by this default sb3 model policy."""
        selected_session_name = self._optional_name(
            getattr(configuration, "selected_default_model_name", None)
        )
        source_session_name = selected_session_name
        source_checkpoint_name = self._optional_name(
            getattr(configuration, "selected_default_checkpoint_name", None)
        )
        source_resolver = getattr(runtime, "get_training_checkpoint_source", None)
        if callable(source_resolver):
            resolved_source = source_resolver()
            if not isinstance(resolved_source, tuple) or len(resolved_source) != 2:
                raise TypeError(
                    "Training checkpoint source resolver must return two names."
                )
            source_session_name = self._optional_name(resolved_source[0])
            source_checkpoint_name = self._optional_name(resolved_source[1])

        checkpoint_metadata: dict[str, object] = {}
        if source_session_name and source_checkpoint_name:
            checkpoint_directory = os.path.join(
                models_directory,
                source_session_name,
                "checkpoints",
            )
            checkpoint_path = resolve_checkpoint_path(
                checkpoint_directory,
                source_checkpoint_name,
            )
            if not checkpoint_path:
                raise CheckpointMetadataError(
                    "Selected checkpoint artifact is unavailable."
                )
            checkpoint_metadata = load_checkpoint_restore_manifest(
                checkpoint_path
            )

        scenario_index = checkpoint_metadata.get("scenario_index")
        return SB3TrainingRestore(
            selected_session_name=selected_session_name,
            source_session_name=source_session_name,
            source_checkpoint_name=source_checkpoint_name,
            scenario_index=(
                scenario_index if type(scenario_index) is int else None
            ),
            curriculum_state=checkpoint_metadata.get("curriculum_state"),
            restore_curriculum_state=bool(checkpoint_metadata),
        )

    @staticmethod
    def _optional_name(value: object) -> str | None:
        """Return the optional name value owned by this default sb3 model policy."""
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    def training_environment_options(
        self,
        configuration: object,
        environment_type: type[object],
    ) -> dict[str, object]:
        """Return the training environment options value owned by this default sb3 model policy."""
        _ = configuration, environment_type
        return {}

    def scenario_switching_options(self) -> SB3ScenarioSwitchingOptions:
        """Return the scenario switching options value owned by this default sb3 model policy."""
        return SB3ScenarioSwitchingOptions()

    def tensorboard_options(self) -> SB3TensorboardOptions:
        """Return the tensorboard options value owned by this default sb3 model policy."""
        return SB3TensorboardOptions()

    def checkpoint_metric_snapshots(
        self,
        checkpoint_callback: SB3CheckpointMetricProvider | None,
        logger: SB3PolicyLogger,
    ) -> list[SB3MetricSnapshot]:
        """Checkpoint the metric snapshots owned by this default sb3 model policy."""
        _ = checkpoint_callback, logger
        return []
