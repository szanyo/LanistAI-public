# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Checkpoint and curriculum restore support shared by Susan frameworks."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    CheckpointMetadataError,
    load_checkpoint_restore_manifest,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    CurriculumDefinition,
    CurriculumState,
    resolve_curriculum_definition,
)


@dataclass(frozen=True)
class TrainingCheckpointSource:
    """Carry the data required for training checkpoint source."""

    selected_session_name: str | None
    selected_checkpoint_name: str | None
    source_session_name: str | None
    source_checkpoint_name: str | None

    @property
    def has_source_checkpoint(self) -> bool:
        """Return the has source checkpoint exposed by this training checkpoint source."""
        return bool(self.source_session_name and self.source_checkpoint_name)

    @property
    def source_matches_selected_session(self) -> bool:
        """Return the source matches selected session exposed by this training checkpoint source."""
        return (
            self.has_source_checkpoint
            and self.source_session_name == self.selected_session_name
        )


@dataclass(frozen=True)
class RestoredTrainingState:
    """Represent runtime state for restored training."""

    scenario_index: int | None
    curriculum_state: object | None


class TrainingRuntimeHelper:
    """Framework-neutral Susan training runtime metadata helpers."""

    @staticmethod
    def resolve_training_checkpoint_source(
        rl_model_wrapper: object,
        configuration: object,
    ) -> TrainingCheckpointSource:
        """Resolve the training checkpoint source required by this training runtime helper."""
        selected_session_name = TrainingRuntimeHelper._optional_name(
            getattr(configuration, "selected_default_model_name", None)
        )
        selected_checkpoint_name = TrainingRuntimeHelper._optional_name(
            getattr(configuration, "selected_default_checkpoint_name", None)
        )
        source_session_name = selected_session_name
        source_checkpoint_name = selected_checkpoint_name
        source_resolver = getattr(rl_model_wrapper, "get_training_checkpoint_source", None)
        if callable(source_resolver):
            resolved_source = source_resolver()
            if not isinstance(resolved_source, tuple) or len(resolved_source) != 2:
                raise TypeError(
                    "Training checkpoint source resolver must return two names."
                )
            source_session_name = TrainingRuntimeHelper._optional_name(
                resolved_source[0]
            )
            source_checkpoint_name = TrainingRuntimeHelper._optional_name(
                resolved_source[1]
            )
        return TrainingCheckpointSource(
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
            source_session_name=source_session_name,
            source_checkpoint_name=source_checkpoint_name,
        )

    @staticmethod
    def _optional_name(value: object) -> str | None:
        """Return the optional name value owned by this training runtime helper."""
        if value is None:
            return None
        normalized = str(value).strip()
        return normalized or None

    @staticmethod
    def load_checkpoint_metadata(
        checkpoint_path: str | os.PathLike[str] | None,
    ) -> dict[str, Any]:
        """Load the checkpoint metadata owned by this training runtime helper."""
        return (
            load_checkpoint_restore_manifest(os.fspath(checkpoint_path))
            if checkpoint_path
            else {}
        )

    @staticmethod
    def load_sb3_source_checkpoint_metadata(
        *,
        models_directory: str,
        checkpoint_source: TrainingCheckpointSource,
        resolve_checkpoint_path: Callable[[str, str], str | None],
    ) -> dict[str, Any]:
        """Load the sb3 source checkpoint metadata owned by this training runtime helper."""
        if not checkpoint_source.has_source_checkpoint:
            return {}
        source_checkpoint_directory = os.path.join(
            str(models_directory),
            str(checkpoint_source.source_session_name),
            "checkpoints",
        )
        source_checkpoint_path = resolve_checkpoint_path(
            source_checkpoint_directory,
            str(checkpoint_source.source_checkpoint_name),
        )
        if not source_checkpoint_path:
            raise CheckpointMetadataError(
                "Selected checkpoint artifact is unavailable."
            )
        return TrainingRuntimeHelper.load_checkpoint_metadata(source_checkpoint_path)

    @staticmethod
    def restored_scenario_index(
        checkpoint_metadata: dict[str, Any],
    ) -> int | None:
        """Return the restored scenario index value owned by this training runtime helper."""
        if not checkpoint_metadata:
            return None
        value = checkpoint_metadata.get("scenario_index")
        if type(value) is not int or value < 0:
            raise CheckpointMetadataError(
                "Checkpoint restore metadata field 'scenario_index' is invalid."
            )
        return value

    @staticmethod
    def restored_curriculum_state(
        checkpoint_metadata: dict[str, Any],
    ) -> object | None:
        """Return the restored curriculum state value owned by this training runtime helper."""
        if not checkpoint_metadata:
            return None
        if "curriculum_state" not in checkpoint_metadata:
            raise CheckpointMetadataError(
                "Checkpoint restore metadata field 'curriculum_state' is missing."
            )
        return cast(object | None, checkpoint_metadata["curriculum_state"])

    @staticmethod
    def restored_training_state(
        checkpoint_metadata: dict[str, Any],
    ) -> RestoredTrainingState:
        """Return the restored training state value owned by this training runtime helper."""
        return RestoredTrainingState(
            scenario_index=TrainingRuntimeHelper.restored_scenario_index(checkpoint_metadata),
            curriculum_state=TrainingRuntimeHelper.restored_curriculum_state(checkpoint_metadata),
        )

    @staticmethod
    def curriculum_definition(
        training_conditions: object,
        state_completer: Callable[
            [list[CurriculumState]],
            list[CurriculumState],
        ]
        | None = None,
    ) -> CurriculumDefinition:
        """Return the curriculum definition value owned by this training runtime helper."""
        if state_completer is None:
            return resolve_curriculum_definition(training_conditions)
        return resolve_curriculum_definition(
            training_conditions,
            state_completer=state_completer,
        )
