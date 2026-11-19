# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Ray checkpoint callback using the shared runtime storage repository."""

from __future__ import annotations

import os
from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import Protocol

import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    canonical_training_metrics_from_result,
)
from lanista.reinforcengine.rl_model.checkpoints.metadata import CheckpointStateReadError
from lanista.reinforcengine.rl_model.checkpoints.runtime_repository import (
    CheckpointArtifactKind,
    RuntimeCheckpointRepository,
)
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_lifecycle import (
    write_checkpoint_metric_stats_csvs,
)


class _RemoteGetter(Protocol):
    """Define the typed boundary for remote getter."""

    def remote(self) -> ray.ObjectRef[object]:
        """Schedule one actor state read."""


class _StateActor(Protocol):
    """Define the typed boundary for state actor."""

    get: _RemoteGetter


class _CheckpointAlgorithm(Protocol):
    """Define the typed boundary for checkpoint algorithm."""

    def save(self, checkpoint_path: str) -> object:
        """Persist one checkpoint at the supplied staging path."""


class _RemoteCall(Protocol):
    """Ray actor method used by the optional episode trace gate."""

    def remote(self, checkpoint_path: str) -> ray.ObjectRef[object]:
        """Schedule one actor operation."""


class _EpisodeTraceCaptureGate(Protocol):
    """Remote gate surface used after a checkpoint commit."""

    arm: _RemoteCall


@dataclass(frozen=True)
class _TrainingStepWindow:
    """Validated lifetime and current-iteration environment step counts."""

    lifetime: int
    iteration: int

    @property
    def previous(self) -> int:
        """Return the lifetime step at the start of this iteration."""

        return self.lifetime - self.iteration


@dataclass(frozen=True, slots=True)
class _SavedCheckpoint:
    """Committed checkpoint identity and final trace target path."""

    name: str
    path: str


class RayCheckpointSavingCallback(DefaultCallbacks):
    """Handle ray checkpoint saving lifecycle callbacks."""

    def __init__(
        self,
        scenario_index_state: _StateActor,
        save_freq: int,
        checkpoint_directory: str,
        name_prefix: str = "rl_model",
        curriculum_index_state: _StateActor | None = None,
        parent_checkpoint_name: str | None = None,
        base_step_offset: int = 0,
        model_package_directory: str | None = None,
        episode_trace_capture_gate: _EpisodeTraceCaptureGate | None = None,
    ) -> None:
        """Initialize the ray checkpoint saving callback and its owned state."""
        self.save_freq = int(save_freq)
        self.checkpoint_directory = checkpoint_directory
        self.name_prefix = name_prefix
        self._scenario_index_state = scenario_index_state
        self._curriculum_index_state = curriculum_index_state
        self._parent_checkpoint_name = parent_checkpoint_name
        self._base_step_offset = max(0, int(base_step_offset or 0))
        self._model_package_directory = model_package_directory
        self._checkpoint_repository = RuntimeCheckpointRepository(
            model_package_directory or os.path.dirname(checkpoint_directory)
        )
        self._episode_trace_capture_gate = episode_trace_capture_gate
        self._historical_metric_snapshots: list[dict[str, float]] = []
        self._quality_degradations: set[str] = set()

    def _effective_step(self, step_num: int | float | str | None) -> int:
        """Return the effective step value owned by this ray checkpoint saving callback."""
        raw_step = int(float(step_num or 0))
        if self._base_step_offset > 0 and raw_step <= self._base_step_offset:
            return self._base_step_offset + raw_step
        return raw_step

    def _checkpoint_path(
        self,
        scenario_num: int,
        step_num: int | float | str | None,
    ) -> str:
        """Checkpoint the path owned by this ray checkpoint saving callback."""
        return os.path.join(
            self.checkpoint_directory,
            f"{self.name_prefix}{scenario_num}_checkpoint_{self._effective_step(step_num)}_steps",
        )

    @staticmethod
    def _actor_value(
        actor: _StateActor | None,
        *,
        field_name: str,
        allow_none: bool = False,
    ) -> object | None:
        """Return the actor value value owned by this ray checkpoint saving callback."""
        if actor is None:
            raise CheckpointStateReadError(
                f"Checkpoint {field_name} state actor is unavailable."
            )
        try:
            value = ray.get(actor.get.remote())
        except Exception as exc:
            raise CheckpointStateReadError(
                f"Checkpoint {field_name} state could not be read."
            ) from exc
        if value is None and not allow_none:
            raise CheckpointStateReadError(
                f"Checkpoint {field_name} state is missing."
            )
        return value

    def _scenario_index(self) -> int:
        """Return the scenario index value owned by this ray checkpoint saving callback."""
        value = self._actor_value(
            self._scenario_index_state,
            field_name="scenario",
        )
        if type(value) is not int or value < 0:
            raise CheckpointStateReadError(
                "Checkpoint scenario state is invalid."
            )
        return value

    @staticmethod
    def _step_counter(value: object, *, field_name: str) -> int:
        """Return the step counter value owned by this ray checkpoint saving callback."""
        if isinstance(value, bool) or not isinstance(value, Real):
            raise CheckpointStateReadError(
                f"Checkpoint {field_name} step counter is invalid."
            )
        if isinstance(value, Integral):
            counter = int(value)
        else:
            numeric_value = float(value)
            if not isfinite(numeric_value) or not numeric_value.is_integer():
                raise CheckpointStateReadError(
                    f"Checkpoint {field_name} step counter is invalid."
                )
            counter = int(numeric_value)
        if counter < 0:
            raise CheckpointStateReadError(
                f"Checkpoint {field_name} step counter is invalid."
            )
        return counter

    def _record_metric_snapshot(self, result: dict[str, object]) -> None:
        """Record the metric snapshot owned by this ray checkpoint saving callback."""
        env_runners = result.get("env_runners", {}) if isinstance(result, dict) else {}
        flattened = (
            canonical_training_metrics_from_result(env_runners)
            if isinstance(env_runners, dict)
            else {}
        )
        if flattened:
            self._historical_metric_snapshots.append(flattened)

    def _training_step_window(
        self,
        result: dict[str, object],
    ) -> _TrainingStepWindow | None:
        """Return the training step window value owned by this ray checkpoint saving callback."""
        env_runners = result.get("env_runners")
        if not isinstance(env_runners, dict):
            print(
                "[ray_checkpoint_saving_callback] WARNING: "
                "'env_runners' missing from train result."
            )
            return None

        lifetime = self._step_counter(
            env_runners.get("num_env_steps_sampled_lifetime", 0),
            field_name="lifetime",
        )
        iteration = self._step_counter(
            env_runners.get("num_env_steps_sampled", 0),
            field_name="iteration",
        )
        if iteration > lifetime:
            raise CheckpointStateReadError(
                "Checkpoint iteration step counter exceeds lifetime steps."
            )
        return _TrainingStepWindow(
            lifetime=lifetime,
            iteration=iteration,
        )

    def _checkpoint_is_due(self, window: _TrainingStepWindow) -> bool:
        """Checkpoint the is due owned by this ray checkpoint saving callback."""
        return self.save_freq > 0 and (
            window.lifetime // self.save_freq
        ) > (
            window.previous // self.save_freq
        )

    def _save_checkpoint(
        self,
        *,
        algorithm: _CheckpointAlgorithm,
        lifetime_step: int,
    ) -> _SavedCheckpoint:
        """Persist the checkpoint owned by this ray checkpoint saving callback."""
        scenario_num = self._scenario_index()
        curriculum_state = self._actor_value(
            self._curriculum_index_state,
            field_name="curriculum",
            allow_none=True,
        )
        final_model_path = self._checkpoint_path(
            scenario_num,
            lifetime_step,
        )
        checkpoint_name = os.path.basename(final_model_path)
        effective_step = self._effective_step(lifetime_step)

        with self._checkpoint_repository.mutation() as mutation:
            model_path = mutation.prepare_checkpoint_target(
                final_model_path,
                artifact_kind=CheckpointArtifactKind.DIRECTORY,
            )
            algorithm.save(model_path)
            mutation.write_checkpoint_manifest(
                model_path,
                checkpoint_name=checkpoint_name,
                checkpoint_kind="checkpoint",
                framework_uuid="ray",
                scenario_index=scenario_num,
                curriculum_state=curriculum_state,
                step_count=effective_step,
                parent_checkpoint_name=self._parent_checkpoint_name,
            )
            write_checkpoint_metric_stats_csvs(
                model_path,
                self._historical_metric_snapshots,
            )
            if self._model_package_directory:
                mutation.update_last_checkpoint_metadata(
                    checkpoint_name,
                    "checkpoint",
                )
        return _SavedCheckpoint(
            name=checkpoint_name,
            path=final_model_path,
        )

    def _arm_episode_trace_capture(self, checkpoint_path: str) -> None:
        """Arm raw capture only after the checkpoint transaction committed."""

        gate = self._episode_trace_capture_gate
        if gate is None:
            return
        try:
            ray.get(gate.arm.remote(checkpoint_path))
        except Exception as error:
            event = "ray_checkpoint_trace_capture_arm_failed"
            if event in self._quality_degradations:
                return
            self._quality_degradations.add(event)
            print(
                "[ray_checkpoint_saving_callback] WARNING: "
                f"event={event} quality=degraded "
                f"error_type={type(error).__name__} error={error}"
            )

    def on_train_result(
        self,
        *,
        algorithm: _CheckpointAlgorithm,
        result: dict[str, object],
        **kwargs: object,
    ) -> None:
        """Handle the train result event for this ray checkpoint saving callback."""
        _ = kwargs
        self._record_metric_snapshot(result)
        step_window = self._training_step_window(result)
        if step_window is None or not self._checkpoint_is_due(step_window):
            return
        saved_checkpoint = self._save_checkpoint(
            algorithm=algorithm,
            lifetime_step=step_window.lifetime,
        )
        self._parent_checkpoint_name = saved_checkpoint.name
        self._arm_episode_trace_capture(saved_checkpoint.path)
