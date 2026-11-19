# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable contracts shared by model-storage administration commands."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.sessions.deletion_unit_of_work import (
    SessionDeletionSelection,
)


@dataclass(frozen=True, slots=True)
class ModelAdministrationOwner:
    """Stable identity of the model that opened an administration workflow."""

    model_uuid: str


@dataclass(frozen=True, slots=True)
class ModelDeletionOutcome:
    """Describe durable model deletion and active-runtime cleanup."""

    model_uuid: str
    model_name: str
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    runtime_state_cleared: bool


@dataclass(frozen=True, slots=True)
class SessionDeletionOutcome:
    """Describe durable session deletion and runtime projection state."""

    session_name: str
    remaining_sessions: tuple[str, ...]
    selection: SessionDeletionSelection
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    projection_synchronized: bool


@dataclass(frozen=True, slots=True)
class CheckpointDeletionOutcome:
    """Describe durable checkpoint deletion and its runtime projection."""

    session_name: str
    checkpoint_name: str
    replacement_checkpoint_name: str | None
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    projection_synchronized: bool


@dataclass(frozen=True, slots=True)
class CheckpointRenameOutcome:
    """Describe durable checkpoint rename and its runtime projection."""

    session_name: str
    previous_checkpoint_name: str
    checkpoint_name: str
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    projection_synchronized: bool


@dataclass(frozen=True, slots=True)
class CheckpointAnnotationOutcome:
    """Describe a committed checkpoint annotation update."""

    session_name: str
    checkpoint_name: str
    annotation: str | None
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    projection_synchronized: bool
