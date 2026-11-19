# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable contracts for run-configuration administration workflows."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from lanista.gui.contracts import UiNotification
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationDialogAction,
)
from lanista.reinforcengine.application_errors import OperationResult
from lanista.reinforcengine.rl_model.administration_contracts import (
    CheckpointDeletionOutcome,
    CheckpointRenameOutcome,
    ModelAdministrationOwner,
    SessionDeletionOutcome,
)


class ConfigurationModelAdministrationPort(Protocol):
    """Model-owner boundary required by the configuration coordinator."""

    def is_owner_current(self, owner: ModelAdministrationOwner) -> bool:
        """Return whether owner current applies to this configuration model administration port."""
        ...


class ConfigurationSessionAdministrationPort(Protocol):
    """Session command boundary required by the configuration coordinator."""

    def delete_session(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[SessionDeletionOutcome]:
        """Delete the session through this configuration session administration port's safety
        boundary.
        """
        ...


class ConfigurationCheckpointAdministrationPort(Protocol):
    """Checkpoint command boundary required by the configuration coordinator."""

    def delete_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        checkpoint_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[CheckpointDeletionOutcome]:
        """Delete the checkpoint through this configuration checkpoint administration port's safety
        boundary.
        """
        ...

    def rename_checkpoint(
        self,
        owner: ModelAdministrationOwner,
        session_name: str | None,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[CheckpointRenameOutcome]:
        """Rename the checkpoint owned by this configuration checkpoint administration port."""
        ...


class SessionReferenceDisplayPort(Protocol):
    """Read boundary used to present a session deletion confirmation."""

    def get_session_reference_display(self, session_name: str | None) -> str:
        """Return the session reference display exposed by this session reference display port."""
        ...


@dataclass(frozen=True, slots=True)
class ConfigurationAdministrationBoundary:
    """Application ports resolved lazily by the UI composition adapter."""

    models: ConfigurationModelAdministrationPort
    session_commands: ConfigurationSessionAdministrationPort
    checkpoint_commands: ConfigurationCheckpointAdministrationPort
    session_references: SessionReferenceDisplayPort


@dataclass(frozen=True, slots=True)
class ConfigurationAdministrationState:
    """Authoritative selection and callback generation for one parent dialog."""

    owner: ModelAdministrationOwner | None
    generation: int
    session_name: str | None
    checkpoint_name: str | None


@dataclass(frozen=True, slots=True)
class SessionSelectorIdentity:
    """Identity captured when a session selector is opened."""

    owner: ModelAdministrationOwner
    generation: int


@dataclass(frozen=True, slots=True)
class CheckpointSelectorIdentity:
    """Identity and session captured when a checkpoint selector is opened."""

    owner: ModelAdministrationOwner
    generation: int
    session_name: str


class SessionAdministrationAction(StrEnum):
    """Supported session selector outcomes."""

    SELECT = "select"
    EDIT = "edit"
    DELETE = "delete"


@dataclass(frozen=True, slots=True)
class SessionAdministrationEvent:
    """Framework-independent session selector result."""

    action: SessionAdministrationAction
    session_name: str | None

    @classmethod
    def from_raw(
        cls,
        action: str,
        session_name: str | None,
    ) -> SessionAdministrationEvent | None:
        """Create this session administration event's value from raw."""
        try:
            parsed_action = SessionAdministrationAction(action)
        except ValueError:
            return None
        return cls(parsed_action, session_name)


class CheckpointAdministrationAction(StrEnum):
    """Supported checkpoint selector outcomes."""

    SELECT = "select"
    EDIT = "edit"
    DELETE = "delete"
    RENAME = "rename"


@dataclass(frozen=True, slots=True)
class CheckpointAdministrationEvent:
    """Framework-independent checkpoint selector result."""

    action: CheckpointAdministrationAction
    checkpoint_name: str | None
    renamed_checkpoint_name: str | None = None

    @classmethod
    def from_raw(
        cls,
        action: str,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
    ) -> CheckpointAdministrationEvent | None:
        """Create this checkpoint administration event's value from raw."""
        try:
            parsed_action = CheckpointAdministrationAction(action)
        except ValueError:
            return None
        return cls(parsed_action, checkpoint_name, renamed_checkpoint_name)


@dataclass(frozen=True, slots=True)
class SessionDeletionRequest:
    """Deletion input frozen before the confirmation dialog is shown."""

    identity: SessionSelectorIdentity
    session_name: str
    display_name: str


@dataclass(frozen=True, slots=True)
class CheckpointDeletionRequest:
    """Checkpoint deletion input frozen before confirmation."""

    identity: CheckpointSelectorIdentity
    checkpoint_name: str


@dataclass(frozen=True, slots=True)
class OpenSessionSelector:
    """Request opening the selector for one session identity."""

    identity: SessionSelectorIdentity


@dataclass(frozen=True, slots=True)
class OpenCheckpointSelector:
    """Request opening the selector for one checkpoint identity."""

    identity: CheckpointSelectorIdentity


@dataclass(frozen=True, slots=True)
class ConfirmSessionDeletion:
    """Carry the data required for confirm session deletion."""

    request: SessionDeletionRequest


@dataclass(frozen=True, slots=True)
class ConfirmCheckpointDeletion:
    """Carry the data required for confirm checkpoint deletion."""

    request: CheckpointDeletionRequest


@dataclass(frozen=True, slots=True)
class PresentAdministrationNotification:
    """Carry the data required for present administration notification."""

    notification: UiNotification


@dataclass(frozen=True, slots=True)
class DismissConfiguration:
    """Request dismissal of the active configuration view."""

    action: ConfigurationDialogAction


@dataclass(frozen=True, slots=True)
class ReopenSessionSelector:
    """Ask the adapter to start a new guarded session selector generation."""


@dataclass(frozen=True, slots=True)
class ReopenCheckpointSelector:
    """Ask the adapter to start a new guarded checkpoint selector generation."""


type ConfigurationAdministrationEffect = (
    OpenSessionSelector
    | OpenCheckpointSelector
    | ConfirmSessionDeletion
    | ConfirmCheckpointDeletion
    | PresentAdministrationNotification
    | DismissConfiguration
    | ReopenSessionSelector
    | ReopenCheckpointSelector
)


@dataclass(frozen=True, slots=True)
class ConfigurationAdministrationTransition:
    """One deterministic state snapshot and its ordered adapter effects."""

    state: ConfigurationAdministrationState
    effects: tuple[ConfigurationAdministrationEffect, ...] = ()
