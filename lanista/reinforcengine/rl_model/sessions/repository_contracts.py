# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed commands and outcomes for training-session metadata persistence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

from lanista.reinforcengine.rl_model.execution_contracts import RLModelState


@dataclass(frozen=True, slots=True)
class SessionManifestSummary:
    """Immutable projection of one validated session manifest."""

    storage_name: str
    identity_uuid: str = ""
    display_name: str = ""
    session_name: str = ""
    session_version: int = 0
    framework_uuid: str | None = None
    parent_session_uuid: str | None = None
    description: str = ""
    created_at: str = ""
    updated_at: str = ""
    status: str = ""


class SessionMetadataField(StrEnum):
    """Allowlisted session-manifest fields writable through the facade."""

    DISPLAY_NAME = "display_name"
    DESCRIPTION = "description"
    UPDATED_AT = "updated_at"
    STATUS = "status"
    FRAMEWORK_UUID = "framework_uuid"
    CHILD_SESSION_UUIDS = "child_session_uuids"


type SessionMetadataValue = str | None | tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SessionMetadataChange:
    """One allowlisted metadata field assignment."""

    field: SessionMetadataField
    value: SessionMetadataValue

    def __post_init__(self) -> None:
        """Validate and finalize the initialized session metadata change state."""
        if not isinstance(self.field, SessionMetadataField):
            raise TypeError("session metadata fields must use SessionMetadataField")

        if self.field in {
            SessionMetadataField.DISPLAY_NAME,
            SessionMetadataField.DESCRIPTION,
        }:
            if self.value is not None and not isinstance(self.value, str):
                raise TypeError(f"{self.field.value} must be text or null")
            return

        if self.field is SessionMetadataField.CHILD_SESSION_UUIDS:
            self._validate_child_session_uuids()
            return

        if not isinstance(self.value, str):
            raise TypeError(f"{self.field.value} must be text")
        if not self.value.strip():
            raise ValueError(f"{self.field.value} cannot be blank")
        if self.field is SessionMetadataField.STATUS:
            try:
                RLModelState(self.value)
            except ValueError as error:
                raise ValueError("status must be a valid RLModelState value") from error

    def _validate_child_session_uuids(self) -> None:
        """Validate the child session uuids required by this session metadata change."""
        if not isinstance(self.value, tuple):
            raise TypeError("child_session_uuids must be a tuple of non-empty strings")
        if any(not isinstance(item, str) or not item.strip() for item in self.value):
            raise ValueError("child_session_uuids cannot contain blank values")
        if len(self.value) != len(set(self.value)):
            raise ValueError("child_session_uuids cannot contain duplicates")


@dataclass(frozen=True, slots=True)
class SessionMetadataUpdate:
    """Validated group of metadata assignments committed as one write."""

    changes: tuple[SessionMetadataChange, ...]

    def __post_init__(self) -> None:
        """Validate and finalize the initialized session metadata update state."""
        if not self.changes:
            raise ValueError("a session metadata update cannot be empty")
        fields = tuple(change.field for change in self.changes)
        if len(fields) != len(set(fields)):
            raise ValueError("a session metadata field may be assigned only once")

    @classmethod
    def details(
        cls,
        *,
        display_name: str | None,
        description: str | None,
        updated_at: str,
    ) -> SessionMetadataUpdate:
        """Build the fields edited by the session-detail dialog."""

        return cls(
            (
                SessionMetadataChange(SessionMetadataField.DISPLAY_NAME, display_name),
                SessionMetadataChange(SessionMetadataField.DESCRIPTION, description),
                SessionMetadataChange(SessionMetadataField.UPDATED_AT, updated_at),
            )
        )

    @classmethod
    def status(
        cls,
        status: RLModelState,
        *,
        updated_at: str,
    ) -> SessionMetadataUpdate:
        """Build a runtime-status projection update."""

        return cls(
            (
                SessionMetadataChange(SessionMetadataField.STATUS, status.value),
                SessionMetadataChange(SessionMetadataField.UPDATED_AT, updated_at),
            )
        )

    @classmethod
    def framework(cls, framework_uuid: str) -> SessionMetadataUpdate:
        """Build a framework inference backfill update."""

        return cls(
            (
                SessionMetadataChange(
                    SessionMetadataField.FRAMEWORK_UUID,
                    framework_uuid,
                ),
            )
        )

    @classmethod
    def child_sessions(
        cls,
        child_session_uuids: tuple[str, ...],
        *,
        updated_at: str,
    ) -> SessionMetadataUpdate:
        """Build a lineage child-list update."""

        return cls(
            (
                SessionMetadataChange(
                    SessionMetadataField.CHILD_SESSION_UUIDS,
                    child_session_uuids,
                ),
                SessionMetadataChange(SessionMetadataField.UPDATED_AT, updated_at),
            )
        )

    @classmethod
    def description(cls, description: str | None) -> SessionMetadataUpdate:
        """Build the smallest storage-level description update."""

        return cls(
            (
                SessionMetadataChange(
                    SessionMetadataField.DESCRIPTION,
                    description,
                ),
            )
        )

    def to_storage_fields(self) -> dict[str, object]:
        """Return wrapper-compatible values without exposing variadic input."""

        fields: dict[str, object] = {}
        for change in self.changes:
            value: object = change.value
            if change.field is SessionMetadataField.CHILD_SESSION_UUIDS:
                value = list(cast(tuple[str, ...], change.value))
            fields[change.field.value] = value
        return fields

    @property
    def field_names(self) -> tuple[SessionMetadataField, ...]:
        """Return the stable changed-field identifiers."""

        return tuple(change.field for change in self.changes)


@dataclass(frozen=True, slots=True)
class SessionMetadataUpdateOutcome:
    """Durable update result including post-commit cache state."""

    session_name: str
    fields: tuple[SessionMetadataField, ...]
    committed: bool = True
    summary_invalidated: bool = True

    @property
    def degraded(self) -> bool:
        """A committed write is degraded only when cache invalidation failed."""

        return self.committed and not self.summary_invalidated
