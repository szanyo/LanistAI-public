# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validate the data schema for session manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SESSION_MANIFEST_ROOT_ALIAS_KEYS = frozenset(
    {
        "uuid",
        "storage_name",
        "session_name",
        "display_name",
        "description",
        "tags",
        "created_at",
        "updated_at",
        "modified_at",
        "parent_session_uuid",
        "child_session_uuids",
        "framework_uuid",
        "load_checkpoint_from_session",
        "load_checkpoint_name",
        "base_step_offset",
        "last_checkpoint_name",
        "last_checkpoint_kind",
        "last_checkpoint_saved_at",
    }
)


def _normalize_text(value: Any) -> str:
    """Normalize the text accepted by this session manifest schema module."""
    return str(value or "").strip()


def _normalize_optional_text(value: Any) -> str | None:
    """Normalize the optional text accepted by this session manifest schema module."""
    text = _normalize_text(value)
    return text or None


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize the string list accepted by this session manifest schema module."""
    if value is None:
        return []
    if isinstance(value, str | bytes):
        text = _normalize_text(value)
        return [text] if text else []
    if isinstance(value, Sequence):
        items = list(value)
    elif isinstance(value, Mapping):
        items = list(value.values())
    else:
        try:
            items = list(value)
        except TypeError:
            text = _normalize_text(value)
            return [text] if text else []
    normalized = []
    for item in items:
        text = _normalize_text(item)
        if text:
            normalized.append(text)
    return normalized


class SessionManifestSessionSchema(BaseModel):
    """Validate and represent session manifest session schema data."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    name: str = ""
    uuid: str = ""
    storage_name: str = ""
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    created_at: str = ""
    modified_at: str = ""
    parent_session_uuid: str | None = None
    child_session_uuids: list[str] = Field(default_factory=list)
    framework_uuid: str | None = None
    load_checkpoint_from_session: str | None = None
    load_checkpoint_name: str | None = None
    base_step_offset: int = 0
    last_checkpoint_name: str | None = None
    last_checkpoint_kind: str | None = None
    last_checkpoint_saved_at: str | None = None

    @field_validator(
        "name",
        "uuid",
        "storage_name",
        "description",
        "created_at",
        "modified_at",
        mode="before",
    )
    @classmethod
    def _normalize_text_field(cls, value: Any) -> str:
        """Normalize the text field accepted by this session manifest session schema."""
        return _normalize_text(value)

    @field_validator(
        "parent_session_uuid",
        "load_checkpoint_from_session",
        "load_checkpoint_name",
        "last_checkpoint_name",
        "last_checkpoint_kind",
        "last_checkpoint_saved_at",
        mode="before",
    )
    @classmethod
    def _normalize_optional_text_field(cls, value: Any) -> str | None:
        """Normalize the optional text field accepted by this session manifest session schema."""
        return _normalize_optional_text(value)

    @field_validator("framework_uuid", mode="before")
    @classmethod
    def _normalize_framework_uuid_field(cls, value: Any) -> str | None:
        """Normalize the framework uuid field accepted by this session manifest session schema."""
        normalized = _normalize_optional_text(value)
        return normalized.lower() if normalized is not None else None

    @field_validator("tags", "child_session_uuids", mode="before")
    @classmethod
    def _normalize_text_list_field(cls, value: Any) -> list[str]:
        """Normalize the text list field accepted by this session manifest session schema."""
        return _normalize_string_list(value)

    @field_validator("base_step_offset", mode="before")
    @classmethod
    def _normalize_base_step_offset(cls, value: Any) -> int:
        """Normalize the base step offset accepted by this session manifest session schema."""
        if value is None:
            return 0
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0


class SessionManifestSchema(BaseModel):
    """Validate and represent session manifest schema data."""

    model_config = ConfigDict(extra="allow", validate_assignment=True)

    schema_version: Literal[1]
    session: SessionManifestSessionSchema

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_exact_schema_version(cls, value: Any) -> int:
        """Return the required exact schema version for this session manifest schema."""
        if type(value) is not int or value != 1:
            raise ValueError("Session manifest schema_version must be integer 1.")
        return value

    @field_validator("session", mode="before")
    @classmethod
    def _coerce_session_section(cls, value: Any) -> Any:
        """Coerce the session section owned by this session manifest schema."""
        if value is None:
            raise ValueError("Session manifest is missing a session section.")
        if isinstance(value, SessionManifestSessionSchema):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        return value


def validate_session_manifest_payload(payload: Any) -> dict[str, Any]:
    """Validate the session manifest payload required by this session manifest schema module."""
    return SessionManifestSchema.model_validate(payload).model_dump()


def canonicalize_session_manifest_payload(payload: Any) -> dict[str, Any]:
    """Validate one manifest after removing legacy root-level session aliases."""

    candidate = dict(payload) if isinstance(payload, Mapping) else payload
    if isinstance(candidate, dict) and "session" in candidate:
        for key in SESSION_MANIFEST_ROOT_ALIAS_KEYS:
            candidate.pop(key, None)
    return validate_session_manifest_payload(candidate)
