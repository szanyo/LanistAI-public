# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validate the data schema for model manifests."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator


def _normalize_text(value: Any) -> str:
    """Normalize the text accepted by this manifest schema module."""
    return str(value or "").strip()


def _normalize_bool(value: Any) -> bool:
    """Normalize the bool accepted by this manifest schema module."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, int | float):
        return bool(value)
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


class ManifestSectionSchema(BaseModel):
    """Validate and represent manifest section schema data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    name: str = ""
    uuid: str = ""
    version: str = ""
    description: str = ""
    created_at: str = ""
    modified_at: str = ""
    parent_model_uuid: str | None = None
    archived: bool = False

    @field_validator(
        "name",
        "uuid",
        "version",
        "description",
        "created_at",
        "modified_at",
        mode="before",
    )
    @classmethod
    def _normalize_text_field(cls, value: Any) -> str:
        """Normalize the text field accepted by this manifest section schema."""
        return _normalize_text(value)

    @field_validator("parent_model_uuid", mode="before")
    @classmethod
    def _normalize_optional_text_field(cls, value: Any) -> str | None:
        """Normalize the optional text field accepted by this manifest section schema."""
        text = _normalize_text(value)
        return text or None

    @field_validator("archived", mode="before")
    @classmethod
    def _normalize_archived_field(cls, value: Any) -> bool:
        """Normalize the archived field accepted by this manifest section schema."""
        return _normalize_bool(value)


class ManifestSchema(BaseModel):
    """Validate and represent manifest schema data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    schema_version: Literal[1]
    model: ManifestSectionSchema

    @field_validator("schema_version", mode="before")
    @classmethod
    def _require_exact_schema_version(cls, value: Any) -> int:
        """Return the required exact schema version for this manifest schema."""
        if type(value) is not int or value != 1:
            raise ValueError("Model manifest schema_version must be integer 1.")
        return value

    @field_validator("model", mode="before")
    @classmethod
    def _coerce_model_section(cls, value: Any) -> Any:
        """Coerce the model section owned by this manifest schema."""
        if value is None:
            raise ValueError("Model manifest is missing a model section.")
        if isinstance(value, ManifestSectionSchema):
            return value
        if isinstance(value, dict):
            return dict(value)
        return value


def validate_manifest_payload(payload: Any) -> dict[str, Any]:
    """Validate the manifest payload required by this manifest schema module."""
    return ManifestSchema.model_validate(payload).model_dump()
