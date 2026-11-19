# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validate the data schema for configuration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)

VALID_FRAMEWORK_UUIDS = {"ray", "sb3"}


def _normalize_text(value: Any) -> str:
    """Normalize the text accepted by this configuration schema module."""
    return str(value or "").strip()


def _normalize_framework_uuid(value: Any, default: str = "ray") -> str:
    """Normalize the framework uuid accepted by this configuration schema module."""
    normalized = _normalize_text(value).lower()
    if normalized in VALID_FRAMEWORK_UUIDS:
        return normalized
    return default


def _resolve_relative_path(path: str, model_folder_path: str | None) -> str:
    """Resolve the relative path required by this configuration schema module."""
    normalized_path = _normalize_text(path)
    if not normalized_path:
        return ""
    if not model_folder_path:
        return normalized_path
    if os.path.isabs(normalized_path):
        return normalized_path
    return os.path.abspath(os.path.join(model_folder_path, normalized_path))


def _serialize_path(path: str, model_folder_path: str | None) -> str:
    """Serialize the path owned by this configuration schema module."""
    normalized_path = _normalize_text(path)
    if not normalized_path:
        return ""
    if not model_folder_path:
        return normalized_path
    if not os.path.isabs(normalized_path):
        return normalized_path
    try:
        return os.path.relpath(normalized_path, start=model_folder_path)
    except ValueError:
        return normalized_path


class FrameworkResourcePaths(BaseModel):
    """Validate and represent framework resource paths data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    ray: str = "resources/ray"
    sb3: str = "resources/sb3"

    @field_validator("ray", "sb3", mode="before")
    @classmethod
    def _normalize_path(cls, value: Any) -> str:
        """Normalize the path accepted by this framework resource paths."""
        return _normalize_text(value)

    def resolve(self, model_folder_path: str | None) -> FrameworkResourcePaths:
        """Return the resolve value owned by this framework resource paths."""
        if not model_folder_path:
            return self
        return self.model_copy(
            update={
                "ray": _resolve_relative_path(self.ray, model_folder_path),
                "sb3": _resolve_relative_path(self.sb3, model_folder_path),
            }
        )

    def serialize(self, model_folder_path: str | None) -> dict[str, str]:
        """Serialize the operation owned by this framework resource paths."""
        return {
            "ray": _serialize_path(self.ray, model_folder_path),
            "sb3": _serialize_path(self.sb3, model_folder_path),
        }


class ConfigurationSchema(BaseModel):
    """Validate and represent configuration schema data."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    preferred_framework_uuid: str = "ray"
    active_runtime_framework_uuid: str = "ray"
    selected_default_model_name: str = ""
    selected_default_checkpoint_name: str = ""
    framework_resource_paths: FrameworkResourcePaths = Field(default_factory=FrameworkResourcePaths)

    @field_validator(
        "preferred_framework_uuid",
        "active_runtime_framework_uuid",
        mode="before",
    )
    @classmethod
    def _normalize_framework(cls, value: Any, info: ValidationInfo) -> str:
        """Normalize the framework accepted by this configuration schema."""
        normalized = _normalize_text(value).lower()
        if info.field_name == "active_runtime_framework_uuid":
            return normalized if normalized in VALID_FRAMEWORK_UUIDS else ""
        return normalized if normalized in VALID_FRAMEWORK_UUIDS else "ray"

    @field_validator("selected_default_model_name", "selected_default_checkpoint_name", mode="before")
    @classmethod
    def _normalize_selection(cls, value: Any) -> str:
        """Normalize the selection accepted by this configuration schema."""
        return _normalize_text(value)

    @field_validator("framework_resource_paths", mode="before")
    @classmethod
    def _coerce_framework_paths(cls, value: Any) -> Any:
        """Coerce the framework paths owned by this configuration schema."""
        if isinstance(value, FrameworkResourcePaths):
            return value
        default_paths = {
            "ray": "resources/ray",
            "sb3": "resources/sb3",
        }
        if value is None:
            return default_paths
        if isinstance(value, Mapping):
            return {
                "ray": value.get("ray", default_paths["ray"]) or default_paths["ray"],
                "sb3": value.get("sb3", default_paths["sb3"]) or default_paths["sb3"],
            }
        ray = getattr(value, "ray", None)
        sb3 = getattr(value, "sb3", None)
        return {
            "ray": ray or default_paths["ray"],
            "sb3": sb3 or default_paths["sb3"],
        }

    @model_validator(mode="after")
    def _default_active_runtime(self) -> ConfigurationSchema:
        """Return the default active runtime value owned by this configuration schema."""
        if self.active_runtime_framework_uuid not in VALID_FRAMEWORK_UUIDS:
            self.active_runtime_framework_uuid = self.preferred_framework_uuid
        return self

    @property
    def framework_resource_paths_as_dict(self) -> dict[str, str]:
        """Return the framework resource paths as dict exposed by this configuration schema."""
        return self.framework_resource_paths.model_dump()

    @classmethod
    def _configuration_payload(cls, value: Any) -> dict[str, Any]:
        """Return the configuration payload value owned by this configuration schema."""
        if value is None:
            return {}
        if isinstance(value, cls):
            return value.model_dump()
        if isinstance(value, Mapping):
            if "model_configuration" in value:
                raise ValueError("Expected the model_configuration section, not a wrapped payload.")
            return dict(value)
        return {
            "preferred_framework_uuid": getattr(value, "preferred_framework_uuid", None),
            "active_runtime_framework_uuid": getattr(value, "active_runtime_framework_uuid", None),
            "selected_default_model_name": getattr(value, "selected_default_model_name", None),
            "selected_default_checkpoint_name": getattr(value, "selected_default_checkpoint_name", None),
            "framework_resource_paths": getattr(value, "framework_resource_paths", None),
        }

    @classmethod
    def from_configuration(
        cls,
        value: Any,
        *,
        model_folder_path: str | None = None,
    ) -> ConfigurationSchema:
        """Create this configuration schema's value from configuration."""
        schema = cls.model_validate(cls._configuration_payload(value) or {})
        if model_folder_path:
            schema = schema.model_copy(
                update={
                    "framework_resource_paths": schema.framework_resource_paths.resolve(model_folder_path),
                }
            )
        return schema

    @classmethod
    def from_model_configuration_payload(
        cls,
        value: Mapping[str, Any],
        *,
        model_folder_path: str | None = None,
    ) -> ConfigurationSchema:
        """Validate a complete v1 model-configuration document."""

        payload = validate_versioned_metadata_payload(
            MetadataKind.MODEL_CONFIGURATION,
            value,
        )
        return cls.from_configuration(
            payload["model_configuration"],
            model_folder_path=model_folder_path,
        )

    def to_model_configuration_payload(self, *, model_folder_path: str | None = None) -> dict[str, Any]:
        """Convert this configuration schema's value to model configuration payload."""
        framework_resource_paths = self.framework_resource_paths.serialize(model_folder_path)
        return {
            "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
            "model_configuration": {
                "preferred_framework_uuid": self.preferred_framework_uuid,
                "active_runtime_framework_uuid": self.active_runtime_framework_uuid,
                "selected_default_model_name": self.selected_default_model_name,
                "selected_default_checkpoint_name": self.selected_default_checkpoint_name,
                "framework_resource_paths": framework_resource_paths,
            }
        }
