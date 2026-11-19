# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Schema-v1 minimal model-manifest persistence."""

from typing import Protocol

from lanista.reinforcengine.rl_model.metadata.manifest_schema import (
    validate_manifest_payload,
)
from lanista.reinforcengine.rl_model.metadata.schema import CURRENT_METADATA_SCHEMA_VERSION
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)


class ModelManifestConfiguration(Protocol):
    """Minimal configuration surface required for manifest publication."""

    def get_config_file_path(self) -> str | None:
        """Return the config file path exposed by this model manifest configuration."""
        ...


class ModelManifestWriter:
    """Persist the minimal model identity used by creation and bootstrap."""

    def save_minimal_manifest(self, configuration: ModelManifestConfiguration) -> None:
        """Write and read back one validated schema-v1 model manifest."""

        config_file_path = configuration.get_config_file_path()
        if not config_file_path:
            raise ValueError("Model manifest config path is missing.")

        model_section = getattr(configuration, "model", None)
        saved_name = str(
            getattr(configuration, "model_name", "")
            or getattr(model_section, "name", "")
            or ""
        ).strip()
        saved_uuid = str(
            getattr(configuration, "model_uuid", "")
            or getattr(model_section, "uuid", "")
            or ""
        ).strip()
        saved_version = str(
            getattr(configuration, "model_version", "")
            or getattr(model_section, "version", "")
            or ""
        ).strip()
        saved_description = str(
            getattr(configuration, "description", "")
            or getattr(model_section, "description", "")
            or ""
        )
        saved_created_at = str(
            getattr(configuration, "created_at", "")
            or getattr(model_section, "created_at", "")
            or ""
        ).strip()
        saved_modified_at = str(
            getattr(configuration, "modified_at", "")
            or getattr(model_section, "modified_at", "")
            or ""
        ).strip()
        saved_parent_uuid = getattr(configuration, "parent_model_uuid", None)
        if saved_parent_uuid in ("", "None"):
            saved_parent_uuid = None
        if saved_parent_uuid is None and model_section is not None:
            saved_parent_uuid = getattr(model_section, "parent_model_uuid", None)

        atomic_write_validated_yaml_file(
            config_file_path,
            {
                "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
                "model": {
                    "name": saved_name,
                    "uuid": saved_uuid,
                    "version": saved_version,
                    "description": saved_description,
                    "created_at": saved_created_at,
                    "modified_at": saved_modified_at,
                    "parent_model_uuid": saved_parent_uuid,
                },
            },
            validator=validate_manifest_payload,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )
