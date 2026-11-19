# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Adapt configuration access for training-session manifests."""

from collections.abc import Mapping
from copy import deepcopy

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.metadata.session_manifest_schema import (
    canonicalize_session_manifest_payload,
    validate_session_manifest_payload,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)


class SessionConfigurationWrapper:
    """Validated configuration wrapper for training session YAML files."""

    def __init__(self, session_yaml_path: str):
        """Initialize the session configuration wrapper and its owned state."""
        self._session_yaml_path = str(session_yaml_path)
        self._cfg = Configuration()
        self._cfg.set_config_file_path(self._session_yaml_path)

    @property
    def path(self) -> str:
        """Return the path exposed by this session configuration wrapper."""
        return self._session_yaml_path

    @property
    def configuration(self) -> Configuration:
        """Expose the validated in-memory configuration view."""
        return self._cfg

    def exists(self) -> bool:
        """Return the exists value owned by this session configuration wrapper."""
        try:
            from os.path import isfile

            return isfile(self._session_yaml_path)
        except Exception:
            return False

    def load_configuration(self) -> Configuration:
        """Load the configuration owned by this session configuration wrapper."""
        payload = validate_versioned_metadata_payload(
            MetadataKind.SESSION_MANIFEST,
            read_yaml_mapping_file(self._session_yaml_path),
        )
        validated_payload = validate_session_manifest_payload(payload)
        self._cfg = Configuration(config_dict=validated_payload)
        self._cfg.set_config_file_path(self._session_yaml_path)
        self._project_session_aliases()
        return self._cfg

    def _project_session_aliases(self) -> None:
        """Project the session aliases owned by this session configuration wrapper."""
        session_section = getattr(self._cfg, "session", None)
        if session_section is None:
            return

        name = str(getattr(session_section, "name", "") or "")
        description = str(getattr(session_section, "description", "") or "")
        tags = list(getattr(session_section, "tags", []) or [])
        created_at = str(getattr(session_section, "created_at", "") or "")
        modified_at = str(getattr(session_section, "modified_at", "") or "")
        parent_session_uuid = getattr(session_section, "parent_session_uuid", None)
        child_session_uuids = list(getattr(session_section, "child_session_uuids", []) or [])
        session_uuid = str(getattr(session_section, "uuid", "") or "")
        storage_name = str(getattr(session_section, "storage_name", "") or "")
        framework_uuid = str(getattr(session_section, "framework_uuid", "") or "").strip().lower()
        load_checkpoint_from_session = str(
            getattr(session_section, "load_checkpoint_from_session", "") or ""
        ).strip()
        load_checkpoint_name = str(
            getattr(session_section, "load_checkpoint_name", "") or ""
        ).strip()
        base_step_offset = getattr(session_section, "base_step_offset", 0)
        last_checkpoint_name = str(
            getattr(session_section, "last_checkpoint_name", "") or ""
        ).strip()
        last_checkpoint_kind = str(
            getattr(session_section, "last_checkpoint_kind", "") or ""
        ).strip()
        last_checkpoint_saved_at = str(
            getattr(session_section, "last_checkpoint_saved_at", "") or ""
        ).strip()

        self._cfg.uuid = session_uuid
        self._cfg.storage_name = storage_name
        self._cfg.session_name = name
        self._cfg.display_name = name
        self._cfg.description = description
        self._cfg.tags = tags
        self._cfg.created_at = created_at
        self._cfg.updated_at = modified_at
        self._cfg.modified_at = modified_at
        self._cfg.parent_session_uuid = str(parent_session_uuid or "").strip() or None
        self._cfg.child_session_uuids = child_session_uuids
        self._cfg.framework_uuid = framework_uuid or None
        self._cfg.load_checkpoint_from_session = load_checkpoint_from_session or None
        self._cfg.load_checkpoint_name = load_checkpoint_name or None
        self._cfg.base_step_offset = base_step_offset or 0
        self._cfg.last_checkpoint_name = last_checkpoint_name or None
        self._cfg.last_checkpoint_kind = last_checkpoint_kind or None
        self._cfg.last_checkpoint_saved_at = last_checkpoint_saved_at or None

    def snapshot(self) -> dict[str, object]:
        """Return a detached mapping of the validated in-memory payload."""
        self.load_configuration()
        payload = deepcopy(getattr(self._cfg, "_config_dict", None))
        if not isinstance(payload, dict):
            raise ValueError("Session configuration payload is not a mapping.")
        return dict(payload)

    def apply_fields(self, fields: Mapping[str, object]) -> bool:
        """Apply explicit storage fields and report whether values changed."""
        changed = False
        session_section = getattr(self._cfg, "session", None)
        if session_section is None:
            session_section = {}
            self._cfg.session = session_section

        session_field_map = {
            "uuid": "uuid",
            "storage_name": "storage_name",
            "session_name": "name",
            "display_name": "name",
            "description": "description",
            "tags": "tags",
            "created_at": "created_at",
            "updated_at": "modified_at",
            "modified_at": "modified_at",
            "parent_session_uuid": "parent_session_uuid",
            "child_session_uuids": "child_session_uuids",
            "framework_uuid": "framework_uuid",
            "load_checkpoint_from_session": "load_checkpoint_from_session",
            "load_checkpoint_name": "load_checkpoint_name",
            "base_step_offset": "base_step_offset",
            "last_checkpoint_name": "last_checkpoint_name",
            "last_checkpoint_kind": "last_checkpoint_kind",
            "last_checkpoint_saved_at": "last_checkpoint_saved_at",
        }

        for key, value in fields.items():
            if value is None:
                continue
            mapped_key = session_field_map.get(key)
            if mapped_key is None:
                old_value = getattr(self._cfg, key, None)
                if old_value != value:
                    setattr(self._cfg, key, value)
                    changed = True
            else:
                old_nested = (
                    getattr(session_section, mapped_key, None)
                    if not isinstance(session_section, dict)
                    else session_section.get(mapped_key)
                )
                if old_nested != value:
                    if isinstance(session_section, dict):
                        session_section[mapped_key] = value
                    else:
                        setattr(session_section, mapped_key, value)
                    changed = True
                old_value = getattr(self._cfg, key, None)
                if old_value != value:
                    setattr(self._cfg, key, value)
        return changed

    def save(self) -> None:
        """Persist the current state owned by this session configuration wrapper."""
        payload = deepcopy(getattr(self._cfg, "_config_dict", None))
        if not isinstance(payload, dict):
            raise ValueError("Session configuration payload is not a mapping.")
        stored_version = payload.get("schema_version")
        if stored_version is not None and stored_version != CURRENT_METADATA_SCHEMA_VERSION:
            raise ValueError(f"Unsupported session manifest schema version: {stored_version!r}.")
        payload["schema_version"] = CURRENT_METADATA_SCHEMA_VERSION
        atomic_write_validated_yaml_file(
            self._session_yaml_path,
            payload,
            validator=canonicalize_session_manifest_payload,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )
