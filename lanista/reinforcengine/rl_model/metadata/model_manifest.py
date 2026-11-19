# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Strict read and persistence boundary for one model manifest projection."""

from __future__ import annotations

import os
from copy import deepcopy
from types import SimpleNamespace
from typing import cast

from yaml import YAMLError

from lanista.foundation import filesystem as IO
from lanista.foundation.configuration import AppConfiguration, Configuration
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    PersistenceError,
    ResourceNotFoundError,
)
from lanista.reinforcengine.rl_model.metadata.configuration_schema import (
    ConfigurationSchema,
)
from lanista.reinforcengine.rl_model.metadata.manifest_schema import (
    validate_manifest_payload,
)
from lanista.reinforcengine.rl_model.metadata.schema import (
    CURRENT_METADATA_SCHEMA_VERSION,
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    find_existing_model_manifest_path,
    get_framework_model_configuration_path,
    get_model_configuration_path,
)
from lanista.reinforcengine.rl_model.storage.persistence import (
    atomic_write_validated_yaml_file,
    read_yaml_mapping_file,
)

APP_CONFIGURATION = AppConfiguration()
DEFAULT_RL_FRAMEWORKS = SimpleNamespace(
    ray=SimpleNamespace(symbol="ray"),
    sb3=SimpleNamespace(symbol="sb3"),
)


class ModelManifest(LoggerInterface):
    """Describe model data."""

    @staticmethod
    def _short_model_path(path: str) -> str:
        """Return the short model path value owned by this model manifest."""
        text = str(path or "").strip()
        if not text:
            return "<unknown-model>"
        marker = f"{os.sep}models{os.sep}"
        marker_index = text.find(marker)
        if marker_index >= 0:
            return text[marker_index + len(marker) :]
        return os.path.basename(text.rstrip(os.sep)) or text

    @classmethod
    def _quoted_model_path(cls, path: str) -> str:
        """Return the quoted model path value owned by this model manifest."""
        return f'"{cls._short_model_path(path)}"'

    def __init__(self, model_folder_path: str, **kwargs: object) -> None:
        """Initialize the model manifest with the path to the model folder."""
        self._RL_FRAMEWORKS = deepcopy(
            getattr(APP_CONFIGURATION, "rl_frameworks", DEFAULT_RL_FRAMEWORKS)
        )
        self._model_folder_path = model_folder_path
        super().__init__(**kwargs)
        self._model_manifest_configuration = Configuration()
        self._model_configuration = Configuration()
        self._model_symbol = "❓"  # Default icon if not found

    def load(self) -> None:
        """Atomically publish one fully validated model/configuration pair."""

        path = find_existing_model_manifest_path(self._model_folder_path)
        if not path or not IO.existf(path):
            raise ResourceNotFoundError("The model manifest metadata is missing.")

        try:
            raw_payload = read_yaml_mapping_file(path)
            versioned_payload = validate_versioned_metadata_payload(
                MetadataKind.MODEL_MANIFEST,
                raw_payload,
            )
            manifest_payload = validate_manifest_payload(versioned_payload)
            candidate_manifest_configuration = Configuration(config_dict=manifest_payload)
            candidate_manifest_configuration.set_config_file_path(path)
            self._project_model_aliases(candidate_manifest_configuration)
            candidate_manifest_configuration.model_folder_path = self._model_folder_path
        except (OSError, TypeError, ValueError, YAMLError) as error:
            raise PersistenceError(
                "Model manifest is unreadable or uses an unsupported schema. "
                "Run the metadata migration before loading the model."
            ) from error

        candidate_model_configuration, schema = self._load_model_configuration_candidate()
        candidate_symbol = self._project_model_configuration_aliases(
            candidate_manifest_configuration,
            schema,
        )
        self._load_framework_model_configuration(candidate_manifest_configuration)

        self._model_manifest_configuration = candidate_manifest_configuration
        self._model_configuration = candidate_model_configuration
        self._model_symbol = candidate_symbol

    def _load_model_configuration_candidate(
        self,
    ) -> tuple[Configuration, ConfigurationSchema]:
        """Load the model configuration candidate owned by this model manifest."""
        path = get_model_configuration_path(self._model_folder_path)
        if not IO.existf(path):
            raise ResourceNotFoundError("The model configuration metadata is missing.")
        try:
            model_configuration_payload = read_yaml_mapping_file(path)
            schema = ConfigurationSchema.from_model_configuration_payload(
                model_configuration_payload,
                model_folder_path=self._model_folder_path,
            )
            candidate_configuration = Configuration(config_dict=model_configuration_payload)
            candidate_configuration.set_config_file_path(path)
        except (OSError, ValueError, YAMLError) as error:
            raise PersistenceError(
                "Model configuration is unreadable or uses an unsupported schema. "
                "Run the metadata migration before loading the model."
            ) from error
        return candidate_configuration, schema

    @staticmethod
    def _project_model_aliases(cfg: Configuration) -> None:
        """Project the model aliases owned by this model manifest."""
        model_section = getattr(cfg, "model", None)
        if model_section is None:
            return

        cfg.model_name = str(getattr(model_section, "name", "") or "")
        cfg.model_uuid = str(getattr(model_section, "uuid", "") or "")
        cfg.model_version = str(getattr(model_section, "version", "") or "")
        cfg.description = str(getattr(model_section, "description", "") or "")
        cfg.created_at = str(getattr(model_section, "created_at", "") or "")
        cfg.modified_at = str(getattr(model_section, "modified_at", "") or "")
        parent_model_uuid = getattr(model_section, "parent_model_uuid", None)
        cfg.parent_model_uuid = str(parent_model_uuid or "").strip() or None
        cfg.archived = bool(getattr(model_section, "archived", False))

    def _project_model_configuration_aliases(
        self,
        cfg: Configuration,
        schema: ConfigurationSchema,
    ) -> str:
        """Project the model configuration aliases owned by this model manifest."""
        framework_paths = schema.framework_resource_paths_as_dict

        cfg.model_configuration_schema_version = CURRENT_METADATA_SCHEMA_VERSION
        cfg.model_configuration_schema = schema
        cfg.framework_resource_paths = framework_paths
        cfg.preferred_framework_uuid = schema.preferred_framework_uuid
        cfg.active_runtime_framework_uuid = schema.active_runtime_framework_uuid
        cfg.selected_default_model_name = schema.selected_default_model_name
        cfg.selected_default_checkpoint_name = schema.selected_default_checkpoint_name

        preferred_framework_uuid = schema.preferred_framework_uuid
        candidate_symbol = "❓"
        if preferred_framework_uuid:
            framework_cfg = getattr(self._RL_FRAMEWORKS, preferred_framework_uuid, None)
            if framework_cfg is not None:
                candidate_symbol = str(
                    getattr(framework_cfg, "symbol", candidate_symbol) or candidate_symbol
                )
        return candidate_symbol

    def _load_framework_model_configuration(
        self,
        cfg: Configuration,
    ) -> None:
        """Load the framework model configuration owned by this model manifest."""
        preferred_framework_uuid = (
            str(getattr(cfg, "preferred_framework_uuid", "") or "").strip().lower()
        )
        if not preferred_framework_uuid:
            return

        path = get_framework_model_configuration_path(
            self._model_folder_path, preferred_framework_uuid
        )
        if not IO.existf(path):
            return

        try:
            framework_payload = validate_versioned_metadata_payload(
                MetadataKind.FRAMEWORK_CONFIGURATION,
                read_yaml_mapping_file(path),
            )
            framework_cfg = Configuration(config_dict=framework_payload)
            framework_cfg.set_config_file_path(path)
        except (OSError, ValueError, YAMLError) as error:
            # The preferred framework overlay is optional during catalog
            # inspection. Keep the model metadata readable, but make the
            # degraded runtime configuration observable for diagnostics.
            self._logger.warning(
                f"Preferred framework configuration could not be loaded [{type(error).__name__}]."
            )
            return

        for attribute_name in (
            "model_implementation",
            "rl_engine_framework",
            "algorithm",
            "scenario_manager_configuration_file_path",
            "scenarios_folder_path",
            "checkpoint_saving_frequency",
            "progress_bar_update_frequency",
            "evaluation_frequency",
            "evaluation_deterministic",
            "runtime",
        ):
            attribute_value = getattr(framework_cfg, attribute_name, None)
            if attribute_value is not None:
                setattr(cfg, attribute_name, attribute_value)

    @staticmethod
    def _section_value(section: object, attribute_name: str, default: object = "") -> object:
        """Return the section value value owned by this model manifest."""
        if isinstance(section, dict):
            return section.get(attribute_name, default)
        return getattr(section, attribute_name, default)

    def get_runtime_selection_defaults(self) -> tuple[str | None, str | None]:
        """Return the runtime selection defaults exposed by this model manifest."""
        cfg = cast(Configuration | None, self.get_configuration())
        if cfg is None:
            return None, None
        session_name = str(getattr(cfg, "selected_default_model_name", "") or "").strip() or None
        checkpoint_name = (
            str(getattr(cfg, "selected_default_checkpoint_name", "") or "").strip() or None
        )
        return session_name, checkpoint_name

    def get_bootstrap_framework_uuid(self) -> str | None:
        """Return the bootstrap framework uuid exposed by this model manifest."""
        cfg = cast(Configuration | None, self.get_configuration())
        if cfg is None:
            return None
        return (
            str(getattr(cfg, "active_runtime_framework_uuid", "") or "").strip().lower()
            or str(getattr(cfg, "preferred_framework_uuid", "") or "").strip().lower()
            or str(getattr(getattr(cfg, "rl_engine_framework", None), "uuid", "") or "")
            .strip()
            .lower()
            or None
        )

    def _build_manifest_payload(self) -> dict[str, object]:
        """Build the manifest payload required by this model manifest."""
        cfg = self._model_manifest_configuration
        model_section = getattr(cfg, "model", None)
        return {
            "name": str(
                getattr(cfg, "model_name", "")
                or self._section_value(model_section, "name", "")
                or ""
            ).strip(),
            "uuid": str(
                getattr(cfg, "model_uuid", "")
                or self._section_value(model_section, "uuid", "")
                or ""
            ).strip(),
            "version": str(
                getattr(cfg, "model_version", "")
                or self._section_value(model_section, "version", "")
                or ""
            ).strip(),
            "description": str(
                getattr(cfg, "description", "")
                or self._section_value(model_section, "description", "")
                or ""
            ),
            "created_at": str(
                getattr(cfg, "created_at", "")
                or self._section_value(model_section, "created_at", "")
                or ""
            ).strip(),
            "modified_at": str(
                getattr(cfg, "modified_at", "")
                or self._section_value(model_section, "modified_at", "")
                or ""
            ).strip(),
            "parent_model_uuid": (
                str(
                    getattr(cfg, "parent_model_uuid", "")
                    or self._section_value(model_section, "parent_model_uuid", "")
                    or ""
                ).strip()
                or None
            ),
            "archived": bool(
                getattr(cfg, "archived", self._section_value(model_section, "archived", False))
            ),
        }

    def _save_model_manifest(self) -> None:
        """Persist the model manifest owned by this model manifest."""
        config_path = self._model_manifest_configuration.get_config_file_path()
        if not config_path:
            config_path = find_existing_model_manifest_path(self._model_folder_path)
        if not config_path:
            raise ResourceNotFoundError("The model manifest metadata is missing.")

        payload = self._build_manifest_payload()
        if not payload["name"] or not payload["uuid"] or not payload["version"]:
            raise ApplicationValidationError(
                "The model manifest requires a name, UUID, and version before saving."
            )
        atomic_write_validated_yaml_file(
            config_path,
            {
                "schema_version": CURRENT_METADATA_SCHEMA_VERSION,
                "model": payload,
            },
            validator=validate_manifest_payload,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )

    def _save_model_configuration(self) -> None:
        """Persist the model configuration owned by this model manifest."""
        model_folder_path = str(self._model_folder_path or "").strip()
        if not model_folder_path:
            raise ApplicationValidationError("The model storage folder is required before saving.")

        schema = self._build_model_configuration_schema(model_folder_path)
        configuration = self._write_model_configuration(
            schema,
            model_folder_path=model_folder_path,
        )
        self._publish_model_configuration(configuration, schema)

    def _build_model_configuration_schema(
        self,
        model_folder_path: str,
    ) -> ConfigurationSchema:
        """Build the model configuration schema required by this model manifest."""
        schema_source = getattr(
            self._model_manifest_configuration, "model_configuration_schema", None
        )
        if schema_source is None:
            raise ApplicationValidationError(
                "A validated model configuration is required before saving."
            )
        base_schema = ConfigurationSchema.from_configuration(
            schema_source,
            model_folder_path=model_folder_path,
        )
        schema = base_schema.model_copy(
            update={
                "preferred_framework_uuid": (
                    str(
                        getattr(self._model_manifest_configuration, "preferred_framework_uuid", "")
                        or ""
                    )
                    .strip()
                    .lower()
                    or base_schema.preferred_framework_uuid
                ),
                "active_runtime_framework_uuid": (
                    str(
                        getattr(
                            self._model_manifest_configuration, "active_runtime_framework_uuid", ""
                        )
                        or ""
                    )
                    .strip()
                    .lower()
                    or base_schema.active_runtime_framework_uuid
                ),
                "selected_default_model_name": str(
                    getattr(self._model_manifest_configuration, "selected_default_model_name", "")
                    or ""
                ).strip(),
                "selected_default_checkpoint_name": str(
                    getattr(
                        self._model_manifest_configuration, "selected_default_checkpoint_name", ""
                    )
                    or ""
                ).strip(),
            }
        )
        return schema

    @staticmethod
    def _write_model_configuration(
        schema: ConfigurationSchema,
        *,
        model_folder_path: str,
    ) -> Configuration:
        """Persist the model configuration owned by this model manifest."""
        raw_payload = {
            "preferred_framework_uuid": schema.preferred_framework_uuid,
            "active_runtime_framework_uuid": schema.active_runtime_framework_uuid,
            "selected_default_model_name": schema.selected_default_model_name,
            "selected_default_checkpoint_name": schema.selected_default_checkpoint_name,
            "framework_resource_paths": schema.framework_resource_paths_as_dict,
        }
        configuration_payload = schema.to_model_configuration_payload(
            model_folder_path=model_folder_path,
        )
        cfg = Configuration()
        config_path = get_model_configuration_path(model_folder_path)
        cfg.set_config_file_path(config_path)

        def _validate_and_build_model_configuration_payload(
            payload: dict[str, object],
        ) -> dict[str, object]:
            """Validate the and build model configuration payload required by this model manifest.
            """
            validated_schema = ConfigurationSchema.from_configuration(
                payload,
                model_folder_path=model_folder_path,
            )
            return validated_schema.to_model_configuration_payload(
                model_folder_path=model_folder_path,
            )

        atomic_write_validated_yaml_file(
            config_path,
            raw_payload,
            validator=_validate_and_build_model_configuration_payload,
            verify_reader=read_yaml_mapping_file,
            smoke_check_reader=read_yaml_mapping_file,
        )
        cfg.model_configuration = configuration_payload["model_configuration"]
        return cfg

    def _publish_model_configuration(
        self,
        configuration: Configuration,
        schema: ConfigurationSchema,
    ) -> None:
        """Publish the model configuration owned by this model manifest."""
        self._model_configuration = configuration
        self._model_manifest_configuration.model_configuration_schema = schema
        self._model_manifest_configuration.framework_resource_paths = (
            schema.framework_resource_paths.resolve(self._model_folder_path).model_dump()
        )
        self._model_manifest_configuration.preferred_framework_uuid = (
            schema.preferred_framework_uuid
        )
        self._model_manifest_configuration.active_runtime_framework_uuid = (
            schema.active_runtime_framework_uuid
        )
        self._model_manifest_configuration.selected_default_model_name = (
            schema.selected_default_model_name
        )
        self._model_manifest_configuration.selected_default_checkpoint_name = (
            schema.selected_default_checkpoint_name
        )
        framework_cfg = getattr(self._RL_FRAMEWORKS, schema.preferred_framework_uuid, None)
        if framework_cfg is not None:
            self._model_symbol = str(
                getattr(framework_cfg, "symbol", self._model_symbol) or self._model_symbol
            )

    def save(self) -> None:
        """Persist the loaded manifest/configuration pair or raise a typed error."""

        if not self._model_manifest_configuration:
            raise ApplicationValidationError("The model manifest must be loaded before saving.")
        self._save_model_manifest()
        self._save_model_configuration()

    def ensure_valid(self) -> None:
        """Raise a typed error unless every projected manifest invariant holds."""

        if not self._model_manifest_configuration:
            self._logger.error("Manifest file is not loaded or invalid.")
            raise ApplicationValidationError("The model manifest is not loaded.")
        if not self._name_validation():
            self._logger.error(
                f"Model name validation FAILED for '{self._model_manifest_configuration.get_config_file_path()}'."
            )
            raise ApplicationValidationError("The model manifest name is invalid.")
        if not self._uuid_validation():
            self._logger.error(
                f"Model UUID validation FAILED for '{self._model_manifest_configuration.get_config_file_path()}'."
            )
            raise ApplicationValidationError("The model manifest UUID is invalid.")
        if not self._version_validation():
            self._logger.error(
                f"Model version validation FAILED for '{self._model_manifest_configuration.get_config_file_path()}'."
            )
            raise ApplicationValidationError("The model manifest version is invalid.")
        if not self._timestamp_validation("created_at"):
            self._logger.error(
                f"Model created_at validation FAILED for '{self._model_manifest_configuration.get_config_file_path()}'."
            )
            raise ApplicationValidationError("The model manifest creation timestamp is invalid.")
        if not self._timestamp_validation("modified_at"):
            self._logger.error(
                f"Model modified_at validation FAILED for '{self._model_manifest_configuration.get_config_file_path()}'."
            )
            raise ApplicationValidationError(
                "The model manifest modification timestamp is invalid."
            )
        if not self._model_folder_path_validation():
            self._logger.error(
                f"[bold red]Manifest INVALID[/bold red]: [magenta]{self._quoted_model_path(self._model_folder_path)}[/magenta]",
                extra={"markup": True},
            )
            raise ApplicationValidationError("The model manifest storage folder is invalid.")
        self._logger.info(
            f"[green]Manifest VALID[/green]: [green]{self._quoted_model_path(self._model_folder_path)}[/green]",
            extra={"markup": True},
        )

    def _name_validation(self) -> bool:
        """Validate the model name."""
        try:
            if self._model_manifest_configuration.model_name == "":
                self._logger.error("model_name is empty in the manifest.")
                return False
        except AttributeError:
            self._logger.error("model_name attribute is missing in the manifest.")
            return False
        return True

    def _uuid_validation(self) -> bool:
        """Validate the model uuid."""
        try:
            if self._model_manifest_configuration.model_uuid == "":
                self._logger.error("model.uuid is empty in the manifest.")
                return False
        except AttributeError:
            self._logger.error("model.uuid attribute is missing in the manifest.")
            return False
        return True

    def _version_validation(self) -> bool:
        """Validate the model version."""
        try:
            if self._model_manifest_configuration.model_version == "":
                self._logger.error("model.version is empty in the manifest.")
                return False
        except AttributeError:
            self._logger.error("model.version attribute is missing in the manifest.")
            return False
        return True

    def _timestamp_validation(self, field_name: str) -> bool:
        """Return the timestamp validation value owned by this model manifest."""
        try:
            value = str(getattr(self._model_manifest_configuration, field_name, "") or "")
            if value == "":
                self._logger.error(f"{field_name} is empty in the manifest.")
                return False
        except AttributeError:
            self._logger.error(f"{field_name} attribute is missing in the manifest.")
            return False
        return True

    def _model_folder_path_validation(self) -> bool:
        """Return the model folder path validation value owned by this model manifest."""
        if not self._model_manifest_configuration.model_folder_path:
            self._logger.error("Model folder path is empty in the manifest.")
            return False
        if not IO.existdir(self._model_manifest_configuration.model_folder_path):
            self._logger.error(
                f"Model folder path '{self._model_manifest_configuration.model_folder_path}' does not exist."
            )
            return False
        if self._model_manifest_configuration.model_folder_path != self._model_folder_path:
            self._logger.error(
                f"Model folder path '{self._model_manifest_configuration.model_folder_path}' does not match the expected path '{self._model_folder_path}'."
            )
            return False
        return True

    def get_configuration(self) -> Configuration:
        """Return the configuration exposed by this model manifest."""
        if self._model_manifest_configuration:
            return self._model_manifest_configuration
        return self._model_configuration
