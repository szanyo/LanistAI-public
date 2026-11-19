# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Register and resolve runtime resource."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

from yaml import YAMLError

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.application_errors import PersistenceError
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
    get_framework_resource_path,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file


class RuntimeConfigurationProvider(Protocol):
    """Minimal manifest surface required by the runtime resource registry."""

    def get_configuration(self) -> object:
        """Return the configuration exposed by this runtime configuration provider."""
        ...


@dataclass(frozen=True)
class RuntimeResource:
    """Carry the data required for runtime resource."""

    framework_uuid: str
    framework_configuration: Configuration | None
    runtime_configuration: object
    module_path: str
    class_name: str
    framework_resource_path: str
    framework_configuration_path: str


class RuntimeResourceRegistry:
    """Resolves framework resources from the model configuration contract."""

    _FRAMEWORK_CONFIG_ATTRIBUTES = (
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
    )

    def __init__(
        self,
        *,
        framework_normalizer: Callable[[str | None], str],
    ) -> None:
        """Initialize the runtime resource registry and its owned state."""
        self._framework_normalizer = framework_normalizer

    def _normalize_framework_uuid(self, framework_uuid: str | None) -> str:
        """Normalize the framework uuid accepted by this runtime resource registry."""
        return self._framework_normalizer(framework_uuid)

    @staticmethod
    def _framework_paths(configuration: object) -> dict[str, str]:
        """Return the framework paths value owned by this runtime resource registry."""
        raw_paths = getattr(configuration, "framework_resource_paths", None)
        if isinstance(raw_paths, dict):
            return {str(key): str(value or "") for key, value in raw_paths.items()}
        return {
            framework_uuid: str(getattr(raw_paths, framework_uuid, "") or "")
            for framework_uuid in ("ray", "sb3")
        }

    @staticmethod
    def _resolve_path(path: str, model_folder_path: str) -> str:
        """Resolve the path required by this runtime resource registry."""
        normalized_path = str(path or "").strip()
        if not normalized_path:
            return ""
        if os.path.isabs(normalized_path):
            return normalized_path
        if not model_folder_path:
            return normalized_path
        return os.path.abspath(os.path.join(model_folder_path, normalized_path))

    def framework_resource_path(
        self,
        configuration: object,
        framework_uuid: str,
    ) -> str:
        """Return the framework resource path value owned by this runtime resource registry."""
        normalized_framework_uuid = self._normalize_framework_uuid(framework_uuid)
        model_folder_path = str(getattr(configuration, "model_folder_path", "") or "").strip()
        configured_path = self._framework_paths(configuration).get(normalized_framework_uuid, "")
        resolved_configured_path = self._resolve_path(configured_path, model_folder_path)
        if resolved_configured_path:
            return resolved_configured_path
        if not model_folder_path or not normalized_framework_uuid:
            return ""
        return get_framework_resource_path(model_folder_path, normalized_framework_uuid)

    def framework_configuration_path(
        self,
        configuration: object,
        framework_uuid: str,
    ) -> str:
        """Return the framework configuration path value owned by this runtime resource registry."""
        framework_resource_path = self.framework_resource_path(configuration, framework_uuid)
        if not framework_resource_path:
            return ""
        return os.path.join(framework_resource_path, FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME)

    def load_framework_configuration(
        self,
        model_manifest: RuntimeConfigurationProvider,
        framework_uuid: str,
    ) -> Configuration | None:
        """Load the framework configuration owned by this runtime resource registry."""
        cfg = model_manifest.get_configuration()
        normalized_framework_uuid = self._normalize_framework_uuid(framework_uuid)
        if not normalized_framework_uuid:
            return None

        framework_cfg_path = self.framework_configuration_path(cfg, normalized_framework_uuid)
        if not framework_cfg_path or not os.path.isfile(framework_cfg_path):
            return None

        try:
            payload = validate_versioned_metadata_payload(
                MetadataKind.FRAMEWORK_CONFIGURATION,
                read_yaml_mapping_file(framework_cfg_path),
            )
            framework_cfg = Configuration(config_dict=payload)
            framework_cfg.set_config_file_path(framework_cfg_path)
            projection = {
                attribute_name: attribute_value
                for attribute_name in self._FRAMEWORK_CONFIG_ATTRIBUTES
                if (
                    attribute_value := getattr(
                        framework_cfg,
                        attribute_name,
                        None,
                    )
                )
                is not None
            }
            projection["active_runtime_framework_uuid"] = (
                normalized_framework_uuid
            )
            self._commit_projection(cfg, projection)
        except PersistenceError:
            raise
        except (OSError, TypeError, ValueError, YAMLError) as error:
            raise PersistenceError(
                "Framework configuration is unreadable or uses an unsupported "
                "schema. Run the metadata migration before loading the runtime."
            ) from error
        return framework_cfg

    @staticmethod
    def _commit_projection(
        configuration: object,
        projection: Mapping[str, object],
    ) -> None:
        """Publish a fully prepared projection in one object-state commit."""

        try:
            candidate_state = deepcopy(vars(configuration))
        except (TypeError, ValueError) as error:
            raise PersistenceError(
                "The active model configuration cannot accept a runtime projection."
            ) from error

        config_dict = candidate_state.get("_config_dict")
        if isinstance(config_dict, dict):
            config_dict.update(deepcopy(dict(projection)))
            candidate_state["_changed"] = True
        else:
            candidate_state.update(deepcopy(dict(projection)))

        active_state = vars(configuration)
        active_state.clear()
        active_state.update(candidate_state)

    def resolve(
        self,
        model_manifest: RuntimeConfigurationProvider,
        framework_uuid: str | None,
    ) -> RuntimeResource:
        """Return the resolve value owned by this runtime resource registry."""
        model_configuration = model_manifest.get_configuration()
        normalized_framework_uuid = (
            self._normalize_framework_uuid(framework_uuid)
            or self._normalize_framework_uuid(getattr(model_configuration, "preferred_framework_uuid", None))
        )
        framework_cfg = self.load_framework_configuration(model_manifest, normalized_framework_uuid)
        model_configuration = model_manifest.get_configuration()
        runtime_configuration = framework_cfg if framework_cfg is not None else model_configuration
        implementation = getattr(runtime_configuration, "model_implementation", None)
        module_path = str(getattr(implementation, "module", "") or "")
        class_name = str(getattr(implementation, "object", "") or "")
        resolved_framework_uuid = normalized_framework_uuid or self._normalize_framework_uuid(
            getattr(getattr(model_configuration, "rl_engine_framework", None), "uuid", None)
        )

        return RuntimeResource(
            framework_uuid=resolved_framework_uuid,
            framework_configuration=framework_cfg,
            runtime_configuration=runtime_configuration,
            module_path=module_path,
            class_name=class_name,
            framework_resource_path=self.framework_resource_path(model_configuration, resolved_framework_uuid),
            framework_configuration_path=self.framework_configuration_path(
                model_configuration,
                resolved_framework_uuid,
            ),
        )
