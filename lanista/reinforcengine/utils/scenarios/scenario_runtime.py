# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide runtime integration for scenario."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Protocol, cast

from yaml import YAMLError

from lanista.foundation.configuration import AppConfiguration, Configuration
from lanista.foundation.filesystem import only_directories_from_dir
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    get_framework_model_configuration_path,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
    extract_number,
)
from lanista.reinforcengine.utils.scenarios.manifest.scenario_manifest import ScenarioManifest
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    resolve_scenario_index,
)

if TYPE_CHECKING:
    from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest

APP_CONFIGURATION = AppConfiguration()


class _ScenarioManagerLoader(Protocol):
    """Expose the legacy scenario-manager load boundary as a typed operation."""

    def load(self) -> None:
        """Load the configured scenario inventory."""
        ...


def discover_evaluation_scenario_inventory(
    model_manifest: ModelManifest,
    framework_uuid: str,
) -> tuple[tuple[int, ...], int | None]:
    """Read the selected framework's scenario inventory without creating a runtime."""

    manifest_configuration = model_manifest.get_configuration()
    model_folder_path = str(
        getattr(manifest_configuration, "model_folder_path", "") or ""
    ).strip()
    if not model_folder_path:
        return (), None
    framework_configuration_path = get_framework_model_configuration_path(
        model_folder_path,
        framework_uuid,
    )
    try:
        payload = validate_versioned_metadata_payload(
            MetadataKind.FRAMEWORK_CONFIGURATION,
            read_yaml_mapping_file(framework_configuration_path),
        )
        framework_configuration = Configuration(config_dict=payload)
    except (OSError, TypeError, ValueError, YAMLError):
        return (), None

    configured_scenario_root = str(
        getattr(framework_configuration, "scenarios_folder_path", "") or ""
    ).strip()
    if not configured_scenario_root:
        return (), None
    scenario_root = os.path.join(model_folder_path, configured_scenario_root)
    if not os.path.isdir(scenario_root):
        return (), None

    try:
        scenario_paths = sorted(
            only_directories_from_dir(scenario_root),
            key=extract_number,
        )
    except OSError:
        return (), None
    loaded_scenario_count = sum(
        ScenarioManifest(scenario_path).load() for scenario_path in scenario_paths
    )
    available_indices = tuple(range(loaded_scenario_count))
    try:
        configured_index = resolve_scenario_index(
            framework_configuration,
            purpose="evaluation",
            default_index=0,
        )
    except ValueError:
        configured_index = None
    return available_indices, configured_index


def build_scenario_manager(
    model_manifest: ModelManifest,
    logger: logging.Logger,
    *,
    route_entities_provider: RouteEntitiesProvider | None,
) -> ScenarioManager:
    """Create and fully initialize a model resource scenario manager."""
    if route_entities_provider is None:
        raise RuntimeError(
            "Model runtime requires a route-entities provider before scenario setup."
        )
    manifest_cfg = model_manifest.get_configuration()
    manager = ScenarioManager(
        route_entities_provider=route_entities_provider,
        parent_logger_name=logger.name,
        child_logger_name="manager",
        child_logger_level=APP_CONFIGURATION.logs.globally.level,
    )

    scenario_cfg = manager.get_configuration()
    scenario_cfg.set_config_file_path(
        os.path.join(
            manifest_cfg.model_folder_path,
            manifest_cfg.scenario_manager_configuration_file_path,
        )
    )
    scenario_cfg.load()
    scenario_cfg.model_folder_path = manifest_cfg.model_folder_path
    scenario_cfg.scenarios_folder_path = os.path.join(
        manifest_cfg.model_folder_path,
        manifest_cfg.scenarios_folder_path,
    )
    cast(_ScenarioManagerLoader, manager).load()
    return manager


def ensure_sumo_home(logger: logging.Logger) -> None:
    """Ensure the sumo home owned by this scenario runtime module."""
    os.environ["SUMO_HOME"] = APP_CONFIGURATION.sumo_home
    logger.warning(f"SUMO_HOME is set to {os.environ['SUMO_HOME']}")
