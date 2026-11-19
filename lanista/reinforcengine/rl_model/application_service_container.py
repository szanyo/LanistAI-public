# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Logger-aware process adapter for the typed application provider graph."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import cast

from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.logging import LoggerInterface
from lanista.locations import get_app_runtime_state_path, get_models_root_path
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
    ProviderLogger,
)
from lanista.reinforcengine.rl_model.application_providers.graph import (
    ApplicationProviderGraph,
)
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import bind_model_plugin_root
from lanista.reinforcengine.rl_model.runtime_slot_store import RuntimeSlotStore

APP_CONFIGURATION = AppConfiguration()


class ApplicationServiceContainer(LoggerInterface):
    """Adapt process configuration into one explicit application provider DAG."""

    def __init__(
        self,
        *,
        parent_logger_name: str | None = None,
        child_logger_name: str | None = None,
        child_logger_level: object = logging.DEBUG,
        logger_interface_configuration: str | None = None,
        provider_context: ApplicationProviderContext | None = None,
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Build every application singleton and restore persisted runtime state."""

        super().__init__(
            parent_logger_name=parent_logger_name,
            child_logger_name=child_logger_name,
            child_logger_level=child_logger_level,
            logger_interface_configuration=logger_interface_configuration,
        )
        context = provider_context or self._default_provider_context(
            native_gui_mutation_guard,
        )
        if provider_context is None:
            bind_model_plugin_root(context.models_root_provider())
        graph = ApplicationProviderGraph.create(context)
        self._application_services = ApplicationServices.from_provider_graph(graph)
        graph.runtime.core.selection.load_last_successful()

    @property
    def application_services(self) -> ApplicationServices:
        """Expose the typed service projection and no internal construction hooks."""

        return self._application_services

    def _default_provider_context(
        self,
        native_gui_mutation_guard: Callable[[], None],
    ) -> ApplicationProviderContext:
        """Return the default provider context value owned by this application service container."""
        return ApplicationProviderContext(
            logger=cast(ProviderLogger, self._logger),
            slot_store=RuntimeSlotStore(),
            models_root_provider=get_models_root_path,
            runtime_state_path_provider=get_app_runtime_state_path,
            log_level_provider=lambda: APP_CONFIGURATION.logs.globally.level,
            manifest_cls=ModelManifest,
            native_gui_mutation_guard=native_gui_mutation_guard,
        )
