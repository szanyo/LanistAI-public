# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Explicit acyclic construction graph for application service providers."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.application_providers.administration import (
    AdministrationProviders,
)
from lanista.reinforcengine.rl_model.application_providers.catalog import CatalogProviders
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.presentation import (
    PresentationProviders,
)
from lanista.reinforcengine.rl_model.application_providers.runtime import RuntimeCoreProviders
from lanista.reinforcengine.rl_model.application_providers.runtime_workflows import (
    RuntimeWorkflowProviders,
)
from lanista.reinforcengine.rl_model.application_providers.sessions import (
    SessionCheckpointProviders,
)
from lanista.reinforcengine.rl_model.application_providers.storage_safety import (
    StorageSafetyProviders,
)
from lanista.reinforcengine.rl_model.evaluation.library_application_service import (
    EvaluationLibraryApplicationService,
)


@dataclass(frozen=True, slots=True)
class RuntimeProviders:
    """Combine runtime core and workflow phases without rebuilding either."""

    core: RuntimeCoreProviders
    workflows: RuntimeWorkflowProviders


@dataclass(frozen=True, slots=True)
class ApplicationProviderGraph:
    """Own each provider group exactly once in a statically ordered DAG."""

    storage: StorageSafetyProviders
    catalog: CatalogProviders
    runtime: RuntimeProviders
    sessions: SessionCheckpointProviders
    presentation: PresentationProviders
    administration: AdministrationProviders
    evaluation_libraries: EvaluationLibraryApplicationService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
    ) -> ApplicationProviderGraph:
        """Construct the complete graph and reject missing wiring immediately."""

        storage = StorageSafetyProviders.create(context)
        catalog = CatalogProviders.create(context, storage)
        runtime_core = RuntimeCoreProviders.create(context, storage, catalog)
        sessions = SessionCheckpointProviders.create(
            context,
            storage,
            catalog,
            runtime_core,
        )
        presentation = PresentationProviders.create(
            context,
            catalog,
            runtime_core,
        )
        workflows = RuntimeWorkflowProviders.create(
            context,
            storage,
            catalog,
            runtime_core,
            sessions,
            presentation,
        )
        administration = AdministrationProviders.create(
            context,
            storage,
            catalog,
            runtime_core,
            sessions,
            presentation,
        )
        evaluation_libraries = EvaluationLibraryApplicationService(
            active_configuration_provider=context.active_configuration,
            models_root_provider=context.models_root_provider,
            route_entities_provider=lambda: getattr(
                context.slot_store.snapshot().model, "route_entities_provider", None
            ),
        )
        return cls(
            storage=storage,
            catalog=catalog,
            runtime=RuntimeProviders(
                core=runtime_core,
                workflows=workflows,
            ),
            sessions=sessions,
            presentation=presentation,
            administration=administration,
            evaluation_libraries=evaluation_libraries,
        )
