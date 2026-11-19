# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed public projection of the internal application provider graph."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationSafetyService,
)
from lanista.reinforcengine.rl_model.application_providers.graph import (
    ApplicationProviderGraph,
)
from lanista.reinforcengine.rl_model.evaluation.library_application_service import (
    EvaluationLibraryApplicationService,
)
from lanista.reinforcengine.rl_model.checkpoints.administration_command_service import (
    CheckpointAdministrationCommandService,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_reader import CheckpointCatalogReader
from lanista.reinforcengine.rl_model.comparison.tensorboard_service import TensorboardPackService
from lanista.reinforcengine.rl_model.inspect_toggle_service import InspectToggleService
from lanista.reinforcengine.rl_model.inspection_target_service import (
    InspectionTargetService,
)
from lanista.reinforcengine.rl_model.metadata.edit_service import MetadataEditService
from lanista.reinforcengine.rl_model.model_creation_service import ModelCreationService
from lanista.reinforcengine.rl_model.model_discovery_service import ModelDiscoveryService
from lanista.reinforcengine.rl_model.model_storage_command_service import (
    ModelStorageCommandService,
)
from lanista.reinforcengine.rl_model.presentation_context_service import (
    PresentationContextService,
)
from lanista.reinforcengine.rl_model.runtime.access_service import RuntimeAccessService
from lanista.reinforcengine.rl_model.runtime.activation_service import RuntimeActivationService
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionService,
)
from lanista.reinforcengine.rl_model.runtime.checklist_service import RuntimeChecklistService
from lanista.reinforcengine.rl_model.runtime.configuration_state_service import (
    ConfigurationRuntimeStateService,
)
from lanista.reinforcengine.rl_model.runtime.context_service import RuntimeContextService
from lanista.reinforcengine.rl_model.runtime.run_precheck_service import RunPrecheckService
from lanista.reinforcengine.rl_model.runtime.selection_service import RuntimeSelectionService
from lanista.reinforcengine.rl_model.runtime.status_service import RuntimeStatusService
from lanista.reinforcengine.rl_model.sessions.checkpoint_catalog_service import (
    SessionCheckpointCatalogService,
)
from lanista.reinforcengine.rl_model.sessions.creation_service import SessionCreationService
from lanista.reinforcengine.rl_model.sessions.deletion_command_service import (
    SessionDeletionCommandService,
)
from lanista.reinforcengine.rl_model.sessions.lineage_service import LineageService
from lanista.reinforcengine.rl_model.sessions.metadata_service import SessionMetadataService
from lanista.reinforcengine.rl_model.sessions.repository_service import SessionRepositoryService
from lanista.reinforcengine.rl_model.sessions.selection_service import SessionSelectionService
from lanista.reinforcengine.rl_model.sessions.summary_service import SessionSummaryService
from lanista.reinforcengine.rl_model.staged_model_selection_service import (
    StagedModelSelectionService,
)
from lanista.reinforcengine.rl_model.storage.layout_service import StorageLayoutService


@dataclass(frozen=True, slots=True)
class CatalogFacade:
    """Catalog discovery and model-creation workflows."""

    models: ModelDiscoveryService
    creation: ModelCreationService
    checkpoints: CheckpointCatalogReader
    session_checkpoints: SessionCheckpointCatalogService


@dataclass(frozen=True, slots=True)
class RuntimeFacade:
    """Runtime loading, selection, validation, and status services."""

    access: RuntimeAccessService
    checklist: RuntimeChecklistService
    activation: RuntimeActivationService
    selection: RuntimeSelectionService
    active_selection: ActiveRuntimeSelectionService
    staging: StagedModelSelectionService
    run_precheck: RunPrecheckService
    configuration_state: ConfigurationRuntimeStateService
    status: RuntimeStatusService
    context: RuntimeContextService


@dataclass(frozen=True, slots=True)
class SessionFacade:
    """Session metadata, persistence, creation, selection, and lineage workflows."""

    summaries: SessionSummaryService
    repository: SessionRepositoryService
    metadata: SessionMetadataService
    selection: SessionSelectionService
    creation: SessionCreationService
    lineage: LineageService


@dataclass(frozen=True, slots=True)
class AdministrationFacade:
    """Typed model, session, checkpoint, and supporting command boundaries."""

    models: ModelStorageCommandService
    sessions: SessionDeletionCommandService
    checkpoints: CheckpointAdministrationCommandService
    metadata_edits: MetadataEditService
    safety: AdministrationSafetyService
    storage: StorageLayoutService
    tensorboard_packs: TensorboardPackService


@dataclass(frozen=True, slots=True)
class InspectionFacade:
    """Inspection policy and presentation-context services."""

    toggle: InspectToggleService
    presentation: PresentationContextService
    targets: InspectionTargetService


@dataclass(frozen=True, slots=True)
class ApplicationServices:
    """Expose stable responsibility groups without leaking internal providers."""

    catalog: CatalogFacade
    runtime: RuntimeFacade
    sessions: SessionFacade
    administration: AdministrationFacade
    inspection: InspectionFacade
    evaluation_libraries: EvaluationLibraryApplicationService

    @classmethod
    def from_provider_graph(
        cls,
        graph: ApplicationProviderGraph,
    ) -> ApplicationServices:
        """Project one already-built graph without constructing new services."""

        runtime = graph.runtime.core
        workflows = graph.runtime.workflows
        catalog = graph.catalog
        sessions = graph.sessions
        administration = graph.administration
        presentation = graph.presentation
        return cls(
            catalog=CatalogFacade(
                models=catalog.models,
                creation=graph.storage.model_creation,
                checkpoints=catalog.checkpoints,
                session_checkpoints=sessions.checkpoints,
            ),
            runtime=RuntimeFacade(
                access=runtime.access,
                checklist=runtime.checklist,
                activation=runtime.activation,
                selection=runtime.selection,
                active_selection=runtime.active_selection,
                staging=runtime.staging,
                run_precheck=workflows.run_precheck,
                configuration_state=workflows.configuration_state,
                status=runtime.status,
                context=workflows.context,
            ),
            sessions=SessionFacade(
                summaries=catalog.session_summaries,
                repository=catalog.sessions,
                metadata=catalog.session_metadata,
                selection=sessions.selection,
                creation=sessions.creation,
                lineage=catalog.lineage,
            ),
            administration=AdministrationFacade(
                models=administration.models,
                sessions=administration.sessions,
                checkpoints=administration.checkpoints,
                metadata_edits=administration.metadata_edits,
                safety=graph.storage.safety,
                storage=graph.storage.layout,
                tensorboard_packs=administration.tensorboard_packs,
            ),
            inspection=InspectionFacade(
                toggle=presentation.toggle,
                presentation=presentation.context,
                targets=presentation.targets,
            ),
            evaluation_libraries=graph.evaluation_libraries,
        )
