# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Runtime loading and active-slot provider group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from lanista.reinforcengine.application_errors import (
    RuntimeUnavailableError,
    exception_from_failure,
)
from lanista.reinforcengine.rl_model.application_providers.catalog import CatalogProviders
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.storage_safety import (
    StorageSafetyProviders,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.runtime.access_service import RuntimeAccessService
from lanista.reinforcengine.rl_model.runtime.activation_service import RuntimeActivationService
from lanista.reinforcengine.rl_model.runtime.active_selection_service import (
    ActiveRuntimeSelectionRequest,
    ActiveRuntimeSelectionService,
)
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    ConfigurationProvider,
    RuntimeChecklistService,
)
from lanista.reinforcengine.rl_model.runtime.import_service import ImportService
from lanista.reinforcengine.rl_model.runtime.load_finalization_service import (
    LoadFinalizationService,
)
from lanista.reinforcengine.rl_model.runtime.load_service import LoadService
from lanista.reinforcengine.rl_model.runtime.requests import (
    RuntimeInitializationRequest,
    RuntimeSelectionSnapshot,
)
from lanista.reinforcengine.rl_model.runtime.retirement_service import RuntimeRetirementService
from lanista.reinforcengine.rl_model.runtime.selection_catalog_service import (
    RuntimeSelectionCatalogService,
)
from lanista.reinforcengine.rl_model.runtime.selection_service import RuntimeSelectionService
from lanista.reinforcengine.rl_model.runtime.status_service import RuntimeStatusService
from lanista.reinforcengine.rl_model.runtime_slot_commit_service import RuntimeSlotCommitService
from lanista.reinforcengine.rl_model.staged_model_selection_service import (
    StagedModelSelectionService,
)


@dataclass(frozen=True, slots=True)
class RuntimeCoreProviders:
    """Own runtime preparation, activation, selection, and status singletons."""

    access: RuntimeAccessService
    checklist: RuntimeChecklistService
    activation: RuntimeActivationService
    selection: RuntimeSelectionService
    active_selection: ActiveRuntimeSelectionService
    staging: StagedModelSelectionService
    status: RuntimeStatusService
    importer: ImportService
    loader: LoadService
    load_finalization: LoadFinalizationService
    slot_commit: RuntimeSlotCommitService
    retirement: RuntimeRetirementService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
        storage: StorageSafetyProviders,
        catalog: CatalogProviders,
    ) -> RuntimeCoreProviders:
        """Build runtime services once in their real dependency order."""

        checklist = RuntimeChecklistService()
        importer = _create_importer(context)
        selection = _create_runtime_selection(context)
        active_selection = _create_active_selection(context, storage, selection)
        slot_commit = _create_slot_commit(context, storage)
        load_finalization = _create_load_finalization(context, selection)
        loader = _create_loader(context, checklist, importer)
        retirement = RuntimeRetirementService(
            logger=context.logger,
            degraded_state_reporter=storage.safety.mark_degraded,
        )
        activation = _create_activation(
            context,
            storage,
            loader,
            load_finalization,
            slot_commit,
            retirement,
        )
        staging = _create_staging(context, selection)
        status = _create_status(context, catalog, checklist, activation, selection)
        return cls(
            access=RuntimeAccessService(slot_store=context.slot_store),
            checklist=checklist,
            activation=activation,
            selection=selection,
            active_selection=active_selection,
            staging=staging,
            status=status,
            importer=importer,
            loader=loader,
            load_finalization=load_finalization,
            slot_commit=slot_commit,
            retirement=retirement,
        )

    def current_framework_uuid(
        self,
        context: ApplicationProviderContext,
    ) -> str | None:
        """Resolve the active framework from the slot and persisted preference."""

        slot = context.slot_store.snapshot()
        return self.selection.resolve_current_model_framework_uuid(
            configuration=context.active_configuration(),
            current_runtime_framework_uuid=slot.framework_uuid,
        )

    def set_selection_or_raise(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        *,
        persist: bool = False,
    ) -> None:
        """Commit a typed runtime selection or raise its application error."""

        result = self.active_selection.set_selection(
            ActiveRuntimeSelectionRequest(
                session_name=session_name,
                checkpoint_name=checkpoint_name,
                persist=persist,
            )
        )
        if result.failure is not None:
            raise exception_from_failure(result.failure)
        if result.value is None:
            raise RuntimeUnavailableError(
                "The active runtime selection was not committed."
            )


def _create_importer(context: ApplicationProviderContext) -> ImportService:
    """Create the importer owned by this runtime module."""
    return ImportService(
        framework_normalizer=context.normalize_framework_uuid,
        progress_reporter=context.post_model_load_progress,
        logger=context.logger,
    )


def _create_runtime_selection(
    context: ApplicationProviderContext,
) -> RuntimeSelectionService:
    """Create the runtime selection owned by this runtime module."""
    def manifest_factory(model_folder: str) -> ModelManifest:
        """Perform the manifest factory step within the enclosing create runtime selection
        workflow.
        """
        return context.manifest_cls(
            model_folder,
            parent_logger_name=context.logger.name,
            child_logger_name="model_manifest_discovery",
            child_logger_level=context.log_level_provider(),
        )

    return RuntimeSelectionService(
        runtime_state_path_provider=context.runtime_state_path_provider,
        framework_normalizer=context.normalize_framework_uuid,
        models_root_provider=context.models_root_provider,
        last_successful_state_provider=context.last_successful_runtime_state,
        last_successful_state_committer=context.commit_last_successful_runtime_state,
        manifest_factory=manifest_factory,
        selection_catalog=RuntimeSelectionCatalogService(
            models_root_provider=context.models_root_provider,
            session_manager_factory=context.session_manager_for_model,
            framework_normalizer=context.normalize_framework_uuid,
        ),
    )


def _create_active_selection(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    selection: RuntimeSelectionService,
) -> ActiveRuntimeSelectionService:
    """Create the active selection owned by this runtime module."""
    return ActiveRuntimeSelectionService(
        active_manifest_provider=context.active_manifest,
        runtime_selection_service=selection,
        current_framework_provider=(
            lambda: context.slot_store.snapshot().framework_uuid
        ),
        state_provider=context.active_selection_state,
        state_committer=context.commit_active_selection,
        administration_operation_factory=storage.safety.administration_operation,
        logger=context.logger,
        native_gui_mutation_guard=context.native_gui_mutation_guard,
    )


def _create_slot_commit(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
) -> RuntimeSlotCommitService:
    """Create the slot commit owned by this runtime module."""
    return RuntimeSlotCommitService(
        slot_store=context.slot_store,
        administration_operation_factory=storage.safety.administration_operation,
        recovery_operation_factory=storage.safety.runtime_recovery_operation,
        storage_recovery_operation_factory=(
            storage.recovery.validated_receipt_operation
        ),
    )


def _create_load_finalization(
    context: ApplicationProviderContext,
    selection: RuntimeSelectionService,
) -> LoadFinalizationService:
    """Create the load finalization owned by this runtime module."""
    return LoadFinalizationService(
        normalize_selection=selection.normalize_selection,
        normalize_framework_uuid=context.normalize_framework_uuid,
        manifest_folder_name_provider=RuntimeStatusService.get_manifest_folder_name,
        persist_last_successful_state=(
            lambda model_name, framework_uuid, session_name, checkpoint_name: (
                selection.persist_loaded_runtime_state(
                    model_name=model_name,
                    framework_uuid=framework_uuid,
                    session_name=session_name,
                    checkpoint_name=checkpoint_name,
                )
            )
        ),
        logger=context.logger,
    )


def _create_loader(
    context: ApplicationProviderContext,
    checklist: RuntimeChecklistService,
    importer: ImportService,
) -> LoadService:
    """Create the loader owned by this runtime module."""
    return LoadService(
        model_load_checklist_builder=checklist.build_model_load_checklist,
        framework_runtime_checklist_builder=(
            checklist.build_framework_runtime_checklist
        ),
        model_runtime_loader=importer.load_model,
        framework_normalizer=context.normalize_framework_uuid,
        progress_reporter=context.post_model_load_progress,
        checklist_summary_formatter=checklist.format_checklist_summary,
        checklist_line_formatter=checklist.format_checklist_line,
        checklist_path_shortener=checklist.shorten_checklist_path,
        logger=context.logger,
    )


def _create_activation(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    loader: LoadService,
    finalization: LoadFinalizationService,
    slot_commit: RuntimeSlotCommitService,
    retirement: RuntimeRetirementService,
) -> RuntimeActivationService:
    """Create the activation owned by this runtime module."""
    def initialization_request(manifest: ModelManifest) -> RuntimeInitializationRequest:
        """Perform the initialization request step within the enclosing create activation workflow.
        """
        configuration = manifest.get_configuration()
        return RuntimeInitializationRequest(
            parent_logger_name=context.logger.name,
            child_logger_name=str(
                getattr(configuration, "model_name", "") or ""
            ).lower(),
            child_logger_level=context.log_level_provider(),
            execution_lease_factory=storage.safety,
            runtime_selection_provider=lambda: _runtime_selection_snapshot(context),
        )

    return RuntimeActivationService(
        load_service=loader,
        finalization_service=finalization,
        slot_commit_service=slot_commit,
        retirement_service=retirement,
        administration_safety=storage.safety,
        storage_recovery_service=storage.recovery,
        slot_store=context.slot_store,
        initialization_request_factory=initialization_request,
        progress_reporter=context.post_model_load_progress,
        info_logger=context.logger.info,
        error_logger=context.logger.error,
        debug_logger=context.logger.debug,
        event_publish_failure_logger=context.log_runtime_event_publish_failure,
        native_gui_mutation_guard=context.native_gui_mutation_guard,
    )


def _runtime_selection_snapshot(
    context: ApplicationProviderContext,
) -> RuntimeSelectionSnapshot:
    """Project the active slot's session/checkpoint pair atomically."""

    slot = context.slot_store.snapshot()
    return RuntimeSelectionSnapshot(
        session_name=slot.session_name,
        checkpoint_name=slot.checkpoint_name,
    )


def _create_staging(
    context: ApplicationProviderContext,
    selection: RuntimeSelectionService,
) -> StagedModelSelectionService:
    """Create the staging owned by this runtime module."""
    return StagedModelSelectionService(
        selected_manifest_provider=lambda: context.slot_store.snapshot().manifest,
        runtime_selection_loader=selection.load_staged_selection,
        empty_selection_factory=lambda: selection.normalize_selection(None, None),
        state_committer=context.commit_staged_selection,
        native_gui_mutation_guard=context.native_gui_mutation_guard,
    )


def _create_status(
    context: ApplicationProviderContext,
    catalog: CatalogProviders,
    checklist: RuntimeChecklistService,
    activation: RuntimeActivationService,
    selection: RuntimeSelectionService,
) -> RuntimeStatusService:
    """Create the status owned by this runtime module."""
    def current_framework() -> str | None:
        """Perform the current framework step within the enclosing create status workflow."""
        slot = context.slot_store.snapshot()
        return selection.resolve_current_model_framework_uuid(
            configuration=context.active_configuration(),
            current_runtime_framework_uuid=slot.framework_uuid,
        )

    def loaded_framework() -> str | None:
        """Perform the loaded framework step within the enclosing create status workflow."""
        slot = context.slot_store.snapshot()
        if slot.model is None:
            return None
        return slot.last_successful_framework_uuid or slot.framework_uuid

    def model_structure_is_valid(manifest: object) -> bool:
        """Return whether model structure is valid applies to this runtime module."""
        configuration_provider = cast(ConfigurationProvider, manifest)
        return checklist.build_model_load_checklist(configuration_provider).valid

    return RuntimeStatusService(
        active_manifest_provider=context.active_manifest,
        loaded_manifest_provider=context.loaded_manifest,
        last_model_load_issue_provider=activation.get_last_issue,
        model_structure_validator=model_structure_is_valid,
        current_framework_provider=current_framework,
        loaded_framework_provider=loaded_framework,
        sessions_provider=catalog.sessions.list_sessions_for_current_model,
        checkpoint_names_provider=catalog.checkpoints.list_for_session,
    )
