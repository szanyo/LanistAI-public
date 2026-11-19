# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Runtime workflow providers built after session and presentation services."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from lanista.reinforcengine.rl_model.application_providers.catalog import CatalogProviders
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.presentation import (
    PresentationProviders,
)
from lanista.reinforcengine.rl_model.application_providers.runtime import RuntimeCoreProviders
from lanista.reinforcengine.rl_model.application_providers.sessions import (
    SessionCheckpointProviders,
)
from lanista.reinforcengine.rl_model.application_providers.storage_safety import (
    StorageSafetyProviders,
)
from lanista.reinforcengine.rl_model.execution_contracts import ACTIVE_EXECUTION_STATES
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    ConfigurationProvider,
    LoadChecklistResult,
    RuntimeChecklistResult,
)
from lanista.reinforcengine.rl_model.runtime.configuration_state_service import (
    ConfigurationRuntimeStateService,
)
from lanista.reinforcengine.rl_model.runtime.context_service import RuntimeContextService
from lanista.reinforcengine.rl_model.runtime.run_precheck_service import (
    ActionPrecheck,
    RunPrecheckService,
    RunStartPrecheck,
)


@dataclass(frozen=True, slots=True)
class RuntimeWorkflowProviders:
    """Own run prechecks and configuration/runtime state projections."""

    run_precheck: RunPrecheckService
    context: RuntimeContextService
    configuration_state: ConfigurationRuntimeStateService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
        storage: StorageSafetyProviders,
        catalog: CatalogProviders,
        runtime: RuntimeCoreProviders,
        sessions: SessionCheckpointProviders,
        presentation: PresentationProviders,
    ) -> RuntimeWorkflowProviders:
        """Build high-level workflows only after their leaf services exist."""

        runtime_context = _create_runtime_context(
            context,
            catalog,
            runtime,
            sessions,
        )
        run_precheck = _create_run_precheck(
            context,
            storage,
            catalog,
            runtime,
            sessions,
        )
        configuration_state = _create_configuration_state(
            context,
            runtime,
            sessions,
            presentation,
            runtime_context,
            run_precheck,
        )
        return cls(
            run_precheck=run_precheck,
            context=runtime_context,
            configuration_state=configuration_state,
        )


def _create_runtime_context(
    context: ApplicationProviderContext,
    catalog: CatalogProviders,
    runtime: RuntimeCoreProviders,
    sessions: SessionCheckpointProviders,
) -> RuntimeContextService:
    """Create the runtime context owned by this runtime workflows module."""
    metadata = catalog.session_metadata
    return RuntimeContextService(
        active_manifest_provider=context.active_manifest,
        current_framework_provider=lambda: runtime.current_framework_uuid(context),
        selected_session_provider=lambda: context.slot_store.snapshot().session_name,
        selected_checkpoint_provider=lambda: (context.slot_store.snapshot().checkpoint_name),
        sessions_provider=sessions.checkpoints.list_sessions_for_current_model,
        checkpoints_provider=(sessions.checkpoints.list_checkpoints_for_session_for_framework),
        current_model_valid_runtime_provider=runtime.status.current_model_has_valid_runtime,
        session_display_name_provider=metadata.get_session_display_name,
        session_parent_uuid_provider=metadata.get_session_parent_uuid,
        session_created_at_provider=metadata.get_session_created_at,
    )


def _create_run_precheck(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    catalog: CatalogProviders,
    runtime: RuntimeCoreProviders,
    sessions: SessionCheckpointProviders,
) -> RunPrecheckService:
    """Create the run precheck owned by this runtime workflows module."""
    return RunPrecheckService(
        selected_manifest_provider=runtime.staging.get_selected_manifest,
        active_manifest_provider=context.active_manifest,
        has_current_model_provider=(
            lambda: runtime.status.is_same_loaded_model(
                runtime.staging.get_selected_manifest(),
                framework_uuid=runtime.current_framework_uuid(context),
            )
        ),
        active_configuration_provider=context.active_configuration,
        current_framework_provider=lambda: runtime.current_framework_uuid(context) or "",
        framework_normalizer=context.normalize_framework_uuid,
        model_load_checklist_builder=lambda manifest: _build_model_load_checklist(
            runtime,
            manifest,
        ),
        framework_runtime_checklist_builder=lambda manifest, framework: (
            _build_framework_runtime_checklist(runtime, manifest, framework)
        ),
        current_model_structure_validator=runtime.status.current_model_has_valid_structure,
        session_closed_for_training_checker=(sessions.checkpoints.session_is_closed_for_training),
        session_checkpoint_checker=sessions.checkpoints.session_has_checkpoints,
        session_training_source_checker=(catalog.session_has_training_checkpoint_source),
        session_framework_uuid_provider=catalog.sessions.get_session_framework_uuid,
        checkpoint_selection_checker=(
            lambda session, checkpoint, framework: checkpoint
            in catalog.checkpoints.list_for_session(session, framework)
        ),
        administration_block_reason_provider=storage.safety.run_start_block_reason,
        native_gui_mutation_guard=context.native_gui_mutation_guard,
    )


def _create_configuration_state(
    context: ApplicationProviderContext,
    runtime: RuntimeCoreProviders,
    sessions: SessionCheckpointProviders,
    presentation: PresentationProviders,
    runtime_context: RuntimeContextService,
    precheck: RunPrecheckService,
) -> ConfigurationRuntimeStateService:
    """Create the configuration state owned by this runtime workflows module."""
    def confirm(
        mode: RLModelRunningModes,
        framework_uuid: str,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> ActionPrecheck:
        """Confirm the operation owned by this runtime workflows module."""
        return precheck.evaluate_configuration_confirm(
            selected_mode=mode,
            selected_framework_uuid=framework_uuid,
            selected_session_name=session_name,
            selected_checkpoint_name=checkpoint_name,
        )

    def run_start(
        mode: RLModelRunningModes,
        session_name: str | None,
        checkpoint_name: str | None,
        has_previous_models: bool,
    ) -> RunStartPrecheck:
        """Perform the run start step within the enclosing create configuration state workflow."""
        return precheck.evaluate_run_start(
            mode=mode,
            selected_session=session_name,
            selected_checkpoint=checkpoint_name,
            has_previous_models=has_previous_models,
        )

    return ConfigurationRuntimeStateService(
        runtime_selection_provider=(runtime_context.get_current_runtime_selection_state),
        active_manifest_provider=context.active_manifest,
        active_configuration_provider=context.active_configuration,
        current_model_display_name_provider=(presentation.context.get_current_model_display_name),
        current_framework_provider=lambda: runtime.current_framework_uuid(context) or "",
        model_load_checklist_builder=lambda manifest: _build_model_load_checklist(
            runtime,
            manifest,
        ),
        framework_runtime_checklist_builder=lambda manifest, framework: (
            _build_framework_runtime_checklist(runtime, manifest, framework)
        ),
        session_options_provider=(sessions.selection.list_session_options_for_current_model),
        checkpoint_options_provider=(sessions.selection.list_checkpoint_options_for_current_model),
        can_evaluate_provider=runtime.status.current_model_can_evaluate,
        confirm_precheck_evaluator=confirm,
        create_model_precheck_evaluator=precheck.evaluate_create_model,
        create_session_precheck_evaluator=precheck.evaluate_create_session,
        run_start_precheck_evaluator=run_start,
        configuration_locked_provider=lambda: _configuration_is_locked(context),
    )


def _configuration_is_locked(context: ApplicationProviderContext) -> bool:
    """Keep configuration read-only until the active worker has fully stopped."""

    runtime = context.slot_store.snapshot().model
    if runtime is None:
        return False
    execution = runtime.get_execution_snapshot()
    return execution.state in ACTIVE_EXECUTION_STATES or execution.worker_alive


def _build_model_load_checklist(
    runtime: RuntimeCoreProviders,
    manifest: object,
) -> LoadChecklistResult:
    """Build the model load checklist required by this runtime workflows module."""
    return runtime.checklist.build_model_load_checklist(cast(ConfigurationProvider, manifest))


def _build_framework_runtime_checklist(
    runtime: RuntimeCoreProviders,
    manifest: object,
    framework_uuid: str,
) -> RuntimeChecklistResult:
    """Build the framework runtime checklist required by this runtime workflows module."""
    return runtime.checklist.build_framework_runtime_checklist(
        cast(ConfigurationProvider, manifest),
        framework_uuid,
    )
