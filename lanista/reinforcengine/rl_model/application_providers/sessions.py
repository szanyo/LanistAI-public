# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Session and checkpoint workflow provider group."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.application_providers.catalog import CatalogProviders
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.runtime import RuntimeCoreProviders
from lanista.reinforcengine.rl_model.application_providers.storage_safety import (
    StorageSafetyProviders,
)
from lanista.reinforcengine.rl_model.sessions.checkpoint_catalog_service import (
    SessionCheckpointCatalogService,
)
from lanista.reinforcengine.rl_model.sessions.creation_service import SessionCreationService
from lanista.reinforcengine.rl_model.sessions.selection_service import SessionSelectionService


@dataclass(frozen=True, slots=True)
class SessionCheckpointProviders:
    """Own session creation, selection, and checkpoint query coordination."""

    checkpoints: SessionCheckpointCatalogService
    selection: SessionSelectionService
    creation: SessionCreationService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
        storage: StorageSafetyProviders,
        catalog: CatalogProviders,
        runtime: RuntimeCoreProviders,
    ) -> SessionCheckpointProviders:
        """Build session services after runtime status becomes available."""

        checkpoints = SessionCheckpointCatalogService(
            session_status_provider=catalog.sessions.get_session_status,
            sessions_provider=catalog.sessions.list_sessions_for_current_model,
            checkpoint_names_provider=catalog.checkpoints.list_for_session,
            checkpoint_tree_entries_provider=catalog.checkpoints.build_tree_entries,
            latest_session_provider=runtime.status.get_latest_session_for_current_model,
            latest_checkpoint_provider=runtime.status.get_latest_checkpoint_for_session,
            current_model_can_evaluate_provider=runtime.status.current_model_can_evaluate,
            current_model_has_training_session_provider=(
                runtime.status.current_model_has_training_session
            ),
            session_has_checkpoints_provider=runtime.status.session_has_checkpoints,
        )
        selection = _create_session_selection(catalog, checkpoints)
        creation = SessionCreationService(
            active_configuration_provider=context.active_configuration,
            sessions_root_provider=context.sessions_root,
            session_config_path_provider=catalog.sessions.get_session_config_path,
            models_root_provider=context.models_root_provider,
            runtime_selection_state_provider=context.active_selection_state,
            runtime_selection_state_committer=context.commit_active_selection,
            runtime_slot_operation_factory=context.slot_store.operation,
            administration_operation_factory=storage.safety.administration_operation,
            logger=context.logger,
        )
        return cls(
            checkpoints=checkpoints,
            selection=selection,
            creation=creation,
        )

    def replacement_session(
        self,
        runtime: RuntimeCoreProviders,
        context: ApplicationProviderContext,
        deleted_session_name: str,
    ) -> str | None:
        """Choose the first compatible session that is not being deleted."""

        framework_uuid = runtime.current_framework_uuid(context)
        sessions = self.checkpoints.list_sessions_for_current_model(framework_uuid)
        return next(
            (name for name in sessions if name != deleted_session_name),
            None,
        )


def _create_session_selection(
    catalog: CatalogProviders,
    checkpoints: SessionCheckpointCatalogService,
) -> SessionSelectionService:
    """Create the session selection owned by this sessions module."""
    metadata = catalog.session_metadata
    return SessionSelectionService(
        session_configuration_loader=catalog.load_session_configuration,
        session_reference_display_provider=(
            catalog.sessions.get_session_reference_display_from_summary
        ),
        session_identity_uuid_provider=metadata.get_session_identity_uuid,
        session_display_name_without_uuid_provider=(
            metadata.get_session_display_name_without_uuid
        ),
        session_parent_uuid_provider=metadata.get_session_parent_uuid,
        session_created_at_provider=metadata.get_session_created_at,
        session_last_modified_at_provider=metadata.get_session_last_modified_at,
        session_description_provider=metadata.get_session_description,
        session_child_uuids_provider=metadata.get_session_child_uuids,
        session_framework_uuid_provider=catalog.sessions.get_session_framework_uuid,
        sessions_provider=checkpoints.list_sessions_for_current_model,
        checkpoint_names_provider=(
            checkpoints.list_checkpoints_for_session_for_framework
        ),
    )
