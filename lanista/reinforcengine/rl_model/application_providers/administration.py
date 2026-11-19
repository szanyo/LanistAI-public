# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Model, session, and checkpoint administration provider group."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

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
from lanista.reinforcengine.rl_model.checkpoints.administration_command_service import (
    CheckpointAdministrationCommandService,
)
from lanista.reinforcengine.rl_model.checkpoints.mutation_repository import (
    CheckpointMutationRepository,
)
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import extract_checkpoint_steps
from lanista.reinforcengine.rl_model.comparison.tensorboard_service import TensorboardPackService
from lanista.reinforcengine.rl_model.metadata.edit_service import MetadataEditService
from lanista.reinforcengine.rl_model.model_storage_command_service import (
    ModelStorageCommandService,
)
from lanista.reinforcengine.rl_model.sessions.deletion_command_service import (
    SessionDeletionCommandService,
)

type SelectionCommitter = Callable[[str | None, str | None, bool], None]


@dataclass(frozen=True, slots=True)
class AdministrationProviders:
    """Own all destructive command services behind one shared policy."""

    models: ModelStorageCommandService
    sessions: SessionDeletionCommandService
    checkpoints: CheckpointAdministrationCommandService
    metadata_edits: MetadataEditService
    checkpoint_mutations: CheckpointMutationRepository
    tensorboard_packs: TensorboardPackService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
        storage: StorageSafetyProviders,
        catalog: CatalogProviders,
        runtime: RuntimeCoreProviders,
        sessions: SessionCheckpointProviders,
        presentation: PresentationProviders,
    ) -> AdministrationProviders:
        """Construct commands with one policy and one active-slot projection."""

        checkpoint_mutations = _create_checkpoint_mutations(
            context,
            catalog,
            runtime,
        )

        def set_selection(
            session_name: str | None,
            checkpoint_name: str | None,
            persist: bool = False,
        ) -> None:
            """Set the selection owned by this administration providers."""
            runtime.set_selection_or_raise(
                session_name,
                checkpoint_name,
                persist=persist,
            )

        return cls(
            models=_create_model_commands(context, storage),
            sessions=_create_session_commands(
                context,
                storage,
                runtime,
                sessions,
                set_selection,
            ),
            checkpoints=_create_checkpoint_commands(
                context,
                storage,
                sessions,
                checkpoint_mutations,
                set_selection,
            ),
            metadata_edits=_create_metadata_edits(context, storage, catalog),
            checkpoint_mutations=checkpoint_mutations,
            tensorboard_packs=_create_tensorboard_packs(
                context,
                storage,
                catalog,
                presentation,
            ),
        )


def _create_checkpoint_mutations(
    context: ApplicationProviderContext,
    catalog: CatalogProviders,
    runtime: RuntimeCoreProviders,
) -> CheckpointMutationRepository:
    """Create the checkpoint mutations owned by this administration module."""
    return CheckpointMutationRepository(
        sessions_root_provider=context.sessions_root,
        session_framework_provider=catalog.sessions.get_session_framework_uuid,
        current_framework_provider=lambda: runtime.current_framework_uuid(context),
        models_root_provider=context.models_root_provider,
        step_extractor=extract_checkpoint_steps,
        logger=context.logger,
    )


def _create_model_commands(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
) -> ModelStorageCommandService:
    """Create the model commands owned by this administration module."""
    return ModelStorageCommandService(
        policy=storage.command_policy,
        models_root_provider=context.models_root_provider,
        clear_current_model_state_provider=context.clear_current_model_state,
    )


def _create_session_commands(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    runtime: RuntimeCoreProviders,
    sessions: SessionCheckpointProviders,
    set_selection: SelectionCommitter,
) -> SessionDeletionCommandService:
    """Create the session commands owned by this administration module."""
    return SessionDeletionCommandService(
        policy=storage.command_policy,
        models_root_provider=context.models_root_provider,
        session_manager_provider=context.active_session_manager,
        selected_session_provider=lambda: context.slot_store.snapshot().session_name,
        selected_checkpoint_provider=lambda: (
            context.slot_store.snapshot().checkpoint_name
        ),
        get_replacement_session_provider=lambda deleted: sessions.replacement_session(
            runtime,
            context,
            deleted,
        ),
        get_latest_checkpoint_provider=(
            sessions.checkpoints.get_latest_checkpoint_for_session
        ),
        set_current_model_runtime_selection_provider=set_selection,
    )


def _create_checkpoint_commands(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    sessions: SessionCheckpointProviders,
    checkpoint_mutations: CheckpointMutationRepository,
    set_selection: SelectionCommitter,
) -> CheckpointAdministrationCommandService:
    """Create the checkpoint commands owned by this administration module."""
    return CheckpointAdministrationCommandService(
        policy=storage.command_policy,
        checkpoint_mutations=checkpoint_mutations,
        selected_session_provider=lambda: context.slot_store.snapshot().session_name,
        selected_checkpoint_provider=lambda: (
            context.slot_store.snapshot().checkpoint_name
        ),
        get_latest_checkpoint_provider=(
            sessions.checkpoints.get_latest_checkpoint_for_session
        ),
        set_current_model_runtime_selection_provider=set_selection,
    )


def _create_metadata_edits(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    catalog: CatalogProviders,
) -> MetadataEditService:
    """Create the metadata edits owned by this administration module."""
    return MetadataEditService(
        active_configuration_provider=context.active_configuration,
        models_root_provider=context.models_root_provider,
        checkpoint_catalog_provider=lambda: catalog.checkpoints,
        administration_operation_factory=lambda: _metadata_administration_operation(
            storage
        ),
        degraded_state_reporter=storage.safety.mark_degraded,
        logger=context.logger,
    )


@contextmanager
def _metadata_administration_operation(
    storage: StorageSafetyProviders,
) -> Iterator[None]:
    """Return the metadata administration operation value provided by this administration
module."""
    with storage.safety.administration_operation():
        yield


def _create_tensorboard_packs(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    catalog: CatalogProviders,
    presentation: PresentationProviders,
) -> TensorboardPackService:
    """Create the tensorboard packs owned by this administration module."""
    return TensorboardPackService(
        configuration_provider=context.active_configuration,
        current_model_display_provider=(
            presentation.context.get_current_model_display_name
        ),
        model_candidates_provider=lambda: catalog.models.discover_model_candidates(None),
        sanitize_name=TensorboardPackService.sanitize_tensorboard_name,
        logger=context.logger,
        models_root_provider=context.models_root_provider,
        administration_operation_factory=storage.safety.administration_operation,
    )
