# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read-side model, session, checkpoint, and lineage provider group."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.storage_safety import (
    StorageSafetyProviders,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_projection import (
    CheckpointCatalogProjector,
)
from lanista.reinforcengine.rl_model.checkpoints.catalog_reader import CheckpointCatalogReader
from lanista.reinforcengine.rl_model.checkpoints.catalog_repository import (
    CheckpointCatalogRepository,
)
from lanista.reinforcengine.rl_model.checkpoints.name_metadata import extract_checkpoint_steps
from lanista.reinforcengine.rl_model.model_discovery_service import ModelDiscoveryService
from lanista.reinforcengine.rl_model.sessions.lineage_service import (
    LineageService,
    ModelLineageCandidate,
)
from lanista.reinforcengine.rl_model.sessions.metadata_service import SessionMetadataService
from lanista.reinforcengine.rl_model.sessions.repository_service import SessionRepositoryService
from lanista.reinforcengine.rl_model.sessions.summary_service import SessionSummaryService


@dataclass(frozen=True, slots=True)
class CatalogProviders:
    """Own the shared read model consumed by later provider groups."""

    models: ModelDiscoveryService
    session_summaries: SessionSummaryService
    sessions: SessionRepositoryService
    session_metadata: SessionMetadataService
    checkpoints: CheckpointCatalogReader
    lineage: LineageService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
        storage: StorageSafetyProviders,
    ) -> CatalogProviders:
        """Construct catalog services in dependency order without root callbacks."""

        session_summaries = SessionSummaryService(
            context.active_session_manager,
            logger=context.logger,
        )
        sessions = _create_session_repository(context, session_summaries)
        session_metadata = _create_session_metadata(
            context,
            storage,
            sessions,
        )
        models = ModelDiscoveryService(
            models_root_provider=context.models_root_provider,
            logger_name_provider=lambda: context.logger.name,
            logger_level_provider=context.log_level_provider,
            manifest_cls=context.manifest_cls,
        )
        checkpoints = _create_checkpoint_reader(context, sessions, session_metadata)
        lineage = LineageService(
            model_candidates_provider=lambda: _discover_lineage_candidates(models),
            session_summaries_provider=(sessions.list_session_summaries_for_current_model),
            session_reference_display_provider=(
                sessions.get_session_reference_display_from_summary
            ),
            logger=context.logger,
        )
        return cls(
            models=models,
            session_summaries=session_summaries,
            sessions=sessions,
            session_metadata=session_metadata,
            checkpoints=checkpoints,
            lineage=lineage,
        )

    def load_session_configuration(self, session_name: str) -> Configuration | None:
        """Return a session configuration only when the repository read succeeded."""

        result = self.sessions.load_session_configuration(session_name)
        return result.value if result.succeeded else None

    def session_has_training_checkpoint_source(
        self,
        session_name: str | None,
    ) -> bool:
        """Report whether a session resumes from an explicit checkpoint source."""

        if not session_name:
            return False
        configuration = self.load_session_configuration(session_name)
        if configuration is None:
            return False
        source_session = str(
            getattr(configuration, "load_checkpoint_from_session", "") or ""
        ).strip()
        source_checkpoint = str(
            getattr(configuration, "load_checkpoint_name", "") or ""
        ).strip()
        return bool(source_session and source_checkpoint)


def _discover_lineage_candidates(
    models: ModelDiscoveryService,
) -> tuple[Sequence[ModelLineageCandidate], Sequence[object]]:
    """Return the discover lineage candidates value owned by this catalog module."""
    valid, invalid = models.discover_model_candidates(None)
    return cast(Sequence[ModelLineageCandidate], valid), invalid


def _create_session_repository(
    context: ApplicationProviderContext,
    summaries: SessionSummaryService,
) -> SessionRepositoryService:
    """Create the session repository owned by this catalog module."""
    return SessionRepositoryService(
        sessions_root_provider=context.sessions_root,
        session_manager_provider=context.active_session_manager,
        session_summary_service_provider=lambda: summaries,
        framework_normalizer=context.normalize_framework_uuid,
        logger=context.logger,
    )


def _create_session_metadata(
    context: ApplicationProviderContext,
    storage: StorageSafetyProviders,
    sessions: SessionRepositoryService,
) -> SessionMetadataService:
    """Create the session metadata owned by this catalog module."""
    def load_configuration(session_name: str) -> Configuration | None:
        """Load the configuration owned by this catalog module."""
        result = sessions.load_session_configuration(session_name)
        return result.value if result.succeeded else None

    return SessionMetadataService(
        active_configuration_provider=context.active_configuration,
        session_configuration_loader=load_configuration,
        sessions_provider=lambda: sessions.list_sessions_for_current_model(None),
        session_manager_provider=context.active_session_manager,
        model_manifest_saver=context.save_minimal_model_manifest,
        models_root_provider=context.models_root_provider,
        administration_operation_factory=storage.safety.administration_operation,
        logger=context.logger,
    )


def _create_checkpoint_reader(
    context: ApplicationProviderContext,
    sessions: SessionRepositoryService,
    metadata: SessionMetadataService,
) -> CheckpointCatalogReader:
    """Create the checkpoint reader owned by this catalog module."""
    repository = CheckpointCatalogRepository(
        sessions_root_provider=context.sessions_root,
        session_framework_provider=sessions.get_session_framework_uuid,
        framework_normalizer=context.normalize_framework_uuid,
        step_extractor=extract_checkpoint_steps,
    )
    return CheckpointCatalogReader(
        repository=repository,
        projector=CheckpointCatalogProjector(
            step_extractor=extract_checkpoint_steps,
        ),
        session_reference_display_provider=metadata.get_session_reference_display,
        session_framework_provider=sessions.get_session_framework_uuid,
    )
