# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Presentation and local inspection provider group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from lanista.reinforcengine.rl_model.application_providers.catalog import CatalogProviders
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.application_providers.runtime import RuntimeCoreProviders
from lanista.reinforcengine.rl_model.inspect_toggle_service import InspectToggleService
from lanista.reinforcengine.rl_model.inspection_target_service import (
    InspectionTargetService,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.presentation_context_service import (
    PresentationContextService,
)
from lanista.reinforcengine.rl_model.runtime.selection_service import (
    TensorboardRestartContext,
)


@dataclass(frozen=True, slots=True)
class PresentationProviders:
    """Own GUI-neutral presentation projections and TensorBoard inspection."""

    toggle: InspectToggleService
    context: PresentationContextService
    targets: InspectionTargetService

    @classmethod
    def create(
        cls,
        provider_context: ApplicationProviderContext,
        catalog: CatalogProviders,
        runtime: RuntimeCoreProviders,
    ) -> PresentationProviders:
        """Build presentation services from already-owned domain services."""

        presentation = _create_presentation_context(
            provider_context,
            catalog,
            runtime,
        )

        def restart_context(
            mode: RLModelRunningModes,
            selected_session: str | None,
        ) -> TensorboardRestartContext | None:
            """Restart the context owned by this presentation providers."""
            slot = provider_context.slot_store.snapshot()
            return runtime.selection.build_tensorboard_restart_context(
                mode=mode,
                selected_session=selected_session,
                framework_uuid=None,
                configuration=provider_context.active_configuration(),
                current_runtime_framework_uuid=slot.framework_uuid,
            )

        toggle = InspectToggleService(
            has_current_model_provider=(
                lambda: provider_context.slot_store.snapshot().model is not None
            ),
            restart_context_builder=restart_context,
        )
        return cls(
            toggle=toggle,
            context=presentation,
            targets=InspectionTargetService(),
        )


def _create_presentation_context(
    context: ApplicationProviderContext,
    catalog: CatalogProviders,
    runtime: RuntimeCoreProviders,
) -> PresentationContextService:
    """Create the presentation context owned by this presentation module."""
    metadata = catalog.session_metadata

    def lineage_index() -> tuple[dict[str, object], dict[str | None, list[str]]]:
        """Perform the lineage index step within the enclosing create presentation context
        workflow.
        """
        nodes, children = catalog.lineage.build_model_lineage_index()
        return cast(dict[str, object], nodes), children

    return PresentationContextService(
        active_configuration_provider=context.active_configuration,
        current_framework_provider=lambda: runtime.current_framework_uuid(context),
        session_configuration_loader=catalog.load_session_configuration,
        session_reference_display_provider=metadata.get_session_reference_display,
        session_identity_uuid_provider=metadata.get_session_identity_uuid,
        session_display_name_without_uuid_provider=(
            metadata.get_session_display_name_without_uuid
        ),
        session_parent_uuid_provider=metadata.get_session_parent_uuid,
        session_child_uuids_provider=metadata.get_session_child_uuids,
        model_lineage_index_provider=lineage_index,
        model_candidates_provider=lambda: catalog.models.discover_model_candidates(None),
    )
