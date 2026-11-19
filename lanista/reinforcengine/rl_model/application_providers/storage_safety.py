# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Storage and safety provider group at the base of the application DAG."""

from __future__ import annotations

from dataclasses import dataclass

from lanista.reinforcengine.rl_model.administration_command_policy import (
    AdministrationCommandPolicy,
)
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationSafetyService,
)
from lanista.reinforcengine.rl_model.application_providers.contracts import (
    ApplicationProviderContext,
)
from lanista.reinforcengine.rl_model.model_creation_service import ModelCreationService
from lanista.reinforcengine.rl_model.storage.layout import ensure_tensorboard_structure
from lanista.reinforcengine.rl_model.storage.layout_service import StorageLayoutService
from lanista.reinforcengine.rl_model.storage.recovery_service import StorageRecoveryService


@dataclass(frozen=True, slots=True)
class StorageSafetyProviders:
    """Own storage layout, recovery, and the shared administration gate."""

    safety: AdministrationSafetyService
    command_policy: AdministrationCommandPolicy
    recovery: StorageRecoveryService
    layout: StorageLayoutService
    model_creation: ModelCreationService

    @classmethod
    def create(
        cls,
        context: ApplicationProviderContext,
    ) -> StorageSafetyProviders:
        """Construct the dependency-free base provider group exactly once."""

        safety = AdministrationSafetyService(
            native_gui_mutation_guard=context.native_gui_mutation_guard,
        )
        command_policy = AdministrationCommandPolicy(
            active_configuration_provider=context.active_configuration,
            administration_operation_factory=safety.administration_operation,
            degraded_state_reporter=safety.mark_degraded,
            logger=context.logger,
        )
        recovery = StorageRecoveryService(context.models_root_provider)
        layout = StorageLayoutService(
            tensorboard_structure_ensurer=ensure_tensorboard_structure,
            models_root_provider=context.models_root_provider,
            logger=context.logger,
            degraded_state_reporter=safety.mark_degraded,
        )
        model_creation = ModelCreationService(
            models_root_provider=context.models_root_provider,
            administration_operation_factory=safety.administration_operation,
            logger=context.logger,
        )
        return cls(
            safety=safety,
            command_policy=command_policy,
            recovery=recovery,
            layout=layout,
            model_creation=model_creation,
        )
