# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate load finalization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.load_service import LoadOutcome


class PersistLastSuccessfulState(Protocol):
    """Persist the complete association needed to restore one runtime."""

    def __call__(
        self,
        *,
        model_name: str | None,
        framework_uuid: str | None,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> None:
        """Invoke this persist last successful state as a callable."""
        ...


@dataclass(frozen=True)
class LoadFinalizationResult:
    """Report the result of load finalization."""

    model: BaseModelRuntime | None
    current_runtime_framework_uuid: str | None
    selected_session_name: str | None
    selected_checkpoint_name: str | None
    last_successful_model_name: str | None
    last_successful_framework_uuid: str | None


class LoadFinalizationService:
    """Apply and persist the complete post-load runtime association."""

    def __init__(
        self,
        *,
        normalize_selection: Callable[[str | None, str | None], object],
        normalize_framework_uuid: Callable[[str | None], str],
        manifest_folder_name_provider: Callable[[ModelManifest | None], str | None],
        persist_last_successful_state: PersistLastSuccessfulState,
        logger: object | None = None,
    ) -> None:
        """Initialize the load finalization service and its owned state."""
        self._normalize_selection = normalize_selection
        self._normalize_framework_uuid = normalize_framework_uuid
        self._manifest_folder_name_provider = manifest_folder_name_provider
        self._persist_last_successful_state = persist_last_successful_state
        self._logger = logger

    def _warning(self, message: str) -> None:
        """Publish a warning record through this load finalization service."""
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    def finalize_successful_load(
        self,
        *,
        model_manifest: ModelManifest | None,
        load_outcome: LoadOutcome,
        current_runtime_framework_uuid: str | None,
    ) -> LoadFinalizationResult:
        """Build final state and persist its restart association."""

        result = self.build_successful_load(
            model_manifest=model_manifest,
            load_outcome=load_outcome,
            current_runtime_framework_uuid=current_runtime_framework_uuid,
        )
        self.persist_successful_load(result)
        return result

    def build_successful_load(
        self,
        *,
        model_manifest: ModelManifest | None,
        load_outcome: LoadOutcome,
        current_runtime_framework_uuid: str | None,
    ) -> LoadFinalizationResult:
        """Resolve the complete slot state without mutating storage."""

        selection = self._normalize_selection(
            load_outcome.selected_session_name,
            load_outcome.selected_checkpoint_name,
        )
        resolved_current_runtime_framework_uuid = (
            load_outcome.current_runtime_framework_uuid
            or current_runtime_framework_uuid
        )
        last_successful_model_name = (
            load_outcome.last_successful_model_name
            or self._manifest_folder_name_provider(model_manifest)
        )
        last_successful_framework_uuid = (
            load_outcome.last_successful_framework_uuid
            or self._normalize_framework_uuid(resolved_current_runtime_framework_uuid)
            or None
        )

        return LoadFinalizationResult(
            model=load_outcome.model,
            current_runtime_framework_uuid=self._normalize_framework_uuid(
                resolved_current_runtime_framework_uuid
            )
            or None,
            selected_session_name=getattr(selection, "session_name", None),
            selected_checkpoint_name=getattr(selection, "checkpoint_name", None),
            last_successful_model_name=last_successful_model_name,
            last_successful_framework_uuid=last_successful_framework_uuid,
        )

    def persist_successful_load(
        self,
        result: LoadFinalizationResult,
    ) -> None:
        """Persist the restart association or let the slot transaction roll back."""

        try:
            self._persist_last_successful_state(
                model_name=result.last_successful_model_name,
                framework_uuid=result.last_successful_framework_uuid,
                session_name=result.selected_session_name,
                checkpoint_name=result.selected_checkpoint_name,
            )
        except Exception as exc:
            self._warning(f"Failed to persist runtime state: {exc}")
            raise
