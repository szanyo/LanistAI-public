# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate load."""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationFailure,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    LoadChecklistResult,
    RuntimeChecklistResult,
)
from lanista.reinforcengine.rl_model.runtime.events import RuntimeEventPublisher
from lanista.reinforcengine.rl_model.runtime.import_service import ImportResult
from lanista.reinforcengine.rl_model.runtime.requests import RuntimeInitializationRequest


class RuntimeModelLoader(Protocol):
    """Load and initialize one candidate runtime plugin."""

    def __call__(
        self,
        model_manifest: ModelManifest,
        runtime_request: RuntimeInitializationRequest,
        event_publisher: RuntimeEventPublisher | None = None,
        framework_uuid: str | None = None,
        selected_session_name: str | None = None,
        selected_checkpoint_name: str | None = None,
    ) -> Awaitable[ImportResult]:
        """Invoke this runtime model loader as a callable."""
        ...


@dataclass(frozen=True)
class LoadOutcome:
    """Carry the data required for load outcome."""

    model: BaseModelRuntime | None
    current_runtime_framework_uuid: str | None
    selected_session_name: str | None
    selected_checkpoint_name: str | None
    last_successful_model_name: str | None
    last_successful_framework_uuid: str | None
    last_issue: str | None = None
    model_manifest: ModelManifest | None = None
    failure: OperationFailure | None = None


@dataclass(frozen=True, slots=True)
class _ResolvedLoadRequest:
    """Hold normalized inputs while one candidate moves through the load pipeline."""

    manifest: ModelManifest
    configuration: object
    runtime_request: RuntimeInitializationRequest
    event_publisher: RuntimeEventPublisher | None
    framework_uuid: str
    session_name: str | None
    checkpoint_name: str | None


_ChecklistResult = LoadChecklistResult | RuntimeChecklistResult


class LoadService:
    """Orchestrates the model loading workflow around checklist validation and runtime import."""

    def __init__(
        self,
        *,
        model_load_checklist_builder: Callable[
            [ModelManifest],
            LoadChecklistResult,
        ],
        framework_runtime_checklist_builder: Callable[
            [ModelManifest, str],
            RuntimeChecklistResult,
        ],
        model_runtime_loader: RuntimeModelLoader,
        framework_normalizer: Callable[[str | None], str],
        progress_reporter: Callable[
            [RuntimeEventPublisher | None, str, str, float, float],
            None,
        ],
        checklist_summary_formatter: Callable[[str, int, int, bool, str | None], str],
        checklist_line_formatter: Callable[[str, str, bool, str], str],
        checklist_path_shortener: Callable[[str], str],
        logger: object | None = None,
    ) -> None:
        """Initialize the load service and its owned state."""
        self._model_load_checklist_builder = model_load_checklist_builder
        self._framework_runtime_checklist_builder = framework_runtime_checklist_builder
        self._model_runtime_loader = model_runtime_loader
        self._framework_normalizer = framework_normalizer
        self._progress_reporter = progress_reporter
        self._checklist_summary_formatter = checklist_summary_formatter
        self._checklist_line_formatter = checklist_line_formatter
        self._checklist_path_shortener = checklist_path_shortener
        self._logger = logger

    def _info(self, message: str, *, markup: bool = False) -> None:
        """Publish a info record through this load service."""
        if self._logger is not None and hasattr(self._logger, "info"):
            if markup:
                self._logger.info(message, extra={"markup": True})
            else:
                self._logger.info(message)

    def _error(self, message: str, *, markup: bool = False) -> None:
        """Publish a error record through this load service."""
        if self._logger is not None and hasattr(self._logger, "error"):
            if markup:
                self._logger.error(message, extra={"markup": True})
            else:
                self._logger.error(message)

    def _exception(self, exc: Exception) -> None:
        """Publish a exception record through this load service."""
        if self._logger is not None and hasattr(self._logger, "exception"):
            self._logger.exception(exc)

    def _warning(self, message: str) -> None:
        """Publish a warning record through this load service."""
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    def _report_progress(
        self,
        event_publisher: RuntimeEventPublisher | None,
        activity: str,
        step: str,
        progress: float,
        total: float = 100.0,
    ) -> None:
        """Report the progress owned by this load service."""
        self._progress_reporter(event_publisher, activity, step, progress, total)

    @staticmethod
    def _normalize_optional_text(value: str | None) -> str | None:
        """Normalize the optional text accepted by this load service."""
        text = str(value or "").strip()
        return text or None

    def _log_checklist(
        self,
        prefix: str,
        checklist: _ChecklistResult,
    ) -> None:
        """Log the checklist owned by this load service."""
        items = checklist.items
        self._info(
            self._checklist_summary_formatter(
                prefix,
                sum(1 for item in items if item.exists),
                len(items),
                checklist.valid,
                checklist.model_name,
            ),
            markup=True,
        )
        for item in items:
            if item.exists:
                continue
            self._error(
                self._checklist_line_formatter(
                    prefix,
                    item.label,
                    item.exists,
                    self._checklist_path_shortener(item.path),
                ),
                markup=True,
            )

    def _empty_outcome(
        self,
        requested_session_name: str | None,
        requested_checkpoint_name: str | None,
    ) -> LoadOutcome:
        """Return the empty outcome value owned by this load service."""
        return LoadOutcome(
            model=None,
            current_runtime_framework_uuid=None,
            selected_session_name=self._normalize_optional_text(
                requested_session_name
            ),
            selected_checkpoint_name=self._normalize_optional_text(
                requested_checkpoint_name
            ),
            last_successful_model_name=None,
            last_successful_framework_uuid=None,
            last_issue=None,
        )

    def _resolve_request(
        self,
        model_manifest: ModelManifest,
        *,
        runtime_request: RuntimeInitializationRequest,
        event_publisher: RuntimeEventPublisher | None,
        selected_framework_uuid: str | None,
        requested_session_name: str | None,
        requested_checkpoint_name: str | None,
    ) -> _ResolvedLoadRequest:
        """Resolve the request required by this load service."""
        configuration = model_manifest.get_configuration()
        session_name = self._normalize_optional_text(
            requested_session_name
        )
        checkpoint_name = self._normalize_optional_text(
            requested_checkpoint_name
        )
        framework_uuid = self._resolve_framework_uuid(
            model_manifest,
            configuration,
            selected_framework_uuid,
        )
        return _ResolvedLoadRequest(
            manifest=model_manifest,
            configuration=configuration,
            runtime_request=runtime_request,
            event_publisher=event_publisher,
            framework_uuid=framework_uuid,
            session_name=session_name,
            checkpoint_name=checkpoint_name,
        )

    def _resolve_framework_uuid(
        self,
        model_manifest: ModelManifest,
        configuration: object,
        selected_framework_uuid: str | None,
    ) -> str:
        """Resolve the framework uuid required by this load service."""
        bootstrap_provider = getattr(
            model_manifest,
            "get_bootstrap_framework_uuid",
            None,
        )
        manifest_framework_uuid = (
            bootstrap_provider() if callable(bootstrap_provider) else None
        )
        return (
            self._framework_normalizer(selected_framework_uuid)
            or self._framework_normalizer(manifest_framework_uuid)
            or self._framework_normalizer(
                getattr(configuration, "preferred_framework_uuid", None)
            )
            or "ray"
        )

    @staticmethod
    def _failed_outcome(
        request: _ResolvedLoadRequest,
        issue: str,
        *,
        failure: OperationFailure | None = None,
    ) -> LoadOutcome:
        """Return the failed outcome value owned by this load service."""
        return LoadOutcome(
            model=None,
            current_runtime_framework_uuid=None,
            selected_session_name=request.session_name,
            selected_checkpoint_name=request.checkpoint_name,
            last_successful_model_name=None,
            last_successful_framework_uuid=None,
            last_issue=issue,
            failure=failure,
        )

    def _validate_model_structure(
        self,
        request: _ResolvedLoadRequest,
    ) -> LoadOutcome | None:
        """Validate the model structure required by this load service."""
        checklist = self._model_load_checklist_builder(request.manifest)
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "Checking model structure",
            12.0,
        )
        self._log_checklist("load-check", checklist)
        if checklist.valid:
            return None
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "FAILED: structure checklist",
            100.0,
        )
        return self._failed_outcome(
            request,
            checklist.summary or "Model structure checklist failed.",
        )

    def _validate_runtime_configuration(
        self,
        request: _ResolvedLoadRequest,
    ) -> LoadOutcome | None:
        """Validate the runtime configuration required by this load service."""
        checklist = self._framework_runtime_checklist_builder(
            request.manifest,
            request.framework_uuid,
        )
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "Checking runtime configuration",
            18.0,
        )
        self._log_checklist("runtime-check", checklist)
        if checklist.valid:
            return None
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "FAILED: runtime configuration",
            100.0,
        )
        return self._failed_outcome(
            request,
            checklist.summary or "Runtime configuration checklist failed.",
        )

    def _log_import_start(self, request: _ResolvedLoadRequest) -> None:
        """Log the import start owned by this load service."""
        self._info(
            f"Model import started: "
            f"'{getattr(request.configuration, 'model_name', '')}' "
            f"{getattr(request.configuration, 'model_version', '')} "
            f"[{request.framework_uuid}]"
        )

    async def _import_candidate(
        self,
        request: _ResolvedLoadRequest,
    ) -> ImportResult | LoadOutcome:
        """Run the import candidate responsibility for this load service asynchronously."""
        self._log_import_start(request)
        try:
            result: object = await self._model_runtime_loader(
                request.manifest,
                request.runtime_request,
                event_publisher=request.event_publisher,
                framework_uuid=request.framework_uuid,
                selected_session_name=request.session_name,
                selected_checkpoint_name=request.checkpoint_name,
            )
        except ApplicationError as error:
            return self._application_error_outcome(request, error)
        except Exception as error:
            return self._unexpected_error_outcome(request, error)
        if not isinstance(result, ImportResult):
            return self._invalid_result_outcome(request, result)
        return result

    def _application_error_outcome(
        self,
        request: _ResolvedLoadRequest,
        error: ApplicationError,
    ) -> LoadOutcome:
        """Return the application error outcome value owned by this load service."""
        failure = failure_from_exception(error)
        self._error(
            f"Model import rejected for "
            f"'{getattr(request.configuration, 'model_name', '')}': "
            f"[{failure.code}] {failure.message}"
        )
        self._exception(error)
        self._report_progress(
            request.event_publisher,
            "Loading model",
            f"FAILED: {failure.code}",
            100.0,
        )
        return self._failed_outcome(
            request,
            failure.message,
            failure=failure,
        )

    def _unexpected_error_outcome(
        self,
        request: _ResolvedLoadRequest,
        error: Exception,
    ) -> LoadOutcome:
        """Return the unexpected error outcome value owned by this load service."""
        failure = failure_from_exception(error)
        self._error(
            f"Model import FAILED for "
            f"'{getattr(request.configuration, 'model_name', '')}': "
            f"[{type(error).__name__}] {error}"
        )
        self._exception(error)
        self._report_progress(
            request.event_publisher,
            "Loading model",
            f"FAILED: {type(error).__name__}",
            100.0,
        )
        return self._failed_outcome(
            request,
            failure.message,
            failure=failure,
        )

    def _invalid_result_outcome(
        self,
        request: _ResolvedLoadRequest,
        result: object,
    ) -> LoadOutcome:
        """Return the invalid result outcome value owned by this load service."""
        self._warning(
            f"Unexpected model runtime import result type: {type(result).__name__}. "
            "Treating it as a failed load."
        )
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "FAILED: invalid import result",
            100.0,
        )
        failure = failure_from_exception(
            ApplicationValidationError(
                "Model runtime import returned an invalid result."
            )
        )
        return self._failed_outcome(
            request,
            failure.message,
            failure=failure,
        )

    def _successful_outcome(
        self,
        request: _ResolvedLoadRequest,
        result: ImportResult,
    ) -> LoadOutcome:
        """Return the successful outcome value owned by this load service."""
        model_folder_path = str(
            getattr(request.configuration, "model_folder_path", "") or ""
        )
        last_successful_model_name = (
            os.path.basename(model_folder_path.rstrip(os.sep)) or None
        )
        framework_uuid = (
            self._framework_normalizer(
                result.framework_uuid or request.framework_uuid
            )
            or None
        )
        self._report_progress(
            request.event_publisher,
            "Loading model",
            "Completed",
            100.0,
        )
        self._info(
            f"Model import completed: "
            f"'{getattr(request.configuration, 'model_name', '')}'"
        )
        return LoadOutcome(
            model=result.model,
            current_runtime_framework_uuid=framework_uuid,
            selected_session_name=request.session_name,
            selected_checkpoint_name=request.checkpoint_name,
            last_successful_model_name=last_successful_model_name,
            last_successful_framework_uuid=framework_uuid,
            last_issue=None,
            model_manifest=result.model_manifest or request.manifest,
        )

    async def load(
        self,
        model_manifest: ModelManifest | None,
        *,
        runtime_request: RuntimeInitializationRequest,
        event_publisher: RuntimeEventPublisher | None = None,
        selected_framework_uuid: str | None = None,
        requested_session_name: str | None = None,
        requested_checkpoint_name: str | None = None,
    ) -> LoadOutcome:
        """Load and return the state owned by this load service."""
        if model_manifest is None:
            return self._empty_outcome(
                requested_session_name,
                requested_checkpoint_name,
            )

        request = self._resolve_request(
            model_manifest,
            runtime_request=runtime_request,
            event_publisher=event_publisher,
            selected_framework_uuid=selected_framework_uuid,
            requested_session_name=requested_session_name,
            requested_checkpoint_name=requested_checkpoint_name,
        )
        self._report_progress(event_publisher, "Loading model", "Preparing import", 5.0)
        failure = self._validate_model_structure(request)
        if failure is not None:
            return failure
        failure = self._validate_runtime_configuration(request)
        if failure is not None:
            return failure
        imported = await self._import_candidate(request)
        if isinstance(imported, LoadOutcome):
            return imported
        return self._successful_outcome(request, imported)
