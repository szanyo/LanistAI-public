# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate model creation."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Protocol
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationConflict,
    OperationResult,
    OperationTimeout,
    PersistenceError,
    ResourceNotFoundError,
    RetryPolicy,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.model_creation_contracts import (
    FreshModelCreationRequest,
    ModelCloneRequest,
    ModelCreationDraftResolution,
    ModelCreationOutcome,
)
from lanista.reinforcengine.rl_model.model_duplicator import ModelDuplicator
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLockTimeout
from lanista.reinforcengine.rl_model.storage.names import InvalidStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathViolation
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransactionError


class ModelCreationLogger(Protocol):
    """Minimal structured logger used at the application boundary."""

    def exception(self, message: str) -> object:
        """Publish a exception record through this model creation logger."""
        ...


class ModelCreationService:
    """Apply model creation naming rules against one configured storage root."""

    _BASE_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*_project$")
    _COMPOSITE_VERSION_PATTERN = re.compile(r"^v?([0-9A-Za-z-]+)\.([0-9]+)\.([0-9]+)$")
    _MANUAL_MAJOR_PATTERN = re.compile(r"^[0-9A-Za-z-]+$")

    def __init__(
        self,
        models_root_provider: Callable[[], str],
        *,
        administration_operation_factory: Callable[
            [], AbstractContextManager[object]
        ] = nullcontext,
        logger: ModelCreationLogger | None = None,
    ) -> None:
        """Initialize the model creation service and its owned state."""
        self._models_root_provider = models_root_provider
        self._administration_operation_factory = administration_operation_factory
        self._logger = logger

    def suggest_default_version(self, manual_major: str = "1") -> str:
        """Return the next version using the service's configured model root."""

        return self.suggest_default_new_model_version(
            self._models_root_provider(),
            manual_major=manual_major,
        )

    def resolve_draft(
        self,
        model_name_input: str,
        model_version_input: str,
        *,
        existing_version: str = "",
    ) -> ModelCreationDraftResolution:
        """Validate and normalize a draft within the configured model root."""

        return self.resolve_model_creation_draft(
            model_name_input,
            model_version_input,
            existing_version=existing_version,
            models_root=self._models_root_provider(),
        )

    def create_empty_model(
        self,
        request: FreshModelCreationRequest,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ModelCreationOutcome]:
        """Create and publish a fresh model or return a safe typed failure."""

        correlation_id = operation_id or uuid4().hex
        try:
            with self._administration_operation_factory():
                result = EmptyModelGenerator().generate(
                    model_name=request.model_name,
                    model_version=request.model_version,
                    model_description=request.model_description,
                    framework_uuid=request.framework_uuid,
                    models_root=self._models_root_provider(),
                )
        except Exception as error:
            return self._failure_result(
                error,
                operation="creation",
                operation_id=correlation_id,
            )
        return OperationResult.success(
            ModelCreationOutcome(
                target_model_name=result.target_model_name,
                target_model_folder_path=result.target_model_folder_path,
                auto_load=result.auto_load,
            ),
            operation_id=correlation_id,
        )

    def clone_model(
        self,
        request: ModelCloneRequest,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ModelCreationOutcome]:
        """Clone and publish selected resources or return a safe failure."""

        correlation_id = operation_id or uuid4().hex
        try:
            with self._administration_operation_factory():
                result = ModelDuplicator().duplicate(
                    source_model_folder_path=request.source_model_folder_path,
                    target_model_name=request.target_model_name,
                    target_model_version=request.target_model_version,
                    initial_session_description=request.initial_session_description,
                    create_initial_session=False,
                    selected_relative_paths=list(request.selected_relative_paths),
                    models_root=self._models_root_provider(),
                )
        except Exception as error:
            return self._failure_result(
                error,
                operation="clone",
                operation_id=correlation_id,
            )
        return OperationResult.success(
            ModelCreationOutcome(
                target_model_name=result.target_model_name,
                target_model_folder_path=result.target_model_folder_path,
                auto_load=result.auto_load,
            ),
            operation_id=correlation_id,
        )

    def _failure_result(
        self,
        error: Exception,
        *,
        operation: str,
        operation_id: str,
    ) -> OperationResult[ModelCreationOutcome]:
        """Return the failure result value owned by this model creation service."""
        classified = self._classify_failure(error, operation=operation)
        failure = failure_from_exception(
            classified,
            operation_id=operation_id,
            retry_policy=(RetryPolicy.SAFE if isinstance(classified, OperationTimeout) else None),
        )
        self._log_failure(
            error,
            operation=operation,
            operation_id=operation_id,
            failure_code=str(failure.code),
        )
        return OperationResult.failed(failure)

    def _log_failure(
        self,
        error: Exception,
        *,
        operation: str,
        operation_id: str,
        failure_code: str,
    ) -> None:
        """Log the failure owned by this model creation service."""
        if self._logger is None:
            return
        message = (
            "model_creation_failed "
            f"operation_id={operation_id} operation={operation} "
            f"failure_code={failure_code} exception_type={type(error).__name__} "
            f"error={error!r}"
        )
        try:
            self._logger.exception(message)
        except Exception as logging_error:
            # Logging is diagnostic; its failure must not alter the typed
            # operation result or leak an exception into the GUI adapter.
            logging.getLogger(__name__).error(
                "model_creation_failure_logging_failed "
                f"operation_id={operation_id} "
                f"exception_type={type(logging_error).__name__}",
                exc_info=logging_error,
            )

    @staticmethod
    def _classify_failure(error: Exception, *, operation: str) -> Exception:
        """Classify the failure owned by this model creation service."""
        if isinstance(error, ApplicationError):
            return error
        if isinstance(error, StorageMutationLockTimeout):
            return OperationTimeout("Model storage is busy. Try again.")
        if isinstance(error, StoragePathViolation):
            return ApplicationValidationError(
                "The model request references invalid managed storage."
            )
        if isinstance(error, InvalidStorageName):
            return ApplicationValidationError("The requested model name is invalid.")
        if isinstance(error, ValueError):
            return ApplicationValidationError(str(error))
        if isinstance(error, FileExistsError):
            return OperationConflict("A model with the requested storage name already exists.")
        if isinstance(error, FileNotFoundError | NotADirectoryError):
            return ResourceNotFoundError(
                "The source model or a required model template was not found."
            )
        if isinstance(error, OSError | StorageTransactionError | RuntimeError):
            return PersistenceError(f"The model {operation} operation could not be completed.")
        return error

    @staticmethod
    def build_model_display_name(model_name: str, model_version: str = "") -> str:
        """Format a created model through the shared naming policy."""

        return EmptyModelGenerator.build_model_display_name(
            model_name,
            model_version,
        )

    @classmethod
    def parse_composite_version(cls, version_text: str) -> tuple[str, int, int] | None:
        """Parse the composite version owned by this model creation service."""
        return EmptyModelGenerator.parse_composite_version(version_text)

    @classmethod
    def is_valid_manual_major(cls, version_text: str) -> bool:
        """Return whether valid manual major applies to this model creation service."""
        return EmptyModelGenerator.is_valid_manual_major(version_text)

    @staticmethod
    def build_composite_version(manual_major: str, model_number: int, session_number: int) -> str:
        """Build the composite version required by this model creation service."""
        return EmptyModelGenerator.build_composite_version(
            manual_major,
            model_number,
            session_number,
        )

    @staticmethod
    def _normalize_base_model_name(model_name: str) -> str:
        """Normalize the base model name accepted by this model creation service."""
        return EmptyModelGenerator.normalize_model_name(model_name)

    @classmethod
    def suggest_default_new_model_version(cls, models_root: str, manual_major: str = "1") -> str:
        """Return the suggest default new model version value owned by this model creation service.
        """
        return EmptyModelGenerator.suggest_default_version(
            models_root,
            manual_major=manual_major,
        )

    @classmethod
    def resolve_model_version_text(
        cls,
        model_version_input: str,
        *,
        existing_version: str = "",
        models_root: str = "",
    ) -> str:
        """Resolve the model version text required by this model creation service."""
        return EmptyModelGenerator.resolve_model_version(
            model_version_input,
            existing_version=existing_version,
            models_root=models_root,
            default_session_number=0,
        )

    @classmethod
    def normalize_model_name_text(cls, model_name_input: str, resolved_model_version: str) -> str:
        """Normalize the model name text accepted by this model creation service."""
        _ = resolved_model_version
        return cls._normalize_base_model_name(model_name_input)

    @classmethod
    def get_model_folder_name(cls, model_name: str) -> str:
        """Return the model folder name exposed by this model creation service."""
        return EmptyModelGenerator.get_model_folder_name(model_name)

    @classmethod
    def resolve_model_creation_draft(
        cls,
        model_name_input: str,
        model_version_input: str,
        *,
        existing_version: str = "",
        models_root: str = "",
    ) -> ModelCreationDraftResolution:
        """Resolve the model creation draft required by this model creation service."""
        models_root = str(models_root or "").strip()
        resolved_version = cls.resolve_model_version_text(
            model_version_input,
            existing_version=existing_version,
            models_root=models_root,
        )
        resolved_name = cls.normalize_model_name_text(model_name_input, resolved_version)
        target_folder_path = (
            os.path.join(models_root, cls.get_model_folder_name(resolved_name))
            if models_root and resolved_name
            else ""
        )

        if not resolved_name:
            return ModelCreationDraftResolution(
                False,
                "Model name is required.",
                "",
                resolved_version,
                target_folder_path,
            )
        if not cls._BASE_MODEL_NAME_PATTERN.fullmatch(resolved_name):
            return ModelCreationDraftResolution(
                False,
                "Model name must resolve to a unique '<name>_project' value.",
                resolved_name,
                resolved_version,
                target_folder_path,
            )
        if not str(model_version_input or "").strip():
            return ModelCreationDraftResolution(
                False,
                "Model version is required.",
                resolved_name,
                resolved_version,
                target_folder_path,
            )
        if not resolved_version:
            return ModelCreationDraftResolution(
                False,
                "Model version must resolve to exactly 3 components: v<major>.<model_no>.<session_no>",
                resolved_name,
                "",
                target_folder_path,
            )
        if target_folder_path and os.path.exists(target_folder_path):
            return ModelCreationDraftResolution(
                False,
                f"Model '{resolved_name}' already exists.",
                resolved_name,
                resolved_version,
                target_folder_path,
            )

        return ModelCreationDraftResolution(
            True, None, resolved_name, resolved_version, target_folder_path
        )
