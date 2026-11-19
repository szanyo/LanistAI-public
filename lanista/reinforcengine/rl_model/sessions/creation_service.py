# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate training-session creation."""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

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
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationCompletionGuard,
)
from lanista.reinforcengine.rl_model.metadata.configuration_schema import (
    VALID_FRAMEWORK_UUIDS,
    ConfigurationSchema,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.name_formatting_service import NameFormattingService
from lanista.reinforcengine.rl_model.runtime.selection_service import RuntimeSelectionUpdate
from lanista.reinforcengine.rl_model.sessions.creation_repository import (
    PreparedSessionCreationMetadata,
    SessionCreationMetadataRepository,
    SessionCreationMetadataRequest,
)
from lanista.reinforcengine.rl_model.sessions.creation_unit_of_work import (
    SessionCreationRecoveryError,
    SessionCreationTransactionFactory,
    SessionCreationUnitOfWork,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import (
    StorageMutationLock,
    StorageMutationLockError,
    StorageMutationLockTimeout,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    InvalidStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathViolation
from lanista.reinforcengine.rl_model.storage.transaction import (
    StorageDurabilityError,
    StorageTransaction,
    StorageTransactionError,
)


class _Logger(Protocol):
    """Define the typed boundary for logger."""

    def warning(self, message: str) -> None:
        """Publish a warning record through this logger."""
        ...

    def info(self, message: str) -> None:
        """Publish a info record through this logger."""
        ...


class SessionCreationConfiguration(Protocol):
    """Minimal mutable active-model projection required by session creation."""

    model_uuid: str
    model_folder_path: str
    active_runtime_framework_uuid: str | None
    preferred_framework_uuid: str | None


@dataclass(frozen=True, slots=True)
class _SessionCreationOwner:
    """Carry the data required for session creation owner."""

    model_uuid: str
    model_root: Path
    models_root: Path


@dataclass(frozen=True, slots=True)
class _PreparedSessionCreation:
    """Carry the data required for prepared session creation."""

    configuration: SessionCreationConfiguration
    metadata: PreparedSessionCreationMetadata


@dataclass(frozen=True, slots=True)
class _SessionCreationCommand:
    """Represent the session creation workflow operation."""

    strategy: str
    parent_session_uuid: str | None
    parent_checkpoint_name: str | None
    framework_uuid: str | None
    session_display_name: str | None
    session_description: str | None
    selected_session_name: str | None
    expected_model_uuid: str | None


@dataclass(frozen=True, slots=True)
class _AttributeSnapshot:
    """Capture an immutable snapshot of attribute."""

    present: bool
    value: object


@dataclass(frozen=True, slots=True)
class _ProjectionSnapshot:
    """Capture an immutable snapshot of projection."""

    attributes: dict[str, _AttributeSnapshot]
    runtime_selection: RuntimeSelectionUpdate


@dataclass(frozen=True, slots=True)
class SessionCreationOutcome:
    """Describe a durable session creation and its runtime projection."""

    session_name: str
    session_uuid: str
    framework_uuid: str
    parent_session_name: str | None
    committed: bool
    degraded: bool
    reload_required: bool
    storage_recovery_required: bool
    projection_synchronized: bool


class SessionCreationService:
    """Coordinate session creation operations."""

    @dataclass
    class SessionCreationDraft:
        """Carry the data required for session creation draft."""

        session_uuid: str
        session_label: str
        session_name: str
        strategy_key: str
        selected_framework_uuid: str
        resolved_parent_session_uuid: str | None
        load_checkpoint_from_session: str | None
        load_checkpoint_name: str | None
        base_step_offset: int
        created_at: str

    @staticmethod
    def sanitize_session_name(value: str, limit: int = 48) -> str:
        """Return the sanitize session name value owned by this session creation service."""
        return NameFormattingService.sanitize_name(value, limit=limit, fallback="session")

    @classmethod
    def build_unique_session_storage_name(cls, sessions_root: str, display_name: str) -> str:
        """Build the unique session storage name required by this session creation service."""
        return NameFormattingService.build_unique_storage_name(sessions_root, display_name, limit=48, fallback="session")

    @staticmethod
    def build_session_timestamp_label(created_at_raw: str) -> str:
        """Build the session timestamp label required by this session creation service."""
        return NameFormattingService.build_timestamp_label(
            created_at_raw,
            invalid_strategy="current",
            fallback=datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S"),
        )

    @classmethod
    def resolve_session_creation_draft(
        cls,
        *,
        sessions_root: str,
        created_at_raw: str,
        strategy: str = "fresh",
        parent_session_uuid: str | None = None,
        parent_checkpoint_name: str | None = None,
        framework_uuid: str | None = None,
        active_runtime_framework_uuid: str | None = None,
        preferred_framework_uuid: str | None = None,
        session_display_name: str | None = None,
        indexed_sessions: Sequence[str] = (),
        selected_session_name: str | None = None,
        available_checkpoints_resolver: Callable[[str, str], Sequence[str]] | None = None,
    ) -> SessionCreationService.SessionCreationDraft:
        """Resolve the session creation draft required by this session creation service."""
        session_uuid = str(uuid.uuid4())
        session_label = (
            str(session_display_name or "").strip()
            or cls.build_session_timestamp_label(created_at_raw)
        )
        session_name = cls.build_unique_session_storage_name(sessions_root, session_label)
        strategy_key = str(strategy or "fresh").strip().lower()
        if strategy_key not in {"fresh", "fork"}:
            raise OperationConflict("Unsupported session creation strategy.")
        selected_framework_uuid = cls._resolve_framework_uuid(
            framework_uuid,
            active_runtime_framework_uuid,
            preferred_framework_uuid,
        )
        resolved_parent_session_uuid = cls._resolve_parent_session(
            strategy_key=strategy_key,
            requested_parent=parent_session_uuid,
            selected_parent=selected_session_name,
            indexed_sessions=indexed_sessions,
        )
        load_checkpoint_name = cls._resolve_parent_checkpoint(
            parent_session=resolved_parent_session_uuid,
            requested_checkpoint=parent_checkpoint_name,
            framework_uuid=selected_framework_uuid,
            available_checkpoints_resolver=available_checkpoints_resolver,
        )
        load_checkpoint_from_session = (
            resolved_parent_session_uuid if load_checkpoint_name else None
        )

        return cls.SessionCreationDraft(
            session_uuid=session_uuid,
            session_label=session_label,
            session_name=session_name,
            strategy_key=strategy_key,
            selected_framework_uuid=selected_framework_uuid,
            resolved_parent_session_uuid=resolved_parent_session_uuid,
            load_checkpoint_from_session=load_checkpoint_from_session,
            load_checkpoint_name=load_checkpoint_name,
            base_step_offset=0,
            created_at=created_at_raw,
        )

    @staticmethod
    def _resolve_framework_uuid(
        requested: str | None,
        active: str | None,
        preferred: str | None,
    ) -> str:
        """Resolve the framework uuid required by this session creation service."""
        requested_framework = str(requested or "").strip().lower()
        if requested_framework and requested_framework not in VALID_FRAMEWORK_UUIDS:
            raise OperationConflict("Unsupported training framework.")
        return requested_framework or next(
            (
                candidate
                for value in (active, preferred, "ray")
                if (candidate := str(value or "").strip().lower())
                in VALID_FRAMEWORK_UUIDS
            ),
            "ray",
        )

    @staticmethod
    def _resolve_parent_session(
        *,
        strategy_key: str,
        requested_parent: str | None,
        selected_parent: str | None,
        indexed_sessions: Sequence[str],
    ) -> str | None:
        """Resolve the parent session required by this session creation service."""
        if strategy_key == "fresh":
            return None
        candidate = str(requested_parent or selected_parent or "").strip()
        if not candidate:
            raise OperationConflict(
                "Fork session creation requires an existing parent session."
            )
        try:
            candidate = str(SessionStorageName.parse(candidate))
        except ValueError as exc:
            raise OperationConflict(
                "The selected parent session is invalid."
            ) from exc
        indexed = {
            str(value).strip()
            for value in indexed_sessions
            if str(value).strip()
        }
        if candidate not in indexed:
            raise OperationConflict(
                "The selected parent session is no longer available."
            )
        return candidate

    @staticmethod
    def _resolve_parent_checkpoint(
        *,
        parent_session: str | None,
        requested_checkpoint: str | None,
        framework_uuid: str,
        available_checkpoints_resolver: (
            Callable[[str, str], Sequence[str]] | None
        ),
    ) -> str | None:
        """Resolve the parent checkpoint required by this session creation service."""
        if parent_session is None:
            return None
        available = list(
            available_checkpoints_resolver(parent_session, framework_uuid)
            if available_checkpoints_resolver is not None
            else []
        )
        requested = str(requested_checkpoint or "").strip()
        if not requested:
            return available[0] if available else None
        try:
            requested = str(CheckpointStorageName.parse(requested))
        except ValueError as exc:
            raise OperationConflict(
                "The selected parent checkpoint is invalid."
            ) from exc
        if requested not in available:
            raise OperationConflict(
                "The selected parent checkpoint is no longer available."
            )
        return requested

    @staticmethod
    def build_session_configuration_payload(
        draft: SessionCreationService.SessionCreationDraft,
        session_description: str = "",
    ) -> dict[str, object]:
        """Build the session configuration payload required by this session creation service."""
        return {
            "name": draft.session_label,
            "uuid": draft.session_uuid,
            "storage_name": draft.session_name,
            "description": str(session_description or "").strip(),
            "tags": [],
            "framework_uuid": draft.selected_framework_uuid,
            "created_at": draft.created_at,
            "modified_at": draft.created_at,
            "parent_session_uuid": draft.resolved_parent_session_uuid,
            "child_session_uuids": [],
            "load_checkpoint_from_session": draft.load_checkpoint_from_session,
            "load_checkpoint_name": draft.load_checkpoint_name,
            "base_step_offset": draft.base_step_offset,
        }

    def __init__(
        self,
        *,
        active_configuration_provider: (
            Callable[[], SessionCreationConfiguration | None]
        ),
        sessions_root_provider: Callable[[], str],
        session_config_path_provider: Callable[[str], str],
        models_root_provider: Callable[[], str],
        runtime_selection_state_provider: Callable[[], RuntimeSelectionUpdate],
        runtime_selection_state_committer: Callable[[RuntimeSelectionUpdate], None],
        runtime_slot_operation_factory: Callable[
            [], AbstractContextManager[object]
        ],
        administration_operation_factory: Callable[
            [], AbstractContextManager[AdministrationCompletionGuard]
        ],
        logger: _Logger | None = None,
        transaction_factory: SessionCreationTransactionFactory = StorageTransaction,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        lock_timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the session creation service and its owned state."""
        self._active_configuration_provider = active_configuration_provider
        self._sessions_root_provider = sessions_root_provider
        self._session_config_path_provider = session_config_path_provider
        self._models_root_provider = models_root_provider
        self._runtime_selection_state_provider = runtime_selection_state_provider
        self._runtime_selection_state_committer = runtime_selection_state_committer
        self._runtime_slot_operation_factory = runtime_slot_operation_factory
        self._administration_operation_factory = administration_operation_factory
        self._logger = logger
        self._transaction_factory = transaction_factory
        self._lock_factory = lock_factory
        self._lock_timeout_seconds = lock_timeout_seconds

    def _warning(self, message: str) -> None:
        """Publish a warning record through this session creation service."""
        if self._logger is None or not hasattr(self._logger, "warning"):
            return
        try:
            self._logger.warning(message)
        except Exception:
            return

    def _info(self, message: str) -> None:
        """Publish a info record through this session creation service."""
        if self._logger is None or not hasattr(self._logger, "info"):
            return
        try:
            self._logger.info(message)
        except Exception:
            return

    def _exception(self, error: BaseException) -> None:
        """Publish a exception record through this session creation service."""
        if self._logger is None or not hasattr(self._logger, "exception"):
            return
        try:
            self._logger.exception(
                error,
                exc_info=(type(error), error, error.__traceback__),
            )
        except (TypeError, AttributeError):
            try:
                self._logger.exception(error)
            except Exception:
                return
        except Exception:
            return

    def create_new_session_for_current_model(
        self,
        *,
        strategy: str = "fresh",
        parent_session_uuid: str | None = None,
        parent_checkpoint_name: str | None = None,
        framework_uuid: str | None = None,
        session_display_name: str | None = None,
        session_description: str | None = None,
        selected_session_name: str | None = None,
        expected_model_uuid: str | None = None,
        operation_id: str | None = None,
    ) -> OperationResult[SessionCreationOutcome]:
        """Create one session and return a correlated, transport-safe result."""

        correlation_id = operation_id or uuid.uuid4().hex
        command = _SessionCreationCommand(
            strategy=strategy,
            parent_session_uuid=parent_session_uuid,
            parent_checkpoint_name=parent_checkpoint_name,
            framework_uuid=framework_uuid,
            session_display_name=session_display_name,
            session_description=session_description,
            selected_session_name=selected_session_name,
            expected_model_uuid=expected_model_uuid,
        )
        try:
            outcome = self._execute_creation(
                command,
                operation_id=correlation_id,
            )
        except Exception as error:
            return self._creation_failure_result(
                error,
                operation_id=correlation_id,
            )
        return OperationResult.success(outcome, operation_id=correlation_id)

    def _execute_creation(
        self,
        command: _SessionCreationCommand,
        *,
        operation_id: str,
    ) -> SessionCreationOutcome:
        """Execute the creation owned by this session creation service."""
        committed: SessionCreationOutcome | None = None
        completion_guard: AdministrationCompletionGuard | None = None
        try:
            with self._administration_operation_factory() as context_value:
                completion_guard = self._require_completion_guard(context_value)
                try:
                    with self._runtime_slot_operation_factory():
                        committed = self._create_locked(
                            command,
                            operation_id=operation_id,
                        )
                except BaseException as operation_error:
                    if committed is not None:
                        completion_guard.latch_recovery_required()
                    else:
                        self._arm_failed_creation_completion(
                            completion_guard,
                            operation_error,
                        )
                    raise
                try:
                    completion_guard.arm_committed_completion(
                        reload_required=committed.reload_required,
                        storage_recovery_required=(
                            committed.storage_recovery_required
                        ),
                    )
                except BaseException:
                    completion_guard.latch_recovery_required()
                    raise
            if committed is None:
                raise OperationConflict(
                    "Session creation did not produce a durable result."
                )
            completion_guard.confirm_successful_exit()
            return committed
        except BaseException as error:
            self._fail_creation_completion(completion_guard)
            if committed is None or not isinstance(error, Exception):
                raise
            return self._recover_creation_context_exit(
                committed,
                error,
                operation_id=operation_id,
            )

    @staticmethod
    def _require_completion_guard(
        context_value: object,
    ) -> AdministrationCompletionGuard:
        """Return the required completion guard for this session creation service."""
        if not isinstance(context_value, AdministrationCompletionGuard):
            raise OperationConflict(
                "Session creation requires a completion-aware safety gate."
            )
        return context_value

    @staticmethod
    def _arm_failed_creation_completion(
        completion_guard: AdministrationCompletionGuard,
        error: BaseException,
    ) -> None:
        """Arm the failed creation completion owned by this session creation
service."""
        if isinstance(error, SessionCreationRecoveryError | StorageDurabilityError):
            completion_guard.latch_recovery_required()
        elif not isinstance(error, Exception):
            completion_guard.arm_committed_completion(
                reload_required=True,
                storage_recovery_required=True,
            )

    def _fail_creation_completion(
        self,
        completion_guard: AdministrationCompletionGuard | None,
    ) -> None:
        """Fail the creation completion owned by this session creation
service."""
        if completion_guard is None:
            return
        try:
            completion_guard.fail_exit()
        except Exception as error:
            self._warning(
                "session_creation_completion_fail_closed "
                f"exception_type={type(error).__name__}"
            )
            self._exception(error)

    def _create_locked(
        self,
        command: _SessionCreationCommand,
        *,
        operation_id: str,
    ) -> SessionCreationOutcome:
        """Create the locked owned by this session creation service."""
        configuration, owner = self._capture_owner(command.expected_model_uuid)
        unit_of_work = SessionCreationUnitOfWork(
            owner.model_root,
            models_root=owner.models_root,
            transaction_factory=self._transaction_factory,
            lock_factory=self._lock_factory,
            lock_timeout_seconds=self._lock_timeout_seconds,
        )
        owner = _SessionCreationOwner(
            model_uuid=owner.model_uuid,
            model_root=unit_of_work.model_root,
            models_root=owner.models_root,
        )
        repository = SessionCreationMetadataRepository(owner.model_root)
        projection = self._capture_projection(configuration)
        locked_selected_session = self._resolve_locked_session_fallback(
            strategy=command.strategy,
            explicit_parent=command.parent_session_uuid,
            requested_fallback=command.selected_session_name,
            locked_fallback=projection.runtime_selection.session_name,
        )
        outcome = unit_of_work.execute(
            lambda: self._prepare_under_storage_lock(
                owner=owner,
                repository=repository,
                strategy=command.strategy,
                parent_session_uuid=command.parent_session_uuid,
                parent_checkpoint_name=command.parent_checkpoint_name,
                framework_uuid=command.framework_uuid,
                session_display_name=command.session_display_name,
                session_description=command.session_description,
                selected_session_name=locked_selected_session,
            ),
            lambda transaction, prepared: self._commit_storage(
                repository,
                transaction,
                prepared,
            ),
        )

        projection_synchronized = self._publish_projection(
            outcome.result,
            projection,
            operation_id=operation_id,
        )
        storage_recovery_required = outcome.post_commit_error is not None
        reload_required = not projection_synchronized or storage_recovery_required
        if outcome.post_commit_error is not None:
            self._log_creation_post_commit_error(
                outcome.post_commit_error,
                operation_id=operation_id,
            )
        request = outcome.result.metadata.request
        session_name = str(request.storage_name)
        self._info(f"Created new session for current model: {session_name}")
        return SessionCreationOutcome(
            session_name=session_name,
            session_uuid=request.identity_uuid,
            framework_uuid=request.framework_uuid,
            parent_session_name=(
                str(request.parent_session)
                if request.parent_session is not None
                else None
            ),
            committed=True,
            degraded=reload_required or storage_recovery_required,
            reload_required=reload_required,
            storage_recovery_required=storage_recovery_required,
            projection_synchronized=projection_synchronized,
        )

    @staticmethod
    def _resolve_locked_session_fallback(
        *,
        strategy: str,
        explicit_parent: str | None,
        requested_fallback: str | None,
        locked_fallback: str | None,
    ) -> str | None:
        """Resolve the locked session fallback required by this session creation service."""
        if str(strategy or "fresh").strip().lower() != "fork" or explicit_parent:
            return requested_fallback
        requested = str(requested_fallback or "").strip()
        locked = str(locked_fallback or "").strip()
        if requested and requested != locked:
            raise OperationConflict(
                "The selected parent session changed before creation started."
            )
        return locked or None

    def _capture_owner(
        self,
        expected_model_uuid: str | None,
    ) -> tuple[SessionCreationConfiguration, _SessionCreationOwner]:
        """Capture the owner required by this session creation service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            raise OperationConflict("Session creation requires an active model.")
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        expected_uuid = str(expected_model_uuid or "").strip()
        if not active_uuid or (expected_uuid and active_uuid != expected_uuid):
            raise OperationConflict(
                "The active model changed before session creation was committed."
            )
        model_root = Path(
            str(getattr(configuration, "model_folder_path", "") or "")
        ).expanduser()
        models_root = Path(self._models_root_provider()).expanduser()
        return configuration, _SessionCreationOwner(
            model_uuid=active_uuid,
            model_root=model_root,
            models_root=models_root,
        )

    def _prepare_under_storage_lock(
        self,
        *,
        owner: _SessionCreationOwner,
        repository: SessionCreationMetadataRepository,
        strategy: str,
        parent_session_uuid: str | None,
        parent_checkpoint_name: str | None,
        framework_uuid: str | None,
        session_display_name: str | None,
        session_description: str | None,
        selected_session_name: str | None,
    ) -> _PreparedSessionCreation:
        """Prepare the under storage lock required by this session creation service."""
        configuration = self._require_current_owner(owner)
        sessions_root = self._sessions_root_provider()
        repository.validate_sessions_root(sessions_root)
        physical_sessions = repository.session_names()
        draft = self._resolve_creation_draft(
            configuration=configuration,
            repository=repository,
            sessions_root=sessions_root,
            physical_sessions=physical_sessions,
            strategy=strategy,
            parent_session_uuid=parent_session_uuid,
            parent_checkpoint_name=parent_checkpoint_name,
            framework_uuid=framework_uuid,
            session_display_name=session_display_name,
            selected_session_name=selected_session_name,
        )
        storage_name = SessionStorageName.parse(draft.session_name)
        session_path = repository.validate_layout(
            supplied_sessions_root=sessions_root,
            supplied_session_manifest=self._session_config_path_provider(
                draft.session_name
            ),
            storage_name=storage_name,
        )
        request = self._build_metadata_request(
            draft=draft,
            storage_name=storage_name,
            session_description=session_description,
        )
        return _PreparedSessionCreation(
            configuration=configuration,
            metadata=repository.prepare(
                request,
                expected_model_uuid=owner.model_uuid,
                session_path=session_path,
            ),
        )

    def _require_current_owner(
        self,
        owner: _SessionCreationOwner,
    ) -> SessionCreationConfiguration:
        """Return the required current owner for this session creation service."""
        configuration = self._active_configuration_provider()
        active_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        active_root = Path(
            str(getattr(configuration, "model_folder_path", "") or "")
        ).expanduser()
        if (
            configuration is None
            or active_uuid != owner.model_uuid
            or active_root.absolute() != owner.model_root
        ):
            raise OperationConflict(
                "The active model changed before session creation was committed."
            )
        return configuration

    def _resolve_creation_draft(
        self,
        *,
        configuration: SessionCreationConfiguration,
        repository: SessionCreationMetadataRepository,
        sessions_root: str,
        physical_sessions: Sequence[str],
        strategy: str,
        parent_session_uuid: str | None,
        parent_checkpoint_name: str | None,
        framework_uuid: str | None,
        session_display_name: str | None,
        selected_session_name: str | None,
    ) -> SessionCreationDraft:
        """Resolve the creation draft required by this session creation service."""
        created_at = utc_now_iso()
        return self.resolve_session_creation_draft(
            sessions_root=sessions_root,
            created_at_raw=created_at,
            strategy=strategy,
            parent_session_uuid=parent_session_uuid,
            parent_checkpoint_name=parent_checkpoint_name,
            framework_uuid=framework_uuid,
            active_runtime_framework_uuid=getattr(
                configuration,
                "active_runtime_framework_uuid",
                None,
            ),
            preferred_framework_uuid=getattr(
                configuration,
                "preferred_framework_uuid",
                None,
            ),
            session_display_name=session_display_name,
            indexed_sessions=physical_sessions,
            selected_session_name=selected_session_name,
            available_checkpoints_resolver=repository.list_checkpoint_names,
        )

    @staticmethod
    def _build_metadata_request(
        *,
        draft: SessionCreationDraft,
        storage_name: SessionStorageName,
        session_description: str | None,
    ) -> SessionCreationMetadataRequest:
        """Build the metadata request required by this session creation service."""
        return SessionCreationMetadataRequest(
            identity_uuid=draft.session_uuid,
            label=draft.session_label,
            storage_name=storage_name,
            description=str(session_description or "").strip(),
            framework_uuid=draft.selected_framework_uuid,
            parent_session=(
                SessionStorageName.parse(draft.resolved_parent_session_uuid)
                if draft.resolved_parent_session_uuid
                else None
            ),
            load_checkpoint_from_session=(
                SessionStorageName.parse(draft.load_checkpoint_from_session)
                if draft.load_checkpoint_from_session
                else None
            ),
            load_checkpoint_name=(
                CheckpointStorageName.parse(draft.load_checkpoint_name)
                if draft.load_checkpoint_name
                else None
            ),
            base_step_offset=draft.base_step_offset,
            created_at=draft.created_at,
        )

    @staticmethod
    def _commit_storage(
        repository: SessionCreationMetadataRepository,
        transaction: StorageTransaction,
        prepared: _PreparedSessionCreation,
    ) -> _PreparedSessionCreation:
        """Commit the storage owned by this session creation service."""
        staged = repository.stage(transaction, prepared.metadata)
        repository.publish(transaction, prepared.metadata, staged)
        return prepared

    _PROJECTION_FIELDS = (
        "training_sessions",
        "model_version",
        "active_runtime_framework_uuid",
        "selected_default_model_name",
        "selected_default_checkpoint_name",
        "model_configuration",
        "model_configuration_schema",
    )

    def _capture_projection(
        self,
        configuration: SessionCreationConfiguration,
    ) -> _ProjectionSnapshot:
        """Capture the projection required by this session creation service."""
        attributes = {
            name: _AttributeSnapshot(
                present=hasattr(configuration, name),
                value=deepcopy(getattr(configuration, name, None)),
            )
            for name in self._PROJECTION_FIELDS
        }
        return _ProjectionSnapshot(
            attributes=attributes,
            runtime_selection=self._runtime_selection_state_provider(),
        )

    def _publish_projection(
        self,
        prepared: _PreparedSessionCreation,
        snapshot: _ProjectionSnapshot,
        *,
        operation_id: str,
    ) -> bool:
        """Publish the projection owned by this session creation service."""
        metadata = prepared.metadata
        configuration = prepared.configuration
        try:
            configuration_schema = ConfigurationSchema.from_configuration(
                metadata.model_configuration,
                model_folder_path=str(metadata.session_path.parents[1]),
            )
            self._set_projection_attribute(
                configuration,
                "training_sessions",
                list(metadata.indexed_sessions),
            )
            self._set_projection_attribute(
                configuration,
                "model_version",
                metadata.model_version,
            )
            self._set_projection_attribute(
                configuration,
                "active_runtime_framework_uuid",
                metadata.request.framework_uuid,
            )
            self._set_projection_attribute(
                configuration,
                "selected_default_model_name",
                str(metadata.request.storage_name),
            )
            self._set_projection_attribute(
                configuration,
                "selected_default_checkpoint_name",
                "",
            )
            self._set_projection_attribute(
                configuration,
                "model_configuration",
                deepcopy(metadata.model_configuration),
            )
            self._set_projection_attribute(
                configuration,
                "model_configuration_schema",
                configuration_schema,
            )
            self._runtime_selection_state_committer(
                RuntimeSelectionUpdate(
                    session_name=str(metadata.request.storage_name),
                    checkpoint_name=None,
                    active_framework_uuid=metadata.request.framework_uuid,
                )
            )
        except Exception as error:
            try:
                self._restore_projection(configuration, snapshot)
            except Exception as restoration_error:
                self._log_creation_post_commit_error(
                    restoration_error,
                    operation_id=operation_id,
                    phase="projection_restore",
                )
            self._log_creation_post_commit_error(
                error,
                operation_id=operation_id,
                phase="projection_publish",
            )
            return False
        return True

    def _restore_projection(
        self,
        configuration: SessionCreationConfiguration,
        snapshot: _ProjectionSnapshot,
    ) -> None:
        """Restore the projection owned by this session creation service."""
        for name, attribute in snapshot.attributes.items():
            if attribute.present:
                setattr(configuration, name, deepcopy(attribute.value))
            elif hasattr(configuration, name):
                delattr(configuration, name)
        try:
            self._runtime_selection_state_committer(snapshot.runtime_selection)
        except Exception as exc:
            self._warning(
                "Runtime selection restoration failed after a committed "
                f"session projection error ({type(exc).__name__})."
            )

    @staticmethod
    def _set_projection_attribute(
        configuration: SessionCreationConfiguration,
        name: str,
        value: object,
    ) -> None:
        """Set the projection attribute owned by this session creation service."""
        setattr(configuration, name, value)

    def _recover_creation_context_exit(
        self,
        committed: SessionCreationOutcome,
        error: Exception,
        *,
        operation_id: str,
    ) -> SessionCreationOutcome:
        """Return the recover creation context exit value owned by this session creation service."""
        self._log_creation_post_commit_error(
            error,
            operation_id=operation_id,
            phase="context_exit",
        )
        return replace(
            committed,
            degraded=True,
            reload_required=True,
            storage_recovery_required=True,
        )

    def _log_creation_post_commit_error(
        self,
        error: Exception,
        *,
        operation_id: str,
        phase: str = "storage_cleanup",
    ) -> None:
        """Log the creation post commit error owned by this session creation service."""
        if isinstance(error, ExceptionGroup):
            for nested_error in error.exceptions:
                if isinstance(nested_error, Exception):
                    self._log_creation_post_commit_error(
                        nested_error,
                        operation_id=operation_id,
                        phase=phase,
                    )
            return
        self._warning(
            "session_creation_post_commit_failed "
            f"operation_id={operation_id} phase={phase} "
            f"exception_type={type(error).__name__}"
        )
        self._exception(error)

    def _creation_failure_result(
        self,
        error: Exception,
        *,
        operation_id: str,
    ) -> OperationResult[SessionCreationOutcome]:
        """Return the creation failure result value owned by this session creation service."""
        classified = self._classify_creation_failure(error)
        failure = failure_from_exception(
            classified,
            operation_id=operation_id,
            retry_policy=(
                RetryPolicy.SAFE
                if isinstance(classified, OperationTimeout)
                else None
            ),
        )
        self._warning(
            "session_creation_failed "
            f"operation_id={operation_id} failure_code={failure.code} "
            f"exception_type={type(error).__name__}"
        )
        if not isinstance(error, ApplicationError):
            self._exception(error)
        return OperationResult.failed(failure)

    @staticmethod
    def _classify_creation_failure(error: Exception) -> Exception:
        """Classify the creation failure owned by this session creation service."""
        if isinstance(error, ApplicationError):
            return error
        if isinstance(error, StorageMutationLockTimeout):
            return OperationTimeout(
                "Session storage is busy. Try again."
            )
        if isinstance(error, StoragePathViolation | InvalidStorageName):
            return ApplicationValidationError(
                "Session creation references invalid managed storage."
            )
        if isinstance(error, FileExistsError):
            return OperationConflict(
                "The requested session storage name is already in use."
            )
        if isinstance(error, FileNotFoundError | NotADirectoryError):
            return ResourceNotFoundError(
                "Required model or parent-session storage is unavailable."
            )
        if isinstance(
            error,
            StorageTransactionError | StorageMutationLockError | OSError,
        ):
            return PersistenceError(
                "Session storage could not be committed safely."
            )
        return error
