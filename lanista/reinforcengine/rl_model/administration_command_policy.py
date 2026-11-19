# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared ownership, completion, and fail-closed administration policy."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Protocol

from lanista.reinforcengine.application_errors import OperationConflict
from lanista.reinforcengine.rl_model.administration_contracts import (
    ModelAdministrationOwner,
)
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationCompletionGuard,
)

_UNKNOWN_FAILED_CLOSED_CONFIGURATION = object()

type ExceptionInfo = tuple[
    type[BaseException],
    BaseException,
    TracebackType | None,
]


class AdministrationLogger(Protocol):
    """Minimal structured logger surface used at the application boundary."""

    def warning(self, message: str) -> None:
        """Publish a warning record through this administration logger."""
        ...

    def exception(
        self,
        error: BaseException,
        *,
        exc_info: ExceptionInfo | None = None,
    ) -> None:
        """Publish a exception record through this administration logger."""
        ...


class DegradedStateReporter(Protocol):
    """Latch recovery requirements after a durable administration commit."""

    def __call__(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> object:
        """Invoke this degraded state reporter as a callable."""
        ...


class CommittedAdministrationOutcome(Protocol):
    """Flags needed to bridge a durable commit across context release."""

    @property
    def reload_required(self) -> bool:
        """Return the reload required exposed by this committed administration outcome."""
        ...

    @property
    def storage_recovery_required(self) -> bool:
        """Return the storage recovery required exposed by this committed administration outcome."""
        ...


class AdministrationCommandPolicy:
    """Coordinate all destructive commands through one shared safety latch.

    One instance must be shared by model, session, and checkpoint commands.
    Otherwise a failed degradation report in one boundary would not block the
    other two boundaries until the active model is reloaded.
    """

    def __init__(
        self,
        *,
        active_configuration_provider: Callable[[], object | None],
        degraded_state_reporter: DegradedStateReporter,
        administration_operation_factory: Callable[
            [], AbstractContextManager[AdministrationCompletionGuard]
        ],
        logger: AdministrationLogger | None = None,
    ) -> None:
        """Initialize the administration command policy and its owned state."""
        self._active_configuration_provider = active_configuration_provider
        self._degraded_state_reporter = degraded_state_reporter
        self._administration_operation_factory = administration_operation_factory
        self._logger = logger
        self._administration_failed_closed = False
        self._failed_closed_configuration: object | None = None

    def active_configuration(self) -> object | None:
        """Return the opaque active configuration from the composition-owned slot."""

        return self._active_configuration_provider()

    def capture_owner(self) -> ModelAdministrationOwner:
        """Capture the active model identity under the administration gate."""

        with self._administration_operation_factory():
            model_uuid = self._active_model_uuid()
            if not model_uuid:
                raise OperationConflict(
                    "Model administration requires an active model identity."
                )
            return ModelAdministrationOwner(model_uuid=model_uuid)

    def is_owner_current(self, owner: ModelAdministrationOwner) -> bool:
        """Return a non-blocking advisory identity snapshot for UI guards."""

        try:
            return self._is_owner_current(owner)
        except Exception as error:
            self.warning(f"Failed to read the active model identity: {error}")
            return False

    def require_current_owner(self, owner: ModelAdministrationOwner) -> None:
        """Reject a stale workflow owner inside the locked command boundary."""

        if not self._is_owner_current(owner):
            raise OperationConflict(
                "The active model changed before administration was committed."
            )

    def require_available(self) -> None:
        """Reject every command while a shared local failure remains latched."""

        if self._local_failure_is_latched():
            raise OperationConflict(
                "Model administration is blocked until the active model is reloaded."
            )

    def execute[OutcomeT: CommittedAdministrationOutcome](
        self,
        operation: Callable[[], OutcomeT],
        *,
        validate: Callable[[], None],
        context_exit_recovery: Callable[[OutcomeT, Exception], OutcomeT],
        recovery_required_error_types: tuple[type[BaseException], ...] = (),
    ) -> OutcomeT:
        """Run one command and preserve its durable result across unlock."""

        committed: OutcomeT | None = None
        completion_guard: AdministrationCompletionGuard | None = None
        try:
            with self._administration_operation_factory() as context_value:
                if not isinstance(context_value, AdministrationCompletionGuard):
                    raise OperationConflict(
                        "Durable administration requires a completion-aware safety gate."
                    )
                completion_guard = context_value
                try:
                    validate()
                    committed = operation()
                except BaseException as operation_error:
                    self._arm_failed_completion(
                        completion_guard,
                        operation_error,
                        recovery_required_error_types=recovery_required_error_types,
                    )
                    raise
                else:
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
                    "Administration context suppressed an incomplete operation."
                )
            completion_guard.confirm_successful_exit()
            return committed
        except BaseException as error:
            self._fail_completion(completion_guard)
            if committed is None or not isinstance(error, Exception):
                raise
            return context_exit_recovery(committed, error)

    def report_degradation(
        self,
        *,
        operation_id: str,
        failure_event: str,
        reload_required: bool,
        storage_recovery_required: bool,
        detail: str = "",
    ) -> None:
        """Report degradation, latching every command if reporting fails."""

        try:
            self._degraded_state_reporter(
                reload_required=reload_required,
                storage_recovery_required=storage_recovery_required,
            )
        except Exception as error:
            self._latch_local_failure()
            detail_suffix = f" {detail}" if detail else ""
            self.warning(
                f"{failure_event} operation_id={operation_id}{detail_suffix} "
                f"exception_type={type(error).__name__}"
            )
            self.exception(error)

    def warning(self, message: str) -> None:
        """Emit best-effort structured warning without masking command state."""

        if self._logger is None or not hasattr(self._logger, "warning"):
            return
        try:
            self._logger.warning(message)
        except Exception:
            return

    def exception(self, error: BaseException) -> None:
        """Emit best-effort exception diagnostics without leaking failures."""

        if self._logger is None or not hasattr(self._logger, "exception"):
            return
        exception_info = (type(error), error, error.__traceback__)
        try:
            self._logger.exception(error, exc_info=exception_info)
        except TypeError:
            try:
                self._logger.exception(error)
            except Exception:
                return
        except Exception:
            return

    def _active_model_uuid(self) -> str:
        """Return the active model uuid value owned by this administration command policy."""
        configuration = self.active_configuration()
        return str(getattr(configuration, "model_uuid", "") or "").strip()

    def _is_owner_current(self, owner: ModelAdministrationOwner) -> bool:
        """Return whether owner current applies to this administration command policy."""
        expected_uuid = str(getattr(owner, "model_uuid", "") or "").strip()
        active_uuid = self._active_model_uuid()
        return bool(expected_uuid and active_uuid and expected_uuid == active_uuid)

    def _local_failure_is_latched(self) -> bool:
        """Return the local failure is latched value owned by this administration command policy."""
        if not self._administration_failed_closed:
            return False
        if self._failed_closed_configuration is _UNKNOWN_FAILED_CLOSED_CONFIGURATION:
            return True
        try:
            active_configuration = self.active_configuration()
        except Exception:
            return True
        if active_configuration is self._failed_closed_configuration:
            return True
        self._administration_failed_closed = False
        self._failed_closed_configuration = None
        return False

    def _latch_local_failure(self) -> None:
        """Latch the local failure owned by this administration command
policy."""
        self._administration_failed_closed = True
        try:
            self._failed_closed_configuration = self.active_configuration()
        except Exception:
            self._failed_closed_configuration = _UNKNOWN_FAILED_CLOSED_CONFIGURATION

    @staticmethod
    def _arm_failed_completion(
        completion_guard: AdministrationCompletionGuard,
        error: BaseException,
        *,
        recovery_required_error_types: tuple[type[BaseException], ...],
    ) -> None:
        """Arm the failed completion owned by this administration command
policy."""
        if isinstance(error, recovery_required_error_types):
            completion_guard.latch_recovery_required()
        elif not isinstance(error, Exception):
            completion_guard.arm_committed_completion(
                reload_required=True,
                storage_recovery_required=True,
            )

    def _fail_completion(
        self,
        completion_guard: AdministrationCompletionGuard | None,
    ) -> None:
        """Fail the completion owned by this administration command policy."""
        if completion_guard is None:
            return
        try:
            completion_guard.fail_exit()
        except Exception as completion_error:
            self._latch_local_failure()
            self.warning(
                "administration_completion_fail_closed "
                f"exception_type={type(completion_error).__name__}"
            )
            self.exception(completion_error)
