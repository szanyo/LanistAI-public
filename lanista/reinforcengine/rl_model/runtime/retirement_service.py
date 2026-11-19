# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Idempotent cleanup for runtimes retired from the active slot."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock, Thread
from typing import Protocol, cast
from weakref import ReferenceType, ref

from lanista.reinforcengine.application_errors import (
    ApplicationValidationError,
    OperationResult,
    OperationTimeout,
    RetryPolicy,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime


class DegradedStateReporter(Protocol):
    """Record that execution and administration require a recovery load."""

    def __call__(
        self,
        *,
        reload_required: bool,
        storage_recovery_required: bool,
    ) -> object:
        """Invoke this degraded state reporter as a callable."""
        ...


@dataclass(frozen=True, slots=True)
class RuntimeRetirementOutcome:
    """Describe whether this call invoked or reused a cleanup claim."""

    runtime_type: str | None
    cleanup_method: str | None
    cleanup_invoked: bool
    already_claimed: bool = False


@dataclass(frozen=True, slots=True)
class _CleanupCompletion:
    """Carry the data required for cleanup completion."""

    error: BaseException | None = None


@dataclass(slots=True)
class _RetirementRecord:
    """Describe retirement data."""

    runtime_ref: ReferenceType[BaseModelRuntime]
    cleanup_method: str
    completion: Future[_CleanupCompletion]


class RuntimeRetirementService:
    """Serialize each retired runtime's cleanup and share its final result."""

    def __init__(
        self,
        *,
        cleanup_timeout_seconds: float = 30.0,
        logger: object | None = None,
        degraded_state_reporter: DegradedStateReporter | None = None,
    ) -> None:
        """Initialize the runtime retirement service and its owned state."""
        self._cleanup_timeout_seconds = max(float(cleanup_timeout_seconds), 0.001)
        self._logger = logger
        self._degraded_state_reporter = degraded_state_reporter
        self._record_lock = RLock()
        self._records: dict[int, _RetirementRecord] = {}

    def _log_failure(
        self,
        *,
        error: BaseException,
        operation_id: str,
        runtime_type: str,
        cleanup_method: str | None,
    ) -> None:
        """Log the failure owned by this runtime retirement service."""
        failure = failure_from_exception(error, operation_id=operation_id)
        logger = self._logger
        if logger is not None and hasattr(logger, "warning"):
            logger.warning(
                "runtime_retirement_failed "
                f"operation_id={operation_id} failure_code={failure.code} "
                f"runtime_type={runtime_type} cleanup_method={cleanup_method or '-'}"
            )
        if logger is not None and hasattr(logger, "exception"):
            logger.exception(error)

    def _report_degraded_state(
        self,
        *,
        runtime: BaseModelRuntime,
        replacement: BaseModelRuntime | None,
    ) -> None:
        """Report the degraded state owned by this runtime retirement
service."""
        reporter = self._degraded_state_reporter
        if reporter is None:
            for owner in (replacement, runtime):
                candidate = getattr(owner, "_degraded_state_reporter", None)
                if callable(candidate):
                    reporter = cast(DegradedStateReporter, candidate)
                    break
        if reporter is None:
            return
        try:
            reporter(
                reload_required=True,
                storage_recovery_required=False,
            )
        except Exception as error:
            logger = self._logger
            if logger is not None and hasattr(logger, "exception"):
                logger.exception(error)

    def _failed_retirement(
        self,
        *,
        error: BaseException,
        runtime: BaseModelRuntime,
        replacement: BaseModelRuntime | None,
        operation_id: str,
        cleanup_method: str | None,
    ) -> OperationResult[RuntimeRetirementOutcome]:
        """Return the failed retirement value owned by this runtime retirement service."""
        self._log_failure(
            error=error,
            operation_id=operation_id,
            runtime_type=type(runtime).__name__,
            cleanup_method=cleanup_method,
        )
        self._report_degraded_state(
            runtime=runtime,
            replacement=replacement,
        )
        return OperationResult.failed(
            failure_from_exception(
                error,
                operation_id=operation_id,
                retry_policy=RetryPolicy.SAFE,
            )
        )

    def _forget_runtime(
        self,
        runtime_id: int,
        runtime_ref: ReferenceType[BaseModelRuntime],
    ) -> None:
        """Handle the forget runtime operation owned by this runtime retirement service."""
        with self._record_lock:
            record = self._records.get(runtime_id)
            if record is not None and record.runtime_ref is runtime_ref:
                self._records.pop(runtime_id, None)

    def _release_failed_record(
        self,
        runtime: BaseModelRuntime,
        record: _RetirementRecord,
    ) -> None:
        """Release the failed record owned by this runtime retirement service."""
        with self._record_lock:
            if self._records.get(id(runtime)) is record:
                self._records.pop(id(runtime), None)

    def _get_or_start_record(
        self,
        runtime: BaseModelRuntime,
    ) -> tuple[_RetirementRecord | None, bool]:
        """Return the or start record exposed by this runtime retirement service."""
        runtime_id = id(runtime)
        with self._record_lock:
            existing = self._records.get(runtime_id)
            if existing is not None and existing.runtime_ref() is runtime:
                if not existing.completion.done():
                    return existing, False
                if existing.completion.result().error is None:
                    return existing, False
                self._records.pop(runtime_id, None)

            cleanup_method, cleanup_callable = self._resolve_cleanup(runtime)
            if cleanup_method is None or cleanup_callable is None:
                return None, False

            completion: Future[_CleanupCompletion] = Future()
            runtime_reference = ref(
                runtime,
                lambda dead_ref: self._forget_runtime(runtime_id, dead_ref),
            )
            record = _RetirementRecord(
                runtime_ref=runtime_reference,
                cleanup_method=cleanup_method,
                completion=completion,
            )
            self._records[runtime_id] = record
            Thread(
                target=self._run_cleanup_worker,
                args=(cleanup_callable, completion),
                daemon=True,
                name="runtime-retirement-cleanup",
            ).start()
            return record, True

    @staticmethod
    async def _await_cleanup_result(result: Awaitable[object]) -> object:
        """Run the await cleanup result responsibility for this runtime retirement service
        asynchronously.
        """
        return await result

    @classmethod
    def _run_cleanup_worker(
        cls,
        cleanup_callable: Callable[[], object],
        completion: Future[_CleanupCompletion],
    ) -> None:
        """Run the cleanup worker owned by this runtime retirement service."""
        try:
            result = cleanup_callable()
            if inspect.isawaitable(result):
                asyncio.run(
                    cls._await_cleanup_result(
                        cast(Awaitable[object], result)
                    )
                )
        except BaseException as error:
            completion.set_result(_CleanupCompletion(error=error))
        else:
            completion.set_result(_CleanupCompletion())

    async def _wait_for_completion(
        self,
        record: _RetirementRecord,
        *,
        require_quiescence: bool,
    ) -> _CleanupCompletion:
        """Run the wait for completion responsibility for this runtime retirement service
        asynchronously.
        """
        async def poll() -> _CleanupCompletion:
            """Perform the poll step within the enclosing wait for completion workflow."""
            while not record.completion.done():
                await asyncio.sleep(0.001)
            return record.completion.result()

        if require_quiescence:
            return await poll()
        try:
            async with asyncio.timeout(self._cleanup_timeout_seconds):
                return await poll()
        except TimeoutError as error:
            raise OperationTimeout(
                "Retired runtime cleanup timed out; cleanup remains in progress."
            ) from error

    async def retire(
        self,
        runtime: BaseModelRuntime | None,
        *,
        replacement: BaseModelRuntime | None,
        operation_id: str,
        require_quiescence: bool = False,
    ) -> OperationResult[RuntimeRetirementOutcome]:
        """Retire an inactive runtime without affecting its replacement slot."""

        if runtime is None or runtime is replacement:
            return OperationResult.success(
                RuntimeRetirementOutcome(
                    runtime_type=type(runtime).__name__ if runtime is not None else None,
                    cleanup_method=None,
                    cleanup_invoked=False,
                ),
                operation_id=operation_id,
            )

        runtime_type = type(runtime).__name__
        record, cleanup_started = self._get_or_start_record(runtime)
        if record is None:
            error = ApplicationValidationError(
                "Retired runtime does not expose a supported cleanup method."
            )
            return self._failed_retirement(
                error=error,
                runtime=runtime,
                replacement=replacement,
                operation_id=operation_id,
                cleanup_method=None,
            )

        try:
            completion = await self._wait_for_completion(
                record,
                require_quiescence=require_quiescence,
            )
        except asyncio.CancelledError:
            self._report_degraded_state(
                runtime=runtime,
                replacement=replacement,
            )
            raise
        except Exception as error:
            return self._failed_retirement(
                error=error,
                runtime=runtime,
                replacement=replacement,
                operation_id=operation_id,
                cleanup_method=record.cleanup_method,
            )

        if completion.error is not None:
            self._release_failed_record(runtime, record)
            return self._failed_retirement(
                error=completion.error,
                runtime=runtime,
                replacement=replacement,
                operation_id=operation_id,
                cleanup_method=record.cleanup_method,
            )

        return OperationResult.success(
            RuntimeRetirementOutcome(
                runtime_type=runtime_type,
                cleanup_method=record.cleanup_method,
                cleanup_invoked=cleanup_started,
                already_claimed=not cleanup_started,
            ),
            operation_id=operation_id,
        )

    @staticmethod
    def _resolve_cleanup(
        runtime: BaseModelRuntime,
    ) -> tuple[str | None, Callable[[], object] | None]:
        """Resolve the cleanup required by this runtime retirement service."""
        method_names = (
            "close_async",
            "shutdown_async",
            "close",
            "shutdown",
        )
        if isinstance(runtime, BaseModelRuntime):
            instance_overrides_close = "close_async" in vars(runtime)
            class_close = getattr(type(runtime), "close_async", None)
            if (
                not instance_overrides_close
                and class_close is BaseModelRuntime.close_async
            ):
                method_names = (
                    "close_blocking",
                    "shutdown_async",
                    "close",
                    "shutdown",
                )
        for method_name in method_names:
            method = getattr(runtime, method_name, None)
            if callable(method):
                return method_name, method
        return None, None
