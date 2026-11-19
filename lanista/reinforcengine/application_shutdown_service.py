# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Fail-closed orchestration for process-level application shutdown."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from enum import StrEnum
from threading import Thread
from typing import Protocol


class RuntimeShutdownPort(Protocol):
    """Narrow runtime lifecycle needed by application shutdown."""

    async def close_async(self, timeout_seconds: float) -> object:
        """Stop execution and release all runtime-owned resources."""

    def is_alive(self) -> bool:
        """Return whether the runtime still owns a live execution worker."""


class ShutdownStep(StrEnum):
    """Stable identifiers for observable application shutdown steps."""

    STOP_TENSORBOARD = "stop TensorBoard"
    RESOLVE_ACTIVE_RUNTIME = "resolve active model"
    QUIESCE_ACTIVE_RUNTIME = "quiesce active model"
    VERIFY_RUNTIME_QUIESCENCE = "verify active model quiescence"
    STOP_RAY_RUNTIME = "stop Ray runtime"


@dataclass(frozen=True, slots=True)
class ShutdownFailure:
    """Sanitized evidence for one failed shutdown step."""

    step: ShutdownStep
    error_type: str


@dataclass(frozen=True, slots=True)
class ApplicationShutdownReport:
    """Describe completed, failed, and deliberately skipped cleanup work."""

    failures: tuple[ShutdownFailure, ...]
    runtime_resolution_succeeded: bool
    runtime_quiescent: bool
    ray_shutdown_attempted: bool

    @property
    def failure_steps(self) -> tuple[str, ...]:
        """Return stable presentation labels without exposing error details."""

        return tuple(failure.step.value for failure in self.failures)


@dataclass(frozen=True, slots=True)
class _RuntimeCloseCompletion:
    """Carry the data required for runtime close completion."""

    error: BaseException | None = None


class ApplicationShutdownService:
    """Await runtime quiescence before allowing process-global cleanup."""

    def __init__(
        self,
        *,
        tensorboard_stopper: Callable[[], object],
        active_runtime_provider: Callable[[], RuntimeShutdownPort | None],
        ray_stopper: Callable[[], object],
        runtime_timeout_seconds: float = 30.0,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialize the application shutdown service and its owned state."""
        self._tensorboard_stopper = tensorboard_stopper
        self._active_runtime_provider = active_runtime_provider
        self._ray_stopper = ray_stopper
        self._runtime_timeout_seconds = max(
            float(runtime_timeout_seconds),
            0.001,
        )
        self._logger = logger or logging.getLogger(__name__)

    def _record_failure(
        self,
        failures: list[ShutdownFailure],
        step: ShutdownStep,
        error: Exception,
    ) -> None:
        """Record the failure owned by this application shutdown service."""
        failures.append(
            ShutdownFailure(
                step=step,
                error_type=type(error).__name__,
            )
        )
        self._logger.error(
            "application_shutdown_step_failed step=%s error_type=%s",
            step.value,
            type(error).__name__,
            exc_info=(type(error), error, error.__traceback__),
        )

    def _attempt_sync(
        self,
        failures: list[ShutdownFailure],
        step: ShutdownStep,
        operation: Callable[[], object],
    ) -> bool:
        """Return the attempt sync value owned by this application shutdown service."""
        try:
            operation()
        except Exception as error:
            self._record_failure(failures, step, error)
            return False
        return True

    @staticmethod
    def _run_runtime_close(
        runtime: RuntimeShutdownPort,
        timeout_seconds: float,
        completion: Future[_RuntimeCloseCompletion],
    ) -> None:
        """Run the runtime close owned by this application shutdown service."""
        try:
            asyncio.run(runtime.close_async(timeout_seconds))
        except BaseException as error:
            completion.set_result(_RuntimeCloseCompletion(error=error))
        else:
            completion.set_result(_RuntimeCloseCompletion())

    async def _close_runtime(
        self,
        runtime: RuntimeShutdownPort,
    ) -> None:
        """Bound the complete close operation without abandoning its owner."""

        completion: Future[_RuntimeCloseCompletion] = Future()
        Thread(
            target=self._run_runtime_close,
            args=(runtime, self._runtime_timeout_seconds, completion),
            daemon=True,
            name="application-runtime-shutdown",
        ).start()

        async def wait_for_completion() -> _RuntimeCloseCompletion:
            """Perform the wait for completion step within the enclosing close runtime workflow."""
            while not completion.done():
                await asyncio.sleep(0.001)
            return completion.result()

        try:
            async with asyncio.timeout(self._runtime_timeout_seconds):
                result = await wait_for_completion()
        except TimeoutError as error:
            raise TimeoutError(
                "Active runtime cleanup exceeded the shutdown deadline."
            ) from error
        if result.error is not None:
            raise result.error

    async def shutdown(self) -> ApplicationShutdownReport:
        """Attempt independent cleanup while preserving the Ray safety gate."""

        failures: list[ShutdownFailure] = []
        self._attempt_sync(
            failures,
            ShutdownStep.STOP_TENSORBOARD,
            self._tensorboard_stopper,
        )

        try:
            runtime = self._active_runtime_provider()
        except Exception as error:
            self._record_failure(
                failures,
                ShutdownStep.RESOLVE_ACTIVE_RUNTIME,
                error,
            )
            runtime_resolution_succeeded = False
            runtime_quiescent = False
        else:
            runtime_resolution_succeeded = True
            runtime_quiescent = runtime is None
            if runtime is not None:
                try:
                    await self._close_runtime(runtime)
                except Exception as error:
                    runtime_quiescent = False
                    self._record_failure(
                        failures,
                        ShutdownStep.QUIESCE_ACTIVE_RUNTIME,
                        error,
                    )
                else:
                    try:
                        runtime_quiescent = not runtime.is_alive()
                    except Exception as error:
                        runtime_quiescent = False
                        self._record_failure(
                            failures,
                            ShutdownStep.VERIFY_RUNTIME_QUIESCENCE,
                            error,
                        )
                    else:
                        if not runtime_quiescent:
                            self._record_failure(
                                failures,
                                ShutdownStep.VERIFY_RUNTIME_QUIESCENCE,
                                RuntimeError(
                                    "Active runtime remained alive after shutdown."
                                ),
                            )

        ray_shutdown_attempted = (
            runtime_resolution_succeeded and runtime_quiescent
        )
        if ray_shutdown_attempted:
            self._attempt_sync(
                failures,
                ShutdownStep.STOP_RAY_RUNTIME,
                self._ray_stopper,
            )

        return ApplicationShutdownReport(
            failures=tuple(failures),
            runtime_resolution_succeeded=runtime_resolution_succeeded,
            runtime_quiescent=runtime_quiescent,
            ray_shutdown_attempted=ray_shutdown_attempted,
        )
