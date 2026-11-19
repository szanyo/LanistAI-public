# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Exact-once runtime cleanup and shared blocking/async disposal ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Event, RLock, Thread
from typing import Protocol

from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    ExecutionSnapshot,
    ExecutionStopTimeoutError,
    InvalidExecutionStateError,
)


class RuntimeExecutionDisposalPort(Protocol):
    """Execution state and stop operations needed during disposal."""

    def snapshot(self) -> ExecutionSnapshot:
        """Return the snapshot value provided by this runtime execution disposal port."""
        ...

    def is_alive(self) -> bool:
        """Return whether alive applies to this runtime execution disposal port."""
        ...

    def stop_and_wait(
        self,
        timeout_seconds: float,
    ) -> Awaitable[ExecutionSnapshot]:
        """Stop the and wait workflow owned by this runtime execution disposal port."""
        ...

    def wait_for_current(
        self,
        timeout_seconds: float | None = None,
    ) -> Awaitable[ExecutionSnapshot]:
        """Return whether wait for current applies to this runtime execution disposal port."""
        ...

    def join(self, timeout_seconds: float | None = None) -> None:
        """Join the operation owned by this runtime execution disposal port."""
        ...


class RuntimeWriterCleanupPort(Protocol):
    """Writer reset and retryable cleanup operations owned by the disposer."""

    def reset_cleanup_state(self) -> None:
        """Reset the cleanup state owned by this runtime writer cleanup port."""
        ...

    def cleanup(self) -> list[BaseException]:
        """Clean up the operation owned by this runtime writer cleanup port."""
        ...


@dataclass(slots=True)
class _DisposalAttempt:
    """Thread-safe completion shared by every caller of one disposal attempt."""

    completion: Event
    result: ExecutionSnapshot | None = None
    error: BaseException | None = None


class RuntimeResourceDisposer:
    """Coordinate callbacks, writer, framework hook, and runtime retirement.

    One instance belongs to one runtime. It serializes the start/dispose race,
    shares the exact same result or exception object with concurrent blocking
    and asynchronous callers, and remembers successful cleanup steps so a
    recovery retry repeats only the work that previously failed.
    """

    def __init__(
        self,
        *,
        execution: RuntimeExecutionDisposalPort,
        writer: RuntimeWriterCleanupPort,
        framework_cleanup: Callable[[], None],
    ) -> None:
        """Initialize the runtime resource disposer and its owned state."""
        self._execution = execution
        self._writer = writer
        self._framework_cleanup = framework_cleanup
        self._callbacks: list[Callable[[], None]] = []
        self._completed_callback_ids: set[int] = set()
        self._framework_cleanup_completed = False
        self._cleanup_in_progress = False
        self._lock = RLock()
        self._claimed_cleanup_ids: set[str] = set()
        self._disposal_requested = False
        self._disposal_completed = False
        self._disposal_attempt: _DisposalAttempt | None = None

    @contextmanager
    def start_guard(self) -> Iterator[None]:
        """Serialize worker allocation against the first disposal request."""

        with self._runtime_mutation_guard(
            "Cannot start a disposed model runtime."
        ):
            yield

    @contextmanager
    def configuration_guard(self) -> Iterator[None]:
        """Serialize mutable runtime configuration against start/disposal."""

        with self._runtime_mutation_guard(
            "Cannot configure a disposed model runtime."
        ):
            yield

    @contextmanager
    def resource_creation_guard(self) -> Iterator[None]:
        """Prevent new owned resources once cleanup or disposal has started."""

        with self._runtime_mutation_guard(
            "Cannot create resources for a disposing model runtime."
        ):
            yield

    @contextmanager
    def _runtime_mutation_guard(self, error_message: str) -> Iterator[None]:
        """Hold the ownership lock across one pre-start runtime mutation."""

        with self._lock:
            if self._disposal_requested or self._cleanup_in_progress:
                snapshot = self._execution.snapshot()
                raise InvalidExecutionStateError(
                    error_message,
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            yield

    def prepare_execution_run(self) -> None:
        """Reset per-run successful-step ownership before external setup."""

        with self._lock:
            self._completed_callback_ids.clear()
            self._writer.reset_cleanup_state()
            self._framework_cleanup_completed = False

    def register_execution_cleanup(self, callback: Callable[[], None]) -> None:
        """Register a callback that runs once for every execution run."""

        with self._lock:
            if self._disposal_requested or self._cleanup_in_progress:
                snapshot = self._execution.snapshot()
                raise InvalidExecutionStateError(
                    "Cannot register cleanup after runtime cleanup has started.",
                    state=snapshot.state,
                    run_id=snapshot.run_id,
                )
            self._callbacks.append(callback)

    def cleanup_execution_run(self, run_id: str) -> None:
        """Clean one coordinator run, retaining failed steps for recovery."""

        if not self._claim_cleanup(run_id):
            return
        try:
            self._cleanup_owned_resources()
        finally:
            # Claims prevent concurrent work. Successful-step state provides
            # idempotence and permits a later disposal to collect resources
            # registered after a completed run.
            self._release_cleanup_claim(run_id)

    def close_blocking(self) -> ExecutionSnapshot:
        """Dispose an idle runtime and share the attempt with all waiters."""

        attempt, owns_attempt = self._begin_disposal_attempt()
        if attempt is None:
            return self._execution.snapshot()
        if not owns_attempt:
            return self._wait_for_attempt_blocking(attempt)
        try:
            result = self._dispose_idle_blocking()
        except BaseException as error:
            self._finish_disposal_attempt(attempt, error=error)
            raise
        self._finish_disposal_attempt(attempt, result=result)
        return result

    async def close_async(
        self,
        timeout_seconds: float = 30.0,
    ) -> ExecutionSnapshot:
        """Stop if needed, clean resources, and share one disposal attempt."""

        attempt, owns_attempt = self._begin_disposal_attempt()
        if attempt is None:
            return self._execution.snapshot()
        if not owns_attempt:
            return await self._wait_for_attempt_async(attempt)
        try:
            result = await self._dispose_async(timeout_seconds)
        except BaseException as error:
            self._finish_disposal_attempt(attempt, error=error)
            raise
        self._finish_disposal_attempt(attempt, result=result)
        return result

    def _claim_cleanup(self, cleanup_id: str) -> bool:
        """Return the claim cleanup value owned by this runtime resource disposer."""
        with self._lock:
            if cleanup_id in self._claimed_cleanup_ids:
                return False
            self._claimed_cleanup_ids.add(cleanup_id)
            return True

    def _release_cleanup_claim(self, cleanup_id: str) -> None:
        """Release the cleanup claim owned by this runtime resource disposer."""
        with self._lock:
            self._claimed_cleanup_ids.discard(cleanup_id)

    def _cleanup_registered_callbacks(self) -> list[BaseException]:
        """Clean up the registered callbacks owned by this runtime resource disposer."""
        failures: list[BaseException] = []
        with self._lock:
            callbacks = tuple(self._callbacks)
        for index, callback in enumerate(callbacks):
            with self._lock:
                if index in self._completed_callback_ids:
                    continue
            try:
                callback()
            except BaseException as error:
                failures.append(error)
            else:
                with self._lock:
                    self._completed_callback_ids.add(index)
        return failures

    def _cleanup_framework_resources(self) -> list[BaseException]:
        """Clean up the framework resources owned by this runtime resource disposer."""
        with self._lock:
            if self._framework_cleanup_completed:
                return []
        try:
            self._framework_cleanup()
        except BaseException as error:
            return [error]
        with self._lock:
            self._framework_cleanup_completed = True
        return []

    def _cleanup_owned_resources(self) -> None:
        """Clean up the owned resources owned by this runtime resource disposer."""
        with self._lock:
            if self._cleanup_in_progress:
                raise RuntimeError("Runtime resource cleanup is already active.")
            self._cleanup_in_progress = True
        try:
            failures = [
                *self._cleanup_registered_callbacks(),
                *self._writer.cleanup(),
                *self._cleanup_framework_resources(),
            ]
        finally:
            with self._lock:
                self._cleanup_in_progress = False
        if failures:
            raise RuntimeError(
                f"{len(failures)} execution resource cleanup operation(s) failed."
            ) from failures[0]

    async def _cleanup_owned_resources_async(self) -> None:
        """Clean up the owned resources async owned by this runtime resource disposer."""
        completion = Event()
        errors: list[BaseException] = []

        def cleanup_worker() -> None:
            """Clean up the worker owned by this runtime resource disposer."""
            try:
                self._cleanup_owned_resources()
            except BaseException as error:
                errors.append(error)
            finally:
                completion.set()

        Thread(
            target=cleanup_worker,
            daemon=True,
            name="model-runtime-cleanup",
        ).start()

        async def wait_for_worker() -> None:
            """Perform the wait for worker step within the enclosing cleanup owned resources async
            workflow.
            """
            while not completion.is_set():
                await asyncio.sleep(0.001)

        wait_task = asyncio.create_task(wait_for_worker())
        cancellation: asyncio.CancelledError | None = None
        while not wait_task.done():
            try:
                await asyncio.shield(wait_task)
            except asyncio.CancelledError as error:
                # Python cannot pre-empt the cleanup thread. Repeated caller
                # cancellation must not release the cleanup claim while that
                # thread can still mutate framework-owned resources.
                if cancellation is None:
                    cancellation = error
        if cancellation is not None:
            raise cancellation
        if errors:
            raise errors[0]

    def _begin_disposal_attempt(
        self,
    ) -> tuple[_DisposalAttempt | None, bool]:
        """Begin the disposal attempt owned by this runtime resource disposer."""
        with self._lock:
            if self._disposal_completed:
                return None, False
            self._disposal_requested = True
            attempt = self._disposal_attempt
            if attempt is not None:
                return attempt, False
            attempt = _DisposalAttempt(completion=Event())
            self._disposal_attempt = attempt
            return attempt, True

    def _finish_disposal_attempt(
        self,
        attempt: _DisposalAttempt,
        *,
        result: ExecutionSnapshot | None = None,
        error: BaseException | None = None,
    ) -> None:
        """Finish the disposal attempt owned by this runtime resource
disposer."""
        with self._lock:
            attempt.result = result
            attempt.error = error
            if error is None:
                self._disposal_completed = True
            if self._disposal_attempt is attempt:
                self._disposal_attempt = None
            attempt.completion.set()

    @staticmethod
    def _attempt_result(attempt: _DisposalAttempt) -> ExecutionSnapshot:
        """Return the attempt result value owned by this runtime resource disposer."""
        if attempt.error is not None:
            raise attempt.error
        if attempt.result is None:
            raise RuntimeError("Runtime disposal completed without a result.")
        return attempt.result

    @classmethod
    def _wait_for_attempt_blocking(
        cls,
        attempt: _DisposalAttempt,
    ) -> ExecutionSnapshot:
        """Return the wait for attempt blocking value owned by this runtime resource disposer."""
        attempt.completion.wait()
        return cls._attempt_result(attempt)

    @classmethod
    async def _wait_for_attempt_async(
        cls,
        attempt: _DisposalAttempt,
    ) -> ExecutionSnapshot:
        """Run the wait for attempt async responsibility for this runtime resource disposer
        asynchronously.
        """
        while not attempt.completion.is_set():
            await asyncio.sleep(0.001)
        return cls._attempt_result(attempt)

    def _dispose_idle_blocking(self) -> ExecutionSnapshot:
        """Return the dispose idle blocking value owned by this runtime resource disposer."""
        snapshot = self._execution.snapshot()
        if snapshot.state in ACTIVE_EXECUTION_STATES or snapshot.worker_alive:
            raise InvalidExecutionStateError(
                "Blocking disposal requires an idle model runtime.",
                state=snapshot.state,
                run_id=snapshot.run_id,
            )
        if snapshot.run_id is not None:
            self._execution.join()
        snapshot = self._execution.snapshot()
        cleanup_id = snapshot.run_id or "runtime-disposal"
        if self._claim_cleanup(cleanup_id):
            try:
                self._cleanup_owned_resources()
            finally:
                self._release_cleanup_claim(cleanup_id)
        return self._execution.snapshot()

    async def _dispose_async(self, timeout_seconds: float) -> ExecutionSnapshot:
        """Run the dispose async responsibility for this runtime resource disposer asynchronously.
        """
        await self._wait_for_execution_quiescence(timeout_seconds)

        snapshot = self._execution.snapshot()
        cleanup_id = snapshot.run_id or "runtime-disposal"
        if self._claim_cleanup(cleanup_id):
            try:
                await self._cleanup_owned_resources_async()
            finally:
                self._release_cleanup_claim(cleanup_id)
        return self._execution.snapshot()

    async def _wait_for_execution_quiescence(
        self,
        timeout_seconds: float,
    ) -> None:
        """Run the wait for execution quiescence responsibility for this runtime resource disposer
        asynchronously.
        """
        snapshot = self._execution.snapshot()
        if snapshot.run_id is None:
            return
        if snapshot.state in ACTIVE_EXECUTION_STATES:
            try:
                await self._execution.stop_and_wait(timeout_seconds)
                return
            except InvalidExecutionStateError:
                # The worker can become terminal between snapshot and command.
                snapshot = self._execution.snapshot()
        try:
            await self._execution.wait_for_current(timeout_seconds)
        except TimeoutError as error:
            raise ExecutionStopTimeoutError(
                state=snapshot.state,
                run_id=snapshot.run_id,
                timeout_seconds=timeout_seconds,
            ) from error
