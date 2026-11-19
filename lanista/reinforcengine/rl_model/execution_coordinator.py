# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Thread ownership and state transitions for model execution.

The coordinator deliberately owns every worker object.  Model runtimes remain
ordinary objects and provide only the operation and cleanup callables for a
run.  This boundary prevents Python's single-use ``Thread`` lifecycle from
leaking into reusable model instances.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Event, RLock, Thread, local
from typing import cast
from uuid import UUID, uuid4

from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    ExecutionAlreadyRunningError,
    ExecutionContext,
    ExecutionLease,
    ExecutionLeaseFactory,
    ExecutionSnapshot,
    ExecutionStopTimeoutError,
    InvalidExecutionStateError,
    RLModelState,
)


@dataclass
class _ExecutionRun:
    """Carry the data required for execution run."""

    run_id: UUID
    lease: ExecutionLease
    pause_event: Event
    stop_event: Event
    worker_start_resolved: Event
    completion: Future[ExecutionSnapshot]
    state: RLModelState = RLModelState.STARTING
    worker: Thread | None = None
    monitor: Thread | None = None
    worker_started: bool = False
    monitor_started: bool = False
    failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    cleanup_claimed: bool = False
    cleanup_failure_reported: bool = False
    lease_release_claimed: bool = False
    stop_timed_out: bool = False


TransitionListener = Callable[[ExecutionSnapshot], None]
OperationCallable = Callable[[ExecutionContext], None]
CleanupCallable = Callable[[ExecutionContext], None]
PreparationCallable = Callable[[ExecutionContext], None]
CleanupFailureListener = Callable[[BaseException], None]


class _NoopExecutionLease:
    """Default lease used until a runtime receives an application policy gate."""

    def __init__(self, run_id: str) -> None:
        """Initialize the noop execution lease and its owned state."""
        self._run_id = run_id

    @property
    def run_id(self) -> str:
        """Return the run id exposed by this noop execution lease."""
        return self._run_id

    def mark_reload_required(self) -> None:
        """Mark the reload required owned by this noop execution lease."""
        return

    def release(self) -> None:
        """Release the operation owned by this noop execution lease."""
        return


class _NoopExecutionLeaseFactory:
    """Construct configured noop execution lease instances."""

    def acquire_execution_lease(self, run_id: str) -> ExecutionLease:
        """Return the acquire execution lease value owned by this noop execution lease factory."""
        return _NoopExecutionLease(run_id)


class ExecutionHandle:
    """Awaitable identity and completion handle for one immutable run."""

    def __init__(
        self,
        coordinator: ExecutionCoordinator,
        run_id: str,
        completion: Future[ExecutionSnapshot],
    ) -> None:
        """Initialize the execution handle and its owned state."""
        self._coordinator = coordinator
        self._run_id = run_id
        self._completion = completion

    @property
    def run_id(self) -> str:
        """Return the run id exposed by this execution handle."""
        return self._run_id

    async def wait(self, timeout_seconds: float | None = None) -> ExecutionSnapshot:
        """Wait without coupling cancellation to the worker-owned completion."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout_seconds is None else loop.time() + timeout_seconds
        while not self._completion.done():
            if deadline is not None:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise TimeoutError
                await asyncio.sleep(min(0.01, remaining))
            else:
                await asyncio.sleep(0.01)
        return self._completion.result()

    def snapshot(self) -> ExecutionSnapshot:
        """Return the snapshot value owned by this execution handle."""
        if self._completion.done():
            return self._completion.result()
        return self._coordinator.snapshot()


class ExecutionCoordinator:
    """Own fresh workers and serialize lifecycle commands for one runtime."""

    def __init__(
        self,
        *,
        on_transition: TransitionListener | None = None,
        thread_name_prefix: str = "model-execution",
        execution_lease_factory: ExecutionLeaseFactory | None = None,
        on_cleanup_failure: CleanupFailureListener | None = None,
    ) -> None:
        """Initialize the execution coordinator and its owned state."""
        self._on_transition = on_transition
        self._on_cleanup_failure = on_cleanup_failure
        self._thread_name_prefix = thread_name_prefix
        self._execution_lease_factory = (
            execution_lease_factory or _NoopExecutionLeaseFactory()
        )
        self._lock = RLock()
        self._thread_context = local()
        self._current: _ExecutionRun | None = None
        self._last_transition_listener_failure: Exception | None = None

    def snapshot(self) -> ExecutionSnapshot:
        """Return the snapshot value owned by this execution coordinator."""
        with self._lock:
            if self._current is None:
                return ExecutionSnapshot(None, RLModelState.IDLE, False)
            return self._snapshot_locked(self._current)

    @property
    def current_run_id(self) -> str | None:
        """Return the current run id exposed by this execution coordinator."""
        return self.snapshot().run_id

    def is_current_run(self, run_id: str | None) -> bool:
        """Return whether current run applies to this execution coordinator."""
        if run_id is None:
            return False
        with self._lock:
            return self._current is not None and str(self._current.run_id) == run_id

    def current_worker_run_id(self) -> str | None:
        """Return the current worker run id value owned by this execution coordinator."""
        context = getattr(self._thread_context, "execution_context", None)
        return context.run_id if context is not None else None

    def failure_for_run(self, run_id: str | None) -> BaseException | None:
        """Return the failure for run value owned by this execution coordinator."""
        with self._lock:
            if self._current is None or str(self._current.run_id) != run_id:
                return None
            return self._current.failure or self._current.cleanup_failure

    @property
    def last_transition_listener_failure(self) -> Exception | None:
        """Return the latest adapter-listener failure without changing run state."""

        with self._lock:
            return self._last_transition_listener_failure

    def start(
        self,
        operation: OperationCallable,
        *,
        cleanup: CleanupCallable,
        prepare: PreparationCallable | None = None,
    ) -> ExecutionHandle:
        """Create and start a fresh worker or reject an overlapping run."""
        with self._lock:
            self._reject_overlapping_run_locked()
            run, context, starting_snapshot = self._prepare_run_locked(
                operation=operation,
                cleanup=cleanup,
                prepare=prepare,
            )

        self._notify(starting_snapshot)
        self._start_run_threads(run, context, cleanup)
        return ExecutionHandle(self, str(run.run_id), run.completion)

    def _reject_overlapping_run_locked(self) -> None:
        """Reject the overlapping run locked owned by this execution
coordinator."""
        current = self._current
        if current is None or not (
            self._worker_alive_locked(current)
            or current.state in ACTIVE_EXECUTION_STATES
            or not current.completion.done()
        ):
            return
        snapshot = self._snapshot_locked(current)
        raise ExecutionAlreadyRunningError(
            f"Execution run {snapshot.run_id} is already {snapshot.state.value}.",
            state=snapshot.state,
            run_id=snapshot.run_id,
        )

    def _prepare_run_locked(
        self,
        *,
        operation: OperationCallable,
        cleanup: CleanupCallable,
        prepare: PreparationCallable | None,
    ) -> tuple[_ExecutionRun, ExecutionContext, ExecutionSnapshot]:
        """Acquire policy ownership and build one not-yet-started worker pair."""

        run_id = uuid4()
        lease = self._execution_lease_factory.acquire_execution_lease(
            str(run_id)
        )
        run = _ExecutionRun(
            run_id=run_id,
            lease=lease,
            pause_event=Event(),
            stop_event=Event(),
            worker_start_resolved=Event(),
            completion=Future(),
        )
        context = ExecutionContext(
            run_id=str(run.run_id),
            _pause_event=run.pause_event,
            _stop_event=run.stop_event,
        )
        try:
            self._attach_run_threads(run, context, operation, cleanup)
            if prepare is not None:
                prepare(context)
        except BaseException as exc:
            run.failure = exc
            try:
                self._cleanup_once(run, context, cleanup)
            except BaseException as cleanup_error:
                run.cleanup_failure = cleanup_error
            finally:
                self._report_cleanup_failure_once(run)
                self._release_execution_lease_once(run)
            raise

        self._current = run
        return run, context, self._snapshot_locked(run)

    def _attach_run_threads(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        operation: OperationCallable,
        cleanup: CleanupCallable,
    ) -> None:
        """Attach the run threads owned by this execution coordinator."""
        run.worker = Thread(
            target=self._run_worker,
            args=(run, context, operation, cleanup),
            daemon=True,
            name=f"{self._thread_name_prefix}-{str(run.run_id)[:8]}",
        )
        run.monitor = Thread(
            target=self._observe_worker_completion,
            args=(run, context, cleanup),
            daemon=True,
            name=f"{self._thread_name_prefix}-monitor-{str(run.run_id)[:8]}",
        )

    def _start_run_threads(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        cleanup: CleanupCallable,
    ) -> None:
        """Start monitor then worker, completing failed starts synchronously."""

        monitor = run.monitor
        worker = run.worker
        if monitor is None or worker is None:
            raise RuntimeError("Execution workers were not prepared.")
        try:
            monitor.start()
        except BaseException as exc:
            run.failure = exc
            self._complete_unstarted_run(run, context, cleanup)
            raise
        else:
            with self._lock:
                run.monitor_started = True

        try:
            worker.start()
        except BaseException as exc:
            with self._lock:
                run.failure = exc
                run.state = RLModelState.FAILED
            run.worker_start_resolved.set()
            monitor.join()
            raise
        else:
            with self._lock:
                run.worker_started = True
            run.worker_start_resolved.set()

    def pause(self) -> ExecutionSnapshot:
        """Return the pause value owned by this execution coordinator."""
        with self._lock:
            run = self._require_state_locked("pause", {RLModelState.RUNNING})
            run.pause_event.set()
            run.state = RLModelState.PAUSED
            snapshot = self._snapshot_locked(run)
        self._notify(snapshot)
        return snapshot

    def resume(self) -> ExecutionSnapshot:
        """Return the resume value owned by this execution coordinator."""
        with self._lock:
            run = self._require_state_locked("resume", {RLModelState.PAUSED})
            run.pause_event.clear()
            run.state = RLModelState.RUNNING
            snapshot = self._snapshot_locked(run)
        self._notify(snapshot)
        return snapshot

    def request_stop(self) -> ExecutionHandle:
        """Return the request stop value owned by this execution coordinator."""
        snapshot: ExecutionSnapshot | None = None
        with self._lock:
            run = self._require_state_locked(
                "stop",
                {
                    RLModelState.STARTING,
                    RLModelState.RUNNING,
                    RLModelState.PAUSED,
                    RLModelState.STOPPING,
                },
            )
            if run.state != RLModelState.STOPPING:
                run.stop_event.set()
                run.pause_event.clear()
                run.state = RLModelState.STOPPING
                snapshot = self._snapshot_locked(run)
            handle = ExecutionHandle(self, str(run.run_id), run.completion)
        if snapshot is not None:
            self._notify(snapshot)
        return handle

    async def stop_and_wait(self, timeout_seconds: float) -> ExecutionSnapshot:
        """Stop the and wait workflow owned by this execution coordinator."""
        handle = self.request_stop()
        try:
            return await handle.wait(timeout_seconds)
        except TimeoutError as exc:
            snapshot = self._mark_stop_timeout(handle.run_id)
            raise ExecutionStopTimeoutError(
                state=snapshot.state,
                run_id=snapshot.run_id,
                timeout_seconds=timeout_seconds,
            ) from exc

    async def wait_for_current(self, timeout_seconds: float | None = None) -> ExecutionSnapshot:
        """Return whether wait for current applies to this execution coordinator."""
        with self._lock:
            if self._current is None:
                return ExecutionSnapshot(None, RLModelState.IDLE, False)
            handle = ExecutionHandle(
                self,
                str(self._current.run_id),
                self._current.completion,
            )
        return await handle.wait(timeout_seconds)

    def check_pause(self) -> None:
        """Check the pause invariant owned by this execution coordinator."""
        self._context_for_caller().wait_if_paused()

    def check_stop(self) -> None:
        """Check the stop invariant owned by this execution coordinator."""
        self._context_for_caller().check_stop()

    def is_alive(self) -> bool:
        """Return whether alive applies to this execution coordinator."""
        with self._lock:
            return self._current is not None and self._worker_alive_locked(self._current)

    def join(self, timeout_seconds: float | None = None) -> None:
        """Join the operation owned by this execution coordinator."""
        with self._lock:
            run = self._current
            worker = (
                run.worker
                if run is not None and run.worker_started
                else None
            )
            monitor = (
                run.monitor
                if run is not None and run.monitor_started
                else None
            )
        if worker is not None:
            worker.join(timeout_seconds)
        if monitor is not None:
            monitor.join(timeout_seconds)

    def _run_worker(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        operation: OperationCallable,
        cleanup: CleanupCallable,
    ) -> None:
        """Run the worker owned by this execution coordinator."""
        self._thread_context.execution_context = context
        terminal_state = RLModelState.COMPLETED
        try:
            if self._enter_running_unless_stopping(run):
                operation(context)
            else:
                terminal_state = RLModelState.STOPPED
            if run.stop_event.is_set():
                terminal_state = RLModelState.STOPPED
        except InterruptedError:
            terminal_state = RLModelState.STOPPED
        except BaseException as exc:
            run.failure = exc
            terminal_state = RLModelState.FAILED
        finally:
            try:
                self._cleanup_once(run, context, cleanup)
            except BaseException as exc:
                run.cleanup_failure = exc
                terminal_state = RLModelState.FAILED

            if run.stop_timed_out:
                terminal_state = RLModelState.FAILED
            with self._lock:
                if run.stop_timed_out:
                    terminal_state = RLModelState.FAILED
                run.state = terminal_state
            del self._thread_context.execution_context

    def _enter_running_unless_stopping(self, run: _ExecutionRun) -> bool:
        """Enter RUNNING atomically unless a STARTING run was already stopped."""

        with self._lock:
            if (
                self._current is not run
                or run.stop_event.is_set()
                or run.state == RLModelState.STOPPING
            ):
                return False
            run.state = RLModelState.RUNNING
            snapshot = self._snapshot_locked(run)
        self._notify(snapshot)
        return True

    def _observe_worker_completion(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        cleanup: CleanupCallable,
    ) -> None:
        """Join the worker, finish cleanup, and only then release its lease."""

        try:
            run.worker_start_resolved.wait()
            if run.worker_started:
                worker = run.worker
                if worker is not None:
                    worker.join()
            else:
                try:
                    self._cleanup_once(run, context, cleanup)
                except BaseException as exc:
                    run.cleanup_failure = exc
        except BaseException as exc:
            if run.failure is None:
                run.failure = exc
        finally:
            try:
                self._report_cleanup_failure_once(run)
                self._release_execution_lease_once(run)
            except BaseException as exc:
                if run.cleanup_failure is None:
                    run.cleanup_failure = exc

            with self._lock:
                if (
                    run.failure is not None
                    or run.cleanup_failure is not None
                    or run.stop_timed_out
                ):
                    run.state = RLModelState.FAILED
                terminal_snapshot = self._snapshot_for_completion(run, run.state)
            try:
                self._notify(terminal_snapshot)
            finally:
                if not run.completion.done():
                    run.completion.set_result(terminal_snapshot)

    def _complete_unstarted_run(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        cleanup: CleanupCallable,
    ) -> None:
        """Finalize a run whose monitor could not be started."""

        try:
            self._cleanup_once(run, context, cleanup)
        except BaseException as exc:
            run.cleanup_failure = exc
        try:
            self._report_cleanup_failure_once(run)
            self._release_execution_lease_once(run)
        except BaseException as exc:
            if run.cleanup_failure is None:
                run.cleanup_failure = exc

        with self._lock:
            run.state = RLModelState.FAILED
            terminal_snapshot = self._snapshot_for_completion(
                run,
                RLModelState.FAILED,
            )
        try:
            self._notify(terminal_snapshot)
        finally:
            if not run.completion.done():
                run.completion.set_result(terminal_snapshot)

    def _cleanup_once(
        self,
        run: _ExecutionRun,
        context: ExecutionContext,
        cleanup: CleanupCallable,
    ) -> None:
        """Clean up the once owned by this execution coordinator."""
        with self._lock:
            if run.cleanup_claimed:
                return
            run.cleanup_claimed = True
        cleanup(context)

    def _release_execution_lease_once(self, run: _ExecutionRun) -> None:
        """Release the execution lease once owned by this execution coordinator."""
        with self._lock:
            if run.lease_release_claimed:
                return
            run.lease_release_claimed = True
        run.lease.release()

    def _report_cleanup_failure_once(self, run: _ExecutionRun) -> None:
        """Report the cleanup failure once owned by this execution
coordinator."""
        failure = run.cleanup_failure
        listener = self._on_cleanup_failure
        if failure is None or listener is None:
            return
        with self._lock:
            if run.cleanup_failure_reported:
                return
            run.cleanup_failure_reported = True
        try:
            listener(failure)
        except BaseException as listener_error:
            if run.failure is None:
                run.failure = listener_error

    def _mark_stop_timeout(self, run_id: str) -> ExecutionSnapshot:
        """Return the mark stop timeout value owned by this execution coordinator."""
        with self._lock:
            run = self._current
            if run is None or str(run.run_id) != run_id:
                return self.snapshot()
            run.stop_timed_out = True
            run.failure = TimeoutError(f"Execution run {run_id} stop timed out.")
            run.lease.mark_reload_required()
        return self._transition(run, RLModelState.FAILED) or self.snapshot()

    def _transition(
        self,
        run: _ExecutionRun,
        state: RLModelState,
    ) -> ExecutionSnapshot | None:
        """Return the transition value owned by this execution coordinator."""
        with self._lock:
            if self._current is not run:
                return None
            if run.stop_timed_out and state != RLModelState.FAILED:
                state = RLModelState.FAILED
            if run.state == state:
                return self._snapshot_locked(run)
            run.state = state
            snapshot = self._snapshot_locked(run)
        self._notify(snapshot)
        return snapshot

    def _require_state_locked(
        self,
        action: str,
        allowed_states: set[RLModelState],
    ) -> _ExecutionRun:
        """Return the required state locked for this execution coordinator."""
        run = self._current
        if run is not None and run.state in allowed_states:
            return run
        snapshot = (
            self._snapshot_locked(run)
            if run is not None
            else ExecutionSnapshot(None, RLModelState.IDLE, False)
        )
        raise InvalidExecutionStateError(
            f"Cannot {action} execution while it is {snapshot.state.value}.",
            state=snapshot.state,
            run_id=snapshot.run_id,
        )

    def _context_for_caller(self) -> ExecutionContext:
        """Return the context for caller value owned by this execution coordinator."""
        context = cast(
            ExecutionContext | None,
            getattr(self._thread_context, "execution_context", None),
        )
        if context is not None:
            return context
        with self._lock:
            run = self._current
            if run is not None and run.state in ACTIVE_EXECUTION_STATES:
                return ExecutionContext(
                    run_id=str(run.run_id),
                    _pause_event=run.pause_event,
                    _stop_event=run.stop_event,
                )
            snapshot = (
                self._snapshot_locked(run)
                if run is not None
                else ExecutionSnapshot(None, RLModelState.IDLE, False)
            )
        raise InvalidExecutionStateError(
            f"No active execution context exists while state is {snapshot.state.value}.",
            state=snapshot.state,
            run_id=snapshot.run_id,
        )

    def _snapshot_locked(self, run: _ExecutionRun) -> ExecutionSnapshot:
        """Return the snapshot locked value owned by this execution coordinator."""
        failure = run.failure or run.cleanup_failure
        return ExecutionSnapshot(
            run_id=str(run.run_id),
            state=run.state,
            worker_alive=self._worker_alive_locked(run),
            stop_requested=run.stop_event.is_set(),
            failure_type=type(failure).__name__ if failure is not None else None,
        )

    @staticmethod
    def _worker_alive_locked(run: _ExecutionRun) -> bool:
        """Return the worker alive locked value owned by this execution coordinator."""
        return run.worker is not None and run.worker.is_alive()

    def _snapshot_for_completion(
        self,
        run: _ExecutionRun,
        state: RLModelState,
    ) -> ExecutionSnapshot:
        """Return the snapshot for completion value owned by this execution coordinator."""
        failure = run.failure or run.cleanup_failure
        return ExecutionSnapshot(
            run_id=str(run.run_id),
            state=state,
            worker_alive=False,
            stop_requested=run.stop_event.is_set(),
            failure_type=type(failure).__name__ if failure is not None else None,
        )

    def _notify(self, snapshot: ExecutionSnapshot) -> None:
        """Notify the operation owned by this execution coordinator."""
        if self._on_transition is None:
            return
        try:
            self._on_transition(snapshot)
        except Exception as error:
            # Transition listeners are outbound adapters. Their failure must
            # remain observable without corrupting the coordinator state machine.
            with self._lock:
                self._last_transition_listener_failure = error
