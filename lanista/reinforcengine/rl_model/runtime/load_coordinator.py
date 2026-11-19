# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Serialize runtime initialization and atomically authorize active-slot swaps."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    OperationCancelled,
    OperationResult,
    failure_from_exception,
)


class RuntimeLoadLease(Protocol):
    """Exclusive transition lease held across candidate prepare and commit."""

    def release(self) -> None:
        """Release the load transition idempotently."""


@dataclass(frozen=True)
class RuntimeLoadRequest:
    """Identity used to reject work superseded by a newer load request."""

    operation_id: str
    generation: int


class RuntimeLoadCoordinator[CandidateT, CommittedT]:
    """Monitor-object boundary for serialized, latest-request-wins loading."""

    def __init__(self) -> None:
        """Initialize the runtime load coordinator and its owned state."""
        self._generation = 0
        self._lock: asyncio.Lock | None = None
        self._last_discard_failure: Exception | None = None
        self._last_post_commit_failure: Exception | None = None

    @property
    def last_discard_failure(self) -> Exception | None:
        """Expose a contained cleanup failure for diagnostics and tests."""

        return self._last_discard_failure

    @property
    def last_post_commit_failure(self) -> Exception | None:
        """Expose a contained post-commit failure for diagnostics."""

        return self._last_post_commit_failure

    def begin(self) -> RuntimeLoadRequest:
        """Begin the operation owned by this runtime load coordinator."""
        self._generation += 1
        return RuntimeLoadRequest(operation_id=uuid4().hex, generation=self._generation)

    def is_current(self, request: RuntimeLoadRequest) -> bool:
        """Return whether current applies to this runtime load coordinator."""
        return request.generation == self._generation

    def _get_lock(self) -> asyncio.Lock:
        # Construct lazily so lightweight managers created outside an event loop
        # do not bind their synchronization primitive to an unrelated loop.
        """Return the lock exposed by this runtime load coordinator."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @staticmethod
    async def _invoke(callback: Callable[..., object], *args: object) -> object:
        """Run the invoke responsibility for this runtime load coordinator asynchronously."""
        result = callback(*args)
        if inspect.isawaitable(result):
            return await result
        return result

    @classmethod
    async def _invoke_after_commit(
        cls,
        callback: Callable[..., object],
        *args: object,
    ) -> object:
        """Finish a committed transition even if its caller is cancelled."""

        finalizer = asyncio.create_task(cls._invoke(callback, *args))
        try:
            return await asyncio.shield(finalizer)
        except asyncio.CancelledError:
            if finalizer.cancelled():
                raise
            # The slot already committed.  Cancellation must not open the
            # transition gate while the previous runtime still owns resources.
            return await finalizer

    async def execute(
        self,
        request: RuntimeLoadRequest,
        *,
        prepare: Callable[[], Awaitable[CandidateT]],
        commit: Callable[[CandidateT], CommittedT | Awaitable[CommittedT]],
        discard: Callable[[CandidateT], object | Awaitable[object]] | None = None,
        after_commit: Callable[[CommittedT], object | Awaitable[object]] | None = None,
        load_lease_factory: Callable[[str], RuntimeLoadLease] | None = None,
    ) -> OperationResult[CommittedT]:
        """Prepare under serialization and commit only if the request is current."""

        async with self._get_lock():
            if not self.is_current(request):
                failure = failure_from_exception(
                    OperationCancelled("Runtime load was superseded by a newer request."),
                    operation_id=request.operation_id,
                )
                return OperationResult.failed(failure)

            candidate: CandidateT | None = None
            load_lease: RuntimeLoadLease | None = None
            try:
                if load_lease_factory is not None:
                    load_lease = load_lease_factory(request.operation_id)
                candidate = await prepare()
                if not self.is_current(request):
                    if discard is not None:
                        await self._invoke(discard, candidate)
                    failure = failure_from_exception(
                        OperationCancelled("Runtime load was superseded by a newer request."),
                        operation_id=request.operation_id,
                    )
                    return OperationResult.failed(failure)

                committed = await self._invoke(commit, candidate)
                if after_commit is not None:
                    try:
                        await self._invoke_after_commit(
                            after_commit,
                            committed,
                        )
                    except Exception as post_commit_error:
                        # A committed slot cannot be rolled back safely merely
                        # because retirement/telemetry finalization failed.
                        self._last_post_commit_failure = post_commit_error
                return OperationResult.success(
                    committed,  # type: ignore[arg-type]
                    operation_id=request.operation_id,
                )
            except Exception as error:
                if candidate is not None and discard is not None:
                    try:
                        await self._invoke(discard, candidate)
                    except Exception as discard_error:
                        # Cleanup failure is logged by the owning adapter. It must
                        # not replace the primary load failure returned to callers.
                        self._last_discard_failure = discard_error
                return OperationResult.failed(
                    failure_from_exception(error, operation_id=request.operation_id)
                )
            finally:
                if load_lease is not None:
                    load_lease.release()
