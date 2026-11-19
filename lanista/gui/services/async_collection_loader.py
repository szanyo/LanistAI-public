# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Load async collection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from threading import Thread


async def run_blocking[Result](source: Callable[[], Result]) -> Result:
    """Run one blocking operation while the Textual event loop remains responsive."""

    results: list[Result] = []
    errors: list[BaseException] = []

    def _execute() -> None:
        try:
            results.append(source())
        except BaseException as error:
            errors.append(error)

    thread = Thread(
        target=_execute,
        name="lanistai-ui-loader",
        daemon=True,
    )
    thread.start()
    while thread.is_alive():
        await asyncio.sleep(0.01)
    thread.join()
    if errors:
        raise errors[0]
    return results[0]


@dataclass(frozen=True)
class AsyncCollectionResult[T]:
    """Report the result of async collection."""

    request_id: int
    items: tuple[T, ...] = ()
    error: BaseException | None = None

    @property
    def stale(self) -> bool:
        """Return the stale exposed by this async collection result."""
        return False

    @property
    def ok(self) -> bool:
        """Return the ok exposed by this async collection result."""
        return self.error is None


class AsyncCollectionLoader[T]:
    """Small request-id guarded loader for IO-heavy dialog collections."""

    def __init__(self, source: Callable[[], list[T] | tuple[T, ...]]) -> None:
        """Initialize the async collection loader and its owned state."""
        self._source = source
        self._latest_request_id = 0

    @property
    def latest_request_id(self) -> int:
        """Return the latest request id exposed by this async collection loader."""
        return self._latest_request_id

    def begin_request(self) -> int:
        """Begin the request owned by this async collection loader."""
        self._latest_request_id += 1
        return self._latest_request_id

    def is_current(self, request_id: int) -> bool:
        """Return whether current applies to this async collection loader."""
        return request_id == self._latest_request_id

    async def load(self, request_id: int | None = None) -> AsyncCollectionResult[T]:
        """Load and return the state owned by this async collection loader."""
        active_request_id = request_id if request_id is not None else self.begin_request()
        try:
            items = await run_blocking(self._source)
        except BaseException as exc:
            return AsyncCollectionResult(request_id=active_request_id, error=exc)
        return AsyncCollectionResult(request_id=active_request_id, items=tuple(items))
