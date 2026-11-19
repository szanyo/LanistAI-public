# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Deterministic cleanup for framework logic, algorithms, and environments."""

from __future__ import annotations

from collections.abc import Callable


class FrameworkCleanupError(RuntimeError):
    """Raised after all owned resources were attempted but one or more failed."""


def close_framework_logic(
    logic: object | None,
    *,
    completed_resource_ids: set[int] | None = None,
) -> None:
    """Close each owned framework resource once, including across retries.

    ``completed_resource_ids`` is owned by the runtime and intentionally
    survives a failed cleanup attempt.  A later recovery pass therefore
    retries only the resources whose close hook actually failed.
    """

    if logic is None:
        return

    failures: list[Exception] = []
    completed = completed_resource_ids if completed_resource_ids is not None else set()
    attempted_ids: set[int] = set()
    candidates = (
        getattr(logic, "_algo", None),
        getattr(logic, "_env", None),
        getattr(logic, "_worker_pool", None),
        logic,
    )
    for resource in candidates:
        if resource is None:
            continue
        resource_id = id(resource)
        if resource_id in attempted_ids or resource_id in completed:
            continue
        attempted_ids.add(resource_id)
        close_method = _resolve_close_method(resource)
        if close_method is None:
            completed.add(resource_id)
            continue
        try:
            close_method()
        except Exception as exc:
            failures.append(exc)
        else:
            completed.add(resource_id)

    if failures:
        raise FrameworkCleanupError(
            f"{len(failures)} framework resource cleanup operation(s) failed."
        ) from failures[0]


def _resolve_close_method(resource: object) -> Callable[[], object] | None:
    """Resolve the close method required by this runtime cleanup module."""
    for method_name in ("close", "stop", "shutdown"):
        method = getattr(resource, method_name, None)
        if callable(method):
            return method
    return None
