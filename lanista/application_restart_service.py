# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Idempotent process-image restart ownership for the application root."""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol

_RESTART_INTENT_ENVIRONMENT = "LANISTAI_APPLICATION_RESTART_INTENT"
_MAXIMUM_INTENT_LENGTH = 16_384


class ProcessImageExecutor(Protocol):
    """Replace the current process image without returning on success."""

    def __call__(
        self,
        executable: str,
        arguments: Sequence[str],
        environment: Mapping[str, str],
    ) -> object:
        """Invoke this process image executor as a callable."""
        ...


@dataclass(frozen=True, slots=True)
class ApplicationRestartRequest:
    """Opaque, JSON-safe continuation owned by one restart request."""

    reason: str
    continuation: dict[str, object]

    def to_payload(self) -> dict[str, object]:
        """Convert this application restart request's value to payload."""
        return {
            "schema_version": 1,
            "reason": self.reason,
            "continuation": deepcopy(self.continuation),
        }

    @classmethod
    def from_payload(cls, payload: object) -> ApplicationRestartRequest:
        """Create this application restart request's value from payload."""
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("Unsupported application restart intent.")
        reason = payload.get("reason")
        continuation = payload.get("continuation")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Application restart reason is missing.")
        if not isinstance(continuation, dict):
            raise ValueError("Application restart continuation must be a mapping.")
        return cls(reason=reason.strip(), continuation=deepcopy(continuation))


class ApplicationRestartService:
    """Coalesce restart requests and carry one intent through ``execve``."""

    def __init__(self, *, logger: logging.Logger | None = None) -> None:
        """Initialize the application restart service and its owned state."""
        self._logger = logger or logging.getLogger(__name__)
        self._pending: ApplicationRestartRequest | None = None
        self._startup: ApplicationRestartRequest | None = self._read_startup_intent()
        self._startup_consumed = False

    @property
    def restart_pending(self) -> bool:
        """Return the restart pending exposed by this application restart service."""
        return self._pending is not None

    @property
    def pending_request(self) -> ApplicationRestartRequest | None:
        """Expose the intent awaiting process replacement to its application owner."""

        return self._pending

    def request(self, request: ApplicationRestartRequest) -> bool:
        """Register one restart; identical repeated requests are harmless."""

        if self._pending is None:
            self._validate_serializable(request)
            self._pending = request
            return True
        if self._pending == request:
            return False
        raise RuntimeError("A different application restart is already pending.")

    def consume_startup_request(self) -> ApplicationRestartRequest | None:
        """Return the inherited intent once so it cannot create a restart loop."""

        if self._startup_consumed:
            return None
        self._startup_consumed = True
        return self._startup

    def clear_failed_request(self) -> None:
        """Release a pending request after process replacement failed."""

        self._pending = None

    def replace_process_image(
        self,
        *,
        executor: ProcessImageExecutor = os.execve,
        executable: str | None = None,
        arguments: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        """Replace LanistAI with a fresh ``python -m lanista`` process image."""

        request = self._pending
        if request is None:
            raise RuntimeError("Application restart was not requested.")
        serialized = self._serialize(request)
        restart_environment = dict(environment or os.environ)
        restart_environment[_RESTART_INTENT_ENVIRONMENT] = serialized
        restart_executable = executable or sys.executable
        restart_arguments = list(
            arguments
            or (
                restart_executable,
                "-m",
                "lanista",
                *sys.argv[1:],
            )
        )
        executor(
            restart_executable,
            restart_arguments,
            restart_environment,
        )
        raise RuntimeError("Process image replacement returned unexpectedly.")

    def _read_startup_intent(self) -> ApplicationRestartRequest | None:
        """Read the startup intent without changing this application restart service's state."""
        serialized = os.environ.pop(_RESTART_INTENT_ENVIRONMENT, None)
        if not serialized:
            return None
        if len(serialized) > _MAXIMUM_INTENT_LENGTH:
            self._logger.error("Discarded an oversized application restart intent.")
            return None
        try:
            return ApplicationRestartRequest.from_payload(json.loads(serialized))
        except (TypeError, ValueError, json.JSONDecodeError):
            self._logger.exception("Discarded an invalid application restart intent.")
            return None

    @classmethod
    def _validate_serializable(cls, request: ApplicationRestartRequest) -> None:
        """Validate the serializable required by this application restart service."""
        cls._serialize(request)

    @staticmethod
    def _serialize(request: ApplicationRestartRequest) -> str:
        """Serialize the operation owned by this application restart service."""
        serialized = json.dumps(
            request.to_payload(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        if len(serialized) > _MAXIMUM_INTENT_LENGTH:
            raise ValueError("Application restart intent is too large.")
        return serialized


__all__ = [
    "ApplicationRestartRequest",
    "ApplicationRestartService",
]
