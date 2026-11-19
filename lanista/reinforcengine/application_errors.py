# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed, transport-neutral results for application boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4


class FailureCode(StrEnum):
    """Stable failure identifiers exposed to adapters and user interfaces."""

    VALIDATION_ERROR = "validation_error"
    RESOURCE_NOT_FOUND = "resource_not_found"
    RUNTIME_UNAVAILABLE = "runtime_unavailable"
    PERSISTENCE_ERROR = "persistence_error"
    EXTERNAL_TOOL_ERROR = "external_tool_error"
    OPERATION_CANCELLED = "operation_cancelled"
    OPERATION_CONFLICT = "operation_conflict"
    OPERATION_TIMEOUT = "operation_timeout"
    UNEXPECTED_ERROR = "unexpected_error"


class RetryPolicy(StrEnum):
    """Whether a caller may explicitly retry an idempotent operation."""

    NEVER = "never"
    SAFE = "safe"


class ApplicationError(Exception):
    """Base class for expected, user-presentable application failures."""


class ApplicationValidationError(ApplicationError):
    """Input or plugin configuration violates a public contract."""


class ResourceNotFoundError(ApplicationError):
    """A requested first-party resource does not exist."""


class RuntimeUnavailableError(ApplicationError):
    """A configured runtime cannot currently be used."""


class PersistenceError(ApplicationError):
    """Durable state could not be read or committed."""


class ExternalToolError(ApplicationError):
    """A required external process or native adapter failed."""


class OperationCancelled(ApplicationError):
    """The operation was superseded or explicitly cancelled."""


class OperationConflict(ApplicationError):
    """The operation conflicts with another active state transition."""


class OperationTimeout(ApplicationError):
    """The operation did not finish within its bounded deadline."""


@dataclass(frozen=True)
class OperationFailure:
    """Safe failure details returned across an application boundary."""

    code: FailureCode
    message: str
    retry_policy: RetryPolicy
    operation_id: str

    @property
    def retryable(self) -> bool:
        """Retain a convenient predicate without leaking exception types."""

        return self.retry_policy is RetryPolicy.SAFE


@dataclass(frozen=True)
class OperationResult[ResultT]:
    """A successful value or a stable failure, correlated by operation ID."""

    operation_id: str
    value: ResultT | None = None
    failure: OperationFailure | None = None

    @property
    def succeeded(self) -> bool:
        """Return the succeeded exposed by this operation result."""
        return self.failure is None

    @classmethod
    def success(cls, value: ResultT, *, operation_id: str | None = None) -> OperationResult[ResultT]:
        """Return the success value owned by this operation result."""
        return cls(operation_id=operation_id or uuid4().hex, value=value)

    @classmethod
    def failed(cls, failure: OperationFailure) -> OperationResult[ResultT]:
        """Return the failed value owned by this operation result."""
        return cls(operation_id=failure.operation_id, failure=failure)


_FAILURE_CODE_BY_TYPE: tuple[tuple[type[BaseException], FailureCode], ...] = (
    (ApplicationValidationError, FailureCode.VALIDATION_ERROR),
    (ResourceNotFoundError, FailureCode.RESOURCE_NOT_FOUND),
    (RuntimeUnavailableError, FailureCode.RUNTIME_UNAVAILABLE),
    (PersistenceError, FailureCode.PERSISTENCE_ERROR),
    (ExternalToolError, FailureCode.EXTERNAL_TOOL_ERROR),
    (OperationCancelled, FailureCode.OPERATION_CANCELLED),
    (OperationConflict, FailureCode.OPERATION_CONFLICT),
    (OperationTimeout, FailureCode.OPERATION_TIMEOUT),
)


def failure_from_exception(
    error: BaseException,
    *,
    operation_id: str | None = None,
    retry_policy: RetryPolicy | None = None,
) -> OperationFailure:
    """Convert an exception to safe boundary data.

    Unexpected exceptions deliberately receive a generic message. The original
    exception and traceback belong in structured logs, not in UI-facing data.
    """

    code = next(
        (candidate for error_type, candidate in _FAILURE_CODE_BY_TYPE if isinstance(error, error_type)),
        FailureCode.UNEXPECTED_ERROR,
    )
    default_retry_policy = (
        RetryPolicy.SAFE
        if isinstance(error, RuntimeUnavailableError | ExternalToolError)
        else RetryPolicy.NEVER
    )
    safe_message = (
        str(error)
        if isinstance(error, ApplicationError)
        else "The operation failed unexpectedly. See the application log for details."
    )
    return OperationFailure(
        code=code,
        message=safe_message,
        retry_policy=retry_policy or default_retry_policy,
        operation_id=operation_id or uuid4().hex,
    )


def exception_from_failure(failure: OperationFailure) -> ApplicationError:
    """Rehydrate a safe expected error for an internal orchestration boundary."""

    error_type_by_code: dict[FailureCode, type[ApplicationError]] = {
        FailureCode.VALIDATION_ERROR: ApplicationValidationError,
        FailureCode.RESOURCE_NOT_FOUND: ResourceNotFoundError,
        FailureCode.RUNTIME_UNAVAILABLE: RuntimeUnavailableError,
        FailureCode.PERSISTENCE_ERROR: PersistenceError,
        FailureCode.EXTERNAL_TOOL_ERROR: ExternalToolError,
        FailureCode.OPERATION_CANCELLED: OperationCancelled,
        FailureCode.OPERATION_CONFLICT: OperationConflict,
        FailureCode.OPERATION_TIMEOUT: OperationTimeout,
    }
    error_type = error_type_by_code.get(failure.code, RuntimeUnavailableError)
    return error_type(failure.message)
