# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed boundaries for optional raw diagnostic traces."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TextIO

DiagnosticTraceRow = Mapping[str, object]
QualityReporter = Callable[[str, BaseException], None]


class DiagnosticTraceSource(Protocol):
    """Expose the latest already-composed diagnostic row."""

    def latest_diagnostic_trace_row(
        self,
        *,
        scenario_index: int | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> DiagnosticTraceRow | None:
        """Return the latest completed simulator-step trace row."""


class DiagnosticTraceTensorboardWriter(Protocol):
    """Publish one raw diagnostic scalar."""

    def add_scalar(self, tag: str, value: object, step: int) -> object:
        """Write one scalar sample."""


class DiagnosticTraceFile(Protocol):
    """File operations owned by a diagnostic trace run."""

    def write(self, data: str) -> int:
        """Write text to the trace file."""

    def flush(self) -> None:
        """Flush buffered trace text."""

    def close(self) -> None:
        """Close the trace file."""


TraceFileOpener = Callable[[str], TextIO]


__all__ = [
    "DiagnosticTraceFile",
    "DiagnosticTraceRow",
    "DiagnosticTraceSource",
    "DiagnosticTraceTensorboardWriter",
    "QualityReporter",
    "TraceFileOpener",
]
