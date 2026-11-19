# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Standard text-stream metadata for Textual's captured output streams."""

from __future__ import annotations

import io
from typing import Protocol


class _CapturedTextStream(Protocol):
    """Define the typed boundary for captured text stream."""

    def write(self, text: str) -> object:
        """Return the write value provided by this captured text stream."""
        ...

    def flush(self) -> object:
        """Flush the operation owned by this captured text stream."""
        ...

    def isatty(self) -> bool:
        """Return the isatty value provided by this captured text stream."""
        ...


class _DescriptorTextStream(Protocol):
    """Define the typed boundary for descriptor text stream."""

    @property
    def encoding(self) -> str | None:
        """Return the encoding exposed by this descriptor text stream."""
        ...

    @property
    def errors(self) -> str | None:
        """Return the errors exposed by this descriptor text stream."""
        ...

    def fileno(self) -> int:
        """Return the fileno value provided by this descriptor text stream."""
        ...


class TextualCaptureStreamAdapter:
    """Keep Textual capture while exposing a real terminal stream contract."""

    def __init__(
        self,
        capture_stream: _CapturedTextStream,
        descriptor_stream: _DescriptorTextStream | None,
    ) -> None:
        """Initialize the textual capture stream adapter and its owned state."""
        self._capture_stream = capture_stream
        self._descriptor_stream = descriptor_stream

    @property
    def encoding(self) -> str:
        """Return the encoding exposed by this textual capture stream adapter."""
        if self._descriptor_stream is None:
            return "utf-8"
        return self._descriptor_stream.encoding or "utf-8"

    @property
    def errors(self) -> str:
        """Return the errors exposed by this textual capture stream adapter."""
        if self._descriptor_stream is None:
            return "strict"
        return self._descriptor_stream.errors or "strict"

    def write(self, text: str) -> int:
        """Return the write value owned by this textual capture stream adapter."""
        self._capture_stream.write(text)
        return len(text)

    def flush(self) -> None:
        """Flush the operation owned by this textual capture stream adapter."""
        self._capture_stream.flush()

    def isatty(self) -> bool:
        """Return the isatty value owned by this textual capture stream adapter."""
        return self._capture_stream.isatty()

    def fileno(self) -> int:
        """Return the fileno value owned by this textual capture stream adapter."""
        if self._descriptor_stream is None:
            raise io.UnsupportedOperation("captured stream has no file descriptor")
        return self._descriptor_stream.fileno()
