# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""No-op TensorBoard writer for installations without the optional package."""

from __future__ import annotations


class NoopSummaryWriter:
    """Preserve the runtime writer contract without producing artifacts."""

    def __init__(self, log_dir: str | None = None) -> None:
        """Initialize the noop summary writer and its owned state."""
        self.log_dir = log_dir

    def add_scalar(
        self,
        tag: str,
        scalar_value: object,
        global_step: int | None = None,
    ) -> None:
        """Add the scalar owned by this noop summary writer."""
        del tag, scalar_value, global_step

    def add_histogram(
        self,
        tag: str,
        values: object,
        global_step: int | None = None,
    ) -> None:
        """Add the histogram owned by this noop summary writer."""
        del tag, values, global_step

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
    ) -> None:
        """Add the text owned by this noop summary writer."""
        del tag, text_string, global_step

    def add_figure(
        self,
        tag: str,
        figure: object,
        global_step: int | None = None,
    ) -> None:
        """Add the figure owned by this noop summary writer."""
        del tag, figure, global_step

    def flush(self) -> None:
        """Flush the operation owned by this noop summary writer."""
        return None

    def close(self) -> None:
        """Close resources owned by this noop summary writer."""
        return None
