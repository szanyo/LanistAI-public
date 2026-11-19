# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""TensorBoard log-layout, writer construction, and cleanup ownership."""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, cast
from uuid import uuid4

from lanista.reinforcengine.rl_model.runtime.tensorboard_fallback import (
    NoopSummaryWriter,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_layout_service import (
    TensorboardLayoutLogger,
    TensorboardLayoutMode,
    TensorboardRuntimeLayoutRequest,
    TensorboardRuntimeLayoutService,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_step_continuity import (
    TensorboardStepContinuityService,
)

__all__ = [
    "SummaryWriterFactory",
    "TensorboardWriter",
    "TensorboardWriterOwner",
    "build_tensorboard_writer",
]


class TensorboardWriter(Protocol):
    """Metric operations and resource hooks required from runtime writers."""

    def add_scalar(
        self,
        tag: str,
        scalar_value: object,
        global_step: int | None = None,
    ) -> object:
        """Add the scalar owned by this tensorboard writer."""
        ...

    def add_histogram(
        self,
        tag: str,
        values: object,
        global_step: int | None = None,
    ) -> object:
        """Add the histogram owned by this tensorboard writer."""
        ...

    def add_text(
        self,
        tag: str,
        text_string: str,
        global_step: int | None = None,
    ) -> object:
        """Add the text owned by this tensorboard writer."""
        ...

    def add_figure(
        self,
        tag: str,
        figure: object,
        global_step: int | None = None,
    ) -> object:
        """Add the figure owned by this tensorboard writer."""
        ...

    def flush(self) -> object:
        """Flush the operation owned by this tensorboard writer."""
        ...

    def close(self) -> object:
        """Close resources owned by this tensorboard writer."""
        ...


def build_tensorboard_writer(*, log_dir: str) -> TensorboardWriter:
    """Build the default writer required by this runtime tensorboard writer module."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        return NoopSummaryWriter(log_dir=log_dir)
    return cast(
        TensorboardWriter,
        SummaryWriter(log_dir=log_dir),  # type: ignore[no-untyped-call]
    )


class SummaryWriterFactory(Protocol):
    """Construct a writer bound to one already-resolved run directory."""

    def __call__(self, *, log_dir: str) -> TensorboardWriter:
        """Invoke this summary writer factory as a callable."""
        ...


class RuntimeLogDirResolver(Protocol):
    """Resolve one runtime writer path without taking writer ownership."""

    def resolve_runtime_log_dir(
        self,
        request: TensorboardRuntimeLayoutRequest,
        logger: TensorboardLayoutLogger,
    ) -> str | None:
        """Resolve the runtime log dir required by this runtime log dir resolver."""
        ...


class RuntimeTensorboardConfiguration(Protocol):
    """Configuration fields required to resolve one runtime writer path."""

    model_folder_path: str
    selected_default_model_name: str
    rl_engine_framework: object


class TensorboardWriterOwner:
    """Own one lazy runtime writer and its retryable flush/close lifecycle."""

    def __init__(
        self,
        *,
        configuration_provider: Callable[[], object],
        layout_mode_provider: Callable[[], TensorboardLayoutMode],
        logger_provider: Callable[[], TensorboardLayoutLogger],
        layout_service: RuntimeLogDirResolver | None = None,
        writer_factory: SummaryWriterFactory = build_tensorboard_writer,
        step_continuity: TensorboardStepContinuityService | None = None,
    ) -> None:
        """Initialize the tensorboard writer owner and its owned state."""
        self._configuration_provider = configuration_provider
        self._layout_mode_provider = layout_mode_provider
        self._logger_provider = logger_provider
        self._layout_service = layout_service or TensorboardRuntimeLayoutService()
        self._writer_factory = writer_factory
        self._step_continuity = step_continuity or TensorboardStepContinuityService()
        self._writer: TensorboardWriter | None = None
        self._flush_completed = False
        self._step_floor = 0
        self._evaluation_run_name: str | None = None

    @staticmethod
    def _new_evaluation_run_name() -> str:
        """Return a sortable, collision-resistant safe directory component."""

        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        return f"eval-{timestamp}-{uuid4().hex[:8]}"

    def resolve_log_dir(self) -> str | None:
        """Return the session-specific writer directory for the active mode."""

        configuration = cast(
            RuntimeTensorboardConfiguration,
            self._configuration_provider(),
        )
        selected_session_name = str(getattr(configuration, "selected_default_model_name", "") or "")
        if not selected_session_name:
            return None

        framework_configuration = getattr(
            configuration,
            "rl_engine_framework",
            None,
        )
        framework_uuid = str(getattr(framework_configuration, "uuid", "") or "").strip().lower()
        layout_mode = self._layout_mode_provider()
        if layout_mode is TensorboardLayoutMode.EVALUATION:
            if self._evaluation_run_name is None:
                self._evaluation_run_name = self._new_evaluation_run_name()
        else:
            self._evaluation_run_name = None
        log_dir = self._layout_service.resolve_runtime_log_dir(
            TensorboardRuntimeLayoutRequest(
                model_folder_path=str(configuration.model_folder_path),
                session_name=selected_session_name,
                framework_uuid=framework_uuid,
                mode=layout_mode,
                evaluation_run_name=self._evaluation_run_name,
            ),
            self._logger_provider(),
        )
        if log_dir:
            self._prepare_step_continuity(
                log_dir,
                framework_uuid=framework_uuid,
            )
        return log_dir

    def _prepare_step_continuity(
        self,
        log_dir: str,
        *,
        framework_uuid: str,
    ) -> None:
        """Publish the recovered session floor before a framework opens its writer."""

        self._step_floor = self._step_continuity.last_committed_step(log_dir)
        if framework_uuid == "ray":
            os.environ["LANISTAI_TENSORBOARD_STEP_FLOOR"] = str(self._step_floor)
        else:
            os.environ.pop("LANISTAI_TENSORBOARD_STEP_FLOOR", None)

    def step_floor(self) -> int:
        """Return the last recovered committed step for the resolved session."""

        return self._step_floor

    def refresh(self) -> None:
        """Replace the writer after the selected run directory changes."""

        if self._layout_mode_provider() is TensorboardLayoutMode.EVALUATION:
            self._evaluation_run_name = self._new_evaluation_run_name()
        log_dir = self.resolve_log_dir()
        if not log_dir:
            return

        os.environ["LANISTAI_TENSORBOARD_LOGDIR"] = log_dir
        if self._writer is not None:
            self._writer.close()
        self._writer = self._writer_factory(log_dir=log_dir)
        self._flush_completed = False

    def get_or_create(self) -> TensorboardWriter | None:
        """Return the current writer, creating it lazily when possible."""

        if self._writer is not None:
            return self._writer

        log_dir = self.resolve_log_dir()
        if not log_dir:
            return None

        os.environ["LANISTAI_TENSORBOARD_LOGDIR"] = log_dir
        self._writer = self._writer_factory(log_dir=log_dir)
        self._flush_completed = False
        return self._writer

    def require(self) -> TensorboardWriter:
        """Return a writer or fail when no session can own its log directory."""

        writer = self.get_or_create()
        if writer is None:
            raise RuntimeError("TensorBoard writer unavailable: no selected session configured.")
        return writer

    def current(self) -> TensorboardWriter | None:
        """Return the existing writer without creating a new resource."""

        return self._writer

    def reset_cleanup_state(self) -> None:
        """Allow a new execution run to flush its current writer once."""

        self._flush_completed = False

    def cleanup(self) -> list[BaseException]:
        """Flush before close and retain only the failed step for retry."""

        failures: list[BaseException] = []
        writer = self._writer
        if writer is None:
            return failures
        if not self._flush_completed:
            try:
                writer.flush()
            except BaseException as error:
                failures.append(error)
            else:
                self._flush_completed = True
        if self._flush_completed:
            try:
                writer.close()
            except BaseException as error:
                failures.append(error)
            else:
                self._writer = None
                self._flush_completed = False
        return failures
