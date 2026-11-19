# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Application-owned logging adapters and decision-log value objects."""

from __future__ import annotations

import logging
import logging.config
import re
import warnings
from dataclasses import dataclass
from queue import Queue
from typing import Any

import yaml
from rich.text import Text


@dataclass(frozen=True)
class LoggerConditionCheck:
    """Describe one compact sub-check in a compound logger condition."""

    label: str
    lhs_val: Any
    operator: str
    rhs_val: Any
    result: bool

    def format(self, label_width: int) -> str:
        """Format this sub-check as one aligned diagnostic row."""

        result = "True" if self.result else "False"
        return (
            f"{self.label:<{label_width}}  {self.lhs_val} {self.operator} {self.rhs_val} → {result}"
        )


@dataclass(frozen=True)
class LoggerCondition:
    """Carry the data required for logger condition."""

    lhs_name: str
    lhs_val: Any
    operator: str
    rhs_name: str
    rhs_val: Any
    result: bool
    checks: tuple[LoggerConditionCheck, ...] = ()

    def format(self) -> str:
        """Format this logger condition's value for presentation."""
        result = "True" if self.result else "False"
        return (
            f"{self.lhs_name}={self.lhs_val} {self.operator} "
            f"{self.rhs_name}={self.rhs_val} → {result}"
        )

    def format_lines(self, *, scope: str | None = None) -> tuple[str, ...]:
        """Render one condition as a compact, consistently aligned block."""

        result = "True" if self.result else "False"
        display_name = self._display_name(scope)
        if not self.checks:
            return (f"{display_name}: {self.lhs_val} {self.operator} {self.rhs_val} → {result}",)
        label_width = max(len(check.label) for check in self.checks)
        return (
            f"{display_name} → {result}",
            *(check.format(label_width) for check in self.checks),
        )

    def _display_name(self, scope: str | None) -> str:
        """Return a short human-facing name without losing window scope."""

        name = self.lhs_name
        if scope and name.startswith(f"{scope}."):
            name = name[len(scope) + 1 :]
        name = name.removesuffix(".preferred_gate")
        parts = [part.replace("_", " ") for part in name.split(".")]
        if parts and parts[0] == "window":
            parts[0] = "main window"
        if ".acceptable_range." in self.rhs_name:
            parts.append("acceptable")
        elif self.checks:
            parts.append("preferred")
        return " · ".join(parts)


class LoggerInterface:
    """Mixin that consistently creates and configures a component logger."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the logger interface and its owned state."""
        parent_name = kwargs.get("parent_logger_name")
        child_name = kwargs.get("child_logger_name")
        level = kwargs.get("child_logger_level", logging.DEBUG)
        config_location = kwargs.get("logger_interface_configuration")
        self._logger_parent = str(parent_name) if parent_name is not None else None
        self._logger_name = str(child_name) if child_name is not None else None
        self._logger_level = self._coerce_level(level)
        self._logger_configuration_location = (
            str(config_location) if config_location is not None else None
        )
        logger_name = self._resolve_logger_name()
        self._logger: Any = logging.getLogger(logger_name)
        self._load_logger_configuration()
        self._logger.setLevel(self._logger_level)
        self._logger.debug("Logger interface initialized for %s.", logger_name)

    def _resolve_logger_name(self) -> str:
        """Resolve the logger name required by this logger interface."""
        if self._logger_parent and self._logger_name:
            return f"{self._logger_parent}.{self._logger_name}"
        if self._logger_name:
            return self._logger_name
        return "lanista"

    @staticmethod
    def _coerce_level(level: object) -> int:
        """Coerce the level owned by this logger interface."""
        if isinstance(level, int):
            return level
        if isinstance(level, str):
            resolved = logging.getLevelNamesMapping().get(level.upper())
            if resolved is not None:
                return resolved
        return logging.DEBUG

    def _load_logger_configuration(self) -> None:
        """Load the logger configuration owned by this logger interface."""
        location = self._logger_configuration_location
        if location is None:
            return
        with open(location, encoding="utf-8") as stream:
            configuration = yaml.safe_load(stream)
        if not isinstance(configuration, dict):
            raise ValueError("Logger configuration must be a mapping.")
        logging.config.dictConfig(configuration)

    def log_decision(
        self,
        method: str,
        conditions: list[LoggerCondition],
        decision: bool,
    ) -> None:
        """Log the decision owned by this logger interface."""
        scope = method.rsplit(".", maxsplit=1)[-1]
        lines = [f"[{method}] Conditions"]
        for index, condition in enumerate(conditions, start=1):
            condition_lines = condition.format_lines(scope=scope)
            lines.append(f"    {index:02}. {condition_lines[0]}")
            lines.extend(f"        {line}" for line in condition_lines[1:])
        lines.append(f"    Result → {decision}")
        self._logger.info("\n".join(lines))


class DictFormatter(logging.Formatter):
    """Project a log record into the dictionary expected by GUI consumers."""

    def format(self, record: logging.LogRecord) -> dict[str, object]:  # type: ignore[override]
        """Format this dict formatter's value for presentation."""
        super().format(record)
        if not hasattr(record, "asctime"):
            record.asctime = self.formatTime(record, self.datefmt)
        return dict(record.__dict__)


class RawFormatter(logging.Formatter):
    """Return the original log record for an early in-memory pipeline."""

    def format(self, record: logging.LogRecord) -> logging.LogRecord:  # type: ignore[override]
        """Format this raw formatter's value for presentation."""
        super().format(record)
        if not hasattr(record, "asctime"):
            record.asctime = self.formatTime(record, self.datefmt)
        return record


class DecisionFormatter(logging.Formatter):
    """Remove Rich markup before writing decision logs to plain files."""

    def format(self, record: logging.LogRecord) -> str:
        """Format this decision formatter's value for presentation."""
        formatted = super().format(record)
        try:
            return Text.from_markup(formatted).plain
        except Exception:
            return re.sub(r"\[/?[^\]\[]+\]", "", formatted)


class PipelineHandler(logging.Handler):
    """Buffer formatted records until a consumer queue is attached."""

    def __init__(self, level: int = logging.NOTSET) -> None:
        """Initialize the pipeline handler and its owned state."""
        super().__init__(level)
        self._temporal_buffer_queue: Queue[object] = Queue()
        self._pipeline: Queue[object] | None = None

    def set_pipeline(self, pipeline: Queue[object]) -> None:
        """Set the pipeline owned by this pipeline handler."""
        self._pipeline = pipeline
        self._flush_temporal_buffer()

    def get_pipeline(self) -> Queue[object] | None:
        """Return the pipeline exposed by this pipeline handler."""
        return self._pipeline

    def _flush_temporal_buffer(self) -> None:
        """Flush the temporal buffer owned by this pipeline handler."""
        if self._pipeline is None:
            return
        while not self._temporal_buffer_queue.empty():
            self._pipeline.put(self._temporal_buffer_queue.get())

    def emit(self, record: logging.LogRecord) -> None:
        """Publish one record through this pipeline handler."""
        try:
            formatted = self.format(record)
            if self._pipeline is None:
                self._temporal_buffer_queue.put(formatted)
                return
            self._flush_temporal_buffer()
            self._pipeline.put(formatted)
        except Exception:
            self.handleError(record)


class CustomWarning(LoggerInterface):
    """Route Python warnings through the configured application logger."""

    def __init__(self, **kwargs: object) -> None:
        """Initialize the custom warning and its owned state."""
        super().__init__(**kwargs)
        warnings.showwarning = self._show_warning
        warnings.filterwarnings("always")

    def _show_warning(
        self,
        message: Warning | str,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: object | None = None,
        line: str | None = None,
    ) -> None:
        """Show the warning owned by this custom warning."""
        del file
        self._logger.warning(warnings.formatwarning(message, category, filename, lineno, line))
