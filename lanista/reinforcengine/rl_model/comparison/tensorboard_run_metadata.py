# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Read bounded, cached statistics from TensorBoard event logs."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import (
    TensorboardRunMetadata,
)

_EVENT_FILE_PREFIX = "events.out.tfevents."


@dataclass(frozen=True, slots=True)
class _EventFileRecord:
    """Carry one stable event-file inventory row."""

    path: Path
    relative_path: str
    size_bytes: int
    modified_ns: int
    started_at_epoch: float | None


@dataclass
class _ScanState:
    """Accumulate bounded event records without retaining metric values."""

    started_at_epoch: float | None = None
    finished_at_epoch: float | None = None
    event_record_count: int = 0
    summary_value_count: int = 0
    minimum_step: int | None = None
    maximum_step: int | None = None
    unreadable_event_file_count: int = 0
    truncated: bool = False
    tags: set[str] = field(default_factory=set)
    scalar_tags: set[str] = field(default_factory=set)
    histogram_tags: set[str] = field(default_factory=set)
    image_tags: set[str] = field(default_factory=set)
    text_tags: set[str] = field(default_factory=set)
    other_tags: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _MergedTagStatistics:
    """Carry unique TensorBoard tags across comparison sources."""

    charts: tuple[str, ...]
    scalars: tuple[str, ...]
    histograms: tuple[str, ...]
    images: tuple[str, ...]
    texts: tuple[str, ...]
    others: tuple[str, ...]


class TensorboardRunMetadataReader:
    """Inspect event logs once per unchanged filesystem fingerprint."""

    _MAX_EVENT_FILES = 64
    _MAX_EVENTS_PER_FILE = 64

    def __init__(self) -> None:
        """Initialize an in-memory metadata cache for the application lifetime."""

        self._cache: dict[
            tuple[str, int],
            tuple[tuple[tuple[str, int, int], ...], TensorboardRunMetadata],
        ] = {}

    def summarize_run(
        self,
        log_dir: str,
        *,
        source_run_count: int = 1,
    ) -> TensorboardRunMetadata:
        """Summarize one logical run, including nested event-file segments."""

        root = Path(str(log_dir or "")).absolute()
        inventory = self._event_inventory(root, recursive=True)
        fingerprint = self._fingerprint(inventory)
        cache_key = (str(root), max(0, int(source_run_count)))
        cached = self._cache.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]
        metadata = self._scan_inventory(
            inventory,
            source_run_count=max(0, int(source_run_count)),
        )
        self._cache[cache_key] = (fingerprint, metadata)
        return metadata

    def summarize_direct_run(
        self,
        log_dir: str,
        *,
        source_run_count: int = 1,
    ) -> TensorboardRunMetadata:
        """Summarize only event files stored directly in one run directory."""

        root = Path(str(log_dir or "")).absolute()
        inventory = self._event_inventory(root, recursive=False)
        return self._scan_inventory(
            inventory,
            source_run_count=max(0, int(source_run_count)),
        )

    def summarize_comparison(
        self,
        log_dir: str,
        *,
        declared_source_count: int = 0,
    ) -> TensorboardRunMetadata:
        """Aggregate copied source-run statistics for one comparison pack."""

        root = Path(str(log_dir or "")).absolute()
        summaries = self._comparison_source_summaries(root)
        source_count = max(0, int(declared_source_count)) or len(summaries)
        if summaries:
            return self._aggregate(summaries, source_run_count=source_count)
        return TensorboardRunMetadata(source_run_count=source_count)

    def _comparison_source_summaries(
        self,
        root: Path,
    ) -> tuple[TensorboardRunMetadata, ...]:
        """Return non-empty direct source summaries without following links."""

        if not self._safe_directory(root):
            return ()
        summaries: list[TensorboardRunMetadata] = []
        direct_inventory = self._event_inventory(root, recursive=False)
        if direct_inventory:
            summaries.append(self._scan_inventory(direct_inventory, source_run_count=1))
        try:
            children = sorted(root.iterdir(), key=lambda child: child.name)
        except OSError:
            return tuple(summaries)
        for child in children:
            if not self._safe_directory(child):
                continue
            metadata = self.summarize_run(str(child))
            if metadata.event_file_count:
                summaries.append(metadata)
        return tuple(summaries)

    @classmethod
    def _event_inventory(
        cls,
        root: Path,
        *,
        recursive: bool,
    ) -> tuple[_EventFileRecord, ...]:
        """Inventory safe event files below one directory."""

        if not cls._safe_directory(root):
            return ()
        records: list[_EventFileRecord] = []
        if not recursive:
            cls._append_directory_events(root, root, records)
            return tuple(records)
        for current, directory_names, _file_names in os.walk(
            root,
            topdown=True,
            followlinks=False,
            onerror=lambda _error: None,
        ):
            current_path = Path(current)
            directory_names[:] = [
                name
                for name in directory_names
                if cls._safe_directory(current_path / name)
            ]
            cls._append_directory_events(root, current_path, records)
        records.sort(key=lambda record: record.relative_path)
        return tuple(records)

    @staticmethod
    def _append_directory_events(
        root: Path,
        directory: Path,
        records: list[_EventFileRecord],
    ) -> None:
        """Append safe event files from one already-confined directory."""

        try:
            children = tuple(directory.iterdir())
        except OSError:
            return
        for child in children:
            if not child.name.startswith(_EVENT_FILE_PREFIX) or child.is_symlink():
                continue
            try:
                if not child.is_file():
                    continue
                status = child.stat()
                relative_path = str(child.relative_to(root))
            except (OSError, ValueError):
                continue
            records.append(
                _EventFileRecord(
                    path=child,
                    relative_path=relative_path,
                    size_bytes=max(0, int(status.st_size)),
                    modified_ns=max(0, int(status.st_mtime_ns)),
                    started_at_epoch=TensorboardRunMetadataReader._event_file_started_at(
                        child.name
                    ),
                )
            )

    @classmethod
    def _scan_inventory(
        cls,
        inventory: tuple[_EventFileRecord, ...],
        *,
        source_run_count: int,
    ) -> TensorboardRunMetadata:
        """Parse the bounded inventory into one immutable summary."""

        state = _ScanState()
        selected = cls._bounded_records(inventory)
        state.truncated = len(selected) < len(inventory)
        try:
            from tensorboard.backend.event_processing.event_file_loader import (
                EventFileLoader,
            )
        except ImportError:
            state.unreadable_event_file_count = len(inventory)
            return cls._metadata(inventory, state, source_run_count)
        for record in selected:
            cls._scan_event_file(record, state, EventFileLoader)
        return cls._metadata(inventory, state, source_run_count)

    @classmethod
    def _scan_event_file(
        cls,
        record: _EventFileRecord,
        state: _ScanState,
        loader_type: Any,
    ) -> None:
        """Inspect one event file while isolating corrupt segments."""

        try:
            loader = loader_type(str(record.path))
            for event_index, event in enumerate(loader.Load()):
                if event_index >= cls._MAX_EVENTS_PER_FILE:
                    state.truncated = True
                    break
                cls._include_event(state, event)
        except Exception:
            # Inspection metadata is advisory; one corrupt or concurrently
            # rotated segment must not hide the remaining valid runs.
            state.unreadable_event_file_count += 1

    @classmethod
    def _include_event(cls, state: _ScanState, event: Any) -> None:
        """Accumulate one event record and its summary-value categories."""

        state.event_record_count += 1
        values = tuple(getattr(getattr(event, "summary", None), "value", ()) or ())
        if not values:
            return
        wall_time = cls._finite_float(getattr(event, "wall_time", None))
        if wall_time is not None and wall_time > 0:
            state.started_at_epoch = min(
                wall_time,
                state.started_at_epoch if state.started_at_epoch is not None else wall_time,
            )
            state.finished_at_epoch = max(
                wall_time,
                state.finished_at_epoch if state.finished_at_epoch is not None else wall_time,
            )
        step = max(0, int(getattr(event, "step", 0) or 0))
        state.minimum_step = min(
            step,
            state.minimum_step if state.minimum_step is not None else step,
        )
        state.maximum_step = max(
            step,
            state.maximum_step if state.maximum_step is not None else step,
        )
        for value in values:
            state.summary_value_count += 1
            cls._include_value(state, value)

    @classmethod
    def _include_value(cls, state: _ScanState, value: Any) -> None:
        """Record one unique tag in its TensorBoard plugin category."""

        tag = str(getattr(value, "tag", "") or "").strip()
        if not tag:
            return
        state.tags.add(tag)
        plugin_data = getattr(getattr(value, "metadata", None), "plugin_data", None)
        plugin_name = str(getattr(plugin_data, "plugin_name", "") or "").lower()
        if plugin_name in {"scalar", "scalars"} or cls._has_field(value, "simple_value"):
            state.scalar_tags.add(tag)
        elif plugin_name in {"histogram", "histograms", "distributions"} or cls._has_field(
            value,
            "histo",
        ):
            state.histogram_tags.add(tag)
        elif plugin_name in {"image", "images"} or cls._has_field(value, "image"):
            state.image_tags.add(tag)
        elif plugin_name == "text":
            state.text_tags.add(tag)
        else:
            state.other_tags.add(tag)

    @classmethod
    def _metadata(
        cls,
        inventory: tuple[_EventFileRecord, ...],
        state: _ScanState,
        source_run_count: int,
    ) -> TensorboardRunMetadata:
        """Freeze accumulated state with complete filesystem statistics."""

        started_at, finished_at, duration, estimated = cls._time_bounds(
            inventory,
            state,
        )
        return TensorboardRunMetadata(
            duration_seconds=duration,
            duration_estimated=estimated,
            started_at_epoch=started_at,
            finished_at_epoch=finished_at,
            event_file_count=len(inventory),
            unreadable_event_file_count=state.unreadable_event_file_count,
            log_size_bytes=sum(record.size_bytes for record in inventory),
            event_record_count=state.event_record_count,
            summary_value_count=state.summary_value_count,
            chart_count=len(state.tags),
            scalar_tag_count=len(state.scalar_tags),
            histogram_tag_count=len(state.histogram_tags),
            image_tag_count=len(state.image_tags),
            text_tag_count=len(state.text_tags),
            other_tag_count=len(state.other_tags),
            minimum_step=state.minimum_step,
            maximum_step=state.maximum_step,
            source_run_count=source_run_count,
            truncated=state.truncated,
            chart_tags=tuple(sorted(state.tags)),
            scalar_tags=tuple(sorted(state.scalar_tags)),
            histogram_tags=tuple(sorted(state.histogram_tags)),
            image_tags=tuple(sorted(state.image_tags)),
            text_tags=tuple(sorted(state.text_tags)),
            other_tags=tuple(sorted(state.other_tags)),
        )

    @classmethod
    def _time_bounds(
        cls,
        inventory: tuple[_EventFileRecord, ...],
        state: _ScanState,
    ) -> tuple[float | None, float | None, float | None, bool]:
        """Prefer exact event times and use the file envelope for bounded scans."""

        started_at = state.started_at_epoch
        finished_at = state.finished_at_epoch
        estimated = False
        if state.truncated and inventory:
            substantial = tuple(record for record in inventory if record.size_bytes > 88)
            evidence = substantial or inventory
            file_starts = [
                record.started_at_epoch
                for record in evidence
                if record.started_at_epoch is not None
            ]
            file_finishes = [
                record.modified_ns / 1_000_000_000
                for record in evidence
                if record.modified_ns > 0
            ]
            if file_starts:
                started_at = min([*file_starts, *([started_at] if started_at else [])])
            if file_finishes:
                finished_at = max(
                    [*file_finishes, *([finished_at] if finished_at else [])]
                )
            estimated = bool(file_starts or file_finishes)
        duration = None
        if started_at is not None and finished_at is not None:
            duration = max(0.0, finished_at - started_at)
        return started_at, finished_at, duration, estimated

    @classmethod
    def _aggregate(
        cls,
        summaries: tuple[TensorboardRunMetadata, ...],
        *,
        source_run_count: int,
    ) -> TensorboardRunMetadata:
        """Aggregate copied source summaries without inventing a wall-time span."""

        durations = [
            metadata.duration_seconds
            for metadata in summaries
            if metadata.duration_seconds is not None
        ]
        starts = [
            metadata.started_at_epoch
            for metadata in summaries
            if metadata.started_at_epoch is not None
        ]
        finishes = [
            metadata.finished_at_epoch
            for metadata in summaries
            if metadata.finished_at_epoch is not None
        ]
        minimum_steps = [
            metadata.minimum_step
            for metadata in summaries
            if metadata.minimum_step is not None
        ]
        maximum_steps = [
            metadata.maximum_step
            for metadata in summaries
            if metadata.maximum_step is not None
        ]
        tags = cls._aggregate_tags(summaries)
        return TensorboardRunMetadata(
            duration_seconds=sum(durations) if durations else None,
            duration_estimated=any(
                metadata.duration_estimated for metadata in summaries
            ),
            started_at_epoch=min(starts) if starts else None,
            finished_at_epoch=max(finishes) if finishes else None,
            event_file_count=sum(metadata.event_file_count for metadata in summaries),
            unreadable_event_file_count=sum(
                metadata.unreadable_event_file_count for metadata in summaries
            ),
            log_size_bytes=sum(metadata.log_size_bytes for metadata in summaries),
            event_record_count=sum(metadata.event_record_count for metadata in summaries),
            summary_value_count=sum(metadata.summary_value_count for metadata in summaries),
            chart_count=cls._merged_count(tags.charts, summaries, "chart_count"),
            scalar_tag_count=cls._merged_count(
                tags.scalars,
                summaries,
                "scalar_tag_count",
            ),
            histogram_tag_count=cls._merged_count(
                tags.histograms,
                summaries,
                "histogram_tag_count",
            ),
            image_tag_count=cls._merged_count(tags.images, summaries, "image_tag_count"),
            text_tag_count=cls._merged_count(tags.texts, summaries, "text_tag_count"),
            other_tag_count=cls._merged_count(tags.others, summaries, "other_tag_count"),
            minimum_step=min(minimum_steps) if minimum_steps else None,
            maximum_step=max(maximum_steps) if maximum_steps else None,
            source_run_count=source_run_count,
            truncated=any(metadata.truncated for metadata in summaries),
            chart_tags=tags.charts,
            scalar_tags=tags.scalars,
            histogram_tags=tags.histograms,
            image_tags=tags.images,
            text_tags=tags.texts,
            other_tags=tags.others,
        )

    @classmethod
    def _aggregate_tags(
        cls,
        summaries: tuple[TensorboardRunMetadata, ...],
    ) -> _MergedTagStatistics:
        """Merge every tag category across comparison source runs."""

        return _MergedTagStatistics(
            charts=cls._merged_tags(summaries, "chart_tags"),
            scalars=cls._merged_tags(summaries, "scalar_tags"),
            histograms=cls._merged_tags(summaries, "histogram_tags"),
            images=cls._merged_tags(summaries, "image_tags"),
            texts=cls._merged_tags(summaries, "text_tags"),
            others=cls._merged_tags(summaries, "other_tags"),
        )

    @staticmethod
    def _merged_count(
        tags: tuple[str, ...],
        summaries: tuple[TensorboardRunMetadata, ...],
        count_attribute: str,
    ) -> int:
        """Use unique tags when available and legacy counts otherwise."""

        if tags:
            return len(tags)
        return sum(int(getattr(metadata, count_attribute)) for metadata in summaries)

    @staticmethod
    def _merged_tags(
        summaries: tuple[TensorboardRunMetadata, ...],
        attribute: str,
    ) -> tuple[str, ...]:
        """Merge a tag-identity field across comparison source runs."""

        return tuple(
            sorted(
                {
                    tag
                    for metadata in summaries
                    for tag in getattr(metadata, attribute)
                }
            )
        )

    @classmethod
    def _bounded_records(
        cls,
        inventory: tuple[_EventFileRecord, ...],
    ) -> tuple[_EventFileRecord, ...]:
        """Retain oldest and newest segments when the file limit is exceeded."""

        if len(inventory) <= cls._MAX_EVENT_FILES:
            return inventory
        ordered = sorted(
            inventory,
            key=lambda record: (record.modified_ns, record.relative_path),
        )
        oldest_count = cls._MAX_EVENT_FILES // 2
        selected = ordered[:oldest_count] + ordered[-(cls._MAX_EVENT_FILES - oldest_count) :]
        return tuple(selected)

    @staticmethod
    def _fingerprint(
        inventory: tuple[_EventFileRecord, ...],
    ) -> tuple[tuple[str, int, int], ...]:
        """Return the cache key material for one event-file inventory."""

        return tuple(
            (record.relative_path, record.size_bytes, record.modified_ns)
            for record in inventory
        )

    @staticmethod
    def _event_file_started_at(file_name: str) -> float | None:
        """Read the writer-start epoch embedded in a TensorBoard event name."""

        suffix = str(file_name or "").removeprefix(_EVENT_FILE_PREFIX)
        raw_epoch = suffix.split(".", maxsplit=1)[0]
        try:
            epoch = float(raw_epoch)
        except (TypeError, ValueError):
            return None
        return epoch if math.isfinite(epoch) and epoch > 0 else None

    @staticmethod
    def _safe_directory(path: Path) -> bool:
        """Return whether a directory can be inspected without following a link."""

        try:
            return path.is_dir() and not path.is_symlink()
        except OSError:
            return False

    @staticmethod
    def _finite_float(value: object) -> float | None:
        """Return one finite floating-point value."""

        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    @staticmethod
    def _has_field(value: Any, field_name: str) -> bool:
        """Safely query an optional protobuf oneof field."""

        try:
            return bool(value.HasField(field_name))
        except (AttributeError, TypeError, ValueError):
            return False


__all__ = ["TensorboardRunMetadataReader"]
