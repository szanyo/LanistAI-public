# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Describe one TensorBoard comparison-pack entry."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TensorboardRunMetadata:
    """Summarize bounded event-log evidence for one inspection target."""

    duration_seconds: float | None = None
    duration_estimated: bool = False
    started_at_epoch: float | None = None
    finished_at_epoch: float | None = None
    event_file_count: int = 0
    unreadable_event_file_count: int = 0
    log_size_bytes: int = 0
    event_record_count: int = 0
    summary_value_count: int = 0
    chart_count: int = 0
    scalar_tag_count: int = 0
    histogram_tag_count: int = 0
    image_tag_count: int = 0
    text_tag_count: int = 0
    other_tag_count: int = 0
    minimum_step: int | None = None
    maximum_step: int | None = None
    source_run_count: int = 1
    truncated: bool = False
    chart_tags: tuple[str, ...] = ()
    scalar_tags: tuple[str, ...] = ()
    histogram_tags: tuple[str, ...] = ()
    image_tags: tuple[str, ...] = ()
    text_tags: tuple[str, ...] = ()
    other_tags: tuple[str, ...] = ()


@dataclass
class TensorboardPackEntry:
    """Describe tensorboard pack data."""

    pack_type: str
    framework_uuid: str
    run_name: str
    path: str
    search_text: str
    annotation: str | None = None
    source_paths: list[str] | None = None
    model_name: str | None = None
    model_display_name: str | None = None
    metadata: TensorboardRunMetadata | None = None
    recursive_copy: bool = True
