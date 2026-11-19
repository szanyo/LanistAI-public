# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Project TensorBoard event statistics into compact GUI text."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import (
    TensorboardPackEntry,
)


@dataclass(frozen=True, slots=True)
class TensorboardMetadataDisplayRow:
    """Carry one presentation-neutral details-panel row."""

    label: str
    value: str
    value_style: str = "bold"


def compact_tensorboard_metadata(entry: TensorboardPackEntry) -> str:
    """Return the short statistics suffix used by inspection trees."""

    metadata = entry.metadata
    if metadata is None:
        return ""
    if entry.pack_type != "comparison" and metadata.event_file_count == 0:
        return ""
    parts: list[str] = []
    if entry.pack_type == "comparison":
        source_count = max(metadata.source_run_count, len(entry.source_paths or ()))
        parts.append(f"{source_count} runs")
    duration = format_duration(metadata.duration_seconds)
    parts.append(f"~{duration}" if metadata.duration_estimated and duration != "-" else duration)
    chart_prefix = "≥" if metadata.truncated else ""
    parts.append(f"{chart_prefix}{metadata.chart_count} charts")
    if entry.pack_type != "comparison" and metadata.maximum_step is not None:
        step_prefix = "≥" if metadata.truncated else ""
        parts.append(
            f"step {step_prefix}{format_compact_number(metadata.maximum_step)}"
        )
    return " · ".join(parts)


def tensorboard_metadata_rows(
    entry: TensorboardPackEntry,
) -> tuple[TensorboardMetadataDisplayRow, ...]:
    """Return shared detailed statistics for one inspection target."""

    metadata = entry.metadata
    if metadata is None:
        return ()
    rows = list(_primary_metadata_rows(entry))
    rows.extend(_internal_metadata_rows(entry))
    if metadata.unreadable_event_file_count:
        rows.append(
            TensorboardMetadataDisplayRow(
                "Unreadable Files",
                f"{metadata.unreadable_event_file_count:,}",
                value_style="yellow",
            )
        )
    if metadata.truncated:
        rows.append(
            TensorboardMetadataDisplayRow(
                "Scan",
                "Bounded sample (counts marked ≥)",
                value_style="yellow",
            )
        )
    return tuple(rows)


def _primary_metadata_rows(
    entry: TensorboardPackEntry,
) -> tuple[TensorboardMetadataDisplayRow, ...]:
    """Project time and chart statistics used for run identification."""

    metadata = entry.metadata
    if metadata is None:
        return ()
    duration_label = {
        "training": "Training Time",
        "evaluation": "Evaluation Time",
        "comparison": "Combined Time",
    }.get(entry.pack_type, "Logged Time")
    if metadata.duration_estimated:
        duration_label = f"{duration_label} (est.)"
    duration = format_duration(metadata.duration_seconds)
    return (
        TensorboardMetadataDisplayRow(
            duration_label,
            f"~{duration}" if metadata.duration_estimated and duration != "-" else duration,
        ),
        TensorboardMetadataDisplayRow(
            "Started",
            _estimated_value(
                format_epoch(metadata.started_at_epoch),
                metadata.duration_estimated,
            ),
        ),
        TensorboardMetadataDisplayRow(
            "Finished",
            _estimated_value(
                format_epoch(metadata.finished_at_epoch),
                metadata.duration_estimated,
            ),
        ),
        TensorboardMetadataDisplayRow(
            "Charts / Tags",
            _bounded_quantity(metadata.chart_count, metadata.truncated),
        ),
        TensorboardMetadataDisplayRow(
            "Scalar / Histogram",
            " / ".join(
                (
                    _bounded_quantity(metadata.scalar_tag_count, metadata.truncated),
                    _bounded_quantity(metadata.histogram_tag_count, metadata.truncated),
                )
            ),
        ),
    )


def _internal_metadata_rows(
    entry: TensorboardPackEntry,
) -> tuple[TensorboardMetadataDisplayRow, ...]:
    """Project lower-level event, step, and storage statistics."""

    metadata = entry.metadata
    if metadata is None:
        return ()
    rows: list[TensorboardMetadataDisplayRow] = []
    if metadata.image_tag_count or metadata.text_tag_count or metadata.other_tag_count:
        rows.append(
            TensorboardMetadataDisplayRow(
                "Image / Text / Other",
                " / ".join(
                    _bounded_quantity(value, metadata.truncated)
                    for value in (
                        metadata.image_tag_count,
                        metadata.text_tag_count,
                        metadata.other_tag_count,
                    )
                ),
            )
        )
    rows.extend(
        (
            TensorboardMetadataDisplayRow(
                "Step Range",
                format_step_range(
                    metadata.minimum_step,
                    metadata.maximum_step,
                    upper_is_lower_bound=metadata.truncated,
                ),
            ),
            TensorboardMetadataDisplayRow(
                "Records / Values",
                " / ".join(
                    (
                        _bounded_quantity(metadata.event_record_count, metadata.truncated),
                        _bounded_quantity(metadata.summary_value_count, metadata.truncated),
                    )
                ),
            ),
            TensorboardMetadataDisplayRow(
                "Event Files",
                f"{metadata.event_file_count:,}",
            ),
            TensorboardMetadataDisplayRow(
                "Event Data",
                format_byte_size(metadata.log_size_bytes),
            ),
        )
    )
    return tuple(rows)


def format_duration(seconds: float | None) -> str:
    """Format a non-negative duration compactly."""

    if seconds is None:
        return "-"
    total_seconds = max(0, round(float(seconds)))
    days, day_remainder = divmod(total_seconds, 86_400)
    hours, hour_remainder = divmod(day_remainder, 3_600)
    minutes, remaining_seconds = divmod(hour_remainder, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remaining_seconds:02d}s"
    return f"{remaining_seconds}s"


def format_epoch(epoch_seconds: float | None) -> str:
    """Format one event wall time in the local timezone."""

    if epoch_seconds is None:
        return "-"
    try:
        timestamp = datetime.fromtimestamp(float(epoch_seconds)).astimezone()
    except (OSError, OverflowError, TypeError, ValueError):
        return "-"
    return timestamp.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_step_range(
    minimum_step: int | None,
    maximum_step: int | None,
    *,
    upper_is_lower_bound: bool = False,
) -> str:
    """Format the inclusive event-step range."""

    if minimum_step is None or maximum_step is None:
        return "-"
    upper_prefix = "≥" if upper_is_lower_bound else ""
    return f"{minimum_step:,} → {upper_prefix}{maximum_step:,}"


def _bounded_quantity(value: int, lower_bound: bool) -> str:
    """Format a sampled quantity without presenting it as an exact total."""

    return f"{'≥' if lower_bound else ''}{max(0, int(value)):,}"


def _estimated_value(value: str, estimated: bool) -> str:
    """Mark a timestamp inferred partly from event-file metadata."""

    return f"~{value}" if estimated and value != "-" else value


def format_byte_size(size_bytes: int) -> str:
    """Format event-log bytes with binary units."""

    size = max(0, int(size_bytes))
    amount = float(size)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{int(amount)} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return f"{size} B"


def format_compact_number(value: int) -> str:
    """Format a large integer for a narrow tree label."""

    number = max(0, int(value))
    if number >= 1_000_000:
        return f"{number / 1_000_000:.1f}m"
    if number >= 1_000:
        return f"{number / 1_000:.1f}k"
    return str(number)


__all__ = [
    "TensorboardMetadataDisplayRow",
    "compact_tensorboard_metadata",
    "format_byte_size",
    "format_compact_number",
    "format_duration",
    "format_epoch",
    "format_step_range",
    "tensorboard_metadata_rows",
]
