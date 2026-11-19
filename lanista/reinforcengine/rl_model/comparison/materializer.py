# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Source selection and filesystem materialization for comparison packs."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Sequence

from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import (
    TensorboardPackEntry,
)
from lanista.reinforcengine.rl_model.name_formatting_service import NameSanitizer


class ComparisonPackMaterializer:
    """Resolve selected TensorBoard runs and copy them into a staged pack."""

    def __init__(
        self,
        *,
        source_entries_provider: Callable[[], list[TensorboardPackEntry]],
        sanitize_name: NameSanitizer,
    ) -> None:
        """Initialize the comparison pack materializer and its owned state."""
        self._source_entries_provider = source_entries_provider
        self._sanitize_name = sanitize_name

    @staticmethod
    def normalize_source_paths(source_paths: Sequence[str]) -> tuple[str, ...]:
        """Return unique absolute source paths while preserving input order."""

        normalized_paths: list[str] = []
        for path in source_paths:
            raw_path = str(path or "").strip()
            if not raw_path:
                continue
            normalized = os.path.abspath(raw_path)
            if normalized not in normalized_paths:
                normalized_paths.append(normalized)
        return tuple(normalized_paths)

    def resolve_selected_entries(
        self,
        normalized_paths: Sequence[str],
    ) -> tuple[TensorboardPackEntry, ...]:
        """Resolve available entries in the caller's requested path order."""

        available_entries = self._source_entries_provider()
        entry_by_path = {
            os.path.abspath(entry.path): entry for entry in available_entries
        }
        return tuple(
            entry_by_path[path]
            for path in normalized_paths
            if path in entry_by_path
        )

    def copy_selected_entries(
        self,
        pack_path: str,
        selected_entries: Sequence[TensorboardPackEntry],
    ) -> None:
        """Copy selected runs into a deterministic staged directory layout."""

        for index, entry in enumerate(selected_entries, start=1):
            label = self._sanitize_name(
                f"{entry.model_name or entry.model_display_name or 'model'}_"
                f"{entry.pack_type}_{entry.framework_uuid}_{entry.run_name}",
                limit=96,
            ) or f"pack_{index:02d}"
            destination = os.path.join(pack_path, f"{index:02d}_{label}")
            if entry.recursive_copy:
                shutil.copytree(entry.path, destination)
                continue
            os.makedirs(destination)
            with os.scandir(entry.path) as children:
                for child in children:
                    if not child.is_file(follow_symlinks=False):
                        continue
                    shutil.copy2(child.path, os.path.join(destination, child.name))


__all__ = ["ComparisonPackMaterializer"]
