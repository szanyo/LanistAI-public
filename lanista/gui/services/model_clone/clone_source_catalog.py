# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Filesystem catalog and path policy for model-clone selection."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CloneTreeEntry:
    """One normalized entry in a clone candidate tree."""

    relative_path: str
    is_dir: bool


class CloneSourceCatalog:
    """Discover clone candidates without leaking filesystem policy into the UI."""

    @staticmethod
    def normalize_relative_path(value: str) -> str:
        """Return a safe slash-separated relative path or an empty sentinel."""

        raw_value = str(value or "").strip().replace("\\", "/")
        if not raw_value or raw_value.startswith("/"):
            return ""
        components: list[str] = []
        for component in raw_value.split("/"):
            if component in {"", "."}:
                continue
            if component == ".." or (not components and ":" in component):
                return ""
            components.append(component)
        return "/".join(components)

    @staticmethod
    def is_ignored_file(name: str) -> bool:
        """Return whether an entry is generated interpreter cache data."""

        return name == "__pycache__" or name.endswith((".pyc", ".pyo"))

    @classmethod
    def is_non_clonable(
        cls,
        relative_path: str,
        *,
        root_dirs: set[str],
        generated_paths: set[str],
    ) -> bool:
        """Apply the clone policy to one normalized source-relative path."""

        normalized = cls.normalize_relative_path(relative_path)
        if not normalized:
            return False
        return normalized in generated_paths or normalized.split("/", 1)[0] in root_dirs

    @classmethod
    def discover(
        cls,
        root_path: str,
        *,
        root_dirs: set[str],
        generated_paths: set[str],
    ) -> list[CloneTreeEntry]:
        """Return a deterministic inventory below an existing source root."""

        if not root_path or not os.path.isdir(root_path):
            return []
        source_root = os.path.abspath(root_path)
        discovered: list[CloneTreeEntry] = []
        for current_root, dir_names, file_names in os.walk(source_root):
            relative_root = cls.normalize_relative_path(os.path.relpath(current_root, source_root))
            if relative_root and cls.is_non_clonable(
                relative_root,
                root_dirs=root_dirs,
                generated_paths=generated_paths,
            ):
                dir_names[:] = []
                continue
            filtered_dirs: list[str] = []
            for name in sorted(dir_names, key=str.lower):
                relative_path = cls.normalize_relative_path(os.path.join(relative_root, name))
                if cls.is_ignored_file(name) or cls.is_non_clonable(
                    relative_path,
                    root_dirs=root_dirs,
                    generated_paths=generated_paths,
                ):
                    continue
                filtered_dirs.append(name)
                discovered.append(CloneTreeEntry(relative_path, True))
            dir_names[:] = filtered_dirs
            for name in sorted(file_names, key=str.lower):
                relative_path = cls.normalize_relative_path(os.path.join(relative_root, name))
                if cls.is_ignored_file(name) or cls.is_non_clonable(
                    relative_path,
                    root_dirs=root_dirs,
                    generated_paths=generated_paths,
                ):
                    continue
                discovered.append(CloneTreeEntry(relative_path, False))
        return sorted(
            discovered,
            key=lambda entry: (
                entry.relative_path.count("/"),
                entry.relative_path.lower(),
            ),
        )
