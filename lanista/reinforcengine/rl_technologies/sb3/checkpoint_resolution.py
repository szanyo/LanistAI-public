# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Root-confined Stable-Baselines3 checkpoint path resolution."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

from lanista.reinforcengine.rl_model.storage.names import CheckpointStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard

SB3_CHECKPOINT_ROOT_DIRECTORY = "stablebaselines"


def checkpoint_folder_stem(
    checkpoint_name: str | CheckpointStorageName,
) -> CheckpointStorageName:
    """Return the validated organized-layout directory name."""

    validated_name = CheckpointStorageName.parse(checkpoint_name)
    stem = os.path.splitext(validated_name.value)[0]
    return CheckpointStorageName.parse(stem)


def build_sb3_checkpoint_path(
    checkpoint_directory: str | Path,
    checkpoint_name: str | CheckpointStorageName,
) -> str:
    """Build one organized checkpoint path without allowing root escape."""

    validated_name = CheckpointStorageName.parse(checkpoint_name)
    checkpoint_path = StoragePathGuard(checkpoint_directory).resolve(
        SB3_CHECKPOINT_ROOT_DIRECTORY,
        checkpoint_folder_stem(validated_name),
        validated_name,
    )
    return str(checkpoint_path)


def reserve_sb3_checkpoint_path(
    checkpoint_directory: str | Path,
    checkpoint_name: str | CheckpointStorageName,
) -> str:
    """Reserve a collision-free organized checkpoint path."""

    validated_name = CheckpointStorageName.parse(checkpoint_name)
    candidate = Path(
        build_sb3_checkpoint_path(
            checkpoint_directory,
            validated_name,
        )
    )
    if not candidate.exists() and not candidate.parent.exists():
        return str(candidate)
    suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    root, extension = os.path.splitext(validated_name.value)
    suffixed_name = CheckpointStorageName.parse(
        f"{root}_{suffix}{extension}"
    )
    return build_sb3_checkpoint_path(
        checkpoint_directory,
        suffixed_name,
    )


def resolve_sb3_checkpoint_path(
    checkpoint_directory: str | Path,
    checkpoint_name: str | CheckpointStorageName,
) -> str | None:
    """Resolve flat and organized SB3 layouts below one checkpoint root."""

    raw_checkpoint_name = str(checkpoint_name or "")
    if not raw_checkpoint_name:
        return None
    validated_name = CheckpointStorageName.parse(raw_checkpoint_name)
    checkpoint_guard = StoragePathGuard(checkpoint_directory)

    direct_path = checkpoint_guard.resolve(validated_name)
    if direct_path.is_file():
        return str(direct_path)

    organized_path = Path(
        build_sb3_checkpoint_path(
            checkpoint_directory,
            validated_name,
        )
    )
    if organized_path.is_file():
        return str(organized_path)

    if not validated_name.value.lower().endswith(".zip"):
        zipped_name = CheckpointStorageName.parse(
            f"{validated_name.value}.zip"
        )
        organized_zip_path = Path(
            build_sb3_checkpoint_path(
                checkpoint_directory,
                zipped_name,
            )
        )
        if organized_zip_path.is_file():
            return str(organized_zip_path)

    candidate_directory = _candidate_directory(
        checkpoint_guard,
        direct_path,
        validated_name,
    )
    if candidate_directory is None:
        return None
    expected_name = _expected_zip_name(validated_name)
    directory_guard = StoragePathGuard(candidate_directory)
    expected_path = directory_guard.resolve(expected_name)
    if expected_path.is_file():
        return str(expected_path)
    latest_zip = _latest_guarded_zip(directory_guard)
    return str(latest_zip) if latest_zip is not None else None


def list_sb3_checkpoint_paths(
    checkpoint_directory: str | Path,
) -> list[str]:
    """List valid flat and organized checkpoints by descending mtime."""

    checkpoint_guard = StoragePathGuard(checkpoint_directory)
    paths = _flat_checkpoint_paths(checkpoint_guard)
    paths.extend(_organized_checkpoint_paths(checkpoint_guard))
    unique_paths = sorted(
        set(paths),
        key=lambda path: (path.stat().st_mtime_ns, str(path)),
        reverse=True,
    )
    return [str(path) for path in unique_paths]


def _candidate_directory(
    checkpoint_guard: StoragePathGuard,
    direct_path: Path,
    checkpoint_name: CheckpointStorageName,
) -> Path | None:
    """Return the candidate directory value owned by this checkpoint resolution module."""
    if direct_path.is_dir():
        return direct_path
    organized_directory = checkpoint_guard.resolve(
        SB3_CHECKPOINT_ROOT_DIRECTORY,
        checkpoint_folder_stem(checkpoint_name),
    )
    return organized_directory if organized_directory.is_dir() else None


def _expected_zip_name(
    checkpoint_name: CheckpointStorageName,
) -> CheckpointStorageName:
    """Return the expected zip name value owned by this checkpoint resolution module."""
    if checkpoint_name.value.lower().endswith(".zip"):
        return checkpoint_name
    return CheckpointStorageName.parse(f"{checkpoint_name.value}.zip")


def _flat_checkpoint_paths(
    checkpoint_guard: StoragePathGuard,
) -> list[Path]:
    """Return the flat checkpoint paths value owned by this checkpoint resolution module."""
    if not checkpoint_guard.root.is_dir():
        return []
    paths: list[Path] = []
    for entry in sorted(os.listdir(checkpoint_guard.root)):
        validated_name = CheckpointStorageName.parse(entry)
        full_path = checkpoint_guard.resolve(validated_name)
        if full_path.is_file() and entry.lower().endswith(".zip"):
            paths.append(full_path)
    return paths


def _organized_checkpoint_paths(
    checkpoint_guard: StoragePathGuard,
) -> list[Path]:
    """Return the organized checkpoint paths value owned by this checkpoint resolution module."""
    organized_root = checkpoint_guard.resolve(
        SB3_CHECKPOINT_ROOT_DIRECTORY
    )
    if not organized_root.is_dir():
        return []
    organized_guard = StoragePathGuard(organized_root)
    paths: list[Path] = []
    for entry in sorted(os.listdir(organized_root)):
        validated_name = CheckpointStorageName.parse(entry)
        full_path = organized_guard.resolve(validated_name)
        checkpoint_path = _organized_entry_checkpoint(
            validated_name,
            full_path,
        )
        if checkpoint_path is not None:
            paths.append(checkpoint_path)
    return paths


def _organized_entry_checkpoint(
    entry_name: CheckpointStorageName,
    full_path: Path,
) -> Path | None:
    """Return the organized entry checkpoint value owned by this checkpoint resolution module."""
    if full_path.is_file():
        return full_path if entry_name.value.lower().endswith(".zip") else None
    if not full_path.is_dir():
        return None
    directory_guard = StoragePathGuard(full_path)
    expected_path = directory_guard.resolve(
        CheckpointStorageName.parse(f"{entry_name.value}.zip")
    )
    if expected_path.is_file():
        return expected_path
    return _latest_guarded_zip(directory_guard)


def _latest_guarded_zip(guard: StoragePathGuard) -> Path | None:
    """Return the latest guarded zip value owned by this checkpoint resolution module."""
    zip_files: list[Path] = []
    for entry in sorted(os.listdir(guard.root)):
        if not entry.lower().endswith(".zip"):
            continue
        validated_name = CheckpointStorageName.parse(entry)
        candidate = guard.resolve(validated_name)
        if candidate.is_file():
            zip_files.append(candidate)
    return zip_files[-1] if zip_files else None
