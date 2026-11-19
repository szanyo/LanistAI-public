# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed filesystem helpers retained from the former utility dependency."""

from __future__ import annotations

import os
from pathlib import Path


def writef(name: str | os.PathLike[str], content: object) -> None:
    """Handle the writef operation owned by this filesystem module."""
    Path(name).write_text(str(content), encoding="utf-8")


def readf(name: str | os.PathLike[str]) -> list[str]:
    """Return the readf value owned by this filesystem module."""
    return Path(name).read_text(encoding="utf-8").splitlines()


def existf(name: str | os.PathLike[str]) -> bool:
    """Return the existf value owned by this filesystem module."""
    return Path(name).is_file()


def existdir(name: str | os.PathLike[str]) -> bool:
    """Return the existdir value owned by this filesystem module."""
    return Path(name).is_dir()


def makedirs(name: str | os.PathLike[str], exist_ok: bool = False) -> None:
    """Handle the makedirs operation owned by this filesystem module."""
    Path(name).mkdir(parents=True, exist_ok=exist_ok)


def only_directories_from_dir(directory: str | os.PathLike[str]) -> list[str]:
    """Return the only directories from dir value owned by this filesystem module."""
    return [str(path) for path in Path(directory).iterdir() if path.is_dir()]


def only_files_from_dir(directory: str | os.PathLike[str]) -> list[str]:
    """Return the only files from dir value owned by this filesystem module."""
    return [str(path) for path in Path(directory).iterdir() if path.is_file()]

