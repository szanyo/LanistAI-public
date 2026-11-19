# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validated single-component names used at storage boundaries."""

from __future__ import annotations

import os
import unicodedata
from dataclasses import dataclass
from typing import ClassVar, Self


class InvalidStorageName(ValueError):
    """Raised when an untrusted storage name is not one safe path component."""


@dataclass(frozen=True, slots=True)
class StorageName:
    """A validated, portable, single-component filesystem name."""

    value: str

    kind: ClassVar[str] = "storage"
    reserved_values: ClassVar[frozenset[str]] = frozenset({".storage-transactions"})
    reserved_windows_stems: ClassVar[frozenset[str]] = frozenset(
        {"con", "prn", "aux", "nul"}
        | {f"com{index}" for index in range(1, 10)}
        | {f"lpt{index}" for index in range(1, 10)}
    )
    forbidden_characters: ClassVar[frozenset[str]] = frozenset('<>:"/\\|?*')
    maximum_length: ClassVar[int] = 255

    def __post_init__(self) -> None:
        """Validate and finalize the initialized storage name state."""
        raw_value = str(self.value)
        if not raw_value or raw_value != raw_value.strip():
            raise InvalidStorageName(f"{self.kind.title()} name must not be empty or padded with whitespace.")
        if raw_value in {".", ".."}:
            raise InvalidStorageName(f"{self.kind.title()} name cannot be a relative path marker.")
        if os.path.isabs(raw_value) or any(character in self.forbidden_characters for character in raw_value):
            raise InvalidStorageName(f"{self.kind.title()} name must be one relative path component.")
        if raw_value.endswith(".") or raw_value.split(".", maxsplit=1)[0].casefold() in self.reserved_windows_stems:
            raise InvalidStorageName(f"{self.kind.title()} name is not portable across supported filesystems.")
        if len(raw_value.encode("utf-8")) > self.maximum_length:
            raise InvalidStorageName(
                f"{self.kind.title()} name exceeds the {self.maximum_length}-byte filesystem limit."
            )
        if raw_value.casefold() in {value.casefold() for value in self.reserved_values}:
            raise InvalidStorageName(f"{self.kind.title()} name is reserved for storage administration.")
        if any(unicodedata.category(character).startswith("C") for character in raw_value):
            raise InvalidStorageName(f"{self.kind.title()} name contains a control character.")

    @classmethod
    def parse(cls, value: object) -> Self:
        """Return ``value`` as this validated name type."""
        if isinstance(value, cls):
            return value
        return cls(str(value))

    def __str__(self) -> str:
        """Return the human-readable representation of this storage name."""
        return self.value

    def __fspath__(self) -> str:
        """Return the filesystem representation owned by this storage name."""
        return self.value


class ModelStorageName(StorageName):
    """Validated model-directory name."""

    kind = "model"


class SessionStorageName(StorageName):
    """Validated training-session directory name."""

    kind = "session"


class CheckpointStorageName(StorageName):
    """Validated checkpoint file or directory name."""

    kind = "checkpoint"


class ComparisonPackStorageName(StorageName):
    """Validated TensorBoard comparison-pack directory name."""

    kind = "comparison pack"
