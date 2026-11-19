# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate name formatting."""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from typing import Protocol


class NameSanitizer(Protocol):
    """Callable contract for storage-safe display-name normalization."""

    def __call__(self, value: str, limit: int = 48) -> str:
        """Invoke this name sanitizer as a callable."""
        ...


class NameFormattingService:
    """Coordinate name formatting operations."""

    _NAME_PATTERN = re.compile(r"[^a-zA-Z0-9._-]+")
    _DASH_PATTERN = re.compile(r"-+")

    @classmethod
    def sanitize_name(cls, value: str, limit: int = 48, fallback: str = "session") -> str:
        """Return the sanitize name value owned by this name formatting service."""
        sanitized = cls._NAME_PATTERN.sub("-", str(value or "").strip())
        sanitized = cls._DASH_PATTERN.sub("-", sanitized).strip("-._")
        sanitized = sanitized[:limit] if sanitized else ""
        return sanitized or fallback

    @classmethod
    def build_timestamp_label(
        cls,
        created_at_raw: str | None,
        *,
        invalid_strategy: str = "current",
        fallback: str = "unknown-time",
    ) -> str:
        """Build the timestamp label required by this name formatting service."""
        if created_at_raw:
            try:
                return datetime.fromisoformat(str(created_at_raw).replace("Z", "+00:00")).strftime("%Y-%m-%d_%H-%M-%S")
            except ValueError:
                if invalid_strategy == "sanitize":
                    return cls.sanitize_name(created_at_raw, limit=24, fallback=fallback)
                if invalid_strategy == "fallback":
                    return fallback
                return datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        return fallback

    @classmethod
    def build_unique_storage_name(
        cls,
        root_path: str,
        display_name: str,
        *,
        limit: int = 48,
        fallback: str = "session",
    ) -> str:
        """Build the unique storage name required by this name formatting service."""
        if limit < 4:
            raise ValueError("Unique storage names require a limit of at least four characters.")
        base_name = cls.sanitize_name(display_name, limit=limit, fallback=fallback)
        candidate = base_name
        suffix = 2
        while os.path.lexists(os.path.join(str(root_path), candidate)):
            suffix_token = f"-{suffix}"
            stem_limit = limit - len(suffix_token)
            if stem_limit < 1:
                raise FileExistsError(
                    "The configured storage-name namespace is exhausted."
                )
            stem = base_name[:stem_limit].rstrip("-._")
            if not stem:
                stem = cls.sanitize_name(
                    fallback,
                    limit=stem_limit,
                    fallback="x",
                )
            candidate = f"{stem}{suffix_token}"
            suffix += 1
        return candidate

    @classmethod
    def build_run_name(
        cls,
        session_uuid: str,
        created_at_raw: str | None,
        *,
        short_uuid_limit: int = 12,
        timestamp_limit: int = 24,
        invalid_strategy: str = "sanitize",
        fallback: str = "unknown-time",
    ) -> str | None:
        """Build the run name required by this name formatting service."""
        timestamp_token = cls.build_timestamp_label(
            created_at_raw,
            invalid_strategy=invalid_strategy,
            fallback=fallback,
        )
        if invalid_strategy == "sanitize" and timestamp_token not in {fallback, "unknown-time"} and len(timestamp_token) > timestamp_limit:
            timestamp_token = timestamp_token[:timestamp_limit]
        short_uuid = cls.sanitize_name(session_uuid, limit=short_uuid_limit, fallback="")
        if not short_uuid:
            return None
        return f"{short_uuid}__{timestamp_token}"
