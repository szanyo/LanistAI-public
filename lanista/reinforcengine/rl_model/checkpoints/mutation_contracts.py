# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Internal port and durable-result envelope for checkpoint mutations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class CheckpointMutationCommit[ResultT]:
    """Carry authoritative committed state and deferred recovery evidence."""

    value: ResultT
    post_commit_error: Exception | None = None


class CheckpointMutationPort(Protocol):
    """Storage mutation operations required by checkpoint administration."""

    def delete(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
    ) -> CheckpointMutationCommit[str]:
        """Delete one artifact and return its authoritative committed name."""

        ...

    def rename(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        renamed_checkpoint_name: str | None,
    ) -> CheckpointMutationCommit[str]:
        """Rename one artifact and return its authoritative committed name."""

        ...

    def update_annotation(
        self,
        session_name: str | None,
        checkpoint_name: str | None,
        annotation: str | None,
    ) -> CheckpointMutationCommit[str | None]:
        """Store a cleaned annotation and return the committed value."""

        ...
