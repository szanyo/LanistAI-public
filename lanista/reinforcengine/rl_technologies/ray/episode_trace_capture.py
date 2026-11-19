# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Atomic checkpoint-to-episode claims for optional Ray training traces."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock

import ray


@dataclass(frozen=True, slots=True)
class EpisodeTraceClaim:
    """Checkpoint targets owned by one complete post-checkpoint episode."""

    token: str
    checkpoint_paths: tuple[str, ...]


class EpisodeTraceCaptureGate:
    """Let exactly one episode claim all currently pending checkpoint paths."""

    def __init__(self) -> None:
        """Initialize an empty process-local or Ray actor-owned gate."""

        self._lock = Lock()
        self._pending_paths: list[str] = []
        self._active_claim: EpisodeTraceClaim | None = None

    def arm(self, checkpoint_path: str) -> None:
        """Queue one committed checkpoint for the next clean episode."""

        path = str(checkpoint_path)
        with self._lock:
            active_paths = (
                self._active_claim.checkpoint_paths
                if self._active_claim is not None
                else ()
            )
            if path not in self._pending_paths and path not in active_paths:
                self._pending_paths.append(path)

    def claim(self, token: str) -> EpisodeTraceClaim | None:
        """Atomically assign pending targets to one episode token."""

        normalized_token = str(token)
        with self._lock:
            if self._active_claim is not None or not self._pending_paths:
                return None
            claim = EpisodeTraceClaim(
                token=normalized_token,
                checkpoint_paths=tuple(self._pending_paths),
            )
            self._pending_paths.clear()
            self._active_claim = claim
            return claim

    def complete(self, token: str) -> bool:
        """Acknowledge a successfully written episode claim."""

        with self._lock:
            if self._active_claim is None or self._active_claim.token != str(token):
                return False
            self._active_claim = None
            return True

    def abandon(self, token: str) -> bool:
        """Requeue an incomplete claim without losing checkpoint order."""

        with self._lock:
            if self._active_claim is None or self._active_claim.token != str(token):
                return False
            claimed_paths = self._active_claim.checkpoint_paths
            self._pending_paths = list(
                dict.fromkeys((*claimed_paths, *self._pending_paths))
            )
            self._active_claim = None
            return True

    def pending_paths(self) -> tuple[str, ...]:
        """Return a stable inspection snapshot for tests and diagnostics."""

        with self._lock:
            return tuple(self._pending_paths)


RayEpisodeTraceCaptureGate = ray.remote(EpisodeTraceCaptureGate)


__all__ = [
    "EpisodeTraceCaptureGate",
    "EpisodeTraceClaim",
    "RayEpisodeTraceCaptureGate",
]
