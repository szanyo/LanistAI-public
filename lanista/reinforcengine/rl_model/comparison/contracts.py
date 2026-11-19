# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed administration contracts for comparison packs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction


@dataclass(frozen=True, slots=True)
class ComparisonPackCreationOutcome:
    """Describe a durably created comparison pack."""

    pack_name: str
    log_directory: str
    source_paths: tuple[str, ...]
    committed: bool
    degraded: bool = False
    reload_required: bool = False
    storage_recovery_required: bool = False


@dataclass(frozen=True, slots=True)
class ComparisonPackMutationOutcome:
    """Describe a durable comparison-pack mutation."""

    pack_name: str
    previous_pack_name: str | None = None
    log_directory: str | None = None
    annotation: str | None = None
    source_paths: tuple[str, ...] = ()
    committed: bool = True
    degraded: bool = False
    reload_required: bool = False
    storage_recovery_required: bool = False


class ComparisonPackTransactionFactory(Protocol):
    """Construct one journaled comparison-pack storage transaction."""

    def __call__(
        self,
        root: str | os.PathLike[str],
        *,
        operation: str,
    ) -> StorageTransaction:
        """Invoke this comparison pack transaction factory as a callable."""
        ...


__all__ = [
    "ComparisonPackCreationOutcome",
    "ComparisonPackMutationOutcome",
    "ComparisonPackTransactionFactory",
]
