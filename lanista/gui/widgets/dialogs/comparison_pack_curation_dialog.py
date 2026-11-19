# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Comparison Pack Curation dialog scaffold.

This module provides a minimal, safe-to-import stub for the
Comparison Pack Curation UI. It intentionally keeps Textual/UI
dependencies out of the import path so it can be imported in tests
that don't initialize the full TUI.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from lanista.reinforcengine.application_errors import OperationResult
from lanista.reinforcengine.rl_model.comparison.contracts import (
    ComparisonPackMutationOutcome,
)


class ComparisonPackCurationPort(Protocol):
    """TensorBoard comparison-pack operations required by the dialog."""

    def replace_comparison_pack_sources(
        self,
        pack_name: str,
        source_paths: list[str],
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Replace the comparison pack sources owned by this comparison pack curation port."""
        ...

    def update_comparison_pack_annotation(
        self,
        pack_name: str,
        annotation: str | None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Update the comparison pack annotation owned by this comparison pack curation port."""
        ...

    def rename_comparison_pack(
        self,
        pack_name: str,
        new_pack_name: str,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Rename the comparison pack owned by this comparison pack curation port."""
        ...

    def delete_comparison_pack(
        self,
        pack_name: str,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Delete the comparison pack through this comparison pack curation port's safety boundary.
        """
        ...


class ComparisonPackCurationDialog:
    """Lightweight dialog wrapper for comparison pack curation actions.

    This small adapter keeps UI-independent action results testable and
    delegates storage mutations to the typed curation service.
    """

    def __init__(self, pack_service: ComparisonPackCurationPort) -> None:
        """Initialize the comparison pack curation dialog and its owned state."""
        self._pack_service = pack_service

    def open(self, pack_name: str) -> dict[str, str]:
        """Open the curation dialog for `pack_name`.

        Returns a small status dict to keep callers testable without a UI.
        """
        return {"action": "open", "pack_name": pack_name}

    def update_sources(
        self,
        pack_name: str,
        source_paths: Iterable[str],
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Replace the sources of an existing pack."""

        return self._pack_service.replace_comparison_pack_sources(
            pack_name, list(source_paths)
        )

    def update_annotation(
        self,
        pack_name: str,
        annotation: str | None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Update comparison pack notes."""

        return self._pack_service.update_comparison_pack_annotation(
            pack_name, annotation
        )

    def rename_pack(
        self,
        pack_name: str,
        new_pack_name: str,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Rename an existing comparison pack."""

        return self._pack_service.rename_comparison_pack(
            pack_name, new_pack_name
        )

    def delete_pack(
        self,
        pack_name: str,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Delete an existing comparison pack."""

        return self._pack_service.delete_comparison_pack(pack_name)


__all__ = ["ComparisonPackCurationDialog", "ComparisonPackCurationPort"]
