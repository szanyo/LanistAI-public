# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the training chain viewer interface component."""

from typing import Optional

from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer


class TrainingChainViewer(LineageTreeViewer):
    """Compatibility wrapper for detailed session lineage views."""

    def __init__(self, session_uuid: Optional[str] = None, **kwargs):
        """Initialize the training chain viewer and its owned state."""
        super().__init__(
            kind="session",
            item_uuid=session_uuid,
            compact=False,
            show_search=True,
            show_details=True,
            **kwargs,
        )

