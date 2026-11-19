# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the model lineage viewer interface component."""

from typing import Optional

from lanista.gui.widgets.fragments.lineage_tree_viewer import LineageTreeViewer


class ModelLineageViewer(LineageTreeViewer):
    """Compatibility wrapper for detailed model lineage views."""

    def __init__(self, model_uuid: Optional[str] = None, **kwargs):
        """Initialize the model lineage viewer and its owned state."""
        super().__init__(
            kind="model",
            item_uuid=model_uuid,
            compact=False,
            show_search=True,
            show_details=True,
            **kwargs,
        )

