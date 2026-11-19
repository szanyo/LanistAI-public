# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the model selector panel interface component."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from functools import partial
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Input, Static, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.services.async_collection_loader import run_blocking
from lanista.gui.utils.text_cells import fit_text_cell
from lanista.gui.utils.tree_nodes import first_data_node
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model_discovery_service import (
    DiscoveredInvalidModelEntry,
    DiscoveredSessionMetadataStatus,
    DiscoveredValidModelEntry,
)

LOGGER = logging.getLogger(__name__)


class ModelSelectorPanel(Widget):
    """Present model selector in the Textual interface."""

    _APPLICATION = LanistAIApplication()
    _DISPLAY_NAME_WIDTH = 28
    _VERSION_WIDTH = 12
    _UUID_WIDTH = 8
    _CREATED_WIDTH = 19
    _SESSIONS_WIDTH = 4
    _CURRENT_WIDTH = 1

    @dataclass
    class ModelSelected(Message):
        """Carry the model selected interface message."""

        model_manifest: ModelManifest | None

    def __init__(
        self,
        current_model_uuid: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model selector panel and its owned state."""
        super().__init__(**kwargs)
        self.border_title = "Model Discovery"
        self._selected_manifest: ModelManifest | None = None
        self._manifest_entries: list[DiscoveredValidModelEntry] = []
        self._invalid_entries: list[DiscoveredInvalidModelEntry] = []
        self._current_model_uuid = str(current_model_uuid or "").strip()
        self._loading = False
        self._load_error: BaseException | None = None
        self._load_generation = 0
        self._discovery_task: asyncio.Task[None] | None = None

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this model selector panel."""
        yield Input(placeholder="Search models...", id="model_selector_search")
        yield Static(self._build_header_label(), id="model_selector_header")
        tree: Tree[ModelManifest | None] = Tree("Models", id="rl_model_tree")
        tree.show_root = False
        tree.show_guides = False
        tree.border_title = "Valid Model Folders"
        yield tree

    @classmethod
    def _build_header_label(cls) -> Text:
        """Build the header label required by this model selector panel."""
        header = Text()
        header.append(" ")
        header.append(fit_text_cell("Name", cls._DISPLAY_NAME_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("C", cls._CURRENT_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("Version", cls._VERSION_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("UUID", cls._UUID_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("Created", cls._CREATED_WIDTH), style="bold dim")
        header.append("  ")
        header.append("Ses", style="bold dim")
        return header

    @classmethod
    def _build_entry_label(cls, entry: DiscoveredValidModelEntry) -> Text:
        """Build the entry label required by this model selector panel."""
        label = Text()
        label.append(
            fit_text_cell(entry.display_name or "Unnamed model", cls._DISPLAY_NAME_WIDTH),
            style="bold",
        )
        label.append("  ")
        if entry.is_current:
            label.append(fit_text_cell("•", cls._CURRENT_WIDTH), style="bold yellow")
        else:
            label.append(fit_text_cell("", cls._CURRENT_WIDTH))
        label.append("  ")
        label.append(
            fit_text_cell(entry.effective_version or "-", cls._VERSION_WIDTH),
            style="green",
        )
        label.append("  ")
        label.append(fit_text_cell(entry.short_uuid or "-", cls._UUID_WIDTH), style="dim")
        label.append("  ")
        label.append(
            fit_text_cell(entry.created_at_display, cls._CREATED_WIDTH),
            style="cyan",
        )
        label.append("  ")
        session_summary = str(entry.session_count)
        session_style = "italic cyan"
        if entry.session_metadata_status is not DiscoveredSessionMetadataStatus.CLEAN:
            invalid_suffix = (
                str(entry.invalid_session_count)
                if entry.invalid_session_count > 0
                else ""
            )
            session_summary = f"{session_summary} ⚠{invalid_suffix}"
            session_style = "bold yellow"
        label.append(session_summary, style=session_style)
        return label

    def _rebuild_tree(self, filter_text: str = "") -> None:
        """Rebuild the tree owned by this model selector panel."""
        tree = self.query_one("#rl_model_tree", Tree)
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False

        if self._loading:
            node = tree.root.add_leaf(Text("Loading models...", style="dim"), data=None)
            node.allow_expand = False
            return
        if self._load_error is not None:
            node = tree.root.add_leaf(Text("Failed to load models.", style="red"), data=None)
            node.allow_expand = False
            return

        normalized_filter = str(filter_text or "").strip().lower()
        visible_valid_entries = [
            entry for entry in self._manifest_entries
            if not normalized_filter or normalized_filter in entry.search_text
        ]

        if not visible_valid_entries:
            empty_node = tree.root.add_leaf(Text("No matching models.", style="dim"), data=None)
            empty_node.allow_expand = False
            self._selected_manifest = None
            self.post_message(self.ModelSelected(None))
            return

        selected_node: TreeNode[ModelManifest | None] | None = None
        for manifest_entry in visible_valid_entries:
            node = tree.root.add_leaf(self._build_entry_label(manifest_entry), data=manifest_entry.manifest)
            node.allow_expand = False
            if self._selected_manifest is manifest_entry.manifest or (
                self._selected_manifest is not None
                and getattr(self._selected_manifest.get_configuration(), "model_uuid", None)
                == getattr(manifest_entry.manifest.get_configuration(), "model_uuid", None)
            ):
                selected_node = node

        if selected_node is None:
            selected_node = first_data_node(tree.root, ModelManifest)

        if selected_node is not None and isinstance(selected_node.data, ModelManifest):
            self._selected_manifest = selected_node.data
            tree.select_node(selected_node)
            self.post_message(self.ModelSelected(self._selected_manifest))
        else:
            self._selected_manifest = None
            self.post_message(self.ModelSelected(None))

    def on_mount(self) -> None:
        """Handle the mount event for this model selector panel."""
        self._start_discovery()
        self.query_one("#model_selector_search", Input).focus()

    def _start_discovery(self) -> None:
        """Discover model manifests without blocking Textual's event loop."""

        self._loading = True
        self._load_error = None
        self._load_generation += 1
        generation = self._load_generation
        self._rebuild_tree()
        discovery = self._APPLICATION.get_application_services().catalog.models

        async def _load() -> None:
            try:
                valid_entries, invalid_entries = await run_blocking(
                    partial(
                        discovery.discover_model_candidates,
                        current_model_uuid=self._current_model_uuid,
                    )
                )
            except asyncio.CancelledError:
                return
            except BaseException as error:
                if generation != self._load_generation or not self.is_mounted:
                    return
                self._loading = False
                self._load_error = error
                self._rebuild_tree()
                LOGGER.error(
                    "Model selector discovery failed.",
                    exc_info=(type(error), error, error.__traceback__),
                )
                return
            if generation != self._load_generation or not self.is_mounted:
                return
            self._loading = False
            self._manifest_entries = list(valid_entries)
            self._invalid_entries = list(invalid_entries)
            if self._selected_manifest is None and self._current_model_uuid:
                self._selected_manifest = next(
                    (
                        entry.manifest
                        for entry in self._manifest_entries
                        if str(
                            getattr(
                                entry.manifest.get_configuration(),
                                "model_uuid",
                                "",
                            )
                            or ""
                        ).strip()
                        == self._current_model_uuid
                    ),
                    None,
                )
            self._rebuild_tree(self.query_one("#model_selector_search", Input).value)

        self._discovery_task = asyncio.create_task(_load())

    def on_unmount(self) -> None:
        """Invalidate and cancel outstanding discovery work."""

        self._load_generation += 1
        task = self._discovery_task
        if task is not None and not task.done():
            task.cancel()

    def get_selected_manifest(self) -> ModelManifest | None:
        """Return the selected manifest exposed by this model selector panel."""
        return self._selected_manifest

    def get_invalid_entries(self) -> list[DiscoveredInvalidModelEntry]:
        """Return the invalid entries exposed by this model selector panel."""
        return list(self._invalid_entries)

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this model selector panel."""
        if event.input.id == "model_selector_search":
            self._rebuild_tree(event.value)

    def on_tree_node_selected(
        self,
        event: Tree.NodeSelected[ModelManifest | None],
    ) -> None:
        """Handle the tree node selected event for this model selector panel."""
        model_manifest = event.node.data if isinstance(event.node.data, ModelManifest) else None
        self._selected_manifest = model_manifest
        self.post_message(self.ModelSelected(model_manifest))
