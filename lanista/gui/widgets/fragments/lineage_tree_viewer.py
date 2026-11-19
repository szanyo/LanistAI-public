# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the lineage tree viewer interface component."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.utils.timestamps import format_local_timestamp


class LineageTreeViewer(Vertical):
    """Reusable model/session lineage tree with compact and detailed modes."""

    _APPLICATION = LanistAIApplication()
    _DETAIL_LABEL_WIDTH = 12
    DEFAULT_CSS = """
    LineageTreeViewer {
        padding: 0;
    }

    .lineage_tree_widget_search {
        height: 3;
        min-height: 3;
        margin-bottom: 1;
    }

    .lineage_tree_widget_tree {
        width: 1fr;
        height: 1fr;
        min-height: 6;
        padding: 0 1;
        overflow-x: auto;
        overflow-y: auto;
        border-title-align: left;
        border-title-color: $text-muted;
        background: $surface;
    }

    .lineage_tree_widget_tree:focus {
        border: round $primary-lighten-1 50%;
    }

    .lineage_tree_widget_tree > .tree--cursor {
        color: $text;
        background: $primary 20%;
        text-style: bold;
    }

    .lineage_tree_widget_tree > .tree--highlight-line {
        background: $primary 10%;
    }

    .lineage_tree_widget_details {
        height: 9;
        min-height: 9;
        margin-top: 1;
        padding: 0 1;
        border-title-align: left;
        color: $text;
    }
    """

    @dataclass
    class LineageEntry:
        """Describe lineage data."""

        item_uuid: str
        kind: str
        display_name: str
        identity_uuid: str
        version_or_framework: str
        parent_uuid: str | None
        description: str
        created_at_raw: str | None
        modified_at_raw: str | None
        is_current: bool
        child_count: int
        search_text: str

    def __init__(
        self,
        *,
        kind: str,
        item_uuid: str | None = None,
        compact: bool = False,
        show_search: bool | None = None,
        show_details: bool | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the lineage tree viewer and its owned state."""
        super().__init__(**kwargs)
        self._kind = "model" if str(kind or "").strip().lower() == "model" else "session"
        self._item_uuid = str(item_uuid or "").strip() or None
        self._compact = bool(compact)
        self._show_search = (not self._compact) if show_search is None else bool(show_search)
        self._show_details = (not self._compact) if show_details is None else bool(show_details)
        key = str(kwargs.get("id") or f"{self._kind}_lineage_viewer")
        self._search_id = f"{key}_search"
        self._tree_id = f"{key}_tree"
        self._details_id = f"{key}_details"
        self._filter_text = ""

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this lineage tree viewer."""
        if self._show_search:
            search = Input(placeholder=f"Search {self._kind} lineage...", id=self._search_id)
            search.add_class("lineage_tree_widget_search")
            yield search

        tree_title = "Model Lineage" if self._kind == "model" else "Session Lineage"
        tree: Tree[LineageTreeViewer.LineageEntry] = Tree(
            tree_title,
            id=self._tree_id,
        )
        tree.add_class("lineage_tree_widget_tree")
        tree.show_root = False
        tree.show_guides = True
        tree.guide_depth = 2
        tree.auto_expand = False
        tree.border_title = tree_title if not self._compact else f"{tree_title} Preview"
        yield tree

        if self._show_details:
            details = Static(id=self._details_id)
            details.add_class("lineage_tree_widget_details")
            details.border_title = "Selected Model" if self._kind == "model" else "Selected Session"
            yield details

    @staticmethod
    def _dedupe(values: list[str]) -> list[str]:
        """Return the dedupe value owned by this lineage tree viewer."""
        seen: set[str] = set()
        unique: list[str] = []
        for value in values:
            if not value or value in seen:
                continue
            seen.add(value)
            unique.append(value)
        return unique

    def _format_label(self, entry: LineageTreeViewer.LineageEntry) -> Text:
        """Format the label for presentation by this lineage tree viewer."""
        label = Text(entry.display_name or entry.item_uuid)
        if entry.version_or_framework:
            label.append(f"  {entry.version_or_framework}", style="dim")
        if entry.is_current:
            label.append("  [CURRENT]", style="bold green")
        if entry.child_count:
            suffix = "child" if entry.child_count == 1 else "children"
            label.append(f"  ({entry.child_count} {suffix})", style="dim")
        return label

    def _empty_message(self) -> str:
        """Return the empty message value owned by this lineage tree viewer."""
        return "No model selected." if self._kind == "model" else "No session selected."

    def _render_details(
        self,
        entry: LineageTreeViewer.LineageEntry | None,
    ) -> None:
        """Handle the render details operation owned by this lineage tree viewer."""
        if not self._show_details:
            return

        details = self.query_one(f"#{self._details_id}", Static)
        if entry is None:
            details.update(self._empty_message())
            return

        services = self._APPLICATION.get_application_services()
        if self._kind == "model":
            title = "Model"
            display_name = entry.display_name or entry.item_uuid
            parent_label = (
                services.inspection.presentation.get_model_reference_display(entry.parent_uuid)
                if entry.parent_uuid
                else "-"
            )
            aux_label = "Version"
        else:
            title = "Session"
            display_name = (
                services.sessions.metadata.get_session_display_name_without_uuid(entry.item_uuid)
                or entry.display_name
                or entry.item_uuid
            )
            parent_label = (
                services.sessions.metadata.get_session_reference_display(entry.parent_uuid)
                if entry.parent_uuid
                else "-"
            )
            aux_label = "Framework"

        child_suffix = "child" if entry.child_count == 1 else "children"
        rows = [
            (title, display_name),
            ("UUID", entry.identity_uuid or entry.item_uuid),
            (aux_label, entry.version_or_framework or "-"),
            ("Parent", parent_label),
            ("Children", f"{entry.child_count} {child_suffix}"),
            ("Created At", format_local_timestamp(entry.created_at_raw)),
            ("Modified At", format_local_timestamp(entry.modified_at_raw)),
        ]

        rendered = Text()
        for label, value in rows:
            rendered.append(f"{label:<{self._DETAIL_LABEL_WIDTH}}", style="dim")
            rendered.append("  ")
            rendered.append(str(value or "-"), style="bold")
            rendered.append("\n")
        rendered.append("Description", style="dim")
        rendered.append("\n")
        rendered.append(entry.description or "(no description)", style="bold")
        details.update(rendered)

    def _build_model_entries(
        self,
    ) -> tuple[dict[str, LineageTreeViewer.LineageEntry], dict[str | None, list[str]]]:
        """Build the model entries required by this lineage tree viewer."""
        services = self._APPLICATION.get_application_services()
        nodes, children_by_parent = services.sessions.lineage.build_model_lineage_index()
        entries: dict[str, LineageTreeViewer.LineageEntry] = {}

        for model_uuid, node in nodes.items():
            version = str(node.model_version or "").strip()
            child_uuids = list(children_by_parent.get(model_uuid, []))
            search_text = " ".join(
                [
                    node.display_name,
                    model_uuid,
                    version,
                    node.description,
                    node.parent_model_uuid or "",
                    node.created_at_raw or "",
                    node.modified_at_raw or "",
                ]
            ).lower()
            entries[model_uuid] = self.LineageEntry(
                item_uuid=model_uuid,
                kind="model",
                display_name=node.display_name,
                identity_uuid=model_uuid,
                version_or_framework=version,
                parent_uuid=node.parent_model_uuid if node.parent_model_uuid in nodes else None,
                description=node.description,
                created_at_raw=node.created_at_raw,
                modified_at_raw=node.modified_at_raw,
                is_current=model_uuid == self._item_uuid,
                child_count=len(child_uuids),
                search_text=search_text,
            )

        cleaned_children: dict[str | None, list[str]] = {}
        for parent_uuid, child_uuids in children_by_parent.items():
            cleaned_children[parent_uuid] = self._dedupe(
                [uuid for uuid in child_uuids if uuid in entries]
            )
        return entries, cleaned_children

    def _build_session_entries(
        self,
    ) -> tuple[dict[str, LineageTreeViewer.LineageEntry], dict[str | None, list[str]]]:
        """Build the session entries required by this lineage tree viewer."""
        services = self._APPLICATION.get_application_services()
        nodes, children_by_parent = services.sessions.lineage.build_session_lineage_index()

        entries: dict[str, LineageTreeViewer.LineageEntry] = {}
        for session_uuid, node in nodes.items():
            framework_uuid = node.framework_uuid or "-"
            search_text = " ".join(
                [
                    node.display_name,
                    session_uuid,
                    node.identity_uuid,
                    framework_uuid,
                    node.description,
                    node.parent_session_uuid or "",
                    node.created_at_raw or "",
                    node.modified_at_raw or "",
                ]
            ).lower()
            entries[session_uuid] = self.LineageEntry(
                item_uuid=session_uuid,
                kind="session",
                display_name=node.display_name,
                identity_uuid=node.identity_uuid,
                version_or_framework=framework_uuid,
                parent_uuid=node.parent_session_uuid,
                description=node.description,
                created_at_raw=node.created_at_raw,
                modified_at_raw=node.modified_at_raw,
                is_current=session_uuid == self._item_uuid,
                child_count=node.child_count,
                search_text=search_text,
            )

        return entries, children_by_parent

    def _build_entries(
        self,
    ) -> tuple[dict[str, LineageTreeViewer.LineageEntry], dict[str | None, list[str]]]:
        """Build the entries required by this lineage tree viewer."""
        if self._compact and not self._item_uuid:
            return {}, {}
        if self._kind == "model":
            return self._build_model_entries()
        return self._build_session_entries()

    def _visible_with_ancestors(
        self,
        entries: dict[str, LineageTreeViewer.LineageEntry],
        filter_text: str,
    ) -> set[str]:
        """Return the visible with ancestors value owned by this lineage tree viewer."""
        normalized_filter = str(filter_text or "").strip().lower()
        if not normalized_filter:
            return set(entries)

        visible: set[str] = set()
        for item_uuid, entry in entries.items():
            if normalized_filter not in entry.search_text:
                continue
            current_uuid: str | None = item_uuid
            visited: set[str] = set()
            while current_uuid and current_uuid in entries and current_uuid not in visited:
                visited.add(current_uuid)
                visible.add(current_uuid)
                current_uuid = entries[current_uuid].parent_uuid
        return visible

    def _compact_visible_set(
        self,
        entries: dict[str, LineageTreeViewer.LineageEntry],
        children_by_parent: dict[str | None, list[str]],
    ) -> set[str]:
        """Return the compact visible set value owned by this lineage tree viewer."""
        if not self._item_uuid or self._item_uuid not in entries:
            return set()
        current_entry = entries[self._item_uuid]
        visible = {self._item_uuid}
        if current_entry.parent_uuid and current_entry.parent_uuid in entries:
            visible.add(current_entry.parent_uuid)
        visible.update(children_by_parent.get(self._item_uuid, []))
        return visible

    def _root_uuids(
        self,
        entries: dict[str, LineageTreeViewer.LineageEntry],
        visible: set[str],
    ) -> list[str]:
        """Return the root uuids value owned by this lineage tree viewer."""
        roots: list[str] = []
        for item_uuid, entry in entries.items():
            if item_uuid not in visible:
                continue
            if (
                not entry.parent_uuid
                or entry.parent_uuid not in entries
                or entry.parent_uuid not in visible
            ):
                roots.append(item_uuid)
        roots.sort(key=lambda uuid: (entries[uuid].display_name.lower(), uuid))
        return roots

    def _populate_tree(self) -> None:
        """Return the populate tree value owned by this lineage tree viewer."""
        tree = self.query_one(f"#{self._tree_id}", Tree)
        tree.clear()
        tree.root.expand()

        entries, children_by_parent = self._build_entries()
        if not entries:
            tree.root.add_leaf(Text(self._empty_message(), style="dim"), data=None)
            self._render_details(None)
            return

        visible = (
            self._compact_visible_set(entries, children_by_parent)
            if self._compact
            else self._visible_with_ancestors(entries, self._filter_text)
        )
        if not visible:
            tree.root.add_leaf(Text("No matching lineage nodes.", style="dim"), data=None)
            self._render_details(None)
            return

        selected_node: TreeNode[LineageTreeViewer.LineageEntry] | None = None
        visited_path: set[str] = set()

        def add_node(
            parent_node: TreeNode[LineageTreeViewer.LineageEntry],
            item_uuid: str,
        ) -> TreeNode[LineageTreeViewer.LineageEntry] | None:
            """Add the node owned by this lineage tree viewer."""
            nonlocal selected_node
            if item_uuid in visited_path:
                return None
            if item_uuid not in visible or item_uuid not in entries:
                return None

            visited_path.add(item_uuid)
            entry = entries[item_uuid]
            node = parent_node.add(
                self._format_label(entry),
                data=entry,
                expand=True,
                allow_expand=bool(
                    [
                        child_uuid
                        for child_uuid in children_by_parent.get(item_uuid, [])
                        if child_uuid in visible
                    ]
                ),
            )
            if item_uuid == self._item_uuid:
                selected_node = node

            for child_uuid in children_by_parent.get(item_uuid, []):
                add_node(node, child_uuid)
            visited_path.remove(item_uuid)
            return node

        for root_uuid in self._root_uuids(entries, visible):
            add_node(tree.root, root_uuid)

        selected_entry = entries.get(self._item_uuid or "")
        if selected_node is None:
            first_child = tree.root.children[0] if tree.root.children else None
            if first_child is not None and isinstance(first_child.data, self.LineageEntry):
                selected_node = first_child
                selected_entry = first_child.data

        self._render_details(selected_entry)
        if selected_node is not None:
            tree.move_cursor(selected_node, animate=False)
            tree.scroll_to_node(selected_node, animate=False)

    def set_item(self, item_uuid: str | None) -> None:
        """Set the item owned by this lineage tree viewer."""
        self._item_uuid = str(item_uuid or "").strip() or None
        if self.is_mounted:
            self.refresh_lineage()

    def set_model(self, model_uuid: str | None) -> None:
        """Set the model owned by this lineage tree viewer."""
        self._kind = "model"
        self.set_item(model_uuid)

    def set_session(self, session_uuid: str | None) -> None:
        """Set the session owned by this lineage tree viewer."""
        self._kind = "session"
        self.set_item(session_uuid)

    def refresh_lineage(self) -> None:
        """Refresh the lineage owned by this lineage tree viewer."""
        self._populate_tree()

    def on_mount(self) -> None:
        """Handle the mount event for this lineage tree viewer."""
        self._populate_tree()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this lineage tree viewer."""
        if event.input.id != self._search_id:
            return
        self._filter_text = event.value
        self._populate_tree()

    def on_tree_node_highlighted(
        self,
        event: Tree.NodeHighlighted[LineageTreeViewer.LineageEntry],
    ) -> None:
        """Handle the tree node highlighted event for this lineage tree viewer."""
        entry = event.node.data if isinstance(event.node.data, self.LineageEntry) else None
        self._render_details(entry)

    def on_tree_node_selected(
        self,
        event: Tree.NodeSelected[LineageTreeViewer.LineageEntry],
    ) -> None:
        """Handle the tree node selected event for this lineage tree viewer."""
        entry = event.node.data if isinstance(event.node.data, self.LineageEntry) else None
        self._render_details(entry)
