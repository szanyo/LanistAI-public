# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for comparison pack builder."""

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from typing import ClassVar, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.services.async_collection_loader import run_blocking
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.tensorboard_metadata import (
    compact_tensorboard_metadata,
    tensorboard_metadata_rows,
)
from lanista.gui.utils.tree_nodes import first_string_data_node
from lanista.gui.widgets.dialogs.button_templates import (
    cancel_button,
    destructive_button,
    dialog_action_row,
    primary_button,
    success_button,
    warning_button,
)
from lanista.gui.widgets.dialogs.comparison_pack_curation_dialog import (
    ComparisonPackCurationDialog,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.reinforcengine.application_errors import OperationResult
from lanista.reinforcengine.rl_model.comparison.contracts import (
    ComparisonPackCreationOutcome,
    ComparisonPackMutationOutcome,
)
from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import TensorboardPackEntry

LOGGER = logging.getLogger(__name__)


@dataclass
class ComparisonPackBuilderDialogResult:
    """Report the result of comparison pack builder dialog."""

    action: str
    pack_name: str
    logdir: str | None = None
    source_count: int = 0


class ComparisonPackBuilderDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[ComparisonPackBuilderDialogResult | None],
):
    """Present comparison pack builder in the Textual interface."""

    _HELP_TOPIC = "comparison_pack"
    _APPLICATION = LanistAIApplication()

    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        Binding("f2,a", "shortcut_approve", "Add", key_display="f2"),
        Binding("f3,r", "shortcut_remove", "Remove", key_display="f3"),
        Binding("f4,e", "shortcut_save_description", "Save description", key_display="f4"),
        Binding("f5,u", "shortcut_update_sources", "Update sources", key_display="f5"),
        Binding("f6,n", "shortcut_rename_pack", "Rename pack", key_display="f6"),
        Binding("f7,d", "shortcut_delete_pack", "Delete pack", key_display="f7"),
        Binding("enter", "shortcut_confirm", "Create", key_display="enter"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "compare_pack_builder_name": {
            "down": "compare_pack_builder_source_tree",
        },
        "compare_pack_builder_source_tree": {
            "up": "compare_pack_builder_name",
            "right": "compare_pack_builder_add",
        },
        "compare_pack_builder_add": {
            "left": "compare_pack_builder_source_tree",
            "right": "compare_pack_builder_remove",
            "down": "compare_pack_builder_comparison_tree",
        },
        "compare_pack_builder_remove": {
            "left": "compare_pack_builder_add",
            "down": "compare_pack_builder_comparison_tree",
        },
        "compare_pack_builder_comparison_tree": {
            "up": "compare_pack_builder_add",
            "down": "compare_pack_builder_rename_name",
        },
        "compare_pack_builder_rename_name": {
            "up": "compare_pack_builder_comparison_tree",
            "down": "compare_pack_builder_description_input",
        },
        "compare_pack_builder_description_input": {
            "up": "compare_pack_builder_rename_name",
            "down": "compare_pack_builder_save_description",
        },
        "compare_pack_builder_save_description": {
            "up": "compare_pack_builder_description_input",
            "right": "compare_pack_builder_update_sources",
            "down": "compare_pack_builder_rename",
        },
        "compare_pack_builder_update_sources": {
            "up": "compare_pack_builder_description_input",
            "left": "compare_pack_builder_save_description",
            "down": "compare_pack_builder_delete",
        },
        "compare_pack_builder_rename": {
            "up": "compare_pack_builder_save_description",
            "right": "compare_pack_builder_delete",
            "down": "compare_pack_builder_create",
        },
        "compare_pack_builder_delete": {
            "up": "compare_pack_builder_update_sources",
            "left": "compare_pack_builder_rename",
            "down": "compare_pack_builder_cancel",
        },
        "compare_pack_builder_create": {
            "up": "compare_pack_builder_rename",
            "right": "compare_pack_builder_cancel",
        },
        "compare_pack_builder_cancel": {
            "up": "compare_pack_builder_delete",
            "left": "compare_pack_builder_create",
        },
    }

    def __init__(self) -> None:
        """Initialize the comparison pack builder dialog and its owned state."""
        super().__init__()
        self._pack_service = (
            self._APPLICATION.get_application_services().administration.tensorboard_packs
        )
        self._curation_dialog = ComparisonPackCurationDialog(self._pack_service)
        self._source_entries: list[TensorboardPackEntry] = []
        self._comparison_entries: list[TensorboardPackEntry] = []
        self._selected_paths: list[str] = []
        self._source_value: str | None = None
        self._selected_comparison_value: str | None = None
        self._entry_load_request_id = 0
        self._entry_load_task: asyncio.Task[None] | None = None
        self._loading_entries = False
        self._entry_load_error: BaseException | None = None

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this comparison pack builder dialog."""
        with Vertical(id="compare_pack_builder_dialog_container") as container:
            container.border_title = "Comparison Packs"
            with (
                ScrollableContainer(id="compare_pack_builder_scroll"),
                Vertical(id="compare_pack_builder_content"),
            ):
                yield Static(
                    "Create or update curated TensorBoard comparison packs from training and evaluation runs.",
                    id="compare_pack_builder_hint",
                )
                with Vertical(id="compare_pack_builder_workspace"):
                    with Vertical(id="compare_pack_builder_sources_panel"):
                        yield Input(
                            value=f"compare_{datetime.now(UTC).strftime('%Y-%m-%d_%H-%M-%S')}",
                            placeholder="New comparison pack name",
                            id="compare_pack_builder_name",
                        )

                        source_tree: Tree[str | None] = Tree(
                            "Available TensorBoard Sources",
                            id="compare_pack_builder_source_tree",
                        )
                        source_tree.border_title = "Available Source Runs"
                        source_tree.show_root = False
                        source_tree.show_guides = True
                        with Horizontal(id="compare_pack_builder_source_region"):
                            yield source_tree
                            yield SelectionDetailsPanel(
                                title="Selected Sources",
                                empty_message="No source runs selected.",
                                id="compare_pack_builder_selected",
                            )

                    yield dialog_action_row(
                        primary_button(
                            "Add",
                            id="compare_pack_builder_add",
                            disabled=not bool(self._source_entries),
                        ),
                        warning_button(
                            "Remove",
                            id="compare_pack_builder_remove",
                            disabled=True,
                        ),
                        id="compare_pack_builder_source_actions",
                    )

                    with Vertical(id="compare_pack_builder_admin_panel"):
                        with Horizontal(id="compare_pack_builder_review_region"):
                            comparison_tree: Tree[str | None] = Tree(
                                "Existing Comparison Packs",
                                id="compare_pack_builder_comparison_tree",
                            )
                            comparison_tree.border_title = "Existing Packs"
                            comparison_tree.show_root = False
                            comparison_tree.show_guides = False
                            yield comparison_tree

                            yield SelectionDetailsPanel(
                                title="Selected Pack",
                                empty_message="No existing pack selected.",
                                id="compare_pack_builder_comparison_summary",
                            )

                        rename_input = Input(
                            value="",
                            placeholder="Selected comparison pack name",
                            id="compare_pack_builder_rename_name",
                            disabled=not bool(self._comparison_entries),
                        )
                        rename_input.border_title = "Pack Name"
                        rename_input.display = bool(self._comparison_entries)
                        yield rename_input

                        description_input = TextArea(
                            text="",
                            id="compare_pack_builder_description_input",
                            disabled=not bool(self._comparison_entries),
                        )
                        description_input.border_title = "Description"
                        description_input.display = bool(self._comparison_entries)
                        yield description_input

                        yield dialog_action_row(
                            primary_button(
                                "Save Description",
                                id="compare_pack_builder_save_description",
                                disabled=not bool(self._comparison_entries),
                            ),
                            warning_button(
                                "Update Sources",
                                id="compare_pack_builder_update_sources",
                                disabled=not bool(self._comparison_entries),
                            ),
                            id="compare_pack_builder_admin_actions",
                            displayed=bool(self._comparison_entries),
                        )

                        yield dialog_action_row(
                            primary_button(
                                "Rename Pack",
                                id="compare_pack_builder_rename",
                                disabled=not bool(self._comparison_entries),
                            ),
                            destructive_button(
                                "Delete Pack",
                                id="compare_pack_builder_delete",
                                disabled=not bool(self._comparison_entries),
                            ),
                            id="compare_pack_builder_admin_danger_actions",
                            displayed=bool(self._comparison_entries),
                        )

                    yield dialog_action_row(
                        success_button(
                            "Create",
                            id="compare_pack_builder_create",
                            disabled=not bool(self._source_entries),
                        ),
                        cancel_button(id="compare_pack_builder_cancel"),
                        id="compare_pack_builder_create_actions",
                    )

                yield Static("", id="compare_pack_builder_status")
                yield Static(
                    "Select two or more source runs, then create a new pack or update the selected existing pack.",
                    id="compare_pack_builder_footer_hint",
                )

            yield ModalFooter.from_binding_rows(
                cast(list[Binding], self.BINDINGS),
                [
                    ["shortcut_approve", "shortcut_remove", "shortcut_confirm"],
                    ["shortcut_save_description", "shortcut_update_sources", "dismiss(None)"],
                    ["shortcut_rename_pack", "shortcut_delete_pack", "open_help_dialog"],
                ],
                id="compare_pack_builder_modal_footer",
                label_overrides={
                    "shortcut_approve": "Add",
                    "shortcut_remove": "Remove",
                    "shortcut_confirm": "Create",
                    "shortcut_save_description": "Save",
                    "shortcut_update_sources": "Update",
                    "shortcut_rename_pack": "Rename",
                    "shortcut_delete_pack": "Delete",
                    "dismiss(None)": "Cancel",
                },
                max_per_row=3,
            )

    def action_shortcut_approve(self) -> None:
        """Handle the shortcut approve user action for this comparison pack builder dialog."""
        self._press_button_if_enabled("compare_pack_builder_add")

    def action_shortcut_remove(self) -> None:
        """Handle the shortcut remove user action for this comparison pack builder dialog."""
        self._press_button_if_enabled("compare_pack_builder_remove")

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this comparison pack builder dialog."""
        self._press_button_if_enabled("compare_pack_builder_create")

    def action_shortcut_save_description(self) -> None:
        """Handle the shortcut save description user action for this comparison pack builder
        dialog.
        """
        self._press_button_if_enabled("compare_pack_builder_save_description")

    def action_shortcut_update_sources(self) -> None:
        """Handle the shortcut update sources user action for this comparison pack builder dialog.
        """
        self._press_button_if_enabled("compare_pack_builder_update_sources")

    def action_shortcut_rename_pack(self) -> None:
        """Handle the shortcut rename pack user action for this comparison pack builder dialog."""
        self._press_button_if_enabled("compare_pack_builder_rename")

    def action_shortcut_delete_pack(self) -> None:
        """Handle the shortcut delete pack user action for this comparison pack builder dialog."""
        self._press_button_if_enabled("compare_pack_builder_delete")

    def _press_button_if_enabled(self, button_id: str) -> None:
        """Return whether press button if enabled applies to this comparison pack builder dialog."""
        button = self.query_one(f"#{button_id}", Button)
        if not getattr(button, "display", True) or button.disabled:
            return
        button.press()

    def on_key(self, event: Key) -> None:
        """Apply the directional-focus contract used by the established dialogs."""

        focused_id = self.focused.id if self.focused is not None else None
        if focused_id in {
            "compare_pack_builder_source_tree",
            "compare_pack_builder_comparison_tree",
        } and event.key in {"up", "down"}:
            return
        if focused_id == "compare_pack_builder_description_input":
            return
        if focused_id in {
            "compare_pack_builder_name",
            "compare_pack_builder_rename_name",
        } and event.key in {"left", "right", "up"}:
            return
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(
            event.key
        ):
            event.stop()
            event.prevent_default()

    @staticmethod
    def _compact_run_label(run_name: str, *, max_length: int = 28) -> str:
        """Return the compact run label value owned by this comparison pack builder dialog."""
        text = str(run_name or "").strip()
        if len(text) <= max_length:
            return text
        if "__" in text:
            prefix, suffix = text.rsplit("__", 1)
            compact = f"{prefix[:8]}..{suffix}"
            if len(compact) <= max_length:
                return compact
        return f"{text[: max_length - 1]}~"

    @classmethod
    def _source_label(
        cls,
        entry: TensorboardPackEntry,
        *,
        include_model: bool = False,
    ) -> str:
        """Return the source label value owned by this comparison pack builder dialog."""
        run_label = cls._compact_run_label(entry.run_name)
        prefix = f"{entry.pack_type}/{entry.framework_uuid}"
        compact_metadata = compact_tensorboard_metadata(entry)
        suffix = f"  [{compact_metadata}]" if compact_metadata else ""
        if include_model:
            model_label = entry.model_display_name or entry.model_name or "model"
            return f"{model_label}  {prefix}  {run_label}{suffix}"
        return f"{prefix}  {run_label}{suffix}"

    def _build_selected_text(self) -> str:
        """Build the selected text required by this comparison pack builder dialog."""
        if not self._selected_paths:
            return "No sources selected."

        by_path = {entry.path: entry for entry in self._source_entries}
        lines = [f"{len(self._selected_paths)} selected source runs:"]
        for path in self._selected_paths:
            entry = by_path.get(path)
            if entry is None:
                continue
            lines.append(f"- {self._source_label(entry, include_model=True)}")
        return "\n".join(lines)

    def _selected_detail_rows(self) -> list[SelectionDetailsPanel.DetailRow]:
        """Project selected sources through the shared details-panel row contract."""

        by_path = {entry.path: entry for entry in self._source_entries}
        rows: list[SelectionDetailsPanel.DetailRow] = []
        for index, path in enumerate(self._selected_paths, start=1):
            entry = by_path.get(path)
            rows.append(
                SelectionDetailsPanel.DetailRow(
                    f"Source {index}",
                    self._source_label(entry, include_model=True) if entry is not None else path,
                    value_style="",
                )
            )
        return rows

    def _build_comparison_summary_text(
        self,
        entry: TensorboardPackEntry | None,
    ) -> str:
        """Build the comparison summary text required by this comparison pack builder dialog."""
        if entry is None:
            return "No existing pack selected."

        source_paths = [
            str(source_path or "").strip()
            for source_path in (entry.source_paths or [])
            if str(source_path or "").strip()
        ]
        lines = [
            f"Pack: {entry.run_name}",
            f"Sources: {len(source_paths)}",
        ]
        if entry.annotation:
            lines.append(f"Description: {entry.annotation}")
        if source_paths:
            by_path = {source_entry.path: source_entry for source_entry in self._source_entries}
            lines.append("Sources:")
            for source_path in source_paths:
                source_entry = by_path.get(source_path)
                if source_entry is None:
                    lines.append(f"- {source_path}")
                    continue
                lines.append(f"- {self._source_label(source_entry, include_model=True)}")
        return "\n".join(lines)

    def _comparison_detail_rows(
        self,
        entry: TensorboardPackEntry | None,
    ) -> list[SelectionDetailsPanel.DetailRow]:
        """Project one existing pack through the shared details-panel row contract."""

        if entry is None:
            return []
        rows = [
            SelectionDetailsPanel.DetailRow("Pack", entry.run_name),
            SelectionDetailsPanel.DetailRow("Sources", str(len(entry.source_paths or []))),
        ]
        rows.extend(
            SelectionDetailsPanel.DetailRow(
                metadata_row.label,
                metadata_row.value,
                value_style=metadata_row.value_style,
            )
            for metadata_row in tensorboard_metadata_rows(entry)
        )
        if entry.annotation:
            rows.append(
                SelectionDetailsPanel.DetailRow(
                    "Description",
                    entry.annotation,
                    value_style="",
                )
            )
        return rows

    def _refresh_selected_state(self) -> None:
        """Refresh the selected state owned by this comparison pack builder
dialog."""
        selected_panel = self.query_one(
            "#compare_pack_builder_selected",
            SelectionDetailsPanel,
        )
        selected_count = len(self._selected_paths)
        selected_panel.set_title(f"Selected Sources ({selected_count})")
        if selected_count:
            selected_panel.show_rows(self._selected_detail_rows())
        else:
            selected_panel.show_message()
        remove_button = self.query_one("#compare_pack_builder_remove", Button)
        remove_button.disabled = not bool(self._selected_paths)
        create_button = self.query_one("#compare_pack_builder_create", Button)
        create_button.disabled = len(self._selected_paths) < 2

        update_button = self.query_one("#compare_pack_builder_update_sources", Button)
        update_button.disabled = (
            self._selected_comparison_entry() is None or len(self._selected_paths) < 2
        )

    def _set_status(self, message: str) -> None:
        """Set the status owned by this comparison pack builder dialog."""
        self.query_one("#compare_pack_builder_status", Static).update(message)

    def _refresh_comparison_entries(self, selected_pack_name: str | None = None) -> None:
        """Refresh the comparison entries owned by this comparison pack
builder dialog."""
        self._comparison_entries = [
            entry
            for entry in self._pack_service.list_for_current_model()
            if entry.pack_type == "comparison"
        ]
        if selected_pack_name:
            selected_entry = next(
                (
                    entry
                    for entry in self._comparison_entries
                    if entry.run_name == selected_pack_name
                ),
                None,
            )
            self._selected_comparison_value = (
                selected_entry.path if selected_entry is not None else None
            )
        elif self._comparison_entries and self._selected_comparison_value is None:
            self._selected_comparison_value = self._comparison_entries[0].path
        self._rebuild_comparison_tree()

    def _rebuild_source_tree(self) -> None:
        """Rebuild the source tree owned by this comparison pack builder dialog."""
        tree = self.query_one("#compare_pack_builder_source_tree", Tree)
        tree.clear()
        tree.root.expand()

        if self._loading_entries:
            empty_node = tree.root.add_leaf(
                Text("Loading TensorBoard source metadata...", style="dim"),
                data=None,
            )
            empty_node.allow_expand = False
            return
        if not self._source_entries:
            empty_node = tree.root.add_leaf(
                Text(
                    (
                        "TensorBoard metadata could not be loaded."
                        if self._entry_load_error is not None
                        else "No training or evaluation TensorBoard packs available."
                    ),
                    style="dim",
                ),
                data=None,
            )
            empty_node.allow_expand = False
            return

        selected_node: TreeNode[str | None] | None = None
        groups: dict[str, TreeNode[str | None]] = {}
        for entry in self._source_entries:
            model_label = entry.model_display_name or entry.model_name or "model"
            group = groups.get(model_label)
            if group is None:
                group = tree.root.add(model_label, expand=True, data=None)
                groups[model_label] = group
            label = self._source_label(entry)
            node = group.add_leaf(label, data=entry.path)
            node.allow_expand = False
            if entry.path == self._source_value:
                selected_node = node

        if selected_node is None:
            selected_node = first_string_data_node(tree.root)
        if selected_node is not None and isinstance(selected_node.data, str):
            self._source_value = selected_node.data
            tree.select_node(selected_node)

    def _rebuild_comparison_tree(self) -> None:
        """Rebuild the comparison tree owned by this comparison pack builder dialog."""
        tree = self.query_one("#compare_pack_builder_comparison_tree", Tree)
        tree.clear()
        tree.root.expand()

        if self._loading_entries:
            empty_node = tree.root.add_leaf(
                Text("Loading comparison-pack metadata...", style="dim"),
                data=None,
            )
            empty_node.allow_expand = False
            self._selected_comparison_value = None
            self._refresh_description_input()
            return
        if not self._comparison_entries:
            empty_node = tree.root.add_leaf(
                Text("No comparison packs available yet.", style="dim"), data=None
            )
            empty_node.allow_expand = False
            self._selected_comparison_value = None
            self._refresh_description_input()
            return

        selected_node: TreeNode[str | None] | None = None
        for entry in self._comparison_entries:
            compact_metadata = compact_tensorboard_metadata(entry)
            label = entry.run_name
            if compact_metadata:
                label = f"{label}  [{compact_metadata}]"
            node = tree.root.add_leaf(label, data=entry.path)
            node.allow_expand = False
            if entry.path == self._selected_comparison_value:
                selected_node = node

        if selected_node is None:
            selected_node = tree.root.children[0] if tree.root.children else None
        if selected_node is not None and isinstance(selected_node.data, str):
            self._selected_comparison_value = selected_node.data
            tree.select_node(selected_node)
        self._refresh_description_input()

    def _selected_comparison_entry(
        self,
    ) -> TensorboardPackEntry | None:
        """Return the selected comparison entry value owned by this comparison pack builder dialog.
        """
        for entry in self._comparison_entries:
            if entry.path == self._selected_comparison_value:
                return entry
        return None

    def _select_source_path(self, path: str | None) -> None:
        """Select the source path owned by this comparison pack builder dialog."""
        selected_path = str(path or "").strip()
        if not selected_path:
            return
        tree = self.query_one("#compare_pack_builder_source_tree", Tree)
        node = self._find_node_with_data(tree.root, selected_path)
        if node is not None:
            tree.select_node(node)

    def _find_node_with_data(
        self,
        node: TreeNode[str | None],
        path: str,
    ) -> TreeNode[str | None] | None:
        """Find the node with data required by this comparison pack builder dialog."""
        for child in node.children:
            if child.data == path:
                return child
            nested = self._find_node_with_data(child, path)
            if nested is not None:
                return nested
        return None

    def _refresh_description_input(self) -> None:
        """Refresh the description input owned by this comparison pack
builder dialog."""
        description_input = self.query_one("#compare_pack_builder_description_input", TextArea)
        rename_input = self.query_one("#compare_pack_builder_rename_name", Input)
        save_button = self.query_one("#compare_pack_builder_save_description", Button)
        rename_button = self.query_one("#compare_pack_builder_rename", Button)
        delete_button = self.query_one("#compare_pack_builder_delete", Button)
        summary = self.query_one(
            "#compare_pack_builder_comparison_summary",
            SelectionDetailsPanel,
        )
        entry = self._selected_comparison_entry()
        admin_visible = entry is not None
        description_input.display = admin_visible
        rename_input.display = admin_visible
        self.query_one("#compare_pack_builder_admin_actions").display = admin_visible
        self.query_one("#compare_pack_builder_admin_danger_actions").display = admin_visible
        description_input.disabled = entry is None
        rename_input.disabled = entry is None
        save_button.disabled = entry is None
        rename_button.disabled = entry is None
        delete_button.disabled = entry is None
        description_input.text = entry.annotation or "" if entry is not None else ""
        rename_input.value = entry.run_name if entry is not None else ""
        summary_rows = self._comparison_detail_rows(entry)
        if summary_rows:
            summary.show_rows(summary_rows)
        else:
            summary.show_message()
        if entry is None:
            self._refresh_selected_state()
            return

        source_paths = [
            str(source_path or "").strip()
            for source_path in (entry.source_paths or [])
            if str(source_path or "").strip()
        ]
        self._selected_paths = source_paths
        if source_paths:
            self._source_value = source_paths[0]
            self._select_source_path(self._source_value)
        self._refresh_selected_state()

    def on_mount(self) -> None:
        """Handle the mount event for this comparison pack builder dialog."""
        self._loading_entries = True
        self._rebuild_source_tree()
        self._rebuild_comparison_tree()
        self._refresh_selected_state()
        self._focus_if_navigable("compare_pack_builder_cancel")
        self._start_entry_load()

    def on_unmount(self) -> None:
        """Invalidate detached metadata loading when the dialog closes."""

        self._entry_load_request_id += 1
        if self._entry_load_task is not None and not self._entry_load_task.done():
            self._entry_load_task.cancel()

    def _start_entry_load(self) -> None:
        """Load source and comparison metadata without blocking Textual."""

        self._entry_load_request_id += 1
        request_id = self._entry_load_request_id

        def _collect() -> tuple[
            list[TensorboardPackEntry],
            list[TensorboardPackEntry],
        ]:
            source_entries = self._pack_service.list_sources_for_comparison(
                include_all_models=True
            )
            comparison_entries = [
                entry
                for entry in self._pack_service.list_for_current_model()
                if entry.pack_type == "comparison"
            ]
            return source_entries, comparison_entries

        async def _load() -> None:
            try:
                source_entries, comparison_entries = await run_blocking(_collect)
            except asyncio.CancelledError:
                return
            except BaseException as error:
                self._apply_entry_load_result(request_id, (), (), error)
                return
            self._apply_entry_load_result(
                request_id,
                source_entries,
                comparison_entries,
                None,
            )

        self._entry_load_task = asyncio.create_task(_load())
        self._entry_load_task.add_done_callback(self._observe_entry_load_task)

    def _apply_entry_load_result(
        self,
        request_id: int,
        source_entries: Sequence[TensorboardPackEntry],
        comparison_entries: Sequence[TensorboardPackEntry],
        error: BaseException | None,
    ) -> None:
        """Apply only the latest metadata collection to a mounted dialog."""

        if request_id != self._entry_load_request_id or not self.is_mounted:
            return
        self._loading_entries = False
        self._entry_load_error = error
        self._source_entries = list(source_entries)
        self._comparison_entries = list(comparison_entries)
        self._source_value = (
            self._source_entries[0].path if self._source_entries else None
        )
        self._selected_comparison_value = (
            self._comparison_entries[0].path if self._comparison_entries else None
        )
        self._rebuild_source_tree()
        self._rebuild_comparison_tree()
        self._refresh_selected_state()
        if error is not None:
            self._set_status("TensorBoard metadata could not be loaded.")
            return
        if self._source_entries:
            self._focus_if_navigable("compare_pack_builder_name")

    @staticmethod
    def _observe_entry_load_task(task: asyncio.Task[None]) -> None:
        """Log failures escaping the detached metadata loader."""

        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            LOGGER.exception(
                "Comparison-pack metadata rendering failed.",
                exc_info=(type(error), error, error.__traceback__),
            )

    def on_tree_node_selected(
        self,
        event: Tree.NodeSelected[str | None],
    ) -> None:
        """Handle the tree node selected event for this comparison pack builder dialog."""
        if event.control.id == "compare_pack_builder_source_tree":
            self._source_value = event.node.data if isinstance(event.node.data, str) else None
            return
        if event.control.id == "compare_pack_builder_comparison_tree":
            self._selected_comparison_value = (
                event.node.data if isinstance(event.node.data, str) else None
            )
            self._refresh_description_input()

    def _handle_add_source(self) -> None:
        """Handle the add source workflow for this comparison pack builder dialog."""
        selected_path = str(self._source_value or "").strip()
        if selected_path and selected_path not in self._selected_paths:
            self._selected_paths.append(selected_path)
            self._refresh_selected_state()

    def _handle_remove_source(self) -> None:
        """Handle the remove source workflow for this comparison pack builder dialog."""
        if self._selected_paths:
            self._selected_paths.pop()
            self._refresh_selected_state()

    def _handle_create_pack(self) -> None:
        """Handle the create pack workflow for this comparison pack builder dialog."""
        if len(self._selected_paths) < 2:
            self.notify(
                "Select at least two TensorBoard packs to create a comparison pack.",
                title="Compare",
                severity="warning",
                timeout=8,
            )
            return
        pack_name = self.query_one("#compare_pack_builder_name", Input).value
        result = self._pack_service.create_comparison_pack(
            pack_name,
            self._selected_paths,
        )
        outcome = ComparisonPackBuilderDialog._command_outcome(self, result)
        if outcome is None:
            return
        self.dismiss(
            ComparisonPackBuilderDialogResult(
                action="create",
                pack_name=outcome.pack_name,
                logdir=outcome.log_directory,
                source_count=len(self._selected_paths),
            )
        )

    def _handle_save_description(self) -> None:
        """Handle the save description workflow for this comparison pack builder dialog."""
        entry = self._selected_comparison_entry()
        if entry is None:
            return
        result = self._curation_dialog.update_annotation(
            entry.run_name,
            self.query_one("#compare_pack_builder_description_input", TextArea).text,
        )
        outcome = ComparisonPackBuilderDialog._command_outcome(self, result)
        if outcome is None:
            return
        self._refresh_comparison_entries(outcome.pack_name)
        self._set_status(f"Description saved: {entry.run_name}")

    def _handle_update_sources(self) -> None:
        """Handle the update sources workflow for this comparison pack builder dialog."""
        entry = self._selected_comparison_entry()
        if entry is None:
            return
        self._curation_dialog.open(entry.run_name)
        if len(self._selected_paths) < 2:
            self.notify(
                "Select at least two TensorBoard packs before updating a comparison pack.",
                title="Compare",
                severity="warning",
                timeout=8,
            )
            return
        result = self._curation_dialog.update_sources(
            entry.run_name,
            self._selected_paths,
        )
        outcome = ComparisonPackBuilderDialog._command_outcome(self, result)
        if outcome is None:
            return
        self._refresh_comparison_entries(outcome.pack_name)
        self._set_status(
            f"Sources updated: {entry.run_name} ({len(self._selected_paths)} source logs)"
        )

    def _handle_rename_pack(self) -> None:
        """Handle the rename pack workflow for this comparison pack builder dialog."""
        entry = self._selected_comparison_entry()
        if entry is None:
            return
        new_pack_name = self.query_one("#compare_pack_builder_rename_name", Input).value
        result = self._curation_dialog.rename_pack(entry.run_name, new_pack_name)
        outcome = ComparisonPackBuilderDialog._command_outcome(self, result)
        if outcome is None:
            return
        self._refresh_comparison_entries(outcome.pack_name)
        self._set_status(f"Renamed comparison pack: {outcome.pack_name}")

    def _confirm_delete_pack(
        self,
        entry: TensorboardPackEntry,
        confirmed: bool,
    ) -> None:
        """Confirm the delete pack owned by this comparison pack builder dialog."""
        if not confirmed:
            return
        result = self._curation_dialog.delete_pack(entry.run_name)
        outcome = ComparisonPackBuilderDialog._command_outcome(self, result)
        if outcome is None:
            return
        self._selected_paths = []
        self._selected_comparison_value = None
        self._refresh_comparison_entries()
        self._set_status(f"Deleted comparison pack: {outcome.pack_name}")

    def _command_outcome[
        OutcomeT: (
            ComparisonPackCreationOutcome,
            ComparisonPackMutationOutcome,
        )
    ](
        self,
        result: OperationResult[OutcomeT],
    ) -> OutcomeT | None:
        """Return the command outcome value owned by this comparison pack builder dialog."""
        if result.failure is not None:
            self.notify(
                result.failure.message,
                title="Compare",
                severity="error",
                timeout=10,
            )
            return None
        if result.value is None:
            self.notify(
                "The comparison-pack operation returned no committed state.",
                title="Compare",
                severity="error",
                timeout=10,
            )
            return None
        if result.value.degraded:
            self.notify(
                "The comparison pack was committed, but storage recovery and "
                "model reload are required before further administration.",
                title="Compare",
                severity="warning",
                timeout=12,
            )
        return result.value

    def _handle_delete_pack(self) -> None:
        """Handle the delete pack workflow for this comparison pack builder dialog."""
        entry = self._selected_comparison_entry()
        if entry is None:
            return
        from lanista.gui.predefs.secure_delete_dialog_definitions import (
            secure_delete_tensorboard_comparison_pack_dialog,
        )

        self.app.push_screen(
            secure_delete_tensorboard_comparison_pack_dialog(entry.run_name),
            partial(self._confirm_delete_pack, entry),
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this comparison pack builder dialog."""
        handlers = {
            "compare_pack_builder_add": self._handle_add_source,
            "compare_pack_builder_remove": self._handle_remove_source,
            "compare_pack_builder_create": self._handle_create_pack,
            "compare_pack_builder_save_description": self._handle_save_description,
            "compare_pack_builder_update_sources": self._handle_update_sources,
            "compare_pack_builder_rename": self._handle_rename_pack,
            "compare_pack_builder_delete": self._handle_delete_pack,
        }
        handler = handlers.get(event.button.id or "")
        if handler is not None:
            handler()
            return
        self.dismiss(None)
