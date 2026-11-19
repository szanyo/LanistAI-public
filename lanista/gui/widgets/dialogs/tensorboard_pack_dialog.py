# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for tensorboard pack."""

import asyncio
import logging
from dataclasses import dataclass
from functools import partial

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, Tree
from textual.widgets._tree import TreeNode

from lanista.application import LanistAIApplication
from lanista.gui.services.async_collection_loader import (
    AsyncCollectionResult,
    run_blocking,
)
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.tensorboard_metadata import (
    compact_tensorboard_metadata,
    tensorboard_metadata_rows,
)
from lanista.gui.utils.tree_nodes import first_string_data_node
from lanista.gui.widgets.dialogs.button_templates import (
    cancel_button,
    dialog_action_row,
    success_button,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.fragments.model_selector_panel import ModelSelectorPanel
from lanista.gui.widgets.fragments.selection_details_panel import SelectionDetailsPanel
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import TensorboardPackEntry
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest

LOGGER = logging.getLogger(__name__)


@dataclass
class TensorboardPackDialogResult:
    """Report the result of tensorboard pack dialog."""

    pack_type: str
    framework_uuid: str
    run_name: str
    logdir: str
    model_manifest: ModelManifest | None = None


class TensorboardPackDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[TensorboardPackDialogResult | None],
):
    """Present tensorboard pack in the Textual interface."""

    _HELP_TOPIC = "inspect"
    _APPLICATION = LanistAIApplication()

    BINDINGS = [
        Binding("enter", "shortcut_open", "Open", key_display="enter"),
        Binding("f2,v", "dismiss(None)", "Close tensorboard packs"),
        Binding("i", "dismiss(None)", "Close tensorboard packs"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss(None)", "Cancel", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "model_selector_search": {
            "down": "rl_model_tree",
            "right": "rl_model_tree",
        },
        "rl_model_tree": {
            "up": "model_selector_search",
            "down": "tensorboard_pack_filter",
            "right": "tensorboard_pack_filter",
        },
        "tensorboard_pack_filter": {
            "up": "rl_model_tree",
            "down": "tensorboard_pack_tree",
            "right": "tensorboard_pack_tree",
        },
        "tensorboard_pack_tree": {
            "up": "tensorboard_pack_filter",
            "down": "tensorboard_pack_open",
            "right": "tensorboard_pack_open",
        },
        "tensorboard_pack_open": {
            "up": "tensorboard_pack_tree",
            "right": "tensorboard_pack_cancel",
        },
        "tensorboard_pack_cancel": {
            "up": "tensorboard_pack_tree",
            "left": "tensorboard_pack_open",
        },
    }

    def __init__(self) -> None:
        """Initialize the tensorboard pack dialog and its owned state."""
        super().__init__()
        self._pack_entries: tuple[TensorboardPackEntry, ...] = ()
        self._selected_value: str | None = None
        self._loading_packs = False
        self._pack_load_error: BaseException | None = None
        self._pack_render_error: BaseException | None = None
        self._pack_service = (
            self._APPLICATION.get_application_services().administration.tensorboard_packs
        )
        self._selected_manifest: ModelManifest | None = None
        self._selected_model_uuid = ""
        self._pack_request_id = 0
        self._pack_load_task: asyncio.Task | None = None
        configuration = (
            self._APPLICATION.get_application_services()
            .runtime.context.get_active_configuration()
        )
        self._current_model_uuid = str(
            getattr(configuration, "model_uuid", "") or ""
        ).strip()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this tensorboard pack dialog."""
        with Vertical(id="tensorboard_pack_dialog_container") as container:
            container.border_title = "Inspect TensorBoard Pack"
            with (
                ScrollableContainer(id="tensorboard_pack_dialog_scroll"),
                Vertical(id="tensorboard_pack_dialog_content"),
            ):
                yield Static(
                    "Choose a model, then select one of its training, evaluation, "
                    "or comparison packs.",
                    id="tensorboard_pack_hint",
                )
                yield ModelSelectorPanel(
                    current_model_uuid=self._current_model_uuid,
                    id="tensorboard_pack_model_selector",
                )
                with Vertical(id="tensorboard_pack_selection_region"):
                    yield Input(
                        placeholder="Filter by pack type, framework, or name",
                        id="tensorboard_pack_filter",
                    )
                    with Horizontal(id="tensorboard_pack_workspace"):
                        tree = Tree("TensorBoard Packs", id="tensorboard_pack_tree")
                        tree.show_root = False
                        tree.show_guides = True
                        yield tree

                        yield SelectionDetailsPanel(
                            title="Selected TensorBoard Pack",
                            empty_message="No TensorBoard pack available for the current model.",
                            id="tensorboard_pack_details",
                        )

                    yield dialog_action_row(
                        success_button(
                            "Inspect",
                            id="tensorboard_pack_open",
                            disabled=True,
                        ),
                        cancel_button(id="tensorboard_pack_cancel"),
                        id="tensorboard_pack_actions",
                    )
                    yield Static(
                        "Auto-open preferences are available under Settings (F8).",
                        id="tensorboard_pack_preferences_hint",
                    )
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["shortcut_open", "open_help_dialog"], ["dismiss(None)"]],
                id="tensorboard_pack_modal_footer",
                label_overrides={
                    "shortcut_open": "Inspect",
                    "dismiss(None)": "Close",
                },
            )

    def _update_details(self) -> None:
        """Update the details owned by this tensorboard pack dialog."""
        panel = self.query_one("#tensorboard_pack_details", SelectionDetailsPanel)
        entry = self._get_selected_entry()
        if entry is None:
            panel.show_message()
            return

        rows = [
            SelectionDetailsPanel.DetailRow("Pack", entry.run_name),
            SelectionDetailsPanel.DetailRow("Type", entry.pack_type.title()),
        ]
        if entry.framework_uuid:
            rows.append(SelectionDetailsPanel.DetailRow("Framework", entry.framework_uuid))
        if entry.pack_type == "comparison":
            rows.append(
                SelectionDetailsPanel.DetailRow(
                    "Sources",
                    str(max(len(entry.source_paths or ()), entry.metadata.source_run_count if entry.metadata else 0)),
                )
            )
            rows.append(SelectionDetailsPanel.DetailRow("Note", entry.annotation or "-", value_style=""))
        rows.extend(
            SelectionDetailsPanel.DetailRow(
                metadata_row.label,
                metadata_row.value,
                value_style=metadata_row.value_style,
            )
            for metadata_row in tensorboard_metadata_rows(entry)
        )
        rows.append(SelectionDetailsPanel.DetailRow("Logdir", entry.path, value_style=""))
        panel.show_rows(rows)

    def action_shortcut_open(self) -> None:
        """Handle the shortcut open user action for this tensorboard pack dialog."""
        button = self.query_one("#tensorboard_pack_open", Button)
        if button.display and not button.disabled:
            button.press()

    def on_key(self, event: Key) -> None:
        """Apply the directional-focus behavior used by selection dialogs."""

        focused_id = self.focused.id if self.focused is not None else None
        if focused_id in {"rl_model_tree", "tensorboard_pack_tree"} and event.key in {
            "up",
            "down",
        }:
            return
        if focused_id in {"model_selector_search", "tensorboard_pack_filter"} and event.key not in {
            "up",
            "down",
        }:
            return
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(
            event.key
        ):
            event.stop()
            event.prevent_default()

    def _get_selected_entry(self):
        """Return the selected entry exposed by this tensorboard pack dialog."""
        for entry in self._pack_entries:
            if entry.path == self._selected_value:
                return entry
        return None

    def _show_tree_message(self, message: str) -> None:
        """Show the tree message owned by this tensorboard pack dialog."""
        tree = self.query_one("#tensorboard_pack_tree", Tree)
        open_button = self.query_one("#tensorboard_pack_open", Button)
        tree.clear()
        tree.root.expand()
        node = tree.root.add_leaf(message, data=None)
        node.allow_expand = False
        open_button.disabled = True
        self.query_one("#tensorboard_pack_details", SelectionDetailsPanel).show_message(message)

    def _start_pack_load(self, manifest: ModelManifest) -> None:
        """Start the pack load workflow owned by this tensorboard pack dialog."""
        self._loading_packs = True
        self._pack_load_error = None
        self._pack_request_id += 1
        request_id = self._pack_request_id
        if self._pack_load_task is not None and not self._pack_load_task.done():
            self._pack_load_task.cancel()
        self._show_tree_message("Loading TensorBoard packs...")
        configuration = manifest.get_configuration()
        model_folder_path = str(
            getattr(configuration, "model_folder_path", "") or ""
        ).strip()
        model_name = str(getattr(configuration, "model_name", "") or "").strip()
        model_version = str(
            getattr(configuration, "model_version", "") or ""
        ).strip()
        model_display_name = " ".join(
            item for item in (model_name, model_version) if item
        ) or self._selected_model_uuid

        async def _load() -> None:
            """Load and return the state owned by this tensorboard pack dialog."""
            try:
                items = await run_blocking(
                    partial(
                        self._pack_service.list_for_model,
                        model_folder_path,
                        model_display_name,
                    )
                )
                result = AsyncCollectionResult(
                    request_id=request_id,
                    items=tuple(items),
                )
            except asyncio.CancelledError:
                return
            except BaseException as error:
                result = AsyncCollectionResult(
                    request_id=request_id,
                    error=error,
                )
            self._apply_pack_load_result(result)

        self._pack_load_task = asyncio.create_task(_load())
        self._pack_load_task.add_done_callback(self._observe_pack_load_task)

    def _apply_pack_load_result(
        self,
        result: AsyncCollectionResult[TensorboardPackEntry],
    ) -> None:
        """Apply the latest load result while respecting dialog cancellation."""
        if result.request_id != self._pack_request_id or not self.is_mounted:
            return
        if isinstance(result.error, asyncio.CancelledError):
            LOGGER.debug("TensorBoard pack rendering skipped after dialog cancellation.")
            return

        self._loading_packs = False
        self._pack_load_error = result.error
        self._pack_entries = tuple(result.items) if result.ok else ()
        self._selected_value = self._pack_entries[0].path if self._pack_entries else None
        self._rebuild_tree()

    def _observe_pack_load_task(self, task: asyncio.Task[None]) -> None:
        """Record unexpected rendering failures from the detached UI task."""
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        self._pack_render_error = error
        LOGGER.error(
            "TensorBoard pack dialog rendering failed.",
            exc_info=(type(error), error, error.__traceback__),
        )

    def _rebuild_tree(self) -> None:
        """Rebuild the tree owned by this tensorboard pack dialog."""
        tree = self.query_one("#tensorboard_pack_tree", Tree)
        open_button = self.query_one("#tensorboard_pack_open", Button)
        tree.clear()
        tree.root.expand()

        if self._loading_packs:
            self._show_tree_message("Loading TensorBoard packs...")
            return
        if self._pack_load_error is not None:
            self._show_tree_message("Failed to load TensorBoard packs.")
            return

        if not self._pack_entries:
            self._selected_value = None
            empty_node = tree.root.add_leaf("No TensorBoard packs available.", data=None)
            empty_node.allow_expand = False
            open_button.disabled = True
            self._update_details()
            return

        selected_node: TreeNode | None = None
        type_labels = {
            "training": "Training",
            "evaluation": "Evaluation",
            "comparison": "Pack",
        }
        sort_order = {"comparison": 0, "training": 1, "evaluation": 2}
        filter_widget = self.query_one("#tensorboard_pack_filter", Input)
        filter_text = filter_widget.value.strip().casefold()
        visible_entries = tuple(
            entry
            for entry in self._pack_entries
            if not filter_text or filter_text in entry.search_text.casefold()
        )
        if not visible_entries:
            self._selected_value = None
            empty_node = tree.root.add_leaf("No packs match the current filter.", data=None)
            empty_node.allow_expand = False
            open_button.disabled = True
            self._update_details()
            return
        sorted_entries = sorted(
            visible_entries,
            key=lambda entry: (sort_order.get(entry.pack_type, 99), entry.run_name.lower(), entry.path),
        )
        for entry in sorted_entries:
            pack_label = type_labels.get(entry.pack_type, entry.pack_type.title())
            label = f"{pack_label} - {entry.run_name}"
            compact_metadata = compact_tensorboard_metadata(entry)
            if compact_metadata:
                label = f"{label}  [{compact_metadata}]"
            node = tree.root.add_leaf(label, data=entry.path)
            node.allow_expand = False
            if entry.path == self._selected_value:
                selected_node = node

        if selected_node is None:
            selected_node = first_string_data_node(tree.root)
        if selected_node is not None and isinstance(selected_node.data, str):
            self._selected_value = selected_node.data
            tree.select_node(selected_node)
        open_button.disabled = self._selected_value is None
        self._update_details()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Filter the immutable loaded collection without another filesystem read."""

        if event.input.id == "tensorboard_pack_filter":
            self._rebuild_tree()

    def on_mount(self) -> None:
        """Handle the mount event for this tensorboard pack dialog."""
        self._show_tree_message("Select a model to load TensorBoard packs.")
        self._focus_if_navigable("model_selector_search")

    def on_unmount(self) -> None:
        """Handle the unmount event for this tensorboard pack dialog."""
        self._pack_request_id += 1
        if self._pack_load_task is not None and not self._pack_load_task.done():
            self._pack_load_task.cancel()

    def on_model_selector_panel_model_selected(
        self,
        event: ModelSelectorPanel.ModelSelected,
    ) -> None:
        """Load packs for the intentional Inspect model selection."""

        manifest = event.model_manifest
        if manifest is None:
            self._selected_manifest = None
            self._selected_model_uuid = ""
            self._pack_entries = ()
            self._selected_value = None
            self._pack_request_id += 1
            self._show_tree_message("Select a model to load TensorBoard packs.")
            return
        configuration = manifest.get_configuration()
        model_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if model_uuid and model_uuid == self._selected_model_uuid:
            return
        self._selected_manifest = manifest
        self._selected_model_uuid = model_uuid
        self._selected_value = None
        self._start_pack_load(manifest)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle the tree node selected event for this tensorboard pack dialog."""
        if event.control.id != "tensorboard_pack_tree":
            return
        self._selected_value = event.node.data if isinstance(event.node.data, str) else None
        self.query_one("#tensorboard_pack_open", Button).disabled = self._selected_value is None
        self._update_details()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this tensorboard pack dialog."""
        if event.button.id == "tensorboard_pack_open":
            entry = self._get_selected_entry()
            if entry is None:
                self.dismiss(None)
                return
            self.dismiss(
                TensorboardPackDialogResult(
                    pack_type=entry.pack_type,
                    framework_uuid=entry.framework_uuid,
                    run_name=entry.run_name,
                    logdir=entry.path,
                    model_manifest=self._selected_manifest,
                )
            )
            return

        self.dismiss(None)
