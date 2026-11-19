# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the screen for model creation."""

from dataclasses import dataclass
from typing import Any

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.text_cells import fit_text_cell
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.model_clone_selection_dialog import (
    ModelCloneSelectionDialog,
    ModelCloneSelectionResult,
)
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_model_creation_screen_css_path
from lanista.reinforcengine.application_errors import OperationFailure
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.model_creation_contracts import (
    FreshModelCreationRequest,
    ModelCloneRequest,
)
from lanista.reinforcengine.rl_model.model_discovery_service import (
    DiscoveredSessionMetadataStatus,
    DiscoveredValidModelEntry,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea, Tree
from textual.widgets._tree import TreeNode


@dataclass
class ModelCreationResult:
    """Report the result of model creation."""

    target_model_folder_path: str
    target_model_name: str
    creation_strategy: str = "fresh"
    auto_load: bool = False


class ModelCreationScreen(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[ModelCreationResult | None],
):
    """Present model creation in the Textual interface."""

    _HELP_TOPIC = "model_creation"
    CSS_PATH = get_model_creation_screen_css_path()  # type: ignore[no-untyped-call]
    _APPLICATION = LanistAIApplication()

    _DISPLAY_NAME_WIDTH = 28
    _VERSION_WIDTH = 12
    _UUID_WIDTH = 8
    _CREATED_WIDTH = 19

    _START_EMPTY_SELECTOR = "__start_empty_model__"

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "model_creation_parent_search": {
            "down": "model_creation_parent_tree",
        },
        "model_creation_parent_tree": {
            "left": "model_creation_parent_search",
            "right": "model_creation_name_input",
        },
        "model_creation_name_input": {
            "up": "model_creation_parent_tree",
            "down": "model_creation_version_input",
        },
        "model_creation_version_input": {
            "up": "model_creation_name_input",
            "down": "model_creation_description_input",
        },
        "model_creation_description_input": {
            "up": "model_creation_version_input",
            "down": "model_creation_confirm",
        },
        "model_creation_confirm": {
            "up": "model_creation_description_input",
            "right": "model_creation_cancel",
        },
        "model_creation_cancel": {
            "up": "model_creation_description_input",
            "left": "model_creation_confirm",
        },
    }

    _FOOTER_BINDINGS: tuple[Binding, ...] = (
        Binding("f4,enter", "shortcut_confirm", "Confirm"),
        Binding("f6,m", "shortcut_cancel", "Dismiss creation"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "shortcut_cancel", "Dismiss creation", key_display="esc"),
    )
    BINDINGS = list(_FOOTER_BINDINGS)

    def __init__(
        self,
        framework_uuid: str = "ray",
        application_services: ApplicationServices | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the model creation screen and its owned state."""
        super().__init__(**kwargs)
        self._framework_uuid = str(framework_uuid or "ray").strip().lower()
        self._services = application_services or self._APPLICATION.get_application_services()

        self._parent_entries: list[DiscoveredValidModelEntry] = []
        self._selected_parent_manifest: ModelManifest | None = None

    @classmethod
    def _build_parent_header_label(cls) -> Text:
        """Build the parent header label required by this model creation screen."""
        header = Text()
        header.append(" ")
        header.append(fit_text_cell("Parent model", cls._DISPLAY_NAME_WIDTH), style="bold dim")
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
    def _build_parent_entry_label(cls, entry: DiscoveredValidModelEntry) -> Text:
        """Build the parent entry label required by this model creation screen."""
        label = Text()
        label.append(
            fit_text_cell(entry.display_name or "Unnamed model", cls._DISPLAY_NAME_WIDTH),
            style="bold",
        )
        label.append("  ")
        label.append(
            fit_text_cell(entry.effective_version or "-", cls._VERSION_WIDTH), style="green"
        )
        label.append("  ")
        label.append(fit_text_cell(entry.short_uuid or "-", cls._UUID_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell(entry.created_at_display, cls._CREATED_WIDTH), style="cyan")
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

    def _build_start_empty_label(self) -> Text:
        """Build the start empty label required by this model creation screen."""
        label = Text()
        label.append(
            fit_text_cell("Start Empty Model", self._DISPLAY_NAME_WIDTH), style="bold green"
        )
        label.append("  ")
        label.append(fit_text_cell("-", self._VERSION_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell("-", self._UUID_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell("-", self._CREATED_WIDTH), style="dim")
        label.append("  ")
        label.append("-", style="dim")
        return label

    @staticmethod
    def _ensure_project_suffix(model_name: str) -> str:
        """Ensure the project suffix owned by this model creation screen."""
        name = str(model_name or "").strip()
        if name and not name.endswith("_project"):
            return f"{name}_project"
        return name

    def _validate(self) -> str | None:
        """Return the validate value owned by this model creation screen."""
        normalized_name = self._ensure_project_suffix(
            self.query_one("#model_creation_name_input", Input).value
        )
        draft = self._services.catalog.creation.resolve_draft(
            normalized_name,
            self.query_one("#model_creation_version_input", Input).value.strip(),
        )
        return draft.error_message

    def _refresh_create_button_state(self) -> None:
        """Refresh the create button state owned by this model creation
screen."""
        confirm_button = self.query_one("#model_creation_confirm", Button)
        error_label = self.query_one("#model_creation_error", Static)

        validation_error = self._validate()
        confirm_button.disabled = validation_error is not None
        if validation_error:
            error_label.update(validation_error)
        else:
            error_label.update("")

    def _select_parent_manifest(self, manifest: ModelManifest | None) -> None:
        """Select the parent manifest owned by this model creation screen."""
        self._selected_parent_manifest = manifest
        self._refresh_create_button_state()

    def _select_parent_node_data(self, data: object) -> None:
        """Select the parent node data owned by this model creation screen."""
        if data == self._START_EMPTY_SELECTOR:
            self._select_parent_manifest(None)
            return

        manifest = data if isinstance(data, ModelManifest) else None
        self._select_parent_manifest(manifest)

    def _sync_parent_selection_from_tree_cursor(self) -> None:
        """Synchronize the parent selection from tree cursor owned by this model
creation screen."""
        try:
            tree = self.query_one("#model_creation_parent_tree", Tree)
        except NoMatches:
            return

        cursor_node = tree.cursor_node
        if cursor_node is not None:
            self._select_parent_node_data(cursor_node.data)

    def _rebuild_parent_tree(self, filter_text: str = "") -> None:
        """Rebuild the parent tree owned by this model creation screen."""
        tree = self.query_one("#model_creation_parent_tree", Tree)
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False

        normalized_filter = str(filter_text or "").strip().lower()

        show_start_empty = (
            not normalized_filter
            or normalized_filter in "start empty model"
            or normalized_filter in "empty"
        )
        if show_start_empty:
            start_node = tree.root.add_leaf(
                self._build_start_empty_label(), data=self._START_EMPTY_SELECTOR
            )
            start_node.allow_expand = False

        visible_entries = [
            entry
            for entry in self._parent_entries
            if not normalized_filter or normalized_filter in entry.search_text
        ]

        selected_node: TreeNode[object] | None = None
        for entry in visible_entries:
            node = tree.root.add_leaf(self._build_parent_entry_label(entry), data=entry.manifest)
            node.allow_expand = False
            if self._selected_parent_manifest is not None and (
                getattr(self._selected_parent_manifest.get_configuration(), "model_uuid", None)
                == getattr(entry.manifest.get_configuration(), "model_uuid", None)
            ):
                selected_node = node

        if (
            selected_node is None
            and self._selected_parent_manifest is None
            and show_start_empty
            and tree.root.children
        ):
            selected_node = tree.root.children[0]

        if selected_node is None and tree.root.children:
            selected_node = tree.root.children[0]

        if selected_node is not None:
            tree.select_node(selected_node)
            self._select_parent_node_data(selected_node.data)

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this model creation screen."""
        with Vertical(id="model_creation_dialog_container") as container:
            container.border_title = "New Model"

            with (
                ScrollableContainer(id="model_creation_dialog_scroll"),
                Vertical(id="model_creation_dialog_content"),
            ):
                yield Input(
                    placeholder="Search parent models...",
                    id="model_creation_parent_search",
                )
                yield Static(
                    self._build_parent_header_label(),
                    id="model_creation_parent_header",
                )

                parent_tree: Tree[object] = Tree(
                    "Parent model selector",
                    id="model_creation_parent_tree",
                )
                parent_tree.show_root = False
                parent_tree.show_guides = False
                parent_tree.border_title = "Parent Model Selector"
                yield parent_tree

                name_input = Input(
                    id="model_creation_name_input",
                    placeholder="<name>",
                )
                name_input.border_title = "Model Name"
                yield name_input

                version_input = Input(
                    id="model_creation_version_input",
                    value="",
                )
                version_input.border_title = "Model Version"
                yield version_input

                description_input = TextArea(
                    id="model_creation_description_input",
                    text="",
                )
                description_input.border_title = "Model Description"
                yield description_input

                yield Static("", id="model_creation_error")
                with Horizontal(id="model_creation_actions"):
                    yield Button(
                        "Confirm",
                        id="model_creation_confirm",
                        variant="success",
                    )
                    yield Button(
                        "Cancel",
                        id="model_creation_cancel",
                        variant="default",
                    )

            yield ModalFooter.from_binding_rows(
                self._FOOTER_BINDINGS,
                [["shortcut_confirm", "shortcut_cancel"], ["open_help_dialog"]],
                id="model_creation_modal_footer",
                label_overrides={
                    "shortcut_confirm": "Confirm",
                    "shortcut_cancel": "Close",
                },
            )

    def on_mount(self) -> None:
        """Handle the mount event for this model creation screen."""
        self._parent_entries, _ = self._services.catalog.models.discover_model_candidates()

        self._rebuild_parent_tree()

        version_input = self.query_one("#model_creation_version_input", Input)
        if not version_input.value.strip():
            version_input.value = self._services.catalog.creation.suggest_default_version()

        self._refresh_create_button_state()
        self.query_one("#model_creation_parent_search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this model creation screen."""
        if event.input.id in {"model_creation_name_input", "model_creation_version_input"}:
            self._refresh_create_button_state()
            return

        if event.input.id == "model_creation_parent_search":
            self._rebuild_parent_tree(event.value)

    def on_tree_node_selected(self, event: Tree.NodeSelected[object]) -> None:
        """Handle the tree node selected event for this model creation screen."""
        tree_widget = getattr(event, "control", None) or getattr(event.node, "tree", None)
        if tree_widget is None or tree_widget.id != "model_creation_parent_tree":
            return

        self._select_parent_node_data(event.node.data)

    def on_tree_node_highlighted(
        self,
        event: Tree.NodeHighlighted[object],
    ) -> None:
        """Handle the tree node highlighted event for this model creation screen."""
        tree_widget = getattr(event, "control", None) or getattr(event.node, "tree", None)
        if tree_widget is None or tree_widget.id != "model_creation_parent_tree":
            return

        self._select_parent_node_data(event.node.data)

    def on_key(self, event: Key) -> None:
        """Handle the key event for this model creation screen."""
        if event.key not in {"up", "left", "down", "right"}:
            return

        focused = self.focused
        focused_id = focused.id if focused is not None else None

        if focused_id == "model_creation_description_input":
            return

        if focused_id == "model_creation_parent_search" and event.key != "down":
            return

        if focused_id in {
            "model_creation_name_input",
            "model_creation_version_input",
        } and event.key in {"left", "right"}:
            return

        if focused_id == "model_creation_parent_tree" and event.key in {"up", "down"}:
            return

        if self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this model creation screen."""
        try:
            button = self.query_one("#model_creation_confirm", Button)
        except NoMatches:
            return
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this model creation screen."""
        self.dismiss(None)

    def _build_creation_draft(self) -> tuple[str, str, str]:
        """Build the creation draft required by this model creation screen."""
        raw_name = self.query_one("#model_creation_name_input", Input).value
        normalized_name = self._ensure_project_suffix(raw_name)
        draft = self._services.catalog.creation.resolve_draft(
            normalized_name,
            self.query_one("#model_creation_version_input", Input).value.strip(),
        )
        model_version = draft.resolved_model_version
        model_name = draft.resolved_model_name
        description = self.query_one("#model_creation_description_input", TextArea).text.strip()

        self.query_one("#model_creation_name_input", Input).value = model_name
        self.query_one("#model_creation_version_input", Input).value = model_version
        return model_name, model_version, description

    @staticmethod
    def _format_creation_failure(
        failure: OperationFailure | None,
        *,
        operation_id: str,
    ) -> str:
        """Format the creation failure for presentation by this model creation screen."""
        message = (
            failure.message
            if failure is not None
            else "The model creation operation returned no result."
        )
        return f"{message} [operation_id={operation_id}]"

    def _open_clone_selection_dialog(
        self, model_name: str, model_version: str, description: str
    ) -> None:
        """Open the clone selection dialog workflow owned by this model creation screen."""
        if self._selected_parent_manifest is None:
            self.query_one("#model_creation_error", Static).update(
                "Select a valid parent model for clone."
            )
            return

        cfg = self._selected_parent_manifest.get_configuration()
        source_path = str(getattr(cfg, "model_folder_path", "") or "").strip()
        source_display = str(getattr(cfg, "model_name", "") or "-")

        if not source_path:
            self.query_one("#model_creation_error", Static).update(
                "Selected parent model path is missing."
            )
            return

        def _on_clone_selection(result: ModelCloneSelectionResult | None) -> None:
            """Handle the clone selection event for this model creation screen."""
            if result is None:
                return
            selected_paths = [
                str(path).strip() for path in result.selected_relative_paths if str(path).strip()
            ]
            if not selected_paths:
                self.query_one("#model_creation_error", Static).update("No clone entries approved.")
                return

            duplication_result = self._services.catalog.creation.clone_model(
                ModelCloneRequest(
                    source_model_folder_path=source_path,
                    target_model_name=model_name,
                    target_model_version=model_version,
                    initial_session_description=description,
                    selected_relative_paths=tuple(selected_paths),
                )
            )
            outcome = duplication_result.value
            if not duplication_result.succeeded or outcome is None:
                self.query_one("#model_creation_error", Static).update(
                    self._format_creation_failure(
                        duplication_result.failure,
                        operation_id=duplication_result.operation_id,
                    )
                )
                return

            self.dismiss(
                ModelCreationResult(
                    target_model_folder_path=outcome.target_model_folder_path,
                    target_model_name=outcome.target_model_name,
                    creation_strategy="clone",
                    auto_load=outcome.auto_load,
                )
            )

        self.app.push_screen(
            ModelCloneSelectionDialog(
                source_model_folder_path=source_path, source_display_name=source_display
            ),
            _on_clone_selection,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this model creation screen."""
        if event.button.id == "model_creation_cancel":
            self.dismiss(None)
            return

        if event.button.id != "model_creation_confirm":
            return

        validation_error = self._validate()
        if validation_error:
            self.query_one("#model_creation_error", Static).update(validation_error)
            return

        model_name, model_version, description = self._build_creation_draft()
        self._sync_parent_selection_from_tree_cursor()

        if self._selected_parent_manifest is not None:
            self._open_clone_selection_dialog(model_name, model_version, description)
            return

        generation_result = self._services.catalog.creation.create_empty_model(
            FreshModelCreationRequest(
                model_name=model_name,
                model_version=model_version,
                model_description=description,
                framework_uuid=self._framework_uuid,
            )
        )
        outcome = generation_result.value
        if not generation_result.succeeded or outcome is None:
            self.query_one("#model_creation_error", Static).update(
                self._format_creation_failure(
                    generation_result.failure,
                    operation_id=generation_result.operation_id,
                )
            )
            return

        self.dismiss(
            ModelCreationResult(
                target_model_folder_path=outcome.target_model_folder_path,
                target_model_name=outcome.target_model_name,
                creation_strategy="fresh",
                auto_load=outcome.auto_load,
            )
        )
