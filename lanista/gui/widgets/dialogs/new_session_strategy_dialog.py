# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for new session strategy."""

from dataclasses import dataclass
from datetime import UTC, datetime

from lanista.application import LanistAIApplication
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.utils.text_cells import fit_text_cell
from lanista.gui.widgets.dialogs.button_templates import selector_button, selector_label
from lanista.gui.widgets.dialogs.checkpoint_configuration_dialog import (
    CheckpointConfigurationDialog,
    CheckpointConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_new_session_strategy_dialog_css_path
from lanista.reinforcengine.application_errors import ApplicationError
from lanista.reinforcengine.rl_model.sessions.repository_contracts import (
    SessionManifestSummary,
)
from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea, Tree
from textual.widgets._tree import TreeNode


@dataclass
class NewSessionStrategyDialogResult:
    """Report the result of new session strategy dialog."""

    strategy: str  # "fresh" | "fork" | "cancel"
    parent_session_uuid: str | None
    parent_checkpoint_name: str | None = None
    session_name: str = ""
    session_description: str = ""


class NewSessionStrategyDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[NewSessionStrategyDialogResult | None],
):
    """Present new session strategy in the Textual interface."""

    _HELP_TOPIC = "new_session"
    CSS_PATH = get_new_session_strategy_dialog_css_path()
    _APPLICATION = LanistAIApplication()

    _FRESH_SESSION_SELECTOR = "__fresh_session__"
    _DISPLAY_NAME_WIDTH = 33
    _UUID_WIDTH = 8
    _CREATED_WIDTH = 20
    _PARENT_WIDTH = 17

    BINDINGS = [
        Binding("f4,enter", "shortcut_confirm", "Confirm"),
        Binding("f7,n", "shortcut_cancel", "Close new session"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "shortcut_cancel", "Cancel", key_display="esc"),
    ]

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "new_session_parent_search": {
            "down": "new_session_parent_tree",
        },
        "new_session_parent_tree": {
            "left": "new_session_parent_search",
            "right": "new_session_source_checkpoint_button",
            "down": "new_session_source_checkpoint_button",
        },
        "new_session_source_checkpoint_button": {
            "up": "new_session_parent_tree",
            "left": "new_session_parent_tree",
            "right": "new_session_name_input",
            "down": "new_session_name_input",
        },
        "new_session_name_input": {
            "up": "new_session_source_checkpoint_button",
            "down": "new_session_description_input",
        },
        "new_session_description_input": {
            "up": "new_session_name_input",
            "down": "new_session_confirm",
        },
        "new_session_confirm": {
            "up": "new_session_description_input",
            "right": "new_session_cancel",
        },
        "new_session_cancel": {
            "up": "new_session_description_input",
            "left": "new_session_confirm",
        },
    }

    def __init__(
        self,
        current_session_uuid: str | None,
        current_session_display_name: str | None,
        framework_uuid: str | None = None,
    ):
        """Initialize the new session strategy dialog and its owned state."""
        super().__init__()
        self._current_session_uuid = current_session_uuid
        self._current_session_display_name = current_session_display_name
        self._framework_uuid = str(framework_uuid or "").strip().lower() or None
        services = self._APPLICATION.get_application_services()
        self._session_repository = services.sessions.repository
        self._session_selection = services.sessions.selection
        self._session_checkpoints = services.catalog.session_checkpoints
        self._session_uuids: list[str] = []
        self._session_summaries: dict[str, SessionManifestSummary] = {}
        self._checkpoint_options_cache: dict[str, list[tuple[str, str]]] = {}
        self._selected_parent_session_uuid: str | None = None
        self._selected_parent_checkpoint_name: str | None = None
        self._base_session_name = ""
        self._using_auto_session_name = True
        self._updating_session_name = False

    @classmethod
    def _build_parent_header_label(cls) -> Text:
        """Build the parent header label required by this new session strategy dialog."""
        header = Text()
        header.append(" ")
        header.append(fit_text_cell("Parent session", cls._DISPLAY_NAME_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("UUID", cls._UUID_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("Created", cls._CREATED_WIDTH), style="bold dim")
        header.append("  ")
        header.append(fit_text_cell("Parent", cls._PARENT_WIDTH), style="bold dim")
        return header

    def _build_fresh_label(self) -> Text:
        """Build the fresh label required by this new session strategy dialog."""
        label = Text()
        label.append(fit_text_cell("Start Fresh Session", self._DISPLAY_NAME_WIDTH), style="bold green")
        label.append("  ")
        label.append(fit_text_cell("-", self._UUID_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell("-", self._CREATED_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell("No parent", self._PARENT_WIDTH), style="dim")
        return label

    def _build_session_entry_label(self, session_uuid: str) -> Text:
        """Build the session entry label required by this new session strategy dialog."""
        summary = self._session_summaries.get(session_uuid)
        display_name = session_uuid or "-"
        identity_uuid = session_uuid
        created_at = "-"
        parent_uuid = ""
        if summary is not None:
            display_name = summary.display_name or summary.session_name or display_name
            identity_uuid = summary.identity_uuid or identity_uuid
            created_at = summary.created_at or created_at
            parent_uuid = summary.parent_session_uuid or ""
        parent_display = "-"
        if parent_uuid:
            parent_summary = self._session_summaries.get(parent_uuid)
            parent_display = (
                self._session_repository.get_session_reference_display_from_summary(
                    parent_uuid,
                    parent_summary,
                )
                if parent_summary is not None
                else parent_uuid
            )

        label = Text()
        label.append(fit_text_cell(display_name, self._DISPLAY_NAME_WIDTH), style="bold")
        label.append("  ")
        label.append(fit_text_cell(identity_uuid[:8] if identity_uuid else "-", self._UUID_WIDTH), style="dim")
        label.append("  ")
        label.append(fit_text_cell(str(created_at)[:19], self._CREATED_WIDTH), style="cyan")
        label.append("  ")
        label.append(fit_text_cell(parent_display, self._PARENT_WIDTH), style="italic cyan")
        return label

    def _session_search_text(self, session_uuid: str) -> str:
        """Return the session search text value owned by this new session strategy dialog."""
        summary = self._session_summaries.get(session_uuid)
        if summary is None:
            return session_uuid.lower()
        identity_uuid = summary.identity_uuid or session_uuid
        parts = [
            session_uuid,
            identity_uuid,
            self._session_repository.get_session_reference_display_from_summary(session_uuid, summary),
            summary.description,
            summary.parent_session_uuid or "",
            summary.created_at,
        ]
        return " ".join(str(part or "") for part in parts).lower()

    def _select_parent_session(self, session_uuid: str | None) -> None:
        """Select the parent session owned by this new session strategy dialog."""
        if session_uuid != self._selected_parent_session_uuid:
            self._selected_parent_checkpoint_name = None
        self._selected_parent_session_uuid = session_uuid
        self._refresh_source_checkpoint_field()
        self._sync_auto_session_name()
        self._refresh_confirm_state()

    def _select_parent_node_data(self, data: object) -> None:
        """Select the parent node data owned by this new session strategy dialog."""
        if data == self._FRESH_SESSION_SELECTOR:
            self._select_parent_session(None)
            return
        self._select_parent_session(data if isinstance(data, str) else None)

    def _sync_parent_selection_from_tree_cursor(self) -> None:
        """Synchronize the parent selection from tree cursor owned by this new session strategy
        dialog.
        """
        tree = self.query_one("#new_session_parent_tree", Tree)
        cursor_node = tree.cursor_node
        if cursor_node is not None:
            self._select_parent_node_data(cursor_node.data)

    def _rebuild_parent_tree(self, filter_text: str = "") -> None:
        """Rebuild the parent tree owned by this new session strategy dialog."""
        tree = self.query_one("#new_session_parent_tree", Tree)
        tree.clear()
        tree.root.expand()
        tree.root.allow_expand = False

        normalized_filter = str(filter_text or "").strip().lower()
        show_fresh = (
            not normalized_filter
            or normalized_filter in "start fresh session"
            or normalized_filter in "fresh"
        )

        selected_node: TreeNode | None = None
        if show_fresh:
            fresh_node = tree.root.add_leaf(self._build_fresh_label(), data=self._FRESH_SESSION_SELECTOR)
            fresh_node.allow_expand = False
            if self._selected_parent_session_uuid is None:
                selected_node = fresh_node

        visible_sessions = [
            session_uuid
            for session_uuid in self._session_uuids
            if not normalized_filter or normalized_filter in self._session_search_text(session_uuid)
        ]

        for session_uuid in visible_sessions:
            node = tree.root.add_leaf(self._build_session_entry_label(session_uuid), data=session_uuid)
            node.allow_expand = False
            if session_uuid == self._selected_parent_session_uuid:
                selected_node = node

        if selected_node is None and tree.root.children:
            selected_node = tree.root.children[0]

        if selected_node is not None:
            tree.select_node(selected_node)
            self._select_parent_node_data(selected_node.data)

    def _parent_checkpoint_options(self) -> list[tuple[str, str]]:
        """Return the parent checkpoint options value owned by this new session strategy dialog."""
        if self._selected_parent_session_uuid is None:
            return []
        cache_key = f"{self._framework_uuid or ''}:{self._selected_parent_session_uuid}"
        if cache_key not in self._checkpoint_options_cache:
            self._checkpoint_options_cache[cache_key] = self._session_selection.list_checkpoint_options_for_current_model(
                self._selected_parent_session_uuid,
                framework_uuid=self._framework_uuid,
            )
        return list(self._checkpoint_options_cache[cache_key])

    def _selected_source_checkpoint_label(self) -> str:
        """Return the selected source checkpoint label value owned by this new session strategy
        dialog.
        """
        if self._selected_parent_session_uuid is None:
            return "fresh training"
        if self._selected_parent_checkpoint_name:
            return self._selected_parent_checkpoint_name
        if self._parent_checkpoint_options():
            return "select checkpoint"
        return "no checkpoints"

    def _refresh_source_checkpoint_field(self) -> None:
        """Refresh the source checkpoint field owned by this new session
strategy dialog."""
        source_button = self.query_one("#new_session_source_checkpoint_button", Button)
        source_button.label = selector_label("Source Checkpoint", self._selected_source_checkpoint_label())
        source_button.disabled = self._selected_parent_session_uuid is None or not bool(self._parent_checkpoint_options())

    def _auto_session_name(self) -> str:
        """Return the auto session name value owned by this new session strategy dialog."""
        if self._selected_parent_session_uuid is None:
            return self._base_session_name
        return f"{self._base_session_name}_fork"

    def _sync_auto_session_name(self) -> None:
        """Synchronize the auto session name owned by this new session strategy
dialog."""
        if not self._using_auto_session_name:
            return
        name_input = self.query_one("#new_session_name_input", Input)
        self._updating_session_name = True
        try:
            name_input.value = self._auto_session_name()
        finally:
            self._updating_session_name = False

    def _open_source_checkpoint_dialog(self) -> None:
        """Open the source checkpoint dialog workflow owned by this new session strategy dialog."""
        if self._selected_parent_session_uuid is None:
            return
        checkpoint_options = self._parent_checkpoint_options()
        if not checkpoint_options:
            return

        def _handle_checkpoint_result(result: CheckpointConfigurationDialogResult | None) -> None:
            """Handle the checkpoint result workflow for this new session strategy dialog."""
            if result is not None and result.action == "select":
                self._selected_parent_checkpoint_name = result.checkpoint_name
                self._refresh_source_checkpoint_field()
                self._refresh_confirm_state()
                self._focus_if_navigable("new_session_source_checkpoint_button")

        self.app.push_screen(
            CheckpointConfigurationDialog(
                session_name=self._selected_parent_session_uuid,
                checkpoint_options=checkpoint_options,
                selected_checkpoint_name=self._selected_parent_checkpoint_name,
                allow_fresh_training=False,
                allow_delete=False,
                allow_rename=False,
                allow_edit=False,
            ),
            _handle_checkpoint_result,
        )

    def _validation_error(self) -> str | None:
        """Return the validation error value owned by this new session strategy dialog."""
        session_name = self.query_one("#new_session_name_input", Input).value.strip()
        if not session_name:
            return "Session name is required."
        if self._selected_parent_session_uuid and not self._parent_checkpoint_options():
            return "Selected parent session has no checkpoints."
        if self._selected_parent_session_uuid and not self._selected_parent_checkpoint_name:
            return "Select a checkpoint from the parent session."
        return None

    def _refresh_confirm_state(self) -> None:
        """Refresh the confirm state owned by this new session strategy
dialog."""
        confirm_button = self.query_one("#new_session_confirm", Button)
        error_label = self.query_one("#new_session_error", Static)

        validation_error = self._validation_error()
        confirm_button.disabled = validation_error is not None
        error_label.update(validation_error or "")

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this new session strategy dialog."""
        with Vertical(id="new_session_strategy_dialog_container") as container:
            container.border_title = "New Session"

            # The outer title must be assigned before entering the nested content.
            with ScrollableContainer(id="new_session_dialog_scroll"):  # noqa: SIM117
                with Vertical(id="new_session_dialog_content"):
                    yield Input(placeholder="Search parent sessions...", id="new_session_parent_search")
                    yield Static(self._build_parent_header_label(), id="new_session_parent_header")

                    parent_tree = Tree("Parent session selector", id="new_session_parent_tree")
                    parent_tree.show_root = False
                    parent_tree.show_guides = False
                    parent_tree.border_title = "Parent Session Selector"
                    yield parent_tree

                    yield selector_button(
                        selector_label("Source Checkpoint", "fresh training"),
                        id="new_session_source_checkpoint_button",
                        disabled=True,
                    )

                    name_input = Input(id="new_session_name_input", placeholder="<session name>")
                    name_input.border_title = "Session Name"
                    yield name_input

                    description_input = TextArea(id="new_session_description_input", text="")
                    description_input.border_title = "Session Description"
                    yield description_input

                    yield Static("", id="new_session_error")
                    with Horizontal(id="new_session_actions"):
                        yield Button("Confirm", id="new_session_confirm", variant="success")
                        yield Button("Cancel", id="new_session_cancel", variant="default")

            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["shortcut_confirm", "shortcut_cancel"], ["open_help_dialog"]],
                id="new_session_modal_footer",
                label_overrides={
                    "shortcut_confirm": "Confirm",
                    "shortcut_cancel": "Close",
                },
            )

    def _load_session_summaries(self) -> None:
        """Load the session summaries owned by this new session strategy dialog."""
        try:
            session_uuids = self._session_checkpoints.list_sessions_for_current_model(
                framework_uuid=self._framework_uuid,
            )
        except ApplicationError as error:
            self._session_uuids = []
            self._session_summaries = {}
            self.notify(
                str(error),
                title="Session metadata unavailable",
                severity="error",
                timeout=10,
            )
            return

        summaries: dict[str, SessionManifestSummary] = {}
        failures = []
        for session_uuid in session_uuids:
            result = self._session_repository.load_session_manifest_summary(session_uuid)
            if result.failure is not None:
                failures.append(result.failure)
            elif result.value is not None:
                summaries[session_uuid] = result.value

        self._session_summaries = summaries
        self._session_uuids = [
            session_uuid for session_uuid in session_uuids if session_uuid in summaries
        ]
        if failures:
            self.notify(
                (
                    f"{len(failures)} session manifest(s) could not be read. "
                    f"{failures[0].message}"
                ),
                title="Session metadata unavailable",
                severity="error",
                timeout=10,
            )

    def on_mount(self) -> None:
        """Handle the mount event for this new session strategy dialog."""
        self._load_session_summaries()
        if self._current_session_uuid in self._session_uuids:
            self._selected_parent_session_uuid = self._current_session_uuid

        self._base_session_name = datetime.now(UTC).strftime("%Y-%m-%d_%H-%M-%S")
        self.query_one("#new_session_name_input", Input).value = self._auto_session_name()

        self._rebuild_parent_tree()
        self._refresh_source_checkpoint_field()
        self._refresh_confirm_state()
        self.query_one("#new_session_parent_search", Input).focus()

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle the input changed event for this new session strategy dialog."""
        if event.input.id == "new_session_name_input":
            if not self._updating_session_name:
                self._using_auto_session_name = event.value.strip() in {
                    self._base_session_name,
                    f"{self._base_session_name}_fork",
                }
            self._refresh_confirm_state()
            return

        if event.input.id == "new_session_parent_search":
            self._rebuild_parent_tree(event.value)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
        """Handle the tree node selected event for this new session strategy dialog."""
        tree_widget = getattr(event, "control", None) or getattr(event.node, "tree", None)
        if tree_widget is None or tree_widget.id != "new_session_parent_tree":
            return
        self._select_parent_node_data(event.node.data)

    def on_tree_node_highlighted(self, event: Tree.NodeHighlighted) -> None:
        """Handle the tree node highlighted event for this new session strategy dialog."""
        tree_widget = getattr(event, "control", None) or getattr(event.node, "tree", None)
        if tree_widget is None or tree_widget.id != "new_session_parent_tree":
            return
        self._select_parent_node_data(event.node.data)

    def on_key(self, event: Key) -> None:
        """Handle the key event for this new session strategy dialog."""
        if event.key not in {"up", "left", "down", "right"}:
            return

        focused = self.focused
        focused_id = focused.id if focused is not None else None

        if focused_id == "new_session_description_input":
            return

        if focused_id == "new_session_parent_search" and event.key != "down":
            return

        if focused_id == "new_session_name_input" and event.key in {"left", "right"}:
            return

        if focused_id == "new_session_parent_tree" and event.key in {"up", "down"}:
            return

        if self._move_focus_in_direction(event.key):
            event.stop()
            event.prevent_default()

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this new session strategy dialog."""
        button = self.query_one("#new_session_confirm", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_cancel(self) -> None:
        """Handle the shortcut cancel user action for this new session strategy dialog."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this new session strategy dialog."""
        if event.button.id == "new_session_cancel":
            self.dismiss(None)
            return

        if event.button.id != "new_session_confirm":
            if event.button.id == "new_session_source_checkpoint_button":
                self._open_source_checkpoint_dialog()
            return

        validation_error = self._validation_error()
        if validation_error:
            self.query_one("#new_session_error", Static).update(validation_error)
            return

        self._sync_parent_selection_from_tree_cursor()
        parent_session_uuid = self._selected_parent_session_uuid
        parent_checkpoint_name = self._selected_parent_checkpoint_name if parent_session_uuid else None
        strategy = "fork" if parent_session_uuid else "fresh"

        self.dismiss(
            NewSessionStrategyDialogResult(
                strategy=strategy,
                parent_session_uuid=parent_session_uuid,
                parent_checkpoint_name=parent_checkpoint_name,
                session_name=self.query_one("#new_session_name_input", Input).value.strip(),
                session_description=self.query_one("#new_session_description_input", TextArea).text.strip(),
            )
        )
