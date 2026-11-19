# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for configuration."""

from collections.abc import Callable, Sequence
from contextlib import suppress
from functools import partial
from time import monotonic
from typing import cast

from rich.text import Text
from textual.app import ComposeResult, ScreenStackError
from textual.binding import Binding
from textual.containers import Grid, ScrollableContainer, Vertical
from textual.css.query import NoMatches
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Select, Static, Tree

from lanista.application import LanistAIApplication
from lanista.gui.adapters.control_panel.evaluation_libraries_dialog_adapter import (
    EvaluationLibrariesDialogAdapter,
)
from lanista.gui.predefs.secure_delete_dialog_definitions import (
    secure_delete_checkpoint_dialog,
    secure_delete_session_dialog,
)
from lanista.gui.services.configuration.configuration_administration_contracts import (
    CheckpointAdministrationEvent,
    CheckpointDeletionRequest,
    CheckpointSelectorIdentity,
    ConfigurationAdministrationBoundary,
    ConfigurationAdministrationEffect,
    ConfigurationAdministrationTransition,
    ConfirmCheckpointDeletion,
    ConfirmSessionDeletion,
    DismissConfiguration,
    OpenCheckpointSelector,
    OpenSessionSelector,
    PresentAdministrationNotification,
    ReopenCheckpointSelector,
    ReopenSessionSelector,
    SessionAdministrationEvent,
    SessionDeletionRequest,
    SessionSelectorIdentity,
)
from lanista.gui.services.configuration.configuration_administration_coordinator import (
    ConfigurationAdministrationCoordinator,
)
from lanista.gui.services.configuration.configuration_view_state import ConfigurationViewState
from lanista.gui.services.configuration.control_panel_configuration_contracts import (
    ConfigurationDialogAction,
    ConfigurationDialogResult,
    EvaluationTarget,
)
from lanista.gui.utils.directional_focus import DirectionalFocusMixin
from lanista.gui.widgets.dialogs.button_templates import selector_button, selector_label
from lanista.gui.widgets.dialogs.checkpoint_configuration_dialog import (
    CheckpointConfigurationDialog,
    CheckpointConfigurationDialogResult,
)
from lanista.gui.widgets.dialogs.evaluation_libraries_dialog import EvaluationLibrariesDialog
from lanista.gui.widgets.dialogs.help_enabled_modal import HelpEnabledModalMixin
from lanista.gui.widgets.dialogs.session_configuration_dialog import (
    SessionConfigurationDialog,
    SessionConfigurationDialogResult,
)
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import get_configuration_dialog_css_path
from lanista.reinforcengine.rl_model.administration_contracts import ModelAdministrationOwner
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationPacing,
    EvaluationVisualization,
)
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.checklist_service import (
    ChecklistItem,
    RuntimeChecklistService,
)
from lanista.reinforcengine.rl_model.runtime.configuration_state_service import (
    ConfigurationFrameworkStatus,
    ConfigurationStatusItem,
)


def _selected_evaluation_scenario_index(
    options: Sequence[tuple[str, int]],
    selected_index: int | None,
) -> int | None:
    """Keep a selected scenario only while it belongs to the current inventory."""

    available_indices = {index for _, index in options}
    if selected_index in available_indices:
        return selected_index
    return options[0][1] if options else None


class RunConfigurationDialog(
    DirectionalFocusMixin,
    HelpEnabledModalMixin,
    ModalScreen[ConfigurationDialogResult | None],
):
    """Present run configuration in the Textual interface."""

    _HELP_TOPIC = "configuration"
    CSS_PATH = get_configuration_dialog_css_path()
    _APPLICATION = LanistAIApplication()
    _STATUS_LABEL_WIDTH = 28
    _FRESH_TRAINING_VALUE = "__fresh_training__"
    _SHORTCUT_GUARD_SECONDS = 0.2
    _HELP_SHORTCUT_GUARD_SECONDS = 0.5
    _ACTION_GATE_LABELS = {
        "Create model available",
        "New session available",
        "Confirm available",
        "Training available",
        "Apply available",
        "Inference unavailable",
        "Run start precheck",
    }

    @staticmethod
    def _display_checklist_path(path: str) -> str:
        """Return the display checklist path value owned by this run configuration dialog."""
        return RuntimeChecklistService.shorten_checklist_path(path)

    BINDINGS = [
        Binding("f1", "dismiss", "Close configure"),
        Binding("n", "shortcut_new_model", "New Model"),
        Binding("s", "shortcut_new_session", "New Session"),
        Binding("i", "shortcut_inspect", "Inspect"),
        Binding("c", "shortcut_compare", "Compare Pack"),
        Binding("enter", "shortcut_confirm", "Confirm"),
        Binding("f9,h,?", "open_help_dialog", "Help", key_display="f9"),
        Binding("escape", "dismiss", "Cancel", key_display="esc"),
    ]

    def __init__(
        self,
        *,
        has_model: bool,
        loaded_model_name: str | None,
        expected_model_uuid: str | None = None,
        can_evaluate: bool,
        can_new_session: bool,
        can_confirm: bool,
        can_start_run: bool,
        configuration_locked: bool = False,
        selected_mode: RLModelRunningModes,
        framework_options: list[tuple[str, str]],
        selected_framework_uuid: str,
        session_options: list[tuple[str, str]],
        selected_session_name: str | None,
        checkpoint_options: list[tuple[str, str]],
        selected_checkpoint_name: str | None,
        selected_evaluation_visualization: EvaluationVisualization = "gui",
        selected_evaluation_pacing: EvaluationPacing = "maximum-throughput",
        evaluation_scenario_options: list[tuple[str, int]] | None = None,
        selected_evaluation_scenario_index: int | None = None,
        selected_evaluation_library_scope: LibraryEvaluationScope | None = None,
        selected_evaluation_target: EvaluationTarget = "single",
        inspect_is_running: bool,
        configuration_status_items: list[ConfigurationStatusItem],
        framework_statuses: list[ConfigurationFrameworkStatus],
        structure_check_items: list[ChecklistItem],
        runtime_check_items: list[ChecklistItem],
        application_services: ApplicationServices | None = None,
        auto_open_session_selector: bool = False,
        auto_open_checkpoint_selector: bool = False,
    ):
        """Initialize the run configuration dialog and its owned state."""
        self._has_model = has_model
        self._loaded_model_name = loaded_model_name
        normalized_model_uuid = str(expected_model_uuid or "").strip()
        if has_model and not normalized_model_uuid:
            raise ValueError(
                "A model UUID is required when configuration administration is enabled."
            )
        self._expected_model_uuid = normalized_model_uuid or None
        administration_owner = (
            ModelAdministrationOwner(normalized_model_uuid) if normalized_model_uuid else None
        )
        self._can_evaluate = can_evaluate
        self._can_new_session = can_new_session
        self._can_confirm = can_confirm
        self._can_start_run = can_start_run
        self._configuration_locked = configuration_locked
        self._selected_mode = selected_mode
        self._framework_options = framework_options
        self._selected_framework_uuid = selected_framework_uuid
        self._initialize_evaluation_selection(
            selected_evaluation_visualization,
            selected_evaluation_pacing,
            evaluation_scenario_options,
            selected_evaluation_scenario_index,
            selected_evaluation_library_scope,
            selected_evaluation_target,
        )
        self._session_options = session_options
        self._checkpoint_options = checkpoint_options
        self._inspect_is_running = inspect_is_running
        self._configuration_status_items = configuration_status_items
        self._framework_statuses = framework_statuses
        self._structure_check_items = structure_check_items
        self._runtime_check_items = runtime_check_items
        self._application_services = application_services
        self._administration_coordinator = ConfigurationAdministrationCoordinator(
            owner=administration_owner,
            session_name=selected_session_name,
            checkpoint_name=selected_checkpoint_name,
            boundary_provider=self._administration_boundary,
        )
        self._auto_open_session_selector = auto_open_session_selector
        self._auto_open_checkpoint_selector = auto_open_checkpoint_selector
        self._ready = False
        self._mounted_at = 0.0

        super().__init__()

    def _initialize_evaluation_selection(
        self,
        visualization: EvaluationVisualization,
        pacing: EvaluationPacing,
        scenario_options: list[tuple[str, int]] | None,
        scenario_index: int | None,
        library_scope: LibraryEvaluationScope | None,
        target: EvaluationTarget,
    ) -> None:
        self._selected_evaluation_visualization = visualization
        self._selected_evaluation_pacing = pacing
        self._evaluation_scenario_options = list(scenario_options or [])
        self._last_single_evaluation_scenario_index = _selected_evaluation_scenario_index(
            self._evaluation_scenario_options, scenario_index
        )
        self._evaluation_target: EvaluationTarget = (
            "library" if target == "library" or library_scope is not None else "single"
        )
        self._selected_evaluation_scenario_index = self._last_single_evaluation_scenario_index
        self._selected_evaluation_library_scope = library_scope
        if self._evaluation_target == "library":
            self._selected_evaluation_scenario_index = None
            self._selected_evaluation_visualization = "headless"
            self._selected_evaluation_pacing = "maximum-throughput"

    def _services(self) -> ApplicationServices:
        """Resolve the typed application boundary used by dialog effects."""

        if self._application_services is None:
            self._application_services = self._APPLICATION.get_application_services()
        return self._application_services

    def _administration_boundary(self) -> ConfigurationAdministrationBoundary:
        """Return the administration boundary value owned by this run configuration dialog."""
        services = self._services()
        return ConfigurationAdministrationBoundary(
            models=services.administration.models,
            session_commands=services.administration.sessions,
            checkpoint_commands=services.administration.checkpoints,
            session_references=services.sessions.metadata,
        )

    @property
    def _selected_session_name(self) -> str | None:
        """Return the selected session name exposed by this run configuration dialog."""
        return self._administration_coordinator.state.session_name

    @_selected_session_name.setter
    def _selected_session_name(self, session_name: str | None) -> None:
        """Handle the selected session name operation owned by this run configuration dialog."""
        self._administration_coordinator.project_session_name(session_name)

    @property
    def _selected_checkpoint_name(self) -> str | None:
        """Return the selected checkpoint name exposed by this run configuration dialog."""
        return self._administration_coordinator.state.checkpoint_name

    @_selected_checkpoint_name.setter
    def _selected_checkpoint_name(self, checkpoint_name: str | None) -> None:
        """Handle the selected checkpoint name operation owned by this run configuration
dialog."""
        self._administration_coordinator.project_checkpoint_name(checkpoint_name)

    def _invalidate_selector_callbacks(self) -> None:
        """Invalidate the selector callbacks owned by this run configuration dialog."""
        self._administration_coordinator.invalidate()

    def _confirm_label(self) -> str:
        """Confirm the label owned by this run configuration dialog."""
        return self._view_state().confirm_label

    def _view_state(self) -> ConfigurationViewState:
        """Build the immutable policy input from the widget's mutable controls."""
        return ConfigurationViewState(
            has_model=self._has_model,
            loaded_model_name=self._loaded_model_name,
            can_new_session=self._can_new_session,
            can_confirm=self._can_confirm,
            can_start_run=self._can_start_run,
            mode=self._selected_mode,
            framework_uuid=self._selected_framework_uuid,
            session_name=self._selected_session_name,
            checkpoint_name=self._selected_checkpoint_name,
            configuration_locked=self._configuration_locked,
            session_values=frozenset(value for _, value in self._session_options),
            checkpoint_values=frozenset(value for _, value in self._checkpoint_options),
            session_labels=tuple((value, display) for display, value in self._session_options),
            checkpoint_labels=tuple(
                (value, display) for display, value in self._checkpoint_options
            ),
        )

    def _session_display_map(self) -> dict[str, str]:
        """Return the session display map value owned by this run configuration dialog."""
        return {value: display for display, value in self._session_options}

    def _checkpoint_display_map(self) -> dict[str, str]:
        """Checkpoint the display map owned by this run configuration dialog."""
        return {value: display for display, value in self._checkpoint_options}

    def _model_button_label(self) -> str:
        """Return the model button label value owned by this run configuration dialog."""
        return selector_label("Model", self._view_state().model_selection_text, leading_space=False)

    def _session_button_label(self) -> str:
        """Return the session button label value owned by this run configuration dialog."""
        return selector_label(
            "Session", self._view_state().session_selection_text, leading_space=False
        )

    def _checkpoint_button_label(self) -> str:
        """Checkpoint the button label owned by this run configuration dialog."""
        return selector_label(
            "Checkpoint", self._view_state().checkpoint_selection_text, leading_space=False
        )

    def _session_picker_enabled(self) -> bool:
        """Return whether session picker enabled applies to this run configuration dialog."""
        return self._view_state().session_picker_enabled

    def _checkpoint_picker_enabled(self) -> bool:
        """Return whether checkpoint picker enabled applies to this run configuration dialog."""
        return self._view_state().checkpoint_picker_enabled

    def _has_valid_checkpoint_selection(self) -> bool:
        """Return whether valid checkpoint selection applies to this run configuration dialog."""
        return self._view_state().has_valid_checkpoint_selection

    def _confirm_enabled(self) -> bool:
        """Return whether confirm enabled applies to this run configuration dialog."""
        if (
            self._selected_mode is RLModelRunningModes.EVALUATION
            and self._evaluation_target == "library"
        ):
            return self._has_model and not self._configuration_locked
        return self._view_state().confirm_enabled

    def _new_session_enabled(self) -> bool:
        """Return whether new session enabled applies to this run configuration dialog."""
        return self._view_state().new_session_enabled

    def _training_actions_enabled(self) -> bool:
        """Return whether training actions enabled applies to this run configuration dialog."""
        return self._view_state().training_actions_enabled

    def _inspect_enabled(self) -> bool:
        """Return whether inspect enabled applies to this run configuration dialog."""
        return self._view_state().inspect_enabled

    def _compare_enabled(self) -> bool:
        """Return whether compare enabled applies to this run configuration dialog."""
        return self._view_state().compare_enabled

    def _format_tree_label(
        self, label: str, ok: bool, detail: str = "", *, selected: bool = False
    ) -> Text:
        """Format the tree label for presentation by this run configuration dialog."""
        text = Text()
        text.append("✓ " if ok else "✗ ", style="green" if ok else "red")
        padded_label = f"{str(label):<{self._STATUS_LABEL_WIDTH}}"
        text.append(padded_label, style="dim")
        if detail:
            text.append("  ")
            text.append(detail, style="green" if ok else "magenta")
        return text

    @staticmethod
    def _checklist_counts(items: Sequence[ChecklistItem]) -> tuple[int, int]:
        """Return the checklist counts value owned by this run configuration dialog."""
        total = len(items)
        passed = sum(1 for item in items if item.exists)
        return passed, total

    @staticmethod
    def _status_counts(items: Sequence[ConfigurationStatusItem]) -> tuple[int, int]:
        """Return the status counts value owned by this run configuration dialog."""
        total = len(items)
        passed = sum(1 for item in items if item.ok)
        return passed, total

    def _add_status_item_group(
        self,
        tree: Tree[object],
        title: str,
        items: Sequence[ConfigurationStatusItem],
    ) -> None:
        """Add the status item group owned by this run configuration dialog."""
        passed, total = self._status_counts(items)
        group_node = tree.root.add(
            f"{title} ({passed}/{total})",
            expand=True,
            allow_expand=False,
        )
        group_node.expand()
        for status_item in items:
            group_node.add_leaf(
                self._format_tree_label(
                    status_item.label,
                    status_item.ok,
                    status_item.detail,
                )
            )

    def _add_structure_checklist_group(self, tree: Tree[object]) -> None:
        """Add the structure checklist group owned by this run configuration dialog."""
        structure_passed, structure_total = self._checklist_counts(self._structure_check_items)
        structure_title = f"Structure Checklist ({structure_passed}/{structure_total})"
        structure_node = tree.root.add(structure_title, expand=True, allow_expand=False)
        structure_node.expand()
        for checklist_item in self._structure_check_items:
            structure_node.add_leaf(
                self._format_tree_label(
                    checklist_item.label,
                    checklist_item.exists,
                    self._display_checklist_path(checklist_item.path),
                )
            )

    def _add_framework_checklist_group(self, tree: Tree[object]) -> None:
        """Add the framework checklist group owned by this run configuration dialog."""
        framework_group_total = sum(
            len(status.checklist_items) for status in self._framework_statuses
        )
        framework_group_passed = sum(
            1
            for status in self._framework_statuses
            for item in status.checklist_items
            if item.exists
        )
        frameworks_node = tree.root.add(
            f"Framework Runtime Checklists ({framework_group_passed}/{framework_group_total})",
            expand=True,
            allow_expand=False,
        )
        frameworks_node.expand()
        for framework_status in self._framework_statuses:
            framework_passed, framework_total = self._checklist_counts(
                framework_status.checklist_items
            )
            runtime_ok = framework_total > 0 and framework_passed == framework_total
            framework_node = frameworks_node.add(
                self._format_tree_label(
                    framework_status.framework_uuid,
                    runtime_ok,
                    f"{framework_passed}/{framework_total}",
                    selected=framework_status.framework_uuid == self._selected_framework_uuid,
                ),
                expand=True,
                allow_expand=False,
            )
            framework_node.expand()
            for checklist_item in framework_status.checklist_items:
                framework_node.add_leaf(
                    self._format_tree_label(
                        checklist_item.label,
                        checklist_item.exists,
                        self._display_checklist_path(checklist_item.path),
                    )
                )

    def _populate_status_tree(self) -> None:
        """Populate the status tree owned by this run configuration dialog."""
        tree = self.query_one("#dialog_configuration_status_tree", Tree)
        tree.clear()
        tree.root.expand()

        core_items = [
            item
            for item in self._configuration_status_items
            if item.label != "Structure check" and item.label not in self._ACTION_GATE_LABELS
        ]
        action_gate_items = [
            item
            for item in self._configuration_status_items
            if item.label in self._ACTION_GATE_LABELS
        ]
        self._add_status_item_group(tree, "Model Summary", core_items)
        self._add_status_item_group(tree, "Action Gates", action_gate_items)
        self._add_structure_checklist_group(tree)
        self._add_framework_checklist_group(tree)

    _NAVIGATION_GRAPH: dict[str, dict[str, str]] = {
        "dialog_model_select": {
            "up": "dialog_configuration_status_tree",
            "down": "dialog_mode_select",
            "right": "dialog_mode_select",
        },
        "dialog_mode_select": {
            "up": "dialog_model_select",
            "left": "dialog_model_select",
            "down": "dialog_evaluation_visualization_select",
            "right": "dialog_evaluation_visualization_select",
        },
        "dialog_evaluation_visualization_select": {
            "up": "dialog_mode_select",
            "left": "dialog_mode_select",
            "down": "dialog_evaluation_scenario_select",
            "right": "dialog_evaluation_scenario_select",
        },
        "dialog_evaluation_scenario_select": {
            "up": "dialog_evaluation_visualization_select",
            "left": "dialog_evaluation_visualization_select",
            "down": "dialog_framework_select",
            "right": "dialog_framework_select",
        },
        "dialog_framework_select": {
            "up": "dialog_evaluation_scenario_select",
            "left": "dialog_evaluation_visualization_select",
            "down": "dialog_session_select",
            "right": "dialog_session_select",
        },
        "dialog_session_select": {
            "up": "dialog_framework_select",
            "left": "dialog_framework_select",
            "down": "dialog_checkpoint_select",
            "right": "dialog_checkpoint_select",
        },
        "dialog_checkpoint_select": {
            "up": "dialog_session_select",
            "left": "dialog_session_select",
            "down": "dialog_new_model",
            "right": "dialog_configuration_status_tree",
        },
        "dialog_configuration_status_tree": {
            "down": "dialog_model_select",
            "left": "dialog_new_model",
            "right": "dialog_model_select",
        },
        "dialog_new_model": {
            "up": "dialog_checkpoint_select",
            "left": "dialog_configuration_status_tree",
            "right": "dialog_new_session",
            "down": "dialog_inspect",
        },
        "dialog_new_session": {
            "up": "dialog_new_model",
            "left": "dialog_new_model",
            "right": "dialog_inspect",
            "down": "dialog_confirm",
        },
        "dialog_inspect": {
            "up": "dialog_new_model",
            "left": "dialog_new_session",
            "right": "dialog_compare",
            "down": "dialog_confirm",
        },
        "dialog_compare": {
            "up": "dialog_new_session",
            "left": "dialog_inspect",
            "down": "dialog_cancel",
        },
        "dialog_confirm": {
            "up": "dialog_inspect",
            "right": "dialog_cancel",
        },
        "dialog_cancel": {
            "up": "dialog_compare",
            "left": "dialog_confirm",
        },
    }

    def _is_active_screen(self) -> bool:
        """Return whether active screen applies to this run configuration dialog."""
        try:
            return self.app.screen is self
        except ScreenStackError:
            return False

    def on_key(self, event: Key) -> None:
        """Handle the key event for this run configuration dialog."""
        if event.key in {"up", "left", "down", "right"} and self._move_focus_in_direction(
            event.key
        ):
            event.stop()
            event.prevent_default()

    def _shortcuts_ready(self) -> bool:
        """Return the shortcuts ready value owned by this run configuration dialog."""
        return (
            self._mounted_at > 0
            and (monotonic() - self._mounted_at) >= self._SHORTCUT_GUARD_SECONDS
        )

    def _press_button_if_enabled(self, button_id: str) -> None:
        """Return whether press button if enabled applies to this run configuration dialog."""
        if not self._shortcuts_ready():
            return
        button = self.query_one(f"#{button_id}", Button)
        if button.display and not button.disabled:
            button.press()

    def action_shortcut_new_model(self) -> None:
        """Handle the shortcut new model user action for this run configuration dialog."""
        if self._training_actions_enabled():
            self._press_button_if_enabled("dialog_new_model")

    def action_shortcut_new_session(self) -> None:
        """Handle the shortcut new session user action for this run configuration dialog."""
        if self._new_session_enabled():
            self._press_button_if_enabled("dialog_new_session")

    def action_shortcut_inspect(self) -> None:
        """Handle the shortcut inspect user action for this run configuration dialog."""
        if self._inspect_enabled():
            self._press_button_if_enabled("dialog_inspect")

    def action_shortcut_compare(self) -> None:
        """Handle the shortcut compare user action for this run configuration dialog."""
        if self._compare_enabled():
            self._press_button_if_enabled("dialog_compare")

    def action_shortcut_confirm(self) -> None:
        """Handle the shortcut confirm user action for this run configuration dialog."""
        if self._confirm_enabled():
            self._press_button_if_enabled("dialog_confirm")

    @staticmethod
    def _dropdown_label(label: str, value: str) -> Text:
        """Return the dropdown label value owned by this run configuration dialog."""
        bullet_column = 15
        prefix = " "
        text = Text(style="bold")
        text.append(prefix)
        text.append(label)
        text.append(" " * max(2, bullet_column - len(prefix) - len(label)))
        text.append("•")
        text.append("  ")
        text.append(value)
        return text

    def _mode_options(self) -> list[tuple[Text, RLModelRunningModes]]:
        """Return the mode options value owned by this run configuration dialog."""
        options = [
            (
                self._dropdown_label(
                    "Running Type",
                    "Training (learn + save checkpoints)",
                ),
                RLModelRunningModes.TRAINING,
            ),
            (
                self._dropdown_label(
                    "Running Type",
                    "Inference (checkpoint deployment run)",
                ),
                RLModelRunningModes.INFERENCE,
            ),
        ]
        if self._can_evaluate:
            options.insert(
                1,
                (
                    self._dropdown_label(
                        "Running Type",
                        "Evaluation (checkpoint benchmark run)",
                    ),
                    RLModelRunningModes.EVALUATION,
                ),
            )
        return options

    def _status_tree(self) -> Tree[None]:
        """Return the status tree value owned by this run configuration dialog."""
        status_tree: Tree[None] = Tree(
            "Configuration Status",
            id="dialog_configuration_status_tree",
        )
        status_tree.show_root = False
        status_tree.show_guides = True
        status_tree.guide_depth = 2
        status_tree.auto_expand = False
        return status_tree

    def _mode_select(self) -> Select[RLModelRunningModes]:
        """Return the mode select value owned by this run configuration dialog."""
        mode_select: Select[RLModelRunningModes] = Select(
            self._mode_options(),
            value=self._selected_mode,
            allow_blank=False,
            id="dialog_mode_select",
            disabled=not self._has_model or self._configuration_locked,
        )
        mode_select.border_title = "Mode"
        return mode_select

    def _framework_select(self) -> Select[str]:
        """Return the framework select value owned by this run configuration dialog."""
        framework_options = [
            (self._dropdown_label("Framework", label), value)
            for label, value in self._framework_options
        ]
        if self._has_model:
            framework_select = Select(
                framework_options,
                value=self._selected_framework_uuid,
                allow_blank=False,
                id="dialog_framework_select",
                disabled=self._configuration_locked,
            )
        else:
            framework_select = Select(
                [
                    (
                        self._dropdown_label(
                            "Framework",
                            "No model loaded yet.",
                        ),
                        "__no_model__",
                    )
                ],
                value="__no_model__",
                allow_blank=False,
                id="dialog_framework_select",
                disabled=True,
            )
        framework_select.border_title = "Framework"
        return framework_select

    def _evaluation_profile_value(self) -> str:
        """Return the evaluation profile value value owned by this run configuration dialog."""
        if self._evaluation_target == "library":
            return "library"
        if self._selected_evaluation_visualization == "gui":
            return "gui"
        return f"headless:{self._selected_evaluation_pacing}"

    def _evaluation_library_hint_text(self) -> str:
        """Explain how a selected library is configured and then run."""

        scope = self._selected_evaluation_library_scope
        if scope is not None:
            return f"Selected: {scope.library_id} • Apply to choose a library"
        return "Apply: choose existing or generate a library"

    def _sync_evaluation_controls(self) -> None:
        """Show only the controls relevant to the selected Evaluation target."""

        with suppress(NoMatches):
            profile = self.query_one("#dialog_evaluation_visualization_select", Select)
            profile.set_class(self._evaluation_target == "library", "library-target")
            scenario = self.query_one("#dialog_evaluation_scenario_select", Select)
            scenario.display = (
                self._selected_mode is RLModelRunningModes.EVALUATION
                and self._evaluation_target == "single"
                and bool(self._evaluation_scenario_options)
            )
            scenario.disabled = (
                self._configuration_locked
                or self._evaluation_target == "library"
                or not self._evaluation_scenario_options
            )
            if self._selected_evaluation_scenario_index is not None:
                scenario.value = self._selected_evaluation_scenario_index
            hint = self.query_one("#dialog_evaluation_library_hint", Static)
            hint.display = (
                self._selected_mode is RLModelRunningModes.EVALUATION
                and self._evaluation_target == "library"
            )
            hint.update(self._evaluation_library_hint_text())
            self.query_one("#dialog_confirm", Button).disabled = not self._confirm_enabled()

    def _evaluation_visualization_select(self) -> Select[str]:
        """Return the evaluation visualization select value owned by this run configuration dialog.
        """
        visualization_select: Select[str] = Select(
            [
                (
                    self._dropdown_label(
                        "Evaluation",
                        "GUI (real-time)",
                    ),
                    "gui",
                ),
                (
                    self._dropdown_label(
                        "Evaluation",
                        "Headless (maximum throughput)",
                    ),
                    "headless:maximum-throughput",
                ),
                (
                    self._dropdown_label(
                        "Evaluation",
                        "Headless (real-time)",
                    ),
                    "headless:realtime",
                ),
                (
                    self._dropdown_label(
                        "Evaluation",
                        "Library (all scenarios)",
                    ),
                    "library",
                ),
            ],
            value=self._evaluation_profile_value(),
            allow_blank=False,
            id="dialog_evaluation_visualization_select",
            disabled=self._configuration_locked,
        )
        visualization_select.border_title = "Evaluation profile"
        visualization_select.display = self._selected_mode is RLModelRunningModes.EVALUATION
        visualization_select.set_class(self._evaluation_target == "library", "library-target")
        return visualization_select

    def _evaluation_scenario_select(self) -> Select[int]:
        """Build the Evaluation-only scenario index selector."""

        options = self._evaluation_scenario_options or [
            ("No evaluation scenarios available", -1)
        ]
        value = (
            self._selected_evaluation_scenario_index
            if self._selected_evaluation_scenario_index is not None
            else options[0][1]
        )
        scenario_select: Select[int] = Select(
            options,
            value=value,
            allow_blank=False,
            id="dialog_evaluation_scenario_select",
            disabled=(
                self._configuration_locked
                or not self._evaluation_scenario_options
                or self._evaluation_target == "library"
            ),
        )
        scenario_select.border_title = "Evaluation scenario"
        scenario_select.display = bool(self._evaluation_scenario_options) and (
            self._selected_mode is RLModelRunningModes.EVALUATION
            and self._evaluation_target == "single"
        )
        return scenario_select

    def _select_evaluation_scenario(self, value: object) -> None:
        """Accept one scenario option emitted by the Evaluation selector."""

        if self._evaluation_target == "library":
            return
        if not isinstance(value, int) or isinstance(value, bool):
            return
        available = {index for _, index in self._evaluation_scenario_options}
        if value in available:
            self._selected_evaluation_scenario_index = value
            self._last_single_evaluation_scenario_index = value
            self._selected_evaluation_library_scope = None

    def _compose_selection_controls(self) -> ComposeResult:
        """Compose the selection controls owned by this run configuration dialog."""
        with ScrollableContainer(id="dialog_configuration_status_scroll") as status_scroll:
            status_scroll.border_title = "Configuration Status"
            yield self._status_tree()
        yield selector_button(
            self._model_button_label(),
            id="dialog_model_select",
            disabled=not self._view_state().model_picker_enabled,
        )
        yield self._mode_select()
        yield self._evaluation_visualization_select()
        yield self._evaluation_scenario_select()
        library_hint = Static(
            self._evaluation_library_hint_text(),
            id="dialog_evaluation_library_hint",
        )
        library_hint.display = (
            self._selected_mode is RLModelRunningModes.EVALUATION
            and self._evaluation_target == "library"
        )
        yield library_hint
        yield self._framework_select()
        yield selector_button(
            self._session_button_label(),
            id="dialog_session_select",
            disabled=not self._session_picker_enabled(),
        )
        yield selector_button(
            self._checkpoint_button_label(),
            id="dialog_checkpoint_select",
            disabled=not self._checkpoint_picker_enabled(),
        )

    def _compose_action_groups(self) -> ComposeResult:
        """Compose the action groups owned by this run configuration dialog."""
        with Grid(id="dialog_model_actions_top"):
            yield Button(
                "New Model",
                id="dialog_new_model",
                variant="primary",
                disabled=not self._training_actions_enabled(),
            )
            yield Button(
                "New Session",
                id="dialog_new_session",
                variant="primary",
                disabled=not self._new_session_enabled(),
            )
        with Grid(id="dialog_model_actions_secondary"):
            yield Button(
                "Inspect",
                id="dialog_inspect",
                variant="primary",
                disabled=not self._inspect_enabled(),
            )
            yield Button(
                "Compare Pack",
                id="dialog_compare",
                variant="primary",
                disabled=not self._compare_enabled(),
            )
        with Grid(id="dialog_model_actions_bottom"):
            yield Button(
                self._confirm_label(),
                id="dialog_confirm",
                variant="success",
                disabled=not self._confirm_enabled(),
            )
            yield Button("Cancel", id="dialog_cancel", variant="default")

    def _configuration_footer(self) -> ModalFooter:
        """Return the configuration footer value owned by this run configuration dialog."""
        return ModalFooter.from_binding_rows(
            cast(Sequence[Binding], self.BINDINGS),
            [
                ["shortcut_new_model", "shortcut_new_session"],
                ["shortcut_inspect", "shortcut_compare"],
                ["shortcut_confirm", "dismiss"],
            ],
            id="configuration_modal_footer",
            label_overrides={
                "shortcut_inspect": "Inspect",
                "shortcut_compare": "Compare Pack",
                "shortcut_confirm": self._confirm_label(),
                "dismiss": "Close",
            },
        )

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this run configuration dialog."""
        model_name = self._loaded_model_name if self._has_model else None
        title = f"Run Configuration • Model: {model_name or '-'}"
        with Vertical(id="configuration_dialog_container") as container:
            container.border_title = title
            with (
                ScrollableContainer(id="dialog_startup_cfg_scroll"),
                Grid(id="dialog_startup_cfg"),
            ):
                yield from self._compose_selection_controls()
                yield from self._compose_action_groups()
            yield self._configuration_footer()

    def _select_evaluation_profile(self, value: object) -> None:
        if value == "library":
            if self._evaluation_target == "single":
                self._last_single_evaluation_scenario_index = (
                    self._selected_evaluation_scenario_index
                )
            self._evaluation_target = "library"
            self._selected_evaluation_scenario_index = None
            self._selected_evaluation_visualization = "headless"
            self._selected_evaluation_pacing = "maximum-throughput"
            self._sync_evaluation_controls()
            return
        profiles: dict[str, tuple[EvaluationVisualization, EvaluationPacing]] = {
            "gui": ("gui", "realtime"),
            "headless": ("headless", "maximum-throughput"),
            "headless:maximum-throughput": ("headless", "maximum-throughput"),
            "headless:realtime": ("headless", "realtime"),
        }
        profile = profiles.get(str(value))
        if profile is None:
            return
        if self._evaluation_target == "library":
            self._selected_evaluation_scenario_index = _selected_evaluation_scenario_index(
                self._evaluation_scenario_options,
                self._last_single_evaluation_scenario_index,
            )
        self._evaluation_target = "single"
        self._selected_evaluation_library_scope = None
        self._selected_evaluation_visualization, self._selected_evaluation_pacing = profile
        self._sync_evaluation_controls()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle the select changed event for this run configuration dialog."""
        if not self._is_active_screen():
            return
        if not self._ready:
            return
        if self._configuration_locked:
            return
        if event.select.id == "dialog_mode_select":
            if event.value == self._selected_mode:
                return
            if not isinstance(event.value, RLModelRunningModes):
                return
            self._selected_mode = event.value
            self._selected_evaluation_scenario_index = None
            self._last_single_evaluation_scenario_index = None
            self._safe_dismiss(self._result(ConfigurationDialogAction.REFRESH))
        elif event.select.id == "dialog_evaluation_visualization_select":
            self._select_evaluation_profile(event.value)
        elif event.select.id == "dialog_evaluation_scenario_select":
            self._select_evaluation_scenario(event.value)
        elif event.select.id == "dialog_framework_select":
            value = str(event.value)
            if value == self._selected_framework_uuid:
                return
            self._selected_framework_uuid = value
            self._selected_evaluation_scenario_index = None
            self._last_single_evaluation_scenario_index = None
            self._safe_dismiss(self._result(ConfigurationDialogAction.REFRESH))

    def _session_selector_dialog(self) -> SessionConfigurationDialog:
        """Return the session selector dialog value owned by this run configuration dialog."""
        return SessionConfigurationDialog(
            session_options=self._session_options,
            selected_session_name=self._selected_session_name,
            allow_delete=True,
            allow_edit=True,
        )

    def _apply_administration_transition(
        self,
        transition: ConfigurationAdministrationTransition,
    ) -> None:
        """Apply the administration transition transition owned by this run configuration dialog."""
        for effect in transition.effects:
            self._execute_administration_effect(effect)

    def _execute_administration_effect(
        self,
        effect: ConfigurationAdministrationEffect,
    ) -> None:
        """Execute the administration effect owned by this run configuration dialog."""
        if isinstance(effect, OpenSessionSelector):
            self.app.push_screen(
                self._session_selector_dialog(),
                partial(self._on_session_configuration_result, effect.identity),
            )
        elif isinstance(effect, OpenCheckpointSelector):
            self.app.push_screen(
                self._checkpoint_selector_dialog(effect.identity.session_name),
                partial(self._on_checkpoint_configuration_result, effect.identity),
            )
        elif isinstance(effect, ConfirmSessionDeletion):
            session_request = effect.request
            self.app.push_screen(
                secure_delete_session_dialog(session_request.display_name),
                partial(self._confirm_session_deletion, session_request),
            )
        elif isinstance(effect, ConfirmCheckpointDeletion):
            checkpoint_request = effect.request
            self.app.push_screen(
                secure_delete_checkpoint_dialog(
                    checkpoint_request.checkpoint_name,
                    checkpoint_request.identity.session_name,
                ),
                partial(self._confirm_checkpoint_deletion, checkpoint_request),
            )
        elif isinstance(effect, PresentAdministrationNotification):
            notification = effect.notification
            self.app.notify(
                notification.message,
                title=notification.title,
                severity=notification.severity,
            )
        elif isinstance(effect, DismissConfiguration):
            self._safe_dismiss(self._result(effect.action))
        elif isinstance(effect, ReopenSessionSelector):
            self._open_session_configuration_dialog()
        elif isinstance(effect, ReopenCheckpointSelector):
            self._open_checkpoint_configuration_dialog()

    def _open_session_configuration_dialog(self) -> None:
        """Open the session configuration dialog workflow owned by this run configuration dialog."""
        self._apply_administration_transition(
            self._administration_coordinator.open_session_selector(
                enabled=self._session_picker_enabled(),
                active=self._is_active_screen(),
            )
        )

    def _on_session_configuration_result(
        self,
        identity: SessionSelectorIdentity,
        result: SessionConfigurationDialogResult | None,
    ) -> None:
        """Handle the session configuration result event for this run configuration dialog."""
        if result is None:
            return
        event = SessionAdministrationEvent.from_raw(
            result.action,
            result.session_name,
        )
        if event is None:
            return
        previous_selection = (
            self._selected_session_name,
            self._selected_checkpoint_name,
        )
        transition = self._administration_coordinator.handle_session_result(
            identity,
            event,
            active=self._is_active_screen(),
        )
        self._reset_evaluation_scenario_if_selection_changed(
            previous_selection,
            transition,
        )
        self._apply_administration_transition(transition)

    def _confirm_session_deletion(
        self,
        request: SessionDeletionRequest,
        confirmed: bool | None,
    ) -> None:
        """Confirm the session deletion owned by this run configuration dialog."""
        previous_selection = (
            self._selected_session_name,
            self._selected_checkpoint_name,
        )
        transition = self._administration_coordinator.confirm_session_deletion(
            request,
            confirmed,
            active=self._is_active_screen(),
        )
        self._reset_evaluation_scenario_if_selection_changed(
            previous_selection,
            transition,
        )
        self._apply_administration_transition(transition)

    def _fresh_checkpoint_copy(self) -> tuple[str, str, str]:
        """Return the fresh checkpoint copy value owned by this run configuration dialog."""
        if self._selected_mode == RLModelRunningModes.EVALUATION:
            return (
                "Constant-speed diagnostic (no checkpoint)",
                "Constant-speed diagnostic",
                "constant speed diagnostic no checkpoint evaluation",
            )
        return (
            "Start fresh (no checkpoint)",
            "Start fresh",
            "start fresh no checkpoint fresh training",
        )

    def _checkpoint_selector_dialog(
        self,
        session_name: str,
    ) -> CheckpointConfigurationDialog:
        """Checkpoint the selector dialog owned by this run configuration dialog."""
        label, mode_label, search_text = self._fresh_checkpoint_copy()
        return CheckpointConfigurationDialog(
            session_name=session_name,
            checkpoint_options=self._checkpoint_options,
            selected_checkpoint_name=self._selected_checkpoint_name,
            allow_fresh_training=self._selected_mode != RLModelRunningModes.INFERENCE,
            fresh_selection_label=label,
            fresh_selection_mode_label=mode_label,
            fresh_selection_search_text=search_text,
            allow_delete=True,
            allow_rename=False,
            allow_edit=True,
        )

    def _open_checkpoint_configuration_dialog(self) -> None:
        """Open the checkpoint configuration dialog workflow owned by this run configuration
        dialog.
        """
        self._apply_administration_transition(
            self._administration_coordinator.open_checkpoint_selector(
                enabled=self._checkpoint_picker_enabled(),
                active=self._is_active_screen(),
            )
        )

    def _on_checkpoint_configuration_result(
        self,
        identity: CheckpointSelectorIdentity,
        result: CheckpointConfigurationDialogResult | None,
    ) -> None:
        """Handle the checkpoint configuration result event for this run configuration dialog."""
        if result is None:
            return
        event = CheckpointAdministrationEvent.from_raw(
            result.action,
            result.checkpoint_name,
            result.renamed_checkpoint_name,
        )
        if event is None:
            return
        previous_selection = (
            self._selected_session_name,
            self._selected_checkpoint_name,
        )
        transition = self._administration_coordinator.handle_checkpoint_result(
            identity,
            event,
            active=self._is_active_screen(),
        )
        self._reset_evaluation_scenario_if_selection_changed(
            previous_selection,
            transition,
        )
        self._apply_administration_transition(transition)

    def _reset_evaluation_scenario_if_selection_changed(
        self,
        previous_selection: tuple[str | None, str | None],
        transition: ConfigurationAdministrationTransition,
    ) -> None:
        """Clear a transient override when its session/checkpoint identity changes."""

        current_selection = (
            transition.state.session_name,
            transition.state.checkpoint_name,
        )
        if current_selection != previous_selection:
            self._selected_evaluation_scenario_index = None

    def _confirm_checkpoint_deletion(
        self,
        request: CheckpointDeletionRequest,
        confirmed: bool | None,
    ) -> None:
        """Confirm the checkpoint deletion owned by this run configuration dialog."""
        previous_selection = (
            self._selected_session_name,
            self._selected_checkpoint_name,
        )
        transition = self._administration_coordinator.confirm_checkpoint_deletion(
            request,
            confirmed,
            active=self._is_active_screen(),
        )
        self._reset_evaluation_scenario_if_selection_changed(
            previous_selection,
            transition,
        )
        self._apply_administration_transition(transition)

    def _result(self, action: ConfigurationDialogAction) -> ConfigurationDialogResult:
        """Return the result value owned by this run configuration dialog."""
        return ConfigurationDialogResult(
            action=action,
            mode=self._selected_mode,
            framework_uuid=self._selected_framework_uuid,
            session_name=self._selected_session_name,
            checkpoint_name=self._selected_checkpoint_name,
            expected_model_uuid=self._expected_model_uuid,
            evaluation_visualization=self._selected_evaluation_visualization,
            evaluation_pacing=self._selected_evaluation_pacing,
            evaluation_scenario_index=self._selected_evaluation_scenario_index,
            evaluation_library_scope=self._selected_evaluation_library_scope,
            evaluation_target=self._evaluation_target,
        )

    def _safe_dismiss(self, result: ConfigurationDialogResult | None) -> None:
        """Return dismiss converted to a safe value for this run configuration dialog."""
        if not self._is_active_screen():
            return
        self._invalidate_selector_callbacks()
        try:
            self.dismiss(result)
        except ScreenStackError:
            return

    def _emit_action_if(
        self,
        enabled: bool,
        action: ConfigurationDialogAction,
    ) -> None:
        """Emit the action if owned by this run configuration dialog."""
        if enabled:
            self._safe_dismiss(self._result(action))

    def _open_model_selector(self) -> None:
        """Open the model selector workflow owned by this run configuration dialog."""
        self._emit_action_if(
            self._view_state().model_picker_enabled,
            ConfigurationDialogAction.LOAD_MODEL,
        )

    def _open_evaluation_libraries(self) -> None:
        if (
            self._configuration_locked
            or not self._has_model
            or self._selected_mode is not RLModelRunningModes.EVALUATION
            or self._evaluation_target != "library"
        ):
            return
        adapter = EvaluationLibrariesDialogAdapter(self._services().evaluation_libraries)
        self.app.push_screen(
            EvaluationLibrariesDialog(adapter), self._on_evaluation_library_selected
        )

    def _on_evaluation_library_selected(
        self, selection: tuple[str, str] | None
    ) -> None:
        if selection is None or not self._is_active_screen():
            return
        self._selected_evaluation_library_scope = LibraryEvaluationScope(
            library_id=selection[0], manifest_sha256=selection[1]
        )
        self._evaluation_target = "library"
        self._selected_evaluation_scenario_index = None
        self._selected_evaluation_visualization = "headless"
        self._selected_evaluation_pacing = "maximum-throughput"
        self._sync_evaluation_controls()
        if self._selected_checkpoint_name is None or not self._view_state().confirm_enabled:
            self.app.notify(
                "Library selected. Choose a checkpoint, then Apply to configure evaluation.",
                severity="warning",
            )
            return
        try:
            self._services().evaluation_libraries.resolve_selection(
                self._selected_evaluation_library_scope
            )
        except Exception as error:
            self.app.notify(str(error), severity="error")
            return
        self._safe_dismiss(self._result(ConfigurationDialogAction.APPLY))

    def _confirm_configuration(self) -> None:
        """Confirm the configuration owned by this run configuration dialog."""
        if (
            self._selected_mode is RLModelRunningModes.EVALUATION
            and self._evaluation_target == "library"
        ):
            self._open_evaluation_libraries()
            return
        self._emit_action_if(
            self._confirm_enabled(),
            ConfigurationDialogAction.APPLY,
        )

    def _create_model(self) -> None:
        """Create the model owned by this run configuration dialog."""
        self._emit_action_if(
            self._training_actions_enabled(),
            ConfigurationDialogAction.CREATE_MODEL,
        )

    def _inspect_tensorboard(self) -> None:
        """Inspect the tensorboard owned by this run configuration dialog."""
        self._emit_action_if(
            self._inspect_enabled(),
            ConfigurationDialogAction.INSPECT_TENSORBOARD,
        )

    def _create_comparison_pack(self) -> None:
        """Create the comparison pack owned by this run configuration dialog."""
        self._emit_action_if(
            self._compare_enabled(),
            ConfigurationDialogAction.CREATE_COMPARISON_PACK,
        )

    def _create_session(self) -> None:
        """Create the session owned by this run configuration dialog."""
        self._emit_action_if(
            self._new_session_enabled(),
            ConfigurationDialogAction.CREATE_SESSION,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this run configuration dialog."""
        if not self._is_active_screen():
            return
        handlers: dict[str, Callable[[], None]] = {
            "dialog_model_select": self._open_model_selector,
            "dialog_session_select": self._open_session_configuration_dialog,
            "dialog_checkpoint_select": self._open_checkpoint_configuration_dialog,
            "dialog_confirm": self._confirm_configuration,
            "dialog_new_model": self._create_model,
            "dialog_inspect": self._inspect_tensorboard,
            "dialog_compare": self._create_comparison_pack,
            "dialog_new_session": self._create_session,
            "dialog_cancel": partial(self._safe_dismiss, None),
        }
        button_id = event.button.id
        if button_id is None:
            return
        handler = handlers.get(button_id)
        if handler is not None:
            handler()

    def on_mount(self) -> None:
        """Ensure the dialog is interactive after every reopen."""
        self._mounted_at = monotonic()
        self._populate_status_tree()
        for candidate in (
            "#dialog_configuration_status_tree",
            "#dialog_model_select",
            "#dialog_mode_select",
            "#dialog_cancel",
            "#dialog_confirm",
            "#dialog_new_model",
        ):
            try:
                widget = self.query_one(candidate)
                if widget.display and not getattr(widget, "disabled", False) and widget.can_focus:
                    widget.focus()
                    break
            except NoMatches:
                continue
        self.call_after_refresh(lambda: setattr(self, "_ready", True))
        if self._auto_open_session_selector:
            self.call_after_refresh(self._open_session_configuration_dialog)
        if self._auto_open_checkpoint_selector:
            self.call_after_refresh(self._open_checkpoint_configuration_dialog)

    def on_unmount(self) -> None:
        """Invalidate child callbacks after this modal leaves the screen stack."""

        self._invalidate_selector_callbacks()
