# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Edit named scenario groups and select one verified model-local library."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from lanista.gui.adapters.control_panel.evaluation_libraries_dialog_adapter import (
    EvaluationLibrariesDialogAdapter,
)
from lanista.gui.services.configuration.evaluation_library_draft_service import (
    EvaluationGroupDraft,
)
from lanista.reinforcengine.rl_model.evaluation.library_application_service import (
    EvaluationLibraryCatalogItem,
)


class EvaluationLibrariesDialog(ModalScreen[tuple[str, str] | None]):
    """Show a model-scoped catalog and an expandable grouped-generation form."""

    BINDINGS = [Binding("escape", "dismiss(None)", "Close")]
    CSS = """
    #evaluation_libraries_dialog { width: 90%; height: 90%; margin: 2 4; padding: 1 2;
        border: round $accent; background: $surface; }
    #evaluation_libraries_content { height: 1fr; }
    #evaluation_libraries_catalog, #evaluation_libraries_groups { height: auto; border: round $secondary; }
    .evaluation_group_row { height: auto; border: solid $secondary; padding: 1; }
    .evaluation_group_fields { height: auto; }
    .evaluation_group_fields Input { width: 1fr; }
    #evaluation_libraries_actions { height: auto; }
    """

    def __init__(self, adapter: EvaluationLibrariesDialogAdapter) -> None:
        super().__init__()
        self._adapter = adapter
        self._draft = adapter.new_draft()
        self._catalog: tuple[EvaluationLibraryCatalogItem, ...] = ()
        self._busy = False

    @staticmethod
    def _input(value: str, title: str, widget_id: str) -> Input:
        field = Input(value=value, id=widget_id)
        field.border_title = title
        return field

    def _group_row(self, group: EvaluationGroupDraft) -> Vertical:
        identity = group.group_id
        return Vertical(
            Static(identity),
            self._input(group.name, "Name", f"group_name_{identity}"),
            Horizontal(
                self._input(group.vehicle_count, "Vehicles", f"group_vehicles_{identity}"),
                self._input(group.pedestrian_count, "Pedestrians", f"group_pedestrians_{identity}"),
                self._input(group.scenario_count, "Scenarios", f"group_scenarios_{identity}"),
                classes="evaluation_group_fields",
            ),
            Button("Remove group", id=f"remove_{identity}", variant="warning"),
            classes="evaluation_group_row",
        )

    def compose(self) -> ComposeResult:
        with Vertical(id="evaluation_libraries_dialog"):
            yield Static("Evaluation Libraries — selected model")
            yield Static("Choose a verified library below, or generate a new one. Start runs the selected library.")
            with Horizontal(id="evaluation_libraries_actions"):
                yield Button("Add group", id="add_group")
                yield Button("Generate library", id="generate_library", variant="primary")
                yield Button("Cancel", id="cancel_libraries")
            yield Static("", id="evaluation_libraries_status")
            with ScrollableContainer(id="evaluation_libraries_content"):
                with Vertical(id="evaluation_libraries_catalog"):
                    yield Static("Loading libraries…", id="evaluation_libraries_catalog_message")
                yield self._input(self._draft.library_id, "New library ID", "library_id")
                yield self._input(self._draft.name, "Display name", "library_name")
                yield self._input(self._draft.description, "Description", "library_description")
                yield self._input(self._draft.master_seed, "Master seed", "library_seed")
                with Vertical(id="evaluation_libraries_groups"):
                    for group in self._draft.groups:
                        yield self._group_row(group)


    async def on_mount(self) -> None:
        await self._refresh_catalog()
        self.set_interval(0.5, self._poll_model)

    async def _render_groups(self) -> None:
        container = self.query_one("#evaluation_libraries_groups", Vertical)
        await container.remove_children()
        if self._draft.groups:
            await container.mount(*(self._group_row(group) for group in self._draft.groups))
        else:
            await container.mount(Static("Add a group to generate scenarios."))

    async def _refresh_catalog(self) -> None:
        previous_model = self._draft.model_root
        try:
            draft, catalog = await asyncio.to_thread(self._adapter.refresh, self._draft)
        except Exception as error:
            self._status(str(error))
            return
        self._draft = draft
        self._catalog = catalog
        if draft.model_root != previous_model:
            for widget_id, value in (
                ("library_id", draft.library_id),
                ("library_name", draft.name),
                ("library_description", draft.description),
                ("library_seed", draft.master_seed),
            ):
                self.query_one(f"#{widget_id}", Input).value = value
            await self._render_groups()
            self._status("Selected model changed; previous library selection was cleared.")
        container = self.query_one("#evaluation_libraries_catalog", Vertical)
        await container.remove_children()
        if not catalog:
            await container.mount(Static("No grouped evaluation libraries for this model."))
        else:
            await container.mount(
                *(
                    Button(
                        f"Use for evaluation: {item.name} ({item.library_id}) — {item.created_at or "unknown date"}"
                        f" — verified — {item.group_count} groups, {item.scenario_count} scenarios"
                        if item.valid
                        else f"{item.library_id} — invalid: {item.diagnostic}",
                        id=f"select_library_{index}",
                        disabled=not item.valid,
                    )
                    for index, item in enumerate(catalog)
                )
            )

    async def _poll_model(self) -> None:
        if self._busy:
            return
        try:
            changed = self._adapter.service.current_model_root() != self._draft.model_root
        except Exception:
            return
        if changed:
            await self._refresh_catalog()

    def _status(self, message: str) -> None:
        self.query_one("#evaluation_libraries_status", Static).update(message)

    def on_input_changed(self, event: Input.Changed) -> None:
        identity = event.input.id or ""
        library_fields = {
            "library_id": "library_id",
            "library_name": "name",
            "library_description": "description",
            "library_seed": "master_seed",
        }
        if identity in library_fields:
            self._draft = replace(self._draft, **{library_fields[identity]: event.value})
            return
        for prefix, field_name in (
            ("group_name_", "name"),
            ("group_vehicles_", "vehicle_count"),
            ("group_pedestrians_", "pedestrian_count"),
            ("group_scenarios_", "scenario_count"),
        ):
            if identity.startswith(prefix):
                group_id = identity.removeprefix(prefix)
                self._draft = self._adapter.draft_service.update_group(
                    self._draft, group_id, **{field_name: event.value}
                )
                return

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        identity = event.button.id or ""
        if self._busy:
            return
        if identity == "cancel_libraries":
            self.dismiss(None)
        elif identity == "add_group":
            self._draft = self._adapter.draft_service.add_group(self._draft)
            await self._render_groups()
        elif identity.startswith("remove_group-"):
            self._draft = self._adapter.draft_service.remove_group(
                self._draft, identity.removeprefix("remove_")
            )
            await self._render_groups()
        elif identity.startswith("select_library_"):
            index = int(identity.removeprefix("select_library_"))
            try:
                selection = await asyncio.to_thread(
                    self._adapter.select, self._draft, self._catalog[index]
                )
            except Exception as error:
                self._status(str(error))
            else:
                self.dismiss(selection)
        elif identity == "generate_library":
            self._busy = True
            event.button.disabled = True
            self._status("Generating and verifying all scenarios…")
            try:
                selection = await asyncio.to_thread(self._adapter.generate, self._draft)
            except Exception as error:
                self._status(str(error))
                event.button.disabled = False
            else:
                self.dismiss(selection)
            finally:
                self._busy = False


__all__ = ["EvaluationLibrariesDialog"]
