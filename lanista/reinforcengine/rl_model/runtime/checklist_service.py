# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate runtime checklist."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

from rich.markup import escape

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.metadata.schema import (
    MetadataKind,
    validate_versioned_metadata_payload,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    RAY_RESOURCES_DIR_NAME,
    RESOURCES_DIR_NAME,
    SB3_RESOURCES_DIR_NAME,
    TENSORBOARD_COMPARISON_DIR_NAME,
    TENSORBOARD_DIR_NAME,
    TENSORBOARD_EVALUATION_DIR_NAME,
    TENSORBOARD_RAY_DIR_NAME,
    TENSORBOARD_SB3_DIR_NAME,
    TENSORBOARD_TRAININGS_DIR_NAME,
    TRAINING_SCENARIOS_DIR_NAME,
    get_framework_model_configuration_path,
)
from lanista.reinforcengine.rl_model.storage.persistence import read_yaml_mapping_file


class ConfigurationProvider(Protocol):
    """Minimal manifest surface required by read-only runtime checklists."""

    def get_configuration(self) -> object:
        """Return the configuration exposed by this configuration provider."""
        ...


@dataclass
class ChecklistItem:
    """Carry the data required for checklist item."""

    label: str
    path: str
    kind: str  # file | dir | attr
    exists: bool


@dataclass
class LoadChecklistResult:
    """Report the result of load checklist."""

    valid: bool
    model_name: str | None
    summary: str
    items: list[ChecklistItem]


@dataclass
class RuntimeChecklistResult:
    """Report the result of runtime checklist."""

    valid: bool
    model_name: str | None
    summary: str
    items: list[ChecklistItem]


class RuntimeChecklistService:
    """Coordinate runtime checklist operations."""

    @staticmethod
    def shorten_checklist_path(path: str) -> str:
        """Return the shorten checklist path value owned by this runtime checklist service."""
        text = str(path or "").strip()
        if not text:
            return "<missing>"
        marker = f"{os.sep}models{os.sep}"
        marker_index = text.find(marker)
        if marker_index >= 0:
            return text[marker_index + len(marker):]
        return text

    @staticmethod
    def quote_checklist_value(value: str) -> str:
        """Return the quote checklist value value owned by this runtime checklist service."""
        text = str(value or "").strip() or "<missing>"
        return f"\"{text}\""

    @classmethod
    def format_checklist_line(cls, prefix: str, label: str, exists: bool, value: str) -> str:
        """Format the checklist line for presentation by this runtime checklist service."""
        marker = "[green]✓[/green]" if exists else "[bold red]✗[/bold red]"
        status = "[green]OK[/green]" if exists else "[bold red]NO[/bold red]"
        formatted_value = escape(cls.quote_checklist_value(value))
        value_style = "green" if exists else "magenta"
        prefix_text = escape(f"[{prefix}]")
        return (
            f"[dim]{prefix_text}[/dim] "
            f"{marker} {status} "
            f"[white]{escape(label)}[/white] "
            f"[{value_style}]{formatted_value}[/{value_style}]"
        )

    @staticmethod
    def format_checklist_summary(
        prefix: str,
        passed_count: int,
        total_count: int,
        valid: bool,
        model_name: str | None = None,
    ) -> str:
        """Format the checklist summary for presentation by this runtime checklist service."""
        status = "[green]PASS[/green]" if valid else "[bold red]FAIL[/bold red]"
        suffix = f" [green]\"{escape(model_name)}\"[/green]" if model_name else ""
        prefix_text = escape(f"[{prefix}]")
        return f"[dim]{prefix_text}[/dim] {status} [white]{passed_count}/{total_count}[/white]{suffix}"

    def build_model_load_checklist(
        self,
        model_manifest: ConfigurationProvider,
    ) -> LoadChecklistResult:
        """Build the model load checklist required by this runtime checklist service."""
        cfg = model_manifest.get_configuration()
        model_folder = str(getattr(cfg, "model_folder_path", "") or "")
        model_name = str(getattr(cfg, "model_name", "") or "") or None

        def _file(label: str, *parts: str) -> ChecklistItem:
            """Perform the file step within the enclosing build model load checklist workflow."""
            path = os.path.join(model_folder, *parts)
            return ChecklistItem(label=label, path=path, kind="file", exists=os.path.isfile(path))

        def _dir(label: str, *parts: str) -> ChecklistItem:
            """Perform the dir step within the enclosing build model load checklist workflow."""
            path = os.path.join(model_folder, *parts)
            return ChecklistItem(label=label, path=path, kind="dir", exists=os.path.isdir(path))

        items = [
            _file("Model manifest", MODEL_MANIFEST_FILE_NAME),
            _file("Model configuration", MODEL_CONFIGURATION_FILE_NAME),
            _dir("Resources", RESOURCES_DIR_NAME),
            _dir("Training scenarios", RESOURCES_DIR_NAME, TRAINING_SCENARIOS_DIR_NAME),
            _dir("Ray resources", RESOURCES_DIR_NAME, RAY_RESOURCES_DIR_NAME),
            _dir("SB3 resources", RESOURCES_DIR_NAME, SB3_RESOURCES_DIR_NAME),
            _dir("TensorBoard root", TENSORBOARD_DIR_NAME),
            _dir("TensorBoard evaluation", TENSORBOARD_DIR_NAME, TENSORBOARD_EVALUATION_DIR_NAME),
            _dir("TensorBoard comparison", TENSORBOARD_DIR_NAME, TENSORBOARD_COMPARISON_DIR_NAME),
            _dir("TensorBoard trainings", TENSORBOARD_DIR_NAME, TENSORBOARD_TRAININGS_DIR_NAME),
            _dir("TensorBoard ray", TENSORBOARD_DIR_NAME, TENSORBOARD_TRAININGS_DIR_NAME, TENSORBOARD_RAY_DIR_NAME),
            _dir("TensorBoard sb3", TENSORBOARD_DIR_NAME, TENSORBOARD_TRAININGS_DIR_NAME, TENSORBOARD_SB3_DIR_NAME),
            _dir("TensorBoard evaluation ray", TENSORBOARD_DIR_NAME, TENSORBOARD_EVALUATION_DIR_NAME, TENSORBOARD_RAY_DIR_NAME),
            _dir("TensorBoard evaluation sb3", TENSORBOARD_DIR_NAME, TENSORBOARD_EVALUATION_DIR_NAME, TENSORBOARD_SB3_DIR_NAME),
        ]

        missing = [item.label for item in items if not item.exists]
        if missing:
            summary = f"Model structure check failed: missing {', '.join(missing)}."
            return LoadChecklistResult(valid=False, model_name=model_name, summary=summary, items=items)

        summary = "Model structure check passed."
        return LoadChecklistResult(valid=True, model_name=model_name, summary=summary, items=items)

    def build_model_runtime_checklist(
        self,
        model_manifest: ConfigurationProvider,
    ) -> RuntimeChecklistResult:
        """Build the model runtime checklist required by this runtime checklist service."""
        cfg = model_manifest.get_configuration()
        model_name = str(getattr(cfg, "model_name", "") or "") or None

        def _attr(label: str, value: object) -> ChecklistItem:
            """Perform the attr step within the enclosing build model runtime checklist workflow."""
            text = str(value or "").strip()
            return ChecklistItem(label=label, path=text, kind="attr", exists=bool(text))

        model_implementation = getattr(cfg, "model_implementation", None)
        items = [
            _attr("Model implementation module", getattr(model_implementation, "module", None)),
            _attr("Model implementation object", getattr(model_implementation, "object", None)),
        ]

        missing = [item.label for item in items if not item.exists]
        if missing:
            summary = f"Model runtime configuration is incomplete: missing {', '.join(missing)}."
            return RuntimeChecklistResult(valid=False, model_name=model_name, summary=summary, items=items)

        return RuntimeChecklistResult(
            valid=True,
            model_name=model_name,
            summary="Model runtime configuration check passed.",
            items=items,
        )

    def build_framework_runtime_checklist(
        self,
        model_manifest: ConfigurationProvider,
        framework_uuid: str,
    ) -> RuntimeChecklistResult:
        """Build the framework runtime checklist required by this runtime checklist service."""
        cfg = model_manifest.get_configuration()
        model_name = str(getattr(cfg, "model_name", "") or "") or None
        model_folder = str(getattr(cfg, "model_folder_path", "") or "")
        normalized_framework_uuid = str(framework_uuid or "").strip().lower()
        framework_cfg_path = get_framework_model_configuration_path(model_folder, normalized_framework_uuid)
        def _item(label: str, value: str, exists: bool, kind: str = "attr") -> ChecklistItem:
            """Perform the item step within the enclosing build framework runtime checklist
            workflow.
            """
            return ChecklistItem(label=label, path=value, kind=kind, exists=exists)

        items: list[ChecklistItem] = [
            _item(
                "Framework config file",
                framework_cfg_path,
                os.path.isfile(framework_cfg_path),
                kind="file",
            ),
        ]

        if not os.path.isfile(framework_cfg_path):
            summary = f"Framework runtime configuration is incomplete for {normalized_framework_uuid}: missing framework config file."
            return RuntimeChecklistResult(valid=False, model_name=model_name, summary=summary, items=items)

        try:
            payload = validate_versioned_metadata_payload(
                MetadataKind.FRAMEWORK_CONFIGURATION,
                read_yaml_mapping_file(framework_cfg_path),
            )
            framework_cfg = Configuration(config_dict=payload)
            framework_cfg.set_config_file_path(framework_cfg_path)
        except Exception:
            summary = f"Framework runtime configuration is unreadable for {normalized_framework_uuid}."
            return RuntimeChecklistResult(valid=False, model_name=model_name, summary=summary, items=items)

        items.extend([
            _item(
                "Framework uuid",
                str(getattr(getattr(framework_cfg, "rl_engine_framework", None), "uuid", "") or ""),
                bool(str(getattr(getattr(framework_cfg, "rl_engine_framework", None), "uuid", "") or "").strip()),
            ),
            _item(
                "Model implementation module",
                str(getattr(getattr(framework_cfg, "model_implementation", None), "module", "") or ""),
                bool(str(getattr(getattr(framework_cfg, "model_implementation", None), "module", "") or "").strip()),
            ),
            _item(
                "Model implementation object",
                str(getattr(getattr(framework_cfg, "model_implementation", None), "object", "") or ""),
                bool(str(getattr(getattr(framework_cfg, "model_implementation", None), "object", "") or "").strip()),
            ),
        ])

        missing = [item.label for item in items if not item.exists]
        if missing:
            summary = f"Framework runtime configuration is incomplete for {normalized_framework_uuid}: missing {', '.join(missing)}."
            return RuntimeChecklistResult(valid=False, model_name=model_name, summary=summary, items=items)

        return RuntimeChecklistResult(
            valid=True,
            model_name=model_name,
            summary=f"Framework runtime configuration check passed for {normalized_framework_uuid}.",
            items=items,
        )
