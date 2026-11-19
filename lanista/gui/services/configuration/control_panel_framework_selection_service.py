# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate control panel framework selection."""

from __future__ import annotations

from collections.abc import Iterable


class ControlPanelFrameworkSelectionService:
    """Decision helpers for ControlPanel framework selection state."""

    _DEFAULT_OPTIONS: tuple[tuple[str, str], ...] = (
        ("🧠 Ray RLlib", "ray"),
        ("🧬 Stable Baselines 3", "sb3"),
    )

    @classmethod
    def build_framework_options(cls, frameworks: object) -> list[tuple[str, str]]:
        """Build the framework options required by this control panel framework selection service.
        """
        framework_options: list[tuple[str, str]] = []
        items = getattr(frameworks, "items", None)
        source = items() if callable(items) else ()
        for framework_uuid, framework_data in source:
            framework_name = str(getattr(framework_data, "name", framework_uuid))
            framework_symbol = str(getattr(framework_data, "symbol", "") or "")
            option_label = f"{framework_symbol} {framework_name}" if framework_symbol else framework_name
            framework_options.append((option_label, str(framework_uuid)))

        if not framework_options:
            framework_options = list(cls._DEFAULT_OPTIONS)
        return framework_options

    @staticmethod
    def _valid_framework_values(framework_options: Iterable[tuple[str, str]]) -> set[str]:
        """Return the valid framework values value owned by this control panel framework selection
        service.
        """
        return {value for _, value in framework_options}

    @classmethod
    def normalize_requested_framework_uuid(
        cls,
        *,
        requested_framework_uuid: str | None,
        framework_options: Iterable[tuple[str, str]],
        fallback_framework_uuid: str,
    ) -> str:
        """Normalize the requested framework uuid accepted by this control panel framework
        selection service.
        """
        requested = str(requested_framework_uuid or "").strip().lower()
        valid_framework_values = cls._valid_framework_values(framework_options)
        if requested in valid_framework_values:
            return requested
        return fallback_framework_uuid

    @classmethod
    def resolve_selected_framework_uuid(
        cls,
        *,
        model_framework_uuid: str | None,
        selected_framework_uuid: str | None,
        framework_options: Iterable[tuple[str, str]],
    ) -> str:
        """Resolve the selected framework uuid required by this control panel framework selection
        service.
        """
        fallback_framework_uuid = str(selected_framework_uuid or "").strip().lower()
        if not fallback_framework_uuid:
            options = list(framework_options)
            fallback_framework_uuid = options[0][1] if options else "ray"
        requested_framework_uuid = str(model_framework_uuid or fallback_framework_uuid or "").strip().lower()
        valid_framework_values = cls._valid_framework_values(framework_options)
        if requested_framework_uuid in valid_framework_values:
            return requested_framework_uuid
        return fallback_framework_uuid
