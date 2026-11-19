# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate presentation context."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Optional

from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator


@dataclass
class DetailPayload:
    """Carry the data required for detail payload."""

    model_name: str
    model_uuid: str
    model_version: str
    model_description: str
    framework_uuid: str
    current_session_uuid: Optional[str]
    current_session_display_name: Optional[str]
    current_session_name: str
    current_session_description: str


@dataclass
class ContextPayload:
    """Carry the data required for context payload."""

    model_name: str
    model_uuid: str
    model_version: str
    model_description: str
    model_created_at_raw: Optional[str]
    model_modified_at_raw: Optional[str]
    parent_model_uuid: Optional[str]
    parent_model_display_name: str
    child_model_uuids: list[str]
    framework_uuid: str
    current_session_uuid: Optional[str]
    current_session_identity_uuid: Optional[str]
    current_session_display_name: Optional[str]
    current_session_name: str
    current_session_description: str
    current_session_parent_display_name: str
    current_session_child_count: int


class PresentationContextService:
    """Builds GUI-facing model/session detail payloads."""

    def __init__(
        self,
        *,
        active_configuration_provider: Callable[[], object | None],
        current_framework_provider: Callable[[], Optional[str]],
        session_configuration_loader: Callable[[str], object | None],
        session_reference_display_provider: Callable[[Optional[str]], str],
        session_identity_uuid_provider: Callable[[Optional[str]], Optional[str]],
        session_display_name_without_uuid_provider: Callable[[Optional[str]], Optional[str]],
        session_parent_uuid_provider: Callable[[Optional[str]], Optional[str]],
        session_child_uuids_provider: Callable[[Optional[str]], list[str]],
        model_lineage_index_provider: Callable[[], tuple[dict[str, object], dict[Optional[str], list[str]]]],
        model_candidates_provider: Callable[[], tuple[Sequence[Any], Sequence[Any]]],
    ) -> None:
        """Initialize the presentation context service and its owned state."""
        self._active_configuration_provider = active_configuration_provider
        self._current_framework_provider = current_framework_provider
        self._session_configuration_loader = session_configuration_loader
        self._session_reference_display_provider = session_reference_display_provider
        self._session_identity_uuid_provider = session_identity_uuid_provider
        self._session_display_name_without_uuid_provider = session_display_name_without_uuid_provider
        self._session_parent_uuid_provider = session_parent_uuid_provider
        self._session_child_uuids_provider = session_child_uuids_provider
        self._model_lineage_index_provider = model_lineage_index_provider
        self._model_candidates_provider = model_candidates_provider

    @staticmethod
    def _session_name_and_description(session_configuration: object | None) -> tuple[str, str]:
        """Return the session name and description value owned by this presentation context
        service.
        """
        if session_configuration is None:
            return "", ""

        session_name = (
            str(getattr(session_configuration, "display_name", "") or "").strip()
            or str(getattr(session_configuration, "session_name", "") or "").strip()
        )
        session_description = str(getattr(session_configuration, "description", "") or "").strip()
        return session_name, session_description

    def get_current_model_display_name(self) -> Optional[str]:
        """Return the current model display name exposed by this presentation context service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            return None
        model_name = str(getattr(configuration, "model_name", "") or "").strip()
        model_version = str(getattr(configuration, "model_version", "") or "").strip()
        return EmptyModelGenerator.build_model_display_name(model_name, model_version) or None

    def get_model_reference_display(self, model_uuid: Optional[str]) -> str:
        """Return the model reference display exposed by this presentation context service."""
        if not model_uuid:
            return "-"

        configuration = self._active_configuration_provider()
        if configuration is not None:
            current_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
            if current_uuid and current_uuid == str(model_uuid).strip():
                return self.get_current_model_display_name() or str(model_uuid)

        valid_entries, _ = self._model_candidates_provider()
        for entry in valid_entries:
            cfg = entry.manifest.get_configuration()
            if str(getattr(cfg, "model_uuid", "") or "").strip() != str(model_uuid).strip():
                continue
            display_name = str(getattr(cfg, "model_name", "") or entry.display_name or "").strip()
            model_version = str(getattr(cfg, "model_version", "") or entry.effective_version or "").strip()
            return EmptyModelGenerator.build_model_display_name(display_name, model_version) or display_name or str(model_uuid)

        return str(model_uuid)

    def build_model_detail_payload(self, session_uuid: Optional[str]) -> Optional[DetailPayload]:
        """Build the model detail payload required by this presentation context service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            return None

        session_name = ""
        session_description = ""
        if session_uuid:
            session_configuration = self._session_configuration_loader(session_uuid)
            session_name, session_description = self._session_name_and_description(session_configuration)

        return DetailPayload(
            model_name=str(getattr(configuration, "model_name", "") or ""),
            model_uuid=str(getattr(configuration, "model_uuid", "") or ""),
            model_version=str(getattr(configuration, "model_version", "") or ""),
            model_description=str(getattr(configuration, "description", "") or ""),
            framework_uuid=self._current_framework_provider() or "",
            current_session_uuid=session_uuid,
            current_session_display_name=self._session_reference_display_provider(session_uuid) if session_uuid else None,
            current_session_name=session_name,
            current_session_description=session_description,
        )

    def build_model_context_payload(self, session_uuid: Optional[str]) -> Optional[ContextPayload]:
        """Build the model context payload required by this presentation context service."""
        configuration = self._active_configuration_provider()
        if configuration is None:
            return None

        model_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        parent_model_uuid = str(getattr(configuration, "parent_model_uuid", "") or "").strip() or None

        parent_model_display_name = "-"
        child_model_uuids: list[str] = []
        if model_uuid:
            _, lineage_children_by_parent = self._model_lineage_index_provider()
            child_model_uuids = list(lineage_children_by_parent.get(model_uuid, []))
        if parent_model_uuid:
            parent_model_display_name = self.get_model_reference_display(parent_model_uuid)

        session_name = ""
        session_description = ""
        session_parent_display_name = "-"
        session_child_count = 0
        if session_uuid:
            session_configuration = self._session_configuration_loader(session_uuid)
            session_name, session_description = self._session_name_and_description(session_configuration)
            session_parent_uuid = self._session_parent_uuid_provider(session_uuid)
            session_parent_display_name = (
                self._session_reference_display_provider(session_parent_uuid) if session_parent_uuid else "-"
            )
            session_child_count = len(self._session_child_uuids_provider(session_uuid))

        return ContextPayload(
            model_name=str(getattr(configuration, "model_name", "") or "").strip(),
            model_uuid=model_uuid,
            model_version=str(getattr(configuration, "model_version", "") or "").strip(),
            model_description=str(getattr(configuration, "description", "") or "").strip(),
            model_created_at_raw=str(getattr(configuration, "created_at", "") or "").strip() or None,
            model_modified_at_raw=str(getattr(configuration, "modified_at", "") or "").strip() or None,
            parent_model_uuid=parent_model_uuid,
            parent_model_display_name=parent_model_display_name,
            child_model_uuids=child_model_uuids,
            framework_uuid=self._current_framework_provider() or "",
            current_session_uuid=session_uuid,
            current_session_identity_uuid=self._session_identity_uuid_provider(session_uuid) if session_uuid else None,
            current_session_display_name=(
                self._session_display_name_without_uuid_provider(session_uuid) if session_uuid else None
            ),
            current_session_name=session_name,
            current_session_description=session_description,
            current_session_parent_display_name=session_parent_display_name,
            current_session_child_count=session_child_count,
        )
