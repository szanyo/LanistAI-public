# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable application contracts for model creation workflows."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelCreationDraftResolution:
    """Normalized draft fields and their validation result."""

    valid: bool
    error_message: str | None
    resolved_model_name: str
    resolved_model_version: str
    target_model_folder_path: str


@dataclass(frozen=True, slots=True)
class FreshModelCreationRequest:
    """Inputs required to generate and publish one fresh model package."""

    model_name: str
    model_version: str
    model_description: str
    framework_uuid: str


@dataclass(frozen=True, slots=True)
class ModelCloneRequest:
    """Inputs required to clone selected resources into a new model package."""

    source_model_folder_path: str
    target_model_name: str
    target_model_version: str
    initial_session_description: str
    selected_relative_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ModelCreationOutcome:
    """Stable success data shared by fresh and cloned model creation."""

    target_model_name: str
    target_model_folder_path: str
    auto_load: bool
