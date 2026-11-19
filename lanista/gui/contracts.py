# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared, framework-independent contracts for GUI adapters and workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

UiSeverity = Literal["information", "warning", "error"]


def normalize_ui_severity(value: str) -> UiSeverity:
    """Map adapter-provided severity text onto the stable UI contract."""

    if value == "information":
        return "information"
    if value == "error":
        return "error"
    return "warning"


@dataclass(frozen=True)
class UiNotification:
    """A user-safe notification produced by an application workflow."""

    message: str
    title: str
    severity: UiSeverity = "information"
