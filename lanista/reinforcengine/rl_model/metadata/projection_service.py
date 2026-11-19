# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Capture, update, and restore the active in-memory metadata projection."""

from __future__ import annotations

from typing import Protocol, cast

from lanista.reinforcengine.rl_model.metadata.edit_contracts import MetadataEditOutcome

ProjectionSnapshot = tuple[tuple[object, str, bool, object], ...]


class _MutableModelProjection(Protocol):
    """Describe the mutable model fields synchronized after a metadata edit."""

    model_name: str
    model_version: str
    description: str


def capture_projection(configuration: object) -> ProjectionSnapshot:
    """Capture only fields that a successful metadata edit may update."""

    targets: list[tuple[object, str]] = [
        (configuration, "model_name"),
        (configuration, "model_version"),
        (configuration, "description"),
    ]
    model_section = getattr(configuration, "model", None)
    if isinstance(model_section, dict):
        return tuple(
            (
                target,
                attribute,
                hasattr(target, attribute),
                getattr(target, attribute, None),
            )
            for target, attribute in targets
        ) + ((model_section, "__mapping__", True, dict(model_section)),)
    if model_section is not None:
        targets.extend(
            (model_section, attribute)
            for attribute in ("name", "version", "description")
        )
    return tuple(
        (
            target,
            attribute,
            hasattr(target, attribute),
            getattr(target, attribute, None),
        )
        for target, attribute in targets
    )


def restore_projection(snapshot: ProjectionSnapshot) -> None:
    """Restore a projection snapshot after an in-memory update failure."""

    for target, attribute, existed, value in reversed(snapshot):
        if attribute == "__mapping__":
            mapping = target
            if not isinstance(mapping, dict) or not isinstance(value, dict):
                raise TypeError("Invalid model projection snapshot.")
            mapping.clear()
            mapping.update(value)
        elif existed:
            setattr(target, attribute, value)
        elif hasattr(target, attribute):
            delattr(target, attribute)


def synchronize_model_projection(
    configuration: object,
    outcome: MetadataEditOutcome,
) -> None:
    """Apply canonical committed model fields to the active configuration."""

    projection = cast(_MutableModelProjection, configuration)
    projection.model_name = outcome.model_name
    projection.model_version = outcome.model_version
    projection.description = outcome.model_description
    model_section = getattr(configuration, "model", None)
    if isinstance(model_section, dict):
        model_section.update(
            name=outcome.model_name,
            version=outcome.model_version,
            description=outcome.model_description,
        )
    elif model_section is not None:
        model_section.name = outcome.model_name
        model_section.version = outcome.model_version
        model_section.description = outcome.model_description


__all__ = [
    "ProjectionSnapshot",
    "capture_projection",
    "restore_projection",
    "synchronize_model_projection",
]
