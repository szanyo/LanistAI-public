# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Persist and restore reinforcement-learning model data."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from contextlib import suppress
from copy import deepcopy
from typing import Any

import yaml


def atomic_write_yaml_file(
    path: str,
    payload: Any,
    *,
    sort_keys: bool = False,
    allow_unicode: bool = False,
    verify_reader: Callable[[str], Any] | None = None,
    smoke_check_reader: Callable[[str], Any] | None = None,
) -> None:
    """Write YAML through a same-directory temporary file and atomic replace."""
    target_path = os.path.abspath(path)
    target_dir = os.path.dirname(target_path)
    if not target_dir:
        raise ValueError(f"Cannot resolve parent directory for YAML path: {path!r}")

    os.makedirs(target_dir, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(target_path)}.",
        suffix=".tmp",
        dir=target_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(
                payload,
                handle,
                sort_keys=sort_keys,
                allow_unicode=allow_unicode,
            )
            handle.flush()
            os.fsync(handle.fileno())

        if verify_reader is not None:
            verify_reader(temp_path)

        os.replace(temp_path, target_path)
        if smoke_check_reader is not None:
            smoke_check_reader(target_path)
    except Exception:
        with suppress(FileNotFoundError):
            os.unlink(temp_path)
        raise


def atomic_write_validated_yaml_file(
    path: str,
    payload: Any,
    *,
    validator: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    allow_unicode: bool = False,
    verify_reader: Callable[[str], Any] | None = None,
    smoke_check_reader: Callable[[str], Any] | None = None,
) -> None:
    """Validate a YAML payload first, then persist it through the atomic writer."""
    validated_payload = validator(payload) if validator is not None else payload
    atomic_write_yaml_file(
        path,
        validated_payload,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
        verify_reader=verify_reader,
        smoke_check_reader=smoke_check_reader,
    )


def atomic_write_configuration_file(
    configuration: Any,
    path: str | None = None,
    *,
    verify_reader: Callable[[str], Any] | None = None,
    smoke_check_reader: Callable[[str], Any] | None = None,
) -> None:
    """Handle the atomic write configuration file operation owned by this persistence module."""
    target_path = str(path or getattr(configuration, "get_config_file_path", lambda: "")() or "").strip()
    if not target_path:
        target_path = str(getattr(configuration, "_config_file_path", "") or "").strip()
    if not target_path:
        raise ValueError("Cannot resolve configuration file path.")

    payload = deepcopy(getattr(configuration, "_config_dict", None))
    if not isinstance(payload, dict):
        raise ValueError("Configuration payload is not a mapping.")

    atomic_write_yaml_file(
        target_path,
        payload,
        verify_reader=verify_reader or read_yaml_mapping_file,
        smoke_check_reader=smoke_check_reader or read_yaml_mapping_file,
    )


def read_yaml_mapping_file(path: str) -> dict[str, Any]:
    """Read the yaml mapping file without changing this persistence module's state."""
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML payload is not a mapping: {path}")
    return payload
