# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Runtime module import helpers with detailed failure diagnostics."""

from __future__ import annotations

import importlib
import importlib.util
import sys
import traceback
from types import ModuleType


def resolve_runtime_module_origin(module_path: str) -> str:
    """Resolve a runtime module without executing it."""

    normalized_module_path = str(module_path or "").strip()
    if not normalized_module_path:
        raise ImportError("Runtime module path is empty.")
    try:
        spec = importlib.util.find_spec(normalized_module_path)
    except Exception as exc:
        raise ImportError(
            f"Runtime importability check failed for '{normalized_module_path}'."
        ) from exc
    if spec is None:
        raise ImportError(
            f"Runtime module '{normalized_module_path}' is not importable: no module spec found."
        )
    origin = str(getattr(spec, "origin", "") or "").strip()
    if not origin or origin in {"built-in", "frozen"}:
        raise ImportError(
            f"Runtime module '{normalized_module_path}' does not resolve to a source-file origin."
        )
    return origin


def _remove_failed_runtime_modules(module_path: str, modules_before: set[str]) -> None:
    """Remove the failed runtime modules owned by this runtime import diagnostics module."""
    prefix = f"{module_path}."
    for cached_name in tuple(sys.modules):
        if cached_name in modules_before:
            continue
        if cached_name == module_path or cached_name.startswith(prefix):
            sys.modules.pop(cached_name, None)


def import_runtime_module_with_diagnostics(module_path: str) -> ModuleType:
    """Return the import runtime module with diagnostics value owned by this runtime import
    diagnostics module.
    """
    normalized_module_path = str(module_path or "").strip()
    if not normalized_module_path:
        raise ImportError("Runtime module path is empty.")

    try:
        origin = resolve_runtime_module_origin(normalized_module_path)
    except ImportError as exc:
        traceback_text = traceback.format_exc().strip()
        raise ImportError(f"{exc}\n{traceback_text}") from exc

    modules_before = set(sys.modules)
    try:
        return importlib.import_module(normalized_module_path)
    except Exception as exc:
        _remove_failed_runtime_modules(normalized_module_path, modules_before)
        traceback_text = traceback.format_exc().strip()
        raise ImportError(
            f"Runtime import failed for '{normalized_module_path}'. "
            f"Importability check passed (origin: {origin}), but the module raised "
            f"{type(exc).__name__}: {exc}\n{traceback_text}"
        ) from exc
