# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validated canonical module names for model-local plugins."""

from __future__ import annotations

import importlib
import os
import re
import sys
from dataclasses import dataclass
from threading import RLock

from lanista.reinforcengine.application_errors import ApplicationValidationError

MODEL_PLUGIN_NAMESPACE_ROOT = "lanista.rl_model_storage.models"
_MODULE_SEGMENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_NAMESPACE_BINDING_LOCK = RLock()
_BOUND_MODELS_ROOT: str | None = None


def _is_descendant(path: str, root: str) -> bool:
    """Return whether a resolved path belongs to the selected models root."""

    try:
        return os.path.commonpath((os.path.realpath(path), root)) == root
    except (TypeError, ValueError):
        return False


def _reject_cached_cross_root_plugins(models_root: str) -> None:
    """Reject rebinding while model plugin modules from another root are cached."""

    prefix = f"{MODEL_PLUGIN_NAMESPACE_ROOT}."
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith(prefix):
            continue
        module_file = str(getattr(module, "__file__", "") or "")
        if module_file and not _is_descendant(module_file, models_root):
            raise ApplicationValidationError(
                "Model plugin namespace cannot change while another model root is imported."
            )


def bind_model_plugin_root(models_root: str | os.PathLike[str]) -> None:
    """Prepend one validated models root to the canonical package search path."""

    global _BOUND_MODELS_ROOT

    raw_root = os.fspath(models_root)
    if not os.path.isabs(raw_root):
        raise ApplicationValidationError("Model plugin root must be an absolute path.")
    resolved_root = os.path.realpath(raw_root)
    root_path = os.path.normpath(resolved_root)
    if os.path.basename(root_path) != "models" or os.path.basename(
        os.path.dirname(root_path)
    ) != "rl_model_storage":
        raise ApplicationValidationError(
            "Model plugin root must end with 'rl_model_storage/models'."
        )

    with _NAMESPACE_BINDING_LOCK:
        if root_path != _BOUND_MODELS_ROOT:
            _reject_cached_cross_root_plugins(root_path)

        package = importlib.import_module(MODEL_PLUGIN_NAMESPACE_ROOT)
        package_file = str(getattr(package, "__file__", "") or "")
        packaged_root = (
            os.path.realpath(os.path.dirname(package_file)) if package_file else ""
        )
        existing_paths = [
            os.path.realpath(os.fspath(path))
            for path in getattr(package, "__path__", ())
        ]
        retained_paths = [
            path
            for path in existing_paths
            if path != root_path
            and (
                path != _BOUND_MODELS_ROOT
                or path == packaged_root
            )
        ]
        search_paths = [root_path, *retained_paths]
        package.__path__ = search_paths
        package_spec = getattr(package, "__spec__", None)
        if package_spec is not None:
            package_spec.submodule_search_locations = search_paths
        _BOUND_MODELS_ROOT = root_path
        importlib.invalidate_caches()


def _validated_module_segment(value: object, *, label: str) -> str:
    """Return one lowercase Python package segment or fail closed."""

    segment = str(value or "").strip()
    if not _MODULE_SEGMENT_PATTERN.fullmatch(segment):
        raise ApplicationValidationError(f"{label} must be one lowercase Python package segment.")
    return segment


@dataclass(frozen=True, slots=True)
class ModelPluginNamespace:
    """Canonical namespace for one model/framework plugin package."""

    model_package: str
    framework_uuid: str

    @classmethod
    def create(
        cls,
        *,
        model_package: object,
        framework_uuid: object,
    ) -> ModelPluginNamespace:
        """Validate the two storage-derived namespace components."""

        return cls(
            model_package=_validated_module_segment(
                model_package,
                label="Model package name",
            ),
            framework_uuid=_validated_module_segment(
                str(framework_uuid or "").strip().lower(),
                label="Framework identifier",
            ),
        )

    @property
    def package(self) -> str:
        """Return the model/framework package prefix."""

        return f"{MODEL_PLUGIN_NAMESPACE_ROOT}.{self.model_package}.resources.{self.framework_uuid}"

    def module(self, *module_segments: object) -> str:
        """Build a canonical module name from validated descendant segments."""

        if not module_segments:
            raise ApplicationValidationError(
                "Plugin module path must identify a module below its framework package."
            )
        validated_segments = tuple(
            _validated_module_segment(segment, label="Plugin module segment")
            for segment in module_segments
        )
        return ".".join((self.package, *validated_segments))

    def require_module_path(self, module_path: object) -> str:
        """Validate that a configured module belongs to this exact namespace."""

        normalized_module = str(module_path or "").strip()
        parts = normalized_module.split(".")
        if any(not _MODULE_SEGMENT_PATTERN.fullmatch(part) for part in parts):
            raise ApplicationValidationError(
                "Plugin module path contains an invalid package segment."
            )
        prefix_parts = self.package.split(".")
        if parts[: len(prefix_parts)] != prefix_parts or len(parts) <= len(prefix_parts):
            raise ApplicationValidationError(
                "Plugin module must belong to the selected model framework package."
            )
        return normalized_module
