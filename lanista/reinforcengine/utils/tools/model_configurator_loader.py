# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Load model-local algorithm configurators from validated origins."""

import hashlib
import importlib
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

from lanista.foundation.configuration import AppConfiguration
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import (
    ModelPluginNamespace,
    bind_model_plugin_root,
)

APP_CONFIGURATION = AppConfiguration()
LOGGER = logging.getLogger(__name__)


class ModelManifestPort(Protocol):
    """Manifest boundary required by configurator resolution."""

    def get_configuration(self) -> object:
        """Return the active model configuration."""


class ConfiguratorLoggerPort(Protocol):
    """Diagnostic operations used while resolving configurator sources."""

    def debug(self, message: str) -> None:
        """Record a resolution detail."""

    def error(self, message: str) -> None:
        """Record a terminal resolution failure."""


@dataclass(frozen=True, slots=True)
class _ConfiguratorResolutionPlan:
    """Describe the executable plan for configurator resolution."""

    trusted_root: str
    local_file_path: str | None
    module_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _CandidateImportResult:
    """Report the result of candidate import."""

    loaded_module: tuple[str, ModuleType] | None
    last_error: Exception | None


def _resolved_descendant(path: str, root: str) -> str:
    """Return the resolved descendant value owned by this model configurator loader module."""
    resolved_path = os.path.realpath(os.path.abspath(path))
    resolved_root = os.path.realpath(os.path.abspath(root))
    try:
        contained = os.path.commonpath((resolved_path, resolved_root)) == resolved_root
    except ValueError:
        contained = False
    if not contained or resolved_path == resolved_root:
        raise ImportError("Algorithm configurator origin is outside its model resource root.")
    return resolved_path


def _cleanup_failed_module(module_name: str, modules_before: set[str]) -> None:
    """Clean up the failed module owned by this model configurator loader module."""
    prefix = f"{module_name}."
    for cached_name in tuple(sys.modules):
        if cached_name in modules_before:
            continue
        if cached_name == module_name or cached_name.startswith(prefix):
            sys.modules.pop(cached_name, None)


def _package_module_name_from_file_path(normalized_path: str) -> str | None:
    """Return the package module name from file path value owned by this model configurator loader
    module.
    """
    path = Path(normalized_path)
    if path.suffix != ".py":
        return None
    path_parts = path.parts
    package_index = next(
        (
            index
            for index in range(len(path_parts) - 1)
            if path_parts[index : index + 2] == ("rl_model_storage", "models")
        ),
        None,
    )
    if package_index is None or len(path_parts) <= package_index + 5:
        return None
    if path_parts[package_index + 3] != "resources":
        return None

    namespace = ModelPluginNamespace.create(
        model_package=path_parts[package_index + 2],
        framework_uuid=path_parts[package_index + 4],
    )
    module_segments = list(path_parts[package_index + 5 :])
    module_segments[-1] = path.stem
    if module_segments[-1] == "__init__":
        module_segments.pop()
        if not module_segments:
            return namespace.package
    return namespace.module(*module_segments)


def _load_module_from_file(module_file_path: str, *, trusted_root: str) -> tuple[str, ModuleType]:
    """Load the module from file owned by this model configurator loader module."""
    normalized_path = _resolved_descendant(module_file_path, trusted_root)
    package_module_name = _package_module_name_from_file_path(normalized_path)
    if package_module_name:
        path_parts = Path(normalized_path).parts
        package_index = next(
            (
                index
                for index in range(len(path_parts) - 1)
                if path_parts[index : index + 2] == ("rl_model_storage", "models")
            ),
            None,
        )
        if package_index is not None:
            bind_model_plugin_root(Path(*path_parts[: package_index + 2]))
        try:
            spec = importlib.util.find_spec(package_module_name)
        except (ImportError, AttributeError, ValueError) as exc:
            LOGGER.debug(
                "Package lookup failed; loading the validated configurator file directly.",
                exc_info=exc,
            )
        else:
            spec_origin = str(getattr(spec, "origin", "") or "") if spec is not None else ""
            if os.path.realpath(spec_origin) == normalized_path:
                modules_before = set(sys.modules)
                try:
                    return package_module_name, importlib.import_module(package_module_name)
                except Exception:
                    # Initialization failures are authoritative. Retrying the
                    # same module under a synthetic name could duplicate
                    # module-level side effects and disguise the root cause.
                    _cleanup_failed_module(package_module_name, modules_before)
                    raise
            LOGGER.debug(
                "Package origin differs from the validated configurator path; "
                "loading the selected file directly."
            )

    unique_hash = hashlib.sha256(normalized_path.encode("utf-8")).hexdigest()[:32]
    module_name = f"lanistai_dynamic_algorithm_configurator_{unique_hash}"

    if module_name in sys.modules:
        cached_module = sys.modules[module_name]
        cached_origin = os.path.realpath(str(getattr(cached_module, "__file__", "") or ""))
        if cached_origin == normalized_path:
            return module_name, cached_module
        sys.modules.pop(module_name, None)

    spec = importlib.util.spec_from_file_location(module_name, normalized_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec from file: {normalized_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise

    return module_name, module


def _import_module_from_root(module_name: str, *, trusted_root: str) -> ModuleType:
    """Return the import module from root value owned by this model configurator loader module."""
    spec = importlib.util.find_spec(module_name)
    origin = str(getattr(spec, "origin", "") or "") if spec is not None else ""
    if not origin:
        raise ImportError("Algorithm configurator module has no source-file origin.")
    _resolved_descendant(origin, trusted_root)
    modules_before = set(sys.modules)
    try:
        return importlib.import_module(module_name)
    except Exception:
        _cleanup_failed_module(module_name, modules_before)
        raise


def _runtime_framework_uuid(model_configuration: object) -> str:
    """Return the runtime framework uuid value owned by this model configurator loader module."""
    framework_uuid = (
        str(getattr(model_configuration, "active_runtime_framework_uuid", "") or "").strip().lower()
    )
    if framework_uuid:
        return framework_uuid

    framework_uuid = (
        str(getattr(getattr(model_configuration, "rl_engine_framework", None), "uuid", "") or "")
        .strip()
        .lower()
    )
    if framework_uuid:
        return framework_uuid

    return str(getattr(model_configuration, "preferred_framework_uuid", "") or "").strip().lower()


def _default_module_path_for_framework(model_configuration: object) -> str | None:
    """Return the default module path for framework value owned by this model configurator loader
    module.
    """
    framework_uuid = _runtime_framework_uuid(model_configuration)
    if not framework_uuid:
        return None

    model_folder_path = str(getattr(model_configuration, "model_folder_path", "") or "").strip()
    model_folder_name = (
        os.path.basename(model_folder_path.rstrip(os.sep)) if model_folder_path else ""
    )
    model_name = str(getattr(model_configuration, "model_name", "") or "").strip()
    target_model_name = model_folder_name or model_name
    if not target_model_name:
        return None

    return ModelPluginNamespace.create(
        model_package=target_model_name,
        framework_uuid=framework_uuid,
    ).module("algorithm_configurator")


def _framework_resource_directory(model_configuration: object, framework_uuid: str) -> str:
    """Return the framework resource directory value owned by this model configurator loader
    module.
    """
    raw_paths = getattr(model_configuration, "framework_resource_paths", None)
    if isinstance(raw_paths, dict):
        return str(raw_paths.get(framework_uuid, "") or "")
    return str(getattr(raw_paths, framework_uuid, "") or "")


def _configurator_root(model_configuration: object, framework_uuid: str) -> str:
    """Return the configurator root value owned by this model configurator loader module."""
    model_folder_path = str(getattr(model_configuration, "model_folder_path", "") or "")
    configurator_root = _framework_resource_directory(model_configuration, framework_uuid)
    configurator_root = configurator_root or model_folder_path
    if configurator_root and model_folder_path and not os.path.isabs(configurator_root):
        configurator_root = os.path.join(model_folder_path, configurator_root)
    if (
        model_folder_path
        and configurator_root
        and os.path.realpath(configurator_root) != os.path.realpath(model_folder_path)
    ):
        _resolved_descendant(configurator_root, model_folder_path)
    return configurator_root


def _module_candidates(model_configuration: object) -> tuple[str, ...]:
    """Return the module candidates value owned by this model configurator loader module."""
    candidates: list[str] = []
    default_path = _default_module_path_for_framework(model_configuration)
    algorithm_configuration = getattr(model_configuration, "algorithm", None)
    configured_path = getattr(algorithm_configuration, "configurator_module", None)
    if configured_path:
        if default_path is None:
            return ()
        framework_uuid = _runtime_framework_uuid(model_configuration)
        model_folder_path = str(getattr(model_configuration, "model_folder_path", "") or "").strip()
        model_folder_name = (
            os.path.basename(model_folder_path.rstrip(os.sep)) if model_folder_path else ""
        )
        model_name = str(getattr(model_configuration, "model_name", "") or "").strip()
        namespace = ModelPluginNamespace.create(
            model_package=model_folder_name or model_name,
            framework_uuid=framework_uuid,
        )
        candidates.append(namespace.require_module_path(configured_path))

    if default_path and default_path not in candidates:
        candidates.append(default_path)
    return tuple(candidates)


def _build_resolution_plan(model_configuration: object) -> _ConfiguratorResolutionPlan:
    """Build the resolution plan required by this model configurator loader module."""
    framework_uuid = _runtime_framework_uuid(model_configuration)
    trusted_root = _configurator_root(model_configuration, framework_uuid)
    local_file_path = (
        os.path.join(trusted_root, "algorithm_configurator.py") if trusted_root else None
    )
    if local_file_path is not None and not os.path.isfile(local_file_path):
        local_file_path = None
    return _ConfiguratorResolutionPlan(
        trusted_root=trusted_root,
        local_file_path=local_file_path,
        module_candidates=_module_candidates(model_configuration),
    )


def _load_module_candidate(
    plan: _ConfiguratorResolutionPlan,
    logger: ConfiguratorLoggerPort | None,
) -> _CandidateImportResult:
    """Load the module candidate owned by this model configurator loader module."""
    last_error: Exception | None = None
    for candidate in plan.module_candidates:
        try:
            if logger:
                logger.debug(f"Loading algorithm configurator by module path: {candidate}")
            loaded_module = _import_module_from_root(candidate, trusted_root=plan.trusted_root)
        except Exception as exc:
            last_error = exc
        else:
            return _CandidateImportResult(
                loaded_module=(candidate, loaded_module),
                last_error=None,
            )
    return _CandidateImportResult(loaded_module=None, last_error=last_error)


def load_algorithm_configurator_module(
    rl_model_manifest: ModelManifestPort,
    logger: ConfiguratorLoggerPort | None = None,
) -> tuple[str, ModuleType]:
    """Load algorithm configurator module from model folder or fallback import path.

    Returns:
        tuple[str, ModuleType]: (module_reference_name, imported_module)
    """

    model_configuration = rl_model_manifest.get_configuration()
    plan = _build_resolution_plan(model_configuration)
    if plan.local_file_path is not None:
        if logger:
            logger.debug(f"Loading algorithm configurator from file: {plan.local_file_path}")
        return _load_module_from_file(
            plan.local_file_path,
            trusted_root=plan.trusted_root,
        )

    candidate_result = _load_module_candidate(plan, logger)
    if candidate_result.loaded_module is not None:
        return candidate_result.loaded_module

    msg = (
        "Unable to load algorithm configurator module. "
        "Expected either a local 'algorithm_configurator.py' file in the model folder, "
        "or a valid importable module path in configuration."
    )
    if logger:
        logger.error(msg)

    if candidate_result.last_error is not None:
        raise ImportError(msg) from candidate_result.last_error
    raise ImportError(msg)
