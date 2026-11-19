# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Validated runtime-plugin identity and filesystem-origin contract."""

from __future__ import annotations

import os
from dataclasses import dataclass

from lanista.reinforcengine.application_errors import ApplicationValidationError
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import ModelPluginNamespace
from lanista.reinforcengine.rl_model.runtime.requests import RuntimeInitializationRequest
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    EnvironmentFactory,
    FrameworkRuntimeFactory,
)


def _is_strict_descendant(path: str, root: str) -> bool:
    """Return whether strict descendant applies to this runtime plugin module."""
    resolved_path = os.path.realpath(path)
    resolved_root = os.path.realpath(root)
    try:
        return (
            os.path.commonpath((resolved_path, resolved_root)) == resolved_root
            and resolved_path != resolved_root
        )
    except ValueError:
        return False


@dataclass(frozen=True)
class RuntimePluginSpec:
    """A plugin resolved to a trusted model-local source file."""

    framework_uuid: str
    module_path: str
    class_name: str
    origin: str
    resource_root: str
    model_root: str

    @staticmethod
    def validate_namespace(
        *,
        module_path: str,
        framework_uuid: str,
        model_root: str,
    ) -> None:
        """Reject arbitrary import paths before Python resolves parent packages."""

        normalized_framework = str(framework_uuid or "").strip().lower()
        model_folder_name = os.path.basename(os.path.realpath(str(model_root or "")).rstrip(os.sep))
        namespace = ModelPluginNamespace.create(
            model_package=model_folder_name,
            framework_uuid=normalized_framework,
        )
        namespace.require_module_path(module_path)

    @classmethod
    def validate(
        cls,
        *,
        framework_uuid: str,
        module_path: str,
        class_name: str,
        origin: str,
        resource_root: str,
        model_root: str,
    ) -> RuntimePluginSpec:
        """Return the validate value owned by this runtime plugin spec."""
        normalized_module = str(module_path or "").strip()
        normalized_class = str(class_name or "").strip()
        normalized_resource_root = os.path.realpath(str(resource_root or "").strip())
        normalized_model_root = os.path.realpath(str(model_root or "").strip())
        normalized_origin = os.path.realpath(str(origin or "").strip())

        if not normalized_module:
            raise ApplicationValidationError("Runtime plugin module is not configured.")
        if not normalized_class:
            raise ApplicationValidationError("Runtime plugin class is not configured.")
        cls.validate_namespace(
            module_path=normalized_module,
            framework_uuid=framework_uuid,
            model_root=normalized_model_root,
        )
        if not normalized_model_root or not os.path.isdir(normalized_model_root):
            raise ApplicationValidationError("The selected model resource root is unavailable.")
        if not normalized_resource_root or not os.path.isdir(normalized_resource_root):
            raise ApplicationValidationError("The selected framework resource root is unavailable.")
        if not _is_strict_descendant(normalized_resource_root, normalized_model_root):
            raise ApplicationValidationError(
                "Framework resources must remain inside the selected model root."
            )
        if not os.path.isfile(normalized_origin):
            raise ApplicationValidationError("Runtime plugin origin is not a source file.")
        if not _is_strict_descendant(normalized_origin, normalized_resource_root):
            raise ApplicationValidationError(
                "Runtime plugin origin is outside the selected framework resource root."
            )

        return cls(
            framework_uuid=str(framework_uuid or "").strip().lower(),
            module_path=normalized_module,
            class_name=normalized_class,
            origin=normalized_origin,
            resource_root=normalized_resource_root,
            model_root=normalized_model_root,
        )


@dataclass(frozen=True)
class ImportedFrameworkRuntimeFactory:
    """Construct one runtime from a class validated against public contracts."""

    framework_uuid: str
    runtime_type: type[BaseModelRuntime]

    @classmethod
    def resolve(
        cls,
        module_object: object,
        plugin: RuntimePluginSpec,
    ) -> FrameworkRuntimeFactory:
        """Return the resolve value owned by this imported framework runtime factory."""
        try:
            runtime_type = getattr(module_object, plugin.class_name)
        except AttributeError as error:
            raise ApplicationValidationError(
                "The runtime module does not expose its configured plugin class."
            ) from error
        if not isinstance(runtime_type, type) or not issubclass(runtime_type, BaseModelRuntime):
            raise ApplicationValidationError(
                "The configured plugin class does not satisfy the model runtime contract."
            )

        declared_framework_uuid = (
            str(getattr(runtime_type, "framework_uuid", "") or "").strip().lower()
        )
        if declared_framework_uuid != plugin.framework_uuid:
            raise ApplicationValidationError(
                "The configured plugin class belongs to a different framework."
            )
        environment_factory = getattr(runtime_type, "environment_factory", None)
        if not isinstance(environment_factory, EnvironmentFactory):
            raise ApplicationValidationError(
                "The configured plugin class does not expose an EnvironmentFactory."
            )
        return cls(
            framework_uuid=plugin.framework_uuid,
            runtime_type=runtime_type,
        )

    def create(
        self,
        model_manifest: ModelManifest,
        request: RuntimeInitializationRequest,
    ) -> BaseModelRuntime:
        """Create a validated value owned by this imported framework runtime factory."""
        return self.runtime_type(
            model_manifest,
            parent_logger_name=request.parent_logger_name,
            child_logger_name=request.child_logger_name,
            child_logger_level=request.child_logger_level,
            execution_lease_factory=request.execution_lease_factory,
        )
