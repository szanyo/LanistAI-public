# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate import."""

from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.application_errors import (
    ApplicationError,
    ApplicationValidationError,
    OperationTimeout,
    RuntimeUnavailableError,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.events import RuntimeEventPublisher
from lanista.reinforcengine.rl_model.runtime.import_diagnostics import (
    import_runtime_module_with_diagnostics,
    resolve_runtime_module_origin,
)
from lanista.reinforcengine.rl_model.runtime.plugin import (
    ImportedFrameworkRuntimeFactory,
    RuntimePluginSpec,
)
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import bind_model_plugin_root
from lanista.reinforcengine.rl_model.runtime.requests import RuntimeInitializationRequest
from lanista.reinforcengine.rl_model.runtime.resource_registry import (
    RuntimeResource,
    RuntimeResourceRegistry,
)
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    EnvironmentFactory,
    FrameworkRuntimeFactory,
    ModelPlugin,
)


@dataclass(frozen=True)
class ImportResult:
    """Report the result of import."""

    model: BaseModelRuntime
    framework_uuid: str
    model_manifest: ModelManifest | None = None
    plugin: RuntimePluginSpec | None = None


class ImportService:
    """Loads framework runtime configuration and imports model implementations."""

    def __init__(
        self,
        *,
        framework_normalizer: Callable[[str | None], str],
        progress_reporter: Callable[
            [RuntimeEventPublisher | None, str, str, float, float],
            None,
        ],
        module_importer: Callable[[str], object] = import_runtime_module_with_diagnostics,
        module_origin_resolver: Callable[[str], str] = resolve_runtime_module_origin,
        namespace_binder: Callable[[str], None] = bind_model_plugin_root,
        manifest_cloner: Callable[[ModelManifest], ModelManifest] = deepcopy,
        runtime_registry: RuntimeResourceRegistry | None = None,
        initialization_timeout_seconds: float = 120.0,
        logger: object | None = None,
    ) -> None:
        """Initialize the import service and its owned state."""
        self._framework_normalizer = framework_normalizer
        self._progress_reporter = progress_reporter
        self._module_importer = module_importer
        self._module_origin_resolver = module_origin_resolver
        self._namespace_binder = namespace_binder
        self._manifest_cloner = manifest_cloner
        self._runtime_registry = runtime_registry or RuntimeResourceRegistry(
            framework_normalizer=framework_normalizer,
        )
        self._initialization_timeout_seconds = max(float(initialization_timeout_seconds), 0.001)
        self._logger = logger

    def _info(self, message: str) -> None:
        """Publish a info record through this import service."""
        if self._logger is not None and hasattr(self._logger, "info"):
            self._logger.info(message)

    def _exception(self, error: BaseException) -> None:
        """Publish a exception record through this import service."""
        if self._logger is not None and hasattr(self._logger, "exception"):
            self._logger.exception(error)

    def _warning(self, message: str) -> None:
        """Publish a warning record through this import service."""
        if self._logger is not None and hasattr(self._logger, "warning"):
            self._logger.warning(message)

    @staticmethod
    def _has_confined_module_file(
        module_object: object,
        resolved_model_root: str,
    ) -> bool:
        """Return whether a cache entry resolves to a model-local regular file."""

        try:
            raw_module_file = getattr(module_object, "__file__", None)
            if isinstance(raw_module_file, str):
                module_file = raw_module_file
            elif isinstance(raw_module_file, bytes | os.PathLike):
                module_file = os.fsdecode(raw_module_file)
            else:
                return False
            resolved_module_file = os.path.realpath(module_file)
            return (
                os.path.isfile(resolved_module_file)
                and resolved_module_file != resolved_model_root
                and os.path.commonpath(
                    (resolved_module_file, resolved_model_root)
                )
                == resolved_model_root
            )
        except (OSError, TypeError, ValueError, RuntimeError):
            return False

    @staticmethod
    def _remove_new_runtime_modules(
        validated_model_root: str,
        modules_before: set[str],
    ) -> None:
        """Remove new regular-file modules confined to one validated model root."""

        resolved_model_root = os.path.realpath(validated_model_root)
        for cached_name, cached_module in tuple(sys.modules.items()):
            if cached_name in modules_before:
                continue
            if ImportService._has_confined_module_file(
                cached_module,
                resolved_model_root,
            ) and sys.modules.get(cached_name) is cached_module:
                sys.modules.pop(cached_name, None)

    async def _discard_constructed_candidate(
        self,
        candidate: BaseModelRuntime,
    ) -> None:
        """Invoke the first supported cleanup hook exactly once."""

        for method_name in ("close_async", "shutdown_async", "close", "shutdown"):
            method = getattr(candidate, method_name, None)
            if not callable(method):
                continue
            try:
                cleanup_result = method()
                if inspect.isawaitable(cleanup_result):
                    await cleanup_result
            except BaseException as error:
                self._warning(
                    "Rejected runtime candidate cleanup failed "
                    f"[{type(error).__name__}]."
                )
                self._exception(error)
            break

    def load_framework_configuration(
        self,
        model_manifest: ModelManifest,
        framework_uuid: str,
    ) -> Configuration | None:
        """Load the framework configuration owned by this import service."""
        return self._runtime_registry.load_framework_configuration(model_manifest, framework_uuid)

    def _prepare_candidate_manifest(
        self,
        model_manifest: ModelManifest,
        *,
        selected_session_name: str | None,
        selected_checkpoint_name: str | None,
    ) -> ModelManifest:
        """Clone and stage runtime selection without touching the active model."""

        candidate_manifest = self._manifest_cloner(model_manifest)
        candidate_configuration = candidate_manifest.get_configuration()
        if selected_session_name is not None:
            candidate_configuration.selected_default_model_name = selected_session_name
        if selected_checkpoint_name is not None:
            candidate_configuration.selected_default_checkpoint_name = selected_checkpoint_name
        return candidate_manifest

    def _resolve_runtime_plugin(
        self,
        candidate_manifest: ModelManifest,
        runtime_resource: RuntimeResource,
    ) -> RuntimePluginSpec:
        """Validate plugin namespace, origin, and resource-root confinement."""

        candidate_configuration = candidate_manifest.get_configuration()
        model_root = str(
            getattr(candidate_configuration, "model_folder_path", "") or ""
        )
        self._namespace_binder(os.path.dirname(os.path.realpath(model_root)))
        RuntimePluginSpec.validate_namespace(
            module_path=runtime_resource.module_path,
            framework_uuid=runtime_resource.framework_uuid,
            model_root=model_root,
        )
        try:
            module_origin = self._module_origin_resolver(
                runtime_resource.module_path
            )
        except Exception as error:
            raise RuntimeUnavailableError(
                "The configured runtime module cannot be resolved."
            ) from error
        return RuntimePluginSpec.validate(
            framework_uuid=runtime_resource.framework_uuid,
            module_path=runtime_resource.module_path,
            class_name=runtime_resource.class_name,
            origin=module_origin,
            resource_root=runtime_resource.framework_resource_path,
            model_root=model_root,
        )

    def _import_candidate_module(
        self,
        plugin: RuntimePluginSpec,
        event_publisher: RuntimeEventPublisher | None,
    ) -> tuple[object, set[str]]:
        """Import a plugin and retain the cache baseline for later rollback."""

        module_path = plugin.module_path
        self._info(f"[1/4] Importing module: {module_path}")
        self._progress_reporter(
            event_publisher,
            "Loading model",
            "Importing module",
            20.0,
            100.0,
        )
        modules_before_import = set(sys.modules)
        try:
            return self._module_importer(module_path), modules_before_import
        except Exception as error:
            self._remove_new_runtime_modules(
                plugin.model_root,
                modules_before_import,
            )
            raise RuntimeUnavailableError(
                "The configured runtime module could not be imported."
            ) from error

    def _resolve_runtime_factory(
        self,
        module_object: object,
        plugin: RuntimePluginSpec,
        event_publisher: RuntimeEventPublisher | None,
    ) -> FrameworkRuntimeFactory:
        """Bind the imported plugin class to the framework factory contract."""

        self._info(f"[2/4] Locating plugin class: {plugin.class_name}")
        self._progress_reporter(
            event_publisher,
            "Loading model",
            "Resolving model plugin",
            45.0,
            100.0,
        )
        return ImportedFrameworkRuntimeFactory.resolve(
            module_object,
            plugin,
        )

    def _construct_candidate(
        self,
        runtime_factory: FrameworkRuntimeFactory,
        candidate_manifest: ModelManifest,
        runtime_request: RuntimeInitializationRequest,
        event_publisher: RuntimeEventPublisher | None,
    ) -> BaseModelRuntime:
        """Construct one isolated candidate through the framework factory."""

        self._info(
            f"[3/4] Instantiating {runtime_factory.framework_uuid} model plugin"
        )
        self._progress_reporter(
            event_publisher,
            "Loading model",
            "Instantiating model plugin",
            70.0,
            100.0,
        )
        try:
            return runtime_factory.create(
                candidate_manifest,
                runtime_request,
            )
        except ApplicationError:
            raise
        except Exception as error:
            raise RuntimeUnavailableError(
                "The model runtime could not be constructed."
            ) from error

    @staticmethod
    def _validate_candidate_contract(
        candidate: BaseModelRuntime,
        runtime_factory: FrameworkRuntimeFactory,
    ) -> None:
        """Reject a constructed candidate that changed its plugin contract."""

        if not isinstance(candidate, ModelPlugin):
            raise ApplicationValidationError(
                "The constructed runtime does not satisfy the ModelPlugin contract."
            )
        if not isinstance(candidate.environment_factory, EnvironmentFactory):
            raise ApplicationValidationError(
                "The constructed runtime discarded its EnvironmentFactory."
            )
        candidate_framework_uuid = str(
            candidate.framework_uuid or ""
        ).strip().lower()
        if candidate_framework_uuid != runtime_factory.framework_uuid:
            raise ApplicationValidationError(
                "The constructed runtime changed its declared framework."
            )

    async def _initialize_candidate(
        self,
        rl_model: BaseModelRuntime,
        event_publisher: RuntimeEventPublisher | None,
    ) -> BaseModelRuntime:
        """Initialize and validate a candidate before it can enter the active slot."""

        self._info("[4/4] Running async initialization...")
        self._progress_reporter(
            event_publisher,
            "Loading model",
            "Initializing runtime",
            90.0,
            100.0,
        )
        initialization_task = asyncio.create_task(
            self._initialize_to_quiescence(rl_model)
        )
        try:
            initialized_model = await asyncio.wait_for(
                asyncio.shield(initialization_task),
                timeout=self._initialization_timeout_seconds,
            )
        except TimeoutError as error:
            await self._cancel_and_join_initialization(
                rl_model,
                initialization_task,
            )
            raise OperationTimeout(
                "Model runtime initialization timed out."
            ) from error
        except asyncio.CancelledError:
            await self._cancel_and_join_initialization(
                rl_model,
                initialization_task,
            )
            raise
        except ApplicationError:
            await self._cancel_and_join_initialization(
                rl_model,
                initialization_task,
            )
            raise
        except Exception as error:
            await self._cancel_and_join_initialization(
                rl_model,
                initialization_task,
            )
            raise RuntimeUnavailableError(
                "The model runtime could not be initialized."
            ) from error
        if initialized_model is not rl_model:
            if isinstance(initialized_model, BaseModelRuntime):
                await self._discard_constructed_candidate(initialized_model)
                raise ApplicationValidationError(
                    "Runtime initialization must return the constructed model candidate."
                )
            raise ApplicationValidationError(
                "Runtime initialization returned an object outside the model runtime contract."
            )
        return rl_model

    @staticmethod
    async def _initialize_to_quiescence(
        rl_model: BaseModelRuntime,
    ) -> BaseModelRuntime:
        """Initialize the to quiescence owned by this import service."""
        initialized_model = await rl_model.initialize_async()
        await rl_model.wait_for_initialization_quiescence()
        return initialized_model

    async def _cancel_and_join_initialization(
        self,
        rl_model: BaseModelRuntime,
        initialization_task: asyncio.Task[BaseModelRuntime],
    ) -> None:
        """Keep the join alive if the load caller cancels during rollback."""

        join_task = asyncio.create_task(
            rl_model.cancel_initialization_and_wait(initialization_task)
        )
        try:
            await asyncio.shield(join_task)
        except asyncio.CancelledError:
            try:
                await join_task
            except BaseException as cleanup_error:
                self._record_initialization_cleanup_failure(
                    rl_model,
                    cleanup_error,
                )
            raise
        except BaseException as cleanup_error:
            self._record_initialization_cleanup_failure(
                rl_model,
                cleanup_error,
            )

    def _record_initialization_cleanup_failure(
        self,
        rl_model: BaseModelRuntime,
        error: BaseException,
    ) -> None:
        """Record the initialization cleanup failure owned by this import service."""
        self._warning(
            "Runtime initialization cleanup could not prove quiescence "
            f"[{type(error).__name__}]."
        )
        self._exception(error)
        rl_model._report_execution_cleanup_failure(error)

    async def load_model(
        self,
        model_manifest: ModelManifest,
        runtime_request: RuntimeInitializationRequest,
        event_publisher: RuntimeEventPublisher | None = None,
        framework_uuid: str | None = None,
        selected_session_name: str | None = None,
        selected_checkpoint_name: str | None = None,
    ) -> ImportResult:
        """Load the model owned by this import service."""
        candidate_manifest = self._prepare_candidate_manifest(
            model_manifest,
            selected_session_name=selected_session_name,
            selected_checkpoint_name=selected_checkpoint_name,
        )
        runtime_resource = self._runtime_registry.resolve(
            candidate_manifest,
            framework_uuid,
        )
        try:
            plugin = self._resolve_runtime_plugin(
                candidate_manifest,
                runtime_resource,
            )
            module_object, modules_before_import = self._import_candidate_module(
                plugin,
                event_publisher,
            )
            rl_model: BaseModelRuntime | None = None
            try:
                runtime_factory = self._resolve_runtime_factory(
                    module_object,
                    plugin,
                    event_publisher,
                )
                rl_model = self._construct_candidate(
                    runtime_factory,
                    candidate_manifest,
                    runtime_request,
                    event_publisher,
                )
                self._validate_candidate_contract(
                    rl_model,
                    runtime_factory,
                )
                if not rl_model.has_managed_execution_lifecycle():
                    raise ApplicationValidationError(
                        "The runtime plugin discarded the managed execution lifecycle dependency."
                    )
                initialized_model = await self._initialize_candidate(
                    rl_model,
                    event_publisher,
                )
                return ImportResult(
                    model=initialized_model,
                    framework_uuid=runtime_resource.framework_uuid,
                    model_manifest=candidate_manifest,
                    plugin=plugin,
                )
            except BaseException:
                if rl_model is not None:
                    await self._discard_constructed_candidate(rl_model)
                self._remove_new_runtime_modules(
                    plugin.model_root,
                    modules_before_import,
                )
                raise
        except ApplicationError as error:
            self._exception(error)
            raise
