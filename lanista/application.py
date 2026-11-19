# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the framework-independent LanistAI application services."""

import logging.config
import multiprocessing
import os
import random
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

import yaml

from lanista.application_bootstrap import APP_CONFIGURATION as APP_CONFIGURATION
from lanista.application_restart_service import (
    ApplicationRestartRequest,
    ApplicationRestartService,
)
from lanista.foundation.filesystem import makedirs
from lanista.foundation.logging import CustomWarning, PipelineHandler
from lanista.inspection_lifecycle import (
    InspectionLifecycleCoordinator,
    InspectionPreferences,
    InspectionStatus,
)
from lanista.locations import (
    get_app_logging_config_path,
    get_app_runtime_state_path,
    get_legacy_app_runtime_state_path,
    get_logs_path,
    get_models_root_path,
)
from lanista.native_gui_restart_gate import NativeGuiRestartGate
from lanista.reinforcengine.rl_model.application_service_container import (
    ApplicationServiceContainer,
)
from lanista.reinforcengine.rl_model.application_services import ApplicationServices
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)
from lanista.reinforcengine.rl_model.inspection_target_service import InspectionTarget
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.state_bootstrap_service import (
    materialize_legacy_runtime_state,
)
from lanista.reinforcengine.rl_technologies.training_device import (
    configured_training_gpu_enabled,
)
from lanista.reinforcengine.rl_model.runtime.tensorboard_layout_service import (
    TensorboardApplicationLayoutRequest,
    TensorboardLayoutMode,
    TensorboardRuntimeLayoutService,
)
from lanista.reinforcengine.utils.tensorboard_lifecycle import TensorboardLifecycle


class _LogRecordQueue(Protocol):
    """Minimal queue surface exposed by the logging pipeline handler."""

    def empty(self) -> bool:
        """Return whether no buffered record is available."""

    def get(self) -> logging.LogRecord:
        """Remove and return the next buffered record."""


class _ApplicationServiceContainerFactory(Protocol):
    """Typed constructor boundary for the legacy logger-aware container."""

    def __call__(
        self,
        *,
        parent_logger_name: str,
        child_logger_name: str,
        child_logger_level: object,
        native_gui_mutation_guard: Callable[[], None],
    ) -> ApplicationServiceContainer:
        """Construct the application-service container."""


class _TensorboardLifecycleFactory(Protocol):
    """Typed constructor boundary for the legacy TensorBoard lifecycle."""

    def __call__(self) -> TensorboardLifecycle:
        """Construct a deferred TensorBoard lifecycle."""


class LanistAIApplication:
    """Coordinate the LanistAI application lifecycle."""

    _instance = None
    _logger: logging.Logger | None = None
    _application_services: ApplicationServices | None = None
    _tensorboard_lifecycle: TensorboardLifecycle | None = None
    _inspection_coordinator: InspectionLifecycleCoordinator | None = None
    _tensorboard_layout_service = TensorboardRuntimeLayoutService()
    _restart_service = ApplicationRestartService()
    _native_gui_restart_gate = NativeGuiRestartGate()
    _gui_evaluation_post_run_continuation: GuiEvaluationRestartContinuation | None = None
    _restart_failure: str | None = None

    def __new__(cls) -> Self:
        """Creates the single instance of the class or returns it if it already exists.
        Sets the default controller to the idle controller.

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __enter__(self) -> Self:
        """Initialize application-owned services and return the composition root."""

        multiprocessing.freeze_support()
        self._configure_application_logging()
        self._materialize_runtime_state()
        self._configure_custom_warning()
        self._configure_application_services()
        self._configure_tensorboard_lifecycle()
        self._attach_tensorflow_logger()
        logging.captureWarnings(True)
        return self

    @staticmethod
    def _materialize_runtime_state() -> None:
        """Copy valid repository-local selection state into a fresh data home."""

        materialize_legacy_runtime_state(
            get_legacy_app_runtime_state_path(),
            get_app_runtime_state_path(),
        )

    def _configure_application_logging(self) -> None:
        """Load logging configuration before constructing dependent services."""

        full_logging_config_file_location = get_app_logging_config_path()
        full_logging_save_path = get_logs_path()
        makedirs(full_logging_save_path, exist_ok=True)

        logging_configuration = self._load_logging_configuration(full_logging_config_file_location)
        logging.config.dictConfig(logging_configuration)

        self._logger = logging.getLogger("lanistai")
        self._logger.info("LanistAI logger registered as the application root.")
        self._logger.info("Logger ONLINE.")
        self._logger.info("Application Configuration ONLINE.")
        self._logger.info("Application Configuration LOADED.")

    @staticmethod
    def _load_logging_configuration(path: str) -> dict[str, object]:
        """Load logging config and anchor relative file handlers to the data home."""

        with open(path, encoding="utf-8") as config_file:
            loaded = yaml.safe_load(config_file)
        if not isinstance(loaded, Mapping):
            raise ValueError("Application logging configuration must be a mapping.")

        configuration = dict(loaded)
        handlers_value = configuration.get("handlers")
        if isinstance(handlers_value, Mapping):
            handlers: dict[str, object] = {}
            for name, handler_value in handlers_value.items():
                if not isinstance(name, str) or not isinstance(handler_value, Mapping):
                    raise ValueError("Logging handlers must be string-keyed mappings.")
                handler = dict(handler_value)
                filename = handler.get("filename")
                if isinstance(filename, str) and not os.path.isabs(filename):
                    relative_path = Path(filename)
                    if relative_path.parts[:2] == ("data", "logs"):
                        relative_path = Path(*relative_path.parts[2:])
                    elif relative_path.parts[:1] == ("logs",):
                        relative_path = Path(*relative_path.parts[1:])
                    log_root = Path(get_logs_path()).resolve()
                    target_path = (log_root / relative_path).resolve()
                    if not target_path.is_relative_to(log_root):
                        raise ValueError("Relative logging path escapes the application log root.")
                    handler["filename"] = str(target_path)
                handlers[name] = handler
            configuration["handlers"] = handlers
        return configuration

    def _require_logger(self) -> logging.Logger:
        """Return the configured application logger."""

        if self._logger is None:
            raise RuntimeError("Application logging is not initialized.")
        return self._logger

    def _configure_custom_warning(self) -> None:
        """Create the warning adapter with the application logger as parent."""

        logger = self._require_logger()
        self._custom_warning = CustomWarning(
            parent_logger_name=logger.name,
            child_logger_name="custom_warning",
            child_logger_level=APP_CONFIGURATION.logs.globally.level,
        )
        logger.info("Custom warning ONLINE.")

    def _configure_application_services(self) -> None:
        """Construct the typed application-service composition root."""

        logger = self._require_logger()
        container_factory = cast(
            _ApplicationServiceContainerFactory,
            ApplicationServiceContainer,
        )
        service_container = container_factory(
            parent_logger_name=logger.name,
            child_logger_name="application_services",
            child_logger_level=APP_CONFIGURATION.logs.globally.level,
            native_gui_mutation_guard=(
                self._native_gui_restart_gate.require_fresh_process
            ),
        )
        self._application_services = service_container.application_services
        logger.info("Application service container ONLINE.")

    def _configure_tensorboard_lifecycle(self) -> None:
        """Configure native and Analysis server ownership for the application."""

        logger = self._require_logger()
        lifecycle_factory = cast(
            _TensorboardLifecycleFactory,
            TensorboardLifecycle,
        )
        self._tensorboard_lifecycle = lifecycle_factory()
        self._tensorboard_lifecycle.configure(
            logger=logger,
            executable=self._get_tensorboard_executable(),
            subprocess_kwargs=self._get_tensorboard_subprocess_kwargs(),
            port="6006",
        )
        self._inspection_coordinator = InspectionLifecycleCoordinator(
            native_lifecycle=self._tensorboard_lifecycle,
            native_executable=self._get_tensorboard_executable(),
            configuration=APP_CONFIGURATION,
            logger=logger,
            subprocess_kwargs=self._get_tensorboard_subprocess_kwargs(),
        )
        status = self._inspection_coordinator.start_analysis_service(
            Path(get_models_root_path())
        )
        if status.analysis.ready:
            logger.info(
                "Analysis Dashboard ONLINE at [link=%s]%s[/link]",
                status.analysis.url,
                status.analysis.url,
                extra={"markup": True},
            )
        else:
            logger.warning(
                "Analysis Dashboard failed to start: %s",
                status.analysis.error or "unknown failure",
            )
        logger.info("Native TensorBoard deferred until an inspection target is selected.")

    @staticmethod
    def _find_tensorflow_pipeline(
        tensorflow_logger: logging.Logger,
    ) -> _LogRecordQueue | None:
        """Return the pipeline queue installed before TensorFlow logger adoption."""

        for handler in tensorflow_logger.handlers:
            if handler.name == "pipeline" and isinstance(handler, PipelineHandler):
                return cast(_LogRecordQueue, handler.get_pipeline())
        return None

    def _attach_tensorflow_logger(self) -> None:
        """Attach TensorFlow logging and replay records buffered during import."""

        logger = self._require_logger()
        tensorflow_logger = logging.getLogger("tensorflow")
        tensorflow_logger_queue = self._find_tensorflow_pipeline(tensorflow_logger)
        if tensorflow_logger_queue is None:
            logger.error("TensorFlow logger DETACHED.")
            logger.warning(
                "TensorFlow logger NOT FOUND.\nTensorflow import has not been happened before."
            )
            return

        tensorflow_logger.handlers.clear()
        tensorflow_logger.setLevel(logging.DEBUG)
        tensorflow_logger.propagate = True
        tensorflow_logger.parent = logger
        logger.info("TensorFlow logger ATTACHED.")

        while not tensorflow_logger_queue.empty():
            logger.handle(tensorflow_logger_queue.get())

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop application-owned context-manager resources."""

        self.shutdown_inspection()

    def in_critical_phase(self) -> bool:
        """Return the execution-owned administration boundary snapshot."""

        if self._application_services is None:
            return False
        return self._application_services.administration.safety.snapshot().execution_active

    def request_restart(self, request: ApplicationRestartRequest) -> bool:
        """Idempotently request a fresh LanistAI process image."""

        return self._restart_service.request(request)

    def restart_pending(self) -> bool:
        """Return whether normal shutdown must yield to process replacement."""

        return self._restart_service.restart_pending

    def native_gui_restart_required(self) -> bool:
        """Return whether embedded GUI reuse requires process replacement."""

        return self._native_gui_restart_gate.restart_required

    def require_fresh_native_gui_process(self) -> None:
        """Reject a run or configuration mutation in a used native process."""

        self._native_gui_restart_gate.require_fresh_process()

    def set_gui_evaluation_post_run_continuation(
        self,
        continuation: GuiEvaluationRestartContinuation | None,
    ) -> None:
        """Retain the terminal GUI run until the user chooses its next action."""

        self._gui_evaluation_post_run_continuation = continuation

    def gui_evaluation_post_run_continuation(
        self,
    ) -> GuiEvaluationRestartContinuation | None:
        """Return the terminal GUI run awaiting a user decision."""

        return self._gui_evaluation_post_run_continuation

    def consume_startup_restart_request(self) -> ApplicationRestartRequest | None:
        """Return the inherited one-shot continuation, at most once."""

        return self._restart_service.consume_startup_request()

    def clear_failed_restart_request(self) -> None:
        """Restore a rejected GPU choice before clearing a failed restart."""

        request = self._restart_service.pending_request
        if request is not None and request.reason == "training-gpu-change":
            previous = request.continuation.get("previous_use_gpu")
            if isinstance(previous, bool):
                self.set_training_gpu_enabled(previous)
        self._restart_service.clear_failed_request()

    def record_restart_failure(self, message: str) -> None:
        """Retain one process-replacement failure for the next Textual mount."""

        self._restart_failure = str(message).strip() or "unknown failure"

    def consume_restart_failure(self) -> str | None:
        """Return and clear the process-replacement failure notification."""

        failure = self._restart_failure
        self._restart_failure = None
        return failure

    def replace_process_image(self) -> None:
        """Replace the complete application after Textual has unmounted."""

        self.shutdown_inspection()
        for handler in logging.getLogger().handlers:
            handler.flush()
        try:
            self._restart_service.replace_process_image()
        except Exception:
            coordinator = self.__dict__.get("_inspection_coordinator")
            try:
                if coordinator is not None:
                    coordinator.start_analysis_service(Path(get_models_root_path()))
            except Exception:
                logger = self.__dict__.get("_logger")
                if logger is not None:
                    logger.exception(
                        "Failed to restore Analysis Dashboard after process replacement failure."
                    )
            raise

    def debug_show_log_handlers(self) -> None:
        """Log the handlers attached to registered child loggers."""

        application_logger = self._require_logger()
        log_methods = [
            application_logger.debug,
            application_logger.info,
            application_logger.warning,
            application_logger.error,
            application_logger.critical,
        ]

        for name, logger in logging.root.manager.loggerDict.items():
            if isinstance(logger, logging.Logger):
                handler_names = [h.__class__.__name__ for h in logger.handlers]
                random.choice(log_methods)(f"Logger {name} has handlers: {handler_names}")

    def get_logger(self) -> logging.Logger | None:
        """Return the application logger when bootstrap has configured it."""

        return self._logger

    def get_application_services(self) -> ApplicationServices:
        """Return the typed model-service composition root."""

        if self._application_services is None:
            raise RuntimeError("Application services are not initialized.")
        return self._application_services

    def _get_tensorboard_subprocess_kwargs(self) -> dict[str, object]:
        """Return the tensorboard subprocess kwargs exposed by this LanistAI application."""
        logger = self._require_logger()
        return {
            "parent_logger_name": logger.name,
            "child_logger_name": "tensorboard_subprocess",
            "child_logger_level": APP_CONFIGURATION.logs.globally.level,
        }

    def _get_tensorboard_executable(self) -> str:
        """Return the tensorboard executable exposed by this LanistAI application."""
        return os.path.join(sys.prefix, "bin", "tensorboard")

    def get_current_model_tensorboard_path(self, create_if_missing: bool = True) -> str | None:
        """Return the current model tensorboard path exposed by this LanistAI application."""
        if self._application_services is None:
            return None

        active_manifest = self._application_services.runtime.staging.get_selected_manifest()
        current_model = self._application_services.runtime.access.get_active_runtime()
        if current_model is not None:
            active_manifest = current_model.get_manifest()
        if active_manifest is None:
            return None

        configuration = active_manifest.get_configuration()
        _ = create_if_missing
        return str(configuration.model_folder_path)

    def _resolve_tensorboard_logdir_for_mode(
        self, mode: RLModelRunningModes, framework_uuid: str | None, selected_session: str | None
    ) -> str | None:
        """Resolve the tensorboard logdir for mode required by this LanistAI application."""
        model_folder_path = self.get_current_model_tensorboard_path(create_if_missing=False)
        if not model_folder_path:
            return None
        if mode == RLModelRunningModes.EVALUATION:
            layout_mode = TensorboardLayoutMode.EVALUATION
        elif mode == RLModelRunningModes.TRAINING:
            layout_mode = TensorboardLayoutMode.TRAINING
        else:
            return None
        return self._tensorboard_layout_service.resolve_application_log_dir(
            TensorboardApplicationLayoutRequest(
                model_folder_path=model_folder_path,
                selected_session=selected_session,
                framework_uuid=framework_uuid,
                mode=layout_mode,
            )
        )

    def get_active_tensorboard_logdir(self) -> str | None:
        """Return the active tensorboard logdir exposed by this LanistAI application."""
        if self._tensorboard_lifecycle is None:
            return None
        return self._tensorboard_lifecycle.get_active_logdir()

    def get_tensorboard_status_label(self, selected_session: str | None = None) -> str:
        """Return the tensorboard status label exposed by this LanistAI application."""
        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is not None:
            status = coordinator.status()
            if status.running:
                return (
                    f"native {status.native.state.value}; "
                    f"analysis {status.analysis.state.value}"
                )
        if self._tensorboard_lifecycle is None:
            return "offline"
        return self._tensorboard_lifecycle.build_status_label(selected_session)

    def start_inspection(self, target: InspectionTarget) -> InspectionStatus:
        """Start both inspection viewers for one validated immutable target."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is None:
            raise RuntimeError("Inspection lifecycle is not initialized.")
        return coordinator.start(target)

    def get_inspection_status(self) -> InspectionStatus | None:
        """Return the current dual-viewer status when inspection is configured."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        return None if coordinator is None else coordinator.status()

    def open_native_inspection(self) -> bool:
        """Open native TensorBoard manually when its server is ready."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        return False if coordinator is None else coordinator.open_native()

    def open_analysis_inspection(self) -> bool:
        """Open the Analysis Dashboard manually when its server is ready."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        return False if coordinator is None else coordinator.open_analysis()

    def get_inspection_preferences(self) -> InspectionPreferences:
        """Return the global automatic browser-opening preferences."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is None:
            return InspectionPreferences()
        return coordinator.preferences.snapshot()

    def set_auto_open_native_tensorboard(self, enabled: bool) -> InspectionPreferences:
        """Persist the native TensorBoard automatic-open preference."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is None:
            raise RuntimeError("Inspection lifecycle is not initialized.")
        return coordinator.preferences.set_auto_open_native_tensorboard(enabled)

    def set_auto_open_analysis_dashboard(self, enabled: bool) -> InspectionPreferences:
        """Persist the Analysis Dashboard automatic-open preference."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is None:
            raise RuntimeError("Inspection lifecycle is not initialized.")
        return coordinator.preferences.set_auto_open_analysis_dashboard(enabled)

    def get_training_gpu_enabled(self) -> bool:
        """Return the saved global training GPU choice."""

        return configured_training_gpu_enabled(APP_CONFIGURATION)

    def set_training_gpu_enabled(self, enabled: bool) -> bool:
        """Persist the GPU choice for the next application process."""

        training_configuration = APP_CONFIGURATION.get_data().get("training")
        if not isinstance(training_configuration, dict):
            APP_CONFIGURATION.training = {"use_gpu": bool(enabled)}
        else:
            APP_CONFIGURATION.training.use_gpu = bool(enabled)
        APP_CONFIGURATION.save()
        return self.get_training_gpu_enabled()

    def get_splash_screen_enabled(self) -> bool:
        """Return whether the splash screen should appear at application startup."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, Mapping):
            return True
        value = gui_configuration.get("splash_screen")
        return value if isinstance(value, bool) else True

    def set_splash_screen_enabled(self, enabled: bool) -> bool:
        """Persist whether the splash screen should appear at next startup."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, dict):
            APP_CONFIGURATION.gui = {"splash_screen": bool(enabled)}
        else:
            APP_CONFIGURATION.gui.splash_screen = bool(enabled)
        APP_CONFIGURATION.save()
        return self.get_splash_screen_enabled()

    def get_maximize_terminal_enabled(self) -> bool:
        """Return whether startup should request terminal-window maximization."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, Mapping):
            return False
        value = gui_configuration.get("maximize_terminal")
        return value if isinstance(value, bool) else False

    def set_maximize_terminal_enabled(self, enabled: bool) -> bool:
        """Persist whether startup should request terminal-window maximization."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, dict):
            APP_CONFIGURATION.gui = {"maximize_terminal": bool(enabled)}
        else:
            APP_CONFIGURATION.gui.maximize_terminal = bool(enabled)
        APP_CONFIGURATION.save()
        return self.get_maximize_terminal_enabled()

    def get_fullscreen_enabled(self) -> bool:
        """Return whether startup should request terminal fullscreen mode."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, Mapping):
            return False
        value = gui_configuration.get("fullscreen")
        return value if isinstance(value, bool) else False

    def set_fullscreen_enabled(self, enabled: bool) -> bool:
        """Persist whether startup should request terminal fullscreen mode."""

        gui_configuration = APP_CONFIGURATION.get_data().get("gui")
        if not isinstance(gui_configuration, dict):
            APP_CONFIGURATION.gui = {"fullscreen": bool(enabled)}
        else:
            APP_CONFIGURATION.gui.fullscreen = bool(enabled)
        APP_CONFIGURATION.save()
        return self.get_fullscreen_enabled()

    def stop_tensorboard(self) -> None:
        """Stop target inspection while retaining the application Analysis server."""
        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is not None:
            coordinator.stop_target(clear_target=True)
            return
        if self._tensorboard_lifecycle is None:
            return
        self._tensorboard_lifecycle.stop()

    def shutdown_inspection(self) -> None:
        """Stop every application-owned inspection child process idempotently."""

        coordinator = self.__dict__.get("_inspection_coordinator")
        if coordinator is not None:
            coordinator.stop(clear_target=True)
            return
        if self._tensorboard_lifecycle is not None:
            self._tensorboard_lifecycle.stop()

    def restart_tensorboard_for_current_model(self) -> None:
        """Restart the tensorboard for current model owned by this LanistAI application."""
        tensorboard_logdir = self.get_current_model_tensorboard_path()
        if self._tensorboard_lifecycle is None:
            return
        self._tensorboard_lifecycle.restart(
            tensorboard_logdir,
            online_context="model logdir",
            offline_reason="TensorBoard remains offline because no model is loaded.",
        )

    def restart_tensorboard_for_mode(
        self,
        mode: RLModelRunningModes,
        framework_uuid: str | None = None,
        selected_session: str | None = None,
    ) -> None:
        """Restart the tensorboard for mode owned by this LanistAI application."""
        tensorboard_logdir = self._resolve_tensorboard_logdir_for_mode(
            mode, framework_uuid, selected_session
        )
        if self._tensorboard_lifecycle is None:
            return
        self._tensorboard_lifecycle.restart(
            tensorboard_logdir,
            online_context=f"mode '{mode.value}' logdir",
            offline_reason=f"TensorBoard remains offline for mode '{mode.value}': no existing log directory.",
        )

    def restart_tensorboard_for_logdir(self, logdir: str | None, *, online_context: str) -> None:
        """Restart the tensorboard for logdir owned by this LanistAI application."""
        if self._tensorboard_lifecycle is None:
            return
        self._tensorboard_lifecycle.restart(
            logdir,
            online_context=online_context,
            offline_reason=f"TensorBoard remains offline: no existing log directory for {online_context}.",
        )
