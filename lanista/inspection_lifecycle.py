# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own the two local inspection servers behind one typed lifecycle."""

from __future__ import annotations

import logging
import socket
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock, Thread
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from lanista.foundation.browser import open_browser_tab
from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.subprocess import CustomSubprocess
from lanista.reinforcengine.rl_model.inspection_target_service import (
    InspectionTarget,
    InspectionTargetKind,
)
from lanista.reinforcengine.utils.tensorboard_lifecycle import TensorboardLifecycle


class InspectionViewer(StrEnum):
    """Name one independently managed browser viewer."""

    NATIVE = "native"
    ANALYSIS = "analysis"


class InspectionLifecycleState(StrEnum):
    """Describe one inspection server without conflating partial success."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class InspectionViewerStatus:
    """Project the observable state of one local inspection server."""

    viewer: InspectionViewer
    state: InspectionLifecycleState
    target: str | None = None
    url: str | None = None
    port: int | None = None
    error: str | None = None

    @property
    def ready(self) -> bool:
        """Return whether a manual browser-open action is currently valid."""

        return self.state is InspectionLifecycleState.READY and self.url is not None


@dataclass(frozen=True, slots=True)
class InspectionStatus:
    """Return one atomic status snapshot for both owned viewers."""

    target: InspectionTarget | None
    native: InspectionViewerStatus
    analysis: InspectionViewerStatus

    @property
    def running(self) -> bool:
        """Return whether a target-specific inspection intent is active."""

        return self.target is not None and any(
            status.state
            in {InspectionLifecycleState.STARTING, InspectionLifecycleState.READY}
            for status in (self.native, self.analysis)
        )


@dataclass(frozen=True, slots=True)
class InspectionPreferences:
    """Persisted, application-global browser-opening policy."""

    auto_open_native_tensorboard: bool = True
    auto_open_analysis_dashboard: bool = True


class InspectionPreferencesService:
    """Read and atomically persist inspection preferences in ``AppConfiguration``."""

    _SECTION = "inspection"

    def __init__(self, configuration: AppConfiguration | object) -> None:
        """Initialize preferences without requiring a writable model selection."""

        self._configuration = configuration
        self._fallback = InspectionPreferences()
        self._ensure_defaults()

    def snapshot(self) -> InspectionPreferences:
        """Return normalized values while tolerating older configuration files."""

        data = self._configuration_data()
        section = data.get(self._SECTION, {})
        if not isinstance(section, dict):
            section = {}
        return InspectionPreferences(
            auto_open_native_tensorboard=self._as_bool(
                section.get("auto_open_native_tensorboard"),
                self._fallback.auto_open_native_tensorboard,
            ),
            auto_open_analysis_dashboard=self._as_bool(
                section.get("auto_open_analysis_dashboard"),
                self._fallback.auto_open_analysis_dashboard,
            ),
        )

    def set_auto_open_native_tensorboard(self, enabled: bool) -> InspectionPreferences:
        """Persist the native TensorBoard browser preference."""

        return self._set("auto_open_native_tensorboard", enabled)

    def set_auto_open_analysis_dashboard(self, enabled: bool) -> InspectionPreferences:
        """Persist the Analysis Dashboard browser preference."""

        return self._set("auto_open_analysis_dashboard", enabled)

    def _set(self, name: str, enabled: bool) -> InspectionPreferences:
        data = self._configuration_data()
        if not data:
            current = self.snapshot()
            self._fallback = InspectionPreferences(
                auto_open_native_tensorboard=(
                    bool(enabled)
                    if name == "auto_open_native_tensorboard"
                    else current.auto_open_native_tensorboard
                ),
                auto_open_analysis_dashboard=(
                    bool(enabled)
                    if name == "auto_open_analysis_dashboard"
                    else current.auto_open_analysis_dashboard
                ),
            )
            return self._fallback
        section = data.get(self._SECTION)
        if not isinstance(section, dict):
            self._assign_section({name: bool(enabled)})
        else:
            wrapper = getattr(self._configuration, self._SECTION)
            setattr(wrapper, name, bool(enabled))
        self._save()
        return self.snapshot()

    def _ensure_defaults(self) -> None:
        data = self._configuration_data()
        if not data:
            return
        section = data.get(self._SECTION)
        changed = False
        if not isinstance(section, dict):
            self._assign_section(
                {
                    "auto_open_native_tensorboard": True,
                    "auto_open_analysis_dashboard": True,
                }
            )
            changed = True
        else:
            wrapper = getattr(self._configuration, self._SECTION)
            for name in (
                "auto_open_native_tensorboard",
                "auto_open_analysis_dashboard",
            ):
                if name not in section:
                    setattr(wrapper, name, True)
                    changed = True
        if changed:
            self._save()

    def _configuration_data(self) -> dict[str, object]:
        getter = getattr(self._configuration, "get_data", None)
        if not callable(getter):
            return {}
        data = getter()
        return data if isinstance(data, dict) else {}

    def _assign_section(self, section: dict[str, bool]) -> None:
        setattr(self._configuration, self._SECTION, section)

    def _save(self) -> None:
        save = getattr(self._configuration, "save", None)
        if callable(save):
            save()

    @staticmethod
    def _as_bool(value: object, default: bool) -> bool:
        return value if isinstance(value, bool) else default


class InspectionProcess(Protocol):
    """Minimal subprocess surface required by the analysis server owner."""

    def start(self) -> bool:
        """Start the child process."""

    def stop(self) -> None:
        """Stop the child process idempotently."""


class InspectionProcessFactory(Protocol):
    """Construct one managed subprocess from an immutable command."""

    def __call__(self, command: Sequence[str], **kwargs: object) -> InspectionProcess:
        """Return the process wrapper for ``command``."""


class InspectionLifecycleCoordinator:
    """Start, retarget, observe, open, and stop both inspection servers."""

    _HOST = "127.0.0.1"
    _ANALYSIS_PORT = 8765

    def __init__(
        self,
        *,
        native_lifecycle: TensorboardLifecycle,
        native_executable: str,
        configuration: AppConfiguration | object,
        logger: logging.Logger,
        subprocess_kwargs: dict[str, object] | None = None,
        process_factory: InspectionProcessFactory = CustomSubprocess,
        browser_opener: Callable[[str], object] = open_browser_tab,
        readiness_probe: Callable[[str], bool] | None = None,
        port_allocator: Callable[[int], int] | None = None,
        analysis_port_available: Callable[[int], bool] | None = None,
        startup_timeout_seconds: float = 4.0,
        repository_root: Path | None = None,
    ) -> None:
        """Initialize one focused application-owned dual-server coordinator."""

        self.preferences = InspectionPreferencesService(configuration)
        self._native = native_lifecycle
        self._native_executable = native_executable
        self._logger = logger
        self._subprocess_kwargs = dict(subprocess_kwargs or {})
        self._process_factory = process_factory
        self._browser_opener = browser_opener
        self._readiness_probe = readiness_probe or self._http_ready
        self._native_port_allocator = port_allocator or self._available_port
        self._analysis_port_available = analysis_port_available or self._fixed_port_available
        self._startup_timeout_seconds = max(0.0, float(startup_timeout_seconds))
        self._repository_root = repository_root or Path(__file__).resolve().parents[1]
        self._analysis_process: InspectionProcess | None = None
        self._analysis_models_root: Path | None = None
        self._browser_open_lock = Lock()
        self._browser_opening: set[str] = set()
        self._browser_open_queue: deque[str] = deque()
        self._browser_worker_running = False
        self._target: InspectionTarget | None = None
        self._native_status = self._stopped(InspectionViewer.NATIVE)
        self._analysis_status = self._stopped(InspectionViewer.ANALYSIS)

    def start(self, target: InspectionTarget) -> InspectionStatus:
        """Atomically replace any prior target and start both local viewers."""

        self.stop_target(clear_target=False)
        self._target = target
        target_label = target.display_name
        self._native_status = InspectionViewerStatus(
            InspectionViewer.NATIVE,
            InspectionLifecycleState.STARTING,
            target=target_label,
        )
        self._start_native(target)
        if not self._analysis_status.ready:
            analysis_target = (
                "all models" if self._analysis_models_root is not None else target_label
            )
            self._analysis_status = InspectionViewerStatus(
                InspectionViewer.ANALYSIS,
                InspectionLifecycleState.STARTING,
                target=analysis_target,
            )
            self._start_analysis(
                target=None if self._analysis_models_root is not None else target,
                models_root=self._analysis_models_root,
            )
        preferences = self.preferences.snapshot()
        if preferences.auto_open_native_tensorboard:
            self.open_native()
        if preferences.auto_open_analysis_dashboard:
            self.open_analysis()
        return self.status()

    def start_analysis_service(self, models_root: Path) -> InspectionStatus:
        """Start the application-lifetime multi-model Analysis child process."""

        resolved_root = models_root.expanduser().resolve()
        if self._analysis_models_root == resolved_root and self._analysis_status.ready:
            return self.status()
        self._stop_analysis()
        self._analysis_models_root = resolved_root
        self._analysis_status = InspectionViewerStatus(
            InspectionViewer.ANALYSIS,
            InspectionLifecycleState.STARTING,
            target="all models",
        )
        self._start_analysis(models_root=resolved_root)
        if self.preferences.snapshot().auto_open_analysis_dashboard:
            self.open_analysis()
        return self.status()

    def stop_target(self, *, clear_target: bool = True) -> InspectionStatus:
        """Stop target-scoped viewers while retaining the application service."""

        target_label = self._target.display_name if self._target is not None else None
        self._stop_native(target_label)
        if self._analysis_models_root is None:
            self._stop_analysis(target_label)
        if clear_target:
            self._target = None
        return self.status()

    def stop(self, *, clear_target: bool = False) -> InspectionStatus:
        """Stop both server processes idempotently and retain truthful status."""

        target_label = self._target.display_name if self._target is not None else None
        self._stop_native(target_label)
        self._stop_analysis(target_label)
        self._analysis_models_root = None
        if clear_target:
            self._target = None
        return self.status()

    def _stop_native(self, target_label: str | None) -> None:
        """Stop only the native TensorBoard child and publish its state."""

        try:
            self._native.stop()
        except Exception as error:  # pragma: no cover - defensive adapter boundary
            self._logger.warning("Failed to stop native TensorBoard cleanly: %s", error)
        self._native_status = self._stopped(InspectionViewer.NATIVE, target_label)

    def _stop_analysis(self, target_label: str | None = None) -> None:
        """Stop only the Analysis child process and publish its state."""

        analysis_process = self._analysis_process
        self._analysis_process = None
        if analysis_process is not None:
            try:
                analysis_process.stop()
            except Exception as error:  # pragma: no cover - defensive adapter boundary
                self._logger.warning("Failed to stop Analysis Dashboard cleanly: %s", error)
        self._analysis_status = self._stopped(InspectionViewer.ANALYSIS, target_label)

    def status(self) -> InspectionStatus:
        """Return an immutable, independent status projection for both viewers."""

        return InspectionStatus(
            target=self._target,
            native=self._native_status,
            analysis=self._analysis_status,
        )

    def open_native(self) -> bool:
        """Open native TensorBoard manually when its server is ready."""

        return self._open(self._native_status)

    def open_analysis(self) -> bool:
        """Open the live Analysis Dashboard manually when its server is ready."""

        return self._open(self._analysis_status)

    def _open(self, status: InspectionViewerStatus) -> bool:
        if not status.ready or status.url is None:
            return False
        with self._browser_open_lock:
            if status.url in self._browser_opening:
                return True
            self._browser_opening.add(status.url)
            self._browser_open_queue.append(status.url)
            if self._browser_worker_running:
                return True
            self._browser_worker_running = True
        try:
            Thread(
                target=self._drain_browser_open_queue,
                name="lanistai-browser-opener",
                daemon=True,
            ).start()
        except RuntimeError as error:  # pragma: no cover - thread startup failure
            with self._browser_open_lock:
                queued_urls = tuple(self._browser_open_queue)
                self._browser_open_queue.clear()
                self._browser_opening.difference_update(queued_urls)
                self._browser_worker_running = False
            self._logger.warning("Failed to schedule inspection browser tab: %s", error)
            return False
        return True

    def _drain_browser_open_queue(self) -> None:
        """Process browser requests sequentially without occupying the UI thread."""

        while True:
            with self._browser_open_lock:
                if not self._browser_open_queue:
                    self._browser_worker_running = False
                    return
                url = self._browser_open_queue.popleft()
            self._open_browser(url)

    def _open_browser(self, url: str) -> None:
        """Run the platform browser adapter without blocking the Textual thread."""

        try:
            if not self._browser_opener(url):
                self._logger.warning("The platform browser did not accept %s", url)
        except Exception as error:  # pragma: no cover - platform browser adapter
            self._logger.warning("Failed to open inspection browser tab: %s", error)
        finally:
            with self._browser_open_lock:
                self._browser_opening.discard(url)

    def _start_native(self, target: InspectionTarget) -> None:
        port = self._native_port_allocator(6006)
        url = f"http://{self._HOST}:{port}/"
        target_label = target.display_name
        self._native.configure(
            logger=self._logger,
            executable=self._native_executable,
            subprocess_kwargs=self._subprocess_kwargs,
            port=str(port),
        )
        started = self._native.restart(
            target.native_logdir,
            online_context=target_label,
            offline_reason="Native TensorBoard has no readable target directory.",
            run_name=(
                Path(target.native_logdir).name
                if target.kind is InspectionTargetKind.PACK
                else None
            ),
        )
        if started and self._wait_until_ready(url):
            self._native_status = InspectionViewerStatus(
                InspectionViewer.NATIVE,
                InspectionLifecycleState.READY,
                target=target_label,
                url=url,
                port=port,
            )
            self._logger.info(
                "Native TensorBoard ONLINE at [link=%s]%s[/link]",
                url,
                url,
                extra={"markup": True},
            )
            return
        if started:
            self._native.stop()
        self._native_status = InspectionViewerStatus(
            InspectionViewer.NATIVE,
            InspectionLifecycleState.FAILED,
            target=target_label,
            port=port,
            error=(
                "Native TensorBoard did not become ready before the bounded startup deadline."
                if started
                else "Native TensorBoard failed to start."
            ),
        )

    def _start_analysis(
        self,
        target: InspectionTarget | None = None,
        *,
        models_root: Path | None = None,
    ) -> None:
        if (target is None) == (models_root is None):
            raise ValueError("Configure exactly one Analysis server source.")
        target_label = target.display_name if target is not None else "all models"
        script = (
            self._repository_root
            / ".ai"
            / "analysis-tools"
            / "scripts"
            / "project_inspection"
            / "tensorboard_metrics_server.py"
        )
        inspection_root = self._repository_root / ".ai" / "analysis-tools"
        if not script.is_file() or not inspection_root.is_dir():
            self._analysis_status = InspectionViewerStatus(
                InspectionViewer.ANALYSIS,
                InspectionLifecycleState.FAILED,
                target=target_label,
                error="The Analysis Dashboard server resources are unavailable.",
            )
            return
        port = self._ANALYSIS_PORT
        if not self._analysis_port_available(port):
            self._analysis_status = InspectionViewerStatus(
                InspectionViewer.ANALYSIS,
                InspectionLifecycleState.FAILED,
                target=target_label,
                port=port,
                error=(
                    f"The fixed Analysis Dashboard endpoint "
                    f"{self._HOST}:{port} is already in use."
                ),
            )
            return
        url = f"http://{self._HOST}:{port}/#tensorboard"
        command = self._analysis_command(
            target,
            models_root=models_root,
            script=script,
            inspection_root=inspection_root,
            port=port,
        )
        process = self._process_factory(command, **self._subprocess_kwargs)
        self._analysis_process = process
        started = process.start()
        health_url = f"http://{self._HOST}:{port}/healthz"
        if started and self._wait_until_ready(health_url):
            self._analysis_status = InspectionViewerStatus(
                InspectionViewer.ANALYSIS,
                InspectionLifecycleState.READY,
                target=target_label,
                url=url,
                port=port,
            )
            return
        if started:
            process.stop()
        self._analysis_process = None
        self._analysis_status = InspectionViewerStatus(
            InspectionViewer.ANALYSIS,
            InspectionLifecycleState.FAILED,
            target=target_label,
            port=port,
            error=(
                "Analysis Dashboard did not become ready before the bounded startup deadline."
                if started
                else "Analysis Dashboard failed to start."
            ),
        )

    @staticmethod
    def _analysis_command(
        target: InspectionTarget | None,
        *,
        models_root: Path | None,
        script: Path,
        inspection_root: Path,
        port: int,
    ) -> tuple[str, ...]:
        if (target is None) == (models_root is None):
            raise ValueError("Configure exactly one Analysis server source.")
        source_arguments = (
            ("--model-root", target.model_root)
            if target is not None
            else ("--models-root", str(models_root))
        )
        command = [
            sys.executable,
            str(script),
            *source_arguments,
            "--inspection-root",
            str(inspection_root),
            "--host",
            InspectionLifecycleCoordinator._HOST,
            "--port",
            str(port),
        ]
        if target is None:
            return tuple(command)
        command.extend(
            (
                "--scope-pack-type",
                target.catalog_pack_type,
                "--scope-framework",
                "all" if target.framework_uuid == "mixed" else target.framework_uuid,
            )
        )
        if target.selected_session:
            command.extend(("--initial-session", target.selected_session))
        if target.selected_checkpoint:
            command.extend(("--initial-checkpoint", target.selected_checkpoint))
        if target.kind is InspectionTargetKind.PACK and target.pack_name:
            command.extend(("--initial-pack-name", target.pack_name))
        return tuple(command)

    def _wait_until_ready(self, url: str) -> bool:
        deadline = time.monotonic() + self._startup_timeout_seconds
        while True:
            if self._readiness_probe(url):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    @staticmethod
    def _http_ready(url: str) -> bool:
        request = Request(url, method="HEAD")
        try:
            with urlopen(request, timeout=0.25) as response:  # noqa: S310 - fixed loopback URL
                return 200 <= int(response.status) < 500
        except HTTPError as error:
            return 200 <= int(error.code) < 500
        except (OSError, TimeoutError, URLError, ValueError):
            return False

    @classmethod
    def _available_port(cls, preferred: int) -> int:
        for requested in (int(preferred), 0):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
                    candidate.bind((cls._HOST, requested))
                    return int(candidate.getsockname()[1])
            except OSError:
                continue
        raise RuntimeError("No local inspection server port is available.")

    @classmethod
    def _fixed_port_available(cls, port: int) -> bool:
        """Return whether this application can own the configured fixed endpoint."""

        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
                candidate.bind((cls._HOST, int(port)))
                return True
        except OSError:
            return False

    @staticmethod
    def _stopped(
        viewer: InspectionViewer,
        target: str | None = None,
    ) -> InspectionViewerStatus:
        return InspectionViewerStatus(
            viewer,
            InspectionLifecycleState.STOPPED,
            target=target,
        )


__all__ = [
    "InspectionLifecycleCoordinator",
    "InspectionLifecycleState",
    "InspectionPreferences",
    "InspectionPreferencesService",
    "InspectionStatus",
    "InspectionViewer",
    "InspectionViewerStatus",
]
