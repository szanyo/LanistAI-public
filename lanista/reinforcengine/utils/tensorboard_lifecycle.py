# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate the lifecycle of tensorboard."""

from __future__ import annotations

import os
import time

from lanista.foundation.subprocess import CustomSubprocess


class TensorboardLifecycle:
    """Coordinate the lifecycle of tensorboard."""

    _instance: TensorboardLifecycle | None = None

    def __new__(cls):
        """Return the new value owned by this tensorboard lifecycle."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the tensorboard lifecycle and its owned state."""
        if self._initialized:
            return
        self._initialized = True
        self._logger = None
        self._tensorboard_process: CustomSubprocess | None = None
        self._active_logdir: str | None = None
        self._executable: str = "tensorboard"
        self._subprocess_kwargs: dict = {}
        self._port: str = "6006"
        self._startup_probe_delay_s: float = 0.5

    def configure(self, *, logger, executable: str, subprocess_kwargs: dict, port: str = "6006") -> None:
        """Configure the operation owned by this tensorboard lifecycle."""
        self._logger = logger
        self._executable = executable
        self._subprocess_kwargs = dict(subprocess_kwargs or {})
        self._port = str(port)

    def get_active_logdir(self) -> str | None:
        """Return the active logdir exposed by this tensorboard lifecycle."""
        return self._active_logdir or os.environ.get("LANISTAI_TENSORBOARD_VIEW_LOGDIR") or None

    def get_url(self) -> str | None:
        """Return the loopback URL for the active native TensorBoard process."""

        if self.get_active_logdir() is None:
            return None
        return f"http://127.0.0.1:{self._port}/"

    def build_status_label(self, selected_session: str | None = None) -> str:
        """Build the status label required by this tensorboard lifecycle."""
        tensorboard_logdir = self.get_active_logdir()
        if not tensorboard_logdir:
            return "offline"
        if selected_session:
            run_name = os.path.basename(str(tensorboard_logdir).rstrip(os.sep))
            if run_name in {"ray", "sb3"}:
                return selected_session
            return run_name or selected_session
        return "model scope"

    def stop(self) -> None:
        """Stop the workflow owned by this tensorboard lifecycle."""
        os.environ.pop("LANISTAI_TENSORBOARD_VIEW_LOGDIR", None)
        self._active_logdir = None

        if self._tensorboard_process is None:
            return

        try:
            self._tensorboard_process.stop()
            if self._logger is not None:
                self._logger.info("TensorBoard OFFLINE.")
        except Exception as exc:
            if self._logger is not None:
                self._logger.warning(f"Failed to stop TensorBoard subprocess cleanly: {exc}")
        finally:
            self._tensorboard_process = None

    def restart(
        self,
        logdir: str | None,
        *,
        online_context: str,
        offline_reason: str,
        run_name: str | None = None,
    ) -> bool:
        """Restart the operation owned by this tensorboard lifecycle."""
        self.stop()

        if not logdir or not os.path.isdir(logdir):
            if self._logger is not None:
                self._logger.info(offline_reason)
            return False

        command = self._build_command(str(logdir), run_name)

        os.environ["LANISTAI_TENSORBOARD_VIEW_LOGDIR"] = str(logdir)
        self._active_logdir = str(logdir)

        self._tensorboard_process = CustomSubprocess(command, **self._subprocess_kwargs)
        if self._tensorboard_process.start():
            process_handle = getattr(self._tensorboard_process, "_process", None)
            if process_handle is not None:
                time.sleep(self._startup_probe_delay_s)
                if process_handle.poll() is not None:
                    os.environ.pop("LANISTAI_TENSORBOARD_VIEW_LOGDIR", None)
                    self._active_logdir = None
                    self._tensorboard_process = None
                    if self._logger is not None:
                        self._logger.warning("TensorBoard subprocess exited immediately after startup probe.")
                        self._logger.warning("TensorBoard OFFLINE.")
                    return False
            if self._logger is not None:
                self._logger.info(f"TensorBoard ONLINE for {online_context}: {logdir}")
            return True

        os.environ.pop("LANISTAI_TENSORBOARD_VIEW_LOGDIR", None)
        self._active_logdir = None
        self._tensorboard_process = None
        if self._logger is not None:
            self._logger.warning("TensorBoard subprocess failed to start.")
            self._logger.warning("TensorBoard OFFLINE.")
        return False

    def _build_command(self, logdir: str, run_name: str | None) -> list[str]:
        """Build a named single-run command or the ordinary recursive command."""

        normalized_run_name = str(run_name or "").strip()
        if normalized_run_name:
            safe_run_name = (
                normalized_run_name.replace("%", "%25")
                .replace(",", "%2C")
                .replace(":", "%3A")
            )
            return [
                self._executable,
                "--logdir_spec",
                f"{safe_run_name}:{logdir}",
                "--port",
                self._port,
            ]
        return [
            self._executable,
            "--logdir",
            logdir,
            "--port",
            self._port,
        ]
