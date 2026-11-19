# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Managed subprocess with line-oriented logging."""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Sequence
from typing import IO

from lanista.foundation.logging import LoggerInterface


class CustomSubprocess(LoggerInterface):
    """Own a child process and drain both output pipes without deadlocks."""

    def __init__(
        self,
        command: Sequence[str],
        stdout_level: int = logging.INFO,
        stderr_level: int = logging.INFO,
        **kwargs: object,
    ) -> None:
        """Initialize the custom subprocess and its owned state."""
        super().__init__(**kwargs)
        self._command = tuple(command)
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stdout_level = stdout_level
        self._stderr_level = stderr_level

    def _log_output(self, pipe: IO[str], level: int) -> None:
        """Log the output owned by this custom subprocess."""
        try:
            for line in iter(pipe.readline, ""):
                self._logger.log(level, line.rstrip())
        except Exception:
            self._logger.exception("Failed while reading subprocess output.")
        finally:
            pipe.close()

    def start(self) -> bool:
        """Start the workflow owned by this custom subprocess."""
        if self._process is not None:
            return False
        try:
            process = subprocess.Popen(
                self._command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError:
            self._logger.exception("Failed to start subprocess: %s", self._command)
            return False
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            raise RuntimeError("Subprocess pipes were not created.")
        self._process = process
        self._stdout_thread = self._start_reader(process.stdout, self._stdout_level, "stdout")
        self._stderr_thread = self._start_reader(process.stderr, self._stderr_level, "stderr")
        return True

    def _start_reader(self, pipe: IO[str], level: int, stream_name: str) -> threading.Thread:
        """Start the reader workflow owned by this custom subprocess."""
        thread = threading.Thread(
            target=self._log_output,
            args=(pipe, level),
            name=f"lanista-subprocess-{stream_name}",
            daemon=True,
        )
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop the workflow owned by this custom subprocess."""
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5.0)
        finally:
            for thread in (self._stdout_thread, self._stderr_thread):
                if thread is not None:
                    thread.join(timeout=5.0)
            self._process = None
            self._stdout_thread = None
            self._stderr_thread = None
        self._logger.info("Subprocess terminated and output readers finished.")
