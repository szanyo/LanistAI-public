# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Request best-effort terminal window presentation changes."""

from __future__ import annotations

import ctypes
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from ctypes.util import find_library
from typing import Any, Protocol

type _CommandResult = subprocess.CompletedProcess[str]
type _CommandRunner = Callable[..., _CommandResult]

_GNOME_TERMINAL_BUS_NAME = "org.gnome.Terminal"
_GNOME_TERMINAL_WINDOW_ROOT = "/org/gnome/Terminal/window"
_GNOME_WINDOW_NODE_PATTERN = re.compile(r"^\s*node\s+(\d+)\s*\{", re.MULTILINE)
_GNOME_MAXIMIZED_TERMINAL_SIZE = (500, 200)

_CLIENT_MESSAGE = 33
_SUBSTRUCTURE_NOTIFY_MASK = 1 << 19
_SUBSTRUCTURE_REDIRECT_MASK = 1 << 20
_EWMH_ADD_STATE = 1
_EWMH_SOURCE_APPLICATION = 1


class _XClientMessageData(ctypes.Union):
    """Represent the payload union used by an X11 client message."""

    _fields_ = [
        ("b", ctypes.c_byte * 20),
        ("s", ctypes.c_short * 10),
        ("l", ctypes.c_long * 5),
    ]


class _XClientMessageEvent(ctypes.Structure):
    """Represent the X11 client-message event fields used for EWMH."""

    _fields_ = [
        ("type", ctypes.c_int),
        ("serial", ctypes.c_ulong),
        ("send_event", ctypes.c_int),
        ("display", ctypes.c_void_p),
        ("window", ctypes.c_ulong),
        ("message_type", ctypes.c_ulong),
        ("format", ctypes.c_int),
        ("data", _XClientMessageData),
    ]


class _XEvent(ctypes.Union):
    """Allocate storage compatible with XEvent while exposing client messages."""

    _fields_ = [
        ("xclient", _XClientMessageEvent),
        ("pad", ctypes.c_long * 24),
    ]


class _TerminalStream(Protocol):
    """Describe the small output-stream surface needed by the adapter."""

    def isatty(self) -> bool:
        """Return whether the stream is connected to an interactive terminal."""

    def write(self, text: str) -> object:
        """Write a terminal control sequence."""

    def flush(self) -> object:
        """Flush a terminal control sequence."""


def _load_x11_library() -> Any | None:
    """Load and type the small Xlib surface needed for EWMH maximization."""

    library_name = find_library("X11")
    if library_name is None:
        return None
    try:
        x11 = ctypes.CDLL(library_name)
    except OSError:
        return None

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XInternAtom.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    x11.XInternAtom.restype = ctypes.c_ulong
    x11.XGetWindowProperty.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_long,
        ctypes.c_long,
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.c_ulong),
        ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
    ]
    x11.XGetWindowProperty.restype = ctypes.c_int
    x11.XFree.argtypes = [ctypes.c_void_p]
    x11.XFree.restype = ctypes.c_int
    x11.XSendEvent.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_int,
        ctypes.c_long,
        ctypes.POINTER(_XEvent),
    ]
    x11.XSendEvent.restype = ctypes.c_int
    x11.XFlush.argtypes = [ctypes.c_void_p]
    x11.XFlush.restype = ctypes.c_int
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int
    return x11


def _active_x11_window(x11: Any, display: object, root_window: int) -> int | None:
    """Return the EWMH active window, releasing Xlib-owned property memory."""

    active_window_atom = x11.XInternAtom(display, b"_NET_ACTIVE_WINDOW", 0)
    if not active_window_atom:
        return None

    actual_type = ctypes.c_ulong()
    actual_format = ctypes.c_int()
    item_count = ctypes.c_ulong()
    bytes_after = ctypes.c_ulong()
    property_data = ctypes.POINTER(ctypes.c_ubyte)()
    status = x11.XGetWindowProperty(
        display,
        root_window,
        active_window_atom,
        0,
        1,
        0,
        0,
        ctypes.byref(actual_type),
        ctypes.byref(actual_format),
        ctypes.byref(item_count),
        ctypes.byref(bytes_after),
        ctypes.byref(property_data),
    )
    if status != 0 or item_count.value < 1 or not property_data:
        return None
    try:
        return int(ctypes.cast(property_data, ctypes.POINTER(ctypes.c_ulong))[0])
    finally:
        x11.XFree(ctypes.cast(property_data, ctypes.c_void_p))


def _send_x11_maximize_event(
    x11: Any,
    display: object,
    root_window: int,
    terminal_window: int,
) -> bool:
    """Send one EWMH maximize request for the resolved terminal window."""

    state_atom = x11.XInternAtom(display, b"_NET_WM_STATE", 0)
    vertical_atom = x11.XInternAtom(
        display,
        b"_NET_WM_STATE_MAXIMIZED_VERT",
        0,
    )
    horizontal_atom = x11.XInternAtom(
        display,
        b"_NET_WM_STATE_MAXIMIZED_HORZ",
        0,
    )
    if not state_atom or not vertical_atom or not horizontal_atom:
        return False

    event = _XEvent()
    event.xclient.type = _CLIENT_MESSAGE
    event.xclient.serial = 0
    event.xclient.send_event = 1
    event.xclient.display = display
    event.xclient.window = terminal_window
    event.xclient.message_type = state_atom
    event.xclient.format = 32
    event.xclient.data.l[0] = _EWMH_ADD_STATE
    event.xclient.data.l[1] = vertical_atom
    event.xclient.data.l[2] = horizontal_atom
    event.xclient.data.l[3] = _EWMH_SOURCE_APPLICATION
    event.xclient.data.l[4] = 0
    sent = x11.XSendEvent(
        display,
        root_window,
        0,
        _SUBSTRUCTURE_NOTIFY_MASK | _SUBSTRUCTURE_REDIRECT_MASK,
        ctypes.byref(event),
    )
    if not sent:
        return False
    x11.XFlush(display)
    return True


def _request_x11_terminal_maximize(environment: Mapping[str, str]) -> bool:
    """Ask an EWMH-compatible X11 window manager to maximize the terminal."""

    display_name = environment.get("DISPLAY")
    session_type = environment.get("XDG_SESSION_TYPE", "").casefold()
    if not display_name or session_type not in {"", "x11"}:
        return False

    x11 = None
    display = None
    try:
        x11 = _load_x11_library()
        if x11 is None:
            return False

        display = x11.XOpenDisplay(display_name.encode())
        if not display:
            return False

        root_window = int(x11.XDefaultRootWindow(display))
        try:
            terminal_window = int(environment.get("WINDOWID", ""), 0)
        except ValueError:
            terminal_window = 0
        if not terminal_window:
            terminal_window = _active_x11_window(x11, display, root_window) or 0
        if not terminal_window:
            return False

        return _send_x11_maximize_event(
            x11,
            display,
            root_window,
            terminal_window,
        )
    except (
        AttributeError,
        ctypes.ArgumentError,
        OSError,
        OverflowError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    finally:
        if x11 is not None and display:
            with suppress(
                AttributeError,
                ctypes.ArgumentError,
                OSError,
                OverflowError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                x11.XCloseDisplay(display)


def _run_command(
    command: Sequence[str],
    command_runner: _CommandRunner,
) -> _CommandResult | None:
    """Run one bounded presentation command without exposing startup to errors."""

    try:
        return command_runner(
            list(command),
            capture_output=True,
            text=True,
            timeout=1,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _find_gnome_terminal_window(
    environment: Mapping[str, str],
    command_runner: _CommandRunner,
) -> tuple[str, str] | None:
    """Return the unambiguous GNOME Terminal service and window object path."""

    if not environment.get("GNOME_TERMINAL_SCREEN"):
        return None

    service = environment.get(
        "GNOME_TERMINAL_SERVICE",
        _GNOME_TERMINAL_BUS_NAME,
    )
    inspection = _run_command(
        (
            "gdbus",
            "introspect",
            "--session",
            "--dest",
            service,
            "--object-path",
            _GNOME_TERMINAL_WINDOW_ROOT,
        ),
        command_runner,
    )
    if inspection is None or inspection.returncode != 0:
        return None

    window_nodes = _GNOME_WINDOW_NODE_PATTERN.findall(inspection.stdout)
    if len(window_nodes) != 1:
        return None

    window_path = f"{_GNOME_TERMINAL_WINDOW_ROOT}/{window_nodes[0]}"
    return service, window_path


def _activate_gnome_terminal_action(
    service: str,
    window_path: str,
    action: str,
    parameters: str,
    command_runner: _CommandRunner,
) -> bool:
    """Activate one native GNOME Terminal window action through D-Bus."""

    activation = _run_command(
        (
            "gdbus",
            "call",
            "--session",
            "--dest",
            service,
            "--object-path",
            window_path,
            "--method",
            "org.gtk.Actions.Activate",
            action,
            parameters,
            "{}",
        ),
        command_runner,
    )
    return activation is not None and activation.returncode == 0


def _request_gnome_terminal_maximize(
    environment: Mapping[str, str],
    command_runner: _CommandRunner,
) -> bool:
    """Grow a GNOME Terminal window to the Wayland compositor's size limit."""

    target = _find_gnome_terminal_window(environment, command_runner)
    if target is None:
        return False
    columns, rows = _GNOME_MAXIMIZED_TERMINAL_SIZE
    parameters = f"[<(uint32 {columns}, uint32 {rows})>]"
    return _activate_gnome_terminal_action(
        *target,
        "size-to",
        parameters,
        command_runner,
    )


def _request_gnome_terminal_fullscreen(
    environment: Mapping[str, str],
    command_runner: _CommandRunner,
) -> bool:
    """Enter fullscreen through GNOME Terminal's supported Wayland D-Bus action."""

    target = _find_gnome_terminal_window(environment, command_runner)
    if target is None:
        return False
    return _activate_gnome_terminal_action(
        *target,
        "enter-fullscreen",
        "[]",
        command_runner,
    )


def _request_terminal_window_operation(
    sequence: str,
    stream: _TerminalStream | None,
) -> bool:
    """Send one xterm-compatible window operation as a safe fallback."""

    output = stream or sys.stdout
    try:
        if not output.isatty():
            return False
        output.write(sequence)
        output.flush()
    except (AttributeError, OSError, RuntimeError, ValueError):
        return False
    return True


def _has_terminal_window_target(
    environment: Mapping[str, str],
    stream: _TerminalStream | None,
) -> bool:
    """Return whether X11 can safely identify this process's terminal window."""

    window_id = environment.get("WINDOWID")
    if window_id:
        try:
            if int(window_id, 0) > 0:
                return True
        except ValueError:
            pass

    output = stream or sys.stdout
    try:
        return output.isatty()
    except (AttributeError, OSError, RuntimeError, ValueError):
        return False


def request_terminal_maximize(
    stream: _TerminalStream | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    command_runner: _CommandRunner = subprocess.run,
) -> bool:
    """Ask a compatible terminal to maximize its window without failing startup.

    X11 uses the window manager's EWMH maximize state. GNOME Terminal on
    Wayland falls back to its native character-grid action. Other terminals
    receive the xterm-compatible sequence as a final best-effort fallback.
    """

    current_environment = os.environ if environment is None else environment
    if _has_terminal_window_target(
        current_environment,
        stream,
    ) and _request_x11_terminal_maximize(current_environment):
        return True
    if _request_gnome_terminal_maximize(current_environment, command_runner):
        return True
    return _request_terminal_window_operation("\x1b[9;1t", stream)


def request_terminal_fullscreen(
    stream: _TerminalStream | None = None,
    *,
    environment: Mapping[str, str] | None = None,
    command_runner: _CommandRunner = subprocess.run,
) -> bool:
    """Ask a compatible terminal to enter fullscreen without failing startup."""

    current_environment = os.environ if environment is None else environment
    if _request_gnome_terminal_fullscreen(current_environment, command_runner):
        return True
    return _request_terminal_window_operation("\x1b[10;1t", stream)


__all__ = ["request_terminal_fullscreen", "request_terminal_maximize"]
