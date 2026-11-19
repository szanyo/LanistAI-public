# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Provide the interface widget for byte editor."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label, Switch

from lanista.locations import get_bit_switch_css_path, get_byte_input_css_path, get_byte_editor_css_path

class BitSwitchWidget(Widget):
    """A Switch with a numeric label above it."""

    CSS_PATH = get_bit_switch_css_path()
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        DEFAULT_CSS = file.read()

    class BitChanged(Message):
        """Carry the bit changed interface message."""

        def __init__(self,
                     new_value: bool) -> None:
            """Initialize the bit changed and its owned state."""
            super().__init__()
            self.new_value = new_value

    value = reactive(0)

    def __init__(self,
                 bit: int) -> None:
        """Initialize the bit switch widget and its owned state."""
        self.bit = bit
        self._initializing = True
        super().__init__()

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this bit switch widget."""
        yield Label(str(self.bit))
        yield Switch()

    def on_mount(self) -> None:
        """Set the initial value of the switch based on the bit."""
        switch = self.query_one(Switch)
        # Az értéket a bit paraméter ellentétére állítjuk
        switch.value = not bool(self.bit)
        # Frissítsük a reaktív value tulajdonságot is, hogy összhangban legyen
        self.value = bool(switch.value)
        self._initializing = False

    def synced(self) -> bool:
        """Return the synced value owned by this bit switch widget."""
        return True if self.bit == self.value else False

    def watch_value(self, value: bool) -> None:
        """When the value changes we want to set the switch accordingly."""
        if not self._initializing:
            self.query_one(Switch).value = value

    def on_switch_changed(self, event: Switch.Changed) -> None:
        """When the switch changes, notify the parent via a message."""
        event.stop()
        if not self._initializing:
            self.value = event.value
            self.post_message(self.BitChanged(event.value))

class ByteInputWidget(Widget):
    """A compound widget with 8 switches."""

    CSS_PATH = get_byte_input_css_path()
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        DEFAULT_CSS = file.read()

    def __init__(self, bits: list[bool] | None = None) -> None:
        """Initialize with a list of bits or None to create all 8 switches."""
        super().__init__()
        self._bits = bits

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this byte input widget."""
        for bit in self._bits:
            yield BitSwitchWidget(bit)

class ByteEditorWidget(Widget):

    """Present byte editor in the Textual interface."""

    CSS_PATH = get_byte_editor_css_path()
    with open(CSS_PATH, "r", encoding="utf-8") as file:
        DEFAULT_CSS = file.read()

    def __init__(self,
                 bits: list[bool] | None = None,
                 unlocker=None,
                 **kwargs) -> None:
        """Initialize with a pin and an unlocker function."""
        super().__init__(**kwargs)
        self._bits = bits or [False] * 8
        self._unlocker = unlocker

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this byte editor widget."""
        with Container():
            yield ByteInputWidget(self._bits)

    def on_bit_switch_widget_bit_changed(self, event: BitSwitchWidget.BitChanged) -> None:
        """When a switch changes, update the value."""
        event.stop()
        locked = True
        for switch in self.query(BitSwitchWidget):
            if not switch.synced():
                locked = False
                break
        if locked and self._unlocker:
            self._unlocker()