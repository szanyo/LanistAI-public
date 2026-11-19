# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the screen for splash."""

import inspect
import random
import time
from collections.abc import Callable
from enum import Enum
from importlib import import_module
from typing import Protocol, cast

import pyfiglet
import terminaltexteffects.utils.easing
import yaml
from terminaltexteffects.utils.graphics import Color, Gradient
from textual.screen import Screen

from lanista.gui.screens.home_screen import HomeScreen
from lanista.gui.services.product_identity import (
    COPYRIGHT_NOTICE,
    DEPARTMENT,
    PRODUCT_NAME,
    PROJECT_DESCRIPTION,
    PROJECT_START_YEAR,
    SZTAKI_LOGO,
)
from lanista.gui.utils.splash_screen_formatters import (
    center_multiline_text,
    center_text_block,
    center_text_vertically,
    create_empty_line,
    remove_blank_lines_from_end,
)
from lanista.locations import get_splash_configuration_path


class _SplashScreenFactory(Protocol):
    """Typed boundary for the third-party package that does not ship type metadata."""

    def __call__(
        self,
        text: str,
        *,
        effect: str,
        config: dict[str, object],
    ) -> Screen[None]:
        """Invoke this splash screen factory as a callable."""
        ...


_SPLASH_SCREEN_FACTORY = cast(
    _SplashScreenFactory,
    import_module("textualeffects.widgets").SplashScreen,
)


class Colors(Enum):
    """Enumerate colors used by splash-screen presentation."""

    DARK_BLUE = Color("#034991")
    LIGHT_BLUE = Color("#148fff")
    BLUISH_GREY = Color("#deeaf2")
    GREY = Color("#A0A0A0")


EasingFunction = Callable[[float], float]
EASING_FUNCTIONS: dict[str, EasingFunction] = {
    name: cast(EasingFunction, function)
    for name, function in inspect.getmembers(
        terminaltexteffects.utils.easing,
        inspect.isfunction,
    )
}

MIN_TERMINAL_WIDTH = 120
MAX_VHS_TAPE_CANVAS_AREA = 12_000
LARGE_CANVAS_EXCLUDED_EFFECTS = frozenset({"VHSTape"})

TITLE_TEXT = PRODUCT_NAME
DESCRIPTION = PROJECT_DESCRIPTION
START_YEAR_OF_DEVELOPMENT = PROJECT_START_YEAR
COPYRIGHT = COPYRIGHT_NOTICE


def _resolve_splash_width(width: object) -> int:
    """Return a safe splash canvas width with the legacy minimum preserved."""

    try:
        requested_width = int(width)
    except (TypeError, ValueError):
        requested_width = MIN_TERMINAL_WIDTH
    return max(MIN_TERMINAL_WIDTH, requested_width)


def _resolve_splash_height(height: object, content_height: int) -> int:
    """Return a safe canvas height that never truncates splash content."""

    try:
        requested_height = int(height)
    except (TypeError, ValueError):
        requested_height = content_height
    return max(content_height, requested_height)


def generate_splash_art(
    width: int = MIN_TERMINAL_WIDTH,
    height: int | None = None,
) -> str:
    """Return the generate splash art value owned by this splash screen module."""
    canvas_width = _resolve_splash_width(width)
    ascii_art = pyfiglet.figlet_format(TITLE_TEXT, font="ansi_shadow", width=canvas_width)
    ascii_art = remove_blank_lines_from_end(ascii_art)

    ascii_art = center_multiline_text(ascii_art, canvas_width)
    centered_description = center_multiline_text(DESCRIPTION, canvas_width)
    centered_department = center_multiline_text(DEPARTMENT, canvas_width)
    centered_logo = center_text_block(
        SZTAKI_LOGO,
        canvas_width,
        horizontal_offset=-1,
    )
    centered_copyright = center_multiline_text(COPYRIGHT, canvas_width)

    splash_art = create_empty_line(canvas_width)
    splash_art += ascii_art
    splash_art += create_empty_line(canvas_width)
    splash_art += centered_description
    splash_art += create_empty_line(canvas_width)
    splash_art += centered_department
    splash_art += create_empty_line(canvas_width)
    splash_art += centered_logo
    splash_art += create_empty_line(canvas_width)
    splash_art += centered_copyright
    splash_art += create_empty_line(canvas_width)
    content_height = len(splash_art.splitlines())
    canvas_height = _resolve_splash_height(height, content_height)
    return center_text_vertically(splash_art, canvas_width, canvas_height)


def _select_splash_effect(
    effects: dict[str, dict[str, object]],
    canvas_width: int,
    canvas_height: int,
) -> tuple[str, dict[str, object]]:
    """Choose an effect while avoiding VHS noise on large canvases."""

    candidates = list(effects.items())
    if canvas_width * canvas_height > MAX_VHS_TAPE_CANVAS_AREA:
        filtered_candidates = [
            candidate
            for candidate in candidates
            if candidate[0] not in LARGE_CANVAS_EXCLUDED_EFFECTS
        ]
        if filtered_candidates:
            candidates = filtered_candidates
    return random.choice(candidates)


def load_splash_art_effects() -> dict[str, dict[str, object]]:
    """Load the splash art effects owned by this splash screen module."""
    configuration_path = get_splash_configuration_path()
    with open(configuration_path, encoding="utf-8") as f:
        raw_config: object = yaml.safe_load(f.read())

    if raw_config is None:
        raise ValueError(
            "Configuration file is empty. Please provide a valid "
            f"configuration file. {configuration_path}"
        )

    converted = recursive_enum_converter(raw_config)
    if not isinstance(converted, dict):
        raise ValueError(f"Splash configuration must be a mapping: {configuration_path}")

    effects: dict[str, dict[str, object]] = {}
    for effect_name, effect_config in cast(dict[object, object], converted).items():
        if not isinstance(effect_name, str) or not isinstance(effect_config, dict):
            raise ValueError(
                "Each splash effect must have a string name and mapping configuration: "
                f"{configuration_path}"
            )
        effects[effect_name] = _string_keyed_mapping(effect_config, configuration_path)
    return effects


def _string_keyed_mapping(value: object, source: str) -> dict[str, object]:
    """Return the string keyed mapping value owned by this splash screen module."""
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in splash configuration: {source}")

    result: dict[str, object] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise ValueError(f"Splash configuration keys must be strings: {source}")
        result[key] = item
    return result


def detect_and_convert_enum(value: object) -> object:
    """Resolve only the explicitly supported color, direction and easing tokens."""

    if not isinstance(value, str):
        return value
    if value in Colors.__members__:
        return Colors[value].value
    if value in Gradient.Direction.__members__:
        return Gradient.Direction[value]
    return EASING_FUNCTIONS.get(value, value)


def recursive_enum_converter(value: object) -> object:
    """Return the recursive enum converter value owned by this splash screen module."""
    if isinstance(value, dict):
        converted: dict[str, object] = {}
        for key, item in cast(dict[object, object], value).items():
            if not isinstance(key, str):
                raise ValueError("Splash configuration keys must be strings.")
            converted[key] = recursive_enum_converter(item)
        return converted
    if isinstance(value, list):
        return [recursive_enum_converter(item) for item in cast(list[object], value)]
    return detect_and_convert_enum(value)


class SplashArtScreen(Screen[None]):
    """Present splash art in the Textual interface."""

    _SPLASH_ART = generate_splash_art()
    _EFFECTS = load_splash_art_effects()

    async def on_mount(self) -> None:
        """Handle the mount event for this splash art screen."""
        viewport = getattr(self.app, "size", None)
        viewport_width = getattr(viewport, "width", MIN_TERMINAL_WIDTH)
        viewport_height = getattr(viewport, "height", None)
        splash_art = generate_splash_art(viewport_width, viewport_height)
        splash_lines = splash_art.splitlines()
        random_effect_name, random_effect_config = _select_splash_effect(
            SplashArtScreen._EFFECTS,
            max((len(line) for line in splash_lines), default=0),
            len(splash_lines),
        )

        splash_screen = _SPLASH_SCREEN_FACTORY(
            splash_art, effect=random_effect_name, config=random_effect_config
        )
        await self.app.push_screen(splash_screen, callback=self._do_stuff)

    def _do_stuff(self, _message: None) -> None:
        """Handle the do stuff operation owned by this splash art screen."""
        time.sleep(3.5)
        self.app.push_screen(HomeScreen())
