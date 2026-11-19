# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Present the dialog for about."""

import logging
from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Markdown, Static

from lanista.gui.services.product_identity import (
    CONTRIBUTORS,
    COPYRIGHT_NOTICE,
    CREATORS,
    DEPARTMENT,
    PRODUCT_NAME,
    PRODUCT_VERSION,
    PROJECT_CREATION_YEARS,
    PROJECT_DESCRIPTION,
    RESEARCH_ADVISORS,
    RESEARCH_INSTITUTE,
    SZTAKI_LOGO,
)
from lanista.gui.widgets.modal_footer import ModalFooter
from lanista.locations import (
    get_about_dialog_css_path,
    get_about_dialog_md_path,
    get_app_config_path,
)

LOGGER = logging.getLogger(__name__)


class AboutDialog(ModalScreen[None]):
    """Present about in the Textual interface."""

    CSS_PATH = get_about_dialog_css_path()
    _FALLBACK_MARKDOWN = (
        f"# {PRODUCT_NAME}\n\n"
        f"{PROJECT_DESCRIPTION}\n\n"
        f"**Department:** {DEPARTMENT}\n\n"
        f"**Creators:**\n\n{'\n'.join(f'  - {name}' for name in CREATORS)}\n\n"
        f"**Research advisors:**\n\n"
        f"{'\n'.join(f'  - {name}' for name in RESEARCH_ADVISORS)}\n\n"
        f"**Contributors:**\n\n{'\n'.join(f'  - {name}' for name in CONTRIBUTORS)}\n\n"
        f"**Research institute:** {RESEARCH_INSTITUTE}\n\n"
        f"**Creation year:** {PROJECT_CREATION_YEARS}\n\n"
        f"**Version:** {PRODUCT_VERSION}"
    )

    BINDINGS = [
        Binding("escape,q,ctrl+c", "dismiss(None)", "Close about", key_display="esc"),
        Binding("f10,a,.", "dismiss(None)", "Close about", show=False, key_display="f10"),
    ]

    @classmethod
    def _load_markdown(cls) -> str:
        """Load the markdown owned by this about dialog."""
        try:
            template = Path(get_about_dialog_md_path()).read_text(encoding="utf-8").strip()
            if not template:
                LOGGER.warning("About dialog Markdown is empty; using fallback content.")
                return cls._FALLBACK_MARKDOWN

            payload = yaml.safe_load(Path(get_app_config_path()).read_text(encoding="utf-8")) or {}
            about = payload.get("about") if isinstance(payload, dict) else {}
            about = about if isinstance(about, dict) else {}
            creators = about.get("creators") or CREATORS
            advisors = about.get("research_advisors") or RESEARCH_ADVISORS
            contributors = about.get("contributors") or CONTRIBUTORS
            creators_text = cls._format_credits(creators, CREATORS)
            advisors_text = cls._format_credits(advisors, RESEARCH_ADVISORS)
            contributors_text = cls._format_credits(contributors, CONTRIBUTORS)
            institute = str(about.get("research_institute") or RESEARCH_INSTITUTE)
            creation_year = str(about.get("creation_year") or PROJECT_CREATION_YEARS)
            version = str(payload.get("version") or PRODUCT_VERSION)
            return (
                template.replace("{{PRODUCT_NAME}}", PRODUCT_NAME)
                .replace("{{PROJECT_DESCRIPTION}}", PROJECT_DESCRIPTION)
                .replace("{{DEPARTMENT}}", DEPARTMENT)
                .replace("{{CREATORS}}", creators_text)
                .replace("{{RESEARCH_ADVISORS}}", advisors_text)
                .replace("{{CONTRIBUTORS}}", contributors_text)
                .replace("{{RESEARCH_INSTITUTE}}", institute)
                .replace("{{CREATION_YEAR}}", creation_year)
                .replace("{{VERSION}}", version)
            )
        except (OSError, UnicodeError, yaml.YAMLError, TypeError) as exc:
            LOGGER.warning(
                "Unable to load About dialog content; using fallback content: %s",
                exc,
            )
        return cls._FALLBACK_MARKDOWN

    @staticmethod
    def _format_credits(value: object, fallback: tuple[str, ...]) -> str:
        """Render configured credits as an indented Markdown list."""
        if isinstance(value, str):
            items = (value.strip(),) if value.strip() else fallback
            return "\n".join(f"  - {item}" for item in items)
        if isinstance(value, list | tuple):
            items = tuple(str(item).strip() for item in value if str(item).strip()) or fallback
            return "\n".join(f"  - {item}" for item in items)
        return "\n".join(f"  - {item}" for item in fallback)

    @staticmethod
    def _branding_widgets() -> tuple[Static, Static]:
        """Build the centered branding footer in its visual order."""
        return (
            Static(SZTAKI_LOGO, id="about_sztaki_logo", markup=False),
            Static(COPYRIGHT_NOTICE, id="about_copyright", markup=False),
        )

    def compose(self) -> ComposeResult:
        """Compose the child elements owned by this about dialog."""
        with Vertical(id="about_dialog_container") as container:
            container.border_title = "LanistAI About"
            with VerticalScroll(id="about_dialog_content"):
                yield Markdown(self._load_markdown(), id="about_text")
                yield from self._branding_widgets()
            with Horizontal(id="about_dialog_actions"):
                yield Button("Close", id="about_dialog_cancel", variant="default")
            yield ModalFooter.from_binding_rows(
                self.BINDINGS,
                [["dismiss(None)"]],
                id="about_modal_footer",
                label_overrides={"dismiss(None)": "Close"},
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle the button pressed event for this about dialog."""
        if event.button.id == "about_dialog_cancel":
            self.dismiss(None)
