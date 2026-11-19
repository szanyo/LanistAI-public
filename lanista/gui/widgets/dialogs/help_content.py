# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Assemble immutable contextual and central Help documents."""

from dataclasses import dataclass
from pathlib import Path

from lanista.locations import get_help_dialog_md_path


@dataclass(frozen=True)
class HelpTopic:
    """Describe one dialog-specific Help document."""

    topic_id: str
    title: str
    resource_name: str


@dataclass(frozen=True)
class HelpDocument:
    """Provide a rendered Help title and Markdown body."""

    title: str
    markdown: str


HELP_TOPICS: tuple[HelpTopic, ...] = (
    HelpTopic("configuration", "Configure", "help_configuration.md"),
    HelpTopic("model_context", "View Model Context", "help_model_context.md"),
    HelpTopic("model_detail_edit", "Edit Model Details", "help_model_detail_edit.md"),
    HelpTopic("load_model", "Load Model", "help_load_model.md"),
    HelpTopic("model_creation", "New Model", "help_model_creation.md"),
    HelpTopic("edit_target", "Choose Edit Target", "help_edit_target.md"),
    HelpTopic("lineage_detail", "Lineage", "help_lineage_detail.md"),
    HelpTopic("model_clone", "Clone Selection", "help_model_clone.md"),
    HelpTopic("session_configuration", "Session Configuration", "help_session_configuration.md"),
    HelpTopic("session_detail_edit", "Edit Session Details", "help_session_detail_edit.md"),
    HelpTopic("session_switch", "Switch Training Session", "help_session_switch.md"),
    HelpTopic("new_session", "New Session", "help_new_session.md"),
    HelpTopic("checkpoint_configuration", "Checkpoint Configuration", "help_checkpoint_configuration.md"),
    HelpTopic("inspect", "Inspect TensorBoard", "help_inspect.md"),
    HelpTopic("comparison_pack", "Comparison Packs", "help_comparison_pack.md"),
    HelpTopic("settings", "Settings", "help_settings.md"),
)

_TOPICS_BY_ID = {topic.topic_id: topic for topic in HELP_TOPICS}
_GENERAL_FALLBACK = "## Help\n\nGeneral Help content could not be loaded."


def help_topic_ids() -> frozenset[str]:
    """Return every registered contextual Help topic identifier."""

    return frozenset(_TOPICS_BY_ID)


def _read_resource(path: Path, fallback: str) -> str:
    """Read one immutable Help resource with a visible fallback."""

    try:
        content = path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback
    return content or fallback


def _topic_markdown(topic: HelpTopic, resource_root: Path) -> str:
    """Load one contextual Help topic from the packaged resource directory."""

    fallback = f"## {topic.title}\n\nHelp content for this dialog could not be loaded."
    return _read_resource(resource_root / topic.resource_name, fallback)


def build_help_document(topic_id: str | None = None) -> HelpDocument:
    """Build either one contextual topic or the combined central Help document."""

    general_path = Path(get_help_dialog_md_path())
    resource_root = general_path.parent
    if topic_id is not None:
        topic = _TOPICS_BY_ID.get(topic_id)
        if topic is None:
            raise ValueError(f"Unknown Help topic: {topic_id!r}.")
        return HelpDocument(
            title=f"{topic.title} Help",
            markdown=_topic_markdown(topic, resource_root),
        )

    general = _read_resource(general_path, _GENERAL_FALLBACK)
    index = "## Dialog Help Index\n\n" + "\n".join(
        f"- [{topic.title}](#{topic.title.lower().replace(' ', '-')})"
        for topic in HELP_TOPICS
    )
    sections = [_topic_markdown(topic, resource_root) for topic in HELP_TOPICS]
    return HelpDocument(
        title="LanistAI Help",
        markdown="\n\n---\n\n".join((general, index, *sections)),
    )


__all__ = [
    "HELP_TOPICS",
    "HelpDocument",
    "HelpTopic",
    "build_help_document",
    "help_topic_ids",
]
