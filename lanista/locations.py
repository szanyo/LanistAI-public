# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Resolve the application's canonical filesystem locations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import cast

from platformdirs import user_data_path

from lanista.configuration_paths import ConfigurationPaths
from lanista.foundation.configuration import AppConfiguration

APP_CONFIGURATION = AppConfiguration()
_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_HOME_ENVIRONMENT_VARIABLE = "LANISTAI_DATA_HOME"


@dataclass(frozen=True, slots=True)
class ApplicationPaths:
    """Resolved immutable resources and writable application locations."""

    package_root: Path
    data_home: Path
    resource_root: Path
    logs_root: Path
    runtime_state_path: Path
    legacy_runtime_state_path: Path
    model_storage_root: Path
    models_root: Path
    legacy_storage_active: bool


def _has_legacy_model_storage(package_root: Path) -> bool:
    """Return whether a source checkout contains an existing model workspace."""

    models_root = package_root / ConfigurationPaths.RL_MODEL_STORAGE_PATH / "models"
    try:
        projects = tuple(models_root.iterdir())
    except OSError:
        return False
    return any(
        project.is_dir()
        and any(
            (project / filename).is_file()
            for filename in ("model_manifest.yaml", "model_manifest.yml")
        )
        for project in projects
    )


def _is_source_checkout(package_root: Path) -> bool:
    """Return whether the package belongs to this repository checkout."""

    repository_root = package_root.parent
    return (repository_root / "pyproject.toml").is_file() and (
        repository_root / ".git"
    ).exists()


def resolve_application_paths(
    *,
    package_root: str | os.PathLike[str] = _PROJECT_ROOT,
    environment: Mapping[str, str] | None = None,
    platform_data_home: str | os.PathLike[str] | None = None,
) -> ApplicationPaths:
    """Resolve resource and runtime roots with explicit, platform, legacy precedence."""

    resolved_package_root = Path(package_root).resolve()
    selected_environment = os.environ if environment is None else environment
    configured_data_home = str(
        selected_environment.get(_DATA_HOME_ENVIRONMENT_VARIABLE, "") or ""
    ).strip()
    if configured_data_home:
        configured_path = Path(configured_data_home).expanduser()
        if not configured_path.is_absolute():
            raise ValueError("LANISTAI_DATA_HOME must be an absolute path.")
        selected_data_home = configured_path.resolve()
        legacy_storage_active = False
    else:
        default_data_home = Path(
            platform_data_home
            if platform_data_home is not None
            else user_data_path("LanistAI", appauthor=False, ensure_exists=False)
        ).resolve()
        platform_storage = (
            default_data_home / ConfigurationPaths.RUNTIME_MODEL_STORAGE_PATH
        )
        legacy_storage_active = (
            not platform_storage.exists()
            and _is_source_checkout(resolved_package_root)
            and _has_legacy_model_storage(resolved_package_root)
        )
        selected_data_home = (
            resolved_package_root if legacy_storage_active else default_data_home
        )

    legacy_runtime_state_path = (
        resolved_package_root / ConfigurationPaths.APP_RUNTIME_STATE_FILE_PATH
    )
    if legacy_storage_active:
        logs_root = resolved_package_root / ConfigurationPaths.LOGS_SAVE_PATH
        runtime_state_path = legacy_runtime_state_path
        model_storage_root = (
            resolved_package_root / ConfigurationPaths.RL_MODEL_STORAGE_PATH
        )
    else:
        logs_root = selected_data_home / ConfigurationPaths.RUNTIME_LOGS_PATH
        runtime_state_path = (
            selected_data_home / ConfigurationPaths.RUNTIME_STATE_FILE_PATH
        )
        model_storage_root = (
            selected_data_home / ConfigurationPaths.RUNTIME_MODEL_STORAGE_PATH
        )

    return ApplicationPaths(
        package_root=resolved_package_root,
        data_home=selected_data_home,
        resource_root=resolved_package_root / ConfigurationPaths.RESOURCES_PATH,
        logs_root=logs_root,
        runtime_state_path=runtime_state_path,
        legacy_runtime_state_path=legacy_runtime_state_path,
        model_storage_root=model_storage_root,
        models_root=model_storage_root / "models",
        legacy_storage_active=legacy_storage_active,
    )


@lru_cache(maxsize=1)
def get_application_paths() -> ApplicationPaths:
    """Return the process-wide resolved application paths."""

    return resolve_application_paths()


def reset_application_paths_cache() -> None:
    """Forget the cached paths after an explicit environment reconfiguration."""

    get_application_paths.cache_clear()


def get_data_home_path() -> str:
    """Return the selected writable application data home."""

    return str(get_application_paths().data_home)


def get_logs_path() -> str:
    """Return the selected writable application log directory."""

    return str(get_application_paths().logs_root)


def get_models_root_path() -> str:
    """Return the selected writable model project directory."""

    return str(get_application_paths().models_root)


def get_rl_model_storage_path() -> str:
    """Return the rl model storage path exposed by this locations module."""
    return str(get_application_paths().model_storage_root)


def get_app_runtime_state_path() -> str:
    """Return the app runtime state path exposed by this locations module."""
    return str(get_application_paths().runtime_state_path)


def get_legacy_app_runtime_state_path() -> str:
    """Return the repository-local runtime-state path used before external storage."""

    return str(get_application_paths().legacy_runtime_state_path)


def get_app_config_path() -> str:
    """Return the app config path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.APP_CONFIGURATION_FILE_PATH)


def get_app_logging_config_path() -> str:
    """Return the immutable packaged logging configuration path."""

    return str(_PROJECT_ROOT / ConfigurationPaths.APP_LOGGING_FILE_PATH)


def get_model_templates_path() -> str:
    """Return the root containing immutable framework scaffold templates."""

    return str(_PROJECT_ROOT / ConfigurationPaths.MODEL_TEMPLATES_PATH)


def get_splash_configuration_path() -> str:
    """Return the splash-effects configuration independently of the process CWD."""

    return str(_PROJECT_ROOT / ConfigurationPaths.SPLASH_CONFIGURATION_FILE_PATH)


def get_app_css_path() -> str:
    """Return the app css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.APP_CSS_CONFIGURATION_FILE_PATH)


def get_yes_no_dialog_css_path() -> str:
    """Return the yes no dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.YES_NO_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_model_creation_screen_css_path() -> str:
    """Return the model creation screen css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.MODEL_CREATION_SCREEN_CSS_CONFIGURATION_FILE_PATH)


def get_home_screen_css_path() -> str:
    """Return the home screen css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.HOME_SCREEN_CSS_CONFIGURATION_FILE_PATH)


def get_byte_editor_css_path() -> str:
    """Return the byte editor css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.BYTE_EDITOR_WIDGET_CSS_CONFIGURATION_FILE_PATH)


def get_bit_switch_css_path() -> str:
    """Return the bit switch css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.BIT_SWITCH_WIDGET_CSS_CONFIGURATION_FILE_PATH)


def get_byte_input_css_path() -> str:
    """Return the byte input css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.BYTE_INPUT_WIDGET_CSS_CONFIGURATION_FILE_PATH)


def get_secure_yes_no_dialog_css_path() -> str:
    """Return the secure yes no dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.SECURE_YES_NO_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_configuration_dialog_css_path() -> str:
    """Return the configuration dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_load_model_dialog_css_path() -> str:
    """Return the load model dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.LOAD_MODEL_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_session_switch_dialog_css_path() -> str:
    """Return the session switch dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.SESSION_SWITCH_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_session_configuration_dialog_css_path() -> str:
    """Return the session configuration dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.SESSION_CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_checkpoint_configuration_dialog_css_path() -> str:
    """Return the checkpoint configuration dialog css path exposed by this locations module."""
    return str(
        _PROJECT_ROOT
        / ConfigurationPaths.CHECKPOINT_CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH
    )


def get_model_clone_selection_dialog_css_path() -> str:
    """Return the model clone selection dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.MODEL_CLONE_SELECTION_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_model_detail_edit_dialog_css_path() -> str:
    """Return the model detail edit dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.MODEL_DETAIL_EDIT_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_session_detail_edit_dialog_css_path() -> str:
    """Return the session detail edit dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.SESSION_DETAIL_EDIT_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_new_session_strategy_dialog_css_path() -> str:
    """Return the new session strategy dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.NEW_SESSION_STRATEGY_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_help_dialog_css_path() -> str:
    """Return the help dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.HELP_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_about_dialog_css_path() -> str:
    """Return the about dialog css path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.ABOUT_DIALOG_CSS_CONFIGURATION_FILE_PATH)


def get_help_dialog_md_path() -> str:
    """Return the help dialog md path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.HELP_DIALOG_MD_CONFIGURATION_FILE_PATH)


def get_about_dialog_md_path() -> str:
    """Return the about dialog md path exposed by this locations module."""
    return str(_PROJECT_ROOT / ConfigurationPaths.ABOUT_DIALOG_MD_CONFIGURATION_FILE_PATH)


def get_random_route_generator_script_file_path() -> str:
    """Return the random route generator script file path exposed by this locations module."""
    return cast(
        str,
        os.path.join(
            APP_CONFIGURATION.sumo_home, APP_CONFIGURATION.random_route_generator_script_file_path
        ),
    )


def get_tensorboard_logs_path() -> str:
    """Return the tensorboard logs path exposed by this locations module."""
    override_path = os.environ.get("LANISTAI_TENSORBOARD_LOGDIR")
    if override_path:
        return override_path
    return str(
        get_application_paths().model_storage_root
        / os.path.basename(ConfigurationPaths.TENSORBOARD_LOGS_PATH)
    )
