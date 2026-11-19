# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Resolve filesystem paths for configuration."""

import os


class ConfigurationPaths:
    # Data root

    """Define canonical paths for application and interface configuration."""

    DATA_PATH = os.path.join("data")  # */data (legacy writable checkout data)
    RESOURCES_PATH = os.path.join("resources")  # */resources
    DEFAULTS_PATH = os.path.join(RESOURCES_PATH, "defaults")  # */resources/defaults
    MODEL_TEMPLATES_PATH = os.path.join(
        RESOURCES_PATH,
        "model_templates",
    )  # */resources/model_templates

    # Writable runtime data relative to the selected application data home.

    RUNTIME_CONFIGURATION_PATH = os.path.join("config")
    RUNTIME_STATE_FILE_PATH = os.path.join(
        RUNTIME_CONFIGURATION_PATH,
        "runtime_state.yaml",
    )
    RUNTIME_LOGS_PATH = os.path.join("logs")
    RUNTIME_MODEL_STORAGE_PATH = os.path.join("rl_model_storage")
    RUNTIME_MODELS_PATH = os.path.join(RUNTIME_MODEL_STORAGE_PATH, "models")

    # Packaged immutable defaults and legacy writable configuration roots.

    CONFIGURATION_PATH = DEFAULTS_PATH
    LEGACY_CONFIGURATION_PATH = os.path.join(
        DATA_PATH,
        "configurations",
    )  # */data/configurations
    LEGACY_APP_CONFIGURATION_PATH = os.path.join(
        LEGACY_CONFIGURATION_PATH,
        "app",
    )  # */data/configurations/app

    # Application configuration

    APP_CONFIGURATION_PATH = os.path.join(
        CONFIGURATION_PATH,
        "app",
    )  # */resources/defaults/app
    APP_CONFIGURATION_FILE_PATH = os.path.join(
        APP_CONFIGURATION_PATH, "config.yaml"
    )  # */resources/defaults/app/config.yaml
    APP_RUNTIME_STATE_FILE_PATH = os.path.join(
        LEGACY_APP_CONFIGURATION_PATH, "runtime_state.yaml"
    )  # */data/configurations/app/runtime_state.yaml
    APP_LOGGING_FILE_PATH = os.path.join(
        APP_CONFIGURATION_PATH, "logging.yaml"
    )  # */resources/defaults/app/logging.yaml

    # GUI configuration

    GUI_CONFIGURATION_PATH = os.path.join(
        CONFIGURATION_PATH,
        "gui",
    )  # */resources/defaults/gui

    # GUI themes

    THEME_CONFIGURATION_PATH = os.path.join(
        GUI_CONFIGURATION_PATH, "themes"
    )  # */resources/defaults/gui/themes

    # GUI styles and Markdown content

    CSS_CONFIGURATION_PATH = os.path.join(
        GUI_CONFIGURATION_PATH, "css"
    )  # */resources/defaults/gui/css
    MDS_CONFIGURATION_PATH = os.path.join(
        GUI_CONFIGURATION_PATH, "mds"
    )  # */resources/defaults/gui/mds
    APP_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "lanistai.scss"
    )  # */resources/defaults/gui/css/lanistai.scss
    YES_NO_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "yes_no_dialog.scss"
    )  # */resources/defaults/gui/css/yes_no_dialog.scss
    SECURE_YES_NO_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "secure_yes_no_dialog.scss"
    )  # */resources/defaults/gui/css/secure_yes_no_dialog.scss
    CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "configuration_dialog.scss"
    )  # */resources/defaults/gui/css/configuration_dialog.scss
    LOAD_MODEL_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "load_model_dialog.scss"
    )  # */resources/defaults/gui/css/load_model_dialog.scss
    SESSION_SWITCH_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "session_switch_dialog.scss"
    )  # */resources/defaults/gui/css/session_switch_dialog.scss
    SESSION_CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "session_configuration_dialog.scss"
    )  # */resources/defaults/gui/css/session_configuration_dialog.scss
    CHECKPOINT_CONFIGURATION_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "checkpoint_configuration_dialog.scss"
    )  # */resources/defaults/gui/css/checkpoint_configuration_dialog.scss
    MODEL_CLONE_SELECTION_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "model_clone_selection_dialog.scss"
    )  # */resources/defaults/gui/css/model_clone_selection_dialog.scss
    MODEL_DETAIL_EDIT_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "model_detail_edit_dialog.scss"
    )  # */resources/defaults/gui/css/model_detail_edit_dialog.scss
    SESSION_DETAIL_EDIT_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "session_detail_edit_dialog.scss"
    )  # */resources/defaults/gui/css/session_detail_edit_dialog.scss
    NEW_SESSION_STRATEGY_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "new_session_strategy_dialog.scss"
    )  # */resources/defaults/gui/css/new_session_strategy_dialog.scss
    HELP_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "help_dialog.scss"
    )  # */resources/defaults/gui/css/help_dialog.scss
    ABOUT_DIALOG_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "about_dialog.scss"
    )  # */resources/defaults/gui/css/about_dialog.scss
    HELP_DIALOG_MD_CONFIGURATION_FILE_PATH = os.path.join(
        MDS_CONFIGURATION_PATH, "help_dialog.md"
    )  # */resources/defaults/gui/mds/help_dialog.md
    ABOUT_DIALOG_MD_CONFIGURATION_FILE_PATH = os.path.join(
        MDS_CONFIGURATION_PATH, "about_dialog.md"
    )  # */resources/defaults/gui/mds/about_dialog.md
    MODEL_CREATION_SCREEN_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "model_creation_screen.scss"
    )  # */resources/defaults/gui/css/model_creation_screen.scss
    HOME_SCREEN_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "home_screen.scss"
    )  # */resources/defaults/gui/css/home_screen.scss
    BYTE_EDITOR_WIDGET_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "byte_editor_widget.scss"
    )  # */resources/defaults/gui/css/byte_editor_widget.scss
    BIT_SWITCH_WIDGET_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "bit_switch_widget.scss"
    )  # */resources/defaults/gui/css/bit_switch_widget.scss
    BYTE_INPUT_WIDGET_CSS_CONFIGURATION_FILE_PATH = os.path.join(
        CSS_CONFIGURATION_PATH, "byte_input_widget.scss"
    )  # */resources/defaults/gui/css/byte_input_widget.scss

    # Splash screen configuration

    SPLASH_CONFIGURATION_PATH = os.path.join(
        GUI_CONFIGURATION_PATH, "splash"
    )  # */resources/defaults/gui/splash
    SPLASH_CONFIGURATION_FILE_PATH = os.path.join(
        SPLASH_CONFIGURATION_PATH, "effects.yaml"
    )  # */resources/defaults/gui/splash/effects.yaml

    # Logging configuration
    LOGS_SAVE_PATH = os.path.join(DATA_PATH, "logs")  # */data/logs
    LOGS_SAVE_FILE_PERMANENT_PATH = os.path.join(
        LOGS_SAVE_PATH, "permanent.yaml"
    )  # */data/logs/permanent.yaml
    LOGS_SAVE_FILE_ROTATE_PATH = os.path.join(
        LOGS_SAVE_PATH, "rotate.yaml"
    )  # */data/logs/rotate.yaml
    LOGS_SAVE_FILE_TIMED_ROTATE_PATH = os.path.join(
        LOGS_SAVE_PATH, "timed_rotate.yaml"
    )  # */data/logs/timed_rotate.yaml

    # Reinforcement-learning model storage

    RL_MODEL_STORAGE_PATH = os.path.join("rl_model_storage")  # */rl_model_storage
    TENSORBOARD_LOGS_PATH = os.path.join(
        RL_MODEL_STORAGE_PATH, "TENSORBOARD"
    )  # */rl_model_storage/TENSORBOARD
