# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define the storage layout for storage."""

import os

MODEL_MANIFEST_FILE_NAME = "model_manifest.yaml"
MODEL_CONFIGURATION_FILE_NAME = "model_configuration.yml"
CLONE_MANIFEST_FILE_NAME = "clone_manifest.yml"
FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME = "model_manifest_configuration.yaml"

TRAINING_SESSIONS_DIR_NAME = "training_sessions"
SESSION_MANIFEST_FILE_NAME = "session_manifest.yml"
CHECKPOINT_MANIFEST_FILE_NAME = "checkpoint_manifest.yml"

RESOURCES_DIR_NAME = "resources"
TRAINING_SCENARIOS_DIR_NAME = "training_scenarios"
RAY_RESOURCES_DIR_NAME = "ray"
SB3_RESOURCES_DIR_NAME = "sb3"

TENSORBOARD_DIR_NAME = "tensorboard"
TENSORBOARD_LEGACY_INSPECTS_DIR_NAME = "inspects"
TENSORBOARD_EVALUATION_DIR_NAME = "evaluation"
TENSORBOARD_COMPARISON_DIR_NAME = "comparison"
TENSORBOARD_TRAININGS_DIR_NAME = "trainings"
TENSORBOARD_RAY_DIR_NAME = "ray"
TENSORBOARD_SB3_DIR_NAME = "sb3"


def find_existing_model_manifest_path(model_folder_path: str) -> str:
    """Find the existing model manifest path required by this storage layout module."""
    model_folder_path = str(model_folder_path)
    return os.path.join(model_folder_path, MODEL_MANIFEST_FILE_NAME)


def get_preferred_model_manifest_path(model_folder_path: str) -> str:
    """Return the preferred model manifest path exposed by this storage layout module."""
    return os.path.join(str(model_folder_path), MODEL_MANIFEST_FILE_NAME)


def get_model_configuration_path(model_folder_path: str) -> str:
    """Return the model configuration path exposed by this storage layout module."""
    return os.path.join(str(model_folder_path), MODEL_CONFIGURATION_FILE_NAME)


def get_framework_resource_path(model_folder_path: str, framework_uuid: str) -> str:
    """Return the framework resource path exposed by this storage layout module."""
    framework_dir_name = (
        RAY_RESOURCES_DIR_NAME
        if str(framework_uuid or "").strip().lower() == "ray"
        else SB3_RESOURCES_DIR_NAME
    )
    return os.path.join(str(model_folder_path), RESOURCES_DIR_NAME, framework_dir_name)


def get_framework_model_configuration_path(model_folder_path: str, framework_uuid: str) -> str:
    """Return the framework model configuration path exposed by this storage layout module."""
    return os.path.join(
        get_framework_resource_path(model_folder_path, framework_uuid),
        FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
    )


def find_existing_session_manifest_path(session_folder_path: str) -> str:
    """Find the existing session manifest path required by this storage layout module."""
    session_folder_path = str(session_folder_path)
    return os.path.join(session_folder_path, SESSION_MANIFEST_FILE_NAME)


def get_preferred_session_manifest_path(session_folder_path: str) -> str:
    """Return the preferred session manifest path exposed by this storage layout module."""
    return os.path.join(str(session_folder_path), SESSION_MANIFEST_FILE_NAME)


def ensure_tensorboard_structure(model_folder_path: str) -> None:
    """Ensure the tensorboard structure owned by this storage layout module."""
    tensorboard_root = os.path.join(str(model_folder_path), TENSORBOARD_DIR_NAME)
    trainings_root = os.path.join(tensorboard_root, TENSORBOARD_TRAININGS_DIR_NAME)
    os.makedirs(os.path.join(trainings_root, TENSORBOARD_RAY_DIR_NAME), exist_ok=True)
    os.makedirs(os.path.join(trainings_root, TENSORBOARD_SB3_DIR_NAME), exist_ok=True)
    evaluation_root = os.path.join(tensorboard_root, TENSORBOARD_EVALUATION_DIR_NAME)
    os.makedirs(os.path.join(evaluation_root, TENSORBOARD_RAY_DIR_NAME), exist_ok=True)
    os.makedirs(os.path.join(evaluation_root, TENSORBOARD_SB3_DIR_NAME), exist_ok=True)
    os.makedirs(os.path.join(tensorboard_root, TENSORBOARD_COMPARISON_DIR_NAME), exist_ok=True)
