# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Freeze the application-wide training device choice for one process image."""

from __future__ import annotations

from collections.abc import Mapping

from lanista.application_bootstrap import APP_CONFIGURATION
from lanista.foundation.configuration import Configuration


def configured_training_gpu_enabled(configuration: Configuration) -> bool:
    """Read the persisted global choice, defaulting older configs to CPU."""

    training = configuration.get_data().get("training")
    if not isinstance(training, Mapping):
        return False
    value = training.get("use_gpu")
    return value if isinstance(value, bool) else False


ACTIVE_TRAINING_GPU_ENABLED = configured_training_gpu_enabled(APP_CONFIGURATION)


def sb3_device() -> str:
    """Choose the explicit SB3 device for this application process."""

    return "cuda" if ACTIVE_TRAINING_GPU_ENABLED else "cpu"


def ray_gpu_count() -> int:
    """Reserve one GPU for Ray training only when this process opted in."""

    return 1 if ACTIVE_TRAINING_GPU_ENABLED else 0
