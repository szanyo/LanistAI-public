# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Early process configuration required before application module imports."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from queue import Queue

from lanista.foundation.configuration import AppConfiguration
from lanista.foundation.logging import PipelineHandler, RawFormatter
from lanista.locations import get_app_config_path, get_data_home_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def _load_application_configuration() -> AppConfiguration:
    """Load the process-wide configuration from the application directory."""

    application_path = Path(__file__).resolve().parent
    configuration = AppConfiguration()
    configuration.set_config_file_path(get_app_config_path())
    configuration.load()
    configuration.app_path = str(application_path)
    configuration.data_home = get_data_home_path()
    return configuration


def _configure_tensorflow_log_capture() -> None:
    """Buffer early TensorFlow records until the root logger is available."""

    tensorflow_logger = logging.getLogger("tensorflow")
    tensorflow_logger.handlers.clear()
    tensorflow_logger.setLevel(logging.DEBUG)
    tensorflow_logger.propagate = False

    pipeline_handler = PipelineHandler()
    pipeline_handler.set_name("pipeline")
    pipeline_handler.set_pipeline(Queue[object]())
    pipeline_handler.setFormatter(RawFormatter())
    tensorflow_logger.addHandler(pipeline_handler)


APP_CONFIGURATION = _load_application_configuration()
_configure_tensorflow_log_capture()
