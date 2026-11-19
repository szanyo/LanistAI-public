# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Select scenario."""

from typing import Any


def _config_get(configuration: Any, key: str, default=None):
    """Return the config get value owned by this scenario selection module."""
    if configuration is None:
        return default
    if isinstance(configuration, dict):
        return configuration.get(key, default)
    getter = getattr(configuration, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(configuration, key, default)


def _scenario_selection_config(model_configuration: Any):
    """Return the scenario selection config value owned by this scenario selection module."""
    runtime = _config_get(model_configuration, "runtime")
    scenario_selection = _config_get(runtime, "scenario_selection")
    if scenario_selection is not None:
        return scenario_selection
    return _config_get(model_configuration, "scenario_selection")


def resolve_scenario_index(model_configuration: Any, purpose: str, default_index: int = 0) -> int:
    """Resolve the scenario index required by this scenario selection module."""
    selection = _scenario_selection_config(model_configuration)
    key_candidates = {
        "training": ("training_start_index", "training_index", "start_index"),
        "deployment": ("deployment_index", "deployment_scenario_index", "scenario_index"),
        "evaluation": ("evaluation_index", "evaluation_scenario_index", "scenario_index"),
    }.get(str(purpose), (f"{purpose}_index", "scenario_index"))

    selected = None
    for key in key_candidates:
        selected = _config_get(selection, key)
        if selected is not None:
            break

    if selected is None:
        selected = default_index

    try:
        return int(selected)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {purpose} scenario index: {selected}") from exc


def apply_scenario_selection(
        model_configuration: Any,
        scenario_manager,
        *,
        purpose: str,
        default_index: int = 0,
        selected_index: int | None = None,
        logger=None,
) -> int:
    """Apply the scenario selection transition owned by this scenario selection module."""
    scenario_index = (
        resolve_scenario_index(
            model_configuration,
            purpose=purpose,
            default_index=default_index,
        )
        if selected_index is None
        else selected_index
    )
    total = scenario_manager.get_total_number_of_scenarios()
    if scenario_index < 0 or scenario_index >= total:
        message = (
            f"Desired {purpose} scenario index {scenario_index} is out of range. "
            f"The scenario manager has only {total} scenarios."
        )
        if logger is not None:
            logger.error(message)
        raise IndexError(message)

    scenario_manager.reset()
    scenario_manager.set_index(scenario_index)
    if logger is not None:
        logger.info(f"Selected {purpose} scenario index: {scenario_manager.get_index()}")
    return scenario_index
