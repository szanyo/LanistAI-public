# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Immutable run-configuration state and UI policy decisions."""

from dataclasses import dataclass, replace

from lanista.reinforcengine.rl_model.model import RLModelRunningModes


@dataclass(frozen=True)
class ConfigurationViewState:
    """Represent runtime state for configuration view."""

    has_model: bool
    loaded_model_name: str | None
    can_new_session: bool
    can_confirm: bool
    can_start_run: bool
    mode: RLModelRunningModes
    framework_uuid: str
    session_name: str | None
    checkpoint_name: str | None
    configuration_locked: bool = False
    session_values: frozenset[str] = frozenset()
    checkpoint_values: frozenset[str] = frozenset()
    session_labels: tuple[tuple[str, str], ...] = ()
    checkpoint_labels: tuple[tuple[str, str], ...] = ()

    @property
    def confirm_label(self) -> str:
        """Return the confirm label exposed by this configuration view state."""
        if self.mode == RLModelRunningModes.TRAINING:
            return "Training"
        if self.mode == RLModelRunningModes.EVALUATION:
            return "Apply"
        return "Confirm"

    @property
    def model_selection_text(self) -> str:
        """Return the model selection text exposed by this configuration view state."""
        return (
            self.loaded_model_name
            if self.has_model and self.loaded_model_name
            else "Choose model..."
        )

    @property
    def session_selection_text(self) -> str:
        """Return the session selection text exposed by this configuration view state."""
        if not self.has_model:
            return "No model loaded yet."
        if self.session_name:
            return dict(self.session_labels).get(self.session_name, self.session_name)
        return "Choose session..." if self.session_values else "No session available yet."

    @property
    def checkpoint_selection_text(self) -> str:
        """Return the checkpoint selection text exposed by this configuration view state."""
        if not self.has_model:
            return "No model loaded yet."
        if not self.session_name:
            return "Select session first."
        if self.checkpoint_name:
            return dict(self.checkpoint_labels).get(self.checkpoint_name, self.checkpoint_name)
        if self.mode == RLModelRunningModes.INFERENCE:
            return "Choose checkpoint (required)"
        if self.mode == RLModelRunningModes.EVALUATION:
            return "Constant-speed diagnostic (no checkpoint)"
        return "Start fresh (no checkpoint)"

    @property
    def session_picker_enabled(self) -> bool:
        """Return the session picker enabled exposed by this configuration view state."""
        return not self.configuration_locked and self.has_model and bool(self.session_values)

    @property
    def checkpoint_picker_enabled(self) -> bool:
        """Return the checkpoint picker enabled exposed by this configuration view state."""
        return (
            not self.configuration_locked
            and self.has_model
            and self.session_name is not None
        )

    @property
    def model_picker_enabled(self) -> bool:
        """Return whether the active model selection may be changed."""

        return not self.configuration_locked

    @property
    def training_actions_enabled(self) -> bool:
        """Return whether model/session creation actions may be started."""

        return not self.configuration_locked and self.mode == RLModelRunningModes.TRAINING

    @property
    def has_valid_checkpoint_selection(self) -> bool:
        """Return the has valid checkpoint selection exposed by this configuration view state."""
        if self.mode == RLModelRunningModes.EVALUATION and self.checkpoint_name is None:
            return self.session_name is not None
        return self.checkpoint_name is not None and self.checkpoint_name in self.checkpoint_values

    @property
    def confirm_enabled(self) -> bool:
        """Return the confirm enabled exposed by this configuration view state."""
        if self.configuration_locked or not (self.has_model and self.can_confirm):
            return False
        if self.mode == RLModelRunningModes.TRAINING:
            return self.session_name is not None and self.can_start_run
        return self.has_valid_checkpoint_selection

    @property
    def new_session_enabled(self) -> bool:
        """Return the new session enabled exposed by this configuration view state."""
        return (
            not self.configuration_locked
            and self.can_new_session
            and self.mode == RLModelRunningModes.TRAINING
        )

    @property
    def inspect_enabled(self) -> bool:
        """Return the inspect enabled exposed by this configuration view state."""
        return (
            not self.configuration_locked
            and self.has_model
            and self.mode != RLModelRunningModes.INFERENCE
        )

    @property
    def compare_enabled(self) -> bool:
        """Return the compare enabled exposed by this configuration view state."""
        return not self.configuration_locked and self.has_model


def select_mode(state: ConfigurationViewState, mode: RLModelRunningModes) -> ConfigurationViewState:
    """Select the mode owned by this configuration view state module."""
    return replace(state, mode=mode)


def select_framework(state: ConfigurationViewState, framework_uuid: str) -> ConfigurationViewState:
    """Select the framework owned by this configuration view state module."""
    return replace(state, framework_uuid=framework_uuid, session_name=None, checkpoint_name=None)


def select_session(
    state: ConfigurationViewState, session_name: str | None
) -> ConfigurationViewState:
    """Select the session owned by this configuration view state module."""
    return replace(state, session_name=session_name, checkpoint_name=None)


def select_checkpoint(
    state: ConfigurationViewState, checkpoint_name: str | None
) -> ConfigurationViewState:
    """Select the checkpoint owned by this configuration view state module."""
    return replace(state, checkpoint_name=checkpoint_name)
