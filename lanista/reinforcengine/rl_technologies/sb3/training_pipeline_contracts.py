# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed contracts shared by the Stable-Baselines3 training stages."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Logger

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_lifecycle import (
    StableBaselinesCheckpointMetadataCallback,
)
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    SB3ModelPolicy,
    SB3PolicyLogger,
    SB3TrainingRestore,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)

EnvironmentPublisher = Callable[[object], None]
CheckpointCallbackPublisher = Callable[
    [StableBaselinesCheckpointMetadataCallback],
    None,
]
ScenarioPublisher = Callable[[int], None]


class SB3TrainingLogger(SB3PolicyLogger, Protocol):
    """Logging surface shared by the training stages."""

    def debug(self, message: str) -> None:
        """Record a configurator-resolution detail."""

    def error(self, message: str) -> None:
        """Record a mandatory training failure."""


class TrainingProgressReporter(Protocol):
    """UI progress callback used by SB3 training."""

    def __call__(
        self,
        total: float,
        progress: float,
        state: RLModelState,
        message: str,
        /,
    ) -> None:
        """Publish one progress update."""


class CurriculumManagerPort(Protocol):
    """Curriculum operations needed after construction."""

    def initialize(self) -> None:
        """Publish the initial curriculum state."""

    def restore_state(self, state: object | None) -> None:
        """Restore checkpoint curriculum state."""

    def current_state(self) -> object | None:
        """Return the active decorated curriculum state."""


class ReadableStatePort(Protocol):
    """State value read by checkpoint metadata persistence."""

    def get(self) -> object:
        """Return the current state value."""


class ScenarioIndexStatePort(ReadableStatePort, Protocol):
    """Writable scenario index shared with environments."""

    def get(self) -> int:
        """Return the current scenario index."""

    def set(self, value: int) -> None:
        """Publish a scenario index."""


class ScenarioConfigurationPort(Protocol):
    """Scenario configuration values used to derive the training budget."""

    def get_max_timesteps(self) -> int:
        """Return the maximum steps in one scenario."""


class TrainingScenarioManagerPort(Protocol):
    """Scenario-manager operations used by the SB3 training stages."""

    def clone(self) -> object:
        """Return an isolated manager for one environment."""

    def set_scenario_index_state(
        self,
        state: ScenarioIndexStatePort,
    ) -> None:
        """Publish the shared scenario index."""

    def get_configuration(self) -> ScenarioConfigurationPort:
        """Return the scenario budget configuration."""

    def get_total_number_of_scenarios(self) -> int:
        """Return the number of scenarios in the active schedule."""


ScenarioStatePublisher = Callable[[ScenarioIndexStatePort], None]
CurriculumStatePublisher = Callable[[ReadableStatePort], None]


class SB3TrainingAlgorithm(Protocol):
    """Algorithm operations exercised by the training pipeline."""

    num_timesteps: int

    def set_env(self, environment: object) -> None:
        """Replace the algorithm environment."""

    def set_logger(self, logger: Logger) -> None:
        """Replace the algorithm logger after session retargeting."""

    def learn(
        self,
        *,
        total_timesteps: int,
        callback: BaseCallback,
        reset_num_timesteps: bool,
    ) -> object:
        """Run one scenario training window."""

    def predict(
        self,
        observation: object,
        *,
        deterministic: bool,
    ) -> tuple[object, object | None]:
        """Return one deployment action and optional recurrent state."""

    def save(self, path: str) -> None:
        """Persist the algorithm checkpoint."""


@dataclass(frozen=True, slots=True)
class TrainingPreparation:
    """Validated configuration and restore intent for one training run."""

    manifest: ModelManifest
    configuration: Configuration
    configurator_module: object
    model_policy: SB3ModelPolicy
    restore: SB3TrainingRestore


@dataclass(frozen=True, slots=True)
class TrainingStates:
    """Scenario and curriculum state shared with the environments."""

    scenario_index_state: ScenarioIndexStatePort
    curriculum_index_state: ReadableStatePort
    curriculum_manager: CurriculumManagerPort


@dataclass(frozen=True, slots=True)
class TrainingEnvironment:
    """Owned environment plus its deterministic timestep budget."""

    environment: object
    environment_type: type[object]
    constructor_options: Mapping[str, object]
    total_timesteps: int


@dataclass(frozen=True, slots=True)
class TrainingModelResult:
    """Algorithm and continuation offsets used by callbacks and learning."""

    model: SB3TrainingAlgorithm
    continue_training: bool
    continue_same_session: bool
    source_base_step_offset: int


CheckpointLoader = Callable[
    [str, str],
    tuple[SB3TrainingAlgorithm, int],
]
SessionRetargeter = Callable[[str | None], None]
ModelPublisher = Callable[[SB3TrainingAlgorithm], None]
StepOffsetPublisher = Callable[[int], None]


@dataclass(frozen=True, slots=True)
class TrainingModelOperations:
    """Narrow owner callbacks needed while constructing the algorithm."""

    load_checkpoint: CheckpointLoader
    retarget_session: SessionRetargeter
    publish_model: ModelPublisher
    publish_step_offset: StepOffsetPublisher
    configure_tensorboard_logger: Callable[[], None]
    resolve_tensorboard_log_directory: Callable[[], object | None]
    resolve_tensorboard_step_floor: Callable[[], int] = lambda: 0


@dataclass(frozen=True, slots=True)
class TrainingCallbackContext:
    """Immutable inputs for one callback-list assembly."""

    preparation: TrainingPreparation
    states: TrainingStates
    environment: TrainingEnvironment
    model_result: TrainingModelResult
    runtime: BaseModelRuntime
    scenario_manager: ScenarioManager
    progress_reporter: TrainingProgressReporter | None
    checkpoint_directory: str
    model_package_directory: str
    logger: SB3TrainingLogger


@dataclass(frozen=True, slots=True)
class TrainingCallbacks:
    """Callback list and its checkpoint telemetry owner."""

    callback_list: BaseCallback
    checkpoint_callback: StableBaselinesCheckpointMetadataCallback


@dataclass(frozen=True, slots=True)
class TrainingLoopContext:
    """Inputs that stay stable across scenario training windows."""

    runtime: BaseModelRuntime
    scenario_manager: ScenarioManager
    model_result: TrainingModelResult
    total_timesteps: int
    progress_reporter: TrainingProgressReporter | None
    logger: SB3TrainingLogger


@dataclass(frozen=True, slots=True)
class TrainingLoopResult:
    """Observable result of a completed training loop."""

    iteration_count: int
    last_scenario_index: int | None
