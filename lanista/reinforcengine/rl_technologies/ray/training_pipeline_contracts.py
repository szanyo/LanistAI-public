# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed immutable contracts for the composed Ray training pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ray.rllib.algorithms import Algorithm

from lanista.reinforcengine.rl_environment.training_progress import (
    TrainingProgressEvidence,
)
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_technologies.ray.model_policy import (
    CurriculumState,
    RayCurriculumRestoreManager,
    RayCurriculumStateActor,
    RayCurriculumWindowManager,
    RayModelPolicy,
    RayPolicyRollingStats,
)


class RayTrainingLogger(Protocol):
    """Logging surface required by the Ray training stages."""

    name: str
    level: int

    def debug(self, message: str) -> None:
        """Record diagnostic lifecycle detail."""

    def info(self, message: str) -> None:
        """Record an informational lifecycle event."""

    def warning(self, message: str) -> None:
        """Record a quality degradation."""

    def error(self, message: str) -> None:
        """Record a failed mandatory operation."""

    def critical(self, message: str) -> None:
        """Record an unrecoverable configuration problem."""


class RayAlgorithmComponents(Protocol):
    """Configured algorithm entry points used for training."""

    training_configurator: str


class RayTrainingAlgorithmConfiguration(Protocol):
    """Mutable algorithm configuration projection stored in the manifest."""

    components: RayAlgorithmComponents
    training_conditions: object
    configurator_module: str


class RayTrainingConfiguration(Protocol):
    """Model configuration values consumed by the training pipeline."""

    algorithm: RayTrainingAlgorithmConfiguration
    checkpoint_saving_frequency: int


class RayTrainingScenarioManager(Protocol):
    """Scenario operations required by the shared training lifecycle."""

    def get_total_number_of_scenarios(self) -> int:
        """Return the number of configured scenarios."""

    def get_index(self) -> int:
        """Return the active scenario index."""

    def set_index(self, index: int) -> None:
        """Select the active scenario."""

    def set_scenario_index_state(self, state: object) -> None:
        """Attach the Ray scenario state actor."""

    def clone(self) -> object:
        """Create the environment-owned scenario manager."""

    def finished(self) -> bool:
        """Return whether all training scenarios are complete."""

    def next(self) -> None:
        """Advance to the next scenario."""


class RayTrainingRuntime(Protocol):
    """Runtime controls required by the training loop."""

    def check_stop(self) -> None:
        """Raise when the active run must stop."""

    def check_pause(self) -> None:
        """Block while the active run is paused."""


class RayProgressCallback(Protocol):
    """Progress callback used by the application runtime."""

    def __call__(
        self,
        total: float,
        current: float,
        state: RLModelState,
        message: str,
        /,
    ) -> None:
        """Publish one training progress update."""


class RayAlgorithmConfiguration(Protocol):
    """RLlib configuration operations used before algorithm construction."""

    def callbacks(self, callbacks: list[object]) -> object:
        """Configure callback classes and factories."""

    def build_algo(self) -> Algorithm:
        """Construct the owned RLlib algorithm."""


class RayTrainingAlgorithm(Protocol):
    """Typed subset of the otherwise untyped RLlib algorithm lifecycle."""

    def restore(self, checkpoint_path: str) -> None:
        """Restore algorithm state from a checkpoint."""

    def train(self) -> dict[str, object]:
        """Execute one training iteration."""


class RayAlgorithmConfigurator(Protocol):
    """Model-local algorithm configurator function."""

    def __call__(self, model_name: str) -> RayAlgorithmConfiguration:
        """Build an RLlib configuration for one model session."""


class RayConfiguratorLoader(Protocol):
    """Validated dynamic configurator loader."""

    def __call__(
        self,
        manifest: ModelManifest,
        logger: RayTrainingLogger,
    ) -> tuple[str, object]:
        """Return the stable module reference and imported module."""


class RayScenarioSelection(Protocol):
    """Configured training scenario selection boundary."""

    def __call__(
        self,
        configuration: object,
        scenario_manager: RayTrainingScenarioManager,
        *,
        purpose: str,
        default_index: int,
        logger: RayTrainingLogger,
    ) -> int | None:
        """Select and publish the initial scenario."""


class RayStateActorFactory(Protocol):
    """Create one Ray-backed state actor."""

    def __call__(self) -> RayCurriculumStateActor:
        """Return the newly created actor handle."""


class RayEpisodeTraceCaptureGateFactory(Protocol):
    """Create the shared atomic post-checkpoint episode capture gate."""

    def __call__(self) -> object:
        """Return one Ray actor handle shared by callbacks and environments."""


class RayRestoreManifestLoader(Protocol):
    """Load mandatory checkpoint restore metadata."""

    def __call__(self, checkpoint_path: str) -> dict[str, object]:
        """Return validated restore metadata."""


class CurriculumDefinition(Protocol):
    """Resolved curriculum definition used by the Ray manager."""

    @property
    def states(self) -> list[dict[str, object]]:
        """Return normalized curriculum states."""

    @property
    def base_min_terminated_episodes(self) -> int:
        """Return the base transition threshold."""


class CurriculumDefinitionResolver(Protocol):
    """Resolve curriculum state-machine settings."""

    def __call__(self, training_conditions: object) -> CurriculumDefinition:
        """Return the normalized curriculum definition."""


class RayCurriculumManagerPort(
    RayCurriculumWindowManager,
    RayCurriculumRestoreManager,
    Protocol,
):
    """Typed lifecycle surface of the Ray curriculum manager."""

    def initialize(self) -> None:
        """Publish the initial curriculum state."""


class RayCurriculumManagerFactory(Protocol):
    """Construct the Ray curriculum manager with its logger context."""

    def __call__(
        self,
        states: list[dict[str, object]],
        *,
        curriculum_index_state: RayCurriculumStateActor,
        base_min_terminated_episodes: int,
        parent_logger_name: str,
        child_logger_name: str,
        child_logger_level: int,
    ) -> RayCurriculumManagerPort:
        """Return a configured curriculum manager."""


class RayPicklabilityCheck(Protocol):
    """Validate the payload captured by the environment creator."""

    def __call__(
        self,
        value: object,
        *,
        logger: RayTrainingLogger,
    ) -> bool:
        """Return whether Ray can serialize the payload."""


class RayEnvironmentRegistrar(Protocol):
    """Register a named RLlib environment creator."""

    def __call__(
        self,
        name: str,
        creator: Callable[[object], object],
    ) -> None:
        """Register the creator under the model session name."""


class RayCheckpointCallbackFactory(Protocol):
    """Create a periodic checkpoint callback."""

    def __call__(
        self,
        *,
        scenario_index_state: RayCurriculumStateActor,
        save_freq: int,
        checkpoint_directory: str,
        name_prefix: str,
        curriculum_index_state: RayCurriculumStateActor,
        parent_checkpoint_name: str | None,
        base_step_offset: int,
        model_package_directory: str,
        episode_trace_capture_gate: object,
    ) -> object:
        """Return one independently owned callback instance."""


class RayRollingStatsFactory(Protocol):
    """Construct the rolling-statistics accumulator."""

    def __call__(
        self,
        training_conditions: object,
        *,
        parent_logger_name: str,
        child_logger_name: str,
        child_logger_level: int,
    ) -> RayTrainingRollingStats:
        """Return the accumulator for one training run."""


class RayObjectResolver(Protocol):
    """Wait for one Ray actor operation."""

    def __call__(self, reference: object) -> object:
        """Resolve the remote operation result."""


class RayTrainingRollingStats(RayPolicyRollingStats, Protocol):
    """Rolling-statistics lifecycle required by the training pipeline."""

    def add_episode(self, train_result: dict[str, object]) -> bool:
        """Ingest one training result and report whether it added episodes."""

    def log_window_metric_stats(self) -> None:
        """Log the completed rolling window."""

    def metrics_ready(self) -> bool:
        """Return whether training metrics permit a scenario switch."""

    def training_progress_evidence(
        self,
        curriculum_state: object | None,
    ) -> TrainingProgressEvidence:
        """Return active-scenario milestone counters."""


@dataclass(frozen=True, slots=True)
class RayTrainingOwnership:
    """Callbacks that keep mutable algorithm state owned by the framework core."""

    publish_algorithm: Callable[[Algorithm], None]
    publish_scenario_index: Callable[[int], None]
    publish_iteration: Callable[[dict[str, object], int], None]
    publish_curriculum_state: Callable[[CurriculumState], None]
    current_checkpoint_step: Callable[[], int]


@dataclass(frozen=True, slots=True)
class RayTrainingContext:
    """Immutable inputs shared by all stages of one Ray training run."""

    manifest: ModelManifest
    configuration: RayTrainingConfiguration
    runtime: RayTrainingRuntime
    logger: RayTrainingLogger
    policy: RayModelPolicy
    ownership: RayTrainingOwnership
    environment_type: type[object]
    scenario_manager: RayTrainingScenarioManager
    progress_callback: RayProgressCallback | None
    model_name: str
    restoration_path: str | None
    checkpoint_directory: str
    parent_checkpoint_name: str | None
    base_step_offset: int
    model_package_directory: str


@dataclass(frozen=True, slots=True)
class RayTrainingDependencies:
    """Injectable framework boundaries used by the composed stages."""

    load_configurator: RayConfiguratorLoader
    apply_scenario_selection: RayScenarioSelection
    scenario_state_factory: RayStateActorFactory
    curriculum_state_factory: RayStateActorFactory
    load_restore_manifest: RayRestoreManifestLoader
    resolve_curriculum_definition: CurriculumDefinitionResolver
    curriculum_manager_factory: RayCurriculumManagerFactory
    is_picklable: RayPicklabilityCheck
    register_environment: RayEnvironmentRegistrar
    checkpoint_callback_factory: RayCheckpointCallbackFactory
    episode_trace_capture_gate_factory: RayEpisodeTraceCaptureGateFactory
    scenario_switching_callback: object
    rolling_stats_factory: RayRollingStatsFactory
    resolve_object: RayObjectResolver


@dataclass(frozen=True, slots=True)
class RayConfiguratorSelectionResult:
    """Configurator and scenario-selection output."""

    configurator_module: object
    configurator_name: str
    training_conditions: object


@dataclass(frozen=True, slots=True)
class RayActorRestoreResult:
    """Actor handles and restored curriculum manager."""

    scenario_index_state: RayCurriculumStateActor
    curriculum_index_state: RayCurriculumStateActor
    curriculum_manager: RayCurriculumManagerPort
    episode_trace_capture_gate: object


@dataclass(frozen=True, slots=True)
class RayAlgorithmBuildResult:
    """Algorithm constructed from one immutable setup projection."""

    algorithm: Algorithm


@dataclass(frozen=True, slots=True)
class RayTrainingResult:
    """Terminal result of a non-exceptional training pipeline run."""

    algorithm: Algorithm | None
    environment_registered: bool
    completed: bool
