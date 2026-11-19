# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed contracts for the shared Stable-Baselines3 deployment pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from lanista.reinforcengine.rl_metrics.diagnostic_trace_contracts import (
    DiagnosticTraceRow,
)
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_technologies.sb3 import (
    training_pipeline_contracts as training_contracts,
)

EnvironmentPublisher = Callable[[object], None]
DeterministicModeResolver = Callable[[], bool]
ScenarioIndexStatePort = training_contracts.ScenarioIndexStatePort


class SB3DeploymentLogger(Protocol):
    """Logging surface used by configurator resolution and deployment."""

    name: str

    def debug(self, message: str) -> None:
        """Record a configurator-resolution detail."""

    def error(self, message: str) -> None:
        """Record a mandatory configuration failure."""

    def info(self, message: str) -> object:
        """Record one deployment lifecycle event."""


class SB3DeploymentWriter(Protocol):
    """TensorBoard operations used by deployment telemetry."""

    def add_scalar(
        self,
        tag: str,
        value: object,
        step: int,
    ) -> object:
        """Write one scalar sample."""

    def add_histogram(
        self,
        tag: str,
        value: object,
        step: int,
    ) -> object:
        """Write one observation histogram."""


class SB3DeploymentRuntime(Protocol):
    """Runtime control and telemetry operations used during inference."""

    def get_manifest(self) -> ModelManifest:
        """Return the active model manifest."""

    def get_tensorboard_writer(self) -> SB3DeploymentWriter:
        """Return the writer owned by the active runtime."""

    def get_resolved_tensorboard_log_dir(self) -> str | None:
        """Return the active writer-owned evaluation run directory."""

    def get_evaluation_scenario_index(self) -> int:
        """Return the scenario index selected for Evaluation."""

    def report_diagnostic_quality_degradation(
        self,
        event: str,
        error: BaseException,
    ) -> None:
        """Report an optional trace failure without failing evaluation."""

    def check_pause(self) -> None:
        """Block while the active run is paused."""

    def check_stop(self) -> None:
        """Raise when the active run must stop."""


class SB3DeploymentAlgorithm(Protocol):
    """Narrow SB3 algorithm surface used for deterministic inference."""

    def predict(
        self,
        observation: object,
        *,
        deterministic: bool,
    ) -> tuple[object, object | None]:
        """Return one action and optional recurrent state."""


class SB3DeploymentEnvironment(Protocol):
    """Environment operations used by the SB3 inference loop."""

    def reset(self) -> tuple[object, object]:
        """Reset and return the first observation."""

    def step(
        self,
        action: object,
    ) -> tuple[object, object, bool, bool, object]:
        """Advance one simulator step."""

    def get_step_cnt(self) -> int:
        """Return the simulator step used by telemetry."""

    def diagnostic_trace_fieldnames(self) -> tuple[str, ...]:
        """Return the optional model-owned evaluation trace columns."""

    def diagnostic_trace_tensorboard_fields(self) -> frozenset[str]:
        """Return model-owned numeric fields allowed on TensorBoard."""

    def latest_diagnostic_trace_row(
        self,
        *,
        scenario_index: int | None,
        terminated: bool,
        truncated: bool,
    ) -> DiagnosticTraceRow | None:
        """Return the latest already-composed simulator-step trace row."""


class SB3DeploymentScheduler(Protocol):
    """Realtime scheduler operations used around inference steps."""

    def reset(self) -> None:
        """Reset the realtime deadline after environment startup."""

    def delay(self) -> None:
        """Wait until the next realtime deadline."""


class DeploymentAlgorithmConfiguration(Protocol):
    """Mutable configurator reference stored in the model configuration."""

    configurator_module: object


class DeploymentModelConfiguration(Protocol):
    """Configuration values read by the deployment stages."""

    algorithm: DeploymentAlgorithmConfiguration
    evaluation_deterministic: object


class DeploymentScenarioConfiguration(Protocol):
    """Scenario timing required by the realtime scheduler."""

    def get_step_length(self) -> float:
        """Return the simulator step length in seconds."""


class DeploymentScenarioManager(Protocol):
    """Scenario-manager operations used only by SB3 deployment."""

    def set_scenario_index_state(
        self,
        state: ScenarioIndexStatePort,
    ) -> None:
        """Publish the scenario index shared with the environment."""

    def get_configuration(self) -> DeploymentScenarioConfiguration:
        """Return deployment timing configuration."""


@dataclass(frozen=True, slots=True)
class DeploymentPreparation:
    """Validated manifest and scenario state for one deployment run."""

    manifest: ModelManifest
    scenario_index_state: ScenarioIndexStatePort


@dataclass(frozen=True, slots=True)
class DeploymentResources:
    """Environment and realtime scheduler owned by one deployment run."""

    environment: SB3DeploymentEnvironment
    scheduler: SB3DeploymentScheduler


@dataclass(frozen=True, slots=True)
class DeploymentLoopContext:
    """Inputs that remain stable throughout one deployment loop."""

    runtime: SB3DeploymentRuntime
    algorithm: SB3DeploymentAlgorithm
    environment: SB3DeploymentEnvironment
    scheduler: SB3DeploymentScheduler
    deterministic_mode: DeterministicModeResolver


@dataclass(frozen=True, slots=True)
class DeploymentStepResult:
    """Observation and termination state produced by one inference step."""

    observation: object
    terminated: bool
    truncated: bool
    reward: float
