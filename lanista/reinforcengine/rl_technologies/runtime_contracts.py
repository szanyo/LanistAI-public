# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral contracts shared by Ray and SB3 runtime cores."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from lanista.reinforcengine.rl_model.execution_contracts import (
    EvaluationVisualization,
)

DeploymentPurpose = Literal["deployment", "evaluation"]

if TYPE_CHECKING:
    from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
    from lanista.reinforcengine.rl_model.model import BaseModelRuntime, ProgressCallback
    from lanista.reinforcengine.rl_model.runtime.requests import (
        RuntimeInitializationRequest,
    )
    from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
        ScenarioManager,
    )


@dataclass(frozen=True)
class FrameworkLogicLoggerContext:
    """Logger ownership supplied to a model-local framework logic adapter."""

    parent_logger_name: str
    child_logger_name: str
    child_logger_level: int


class FrameworkLogic(Protocol):
    """Lifecycle operations implemented by model-specific framework logic."""

    def build(self) -> object:
        """Build the configured result owned by this framework logic."""
        ...

    def load(
        self,
        model_name: str | None = None,
        checkpoint_name: str | None = None,
    ) -> object:
        """Load and return the state owned by this framework logic."""
        ...

    def train(
        self,
        scenario_manager: ScenarioManager,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Train the operation owned by this framework logic."""
        ...

    def save(
        self,
        checkpoint: bool = False,
        manual: bool = False,
    ) -> str | None:
        """Persist the current state owned by this framework logic."""
        ...

    def deploy(
        self,
        scenario_manager: ScenarioManager,
        *,
        purpose: DeploymentPurpose = "deployment",
        repetitions: int = 1,
        visualization: EvaluationVisualization = "gui",
        realtime: bool | None = None,
    ) -> None:
        """Return the deploy value provided by this framework logic."""
        ...


@runtime_checkable
class EnvironmentFactory(Protocol):
    """Create the model-local environment/algorithm adapter for one runtime."""

    def create(
        self,
        runtime: BaseModelRuntime,
        context: FrameworkLogicLoggerContext,
    ) -> FrameworkLogic:
        """Create a validated value owned by this environment factory."""
        ...


EnvironmentBuilder = Callable[
    ["BaseModelRuntime", FrameworkLogicLoggerContext],
    FrameworkLogic,
]


@dataclass(frozen=True)
class CallableEnvironmentFactory:
    """Adapt a typed model-local builder to the environment-factory port."""

    builder: EnvironmentBuilder

    def create(
        self,
        runtime: BaseModelRuntime,
        context: FrameworkLogicLoggerContext,
    ) -> FrameworkLogic:
        """Create a validated value owned by this callable environment factory."""
        return self.builder(runtime, context)


@runtime_checkable
class ModelPlugin(Protocol):
    """Managed runtime exposed by a model-local plugin module."""

    framework_uuid: str
    environment_factory: EnvironmentFactory

    async def initialize_async(self) -> BaseModelRuntime:
        """Initialize the async owned by this model plugin."""
        ...

    def has_managed_execution_lifecycle(self) -> bool:
        """Return whether managed execution lifecycle applies to this model plugin."""
        ...


@runtime_checkable
class FrameworkRuntimeFactory(Protocol):
    """Construct a validated model runtime for one framework."""

    @property
    def framework_uuid(self) -> str:
        """Return the framework uuid exposed by this framework runtime factory."""
        ...

    def create(
        self,
        model_manifest: ModelManifest,
        request: RuntimeInitializationRequest,
    ) -> BaseModelRuntime:
        """Create a validated value owned by this framework runtime factory."""
        ...
