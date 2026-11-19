# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed Ray inference loop with explicit lifecycle and telemetry ordering."""

from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import torch
from ray.rllib.core import Columns

from lanista.reinforcengine.rl_metrics.diagnostic_trace_contracts import (
    DiagnosticTraceRow,
)
from lanista.reinforcengine.rl_metrics.diagnostic_trace_recorder import (
    EvaluationTraceSession,
)
from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    EVALUATION_EPISODE_LENGTH_TAG,
    EVALUATION_EPISODE_RETURN_TAG,
    tensorboard_scalar_value,
)


class RayDeploymentWriter(Protocol):
    """TensorBoard operations used by Ray deployment."""

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


class RayDeploymentRuntime(Protocol):
    """Runtime control and telemetry operations used during inference."""

    def check_pause(self) -> None:
        """Block while the active run is paused."""

    def check_stop(self) -> None:
        """Raise when the active run must stop."""

    def get_tensorboard_writer(self) -> RayDeploymentWriter:
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


class RayActionDistribution(Protocol):
    """Sample actions from one inference distribution."""

    def sample(self) -> torch.Tensor:
        """Sample one action tensor."""

    def to_deterministic(self) -> RayActionDistribution:
        """Return the deterministic inference form of this distribution."""


class RayActionDistributionFactory(Protocol):
    """Construct action distributions from module logits."""

    def from_logits(self, logits: object) -> RayActionDistribution:
        """Construct an action distribution from module logits."""


class RayInferenceModule(Protocol):
    """Narrow RLlib module interface used by deployment."""

    def eval(self) -> object:
        """Put the underlying torch module into evaluation mode."""

    def get_inference_action_dist_cls(
        self,
    ) -> RayActionDistributionFactory:
        """Return the configured inference distribution type."""

    def forward_inference(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, object]:
        """Run one inference forward pass."""


class RayDeploymentEnvironment(Protocol):
    """Multi-agent deployment environment operations."""

    def reset(self) -> tuple[dict[str, np.ndarray], object]:
        """Reset and return the first multi-agent observation."""

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, object],
        dict[str, bool],
        dict[str, bool],
        object,
    ]:
        """Advance one multi-agent simulation step."""

    def get_step_cnt(self) -> int:
        """Return the current simulator step."""

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


class RayDeploymentScheduler(Protocol):
    """Realtime scheduler operations used around inference steps."""

    def reset(self) -> None:
        """Reset the realtime deadline after environment startup."""

    def delay(self) -> None:
        """Wait until the next realtime deadline."""


class RayDeploymentPipeline:
    """Run the ordered pause/stop, inference, simulation, and telemetry loop."""

    def run(
        self,
        *,
        runtime: RayDeploymentRuntime,
        module: RayInferenceModule,
        environment: RayDeploymentEnvironment,
        scheduler: RayDeploymentScheduler,
        repetitions: int = 1,
        publish_evaluation_summary: bool = False,
        realtime: bool = True,
        deterministic: bool = True,
    ) -> None:
        """Run one or more complete multi-agent environment episodes."""

        trace_session = (
            EvaluationTraceSession.create(
                runtime=runtime,
                source=environment,
                framework="ray",
            )
            if publish_evaluation_summary
            else None
        )
        try:
            for episode_index in range(repetitions):
                episode_return, episode_length = self._run_episode(
                    runtime,
                    module,
                    environment,
                    scheduler,
                    episode_index=episode_index,
                    trace_session=trace_session,
                    realtime=realtime,
                    deterministic=deterministic,
                )
                if publish_evaluation_summary:
                    writer = runtime.get_tensorboard_writer()
                    writer.add_scalar(
                        EVALUATION_EPISODE_RETURN_TAG,
                        episode_return,
                        episode_index,
                    )
                    writer.add_scalar(
                        EVALUATION_EPISODE_LENGTH_TAG,
                        episode_length,
                        episode_index,
                    )
        finally:
            if trace_session is not None:
                trace_session.close()

    def _run_episode(
        self,
        runtime: RayDeploymentRuntime,
        module: RayInferenceModule,
        environment: RayDeploymentEnvironment,
        scheduler: RayDeploymentScheduler,
        *,
        episode_index: int,
        trace_session: EvaluationTraceSession | None,
        realtime: bool,
        deterministic: bool,
    ) -> tuple[float, int]:
        """Return the run episode value owned by this ray deployment pipeline."""
        runtime.check_stop()
        action_distribution = module.get_inference_action_dist_cls()
        observations, _ = environment.reset()
        scheduler.reset()
        self._publish_telemetry(runtime, environment, observations)
        terminated = False
        episode_return = 0.0
        episode_length = 0
        while not terminated:
            observations, terminated, truncated, step_reward = self._run_step(
                runtime,
                module,
                action_distribution,
                environment,
                observations,
                deterministic=deterministic,
            )
            episode_return += step_reward
            episode_length += 1
            if trace_session is not None:
                trace_session.append_latest(
                    episode_index=episode_index,
                    terminated=terminated,
                    truncated=truncated,
                )
            runtime.check_stop()
            if realtime:
                scheduler.delay()
        if trace_session is not None:
            trace_session.flush()
        return episode_return, episode_length

    def _run_step(
        self,
        runtime: RayDeploymentRuntime,
        module: RayInferenceModule,
        action_distribution: RayActionDistributionFactory,
        environment: RayDeploymentEnvironment,
        observations: dict[str, np.ndarray],
        *,
        deterministic: bool,
    ) -> tuple[dict[str, np.ndarray], bool, bool, float]:
        """Return the run step value owned by this ray deployment pipeline."""
        runtime.check_pause()
        runtime.check_stop()
        formatted_observation = cast(
            np.ndarray,
            observations.get("EGO_agent_0"),
        )
        batched_observation = torch.as_tensor(
            formatted_observation,
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.inference_mode():
            outputs = module.forward_inference({Columns.OBS: batched_observation})
            action_tensor = outputs.get(Columns.ACTIONS)
            if action_tensor is None:
                logits = outputs.get(Columns.ACTION_DIST_INPUTS)
                if logits is None:
                    raise RuntimeError("RLModule inference returned neither actions nor logits.")
                distribution = action_distribution.from_logits(logits)
                if deterministic:
                    distribution = distribution.to_deterministic()
                action_tensor = distribution.sample()
        action = torch.as_tensor(action_tensor)[0].detach().cpu().numpy()
        actions = {"EGO_agent_0": action}
        next_observations, rewards, terminated, truncated, _ = environment.step(actions)
        self._publish_telemetry(
            runtime,
            environment,
            next_observations,
            actions=actions,
            rewards=rewards,
        )
        step_reward = sum(tensorboard_scalar_value(reward) for reward in rewards.values())
        return (
            next_observations,
            terminated["__all__"],
            truncated["__all__"],
            step_reward,
        )

    @staticmethod
    def _publish_telemetry(
        runtime: RayDeploymentRuntime,
        environment: RayDeploymentEnvironment,
        observations: dict[str, np.ndarray],
        *,
        actions: dict[str, np.ndarray] | None = None,
        rewards: dict[str, object] | None = None,
    ) -> None:
        """Leave step telemetry to the lossless CSV plus curated projection."""

        _ = runtime, environment, observations, actions, rewards
