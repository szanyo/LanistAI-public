# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Composable stages for one Stable-Baselines3 deployment run."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from lanista.reinforcengine.rl_metrics.diagnostic_trace_recorder import (
    EvaluationTraceSession,
)
from lanista.reinforcengine.rl_metrics.tensorboard_surface import (
    EVALUATION_EPISODE_LENGTH_TAG,
    EVALUATION_EPISODE_RETURN_TAG,
    tensorboard_scalar_value,
)
from lanista.reinforcengine.rl_model.execution_contracts import EvaluationVisualization
from lanista.reinforcengine.rl_technologies.runtime_contracts import DeploymentPurpose
from lanista.reinforcengine.rl_technologies.sb3 import (
    deployment_pipeline_contracts as contracts,
)
from lanista.reinforcengine.running_mode import RunningMode
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    apply_scenario_selection,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.scenario_state_sb3 import (
    StableBaselinesScenarioState,
)
from lanista.reinforcengine.utils.tools.model_configurator_loader import (
    load_algorithm_configurator_module,
)
from lanista.reinforcengine.utils.tools.scheduler import Scheduler


class DeploymentPreparationStage:
    """Resolve the configurator and publish the selected scenario state."""

    def prepare(
        self,
        *,
        runtime: contracts.SB3DeploymentRuntime,
        scenario_manager: ScenarioManager,
        purpose: DeploymentPurpose,
        logger: contracts.SB3DeploymentLogger,
    ) -> contracts.DeploymentPreparation:
        """Return the prepare value owned by this deployment preparation stage."""
        manifest = runtime.get_manifest()
        configurator_module_ref, _ = load_algorithm_configurator_module(
            manifest,
            logger,
        )
        configuration = cast(
            contracts.DeploymentModelConfiguration,
            manifest.get_configuration(),
        )
        configuration.algorithm.configurator_module = configurator_module_ref

        selection_configuration = runtime.get_manifest().get_configuration()
        apply_scenario_selection(
            selection_configuration,
            scenario_manager,
            purpose=purpose,
            default_index=0 if purpose == "evaluation" else 20,
            selected_index=(
                runtime.get_evaluation_scenario_index()
                if purpose == "evaluation"
                else None
            ),
            logger=logger,
        )

        scenario_state_factory = cast(
            Callable[[], contracts.ScenarioIndexStatePort],
            StableBaselinesScenarioState,
        )
        scenario_index_state = scenario_state_factory()
        cast(
            contracts.DeploymentScenarioManager,
            scenario_manager,
        ).set_scenario_index_state(scenario_index_state)
        return contracts.DeploymentPreparation(
            manifest=manifest,
            scenario_index_state=scenario_index_state,
        )


class DeploymentEnvironmentStage:
    """Construct and publish the environment before fallible startup."""

    def create(
        self,
        *,
        preparation: contracts.DeploymentPreparation,
        scenario_manager: ScenarioManager,
        environment_type: type[object],
        logger: contracts.SB3DeploymentLogger,
        publish_environment: contracts.EnvironmentPublisher,
        purpose: DeploymentPurpose = "deployment",
        visualization: EvaluationVisualization = "gui",
    ) -> contracts.DeploymentResources:
        """Create a validated value owned by this deployment environment stage."""
        options: dict[str, object] = {
            "verbose": False,
            "detailed_verbose": False,
            "parent_logger_name": logger.name,
            "child_logger_name": "stable_baselines_sim_env",
            "rl_model_manifest": preparation.manifest,
            "rl_model_mode": (
                RunningMode.TESTING if visualization == "headless" else RunningMode.DEPLOYMENT
            ),
            "scenario_manager": scenario_manager,
            "scenario_index_state": preparation.scenario_index_state,
        }
        if purpose == "evaluation" and visualization == "gui":
            options["defer_gui_shutdown_to_process_restart"] = True
        environment = cast(
            contracts.SB3DeploymentEnvironment,
            environment_type(**options),
        )
        publish_environment(environment)

        scheduler_factory = cast(
            Callable[[float], contracts.SB3DeploymentScheduler],
            Scheduler,
        )
        step_length = (
            cast(
                contracts.DeploymentScenarioManager,
                scenario_manager,
            )
            .get_configuration()
            .get_step_length()
        )
        return contracts.DeploymentResources(
            environment=environment,
            scheduler=scheduler_factory(step_length),
        )


class DeploymentTelemetryStage:
    """Retain the deployment hook without duplicating evaluation traces."""

    @staticmethod
    def publish_initial(
        runtime: contracts.SB3DeploymentRuntime,
        environment: contracts.SB3DeploymentEnvironment,
        observation: object,
    ) -> None:
        """Leave initial telemetry to the curated evaluation trace."""

        _ = runtime, environment, observation

    @staticmethod
    def publish_step(
        runtime: contracts.SB3DeploymentRuntime,
        environment: contracts.SB3DeploymentEnvironment,
        *,
        action: object,
        reward: object,
        observation: object,
    ) -> None:
        """Leave step telemetry to the lossless CSV plus curated projection."""

        _ = runtime, environment, action, reward, observation


class DeploymentInferenceStepStage:
    """Run one cooperative inference, simulation, and telemetry step."""

    def run(
        self,
        context: contracts.DeploymentLoopContext,
        observation: object,
    ) -> contracts.DeploymentStepResult:
        """Run the complete workflow owned by this deployment inference step stage."""
        context.runtime.check_pause()
        context.runtime.check_stop()
        action, _ = context.algorithm.predict(
            observation,
            deterministic=context.deterministic_mode(),
        )
        next_observation, reward, terminated, truncated, _ = (
            context.environment.step(action)
        )
        DeploymentTelemetryStage.publish_step(
            context.runtime,
            context.environment,
            action=action,
            reward=reward,
            observation=next_observation,
        )
        return contracts.DeploymentStepResult(
            observation=next_observation,
            terminated=terminated,
            truncated=truncated,
            reward=tensorboard_scalar_value(reward),
        )


class DeploymentLoopStage:
    """Run inference until termination while retaining legacy stop order."""

    def run(
        self,
        context: contracts.DeploymentLoopContext,
        *,
        repetitions: int = 1,
        publish_evaluation_summary: bool = False,
        realtime: bool = True,
    ) -> None:
        """Run the complete workflow owned by this deployment loop stage."""
        trace_session = (
            EvaluationTraceSession.create(
                runtime=context.runtime,
                source=context.environment,
                framework="sb3",
            )
            if publish_evaluation_summary
            else None
        )
        try:
            for episode_index in range(repetitions):
                episode_return, episode_length = self._run_episode(
                    context,
                    episode_index=episode_index,
                    trace_session=trace_session,
                    realtime=realtime,
                )
                if publish_evaluation_summary:
                    writer = context.runtime.get_tensorboard_writer()
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

    @staticmethod
    def _run_episode(
        context: contracts.DeploymentLoopContext,
        *,
        episode_index: int,
        trace_session: EvaluationTraceSession | None,
        realtime: bool,
    ) -> tuple[float, int]:
        """Return the run episode value owned by this deployment loop stage."""
        context.runtime.check_stop()
        observation, _ = context.environment.reset()
        context.scheduler.reset()
        DeploymentTelemetryStage.publish_initial(
            context.runtime,
            context.environment,
            observation,
        )

        terminated = False
        episode_return = 0.0
        episode_length = 0
        step_stage = DeploymentInferenceStepStage()
        while not terminated:
            result = step_stage.run(context, observation)
            observation = result.observation
            terminated = result.terminated
            episode_return += result.reward
            episode_length += 1
            if trace_session is not None:
                trace_session.append_latest(
                    episode_index=episode_index,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            context.runtime.check_stop()
            if realtime:
                context.scheduler.delay()
        if trace_session is not None:
            trace_session.flush()
        return episode_return, episode_length
