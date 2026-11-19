# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Composed stages for the shared Ray training lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from ray.rllib.algorithms import Algorithm

from lanista.reinforcengine.rl_environment.training_progress import (
    TrainingProgressPlan,
)
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_technologies.ray import (
    training_pipeline_contracts as contracts,
)
from lanista.reinforcengine.rl_technologies.ray.model_policy import (
    CurriculumState,
    RayCurriculumStateActor,
    RayTrainingWindowContext,
)
from lanista.reinforcengine.running_mode import RunningMode


@dataclass(frozen=True, slots=True)
class RayCheckpointCallbackProvider:
    """Create callbacks without retaining the host-side training context."""

    callback_factory: contracts.RayCheckpointCallbackFactory
    scenario_index_state: RayCurriculumStateActor
    save_frequency: int
    checkpoint_directory: str
    curriculum_index_state: RayCurriculumStateActor
    parent_checkpoint_name: str | None
    base_step_offset: int
    model_package_directory: str
    episode_trace_capture_gate: object

    def __call__(self) -> object:
        """Invoke this ray checkpoint callback provider as a callable."""
        return self.callback_factory(
            scenario_index_state=self.scenario_index_state,
            save_freq=self.save_frequency,
            checkpoint_directory=self.checkpoint_directory,
            name_prefix="scenario_",
            curriculum_index_state=self.curriculum_index_state,
            parent_checkpoint_name=self.parent_checkpoint_name,
            base_step_offset=self.base_step_offset,
            model_package_directory=self.model_package_directory,
            episode_trace_capture_gate=self.episode_trace_capture_gate,
        )


class RayConfiguratorSelectionStage:
    """Load the configurator and select the initial training scenario."""

    def prepare(
        self,
        context: contracts.RayTrainingContext,
        dependencies: contracts.RayTrainingDependencies,
    ) -> contracts.RayConfiguratorSelectionResult:
        """Return the prepare value owned by this ray configurator selection stage."""
        algorithm = context.configuration.algorithm
        configurator_name = algorithm.components.training_configurator
        module_ref, configurator_module = dependencies.load_configurator(
            context.manifest,
            context.logger,
        )
        algorithm.configurator_module = module_ref
        training_conditions = algorithm.training_conditions

        context.logger.debug(
            "Number of scenarios:" + str(context.scenario_manager.get_total_number_of_scenarios())
        )
        dependencies.apply_scenario_selection(
            context.configuration,
            context.scenario_manager,
            purpose="training",
            default_index=context.policy.training_scenario_default_index,
            logger=context.logger,
        )
        context.logger.info(
            "Successfully set the starting scenario index to "
            f"{context.scenario_manager.get_index()} for training."
        )
        return contracts.RayConfiguratorSelectionResult(
            configurator_module=configurator_module,
            configurator_name=configurator_name,
            training_conditions=training_conditions,
        )


class RayActorCurriculumRestoreStage:
    """Create actor state and restore checkpoint scenario/curriculum data."""

    def prepare(
        self,
        context: contracts.RayTrainingContext,
        selection: contracts.RayConfiguratorSelectionResult,
        dependencies: contracts.RayTrainingDependencies,
    ) -> contracts.RayActorRestoreResult:
        """Return the prepare value owned by this ray actor curriculum restore stage."""
        scenario_state = dependencies.scenario_state_factory()
        context.scenario_manager.set_scenario_index_state(scenario_state)
        curriculum_state = dependencies.curriculum_state_factory()
        episode_trace_capture_gate = (
            dependencies.episode_trace_capture_gate_factory()
        )
        restored_metadata = (
            dependencies.load_restore_manifest(context.restoration_path)
            if context.restoration_path
            else {}
        )
        self._restore_scenario(
            context,
            scenario_state,
            restored_metadata,
            dependencies,
        )
        curriculum_manager = self._create_curriculum_manager(
            context,
            selection.training_conditions,
            curriculum_state,
            dependencies,
        )
        self._restore_curriculum(
            context,
            curriculum_manager,
            curriculum_state,
            restored_metadata,
        )
        return contracts.RayActorRestoreResult(
            scenario_index_state=scenario_state,
            curriculum_index_state=curriculum_state,
            curriculum_manager=curriculum_manager,
            episode_trace_capture_gate=episode_trace_capture_gate,
        )

    @staticmethod
    def _restore_scenario(
        context: contracts.RayTrainingContext,
        scenario_state: RayCurriculumStateActor,
        restored_metadata: dict[str, object],
        dependencies: contracts.RayTrainingDependencies,
    ) -> None:
        """Restore the scenario owned by this ray actor curriculum restore stage."""
        restored_index = restored_metadata.get("scenario_index")
        if restored_index is None:
            return
        try:
            normalized_index = int(cast(int | str, restored_index))
            context.scenario_manager.set_index(normalized_index)
            dependencies.resolve_object(scenario_state.set.remote(normalized_index))
            context.ownership.publish_scenario_index(normalized_index)
        except Exception as exc:
            context.logger.error(
                "event=checkpoint_scenario_restore_failed quality=invalid "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise

    @staticmethod
    def _create_curriculum_manager(
        context: contracts.RayTrainingContext,
        training_conditions: object,
        curriculum_state: RayCurriculumStateActor,
        dependencies: contracts.RayTrainingDependencies,
    ) -> contracts.RayCurriculumManagerPort:
        """Create the curriculum manager owned by this ray actor curriculum restore stage."""
        definition = dependencies.resolve_curriculum_definition(training_conditions)
        manager = dependencies.curriculum_manager_factory(
            definition.states,
            curriculum_index_state=curriculum_state,
            base_min_terminated_episodes=(definition.base_min_terminated_episodes),
            parent_logger_name=context.logger.name,
            child_logger_name="curriculum_manager",
            child_logger_level=context.logger.level,
        )
        manager.initialize()
        return manager

    @staticmethod
    def _restore_curriculum(
        context: contracts.RayTrainingContext,
        curriculum_manager: contracts.RayCurriculumManagerPort,
        curriculum_state: RayCurriculumStateActor,
        restored_metadata: dict[str, object],
    ) -> None:
        """Restore the curriculum owned by this ray actor curriculum restore stage."""
        if "curriculum_state" not in restored_metadata:
            return
        try:
            restored_state = cast(
                CurriculumState,
                restored_metadata["curriculum_state"],
            )
            published_state = context.policy.restore_curriculum_state(
                curriculum_manager,
                curriculum_state,
                restored_state,
            )
            context.ownership.publish_curriculum_state(published_state)
        except Exception as exc:
            context.logger.error(
                "event=checkpoint_curriculum_restore_failed quality=invalid "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise


class RayEnvironmentRegistrationStage:
    """Register the serializable training environment creator."""

    def register(
        self,
        context: contracts.RayTrainingContext,
        actors: contracts.RayActorRestoreResult,
        dependencies: contracts.RayTrainingDependencies,
    ) -> bool:
        """Return the register value owned by this ray environment registration stage."""
        environment_options: dict[str, object] = {
            "parent_logger_name": context.logger.name,
            "child_logger_name": "ray_sim_env",
            "rl_model_manifest": context.manifest,
            "rl_model_mode": RunningMode.TRAINING,
            "scenario_manager": context.scenario_manager.clone(),
            "scenario_index_state": actors.scenario_index_state,
            "curriculum_index_state": actors.curriculum_index_state,
            "diagnostic_trace_capture_gate": (
                actors.episode_trace_capture_gate
            ),
        }
        environment_options.update(
            context.policy.training_environment_options(
                context.configuration,
                context.environment_type,
            )
        )
        if not dependencies.is_picklable(
            context.manifest,
            logger=context.logger,
        ):
            context.logger.critical(
                "RL model wrapper is not picklable. Cannot register environment."
            )
            return False

        environment_type = cast(Callable[..., object], context.environment_type)

        def create_environment(_: object) -> object:
            """Create the environment owned by this ray environment registration stage."""
            return environment_type(**environment_options)

        dependencies.register_environment(
            context.model_name,
            create_environment,
        )
        return True


class RayAlgorithmBuildStage:
    """Configure callbacks and construct the RLlib algorithm."""

    def build(
        self,
        context: contracts.RayTrainingContext,
        selection: contracts.RayConfiguratorSelectionResult,
        actors: contracts.RayActorRestoreResult,
        dependencies: contracts.RayTrainingDependencies,
    ) -> contracts.RayAlgorithmBuildResult:
        """Build the configured result owned by this ray algorithm build stage."""
        configurator = cast(
            contracts.RayAlgorithmConfigurator,
            getattr(
                selection.configurator_module,
                selection.configurator_name,
            ),
        )
        save_frequency = context.policy.checkpoint_frequency(
            context.configuration.checkpoint_saving_frequency,
            context.logger,
        )

        checkpoint_callback_provider = RayCheckpointCallbackProvider(
            callback_factory=dependencies.checkpoint_callback_factory,
            scenario_index_state=actors.scenario_index_state,
            save_frequency=save_frequency,
            checkpoint_directory=context.checkpoint_directory,
            curriculum_index_state=actors.curriculum_index_state,
            parent_checkpoint_name=context.parent_checkpoint_name,
            base_step_offset=context.base_step_offset,
            model_package_directory=context.model_package_directory,
            episode_trace_capture_gate=(
                actors.episode_trace_capture_gate
            ),
        )

        algorithm_configuration = configurator(context.model_name)
        algorithm_configuration.callbacks(
            [
                dependencies.scenario_switching_callback,
                checkpoint_callback_provider,
            ]
        )
        return contracts.RayAlgorithmBuildResult(algorithm=algorithm_configuration.build_algo())


class RayAlgorithmReadyStage:
    """Restore the built algorithm and apply the model-specific ready hook."""

    def prepare(
        self,
        context: contracts.RayTrainingContext,
        actors: contracts.RayActorRestoreResult,
        build: contracts.RayAlgorithmBuildResult,
    ) -> None:
        """Prepare the operation owned by this ray algorithm ready stage."""
        if context.restoration_path:
            context.logger.info(f"Restoring training from checkpoint: {context.restoration_path}")
            cast(contracts.RayTrainingAlgorithm, build.algorithm).restore(context.restoration_path)
        context.policy.on_algorithm_ready(
            build.algorithm,
            actors.curriculum_manager.current_state(),
            context.logger,
        )


class RayTrainingWindowStage:
    """Execute the ordered stop/progress/train/statistics training loop."""

    def run(
        self,
        context: contracts.RayTrainingContext,
        actors: contracts.RayActorRestoreResult,
        build: contracts.RayAlgorithmBuildResult,
        selection: contracts.RayConfiguratorSelectionResult,
        dependencies: contracts.RayTrainingDependencies,
    ) -> None:
        """Run the complete workflow owned by this ray training window stage."""
        rolling_stats = dependencies.rolling_stats_factory(
            selection.training_conditions,
            parent_logger_name=context.logger.name,
            child_logger_name="rolling_stats",
            child_logger_level=context.logger.level,
        )
        progress_plan = TrainingProgressPlan.from_training_conditions(
            selection.training_conditions,
            total_scenarios=(
                context.scenario_manager.get_total_number_of_scenarios()
            ),
        )
        finished = context.scenario_manager.finished()
        self._publish_progress(
            context,
            actors,
            rolling_stats,
            progress_plan,
            finished=finished,
        )
        while not finished:
            self._run_iteration(
                context,
                actors,
                build.algorithm,
                rolling_stats,
            )
            finished = context.scenario_manager.finished()
            self._publish_progress(
                context,
                actors,
                rolling_stats,
                progress_plan,
                finished=finished,
            )
        context.logger.info("Training finished.")

    @staticmethod
    def _run_iteration(
        context: contracts.RayTrainingContext,
        actors: contracts.RayActorRestoreResult,
        algorithm: Algorithm,
        rolling_stats: contracts.RayTrainingRollingStats,
    ) -> None:
        """Run the iteration owned by this ray training window stage."""
        context.runtime.check_stop()
        context.runtime.check_pause()
        context.runtime.check_stop()

        train_result = cast(contracts.RayTrainingAlgorithm, algorithm).train()
        scenario_index = context.scenario_manager.get_index()
        context.ownership.publish_iteration(train_result, scenario_index)
        context.policy.record_training_result(
            context.runtime,
            train_result,
            context.logger,
        )
        new_episodes = rolling_stats.add_episode(train_result)
        if new_episodes:
            rolling_stats.log_window_metric_stats()
            metrics_ready = rolling_stats.metrics_ready()
            curriculum_state = context.policy.process_training_window(
                RayTrainingWindowContext(
                    algorithm=algorithm,
                    runtime=context.runtime,
                    logger=context.logger,
                    scenario_manager=context.scenario_manager,
                    curriculum_manager=actors.curriculum_manager,
                    rolling_stats=rolling_stats,
                    train_result=train_result,
                    env_steps=context.ownership.current_checkpoint_step(),
                    metrics_ready=metrics_ready,
                )
            )
            context.ownership.publish_curriculum_state(curriculum_state)
        context.runtime.check_stop()

    @staticmethod
    def _publish_progress(
        context: contracts.RayTrainingContext,
        actors: contracts.RayActorRestoreResult,
        rolling_stats: contracts.RayTrainingRollingStats,
        progress_plan: TrainingProgressPlan,
        *,
        finished: bool,
    ) -> None:
        """Publish shared milestone progress at a native Ray window boundary."""
        if context.progress_callback is None:
            return
        current_state = actors.curriculum_manager.current_state()
        projection = progress_plan.project(
            scenario_index=context.scenario_manager.get_index(),
            curriculum_state=current_state,
            evidence=rolling_stats.training_progress_evidence(current_state),
            finished=finished,
        )
        context.progress_callback(
            100.0,
            projection.percentage,
            RLModelState.RUNNING,
            projection.message,
        )


class RayTrainingPipeline:
    """Compose the focused stages without taking algorithm ownership."""

    def __init__(self) -> None:
        """Initialize the ray training pipeline and its owned state."""
        self._selection_stage = RayConfiguratorSelectionStage()
        self._restore_stage = RayActorCurriculumRestoreStage()
        self._environment_stage = RayEnvironmentRegistrationStage()
        self._build_stage = RayAlgorithmBuildStage()
        self._ready_stage = RayAlgorithmReadyStage()
        self._window_stage = RayTrainingWindowStage()

    def run(
        self,
        context: contracts.RayTrainingContext,
        dependencies: contracts.RayTrainingDependencies,
    ) -> contracts.RayTrainingResult:
        """Run the complete workflow owned by this ray training pipeline."""
        selection = self._selection_stage.prepare(context, dependencies)
        actors = self._restore_stage.prepare(
            context,
            selection,
            dependencies,
        )
        if not self._environment_stage.register(
            context,
            actors,
            dependencies,
        ):
            return contracts.RayTrainingResult(
                algorithm=None,
                environment_registered=False,
                completed=False,
            )
        build = self._build_stage.build(
            context,
            selection,
            actors,
            dependencies,
        )
        # Ownership must become visible before restore, ready hooks, or learning.
        context.ownership.publish_algorithm(build.algorithm)
        self._ready_stage.prepare(context, actors, build)
        self._window_stage.run(
            context,
            actors,
            build,
            selection,
            dependencies,
        )
        return contracts.RayTrainingResult(
            algorithm=build.algorithm,
            environment_registered=True,
            completed=True,
        )
