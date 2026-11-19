# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Composable stages for one shared Stable-Baselines3 training run."""

from __future__ import annotations

from collections.abc import Callable
from types import MappingProxyType
from typing import cast

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from lanista.foundation.system import get_cpu_count
from lanista.reinforcengine.rl_environment.training_progress import (
    TrainingProgressPlan,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.sb3 import (
    training_pipeline_contracts as contracts,
)
from lanista.reinforcengine.rl_technologies.sb3.algorithm_config import (
    build_ppo_constructor_args,
    stable_baselines_configurator_name,
)
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_lifecycle import (
    StableBaselinesCheckpointMetadataCallback,
)
from lanista.reinforcengine.rl_technologies.sb3.evaluation import (
    build_training_eval_callback,
)
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    SB3ModelPolicy,
    SB3TrainingRestore,
)
from lanista.reinforcengine.rl_technologies.sb3.tensorboard_callback import (
    StableBaselinesTensorboardCallback,
)
from lanista.reinforcengine.rl_technologies.sb3.training_control_callback import (
    StableBaselinesTrainingControlCallback,
)
from lanista.reinforcengine.running_mode import RunningMode
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import (
    ScenarioManager,
)
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    apply_scenario_selection,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.curriculum_state_sb3 import (
    StableBaselinesCurriculumState,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.scenario_state_sb3 import (
    StableBaselinesScenarioState,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_manager import (
    CurriculumManager,
    resolve_curriculum_definition,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.scenario_switching_callback_sb3 import (
    StableBaselinesScenarioSwitchingCallback,
)
from lanista.reinforcengine.utils.tools.model_configurator_loader import (
    load_algorithm_configurator_module,
)


class TrainingPreparationStage:
    """Resolve the configurator and canonical checkpoint restore intent."""

    def prepare(
        self,
        *,
        runtime: BaseModelRuntime,
        model_policy: SB3ModelPolicy,
        models_directory: str,
        logger: contracts.SB3TrainingLogger,
        checkpoint_path_resolver: Callable[[str, str], str | None],
    ) -> contracts.TrainingPreparation:
        """Return the prepare value owned by this training preparation stage."""
        manifest = runtime.get_manifest()
        module_ref, configurator_module = (
            load_algorithm_configurator_module(
                manifest,
                logger,
            )
        )
        configuration = manifest.get_configuration()
        configuration.algorithm.configurator_module = module_ref
        restore = model_policy.resolve_training_restore(
            runtime,
            configuration,
            models_directory=models_directory,
            resolve_checkpoint_path=checkpoint_path_resolver,
        )
        return contracts.TrainingPreparation(
            manifest=manifest,
            configuration=configuration,
            configurator_module=cast(object, configurator_module),
            model_policy=model_policy,
            restore=restore,
        )


class TrainingStateStage:
    """Prepare and restore scenario/curriculum state in legacy order."""

    def prepare(
        self,
        *,
        preparation: contracts.TrainingPreparation,
        scenario_manager: ScenarioManager,
        logger: contracts.SB3TrainingLogger,
        publish_scenario_state: contracts.ScenarioStatePublisher,
        publish_curriculum_state: contracts.CurriculumStatePublisher,
    ) -> contracts.TrainingStates:
        """Return the prepare value owned by this training state stage."""
        apply_scenario_selection(
            preparation.configuration,
            scenario_manager,
            purpose="training",
            default_index=0,
            logger=logger,
        )

        scenario_state_factory = cast(
            Callable[[], contracts.ScenarioIndexStatePort],
            StableBaselinesScenarioState,
        )
        scenario_state = scenario_state_factory()
        scenario_manager_port = cast(
            contracts.TrainingScenarioManagerPort,
            scenario_manager,
        )
        scenario_manager_port.set_scenario_index_state(
            scenario_state
        )
        publish_scenario_state(scenario_state)
        self._restore_scenario(
            preparation.restore,
            scenario_manager,
            scenario_state,
            logger,
        )

        curriculum_definition = resolve_curriculum_definition(
            preparation.configuration.algorithm.training_conditions
        )
        curriculum_state_factory = cast(
            Callable[[], contracts.ReadableStatePort],
            StableBaselinesCurriculumState,
        )
        curriculum_state = curriculum_state_factory()
        publish_curriculum_state(curriculum_state)
        curriculum_manager = cast(
            contracts.CurriculumManagerPort,
            CurriculumManager(
                curriculum_definition.states,
                curriculum_index_state=curriculum_state,
                base_min_terminated_episodes=(
                    curriculum_definition.base_min_terminated_episodes
                ),
                parent_logger_name=logger.name,
                child_logger_name="curriculum_manager",
                child_logger_level=logger.level,
            ),
        )
        curriculum_manager.initialize()
        self._restore_curriculum(
            preparation.restore,
            curriculum_manager,
            logger,
        )
        return contracts.TrainingStates(
            scenario_index_state=scenario_state,
            curriculum_index_state=curriculum_state,
            curriculum_manager=curriculum_manager,
        )

    @staticmethod
    def _restore_scenario(
        restore: SB3TrainingRestore,
        scenario_manager: ScenarioManager,
        scenario_state: contracts.ScenarioIndexStatePort,
        logger: contracts.SB3TrainingLogger,
    ) -> None:
        """Restore the scenario owned by this training state stage."""
        if restore.scenario_index is None:
            return
        try:
            scenario_manager.set_index(restore.scenario_index)
            scenario_state.set(restore.scenario_index)
        except Exception as exc:
            logger.error(
                "event=checkpoint_scenario_restore_failed quality=invalid "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise

    @staticmethod
    def _restore_curriculum(
        restore: SB3TrainingRestore,
        curriculum_manager: contracts.CurriculumManagerPort,
        logger: contracts.SB3TrainingLogger,
    ) -> None:
        """Restore the curriculum owned by this training state stage."""
        if not restore.restore_curriculum_state:
            return
        try:
            curriculum_manager.restore_state(restore.curriculum_state)
        except Exception as exc:
            logger.error(
                "event=checkpoint_curriculum_restore_failed quality=invalid "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise


class TrainingEnvironmentStage:
    """Construct one supported environment profile and its timestep budget."""

    def create(
        self,
        *,
        preparation: contracts.TrainingPreparation,
        states: contracts.TrainingStates,
        scenario_manager: ScenarioManager,
        environment_type: type[object],
        logger: contracts.SB3TrainingLogger,
        publish_environment: contracts.EnvironmentPublisher,
    ) -> contracts.TrainingEnvironment:
        """Create a validated value owned by this training environment stage."""
        options: dict[str, object] = {
            "verbose": False,
            "detailed_verbose": False,
            "parent_logger_name": logger.name,
            "child_logger_name": "stable_baselines_sim_env",
            "rl_model_manifest": preparation.manifest,
            "rl_model_mode": RunningMode.TRAINING,
            "scenario_manager": cast(
                contracts.TrainingScenarioManagerPort,
                scenario_manager,
            ).clone(),
            "scenario_index_state": states.scenario_index_state,
            "curriculum_index_state": states.curriculum_index_state,
        }
        options.update(
            preparation.model_policy.training_environment_options(
                preparation.configuration,
                environment_type,
            )
        )

        profile = preparation.configuration.algorithm.running_profile
        if profile == "single":
            environment = environment_type(**options)
            publish_environment(environment)
            total_timesteps = self._base_timesteps(scenario_manager)
        elif profile == "vectorized":
            # SB3 stubs require a concrete Gym generic after plugin validation.
            cpu_count = int(get_cpu_count())
            environment = SubprocVecEnv(
                [
                    lambda: environment_type(**options)  # type: ignore[misc, return-value]
                    for _ in range(cpu_count)
                ]
            )
            publish_environment(environment)
            total_timesteps = (
                self._base_timesteps(scenario_manager) * cpu_count
            )
        elif profile == "dummy_vectorized":
            # SB3 stubs require a concrete Gym generic after plugin validation.
            cpu_count = int(get_cpu_count())
            environment = DummyVecEnv(
                [
                    lambda: environment_type(**options)  # type: ignore[misc, return-value]
                    for _ in range(cpu_count)
                ]
            )
            publish_environment(environment)
            total_timesteps = (
                self._base_timesteps(scenario_manager) * cpu_count
            )
        else:
            logger.error(f"Unknown running profile: {profile}")
            raise ValueError(f"Unknown running profile: {profile}")

        return contracts.TrainingEnvironment(
            environment=environment,
            environment_type=environment_type,
            constructor_options=MappingProxyType(dict(options)),
            total_timesteps=total_timesteps,
        )

    @staticmethod
    def _base_timesteps(scenario_manager: ScenarioManager) -> int:
        """Return the base timesteps value owned by this training environment stage."""
        manager = cast(
            contracts.TrainingScenarioManagerPort,
            scenario_manager,
        )
        return int(
            manager.get_configuration().get_max_timesteps()
            * manager.get_total_number_of_scenarios()
            * 1000
        )


class TrainingModelStage:
    """Load a continuation model or construct a fresh PPO algorithm."""

    def build(
        self,
        *,
        preparation: contracts.TrainingPreparation,
        environment: contracts.TrainingEnvironment,
        model_name: str,
        operations: contracts.TrainingModelOperations,
        logger: contracts.SB3TrainingLogger,
    ) -> contracts.TrainingModelResult:
        """Build the configured result owned by this training model stage."""
        restore = preparation.restore
        continue_training = restore.has_source_checkpoint
        continue_same_session = restore.source_matches_selected_session
        tensorboard_step_floor = self._tensorboard_step_floor(operations)
        if continue_training:
            model, source_base_step_offset = self._continue_model(
                preparation=preparation,
                environment=environment,
                operations=operations,
                logger=logger,
                tensorboard_step_floor=tensorboard_step_floor,
            )
        else:
            model, source_base_step_offset = self._fresh_model(
                preparation=preparation,
                environment=environment,
                model_name=model_name,
                operations=operations,
                logger=logger,
                tensorboard_step_floor=tensorboard_step_floor,
            )

        return contracts.TrainingModelResult(
            model=model,
            continue_training=continue_training,
            continue_same_session=continue_same_session,
            source_base_step_offset=source_base_step_offset,
        )

    @staticmethod
    def _tensorboard_step_floor(
        operations: contracts.TrainingModelOperations,
    ) -> int:
        """Normalize only concrete numeric values returned by a real runtime."""

        raw_floor = operations.resolve_tensorboard_step_floor()
        if not isinstance(raw_floor, int | float | str) or isinstance(raw_floor, bool):
            return 0
        try:
            return max(0, int(raw_floor or 0))
        except (TypeError, ValueError, OverflowError):
            return 0

    @staticmethod
    def _continue_model(
        *,
        preparation: contracts.TrainingPreparation,
        environment: contracts.TrainingEnvironment,
        operations: contracts.TrainingModelOperations,
        logger: contracts.SB3TrainingLogger,
        tensorboard_step_floor: int,
    ) -> tuple[contracts.SB3TrainingAlgorithm, int]:
        """Restore one checkpoint and preserve only same-session step continuity."""

        restore = preparation.restore
        source_session = cast(str, restore.source_session_name)
        source_checkpoint = cast(str, restore.source_checkpoint_name)
        model, loaded_step_count = operations.load_checkpoint(
            source_session,
            source_checkpoint,
        )
        operations.publish_model(model)
        source_base_step_offset = (
            max(loaded_step_count, tensorboard_step_floor)
            if restore.source_matches_selected_session
            else 0
        )
        if restore.source_matches_selected_session:
            model.num_timesteps = source_base_step_offset
        operations.publish_step_offset(source_base_step_offset)
        if restore.source_session_name != restore.selected_session_name:
            operations.retarget_session(restore.selected_session_name)
        model.set_env(environment.environment)
        operations.configure_tensorboard_logger()
        logger.info(
            f"Continuing training from session='{source_session}', "
            f"checkpoint='{source_checkpoint}' "
            f"into session='{restore.selected_session_name}' "
            f"with step_offset={source_base_step_offset}"
        )
        return model, source_base_step_offset

    @staticmethod
    def _fresh_model(
        *,
        preparation: contracts.TrainingPreparation,
        environment: contracts.TrainingEnvironment,
        model_name: str,
        operations: contracts.TrainingModelOperations,
        logger: contracts.SB3TrainingLogger,
        tensorboard_step_floor: int,
    ) -> tuple[contracts.SB3TrainingAlgorithm, int]:
        """Construct one new PPO algorithm in the resolved logical-session logdir."""

        tensorboard_log = operations.resolve_tensorboard_log_directory()
        if not tensorboard_log:
            raise RuntimeError(
                "Training TensorBoard log directory could not be resolved."
            )
        policy, configured_environment, ppo_kwargs = build_ppo_constructor_args(
            configurator_module=preparation.configurator_module,
            configurator_name=stable_baselines_configurator_name(
                preparation.configuration
            ),
            env_id=model_name,
            env=environment.environment,
            tensorboard_log=str(tensorboard_log),
            default_verbose=environment.constructor_options.get("verbose", True),
            logger=logger,
        )
        model = cast(
            contracts.SB3TrainingAlgorithm,
            PPO(policy, configured_environment, **ppo_kwargs),
        )
        operations.publish_model(model)
        operations.configure_tensorboard_logger()
        if tensorboard_step_floor:
            model.num_timesteps = tensorboard_step_floor
        return model, tensorboard_step_floor


class TrainingCallbackStage:
    """Assemble callbacks in the stable historical order."""

    def __init__(self) -> None:
        """Initialize training-session metric state shared across scenarios."""

        self._episode_info_cumulative_sums: dict[str, float] = {}

    def assemble(
        self,
        context: contracts.TrainingCallbackContext,
        *,
        publish_checkpoint_callback: contracts.CheckpointCallbackPublisher,
    ) -> contracts.TrainingCallbacks:
        """Return the assemble value owned by this training callback stage."""
        checkpoint_callback = self._checkpoint_callback(context)
        publish_checkpoint_callback(checkpoint_callback)
        raw_callbacks = self._base_callbacks(
            context,
            checkpoint_callback,
        )

        evaluation_callback = build_training_eval_callback(
            model_configuration=context.preparation.configuration,
            sim_env_cls=context.environment.environment_type,
            env_kwargs=dict(context.environment.constructor_options),
            scenario_manager=context.scenario_manager,
            scenario_state_factory=StableBaselinesScenarioState,
            logger=context.logger,
        )
        if evaluation_callback is not None:
            raw_callbacks.append(evaluation_callback)

        return contracts.TrainingCallbacks(
            callback_list=CallbackList(raw_callbacks),
            checkpoint_callback=checkpoint_callback,
        )

    @staticmethod
    def _checkpoint_callback(
        context: contracts.TrainingCallbackContext,
    ) -> StableBaselinesCheckpointMetadataCallback:
        """Checkpoint the callback owned by this training callback stage."""
        preparation = context.preparation
        return StableBaselinesCheckpointMetadataCallback(
            scenario_manager=context.scenario_manager,
            scenario_index_state=context.states.scenario_index_state,
            curriculum_index_state=context.states.curriculum_index_state,
            save_freq=preparation.configuration.checkpoint_saving_frequency,
            checkpoint_directory=context.checkpoint_directory,
            parent_checkpoint_name=(
                preparation.restore.source_checkpoint_name
            ),
            base_step_offset=context.model_result.source_base_step_offset,
            diagnostic_logger=context.logger,
            model_package_directory=context.model_package_directory,
        )

    def _base_callbacks(
        self,
        context: contracts.TrainingCallbackContext,
        checkpoint_callback: StableBaselinesCheckpointMetadataCallback,
    ) -> list[BaseCallback]:
        """Return the base callbacks value owned by this training callback stage."""
        preparation = context.preparation
        switching_options = preparation.model_policy.scenario_switching_options()
        progress_plan = TrainingProgressPlan.from_training_conditions(
            preparation.configuration.algorithm.training_conditions,
            total_scenarios=(
                context.scenario_manager.get_total_number_of_scenarios()
            ),
        )
        scenario_switching_callback = StableBaselinesScenarioSwitchingCallback(
            scenario_manager=context.scenario_manager,
            training_conditions=(
                preparation.configuration.algorithm.training_conditions
            ),
            curriculum_manager=context.states.curriculum_manager,
            learning_rate_resolver=switching_options.learning_rate_resolver,
            learning_rate_applier=switching_options.learning_rate_applier,
            progress_reporter=context.progress_reporter,
            training_progress_plan=progress_plan,
            parent_logger_name=context.logger.name,
            child_logger_name=(
                "stablebaselines_scenario_switching_callback"
            ),
            child_logger_level=context.logger.level,
        )
        tensorboard_options = preparation.model_policy.tensorboard_options()
        return [
            StableBaselinesTrainingControlCallback(context.runtime),
            checkpoint_callback,
            StableBaselinesTensorboardCallback(
                curriculum_index_state=context.states.curriculum_index_state,
                scenario_index_state=context.states.scenario_index_state,
                episode_info_cumulative_sums=(
                    self._episode_info_cumulative_sums
                ),
                record_rollout_stat_details=(
                    tensorboard_options.record_rollout_stat_details
                ),
            ),
            scenario_switching_callback,
        ]


class TrainingLoopStage:
    """Run one safety-bounded learn call across all metric milestones."""

    def run(
        self,
        context: contracts.TrainingLoopContext,
        *,
        callback_factory: Callable[[], contracts.TrainingCallbacks],
        publish_last_scenario: contracts.ScenarioPublisher,
    ) -> contracts.TrainingLoopResult:
        """Run the complete workflow owned by this training loop stage."""
        if context.scenario_manager.finished():
            context.logger.info("Training was already finished.")
            return contracts.TrainingLoopResult(
                iteration_count=0,
                last_scenario_index=context.scenario_manager.get_index(),
            )

        context.runtime.check_stop()
        callbacks = callback_factory()
        context.runtime.check_pause()
        context.runtime.check_stop()
        context.model_result.model.learn(
            total_timesteps=context.total_timesteps,
            callback=callbacks.callback_list,
            reset_num_timesteps=(
                not context.model_result.continue_training
                and context.model_result.source_base_step_offset == 0
            ),
        )

        context.runtime.check_stop()
        last_scenario_index = context.scenario_manager.get_index()
        publish_last_scenario(last_scenario_index)
        if not context.scenario_manager.finished():
            error = TrainingMilestoneLimitReachedError(
                total_timesteps=context.total_timesteps,
                scenario_index=last_scenario_index,
                total_scenarios=(
                    context.scenario_manager.get_total_number_of_scenarios()
                ),
            )
            context.logger.error(str(error))
            raise error

        context.logger.info("Training finished.")
        return contracts.TrainingLoopResult(
            iteration_count=1,
            last_scenario_index=last_scenario_index,
        )


class TrainingMilestoneLimitReachedError(RuntimeError):
    """Raised when SB3's safety budget ends before metric completion."""

    def __init__(
        self,
        *,
        total_timesteps: int,
        scenario_index: int,
        total_scenarios: int,
    ) -> None:
        """Describe the incomplete milestone rather than claiming completion."""

        super().__init__(
            "SB3 learn safety limit reached after "
            f"{total_timesteps} timesteps at scenario "
            f"{scenario_index + 1}/{total_scenarios}, before metric-based "
            "training completion. The run is failed and its periodic "
            "checkpoints remain available for continuation."
        )
