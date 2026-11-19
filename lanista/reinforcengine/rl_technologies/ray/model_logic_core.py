# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared Ray algorithm, checkpoint, curriculum, and deployment logic."""

import os
import pickle
from collections.abc import Callable
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import ray
from ray.rllib.algorithms import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune import register_env

from lanista.foundation.configuration import AppConfiguration, Configuration
from lanista.foundation.filesystem import makedirs
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_environment.rolling_stats_ray_rllib import RayRLlibRollingStats
from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    CheckpointMetadataError,
    load_checkpoint_restore_manifest,
)
from lanista.reinforcengine.rl_model.checkpoints.runtime_repository import (
    CheckpointArtifactKind,
    RuntimeCheckpointMutation,
    RuntimeCheckpointRepository,
)
from lanista.reinforcengine.rl_model.evaluation.evidence_runtime import (
    ScenarioEvidenceRuntime,
)
from lanista.reinforcengine.rl_model.execution_contracts import EvaluationVisualization
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime, ProgressCallback
from lanista.reinforcengine.rl_model.storage.names import (
    CheckpointStorageName,
    SessionStorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard
from lanista.reinforcengine.rl_technologies.ray.checkpoint_resolution import (
    RayCheckpointResolver,
    RayCheckpointSelection,
    RayCheckpointSource,
)
from lanista.reinforcengine.rl_technologies.ray.deployment_pipeline import (
    RayDeploymentEnvironment,
    RayDeploymentPipeline,
    RayDeploymentRuntime,
    RayDeploymentScheduler,
    RayInferenceModule,
)
from lanista.reinforcengine.rl_technologies.ray.episode_trace_capture import (
    RayEpisodeTraceCaptureGate,
)
from lanista.reinforcengine.rl_technologies.ray.model_policy import (
    CurriculumState,
    DefaultRayModelPolicy,
    RayCurriculumStateActor,
    RayModelPolicy,
)
from lanista.reinforcengine.rl_technologies.ray.ray_checkpoint_saving_callback import (
    RayCheckpointSavingCallback,
)
from lanista.reinforcengine.rl_technologies.ray.training_pipeline import (
    RayTrainingPipeline,
)
from lanista.reinforcengine.rl_technologies.ray.training_pipeline_contracts import (
    CurriculumDefinition as RayTrainingCurriculumDefinition,
)
from lanista.reinforcengine.rl_technologies.ray.training_pipeline_contracts import (
    RayRollingStatsFactory,
    RayTrainingConfiguration,
    RayTrainingContext,
    RayTrainingDependencies,
    RayTrainingLogger,
    RayTrainingOwnership,
    RayTrainingScenarioManager,
)
from lanista.reinforcengine.rl_technologies.runtime_contracts import DeploymentPurpose
from lanista.reinforcengine.running_mode import RunningMode
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import ScenarioManager
from lanista.reinforcengine.utils.scenarios.scenario_selection import apply_scenario_selection
from lanista.reinforcengine.utils.scenarios.scenario_state.curriculum_state_ray_rllib import (
    RayCurriculumIndexState,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.scenario_state_ray_rllib import (
    RayScenarioState,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_manager import (
    resolve_curriculum_definition,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_manager_ray_rllib import (
    RayCurriculumManager,
)
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.scenario_switching_callback_ray_rllib import (
    RayScenarioSwitchingCallback,
)
from lanista.reinforcengine.utils.tools.model_configurator_loader import (
    load_algorithm_configurator_module,
)
from lanista.reinforcengine.utils.tools.scheduler import Scheduler

APP_CONFIGURATION = AppConfiguration()


class _RayDeploymentConfiguration(Protocol):
    """Scenario configuration values required by the realtime scheduler."""

    def get_step_length(self) -> float:
        """Return the simulation step length in seconds."""


class _RayDeploymentScenarioManager(
    RayTrainingScenarioManager,
    Protocol,
):
    """Scenario operations required by the Ray deployment adapter."""

    def get_configuration(self) -> _RayDeploymentConfiguration:
        """Return scheduler-relevant scenario configuration."""


def _load_training_configurator(
    manifest: ModelManifest,
    logger: RayTrainingLogger,
) -> tuple[str, object]:
    """Adapt the validated dynamic loader to the training pipeline contract."""

    return load_algorithm_configurator_module(manifest, logger)


def _apply_training_scenario_selection(
    configuration: object,
    scenario_manager: RayTrainingScenarioManager,
    *,
    purpose: str,
    default_index: int,
    logger: RayTrainingLogger,
    selected_index: int | None = None,
) -> int | None:
    """Adapt the legacy scenario selector to its typed training boundary."""

    return apply_scenario_selection(
        configuration,
        scenario_manager,
        purpose=purpose,
        default_index=default_index,
        selected_index=selected_index,
        logger=logger,
    )


def _create_ray_state_actor(actor_type: object) -> RayCurriculumStateActor:
    """Create a Ray actor after validating the decorator-provided factory."""

    remote_factory = getattr(actor_type, "remote", None)
    if not callable(remote_factory):
        raise TypeError("Ray state actor type does not expose remote().")
    return cast(RayCurriculumStateActor, remote_factory())


def _create_scenario_state_actor() -> RayCurriculumStateActor:
    """Create the scenario state actor owned by this model logic core module."""
    return _create_ray_state_actor(RayScenarioState)


def _create_curriculum_state_actor() -> RayCurriculumStateActor:
    """Create the curriculum state actor owned by this model logic core module."""
    return _create_ray_state_actor(RayCurriculumIndexState)


def _create_episode_trace_capture_gate() -> object:
    """Create the atomic Ray actor shared by checkpoint and episode callbacks."""

    return RayEpisodeTraceCaptureGate.remote()


def _resolve_training_curriculum_definition(
    training_conditions: object,
) -> RayTrainingCurriculumDefinition:
    """Hide the resolver's optional extension hook from pipeline callers."""

    return cast(
        RayTrainingCurriculumDefinition,
        resolve_curriculum_definition(training_conditions),
    )


def _resolve_ray_object(reference: object) -> object:
    """Resolve the opaque reference returned by a Ray actor adapter."""

    return ray.get(cast("ray.ObjectRef[object]", reference))


def _instantiate_environment(
    environment_type: type[object],
    options: dict[str, object],
) -> object:
    """Invoke a plugin environment constructor at the dynamic kwargs boundary."""

    constructor = cast(Callable[..., object], environment_type)
    return constructor(**options)


_INFERENCE_MODULE_COMPONENTS = (
    "learner_group",
    "learner",
    "rl_module",
    "EGO_policy",
)


def _load_inference_module(checkpoint_path: str) -> RayInferenceModule:
    """Restore only the policy module required by standalone inference."""

    module_checkpoint_path = os.path.abspath(
        os.path.join(checkpoint_path, *_INFERENCE_MODULE_COMPONENTS)
    )
    if not os.path.isdir(module_checkpoint_path):
        raise CheckpointMetadataError(
            "Selected RLlib checkpoint has no EGO_policy module component."
        )
    module = RLModule.from_checkpoint(module_checkpoint_path)
    evaluation_mode = getattr(module, "eval", None)
    if callable(evaluation_mode):
        evaluation_mode()
    return cast(RayInferenceModule, module)


def _build_deployment_scheduler(
    interval_seconds: float,
) -> RayDeploymentScheduler:
    """Adapt the legacy untyped scheduler constructor to its narrow port."""

    scheduler_factory = cast(
        Callable[[float], RayDeploymentScheduler],
        Scheduler,
    )
    return scheduler_factory(interval_seconds)


def _create_checkpoint_callback(
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
    """Adapt callback actor handles across Ray's decorator typing boundary."""

    callback_factory = cast(
        Callable[..., object],
        RayCheckpointSavingCallback,
    )
    return callback_factory(
        scenario_index_state=scenario_index_state,
        save_freq=save_freq,
        checkpoint_directory=checkpoint_directory,
        name_prefix=name_prefix,
        curriculum_index_state=curriculum_index_state,
        parent_checkpoint_name=parent_checkpoint_name,
        base_step_offset=base_step_offset,
        model_package_directory=model_package_directory,
        episode_trace_capture_gate=episode_trace_capture_gate,
    )


def _is_picklable(
    value: object,
    *,
    logger: RayTrainingLogger,
) -> bool:
    """Return whether picklable applies to this model logic core module."""
    try:
        pickle.dumps(value)
        return True
    except (pickle.PickleError, AttributeError, TypeError, RecursionError) as exc:
        logger.warning(
            "event=ray_environment_payload_not_picklable quality=invalid "
            f"error_type={type(exc).__name__} error={exc}"
        )
        return False


# LoggerInterface is supplied by an external package without typing metadata.
class RayModelLogicCore(LoggerInterface):
    """Own the shared Ray algorithm, checkpoint, and deployment lifecycle."""

    def __init__(
        self,
        rl_model_wrapper: BaseModelRuntime,
        environment_type: type[object],
        model_policy: RayModelPolicy | None = None,
        **kwargs: object,
    ) -> None:
        """Create a Ray lifecycle configured by a model-specific policy."""
        LoggerInterface.__init__(self, **kwargs)
        self._rl_model_wrapper: BaseModelRuntime = rl_model_wrapper
        self._environment_type = environment_type
        self._model_policy = model_policy or DefaultRayModelPolicy()

        self._env: object | None = None
        self._algo: Algorithm | None = None

        self._model_name: str | None = None
        self._model_package_directory: str | None = None
        self._checkpoint_directory: str | None = None
        self._best_model_directory: str | None = None
        self._final_model_directory: str | None = None

        self._restoration_path: str | None = None
        self._last_train_result_stats: dict[str, object] | None = None
        self._last_scenario_index: int | None = None
        self._last_curriculum_state: CurriculumState = None
        self._loaded_checkpoint_name: str | None = None
        self._last_saved_checkpoint_name: str | None = None
        self._base_step_offset = 0
        self._evaluation_module: RayInferenceModule | None = None
        self._evaluation_checkpoint_path: str | None = None

        self._models_directory: str = self._get_saved_models_folder_path()
        makedirs(self._models_directory, exist_ok=True)

    def _active_model_policy(self) -> RayModelPolicy:
        """Return the configured strategy, including deserialized legacy state."""

        policy = getattr(self, "_model_policy", None)
        if policy is None:
            policy = DefaultRayModelPolicy()
            self._model_policy = policy
        return policy

    def _get_model_manifest(self) -> ModelManifest:
        """
        Get the model manifest from the RL model wrapper.

        :return: ModelManifest object containing configuration and metadata.
        """
        return self._rl_model_wrapper.get_manifest()

    def _get_model_manifest_configuration(self) -> Configuration:
        """
        Get the configuration from the model manifest.

        :return: Configuration object containing model settings.
        """
        return self._rl_model_wrapper.get_manifest().get_configuration()

    def _get_saved_models_folder_path(self) -> str:
        """Return the saved models folder path exposed by this ray model logic core."""
        model_root = self._get_model_manifest_configuration().model_folder_path
        return str(StoragePathGuard(model_root).resolve("training_sessions"))

    def _prepare_session_layout(
        self,
        session_name: SessionStorageName,
    ) -> tuple[str, str, str, str]:
        """Create and return one root-confined Ray session layout."""

        model_package_directory = StoragePathGuard(self._models_directory).resolve(session_name)
        makedirs(model_package_directory, exist_ok=True)
        session_guard = StoragePathGuard(model_package_directory)
        checkpoint_directory = session_guard.resolve("checkpoints")
        makedirs(checkpoint_directory, exist_ok=True)
        best_model_directory = session_guard.resolve("best_model")
        final_model_directory = session_guard.resolve("final_model")
        return (
            str(model_package_directory),
            str(checkpoint_directory),
            str(best_model_directory),
            str(final_model_directory),
        )

    def _configurate(self) -> None:
        """Publish a validated, root-confined Ray session layout."""

        session_name = SessionStorageName.parse(self._model_name or "")
        layout = self._prepare_session_layout(session_name)
        self._model_name = session_name.value
        (
            self._model_package_directory,
            self._checkpoint_directory,
            self._best_model_directory,
            self._final_model_directory,
        ) = layout

    def build(self) -> None:
        """
        Build the reinforcement learning model. Construct the environment and the model. Set configuration parameters.
        """

        self._model_name = str(
            getattr(self._get_model_manifest_configuration(), "selected_default_model_name", "")
            or ""
        )
        if not self._model_name:
            raise RuntimeError(
                "Training requires a selected session UUID before building the model package."
            )

        self._configurate()

    @property
    def runtime_quality_degradations(self) -> frozenset[str]:
        """Expose optional policy degradation signals without owning them."""

        return self._active_model_policy().quality_degradations

    def train(
        self,
        scenario_manager: ScenarioManager,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Train through the ordered, composed Ray lifecycle stages."""

        manifest = self._rl_model_wrapper.get_manifest()
        context = RayTrainingContext(
            manifest=manifest,
            configuration=cast(
                RayTrainingConfiguration,
                manifest.get_configuration(),
            ),
            runtime=self._rl_model_wrapper,
            logger=self._logger,
            policy=self._active_model_policy(),
            ownership=RayTrainingOwnership(
                publish_algorithm=self._publish_training_algorithm,
                publish_scenario_index=self._publish_training_scenario_index,
                publish_iteration=self._publish_training_iteration,
                publish_curriculum_state=(self._publish_training_curriculum_state),
                current_checkpoint_step=self._current_checkpoint_step,
            ),
            environment_type=self._environment_type,
            scenario_manager=cast(
                RayTrainingScenarioManager,
                scenario_manager,
            ),
            progress_callback=progress_callback,
            model_name=str(self._model_name),
            restoration_path=self._restoration_path,
            checkpoint_directory=str(self._checkpoint_directory),
            parent_checkpoint_name=self._loaded_checkpoint_name,
            base_step_offset=self._base_step_offset,
            model_package_directory=str(self._model_package_directory),
        )
        dependencies = RayTrainingDependencies(
            load_configurator=_load_training_configurator,
            apply_scenario_selection=_apply_training_scenario_selection,
            scenario_state_factory=_create_scenario_state_actor,
            curriculum_state_factory=_create_curriculum_state_actor,
            load_restore_manifest=load_checkpoint_restore_manifest,
            resolve_curriculum_definition=(_resolve_training_curriculum_definition),
            curriculum_manager_factory=RayCurriculumManager,
            is_picklable=_is_picklable,
            register_environment=register_env,
            checkpoint_callback_factory=_create_checkpoint_callback,
            episode_trace_capture_gate_factory=(_create_episode_trace_capture_gate),
            scenario_switching_callback=RayScenarioSwitchingCallback,
            rolling_stats_factory=cast(
                RayRollingStatsFactory,
                RayRLlibRollingStats,
            ),
            resolve_object=_resolve_ray_object,
        )
        RayTrainingPipeline().run(context, dependencies)

    def _publish_training_algorithm(self, algorithm: Algorithm) -> None:
        """Publish the training algorithm owned by this ray model logic core."""
        self._algo = algorithm

    def _publish_training_scenario_index(self, scenario_index: int) -> None:
        """Publish the training scenario index owned by this ray model logic core."""
        self._last_scenario_index = scenario_index

    def _publish_training_iteration(
        self,
        train_result: dict[str, object],
        scenario_index: int,
    ) -> None:
        """Publish the training iteration owned by this ray model logic core."""
        self._last_train_result_stats = train_result
        self._last_scenario_index = scenario_index

    def _publish_training_curriculum_state(
        self,
        curriculum_state: CurriculumState,
    ) -> None:
        """Publish the training curriculum state owned by this ray model logic core."""
        self._last_curriculum_state = curriculum_state

    def _current_checkpoint_step(self) -> int:
        """Return the current checkpoint step value owned by this ray model logic core."""
        return self._active_model_policy().checkpoint_step(self._last_train_result_stats)

    def _effective_checkpoint_step(self, raw_step: int) -> int:
        """Return the effective checkpoint step value owned by this ray model logic core."""
        raw_step = int(raw_step or 0)
        base_step_offset = int(getattr(self, "_base_step_offset", 0) or 0)
        if base_step_offset > 0 and raw_step <= base_step_offset:
            return base_step_offset + raw_step
        return raw_step

    def _current_checkpoint_scenario(self) -> int:
        """Return the current checkpoint scenario value owned by this ray model logic core."""
        try:
            if self._last_scenario_index is not None:
                return int(self._last_scenario_index)
        except (TypeError, ValueError):
            pass
        return 0

    def _build_checkpoint_name(
        self, scenario_num: int, step_num: int, *, manual: bool = False
    ) -> str:
        """Build the checkpoint name required by this ray model logic core."""
        base_name = f"scenario_{int(scenario_num)}_checkpoint_{int(step_num)}_steps"
        return f"{base_name}_manual" if manual else base_name

    def _reserve_checkpoint_path(self, checkpoint_name: str) -> str:
        """Return the reserve checkpoint path value owned by this ray model logic core."""
        validated_name = CheckpointStorageName.parse(checkpoint_name)
        checkpoint_guard = StoragePathGuard(str(self._checkpoint_directory))
        candidate = checkpoint_guard.resolve(validated_name)
        if not candidate.exists():
            return str(candidate)
        suffix = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        suffixed_name = CheckpointStorageName.parse(f"{validated_name.value}_{suffix}")
        return str(checkpoint_guard.resolve(suffixed_name))

    def _checkpoint_repository(self) -> RuntimeCheckpointRepository:
        """Checkpoint the repository owned by this ray model logic core."""
        return RuntimeCheckpointRepository(str(self._model_package_directory))

    def _update_last_checkpoint_metadata(
        self,
        mutation: RuntimeCheckpointMutation,
        checkpoint_name: str,
        checkpoint_kind: str,
    ) -> None:
        """Update the last checkpoint metadata owned by this ray model logic core."""
        try:
            mutation.update_last_checkpoint_metadata(
                checkpoint_name,
                checkpoint_kind,
                logger=self._logger,
                raise_on_error=True,
            )
        except Exception as exc:
            self._logger.error(
                "event=checkpoint_session_metadata_update_failed quality=invalid "
                f"error_type={type(exc).__name__} error={exc}"
            )
            raise

    def deploy(
        self,
        scenario_manager: ScenarioManager,
        *,
        purpose: DeploymentPurpose = "deployment",
        repetitions: int = 1,
        visualization: EvaluationVisualization = "gui",
        realtime: bool | None = None,
    ) -> None:
        """Deploy the restored policy through the ordered inference lifecycle."""

        if purpose == "evaluation" and visualization == "gui":
            repetitions = 1
        deployment_scenarios = cast(
            _RayDeploymentScenarioManager,
            scenario_manager,
        )
        preparation = self._prepare_deployment(
            deployment_scenarios,
            purpose=purpose,
            visualization=visualization,
        )
        env_kwargs = preparation
        module, environment, scheduler = self._build_deployment_resources(
            deployment_scenarios,
            env_kwargs,
        )
        RayDeploymentPipeline().run(
            runtime=cast(
                RayDeploymentRuntime,
                self._rl_model_wrapper,
            ),
            module=module,
            environment=environment,
            scheduler=scheduler,
            repetitions=repetitions,
            publish_evaluation_summary=purpose == "evaluation",
            realtime=visualization == "gui" if realtime is None else realtime,
            deterministic=self._deployment_deterministic_mode(),
        )
        self._logger.info("Deployment finished.")

    def _prepare_deployment(
        self,
        scenario_manager: _RayDeploymentScenarioManager,
        *,
        purpose: DeploymentPurpose,
        visualization: EvaluationVisualization,
    ) -> dict[str, object]:
        """Resolve scenario state and local deployment environment options."""

        manifest = self._rl_model_wrapper.get_manifest()
        configuration = manifest.get_configuration()
        _apply_training_scenario_selection(
            configuration,
            scenario_manager,
            purpose=purpose,
            default_index=20,
            logger=self._logger,
            selected_index=(
                self._rl_model_wrapper.get_evaluation_scenario_index()
                if purpose == "evaluation"
                else None
            ),
        )
        scenario_index_state = _create_scenario_state_actor()
        scenario_manager.set_scenario_index_state(scenario_index_state)
        env_kwargs: dict[str, object] = {
            "parent_logger_name": self._logger.name,
            "child_logger_name": "ray_sim_env",
            "rl_model_manifest": manifest,
            "rl_model_mode": (
                RunningMode.TESTING if visualization == "headless" else RunningMode.DEPLOYMENT
            ),
            "scenario_manager": scenario_manager.clone(),
            "scenario_index_state": scenario_index_state,
        }
        if purpose == "evaluation" and visualization == "gui":
            env_kwargs["defer_gui_shutdown_to_process_restart"] = True
        return env_kwargs

    def _build_deployment_resources(
        self,
        scenario_manager: _RayDeploymentScenarioManager,
        env_kwargs: dict[str, object],
    ) -> tuple[
        RayInferenceModule,
        RayDeploymentEnvironment,
        RayDeploymentScheduler,
    ]:
        """Restore the inference module and build local deployment resources."""

        module = _load_inference_module(cast(str, self._restoration_path))
        environment = cast(
            RayDeploymentEnvironment,
            _instantiate_environment(self._environment_type, env_kwargs),
        )
        self._env = environment
        scheduler = _build_deployment_scheduler(
            scenario_manager.get_configuration().get_step_length()
        )
        return module, environment, scheduler

    def _deployment_deterministic_mode(self) -> bool:
        """Return whether deployment should select deterministic policy actions."""

        return bool(
            getattr(self._get_model_manifest_configuration(), "evaluation_deterministic", True)
        )

    def evaluate_scenario_inference(
        self,
        *,
        scenario_manager: ScenarioManager,
        checkpoint_path: Path,
        runtime: ScenarioEvidenceRuntime,
        deterministic: bool,
        repetitions: int,
    ) -> str:
        """Run one held-out scenario through RLModule forward inference only."""

        resolved_checkpoint = str(Path(checkpoint_path).resolve())
        if self._evaluation_checkpoint_path != resolved_checkpoint:
            load_checkpoint_restore_manifest(resolved_checkpoint)
            self._evaluation_module = _load_inference_module(resolved_checkpoint)
            self._evaluation_checkpoint_path = resolved_checkpoint
        module = self._evaluation_module
        if module is None:
            raise RuntimeError("Ray evaluation RLModule was not restored.")

        configuration = runtime.get_manifest().get_configuration()
        deployment_scenarios = cast(_RayDeploymentScenarioManager, scenario_manager)
        _apply_training_scenario_selection(
            configuration,
            deployment_scenarios,
            purpose="evaluation",
            default_index=0,
            selected_index=0,
            logger=self._logger,
        )
        scenario_index_state = _create_scenario_state_actor()
        deployment_scenarios.set_scenario_index_state(scenario_index_state)
        environment = cast(
            RayDeploymentEnvironment,
            _instantiate_environment(
                self._environment_type,
                {
                    "parent_logger_name": self._logger.name,
                    "child_logger_name": "ray_evaluation_library_env",
                    "rl_model_manifest": runtime.get_manifest(),
                    "rl_model_mode": RunningMode.TESTING,
                    "scenario_manager": deployment_scenarios.clone(),
                    "scenario_index_state": scenario_index_state,
                },
            ),
        )
        self._env = environment
        scheduler = _build_deployment_scheduler(
            deployment_scenarios.get_configuration().get_step_length()
        )
        try:
            RayDeploymentPipeline().run(
                runtime=cast(RayDeploymentRuntime, runtime),
                module=module,
                environment=environment,
                scheduler=scheduler,
                repetitions=repetitions,
                publish_evaluation_summary=True,
                realtime=False,
                deterministic=deterministic,
            )
        finally:
            close_environment = getattr(environment, "close", None)
            if callable(close_environment):
                close_environment()
            self._env = None
            with suppress(AttributeError, RuntimeError, TypeError, ValueError):
                ray.kill(cast(object, scenario_index_state), no_restart=True)
        return "completed"

    def save(
        self,
        checkpoint: bool = False,
        manual: bool = False,
    ) -> str:
        """
        Save the already trained reinforcement learning model into a file.
        """

        self._logger.info("Saving model...")
        scenario_num = self._current_checkpoint_scenario()
        step_num = self._effective_checkpoint_step(self._current_checkpoint_step())
        checkpoint_kind = "manual" if manual else ("checkpoint" if checkpoint else "final")
        with self._checkpoint_repository().mutation() as mutation:
            checkpoint_name = self._build_checkpoint_name(
                scenario_num,
                step_num,
                manual=manual,
            )
            final_target_path = self._reserve_checkpoint_path(checkpoint_name)
            target_path = mutation.prepare_checkpoint_target(
                final_target_path,
                artifact_kind=CheckpointArtifactKind.DIRECTORY,
            )
            resolved_name = os.path.basename(final_target_path)
            algorithm = self._algo
            if algorithm is None:
                raise RuntimeError("Cannot save before a Ray algorithm is built.")
            algorithm.save(target_path)
            mutation.write_checkpoint_manifest(
                target_path,
                checkpoint_name=resolved_name,
                checkpoint_kind=checkpoint_kind,
                framework_uuid="ray",
                scenario_index=scenario_num,
                curriculum_state=getattr(self, "_last_curriculum_state", None),
                step_count=step_num,
                parent_checkpoint_name=(
                    getattr(self, "_last_saved_checkpoint_name", None)
                    or getattr(self, "_loaded_checkpoint_name", None)
                ),
            )
            self._update_last_checkpoint_metadata(
                mutation,
                resolved_name,
                checkpoint_kind,
            )
        self._last_saved_checkpoint_name = resolved_name
        self._logger.info(f"Model saved to {final_target_path}")
        return resolved_name

    def load(
        self,
        model_name: object | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """Validate and atomically publish one managed Ray checkpoint."""

        self._logger.info("Loading model...")
        if model_name is None:
            model_name = self._get_model_manifest_configuration().selected_default_model_name
            self._logger.info(
                f"No model name given, using the default one from configuration: {model_name}"
            )
        session_name = SessionStorageName.parse(str(model_name or ""))
        selected_checkpoint = (
            CheckpointStorageName.parse(checkpoint_name) if checkpoint_name is not None else None
        )
        layout = self._prepare_session_layout(session_name)
        resolver = RayCheckpointResolver(
            final_directory=layout[3],
            checkpoint_directory=layout[1],
        )
        selection = resolver.resolve(selected_checkpoint)
        if selection is None and selected_checkpoint is not None:
            self._logger.error(
                f"No restorable RLlib checkpoint found for given name: {selected_checkpoint.value}"
            )
            raise CheckpointMetadataError("Selected checkpoint artifact is unavailable.")
        if selection is None:
            self._logger.error("No restorable RLlib checkpoint found by auto-detection.")
            raise FileNotFoundError
        checkpoint_metadata = load_checkpoint_restore_manifest(str(selection.path))
        base_step_offset = checkpoint_metadata["step_count"]
        self._publish_checkpoint_selection(
            session_name,
            layout,
            selection,
            base_step_offset,
        )
        self._log_checkpoint_selection(selection)

    def _publish_checkpoint_selection(
        self,
        session_name: SessionStorageName,
        layout: tuple[str, str, str, str],
        selection: RayCheckpointSelection,
        base_step_offset: int,
    ) -> None:
        """Publish only after artifact metadata passed schema validation."""

        self._model_name = session_name.value
        (
            self._model_package_directory,
            self._checkpoint_directory,
            self._best_model_directory,
            self._final_model_directory,
        ) = layout
        self._loaded_checkpoint_name = (
            selection.checkpoint_name.value if selection.is_intermediate else None
        )
        self._base_step_offset = base_step_offset
        self._restoration_path = str(selection.path)

    def _log_checkpoint_selection(
        self,
        selection: RayCheckpointSelection,
    ) -> None:
        """Log the checkpoint selection owned by this ray model logic core."""
        messages = {
            RayCheckpointSource.EXPLICIT_FINAL: ("Using the final model as selected checkpoint"),
            RayCheckpointSource.EXPLICIT_INTERMEDIATE: (
                "Using an intermediate model as selected checkpoint"
            ),
            RayCheckpointSource.LATEST_FINAL: (
                "Using the latest final model checkpoint based on auto-detection"
            ),
            RayCheckpointSource.LATEST_INTERMEDIATE: (
                "Using the latest intermediate model checkpoint based on auto-detection"
            ),
        }
        self._logger.info(f"{messages[selection.source]}: {selection.path}")
