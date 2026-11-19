# Copyright (c) 2024-2026 Benedek Szanyó.
# Copyright (c) 2024-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Shared SB3 algorithm, checkpoint, curriculum, and deployment logic."""

import os
from pathlib import Path
from typing import TypedDict, cast

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure as configure_sb3_logger

from lanista.foundation.configuration import AppConfiguration, Configuration
from lanista.foundation.filesystem import existdir, makedirs
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_model.checkpoints.metadata import (
    CheckpointMetadataError,
    load_checkpoint_restore_manifest,
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
from lanista.reinforcengine.rl_technologies.runtime_contracts import DeploymentPurpose
from lanista.reinforcengine.rl_technologies.training_device import sb3_device
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_lifecycle import (
    StableBaselinesCheckpointMetadataCallback,
    save_sb3_checkpoint,
)
from lanista.reinforcengine.rl_technologies.sb3.checkpoint_resolution import (
    list_sb3_checkpoint_paths,
    resolve_sb3_checkpoint_path,
)
from lanista.reinforcengine.rl_technologies.sb3.deployment_pipeline import (
    DeploymentEnvironmentStage,
    DeploymentLoopStage,
    DeploymentPreparationStage,
)
from lanista.reinforcengine.rl_technologies.sb3.deployment_pipeline_contracts import (
    DeploymentLoopContext,
    SB3DeploymentAlgorithm,
    SB3DeploymentRuntime,
)
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    DefaultSB3ModelPolicy,
    SB3CheckpointObservationProvider,
    SB3CheckpointObservationRow,
    SB3CheckpointObservationUnavailable,
    SB3EvaluationContext,
    SB3EvaluationPolicy,
    SB3ModelPolicy,
)
from lanista.reinforcengine.rl_technologies.sb3.training_pipeline import (
    TrainingCallbackStage,
    TrainingEnvironmentStage,
    TrainingLoopStage,
    TrainingModelStage,
    TrainingPreparationStage,
    TrainingStateStage,
)
from lanista.reinforcengine.rl_technologies.sb3.training_pipeline_contracts import (
    ReadableStatePort,
    SB3TrainingAlgorithm,
    SB3TrainingLogger,
    ScenarioIndexStatePort,
    TrainingCallbackContext,
    TrainingLoopContext,
    TrainingModelOperations,
)
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import ScenarioManager

APP_CONFIGURATION = AppConfiguration()


class _CheckpointLoadOptions(TypedDict, total=False):
    """Typed SB3 options accepted while restoring a checkpoint."""

    tensorboard_log: str


class SB3ModelLogicCore(LoggerInterface):
    """Own the shared SB3 algorithm, checkpoint, and environment lifecycle."""

    def __init__(
        self,
        rl_model_wrapper: BaseModelRuntime,
        environment_type: type[object],
        model_policy: SB3ModelPolicy | None = None,
        evaluation_policy: SB3EvaluationPolicy | None = None,
        **kwargs: object,
    ) -> None:
        """Create one lifecycle with model-specific policies."""
        LoggerInterface.__init__(self, **kwargs)
        self._logger: SB3TrainingLogger
        self._rl_model_wrapper: BaseModelRuntime = rl_model_wrapper
        self._environment_type: type[object] = environment_type
        self._model_policy: SB3ModelPolicy = model_policy or DefaultSB3ModelPolicy()
        self._evaluation_policy: SB3EvaluationPolicy | None = evaluation_policy

        self._env: object | None = None
        self._model: SB3TrainingAlgorithm | None = None

        self._model_name: str | None = None
        self._model_package_directory: str | None = None

        self._checkpoint_directory: str | None = None

        self._loaded_checkpoint_name: str | None = None
        self._loaded_checkpoint_step_count = 0
        self._loaded_checkpoint_path: str | None = None
        self._training_step_offset: int | None = None
        self._last_scenario_index: int | None = None
        self._last_saved_checkpoint_name: str | None = None
        self._scenario_index_state: ScenarioIndexStatePort | None = None
        self._curriculum_index_state: ReadableStatePort | None = None
        self._checkpoint_metadata_callback: StableBaselinesCheckpointMetadataCallback | None = None
        self._runtime_quality_degradations: set[str] = set()
        self._evaluation_model: SB3DeploymentAlgorithm | None = None
        self._evaluation_checkpoint_path: str | None = None

        self._models_directory: str = self._get_saved_models_folder_path()
        makedirs(self._models_directory, exist_ok=True)

    def _active_model_policy(self) -> SB3ModelPolicy:
        """Return the active model policy value owned by this sb3 model logic core."""
        policy: SB3ModelPolicy | None = getattr(
            self,
            "_model_policy",
            None,
        )
        if policy is None:
            policy = DefaultSB3ModelPolicy()
            self._model_policy = policy
        return policy

    def _active_evaluation_policy(self) -> SB3EvaluationPolicy:
        """Return the active evaluation policy value owned by this sb3 model logic core."""
        policy: SB3EvaluationPolicy | None = getattr(
            self,
            "_evaluation_policy",
            None,
        )
        if policy is None:
            raise RuntimeError("This SB3 model does not provide a diagnostic evaluation policy.")
        return policy

    def _configure_training_tensorboard_logger(self) -> None:
        """Configure the training tensorboard logger owned by this sb3 model logic core."""
        tensorboard_log_dir = self._rl_model_wrapper.get_resolved_tensorboard_log_dir()
        if not tensorboard_log_dir or self._model is None:
            return
        self._model.set_logger(configure_sb3_logger(str(tensorboard_log_dir), ["tensorboard"]))

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
        """Return the saved models folder path exposed by this sb3 model logic core."""
        model_root = self._get_model_manifest_configuration().model_folder_path
        return str(StoragePathGuard(model_root).resolve("training_sessions"))

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

    def _configurate(self) -> None:
        """
        Configurate the model package directory and its subdirectories.
        """

        session_name = SessionStorageName.parse(self._model_name or "")
        self._model_name = session_name.value
        model_package_directory = StoragePathGuard(self._models_directory).resolve(session_name)
        self._model_package_directory = str(model_package_directory)
        makedirs(self._model_package_directory, exist_ok=True)

        self._checkpoint_directory = str(
            StoragePathGuard(model_package_directory).resolve("checkpoints")
        )
        makedirs(self._checkpoint_directory, exist_ok=True)

    @staticmethod
    def _checkpoint_step_count(checkpoint_path: str) -> int:
        """Checkpoint the step count owned by this sb3 model logic core."""
        manifest = load_checkpoint_restore_manifest(checkpoint_path)
        return int(manifest["step_count"])

    def _restore_checkpoint(
        self,
        checkpoint_path: str,
        *,
        load_kwargs: _CheckpointLoadOptions,
    ) -> None:
        """Validate metadata and binary before publishing restored state."""

        checkpoint_metadata = load_checkpoint_restore_manifest(checkpoint_path)
        loaded_model = PPO.load(checkpoint_path, device=sb3_device(), **load_kwargs)
        candidate_model = cast(SB3TrainingAlgorithm, loaded_model)
        self._model = candidate_model
        self._loaded_checkpoint_name = os.path.basename(checkpoint_path)
        self._loaded_checkpoint_step_count = int(checkpoint_metadata["step_count"])
        self._loaded_checkpoint_path = checkpoint_path

    def _publish_training_environment(self, environment: object) -> None:
        """Publish the training environment owned by this sb3 model logic core."""
        self._env = environment

    def _publish_training_scenario_state(
        self,
        state: ScenarioIndexStatePort,
    ) -> None:
        """Publish the training scenario state owned by this sb3 model logic core."""
        self._scenario_index_state = state

    def _publish_training_curriculum_state(
        self,
        state: ReadableStatePort,
    ) -> None:
        """Publish the training curriculum state owned by this sb3 model logic core."""
        self._curriculum_index_state = state

    def _publish_training_model(
        self,
        model: SB3TrainingAlgorithm,
    ) -> None:
        """Publish the training model owned by this sb3 model logic core."""
        self._model = model

    def _publish_training_step_offset(self, step_offset: int) -> None:
        """Publish the training step offset owned by this sb3 model logic core."""
        self._training_step_offset = step_offset

    def _publish_training_checkpoint_callback(
        self,
        callback: StableBaselinesCheckpointMetadataCallback,
    ) -> None:
        """Publish the training checkpoint callback owned by this sb3 model logic core."""
        self._checkpoint_metadata_callback = callback

    def _publish_last_training_scenario(self, scenario_index: int) -> None:
        """Publish the last training scenario owned by this sb3 model logic core."""
        self._last_scenario_index = scenario_index

    def _load_training_checkpoint(
        self,
        model_name: str,
        checkpoint_name: str,
    ) -> tuple[SB3TrainingAlgorithm, int]:
        """Load the training checkpoint owned by this sb3 model logic core."""
        self.load(
            model_name=model_name,
            checkpoint_name=checkpoint_name,
        )
        if self._model is None:
            raise RuntimeError("Checkpoint load completed without an SB3 model.")
        return (
            self._model,
            int(self._loaded_checkpoint_step_count),
        )

    def _retarget_training_session(
        self,
        selected_session_name: str | None,
    ) -> None:
        """Retarget the training session owned by this sb3 model logic core."""
        self._model_name = selected_session_name
        self._configurate()

    def _training_model_operations(self) -> TrainingModelOperations:
        """Return the training model operations value owned by this sb3 model logic core."""
        return TrainingModelOperations(
            load_checkpoint=self._load_training_checkpoint,
            retarget_session=self._retarget_training_session,
            publish_model=self._publish_training_model,
            publish_step_offset=self._publish_training_step_offset,
            configure_tensorboard_logger=(self._configure_training_tensorboard_logger),
            resolve_tensorboard_log_directory=(
                self._rl_model_wrapper.get_resolved_tensorboard_log_dir
            ),
            resolve_tensorboard_step_floor=(self._rl_model_wrapper.get_tensorboard_step_floor),
        )

    def train(
        self,
        scenario_manager: ScenarioManager,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Train through small stages while retaining resource ownership."""

        preparation = TrainingPreparationStage().prepare(
            runtime=self._rl_model_wrapper,
            model_policy=self._active_model_policy(),
            models_directory=str(self._models_directory),
            logger=self._logger,
            checkpoint_path_resolver=resolve_sb3_checkpoint_path,
        )
        states = TrainingStateStage().prepare(
            preparation=preparation,
            scenario_manager=scenario_manager,
            logger=self._logger,
            publish_scenario_state=(self._publish_training_scenario_state),
            publish_curriculum_state=(self._publish_training_curriculum_state),
        )
        environment = TrainingEnvironmentStage().create(
            preparation=preparation,
            states=states,
            scenario_manager=scenario_manager,
            environment_type=self._environment_type,
            logger=self._logger,
            publish_environment=self._publish_training_environment,
        )
        model_result = TrainingModelStage().build(
            preparation=preparation,
            environment=environment,
            model_name=str(self._model_name),
            operations=self._training_model_operations(),
            logger=self._logger,
        )
        callback_context = TrainingCallbackContext(
            preparation=preparation,
            states=states,
            environment=environment,
            model_result=model_result,
            runtime=self._rl_model_wrapper,
            scenario_manager=scenario_manager,
            progress_reporter=progress_callback,
            checkpoint_directory=str(self._checkpoint_directory),
            model_package_directory=str(self._model_package_directory),
            logger=self._logger,
        )
        loop_context = TrainingLoopContext(
            runtime=self._rl_model_wrapper,
            scenario_manager=scenario_manager,
            model_result=model_result,
            total_timesteps=environment.total_timesteps,
            progress_reporter=progress_callback,
            logger=self._logger,
        )
        callback_stage = TrainingCallbackStage()
        TrainingLoopStage().run(
            loop_context,
            callback_factory=lambda: callback_stage.assemble(
                callback_context,
                publish_checkpoint_callback=(self._publish_training_checkpoint_callback),
            ),
            publish_last_scenario=self._publish_last_training_scenario,
        )

    def deploy(
        self,
        scenario_manager: ScenarioManager,
        *,
        purpose: DeploymentPurpose = "deployment",
        repetitions: int = 1,
        visualization: EvaluationVisualization = "gui",
        realtime: bool | None = None,
    ) -> None:
        """Deploy the loaded model through the shared inference stages."""

        if purpose == "evaluation" and visualization == "gui":
            repetitions = 1
        preparation = DeploymentPreparationStage().prepare(
            runtime=self._rl_model_wrapper,
            scenario_manager=scenario_manager,
            purpose=purpose,
            logger=self._logger,
        )
        resources = DeploymentEnvironmentStage().create(
            preparation=preparation,
            scenario_manager=scenario_manager,
            environment_type=self._environment_type,
            logger=self._logger,
            publish_environment=self._publish_training_environment,
            purpose=purpose,
            visualization=visualization,
        )
        DeploymentLoopStage().run(
            DeploymentLoopContext(
                runtime=self._rl_model_wrapper,
                algorithm=cast(SB3DeploymentAlgorithm, self._model),
                environment=resources.environment,
                scheduler=resources.scheduler,
                deterministic_mode=self._deployment_deterministic_mode,
            ),
            repetitions=repetitions,
            publish_evaluation_summary=purpose == "evaluation",
            realtime=visualization == "gui" if realtime is None else realtime,
        )
        self._logger.info("Deployment finished.")

    def _deployment_deterministic_mode(self) -> bool:
        """Return the deployment deterministic mode value owned by this sb3 model logic core."""
        return cast(
            bool,
            self._get_model_manifest_configuration().evaluation_deterministic,
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
        """Run one held-out scenario through PPO load plus deterministic predict."""

        resolved_checkpoint = str(Path(checkpoint_path).resolve())
        if self._evaluation_checkpoint_path != resolved_checkpoint:
            load_checkpoint_restore_manifest(resolved_checkpoint)
            self._evaluation_model = cast(
                SB3DeploymentAlgorithm,
                PPO.load(resolved_checkpoint, device=sb3_device()),
            )
            self._evaluation_checkpoint_path = resolved_checkpoint
        algorithm = self._evaluation_model
        if algorithm is None:
            raise RuntimeError("SB3 evaluation policy was not restored.")
        preparation = DeploymentPreparationStage().prepare(
            runtime=cast(SB3DeploymentRuntime, runtime),
            scenario_manager=scenario_manager,
            purpose="evaluation",
            logger=self._logger,
        )
        resources = DeploymentEnvironmentStage().create(
            preparation=preparation,
            scenario_manager=scenario_manager,
            environment_type=self._environment_type,
            logger=self._logger,
            publish_environment=self._publish_training_environment,
            purpose="evaluation",
            visualization="headless",
        )
        try:
            DeploymentLoopStage().run(
                DeploymentLoopContext(
                    runtime=cast(SB3DeploymentRuntime, runtime),
                    algorithm=algorithm,
                    environment=resources.environment,
                    scheduler=resources.scheduler,
                    deterministic_mode=lambda: deterministic,
                ),
                repetitions=repetitions,
                publish_evaluation_summary=True,
                realtime=False,
            )
        finally:
            close_environment = getattr(resources.environment, "close", None)
            if callable(close_environment):
                close_environment()
            self._env = None
        return "completed"

    def evaluate_without_checkpoint(
        self,
        scenario_manager: ScenarioManager,
    ) -> None:
        """Run a model-specific diagnostic while retaining environment ownership."""

        policy = self._active_evaluation_policy()
        context = SB3EvaluationContext(
            runtime=self._rl_model_wrapper,
            logger=self._logger,
            model_configuration=self._get_model_manifest_configuration(),
            model_manifest=self._get_model_manifest(),
            scenario_manager=scenario_manager,
        )
        environment = policy.create_environment(context)
        # Publish ownership before reset or simulator commands can fail so the
        # runtime's finally-path can still close the environment exactly once.
        self._env = environment
        policy.evaluate(environment, context)

    def _current_checkpoint_step(self) -> int:
        """Return the current checkpoint step value owned by this sb3 model logic core."""
        model = getattr(self, "_model", None)
        return int(getattr(model, "num_timesteps", 0) or 0)

    def _current_checkpoint_scenario(self) -> int:
        """Return the current checkpoint scenario value owned by this sb3 model logic core."""
        if self._scenario_index_state is not None:
            return int(self._scenario_index_state.get())
        if self._last_scenario_index is not None:
            return int(self._last_scenario_index)
        return 0

    def _current_curriculum_state(self) -> object | None:
        """Return the current curriculum state value owned by this sb3 model logic core."""
        if self._curriculum_index_state is not None:
            return self._curriculum_index_state.get()
        return None

    @property
    def runtime_quality_degradations(self) -> frozenset[str]:
        """Return the runtime quality degradations exposed by this sb3 model logic core."""
        degradations = set(getattr(self, "_runtime_quality_degradations", set()))
        degradations.update(self._active_model_policy().quality_degradations)
        evaluation_policy = getattr(self, "_evaluation_policy", None)
        if evaluation_policy is not None:
            degradations.update(evaluation_policy.quality_degradations)
        return frozenset(degradations)

    def _record_runtime_quality_degradation(self, event: str, exc: BaseException) -> None:
        """Record the runtime quality degradation owned by this sb3 model logic core."""
        degradations = getattr(self, "_runtime_quality_degradations", None)
        if degradations is None:
            degradations = set()
            self._runtime_quality_degradations = degradations
        first_occurrence = event not in degradations
        degradations.add(event)
        if not first_occurrence:
            return
        self._logger.warning(
            f"event={event} quality=degraded error_type={type(exc).__name__} error={exc}"
        )

    def _current_episode_observation_rows(
        self,
    ) -> list[SB3CheckpointObservationRow]:
        """Return the current episode observation rows value owned by this sb3 model logic core."""
        provider: SB3CheckpointObservationProvider | None = self._checkpoint_metadata_callback
        if provider is None:
            return []
        try:
            return provider.episode_observations_for_checkpoint()
        except SB3CheckpointObservationUnavailable as exc:
            self._record_runtime_quality_degradation(
                "checkpoint_episode_observations_unavailable",
                exc,
            )
            return []

    def save(
        self,
        checkpoint: bool = False,
        manual: bool = False,
    ) -> str:
        """
        Save the already trained reinforcement learning model into a file.
        """

        self._logger.info("Saving model...")

        if checkpoint or manual:
            resolved_name = save_sb3_checkpoint(
                model=self._model,
                checkpoint_directory=str(self._checkpoint_directory),
                model_package_directory=str(self._model_package_directory),
                scenario_index=self._current_checkpoint_scenario(),
                raw_step=self._current_checkpoint_step(),
                base_step_offset=(
                    getattr(self, "_loaded_checkpoint_step_count", 0)
                    if getattr(self, "_training_step_offset", None) is None
                    else getattr(self, "_training_step_offset", 0)
                ),
                curriculum_state=self._current_curriculum_state(),
                parent_checkpoint_name=(
                    getattr(self, "_last_saved_checkpoint_name", None)
                    or getattr(self, "_loaded_checkpoint_name", None)
                ),
                episode_observations=None if manual else self._current_episode_observation_rows(),
                metric_snapshots=(
                    self._active_model_policy().checkpoint_metric_snapshots(
                        getattr(self, "_checkpoint_metadata_callback", None),
                        self._logger,
                    )
                ),
                manual=manual,
                logger=self._logger,
            )
            self._last_saved_checkpoint_name = resolved_name
            return resolved_name

        return self.save(checkpoint=True, manual=False)

    def load(
        self,
        model_name: str | None = None,
        checkpoint_name: str | None = None,
    ) -> None:
        """
        Load the already trained reinforcement learning model from a file.
        """

        self._logger.info("Loading model...")

        if model_name is None:
            model_name = self._get_model_manifest_configuration().selected_default_model_name
        validated_model_name = SessionStorageName.parse(str(model_name or ""))
        validated_checkpoint_name = (
            CheckpointStorageName.parse(checkpoint_name) if checkpoint_name is not None else None
        )
        self._model_name = validated_model_name.value

        self._configurate()
        tensorboard_log_dir = self._rl_model_wrapper.get_resolved_tensorboard_log_dir()
        load_kwargs: _CheckpointLoadOptions = {}
        if tensorboard_log_dir:
            load_kwargs["tensorboard_log"] = str(tensorboard_log_dir)

        if validated_checkpoint_name is not None:
            explicit_checkpoint_path = resolve_sb3_checkpoint_path(
                str(self._checkpoint_directory),
                validated_checkpoint_name.value,
            )
            if explicit_checkpoint_path and os.path.exists(explicit_checkpoint_path):
                self._restore_checkpoint(
                    explicit_checkpoint_path,
                    load_kwargs=load_kwargs,
                )
                self._logger.info(f"Selected checkpoint loaded. ({explicit_checkpoint_path})")
                return
            raise CheckpointMetadataError("Selected checkpoint artifact is unavailable.")

        if os.path.exists(str(self._model_package_directory)):
            if self._checkpoint_directory is None or not existdir(self._checkpoint_directory):
                self._logger.info(
                    f"Checkpoint directory does not exist. ({self._checkpoint_directory})"
                )
                raise FileNotFoundError
            checkpoints = list_sb3_checkpoint_paths(str(self._checkpoint_directory))
            if len(checkpoints) > 0:
                last_checkpoint_path = checkpoints[0]
                self._restore_checkpoint(
                    last_checkpoint_path,
                    load_kwargs=load_kwargs,
                )
                self._logger.info(f"Last checkpoint loaded. ({last_checkpoint_path})")
                return
            else:
                self._logger.info(
                    f"No checkpoints found in {self._checkpoint_directory}. But the package exists."
                )
                raise FileNotFoundError
        else:
            self._logger.error(f"Model package does not exist. ({self._model_package_directory})")
            raise NotADirectoryError
