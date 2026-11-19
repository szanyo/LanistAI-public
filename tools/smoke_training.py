#!/usr/bin/env python3
"""Run a short Susan PPO update in a disposable model workspace."""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


async def run_check(data_home: Path) -> None:
    """Load the configured model and run eight training steps on CPU."""
    import numpy as np
    import sumo
    from stable_baselines3 import PPO

    from lanista.application_bootstrap import APP_CONFIGURATION
    from lanista.reinforcengine.rl_model.administration_safety_service import (
        AdministrationSafetyService,
    )
    from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
    from lanista.reinforcengine.rl_model.runtime.import_service import ImportService
    from lanista.reinforcengine.rl_model.runtime.requests import RuntimeInitializationRequest
    from lanista.reinforcengine.rl_technologies.sb3.algorithm_config import (
        build_ppo_constructor_args,
    )
    from lanista.reinforcengine.running_mode import RunningMode
    from lanista.reinforcengine.utils.scenarios.scenario_state.curriculum_state_sb3 import (
        StableBaselinesCurriculumState,
    )
    from lanista.reinforcengine.utils.scenarios.scenario_state.scenario_state_sb3 import (
        StableBaselinesScenarioState,
    )

    APP_CONFIGURATION.sumo_home = str(Path(sumo.__file__).parent)
    model_root = data_home / "rl_model_storage/models/susan_project"
    manifest = ModelManifest(str(model_root))
    manifest.load()
    service = ImportService(
        framework_normalizer=lambda value: str(value or "").strip().lower(),
        progress_reporter=lambda *args: None,
    )
    request = RuntimeInitializationRequest("smoke", "runtime", "INFO", AdministrationSafetyService())
    result = await service.load_model(manifest, request, framework_uuid="sb3")
    environment = None
    try:
        from lanista.rl_model_storage.models.susan_project.resources.sb3 import (
            algorithm_configurator,
        )
        from lanista.rl_model_storage.models.susan_project.resources.sb3.model_policy import (
            SusanSB3ModelPolicy,
        )
        from lanista.rl_model_storage.models.susan_project.resources.sb3.sim_env import (
            StableBaselinesSimEnv,
        )

        manager = result.model._scenario_manager.clone()
        manager.generate()
        if manager.get_total_number_of_scenarios() != 21:
            raise RuntimeError("The expected training scenario inventory was not loaded")
        configuration = result.model.get_manifest().get_configuration()
        options = SusanSB3ModelPolicy().training_environment_options(configuration, StableBaselinesSimEnv)
        environment = StableBaselinesSimEnv(
            rl_model_manifest=result.model.get_manifest(),
            rl_model_mode=RunningMode.TRAINING,
            scenario_manager=manager,
            scenario_index_state=StableBaselinesScenarioState(),
            curriculum_index_state=StableBaselinesCurriculumState(),
            parent_logger_name="smoke",
            child_logger_name="environment",
            **options,
        )
        policy, configured_env, kwargs = build_ppo_constructor_args(
            configurator_module=algorithm_configurator,
            configurator_name="training_algorithm_configurator",
            env_id="smoke",
            env=environment,
            tensorboard_log=str(data_home / "tensorboard"),
        )
        kwargs.update(n_steps=8, batch_size=8, n_epochs=1, device="cpu", verbose=0)
        model = PPO(policy, configured_env, **kwargs)
        model.learn(total_timesteps=8)
        if model.num_timesteps != 8:
            raise RuntimeError(f"Unexpected number of training steps: {model.num_timesteps}")
        observation, _ = environment.reset()
        expected_action, _ = model.predict(observation, deterministic=True)
        model_path = data_home / "smoke-policy.zip"
        model.save(model_path)
        restored = PPO.load(model_path, device="cpu")
        restored_action, _ = restored.predict(observation, deterministic=True)
        np.testing.assert_allclose(restored_action, expected_action, rtol=0, atol=1e-7)
        _, reward, _, _, _ = environment.step(restored_action)
        if not np.isfinite(reward):
            raise RuntimeError("Restored policy produced a non-finite reward")
        print("Susan SB3 smoke check passed with 21 scenarios, 8 PPO steps and policy reload")
    finally:
        if environment is not None:
            environment.close()
        await result.model.close_async()


def main() -> None:
    """Create temporary writable model data and remove it after the check."""
    source = ROOT / "lanista/rl_model_storage/models/susan_project"
    with tempfile.TemporaryDirectory(prefix="lanistai-smoke-") as temporary:
        data_home = Path(temporary)
        os.environ["LANISTAI_DATA_HOME"] = str(data_home)
        os.environ.setdefault("MPLCONFIGDIR", str(data_home / "matplotlib"))
        target = data_home / "rl_model_storage/models/susan_project"
        target.parent.mkdir(parents=True)
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        asyncio.run(run_check(data_home))


if __name__ == "__main__":
    main()
