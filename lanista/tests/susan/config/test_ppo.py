# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import call, patch

import torch
import torch.nn as nn


class PpoConfigTests(unittest.TestCase):
    def test_sb3_uses_configured_tanh_activation(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        config = runtime_config.build_sb3_ppo_constructor_config("SusanEnv")

        self.assertIs(config["policy_kwargs"]["activation_fn"], nn.Tanh)

    def test_sb3_silu_activation_is_not_treated_as_relu(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        original_activation = runtime_config.PPO_POLICY_ACTIVATION
        try:
            runtime_config.PPO_POLICY_ACTIVATION = "silu"
            config = runtime_config.build_sb3_ppo_constructor_config("SusanEnv")
        finally:
            runtime_config.PPO_POLICY_ACTIVATION = original_activation

        self.assertIs(config["policy_kwargs"]["activation_fn"], nn.SiLU)

    def test_ray_model_config_uses_shared_activation_name(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.ray import (
            algorithm_configurator,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training.ppo_config import (
            PPO_POLICY_ACTIVATION,
        )

        model_config = algorithm_configurator._ray_model_config()
        self.assertEqual(
            model_config["fcnet_activation"],
            PPO_POLICY_ACTIVATION,
        )
        self.assertNotIn("log_std_clip_param", model_config)

    def test_ray_checkpoint_frequency_scales_with_parallel_env_count(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        original_builder = runtime_config.build_ray_training_topology
        try:
            runtime_config.build_ray_training_topology = lambda: runtime_config.RayTrainingTopology(
                number_of_cpu_cores=12,
                number_of_cpu_threads=24,
                number_of_rollout_workers=8,
                number_of_envs_per_worker=1,
                rollout_fragment_length=512,
                batch_mode="truncate_episodes",
            )

            self.assertEqual(runtime_config.build_ray_checkpoint_saving_frequency(50000), 400000)
        finally:
            runtime_config.build_ray_training_topology = original_builder

    def test_shared_minibatch_size_is_used_by_sb3(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training.ppo_config import (
            PPO_MINIBATCH_SIZE,
        )

        config = runtime_config.build_sb3_ppo_constructor_config("SusanEnv")

        self.assertEqual(config["batch_size"], PPO_MINIBATCH_SIZE)
        self.assertEqual(config["batch_size"], 256)

    def test_sb3_and_ray_use_the_same_2048_sample_update_batch(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.ray import (
            algorithm_configurator,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        sb3_config = runtime_config.build_sb3_ppo_constructor_config("SusanEnv")
        ray_config = algorithm_configurator.training_algorithm_configurator("SusanEnv")

        self.assertEqual(sb3_config["n_steps"], 2048)
        self.assertEqual(ray_config.train_batch_size_per_learner, 2048)
        self.assertEqual(ray_config.rollout_fragment_length, "auto")

    def test_ray_policy_uses_sb3_equivalent_free_log_std_initialization(self) -> None:
        from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
            DefaultPPOTorchRLModule,
        )

        from lanista.rl_model_storage.models.susan_project.resources.ray import (
            algorithm_configurator,
        )

        ray_config = algorithm_configurator.training_algorithm_configurator("SusanEnv")
        catalog_class = ray_config.rl_module_spec.catalog_class
        model_config = ray_config.model_config

        self.assertTrue(model_config["free_log_std"])
        self.assertIsNotNone(catalog_class)
        module = DefaultPPOTorchRLModule(
            observation_space=algorithm_configurator._EGO_policy_observation_space(),
            action_space=algorithm_configurator._EGO_policy_action_space(),
            model_config=model_config,
            catalog_class=catalog_class,
        )
        policy_head = module.pi

        self.assertTrue(hasattr(policy_head, "log_std"))
        self.assertTrue(
            torch.equal(
                policy_head.log_std.detach(),
                torch.full_like(policy_head.log_std.detach(), -1.0),
            )
        )
        self.assertFalse(policy_head.clip_log_std)

    def test_shared_entropy_experiment_is_used_by_sb3(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training.ppo_config import (
            PPO_ENTROPY_COEFF,
        )

        config = runtime_config.build_sb3_ppo_constructor_config("SusanEnv")

        self.assertEqual(PPO_ENTROPY_COEFF, 1.0e-3)
        self.assertEqual(config["ent_coef"], PPO_ENTROPY_COEFF)

    def test_shared_entropy_experiment_is_used_by_ray(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training.ppo_config import (
            PPO_ENTROPY_COEFF,
        )

        class ConfigRecorder:
            def __init__(self) -> None:
                self.training_kwargs: dict[str, object] = {}

            def environment(self, *_args, **_kwargs):
                return self

            def framework(self, *_args, **_kwargs):
                return self

            def rl_module(self, *_args, **_kwargs):
                return self

            def env_runners(self, *_args, **_kwargs):
                return self

            def training(self, **kwargs):
                self.training_kwargs = kwargs
                return self

            def learners(self, *_args, **_kwargs):
                return self

            def multi_agent(self, *_args, **_kwargs):
                return self

            def debugging(self, *_args, **_kwargs):
                return self

        recorder = ConfigRecorder()
        topology = runtime_config.RayTrainingTopology(
            number_of_cpu_cores=4,
            number_of_cpu_threads=8,
            number_of_rollout_workers=2,
            number_of_envs_per_worker=1,
            rollout_fragment_length=512,
            batch_mode="truncate_episodes",
        )
        with (
            patch.object(runtime_config, "PPOConfig", return_value=recorder),
            patch.object(
                runtime_config,
                "build_ray_training_topology",
                return_value=topology,
            ),
        ):
            result = runtime_config.build_ray_training_algorithm_config(
                env_id="SusanEnv",
                policies={},
                policy_mapping_fn=lambda *_args: "EGO_policy",
                logger_creator=object(),
                model_config_builder=dict,
            )

        self.assertIs(result, recorder)
        self.assertEqual(
            recorder.training_kwargs["entropy_coeff"],
            PPO_ENTROPY_COEFF,
        )

    def test_full_curriculum_uses_lower_shared_learning_rate_profile(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )
        from lanista.rl_model_storage.models.susan_project.resources.utils.training.ppo_config import (
            PPO_FULL_CURRICULUM_LEARNING_RATE,
            PPO_LEARNING_RATE,
        )

        self.assertEqual(PPO_FULL_CURRICULUM_LEARNING_RATE, 5.0e-5)
        self.assertLess(PPO_FULL_CURRICULUM_LEARNING_RATE, PPO_LEARNING_RATE)
        self.assertEqual(
            runtime_config.ppo_learning_rate_for_curriculum_state({"name": "intersection_6"}),
            PPO_LEARNING_RATE,
        )
        self.assertEqual(
            runtime_config.ppo_learning_rate_for_curriculum_state({"state_name": "full"}),
            PPO_FULL_CURRICULUM_LEARNING_RATE,
        )
        self.assertEqual(
            runtime_config.ppo_learning_rate_for_curriculum_state(None),
            PPO_FULL_CURRICULUM_LEARNING_RATE,
        )

        original_full_rate = runtime_config.PPO_FULL_CURRICULUM_LEARNING_RATE
        try:
            runtime_config.PPO_FULL_CURRICULUM_LEARNING_RATE = 4.0e-5
            self.assertEqual(
                runtime_config.ppo_learning_rate_for_curriculum_state({"name": "intersection_6"}),
                PPO_LEARNING_RATE,
            )
            self.assertEqual(
                runtime_config.ppo_learning_rate_for_curriculum_state({"state_name": "full"}),
                4.0e-5,
            )
        finally:
            runtime_config.PPO_FULL_CURRICULUM_LEARNING_RATE = original_full_rate

    def test_apply_sb3_learning_rate_updates_optimizer_and_schedule(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        optimizer = SimpleNamespace(param_groups=[{"lr": 7.0e-5}, {"lr": 7.0e-5}])
        model = SimpleNamespace(policy=SimpleNamespace(optimizer=optimizer), lr_schedule=None)

        self.assertTrue(runtime_config.apply_sb3_learning_rate(model, 4.0e-5))
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [4.0e-5, 4.0e-5])
        self.assertEqual(model.lr_schedule(0.5), 4.0e-5)

    def test_apply_ray_learning_rate_uses_module_fallback_order(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        class Learner:
            def __init__(self) -> None:
                self.attempts: list[str] = []
                self.applied: list[tuple[object, float]] = []

            def get_optimizer(self, *, module_id: str, optimizer_name: str) -> object:
                self.assert_optimizer_name(optimizer_name)
                self.attempts.append(module_id)
                if module_id == "EGO_policy":
                    raise KeyError(module_id)
                return f"{module_id}-optimizer"

            @staticmethod
            def assert_optimizer_name(optimizer_name: str) -> None:
                if optimizer_name != "default_optimizer":
                    raise AssertionError(optimizer_name)

            def _set_optimizer_lr(self, optimizer: object, learning_rate: float) -> None:
                self.applied.append((optimizer, learning_rate))

        class LearnerGroup:
            def __init__(self, learner: Learner) -> None:
                self.learner = learner

            def foreach_learner(self, operation, *, lr: float, primary_module_id: str):
                update = operation(self.learner, lr, primary_module_id)
                return [SimpleNamespace(ok=True, get=lambda: update)]

        learner = Learner()
        algorithm = SimpleNamespace(learner_group=LearnerGroup(learner))

        self.assertTrue(runtime_config.apply_ray_learning_rate(algorithm, 4.0e-5))
        self.assertEqual(["EGO_policy", "default_policy"], learner.attempts)
        self.assertEqual([("default_policy-optimizer", 4.0e-5)], learner.applied)

    def test_apply_ray_learning_rate_reports_exhausted_optimizer_candidates(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        learner = SimpleNamespace(
            get_optimizer=lambda **_kwargs: (_ for _ in ()).throw(KeyError("missing")),
            _set_optimizer_lr=lambda _optimizer, _learning_rate: None,
        )

        class LearnerGroup:
            @staticmethod
            def foreach_learner(operation, *, lr: float, primary_module_id: str):
                update = operation(learner, lr, primary_module_id)
                return [SimpleNamespace(ok=True, get=lambda: update)]

        with patch.object(runtime_config, "_warn_quality_degradation") as warn:
            applied = runtime_config.apply_ray_learning_rate(
                SimpleNamespace(learner_group=LearnerGroup()),
                4.0e-5,
            )

        self.assertFalse(applied)
        self.assertEqual(
            [
                call(
                    "ray_learning_rate_optimizer_unavailable",
                    attempted_modules=3,
                ),
                call("ray_learning_rate_not_applied"),
            ],
            warn.call_args_list,
        )

    def test_apply_ray_learning_rate_reports_remote_and_unreadable_results(self) -> None:
        from lanista.rl_model_storage.models.susan_project.resources.utils.training import (
            runtime_config,
        )

        remote_error = RuntimeError("remote learner failed")
        remote_result = SimpleNamespace(ok=False, get=lambda: remote_error)
        remote_group = SimpleNamespace(
            foreach_learner=lambda _operation, **_kwargs: [remote_result]
        )
        unreadable_group = SimpleNamespace(
            foreach_learner=lambda _operation, **_kwargs: 42
        )

        with patch.object(runtime_config, "_warn_quality_degradation") as warn:
            self.assertFalse(
                runtime_config.apply_ray_learning_rate(
                    SimpleNamespace(learner_group=remote_group),
                    4.0e-5,
                )
            )
            self.assertFalse(
                runtime_config.apply_ray_learning_rate(
                    SimpleNamespace(learner_group=unreadable_group),
                    4.0e-5,
                )
            )

        self.assertEqual(
            "ray_learning_rate_remote_failure",
            warn.call_args_list[0].args[0],
        )
        self.assertEqual(
            "ray_learning_rate_not_applied",
            warn.call_args_list[1].args[0],
        )
        self.assertEqual(
            "ray_learning_rate_results_unreadable",
            warn.call_args_list[2].args[0],
        )


if __name__ == "__main__":
    unittest.main()
