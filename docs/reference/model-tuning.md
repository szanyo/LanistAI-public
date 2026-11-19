<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Model tuning and framework configuration

Susan's common PPO values are imported by both model-local backend builders.
The framework manifests supply curriculum, checkpoint and evaluation settings.

## Choose the correct configuration surface

| What changes | Current owner | Use it for |
|---|---|---|
| Shared PPO intent | Susan's [`ppo_config.py`](../../lanista/rl_model_storage/models/susan_project/resources/utils/training/ppo_config.py) | Values deliberately aligned across RLlib and SB3, such as learning rate, discounting, PPO clipping, update epochs, and network layout |
| RLlib algorithm and sampling | The model-local [Ray configurator](../../lanista/rl_model_storage/models/susan_project/resources/ray/algorithm_configurator.py) and [builder](../../lanista/rl_model_storage/models/susan_project/resources/utils/training/runtime_config.py) | `PPOConfig`, EnvRunner topology, learner resources, multi-agent policy mapping, RLlib model options, and Ray logging |
| SB3 algorithm and policy | The model-local [SB3 configurator](../../lanista/rl_model_storage/models/susan_project/resources/sb3/algorithm_configurator.py), its builder, and the [core adapter allowlist](../../lanista/reinforcengine/rl_technologies/sb3/algorithm_config.py) | Keyword arguments accepted by the SB3 `PPO` constructor and its `policy_kwargs` |
| Run and evaluation policy | The model-local Ray or SB3 `model_manifest_configuration.yaml` | Checkpoint/evaluation cadence, evaluation behavior, scenario selection, curriculum gates, and plugin identity |
| Task, observation, reward, and scenario | Focused files below the model's `resources/` tree | Environment dynamics, spaces, reward weights, episode limits, and domain-specific scenario difficulty |
| Active run selection | Machine-local runtime state, changed through the application | Framework, mode, session, and checkpoint association |

Do not place arbitrary PPO keys in a model manifest. The current runtime reads
algorithm hyperparameters from the model's configurator, while the manifest
owns lifecycle and experiment-policy metadata.

## Shared PPO parameter map

Following the shared-configuration format used for reproducible experiment
reports, one experimental dimension is shown beside the native name used by
each backend.

| Configuration group | Unified meaning | Ray RLlib mapping | Stable-Baselines3 mapping |
|---|---|---|---|
| Optimizer | Learning-rate value or schedule | `PPOConfig.training(lr=...)` | `PPO(..., learning_rate=...)` |
| PPO update | Optimization passes over collected data | `num_epochs` | `n_epochs` |
| PPO update | Samples per optimizer minibatch | `minibatch_size` | `batch_size` |
| Temporal credit | Future-reward discount | `gamma` | `gamma` |
| Temporal credit | GAE trace-decay factor | `lambda_` | `gae_lambda` |
| Policy constraint | PPO policy-ratio clip | `clip_param` | `clip_range` |
| Regularization | Entropy-loss coefficient | `entropy_coeff` | `ent_coef` |
| Value learning | Value-loss coefficient | `vf_loss_coeff` | `vf_coef` |
| Optimization safety | Global gradient-norm limit | `grad_clip` | `max_grad_norm` |
| Policy/value network | Separate hidden-layer layouts | `model.fcnet_hiddens` with `vf_share_layers=False` | `policy_kwargs.net_arch={"pi": ..., "vf": ...}` |
| Policy/value network | Hidden activation | `model.fcnet_activation` | `policy_kwargs.activation_fn` |

Sampling topology, loss reduction, advantage handling and framework defaults
affect the update in each backend.

## Susan PPO baseline

| Configuration group | Parameter | Current shared value | Source symbol |
|---|---|---|---|
| Optimizer | Learning rate during restricted curriculum | `7e-5` | `PPO_LEARNING_RATE` |
| Optimizer | Learning rate after the curriculum unlocks the full route | `5e-5` | `PPO_FULL_CURRICULUM_LEARNING_RATE` |
| PPO update | Epochs per collected batch | `4` | `PPO_NUM_EPOCHS` |
| PPO update | Configured samples per update | `2048` | `PPO_ROLLOUT_STEPS` |
| PPO update | Minibatch size | `256` | `PPO_MINIBATCH_SIZE` |
| Temporal credit | Discount factor | `0.995` | `PPO_GAMMA` |
| Temporal credit | GAE lambda | `0.97` | `PPO_GAE_LAMBDA` |
| Policy constraint | Policy-ratio clip | `0.10` | `PPO_CLIP_RANGE` |
| Regularization | Entropy coefficient | `1e-3` | `PPO_ENTROPY_COEFF` |
| Value learning | Value-loss coefficient | `0.5` | `PPO_VALUE_LOSS_COEFF` |
| Optimization safety | Gradient-norm limit | `0.3` | `PPO_MAX_GRAD_NORM` |
| Network | Separate policy/value hidden layers | `[64, 64]` | `PPO_POLICY_NET_ARCH` |
| Network | Hidden activation | `tanh` | `PPO_POLICY_ACTIVATION` |

For the bundled Susan task, model-specific configuration also controls a
30-sample history, observation/reward constants, acceleration bounds of
`[-4, 3]`, a `0.05 s` simulation step, a 5000–7000 step curriculum budget, and scenario
difficulty. These values live in the observation, training, and scenario
configuration files.

## Backend-specific behavior

| Dimension | Ray RLlib | Stable-Baselines3 |
|---|---|---|
| Collection | One environment per EnvRunner. The default requests eight runners and caps the count by physical CPU availability. Fragment length defaults to `auto`. | The local environment fills an explicitly configured 2048-step rollout. |
| PPO batch | The learner target is 2048 samples. | Each rollout is split into 256-sample minibatches. |
| Value clipping | `vf_clip_param=80.0` caps the squared value error. | `clip_range_vf=80.0` clips the change in the value prediction. |
| KL control | `use_kl_loss=True` with `kl_target=0.01` adjusts the KL-loss coefficient. | `target_kl=0.01` supplies an early-stopping guard. |
| Curriculum evidence | 200 accepted episodes per restricted level or scenario. | 25 accepted episodes per restricted level or scenario. |

The Ray builder reads `LANISTAI_RAY_ROLLOUT_WORKERS`,
`LANISTAI_RAY_ROLLOUT_FRAGMENT_LENGTH` and `LANISTAI_RAY_BATCH_MODE`.
It reserves two physical cores when capacity permits. When an explicit numeric
fragment length is set, it is at least the shared minibatch size.
