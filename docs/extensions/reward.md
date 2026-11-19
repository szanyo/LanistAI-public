<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Reward calculation

The reward calculation uses four weighted components for command realisation,
command change, speed and jitter. Separate services own simulator context,
numerical calculations, mutable episode state and telemetry.

## Responsibilities

- `lanista/rl_model_storage/models/susan_project/resources/utils/reward/reward_service.py`
  owns the typed reward workflow, state-update order, telemetry projection, and
  component series. Its immutable call boundary is
  `RewardCalculationRequest -> RewardCalculationResult`. The result contains
  the total reward and actual acceleration used by the same step.
- `RewardMathPort` and `RewardStatePort` make numerical and mutable-state
  dependencies explicit. The current numerical implementation is
  `lanista/rl_model_storage/models/susan_project/resources/utils/control/controller_math_service.py`.
  It owns formulas and component weighting.
- `lanista/rl_model_storage/models/susan_project/resources/utils/reward/reward_telemetry.py`
  owns typed reward state and component telemetry.
- `lanista/rl_model_storage/models/susan_project/resources/utils/simulation/simulation_pipeline_service.py`
  resolves step inputs, constructs `RewardCalculationRequest`, invokes
  `RewardService.calculate()`, and returns the total. It does not own reward
  formulas.
- `lanista/reinforcengine/rl_metrics/channel_schema.py` owns shared metric keys,
  nested reward paths, component ordering, lengths, and bounds.
  `lanista/reinforcengine/rl_metrics/snapshot_projection.py` owns environment
  extraction and nested snapshot projection.
  `reward_series_values_from_env()` consumes a direct
  `get_reward_component_series()` hook before its legacy attribute fallback.
- `lanista/reinforcengine/rl_metrics/aggregation.py` owns framework-neutral
  numeric validation, flattening, statistics, grouping, and metric-family
  filters. `metric_table.py` formats and logs the tables.
- `lanista/reinforcengine/rl_metrics/tensorboard_surface.py` separately owns the
  TensorBoard scalar/histogram allowlists, stable tags, and compact dashboard
  aliases.

The model-local framework adapters expose the same service series and must stay
numerically neutral:

- `lanista/rl_model_storage/models/susan_project/resources/ray/sim_env.py`
- `lanista/rl_model_storage/models/susan_project/resources/sb3/sim_env.py`.


The [reproduction guide](../reproduction.md#train-the-model) explains the historical
jitter weighting used in this snapshot.
