<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Susan policy observations

The observation contract specifies field names, order, shape, units, bounds,
normalization and missing values. Changing a
policy field or its order can invalidate stored models even when the Gym shape
still accepts the value.

The framework-neutral contract is
`lanista/reinforcengine/rl_metrics/observation_contract.py`. It defines policy
history/scalar order, telemetry-only keys, lengths, and bounds without importing
a model plugin. Ray and SB3 may adapt the container shape, but they must not
change field meaning, normalization, or order.

## Susan responsibility boundaries

- `lanista/rl_model_storage/models/susan_project/resources/utils/observation/observation_feature_service.py`
  is the stateless numerical boundary for object slots, red-light features, and
  traffic-light flags. It must not read the simulator, own history, or publish
  framework metrics.
- `lanista/rl_model_storage/models/susan_project/resources/utils/observation/observation_service.py`
  owns formatting, policy history projection, telemetry conversion, and the
  observation component series. Its immutable boundary is
  `ObservationFormattingRequest -> FormattedObservation`. Request fields use
  unit-bearing names, and `FormattedObservation.values` is read-only.
- `lanista/rl_model_storage/models/susan_project/resources/utils/observation/observation_config.py`
  binds the neutral key contract to Susan history/scalar channels and documents
  producer normalization and Gym guardrails.
- `lanista/rl_model_storage/models/susan_project/resources/utils/observation/vehicle_telemetry.py`
  and
  `lanista/rl_model_storage/models/susan_project/resources/utils/observation/vehicle_telemetry_container.py`
  own typed telemetry and history storage. They are not the policy schema.
- `lanista/rl_model_storage/models/susan_project/resources/utils/simulation/simulation_pipeline_service.py`
  is orchestration: it reads scenario/control state, constructs the request,
  calls `ObservationService.format()`, appends history, and publishes telemetry.
  Feature calculations do not belong in this pipeline.

The model-local Ray and SB3 adapters expose the same service result through
`get_observation_component_series()`:

- `lanista/rl_model_storage/models/susan_project/resources/ray/sim_env.py`
- `lanista/rl_model_storage/models/susan_project/resources/sb3/sim_env.py`.

Shared collectors consume that hook in
`lanista/reinforcengine/rl_metrics/env_snapshot.py` and
`lanista/reinforcengine/utils/scenarios/scenario_switching_callbacks/ray_episode_metrics_capture.py`.
The descriptor exposes 13 channels of 30 samples each and 15 current scalars,
for 405 policy inputs.
