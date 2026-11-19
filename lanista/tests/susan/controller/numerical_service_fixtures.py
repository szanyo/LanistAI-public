# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Focused composition helpers for Susan numerical-service tests."""

from collections import deque
from types import SimpleNamespace
from typing import cast

from lanista.rl_model_storage.models.susan_project.resources.utils.control.controller_config import (
    CONTROLLER_CONFIG,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.control.controller_math_service import (
    ControllerMathService,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.control.controller_runtime_state import (
    ControllerRuntimeState,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.control.traffic_light_service import (
    TrafficLightService,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.control.vehicle_sim_controller import (
    VehicleSimController,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.observation_config import (
    SPEED_REWARD_JITTER_COMFORT_WINDOW,
    SPEED_REWARD_JITTER_TREND_WINDOW,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.observation_feature_service import (
    ObservationFeatureMathPort,
    ObservationFeatureService,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.observation_service import (
    ObservationMathPort,
    ObservationService,
    TrafficLightObservationPort,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.vehicle_telemetry import (
    VehicleControllerTelemetryState,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.observation.vehicle_telemetry_container import (
    VehicleObservationTelemetryContainer,
    VehicleTelemetryContainer,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.reward.reward_service import (
    RewardMathPort,
    RewardService,
    RewardStatePort,
)
from lanista.rl_model_storage.models.susan_project.resources.utils.reward.reward_telemetry import (
    RewardConfigTelemetry,
    RewardStateTelemetry,
)


def numerical_controller(
    *,
    average_speed_mps: float = 0.0,
) -> VehicleSimController:
    """Create a partial controller with every numerical state owner explicit."""

    controller = VehicleSimController.__new__(VehicleSimController)
    controller._runtime_state = ControllerRuntimeState()
    controller._controller_config = CONTROLLER_CONFIG
    controller.controller_state = VehicleControllerTelemetryState()
    controller.reward_config = RewardConfigTelemetry()
    controller.reward_state = RewardStateTelemetry()
    controller.scenario_context_preprocessor = None
    controller._episode_active_step_limit = None
    controller._current_supervised_cmd_accel_delta = 0.0
    controller.supervised_cmd_accel_delta = 0.0
    controller.abs_supervised_cmd_accel_delta = 0.0
    controller.supervised_cmd_accel_delta_change = 0.0
    controller.abs_supervised_cmd_accel_delta_change = 0.0
    controller.supervised_cmd_accel_local_jitter = 0.0
    controller.supervised_cmd_accel_local_jitter_sign_change = 0.0
    controller.supervised_cmd_accel_local_effective_jitter = 0.0
    controller.supervised_cmd_accel_trend_jitter = 0.0
    controller.supervised_cmd_accel_trend_jitter_sign_change = 0.0
    controller.supervised_cmd_accel_trend_effective_jitter = 0.0
    controller.supervised_cmd_accel_comfort_jitter = 0.0
    controller.supervised_cmd_accel_weighted_comfort_jitter = 0.0
    controller.supervised_cmd_accel_comfort_jitter_sign_change = 0.0
    controller.supervised_cmd_accel_comfort_effective_jitter = 0.0
    controller.supervised_cmd_accel_discount_jitter = 0.0
    controller.normalized_supervised_cmd_accel_discount_jitter = 0.0
    controller.pre_jitter_discount_speed_reward = 0.0
    controller.speed_reward_jitter_discount_applied = 0.0
    controller.speed_reward_jitter_discount = 1.0
    controller.supervised_cmd_accel_history = deque(
        maxlen=max(
            SPEED_REWARD_JITTER_TREND_WINDOW,
            SPEED_REWARD_JITTER_COMFORT_WINDOW,
        ),
    )
    controller.supervised_cmd_accel_delta_history = deque(
        maxlen=max(
            SPEED_REWARD_JITTER_TREND_WINDOW,
            SPEED_REWARD_JITTER_COMFORT_WINDOW,
        ),
    )
    controller.ego_vehicle = SimpleNamespace(AverageSpeed=average_speed_mps)
    return controller


def math_service_for(controller: VehicleSimController) -> ControllerMathService:
    """Return the numerical service under test without a controller facade."""

    return ControllerMathService(controller)


def reward_service_for(controller: VehicleSimController) -> RewardService:
    """Compose the direct reward boundary for a partially initialized controller."""

    math = ControllerMathService(controller)
    return RewardService(
        state=cast(RewardStatePort, controller),
        math=cast(RewardMathPort, math),
    )


def observation_service_for(
    controller: VehicleSimController,
    traffic_lights: TrafficLightService,
    *,
    history: VehicleObservationTelemetryContainer | None = None,
    telemetry: VehicleTelemetryContainer | None = None,
) -> ObservationService:
    """Compose the direct observation boundary around explicit containers."""

    math = ControllerMathService(controller)
    return ObservationService(
        history=history or VehicleObservationTelemetryContainer(maxlen=3),
        telemetry=telemetry or VehicleTelemetryContainer(),
        features=ObservationFeatureService(
            cast(ObservationFeatureMathPort, math)
        ),
        math=cast(ObservationMathPort, math),
        traffic_lights=cast(TrafficLightObservationPort, traffic_lights),
    )
