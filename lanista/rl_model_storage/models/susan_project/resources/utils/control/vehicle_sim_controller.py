# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan's composition owner for the typed per-step control pipeline."""

from collections import deque
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol, cast, overload

import numpy as np

from lanista.domains.automotive.control.ego import EGO
from lanista.reinforcengine.rl_metrics.aggregation import is_valid_number
from lanista.reinforcengine.rl_metrics.channel_schema import (
    TRAINING_DIAGNOSTIC_PARAM_NAMES,
)
from lanista.reinforcengine.utils.simulation.intersection_metadata import (
    INTERSECTION_TYPE_TRAFFIC_LIGHT,
    TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES,
    TRAFFIC_LIGHT_STATE_GREEN,
    TRAFFIC_LIGHT_STATE_RED,
    TRAFFIC_LIGHT_STATE_RED_YELLOW,
    TRAFFIC_LIGHT_STATE_YELLOW,
)
from lanista.reinforcengine.utils.simulation.ports import SimulatorPort

from ..episode.interaction_tracker import EpisodeInteractionTracker, InteractionKind
from ..observation.observation_config import (
    OBSERVATION_HISTORY_LENGTH,
    SPEED_REWARD_JITTER_COMFORT_WINDOW,
    SPEED_REWARD_JITTER_TREND_WINDOW,
)
from ..observation.observation_feature_service import (
    ObservationFeatureMathPort,
    ObservationFeatureService,
)
from ..observation.observation_service import (
    ObservationMathPort,
    ObservationService,
    TrafficLightObservationPort,
)
from ..observation.vehicle_telemetry import (
    OBSERVATION_KEYS,
    VehicleControllerTelemetryState,
)
from ..observation.vehicle_telemetry_container import (
    VehicleObservationTelemetryContainer,
    VehicleTelemetryContainer,
)
from ..reward.reward_service import RewardMathPort, RewardService, RewardStatePort
from ..reward.reward_telemetry import (
    RewardConfigTelemetry,
    RewardStateTelemetry,
    RewardTelemetry,
)
from ..scenario.scenario_context_preprocessor import ScenarioContextPreprocessor
from ..simulation.diagnostic_trace import (
    SusanDiagnosticTraceProjector,
    susan_diagnostic_trace_fieldnames,
)
from ..simulation.simulation_contracts import (
    EgoVehiclePort,
    ScalarActionPort,
    ScenarioContext,
    SimulationMathPort,
    SimulationPipelineControllerPort,
    TelemetrySinkPort,
    is_ego_blocked_by_pedestrian,
)
from ..simulation.simulation_pipeline_service import SimulationPipelineService
from .controller_config import CONTROLLER_CONFIG, ControllerConfig
from .controller_math_service import ControllerMathService
from .controller_runtime_state import ControllerRuntimeState
from .sumo_command_service import SumoCommandService
from .traffic_light_service import TrafficLightService


class _ScenarioManagerConfigurationPort(Protocol):
    """Define the typed boundary for scenario manager configuration."""

    def get_model_folder_path(self) -> str | Path:
        """Return the model folder path exposed by this scenario manager configuration port."""
        ...

    def get_network_filename(self) -> str:
        """Return the network filename exposed by this scenario manager configuration port."""
        ...


_create_ego_vehicle = cast(Callable[[], EgoVehiclePort], EGO)

_DIAGNOSTIC_REWARD_FIELDS = (
    *RewardTelemetry.from_state(RewardStateTelemetry()).flat_series_names(),
    "control_supervised_cmd_accel_local_jitter",
    "control_supervised_cmd_accel_trend_jitter",
    "control_supervised_cmd_accel_weighted_comfort_jitter",
)
_DIAGNOSTIC_TRACE_FIELDS = susan_diagnostic_trace_fieldnames(
    observation_fields=OBSERVATION_KEYS,
    reward_fields=_DIAGNOSTIC_REWARD_FIELDS,
)
_DIAGNOSTIC_TRACE_FIELDS = (*_DIAGNOSTIC_TRACE_FIELDS, "step_length_seconds")
_DIAGNOSTIC_TEXT_FIELDS = frozenset(
    {
        "motion_state",
        "stop_event",
        "stop_cause",
        "active_use_cases",
        "termination_reason",
        "truncation_reason",
        "ego_road_id",
        "traffic_light_tls_id",
        "ego_lead_vehicle_id",
        "pedestrian_id",
        "policy_object_1_id",
        "policy_object_1_kind",
        "policy_object_2_id",
        "policy_object_2_kind",
        "policy_object_3_id",
        "policy_object_3_kind",
        "simulator_read_quality",
    }
)
_DIAGNOSTIC_TENSORBOARD_FIELDS = frozenset(_DIAGNOSTIC_TRACE_FIELDS).difference(
    _DIAGNOSTIC_TEXT_FIELDS
)


class VehicleSimController:
    """Coordinate vehicle sim operations."""

    _episode_active_step_limit: int | None = None
    supervised_cmd_accel_delta: float = 0.0
    abs_supervised_cmd_accel_delta: float = 0.0
    supervised_cmd_accel_delta_change: float = 0.0
    abs_supervised_cmd_accel_delta_change: float = 0.0
    supervised_cmd_accel_local_jitter: float = 0.0
    supervised_cmd_accel_local_jitter_sign_change: float = 0.0
    supervised_cmd_accel_local_effective_jitter: float = 0.0
    supervised_cmd_accel_trend_jitter: float = 0.0
    supervised_cmd_accel_trend_jitter_sign_change: float = 0.0
    supervised_cmd_accel_trend_effective_jitter: float = 0.0
    supervised_cmd_accel_comfort_jitter: float = 0.0
    supervised_cmd_accel_weighted_comfort_jitter: float = 0.0
    supervised_cmd_accel_comfort_jitter_sign_change: float = 0.0
    supervised_cmd_accel_comfort_effective_jitter: float = 0.0
    supervised_cmd_accel_discount_jitter: float = 0.0
    normalized_supervised_cmd_accel_discount_jitter: float = 0.0
    pre_jitter_discount_speed_reward: float = 0.0
    speed_reward_jitter_discount_applied: float = 0.0
    speed_reward_jitter_discount: float = 1.0

    def _get_runtime_state(self) -> ControllerRuntimeState:
        """Return the runtime state exposed by this vehicle sim controller."""
        return self._runtime_state

    @property
    def policy_action(self) -> float:
        """Return the policy action exposed by this vehicle sim controller."""
        return self._get_runtime_state().command.policy_action

    @policy_action.setter
    def policy_action(self, value: float) -> None:
        """Handle the policy action operation owned by this vehicle sim controller."""
        self._get_runtime_state().command.policy_action = float(value)

    @property
    def _prev_upcoming_intersection_type_code(self) -> int | None:
        """Return the prev upcoming intersection type code exposed by this vehicle sim controller.
        """
        return self._get_runtime_state().traffic_signal.previous_intersection_type_code

    @_prev_upcoming_intersection_type_code.setter
    def _prev_upcoming_intersection_type_code(self, value: int | None) -> None:
        """Handle the prev upcoming intersection type code operation owned by this vehicle sim
        controller.
        """
        self._get_runtime_state().traffic_signal.previous_intersection_type_code = value

    @property
    def _prev_traffic_light_state_code(self) -> int | None:
        """Return the prev traffic light state code exposed by this vehicle sim controller."""
        return self._get_runtime_state().traffic_signal.previous_traffic_light_state_code

    @_prev_traffic_light_state_code.setter
    def _prev_traffic_light_state_code(self, value: int | None) -> None:
        """Handle the prev traffic light state code operation owned by this vehicle sim controller.
        """
        self._get_runtime_state().traffic_signal.previous_traffic_light_state_code = value

    @property
    def red_light_overdrive_this_step(self) -> bool:
        """Return the restricted-area decision computed for the current step."""

        return self._get_runtime_state().traffic_signal.red_light_overdrive_this_step

    @red_light_overdrive_this_step.setter
    def red_light_overdrive_this_step(self, value: bool) -> None:
        """Store the restricted-area decision computed for the current step."""

        self._get_runtime_state().traffic_signal.red_light_overdrive_this_step = bool(
            value
        )

    def __init__(self, *, simulator_port: SimulatorPort) -> None:
        """Initialize the vehicle sim controller and its owned state."""
        self._simulator_port = simulator_port
        self._runtime_state = ControllerRuntimeState()
        self._speed_out = 0
        self.cumulative_reward_counter_over_episodes = 0
        self._delta = 0
        self._controller_config = CONTROLLER_CONFIG
        self.ego_vehicle = _create_ego_vehicle()
        self.pedestrian_id_counter = 0
        self.controller_state = VehicleControllerTelemetryState()
        self.reward_config = RewardConfigTelemetry()
        self.reward_state = RewardStateTelemetry()
        self.episode_interactions = EpisodeInteractionTracker()
        self._diagnostic_trace_projector = SusanDiagnosticTraceProjector()
        self._diagnostic_trace_capture_armed = False

        self.scenario_context_preprocessor = ScenarioContextPreprocessor(
            simulator_port=simulator_port,
            network_file_path=self._resolve_network_file_path(),
            controller_config=self._controller_config,
        )
        self._sumo_command_service = SumoCommandService(
            simulator_port,
            self.controller_state,
        )
        self._traffic_light_service = TrafficLightService(self, simulator_port)
        self._math_service = ControllerMathService(self)

        self.observation_history = VehicleObservationTelemetryContainer(maxlen=OBSERVATION_HISTORY_LENGTH)
        self.telemetry = VehicleTelemetryContainer()
        self._observation_feature_service = ObservationFeatureService(
            cast(ObservationFeatureMathPort, self._math_service)
        )
        self._observation_service = ObservationService(
            history=self.observation_history,
            telemetry=self.telemetry,
            features=self._observation_feature_service,
            math=cast(ObservationMathPort, self._math_service),
            traffic_lights=cast(
                TrafficLightObservationPort,
                self._traffic_light_service,
            ),
        )
        self._reward_service = RewardService(
            state=cast(RewardStatePort, self),
            math=cast(RewardMathPort, self._math_service),
        )
        self._simulation_pipeline_service = SimulationPipelineService(
            controller=cast(SimulationPipelineControllerPort, self),
            observation_service=self._observation_service,
            reward_service=self._reward_service,
            math=cast(SimulationMathPort, self._math_service),
            telemetry_sink=cast(TelemetrySinkPort, self.telemetry),
        )

        # Reset the signal pair together to avoid a false red-light crossing.
        self._prev_upcoming_intersection_type_code = None
        self._prev_traffic_light_state_code = None
        self.policy_action = 0.0
        self._current_supervised_cmd_accel_delta = 0.0
        self.supervised_cmd_accel_delta = 0.0
        self.abs_supervised_cmd_accel_delta = 0.0
        self.supervised_cmd_accel_delta_change = 0.0
        self.abs_supervised_cmd_accel_delta_change = 0.0
        self.supervised_cmd_accel_local_jitter = 0.0
        self.supervised_cmd_accel_local_jitter_sign_change = 0.0
        self.supervised_cmd_accel_local_effective_jitter = 0.0
        self.supervised_cmd_accel_trend_jitter = 0.0
        self.supervised_cmd_accel_trend_jitter_sign_change = 0.0
        self.supervised_cmd_accel_trend_effective_jitter = 0.0
        self.supervised_cmd_accel_comfort_jitter = 0.0
        self.supervised_cmd_accel_weighted_comfort_jitter = 0.0
        self.supervised_cmd_accel_comfort_jitter_sign_change = 0.0
        self.supervised_cmd_accel_comfort_effective_jitter = 0.0
        self.supervised_cmd_accel_discount_jitter = 0.0
        self.normalized_supervised_cmd_accel_discount_jitter = 0.0
        self.pre_jitter_discount_speed_reward = 0.0
        self.speed_reward_jitter_discount_applied = 0.0
        self.speed_reward_jitter_discount = 1.0
        jitter_history_length = max(SPEED_REWARD_JITTER_TREND_WINDOW, SPEED_REWARD_JITTER_COMFORT_WINDOW)
        self.supervised_cmd_accel_history: deque[float] = deque(
            maxlen=jitter_history_length,
        )
        self.supervised_cmd_accel_delta_history: deque[float] = deque(
            maxlen=jitter_history_length,
        )

    @property
    def controller_config(self) -> ControllerConfig:
        """Return the controller config exposed by this vehicle sim controller."""
        return self._controller_config

    @property
    def current_scenario_context(self) -> ScenarioContext:
        """Expose the latest immutable scenario snapshot to services."""

        return self.scenario_context_preprocessor.current_context

    def update_scenario_context(self, ego_speed_mps: float) -> ScenarioContext:
        """Refresh the scenario snapshot for one control step."""

        return self.scenario_context_preprocessor.update(ego_speed_mps)

    def _get_simulation_pipeline_service(self) -> SimulationPipelineService:
        """Return the simulation pipeline service exposed by this vehicle sim controller."""
        try:
            return self._simulation_pipeline_service
        except AttributeError as exc:
            raise RuntimeError(
                "Simulation pipeline is unavailable before controller composition."
            ) from exc

    def _get_math_service(self) -> ControllerMathService:
        """Return the math service exposed by this vehicle sim controller."""
        try:
            return self._math_service
        except AttributeError:
            service = ControllerMathService(self)
            self._math_service = service
            return service

    @property
    def traffic_light_service(self) -> TrafficLightService:
        """Return the traffic light service exposed by this vehicle sim controller."""
        try:
            return self._traffic_light_service
        except AttributeError:
            service = TrafficLightService(self, self._simulator_port)
            self._traffic_light_service = service
            return service

    def _get_sumo_command_service(self) -> SumoCommandService:
        """Return the sumo command service exposed by this vehicle sim controller."""
        try:
            return self._sumo_command_service
        except AttributeError:
            service = SumoCommandService(
                self._simulator_port,
                self.controller_state,
            )
            self._sumo_command_service = service
            return service

    def _resolve_network_file_path(self) -> Path | None:
        """Resolve the network file path required by this vehicle sim controller."""
        scenario_configuration = getattr(self, "_scenario_manager_configuration", None)
        if scenario_configuration is None:
            return None

        try:
            configuration = cast(
                _ScenarioManagerConfigurationPort,
                scenario_configuration,
            )
            model_folder_path = configuration.get_model_folder_path()
            network_filename = configuration.get_network_filename()
        except (AttributeError, TypeError, ValueError):
            return None

        if not model_folder_path or not network_filename:
            return None
        return Path(model_folder_path) / network_filename

    def record_interaction_step(self, scenario: ScenarioContext) -> None:
        """Count supervisor-visible actors without consulting ``getLeader``."""

        self.pedestrian_id_counter = max(
            self.pedestrian_id_counter,
            scenario.pedestrian_count,
        )

        kind: InteractionKind | None = None
        actor_id: str | None = None
        if (
            self.reward_state.speed.blocking_profile_active
            and is_ego_blocked_by_pedestrian(scenario.pedestrian_parameters)
        ):
            kind = "pedestrian"
            actor_id = scenario.pedestrian_id
        elif scenario.supervisor_vehicle_conflict:
            kind = "lead"
            actor_id = next(iter(scenario.supervisor_vehicle_ids), None)
        self.episode_interactions.record_step(kind, actor_id)

    def reset(self) -> None:
        """Reset this vehicle sim controller to its initial operational state."""
        self._speed_out = 0
        self._delta = 0
        self.ego_vehicle = _create_ego_vehicle()
        self.pedestrian_id_counter = 0

        self.controller_state.reset_episode_values()
        self.reward_state.reset_step_values()
        self.episode_interactions.reset()

        self.observation_history.reset()
        self.telemetry.clear()
        self._diagnostic_trace_projector = SusanDiagnosticTraceProjector()
        self._simulation_pipeline_service.reset_episode()
        self.scenario_context_preprocessor.reset_episode()

        self.traffic_light_service.reset_episode()

        self._runtime_state.reset_episode()
        self._current_supervised_cmd_accel_delta = 0.0
        self.supervised_cmd_accel_delta = 0.0
        self.abs_supervised_cmd_accel_delta = 0.0
        self.supervised_cmd_accel_delta_change = 0.0
        self.abs_supervised_cmd_accel_delta_change = 0.0
        self.supervised_cmd_accel_local_jitter = 0.0
        self.supervised_cmd_accel_local_jitter_sign_change = 0.0
        self.supervised_cmd_accel_local_effective_jitter = 0.0
        self.supervised_cmd_accel_trend_jitter = 0.0
        self.supervised_cmd_accel_trend_jitter_sign_change = 0.0
        self.supervised_cmd_accel_trend_effective_jitter = 0.0
        self.supervised_cmd_accel_comfort_jitter = 0.0
        self.supervised_cmd_accel_weighted_comfort_jitter = 0.0
        self.supervised_cmd_accel_comfort_jitter_sign_change = 0.0
        self.supervised_cmd_accel_comfort_effective_jitter = 0.0
        self.supervised_cmd_accel_discount_jitter = 0.0
        self.normalized_supervised_cmd_accel_discount_jitter = 0.0
        self.pre_jitter_discount_speed_reward = 0.0
        self.speed_reward_jitter_discount_applied = 0.0
        self.speed_reward_jitter_discount = 1.0
        jitter_history_length = max(SPEED_REWARD_JITTER_TREND_WINDOW, SPEED_REWARD_JITTER_COMFORT_WINDOW)
        self.supervised_cmd_accel_history = deque(
            maxlen=jitter_history_length,
        )
        self.supervised_cmd_accel_delta_history = deque(
            maxlen=jitter_history_length,
        )

    @staticmethod
    def diagnostic_trace_fieldnames() -> tuple[str, ...]:
        """Return Susan's stable raw diagnostic trace schema."""

        return _DIAGNOSTIC_TRACE_FIELDS

    @staticmethod
    def diagnostic_trace_tensorboard_fields() -> frozenset[str]:
        """Return raw numeric fields allowed on evaluation TensorBoard."""

        return _DIAGNOSTIC_TENSORBOARD_FIELDS

    def get_training_diagnostic_component_series(
        self,
        *,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, float]:
        """Project finite diagnostics for framework-native aggregation."""

        snapshot = self._get_simulation_pipeline_service().latest_diagnostic_snapshot
        if snapshot is None:
            return {}
        row = self._diagnostic_trace_projector.project_training_metrics(
            snapshot,
            terminated=terminated,
            truncated=truncated,
        )
        return {
            name: float(cast(float | int, value))
            for name in TRAINING_DIAGNOSTIC_PARAM_NAMES
            if is_valid_number(value := row.get(name))
        }

    def latest_diagnostic_trace_row(
        self,
        *,
        scenario_index: int | None,
        terminated: bool,
        truncated: bool,
    ) -> dict[str, object] | None:
        """Project the latest already-read step without querying the simulator."""

        snapshot = self._get_simulation_pipeline_service().latest_diagnostic_snapshot
        if snapshot is None:
            return None
        row = self._diagnostic_trace_projector.project(
            snapshot,
            scenario_index=scenario_index,
            terminated=terminated,
            truncated=truncated,
            termination_reason=(
                getattr(self, "last_termination_reason", None)
                if terminated
                else None
            ),
            truncation_reason=(
                getattr(self, "last_truncation_reason", None)
                if truncated
                else None
            ),
        )
        configuration = getattr(self, "_scenario_manager_configuration", None)
        step_length_provider = getattr(configuration, "get_step_length", None)
        if callable(step_length_provider):
            step_length = step_length_provider()
            if is_valid_number(step_length) and float(step_length) > 0:
                row["step_length_seconds"] = float(step_length)
        return row

    def arm_diagnostic_trace_capture(self) -> None:
        """Arm lazy row projection for one complete checkpoint episode."""

        self._diagnostic_trace_capture_armed = True

    def capture_training_diagnostic_trace_row(
        self,
        *,
        scenario_index_provider: Callable[[], int],
        terminated: bool,
        truncated: bool,
    ) -> dict[str, object] | None:
        """Project one training row only while a checkpoint capture is armed."""

        if not getattr(self, "_diagnostic_trace_capture_armed", False):
            return None
        try:
            row = self.latest_diagnostic_trace_row(
                scenario_index=scenario_index_provider(),
                terminated=terminated,
                truncated=truncated,
            )
        except Exception:
            row = None
        if terminated or truncated:
            self._diagnostic_trace_capture_armed = False
        return row

    def is_running_into_restricted_area(
        self,
        previous_context: ScenarioContext | None = None,
        current_context: ScenarioContext | None = None,
    ) -> bool:
        """Detect a red crossing at the calibrated signal boundary."""

        crossing_result = self._configured_signal_boundary_crossing_result(
            previous_context,
            current_context,
        )
        if crossing_result is not None:
            return crossing_result

        # If we haven't recorded a previous state yet, no overdrive
        if self._prev_upcoming_intersection_type_code is None or self._prev_traffic_light_state_code is None:
            return False

        # Get previous state
        prev_intersection_type = self._prev_upcoming_intersection_type_code
        prev_traffic_light_state = self._prev_traffic_light_state_code

        # If the previous signal phase allowed crossing, no overdrive
        if (prev_intersection_type != INTERSECTION_TYPE_TRAFFIC_LIGHT or
                prev_traffic_light_state
                not in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES):
            return False

        # Previous state prohibited crossing at a traffic light
        # Now check current state from preprocessor
        try:
            scenario_context = self.scenario_context_preprocessor.current_context
        except AttributeError:
            return False
        current_intersection_type = self._safe_int(
            scenario_context.upcoming_intersection_type_code,
            None,
        )
        current_traffic_light_state = self._safe_int(
            scenario_context.traffic_light_state_code,
            None,
        )

        # If we can't determine current state, assume no overdrive
        if current_intersection_type is None or current_traffic_light_state is None:
            return False

        # Remaining at the same signal includes all SUMO transition phases.
        allowed_states = [
            (INTERSECTION_TYPE_TRAFFIC_LIGHT, TRAFFIC_LIGHT_STATE_RED),
            (INTERSECTION_TYPE_TRAFFIC_LIGHT, TRAFFIC_LIGHT_STATE_YELLOW),
            (INTERSECTION_TYPE_TRAFFIC_LIGHT, TRAFFIC_LIGHT_STATE_RED_YELLOW),
            (INTERSECTION_TYPE_TRAFFIC_LIGHT, TRAFFIC_LIGHT_STATE_GREEN),
        ]

        current_state = (current_intersection_type, current_traffic_light_state)

        # Return True if current state is NOT in allowed states (i.e., overdrive detected)
        return current_state not in allowed_states

    @staticmethod
    def _configured_signal_boundary_crossing_result(
        previous_context: ScenarioContext | None,
        current_context: ScenarioContext | None,
    ) -> bool | None:
        """Classify crossing of one configured calibrated signal boundary."""

        crossed_signal_boundary = (
            VehicleSimController._configured_signal_boundary_crossed_result(
                previous_context,
                current_context,
            )
        )
        if crossed_signal_boundary is None:
            return None
        if not crossed_signal_boundary:
            return False
        assert current_context is not None
        return (
            VehicleSimController._safe_int(
                current_context.upcoming_intersection_type_code,
                None,
            )
            == INTERSECTION_TYPE_TRAFFIC_LIGHT
            and VehicleSimController._safe_int(
                current_context.traffic_light_state_code,
                None,
            )
            in TRAFFIC_LIGHT_PROHIBITED_CROSSING_STATE_CODES
        )

    @staticmethod
    def crossed_configured_signal_boundary(
        previous_context: ScenarioContext | None,
        current_context: ScenarioContext | None,
    ) -> bool:
        """Return whether two same-episode snapshots cross one configured signal."""

        return bool(
            VehicleSimController._configured_signal_boundary_crossed_result(
                previous_context,
                current_context,
            )
        )

    @staticmethod
    def _configured_signal_boundary_crossed_result(
        previous_context: ScenarioContext | None,
        current_context: ScenarioContext | None,
    ) -> bool | None:
        """Return calibrated crossing geometry, or ``None`` for legacy contexts."""

        if previous_context is None or current_context is None:
            return None
        previous_internal_edge = previous_context.traffic_light_route_internal_edge
        current_internal_edge = current_context.traffic_light_route_internal_edge
        if not (current_internal_edge or previous_internal_edge):
            return None
        if not (previous_internal_edge and current_internal_edge):
            return False
        if str(previous_internal_edge) != str(current_internal_edge):
            return False

        previous_target_station = previous_context.target_ego_station_m
        current_target_station = current_context.target_ego_station_m
        if not (
            is_valid_number(previous_target_station)
            and is_valid_number(current_target_station)
        ):
            return False
        return (
            float(previous_target_station) < 0.0
            and float(current_target_station) >= 0.0
        )

    @overload
    @staticmethod
    def _safe_int(value: object, default: int) -> int:
        """Return int converted to a safe value for this vehicle sim controller."""
        ...

    @overload
    @staticmethod
    def _safe_int(value: object, default: None = None) -> int | None:
        """Return int converted to a safe value for this vehicle sim controller."""
        ...

    @staticmethod
    def _safe_int(value: object, default: int | None = None) -> int | None:
        """Convert a simulator code to ``int`` or return the supplied fallback."""

        try:
            return int(cast(int | float | str, value))
        except (TypeError, ValueError):
            return default

    def step(
        self,
        actions: Mapping[str, ScalarActionPort],
        step_counter: int,
    ) -> tuple[dict[str, np.ndarray], float]:
        """Advance the workflow owned by this vehicle sim controller by one step."""
        return self._get_simulation_pipeline_service().step(actions, step_counter)
