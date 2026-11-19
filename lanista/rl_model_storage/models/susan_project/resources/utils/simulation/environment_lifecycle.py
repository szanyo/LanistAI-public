# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Framework-neutral reset and warmup policy for Susan environments."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

ConfiguredJitterSeconds = int | float | str | bytes | bytearray


class ResetSimulationPort(Protocol):
    """Simulator commands used by Susan's common reset sequence."""

    def shutdown_simulation(self) -> bool:
        """Close the current simulator session, if one exists."""

    def start_simulation(
        self,
        gui: bool,
        sumo_configuration_path: str,
    ) -> None:
        """Start one simulator session."""

    def sim_step(self) -> None:
        """Advance the mandatory simulation step."""

    def set_vehicle_speed_mode(
        self,
        vehicle_id: str,
        speed_mode: int,
    ) -> None:
        """Set the SUMO speed-control bitset for a vehicle."""

    def set_vehicle_speed(
        self,
        vehicle_id: str,
        speed_mps: float,
    ) -> None:
        """Set vehicle speed in metres per second."""

    def track_vehicle(self, view_id: str, vehicle_id: str) -> None:
        """Track one vehicle in the SUMO GUI."""

    def set_zoom(self, view_id: str, zoom: float) -> None:
        """Set the SUMO GUI zoom."""


@dataclass(frozen=True, slots=True)
class LifecycleInitialization:
    """Initial lifecycle metrics exposed by each framework adapter."""

    configured_jitter_seconds: int
    current_jitter_seconds: float = 0
    warmup_seconds: float = 0.0
    warmup_steps: int = 0
    maximum_warmup_seconds: float = 0.0
    maximum_warmup_steps: int = 0
    startup_steps: int = 1
    active_policy_step_limit: int = 0
    active_policy_seconds_limit: float = 0.0


@dataclass(frozen=True, slots=True)
class WarmupInput:
    """Explicit numeric inputs for one randomized warmup sample."""

    step_length_seconds: float
    scaled_jitter_seconds: float


@dataclass(frozen=True, slots=True)
class WarmupResult:
    """Warmup sample and externally reported bounds."""

    current_jitter_seconds: float
    warmup_seconds: float
    warmup_steps: int
    maximum_warmup_seconds: float
    maximum_warmup_steps: int


@dataclass(frozen=True, slots=True)
class CurriculumTimeLimitPolicy:
    """Configured curriculum stages used to scale one active-policy budget."""

    current_state: object | None
    configured_states: tuple[object, ...]
    initial_ratio: float


@dataclass(frozen=True, slots=True)
class TrainingBudgetInput:
    """Explicit numeric inputs for the active-policy step budget."""

    step_length_seconds: float
    scaled_jitter_seconds: float
    maximum_timesteps: int
    startup_steps: int
    simulation_begin_time_seconds: float | None = None
    simulation_end_time_seconds: float | None = None
    curriculum_time_limit: CurriculumTimeLimitPolicy | None = None


@dataclass(frozen=True, slots=True)
class TrainingBudgetResult:
    """Active-policy budget and the minimum required SUMO horizon."""

    current_jitter_seconds: float
    maximum_warmup_seconds: float
    maximum_warmup_steps: int
    active_policy_step_limit: int
    active_policy_seconds_limit: float
    required_sumo_horizon_seconds: float


@dataclass(frozen=True, slots=True)
class EnvironmentResetRequest:
    """Explicit collaborators required by the shared reset command sequence."""

    simulator: ResetSimulationPort
    resolve_sumo_configuration_path: Callable[[], str]
    sample_warmup_steps: Callable[[], int]
    calculate_active_step_limit: Callable[[], int]
    mark_episode_start: Callable[[int], None]
    log_info: Callable[[str], None]


class SusanEnvironmentLifecycle:
    """Calculate lifecycle metrics and execute common reset ordering."""

    def __init__(
        self,
        *,
        training_warmup_seconds: int,
        training_startup_steps: int,
    ) -> None:
        """Initialize the susan environment lifecycle and its owned state."""
        self._training_warmup_seconds = training_warmup_seconds
        self._training_startup_steps = training_startup_steps

    def initialize_metrics(
        self,
        configured_warmup_jitter_seconds: ConfiguredJitterSeconds,
    ) -> LifecycleInitialization:
        """Return normalized initial metrics without mutating an adapter."""

        return LifecycleInitialization(
            configured_jitter_seconds=max(
                0,
                int(configured_warmup_jitter_seconds),
            ),
            startup_steps=self._training_startup_steps,
        )

    def sample_warmup(self, request: WarmupInput) -> WarmupResult:
        """Sample the warmup window used by one environment reset."""

        step_length = float(request.step_length_seconds)
        jitter_seconds = float(request.scaled_jitter_seconds)
        minimum_seconds = max(
            0,
            self._training_warmup_seconds - jitter_seconds,
        )
        maximum_seconds = max(
            minimum_seconds,
            self._training_warmup_seconds + jitter_seconds,
        )
        minimum_steps = max(
            0,
            int(round(minimum_seconds / step_length)),
        )
        maximum_steps = max(
            minimum_steps,
            int(round(maximum_seconds / step_length)),
        )
        warmup_steps = random.randint(minimum_steps, maximum_steps)

        return WarmupResult(
            current_jitter_seconds=jitter_seconds,
            warmup_seconds=float(warmup_steps * step_length),
            warmup_steps=warmup_steps,
            maximum_warmup_seconds=float(maximum_seconds),
            maximum_warmup_steps=maximum_steps,
        )

    def calculate_training_budget(
        self,
        request: TrainingBudgetInput,
    ) -> TrainingBudgetResult:
        """Resolve the curriculum budget while preserving the full SUMO horizon."""

        step_length = float(request.step_length_seconds)
        jitter_seconds = float(request.scaled_jitter_seconds)
        maximum_warmup_seconds = max(
            0,
            self._training_warmup_seconds + jitter_seconds,
        )
        maximum_warmup_steps = int(round(maximum_warmup_seconds / step_length))
        maximum_active_limit = max(1, int(request.maximum_timesteps))
        active_limit = self._curriculum_active_limit(
            maximum_active_limit,
            request.curriculum_time_limit,
        )
        required_sumo_horizon_seconds = float(
            (
                maximum_warmup_steps
                + max(0, int(request.startup_steps))
                + maximum_active_limit
            )
            * step_length
        )
        self._validate_sumo_horizon(request, required_sumo_horizon_seconds)

        return TrainingBudgetResult(
            current_jitter_seconds=jitter_seconds,
            maximum_warmup_seconds=float(maximum_warmup_seconds),
            maximum_warmup_steps=maximum_warmup_steps,
            active_policy_step_limit=active_limit,
            active_policy_seconds_limit=float(active_limit * step_length),
            required_sumo_horizon_seconds=required_sumo_horizon_seconds,
        )

    @classmethod
    def _curriculum_active_limit(
        cls,
        maximum_active_limit: int,
        policy: CurriculumTimeLimitPolicy | None,
    ) -> int:
        """Release the active-policy budget linearly across curriculum stages."""

        if policy is None or cls._is_full_curriculum_state(policy.current_state):
            return maximum_active_limit

        initial_ratio = float(policy.initial_ratio)
        if not math.isfinite(initial_ratio) or not 0.0 < initial_ratio <= 1.0:
            raise ValueError(
                "Curriculum initial time-limit ratio must be finite and in (0, 1]."
            )

        restricted_states = tuple(
            state
            for state in policy.configured_states
            if not cls._is_full_curriculum_state(state)
        )
        if not restricted_states:
            return maximum_active_limit

        state_index = cls._curriculum_state_index(
            policy.current_state,
            restricted_states,
        )
        if state_index is None:
            return maximum_active_limit

        progress = state_index / len(restricted_states)
        active_ratio = initial_ratio + ((1.0 - initial_ratio) * progress)
        return max(
            1,
            min(
                maximum_active_limit,
                int(round(maximum_active_limit * active_ratio)),
            ),
        )

    @classmethod
    def _curriculum_state_index(
        cls,
        current_state: object,
        restricted_states: tuple[object, ...],
    ) -> int | None:
        """Resolve a validated restricted-state index with legacy fallback."""

        raw_index = cls._state_value(current_state, "curriculum_index", None)
        if type(raw_index) is int and 0 <= raw_index < len(restricted_states):
            return raw_index

        current_identity = cls._curriculum_state_identity(current_state)
        for index, state in enumerate(restricted_states):
            if cls._curriculum_state_identity(state) == current_identity:
                return index
        return None

    @classmethod
    def _is_full_curriculum_state(cls, state: object | None) -> bool:
        """Return whether a state represents the unrestricted route."""

        if state is None:
            return True
        name, target_type, target_value = cls._curriculum_state_identity(state)
        return (
            name == "full"
            or target_type in (None, "full")
            or target_value in (None, "full")
        )

    @classmethod
    def _curriculum_state_identity(
        cls,
        state: object,
    ) -> tuple[object, object, object]:
        """Return the stable configured identity of one curriculum state."""

        name = cls._state_value(
            state,
            "name",
            cls._state_value(state, "state_name", None),
        )
        return (
            name,
            cls._state_value(state, "target_type", None),
            cls._state_value(state, "target_value", None),
        )

    @staticmethod
    def _state_value(state: object, name: str, default: object) -> object:
        """Read one state/configuration value without framework assumptions."""

        if isinstance(state, Mapping):
            return state.get(name, default)
        return getattr(state, name, default)

    @staticmethod
    def _validate_sumo_horizon(
        request: TrainingBudgetInput,
        required_horizon_seconds: float,
    ) -> None:
        """Reject a SUMO horizon that cannot contain warmup and active control."""

        begin = request.simulation_begin_time_seconds
        end = request.simulation_end_time_seconds
        if begin is None or end is None:
            return
        available_horizon_seconds = float(end) - float(begin)
        if available_horizon_seconds + 1e-9 < required_horizon_seconds:
            raise ValueError(
                "SUMO simulation horizon is shorter than the maximum warmup, "
                "startup, and active-policy budget: "
                f"{available_horizon_seconds}s < {required_horizon_seconds}s"
            )

    @staticmethod
    def prepare_reset(
        reset_episode_state: Callable[[], None],
        reset_vehicle_state: Callable[[], None],
    ) -> None:
        """Reset episode state before vehicle-controller state."""

        reset_episode_state()
        reset_vehicle_state()

    def reset_training(self, request: EnvironmentResetRequest) -> None:
        """Run the common headless training/testing reset sequence."""

        controller = request.simulator
        controller.shutdown_simulation()
        controller.start_simulation(
            False,
            request.resolve_sumo_configuration_path(),
        )
        controller.sim_step()
        controller.set_vehicle_speed_mode("EGO", 39)
        controller.set_vehicle_speed("EGO", 0)
        self._run_warmup(request)
        request.log_info("Traffic flow started")

    def reset_deployment(self, request: EnvironmentResetRequest) -> None:
        """Run the common SUMO-GUI deployment reset sequence."""

        controller = request.simulator
        controller.start_simulation(
            True,
            request.resolve_sumo_configuration_path(),
        )
        controller.sim_step()
        controller.track_vehicle("View #0", "EGO")
        controller.set_zoom("View #0", 800)
        controller.set_vehicle_speed_mode("EGO", 39)
        controller.set_vehicle_speed("EGO", 0)
        request.log_info("Speeding up initial traffic flow...")
        self._run_warmup(request)
        request.log_info("Traffic flow initialized")

    @staticmethod
    def _run_warmup(request: EnvironmentResetRequest) -> None:
        """Run the warmup owned by this susan environment lifecycle."""
        controller = request.simulator
        for _ in range(request.sample_warmup_steps()):
            controller.sim_step()
        request.mark_episode_start(request.calculate_active_step_limit())


SUSAN_ENVIRONMENT_LIFECYCLE = SusanEnvironmentLifecycle(
    training_warmup_seconds=30,
    training_startup_steps=1,
)


def resolve_curriculum_time_limit_policy(
    model_configuration: object,
    current_state: object | None,
) -> CurriculumTimeLimitPolicy | None:
    """Resolve Susan's shared curriculum time-limit policy from configuration."""

    current = model_configuration
    for name in ("algorithm", "training_conditions", "curriculum_switch"):
        if isinstance(current, Mapping):
            current = current.get(name)
        else:
            current = getattr(current, name, None)
        if current is None:
            return None

    raw_states = (
        current.get("states", ())
        if isinstance(current, Mapping)
        else getattr(current, "states", ())
    )
    configured_states = (
        tuple(raw_states)
        if isinstance(raw_states, Sequence)
        and not isinstance(raw_states, str | bytes | bytearray)
        else ()
    )
    raw_ratio = (
        current.get("initial_time_limit_ratio", 1.0)
        if isinstance(current, Mapping)
        else getattr(current, "initial_time_limit_ratio", 1.0)
    )
    return CurriculumTimeLimitPolicy(
        current_state=current_state,
        configured_states=configured_states,
        initial_ratio=float(raw_ratio),
    )
