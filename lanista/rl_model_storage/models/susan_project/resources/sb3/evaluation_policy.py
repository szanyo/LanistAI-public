# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan's constant-speed diagnostic for the shared SB3 lifecycle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np

from lanista.reinforcengine.rl_metrics.diagnostic_trace_recorder import (
    EvaluationTraceSession,
)
from lanista.reinforcengine.rl_technologies.sb3.model_policy import (
    SB3EvaluationContext,
    SB3PolicyLogger,
)
from lanista.reinforcengine.utils.scenarios.scenario_selection import (
    apply_scenario_selection,
)
from lanista.reinforcengine.utils.scenarios.scenario_state.scenario_state_sb3 import (
    StableBaselinesScenarioState,
)

from ..utils.training.runtime_config import RUNTIME_CONFIG
from .sim_env import DEPLOYMENT, TESTING, StableBaselinesSimEnv


class _SpeedController(Protocol):
    """Define the typed boundary for speed controller."""

    def set_vehicle_speed(self, vehicle_id: str, speed_mps: float) -> object:
        """Command one vehicle speed."""


class _EgoVehicle(Protocol):
    """Define the typed boundary for ego vehicle."""

    prev_speed: float


class _SusanEvaluationEnvironment(Protocol):
    """Define the typed boundary for susan evaluation environment."""

    sumo_simulation_controller: _SpeedController
    ego_vehicle: _EgoVehicle
    _speed_out: float

    def reset(self) -> tuple[object, object]:
        """Reset the diagnostic episode."""

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[object, object, bool, bool, object]:
        """Advance one diagnostic step."""

    def get_step_cnt(self) -> int:
        """Return the current environment step."""


class _TensorboardWriter(Protocol):
    """Define the typed boundary for tensorboard writer."""

    def flush(self) -> object:
        """Flush buffered telemetry."""


class SusanSB3EvaluationPolicy:
    """Run Susan's no-checkpoint diagnostic without owning the lifecycle."""

    def __init__(self) -> None:
        """Initialize the susan sb3 evaluation policy and its owned state."""
        self._quality_degradations: set[str] = set()

    @property
    def quality_degradations(self) -> frozenset[str]:
        """Return the quality degradations exposed by this susan sb3 evaluation policy."""
        return frozenset(self._quality_degradations)

    def create_environment(
        self,
        context: SB3EvaluationContext,
    ) -> object:
        """Create the environment owned by this susan sb3 evaluation policy."""
        apply_scenario_selection(
            context.model_configuration,
            context.scenario_manager,
            purpose="evaluation",
            default_index=0,
            selected_index=context.runtime.get_evaluation_scenario_index(),
            logger=context.logger,
        )
        scenario_index_state = StableBaselinesScenarioState()  # type: ignore[no-untyped-call]
        context.scenario_manager.set_scenario_index_state(  # type: ignore[no-untyped-call]
            scenario_index_state
        )
        gui = context.runtime.get_evaluation_visualization() == "gui"
        return StableBaselinesSimEnv(  # type: ignore[no-untyped-call]
            verbose=False,
            detailed_verbose=False,
            parent_logger_name=context.logger.name,
            child_logger_name="stable_baselines_sim_env",
            rl_model_manifest=context.model_manifest,
            rl_model_mode=DEPLOYMENT if gui else TESTING,
            scenario_manager=context.scenario_manager,
            scenario_index_state=scenario_index_state,
            defer_gui_shutdown_to_process_restart=gui,
        )

    def evaluate(
        self,
        environment: object,
        context: SB3EvaluationContext,
    ) -> None:
        """Evaluate the operation owned by this susan sb3 evaluation policy."""
        env = cast(_SusanEvaluationEnvironment, environment)
        writer = cast(
            _TensorboardWriter,
            context.runtime.get_tensorboard_writer(),
        )
        target_speed = (
            RUNTIME_CONFIG.resolve_constant_speed_diagnostic_speed(
                context.model_configuration
            )
        )
        max_steps = (
            RUNTIME_CONFIG.resolve_constant_speed_diagnostic_max_steps(
                context.model_configuration
            )
        )
        trace_session = EvaluationTraceSession.create(
            runtime=context.runtime,
            source=env,
            framework="sb3",
            quality_reporter=self._trace_quality_reporter(context.logger),
        )
        try:
            terminated, truncated = self._run_diagnostic_episode(
                env,
                writer,
                trace_session,
                context,
                target_speed=target_speed,
                max_steps=max_steps,
            )
            trace_session.flush()
            self._flush_writer(writer, context.logger)
            context.logger.info(
                "Constant-speed diagnostic evaluation finished. "
                f"steps={env.get_step_cnt()}, target_speed={target_speed}, "
                f"terminated={terminated}, truncated={truncated}"
            )
        finally:
            trace_session.close()

    def _run_diagnostic_episode(
        self,
        environment: _SusanEvaluationEnvironment,
        writer: _TensorboardWriter,
        trace_session: EvaluationTraceSession,
        context: SB3EvaluationContext,
        *,
        target_speed: float,
        max_steps: int,
    ) -> tuple[bool, bool]:
        """Run one constant-speed episode and return its terminal state."""

        environment.reset()
        action = np.array([0.0], dtype=np.float64)
        self._force_speed(environment, target_speed)
        terminated = False
        truncated = False
        while (
            not (terminated or truncated)
            and environment.get_step_cnt() < max_steps
        ):
            context.runtime.check_pause()
            context.runtime.check_stop()
            self._force_speed(environment, target_speed)
            _, _, terminated, truncated, _ = (
                environment.step(action)
            )
            if not (terminated or truncated):
                self._force_speed(environment, target_speed)
            trace_session.append_latest(
                episode_index=0,
                terminated=terminated,
                truncated=truncated,
            )
            context.runtime.check_stop()
        return terminated, truncated

    def _trace_quality_reporter(
        self,
        logger: SB3PolicyLogger,
    ) -> Callable[[str, BaseException], None]:
        """Return a reporter that joins trace failures to policy quality state."""

        def report(event: str, error: BaseException) -> None:
            """Join one optional trace failure to policy quality state."""

            self._record_quality_degradation(event, logger, error)

        return report

    @staticmethod
    def _force_speed(
        environment: _SusanEvaluationEnvironment,
        speed_mps: float,
    ) -> None:
        """Force the speed owned by this susan sb3 evaluation policy."""
        speed = float(speed_mps)
        environment.sumo_simulation_controller.set_vehicle_speed(
            "EGO",
            speed,
        )
        environment._speed_out = speed
        environment.ego_vehicle.prev_speed = speed

    def _flush_writer(
        self,
        writer: _TensorboardWriter,
        logger: SB3PolicyLogger,
    ) -> bool:
        """Flush the writer owned by this susan sb3 evaluation policy."""
        try:
            writer.flush()
        except (
            AttributeError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            self._record_quality_degradation(
                "sb3_evaluation_tensorboard_flush_failed",
                logger,
                exc,
            )
            return False
        return True

    def _record_quality_degradation(
        self,
        event: str,
        logger: SB3PolicyLogger,
        exc: BaseException,
    ) -> None:
        """Record the quality degradation owned by this susan sb3 evaluation policy."""
        first_occurrence = event not in self._quality_degradations
        self._quality_degradations.add(event)
        if first_occurrence:
            logger.warning(
                f"event={event} quality=degraded "
                f"error_type={type(exc).__name__} error={exc}"
            )
