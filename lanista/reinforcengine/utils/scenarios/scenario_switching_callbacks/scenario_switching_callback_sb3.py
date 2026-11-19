# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Handle Stable-Baselines3 scenario-switching callbacks."""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from lanista.foundation.configuration import NestedDictWrapper
from lanista.foundation.logging import LoggerInterface
from lanista.reinforcengine.rl_environment.rolling_stats_sb3 import StableBaselinesRollingStats
from lanista.reinforcengine.rl_environment.training_progress import (
    TrainingProgressPlan,
)
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.utils.scenarios.manager.scenario_manager import ScenarioManager
from lanista.reinforcengine.utils.scenarios.scenario_switching_callbacks.curriculum_state_machine import (
    is_full_curriculum_state,
)


class StableBaselinesScenarioSwitchingCallback(BaseCallback, LoggerInterface):
    """Handle stable baselines scenario switching lifecycle callbacks."""

    def __init__(
            self,
            scenario_manager: ScenarioManager,
            training_conditions: NestedDictWrapper,
            curriculum_manager=None,
            learning_rate_resolver=None,
            learning_rate_applier=None,
            progress_reporter=None,
            training_progress_plan: TrainingProgressPlan | None = None,
            **kwargs
    ):
        """Initialize the stable baselines scenario switching callback and its owned state."""
        BaseCallback.__init__(self, training_conditions.training_verbosity)
        LoggerInterface.__init__(self, **kwargs)
        self._scenario_manager = scenario_manager
        self._curriculum_manager = curriculum_manager
        self._learning_rate_resolver = learning_rate_resolver
        self._learning_rate_applier = learning_rate_applier
        self._progress_reporter = progress_reporter
        self._training_progress_plan = training_progress_plan
        self._last_learning_rate = None

        kwargs = {
            "parent_logger_name": self._logger.name,
            "child_logger_name": "rolling_stats",
            "child_logger_level": self._logger.level
        }

        self._stats = StableBaselinesRollingStats(training_conditions, **kwargs)

    def _apply_curriculum_learning_rate(self, current_state) -> None:
        """Apply the curriculum learning rate transition owned by this stable baselines scenario
        switching callback.
        """
        if self._learning_rate_resolver is None or self._learning_rate_applier is None:
            return
        try:
            learning_rate = float(self._learning_rate_resolver(current_state))
        except (TypeError, ValueError):
            return
        if self._last_learning_rate == learning_rate:
            return
        applied = bool(self._learning_rate_applier(self.model, learning_rate))
        if applied:
            self._last_learning_rate = learning_rate
            self._logger.info(
                f"[PPO] Learning rate set to {learning_rate:.2e} "
                f"for curriculum={self._curriculum_name(current_state)}"
            )

    def _on_training_start(self) -> None:
        """Handle the training start event for this stable baselines scenario switching callback."""
        if self._curriculum_manager is not None:
            self._apply_curriculum_learning_rate(
                self._curriculum_manager.current_state()
            )
        self._publish_training_progress()

    def _publish_training_progress(self) -> None:
        """Publish metric milestone progress at the native SB3 callback boundary."""

        reporter = getattr(self, "_progress_reporter", None)
        plan = getattr(self, "_training_progress_plan", None)
        if reporter is None or plan is None:
            return
        current_state = (
            self._curriculum_manager.current_state()
            if self._curriculum_manager is not None
            else None
        )
        projection = plan.project(
            scenario_index=self._scenario_manager.get_index(),
            curriculum_state=current_state,
            evidence=self._stats.training_progress_evidence(current_state),
            finished=self._scenario_manager.finished(),
        )
        reporter(
            100.0,
            projection.percentage,
            RLModelState.RUNNING,
            projection.message,
        )

    @staticmethod
    def _curriculum_name(current_state) -> str:
        """Return the curriculum name value owned by this stable baselines scenario switching
        callback.
        """
        if current_state is None:
            return "full"
        if hasattr(current_state, "get"):
            return str(current_state.get("name") or current_state.get("state_name") or "curriculum")
        return str(getattr(current_state, "name", None) or getattr(current_state, "state_name", None) or "curriculum")

    def _log_progress_summary(
            self,
            *,
            current_state,
            metrics_ready: bool,
            scenario_ready: bool,
    ) -> None:
        """Log the progress summary owned by this stable baselines scenario switching callback."""
        episode_counts = self._stats.episode_count_summary()
        self._logger.info(
            f"[Progress] timesteps={self.num_timesteps} "
            f"total_episodes={episode_counts['total']} "
            f"terminated={episode_counts['terminated']} "
            f"truncated={episode_counts['truncated']} "
            f"curriculum={self._curriculum_name(current_state)} "
            f"scenario={self._scenario_manager.get_index()} "
            f"metrics_ready={metrics_ready} "
            f"scenario_ready={scenario_ready}"
        )

    def _on_step(self) -> bool:
        """Handle the step event for this stable baselines scenario switching callback."""
        infos = self.locals.get("infos", [])
        episode_finished = False
        for info in infos:
            if info.get("term") or info.get("trunc"):
                episode_finished = True
                incoming_metrics = self.model.logger.name_to_value
                stats = {
                    "n_episode_steps": self.locals.get("n_steps", 0),
                    "n_rollout_steps": self.locals.get("n_rollout_steps", 0),
                    "n_timesteps": self.num_timesteps,
                    "total_timesteps": self.model.num_timesteps,
                }
                self._stats.add_episode(info, incoming_metrics, stats)
        if episode_finished:
            self._stats.log_window_metric_stats()
            metrics_ready = self._stats.metrics_ready()
            current_state = None
            if self._curriculum_manager is not None:
                current_state = self._curriculum_manager.current_state()
                if current_state is None:
                    curriculum_ready = False
                elif is_full_curriculum_state(current_state):
                    curriculum_ready = True
                else:
                    curriculum_ready = self._stats.curriculum_ready(current_state)
                self._curriculum_manager.evaluate_and_update(curriculum_ready)
                current_state = self._curriculum_manager.current_state()
                self._apply_curriculum_learning_rate(current_state)

            scenario_ready = self._stats.scenario_ready(current_state)
            self._log_progress_summary(
                current_state=current_state,
                metrics_ready=metrics_ready,
                scenario_ready=scenario_ready,
            )
            if metrics_ready and scenario_ready:
                self._scenario_manager.next()
                if self._curriculum_manager is not None:
                    self._curriculum_manager.reset_for_scenario_switch()
                self._stats.reset()
                self._logger.debug(f"Switching to next scenario {self._scenario_manager.get_index()}")
                self._publish_training_progress()
                # The shared scenario state is consumed by each environment on
                # reset, so intermediate transitions stay inside one learn call.
                return not self._scenario_manager.finished()
            self._publish_training_progress()
        return True # Continue training iteration
