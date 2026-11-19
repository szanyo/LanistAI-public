# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Expose SB3 native inference as one-scenario sweep adapter."""

from lanista.reinforcengine.rl_model.evaluation.framework_adapter import (
    FrameworkScenarioEvaluationAdapter,
)


class SB3ScenarioEvaluationAdapter(FrameworkScenarioEvaluationAdapter):
    """Name the common adapter at the Stable-Baselines3 composition boundary."""


__all__ = ["SB3ScenarioEvaluationAdapter"]
