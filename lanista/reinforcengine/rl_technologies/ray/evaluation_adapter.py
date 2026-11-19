# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Expose Ray RLModule inference as one-scenario sweep adapter."""

from lanista.reinforcengine.rl_model.evaluation.framework_adapter import (
    FrameworkScenarioEvaluationAdapter,
)


class RayScenarioEvaluationAdapter(FrameworkScenarioEvaluationAdapter):
    """Name the common adapter at the Ray composition boundary."""


__all__ = ["RayScenarioEvaluationAdapter"]
