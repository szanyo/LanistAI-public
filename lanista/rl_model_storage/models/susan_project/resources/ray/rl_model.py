# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan's Ray plugin for the shared framework runtime lifecycle."""

from lanista.domains.automotive.zala_zone.route_entities import (
    ZALA_ZONE_ROUTE_ENTITIES_PROVIDER,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.ray.model_logic_core import (
    RayModelLogicCore,
)
from lanista.reinforcengine.rl_technologies.ray.runtime_core import RayRuntimeCore
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    CallableEnvironmentFactory,
    FrameworkLogic,
    FrameworkLogicLoggerContext,
)

from .model_policy import SusanRayModelPolicy
from .sim_env import RaySimEnv


def _build_environment(
    runtime: BaseModelRuntime,
    context: FrameworkLogicLoggerContext,
) -> FrameworkLogic:
    """Build the environment required by this rl model module."""
    return RayModelLogicCore(
        runtime,
        environment_type=RaySimEnv,
        model_policy=SusanRayModelPolicy(),
        parent_logger_name=context.parent_logger_name,
        child_logger_name=context.child_logger_name,
        child_logger_level=context.child_logger_level,
    )


class SusanRayModelPlugin(RayRuntimeCore):
    """Bind Susan's specialized algorithm logic to the common Ray runtime."""

    environment_factory = CallableEnvironmentFactory(_build_environment)
    route_entities_provider = ZALA_ZONE_ROUTE_ENTITIES_PROVIDER
