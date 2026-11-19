# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Susan's SB3 plugin for the shared lifecycle and custom evaluation policy."""

from lanista.domains.automotive.zala_zone.route_entities import (
    ZALA_ZONE_ROUTE_ENTITIES_PROVIDER,
)
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_technologies.runtime_contracts import (
    CallableEnvironmentFactory,
    FrameworkLogic,
    FrameworkLogicLoggerContext,
)
from lanista.reinforcengine.rl_technologies.sb3.model_logic_core import (
    SB3ModelLogicCore,
)
from lanista.reinforcengine.rl_technologies.sb3.runtime_core import SB3RuntimeCore

from .model_policy import SusanSB3ModelPolicy
from .sim_env import StableBaselinesSimEnv


def _build_environment(
    runtime: BaseModelRuntime,
    context: FrameworkLogicLoggerContext,
) -> FrameworkLogic:
    """Build the environment required by this rl model module."""
    return SB3ModelLogicCore(
        runtime,
        environment_type=StableBaselinesSimEnv,
        model_policy=SusanSB3ModelPolicy(),
        parent_logger_name=context.parent_logger_name,
        child_logger_name=context.child_logger_name,
        child_logger_level=context.child_logger_level,
    )


class SusanSB3ModelPlugin(SB3RuntimeCore):
    """Bind Susan's specialized SB3 logic to the common runtime lifecycle."""

    environment_factory = CallableEnvironmentFactory(_build_environment)
    route_entities_provider = ZALA_ZONE_ROUTE_ENTITIES_PROVIDER
