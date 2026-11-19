# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Reward-facing alias for the shared bounded telemetry container type."""

from __future__ import annotations

from ..observation.vehicle_telemetry_container import VehicleTelemetryContainer


RewardTelemetryContainer = VehicleTelemetryContainer
