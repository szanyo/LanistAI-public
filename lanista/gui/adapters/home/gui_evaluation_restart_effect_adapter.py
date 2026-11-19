# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Own the confirmation effect for post-evaluation process replacement."""

from __future__ import annotations

import logging
from collections.abc import Callable

from lanista.application_restart_service import ApplicationRestartRequest
from lanista.gui.services.home.gui_evaluation_restart_flow_service import (
    GuiEvaluationRestartFlowService,
)
from lanista.reinforcengine.rl_model.evaluation_restart_contracts import (
    GuiEvaluationRestartContinuation,
)


class GuiEvaluationRestartEffectAdapter:
    """Present one owned warning and execute only a current confirmed action."""

    _QUESTION = (
        "LanistAI will restart and the current SUMO window will close. "
        "Do you want to continue?"
    )

    def __init__(
        self,
        *,
        flow: GuiEvaluationRestartFlowService,
        restart_required: Callable[[], bool],
        replay_provider: Callable[[], GuiEvaluationRestartContinuation | None],
        owner_identity: Callable[[], object],
        push_confirmation: Callable[[str, Callable[[bool | None], None]], None],
        request_restart: Callable[[ApplicationRestartRequest], object],
        notify_failure: Callable[[str], None],
        logger: logging.Logger,
    ) -> None:
        """Initialize the owned modal boundary."""

        self._flow = flow
        self._restart_required = restart_required
        self._replay_provider = replay_provider
        self._owner_identity = owner_identity
        self._push_confirmation = push_confirmation
        self._request_restart = request_restart
        self._notify_failure = notify_failure
        self._logger = logger
        self._dialog_owned = False
        self._active = True
        self._generation = 0

    def activate(self) -> None:
        """Activate a newly mounted Home-screen lifetime."""

        self._active = True

    def request(self, destination: str) -> None:
        """Ask for confirmation without duplicating a pending modal."""

        if not self._active or self._dialog_owned:
            return
        if not self._restart_required():
            self._notify_failure("No completed GUI evaluation requires a restart.")
            return
        try:
            request = self._flow.build_request(
                destination,
                self._replay_provider(),
            )
        except Exception as error:
            self._logger.exception("GUI evaluation restart request is invalid.")
            self._notify_failure(str(error))
            return
        owner = self._owner_identity()
        generation = self._generation
        self._dialog_owned = True

        def resolve(confirmed: bool | None) -> None:
            """Apply one current modal result."""

            self._dialog_owned = False
            if (
                not confirmed
                or not self._active
                or generation != self._generation
            ):
                return
            if owner != self._owner_identity() or not self._restart_required():
                self._logger.warning(
                    "Ignored a stale GUI evaluation restart confirmation."
                )
                return
            try:
                self._request_restart(request)
            except Exception as error:
                self._logger.exception("GUI evaluation process restart failed.")
                self._notify_failure(str(error))

        self._push_confirmation(self._QUESTION, resolve)

    def close(self) -> None:
        """Invalidate callbacks owned by an unmounted Home screen."""

        self._active = False
        self._dialog_owned = False
        self._generation += 1


__all__ = ["GuiEvaluationRestartEffectAdapter"]
