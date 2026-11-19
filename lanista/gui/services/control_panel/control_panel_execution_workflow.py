# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Textual-independent ownership registry for ControlPanel executions."""

from __future__ import annotations

from dataclasses import replace

from lanista.gui.services.control_panel.control_panel_execution_state_service import (
    ControlPanelExecutionStateService,
)
from lanista.gui.services.control_panel.control_panel_execution_workflow_contracts import (
    BeginFailureReconciliation,
    ExecutionAdoption,
    ExecutionBeginCommand,
    ExecutionCommand,
    ExecutionStopCommand,
    ExecutionTransition,
    ExecutionWorkflowContext,
    ExecutionWorkflowHandle,
    ExecutionWorkflowLease,
    ExecutionWorkflowRuntime,
    TerminalSessionEffect,
)
from lanista.reinforcengine.rl_model.execution_contracts import (
    ACTIVE_EXECUTION_STATES,
    ExecutionSnapshot,
    RLModelState,
)

_TERMINAL_STATES = frozenset(
    {
        RLModelState.STOPPED,
        RLModelState.COMPLETED,
        RLModelState.FAILED,
    }
)


class ControlPanelExecutionWorkflowCoordinator:
    """Own execution identity while delegating snapshot policy projection."""

    def __init__(
        self,
        projection_service: ControlPanelExecutionStateService | None = None,
    ) -> None:
        """Initialize the control panel execution workflow coordinator and its owned state."""
        self._projection_service = projection_service or ControlPanelExecutionStateService()
        self._context = ExecutionWorkflowContext(
            generation=0,
            active=True,
            runtime=None,
            run_id=None,
            handle=None,
            originating_session=None,
            snapshot=ExecutionSnapshot(None, RLModelState.IDLE, False),
            terminal_claim=None,
            begin_pending=False,
        )

    @property
    def context(self) -> ExecutionWorkflowContext:
        """Return the current immutable ownership snapshot."""

        return self._context

    def activate(self) -> ExecutionWorkflowContext:
        """Open a new UI lifetime after an earlier close."""

        if self._context.active:
            return self._context
        self._context = replace(
            self._context,
            generation=self._context.generation + 1,
            active=True,
            runtime=None,
            handle=None,
            begin_pending=False,
        )
        return self._context

    def close(self) -> ExecutionWorkflowContext:
        """Invalidate every command, lease, and delayed result."""

        self._context = replace(
            self._context,
            generation=self._context.generation + 1,
            active=False,
            runtime=None,
            handle=None,
            begin_pending=False,
        )
        return self._context

    def detached_run_matches(self, snapshot: ExecutionSnapshot) -> bool:
        """Return whether ``snapshot`` continues the run detached at close."""

        context = self._context
        return bool(
            context.runtime is None
            and context.run_id is not None
            and snapshot.run_id == context.run_id
        )

    def attach_runtime(
        self,
        runtime: ExecutionWorkflowRuntime | None,
        snapshot: ExecutionSnapshot,
        *,
        selected_session: str | None,
    ) -> ExecutionTransition | None:
        """Attach a replacement runtime and project its first snapshot."""

        if not self._context.active:
            return None
        same_detached_run = self.detached_run_matches(snapshot)
        if runtime is not self._context.runtime:
            self._context = replace(
                self._context,
                generation=self._context.generation + 1,
                runtime=runtime,
                run_id=self._context.run_id if same_detached_run else None,
                handle=None,
                originating_session=(
                    self._context.originating_session if same_detached_run else None
                ),
                terminal_claim=(self._context.terminal_claim if same_detached_run else None),
                begin_pending=False,
            )
        if runtime is None:
            projection = self._projection_service.project(snapshot)
            if projection is None:
                return None
            self._context = replace(
                self._context,
                run_id=projection.tracked_run_id,
                snapshot=projection.snapshot,
            )
            return ExecutionTransition(self._context, projection)
        return self.accept_snapshot(
            runtime,
            snapshot,
            selected_session=selected_session,
        )

    def adopt_attached_handle(
        self,
        runtime: ExecutionWorkflowRuntime,
        handle: ExecutionWorkflowHandle,
        snapshot: ExecutionSnapshot,
        *,
        selected_session: str | None,
    ) -> ExecutionWorkflowLease | None:
        """Attach a GUI observer to an already-running runtime worker."""

        context = self._context
        if (
            not context.active
            or runtime is not context.runtime
            or context.handle is not None
            or context.run_id is None
            or snapshot.run_id != context.run_id
            or handle.run_id != context.run_id
            or snapshot.state not in ACTIVE_EXECUTION_STATES
        ):
            return None
        self._context = replace(
            context,
            handle=handle,
            originating_session=context.originating_session or selected_session,
        )
        return ExecutionWorkflowLease(
            generation=self._context.generation,
            runtime=runtime,
            run_id=context.run_id,
            handle=handle,
        )

    def begin(
        self,
        runtime: ExecutionWorkflowRuntime,
        *,
        selected_session: str | None,
    ) -> ExecutionBeginCommand | None:
        """Reserve a fresh generation before calling ``begin_operation``."""

        if (
            not self._context.active
            or runtime is not self._context.runtime
            or self._context.begin_pending
        ):
            return None
        previous = self._context
        pending_snapshot = ExecutionSnapshot(
            run_id=None,
            state=RLModelState.STARTING,
            worker_alive=False,
        )
        self._context = replace(
            previous,
            generation=previous.generation + 1,
            run_id=None,
            handle=None,
            originating_session=selected_session,
            terminal_claim=None,
            snapshot=pending_snapshot,
            begin_pending=True,
        )
        return ExecutionBeginCommand(
            generation=self._context.generation,
            runtime=runtime,
            previous_run_id=previous.run_id,
            previous_session=previous.originating_session,
            previous_snapshot=previous.snapshot,
        )

    def project_pending_begin(
        self,
        command: ExecutionBeginCommand,
    ) -> ExecutionTransition | None:
        """Project the reserved start before blocking runtime preparation begins."""

        if not self._owns_begin(command):
            return None
        projection = self._projection_service.project(self._context.snapshot)
        if projection is None:
            return None
        return ExecutionTransition(self._context, projection)

    def adopt(
        self,
        command: ExecutionBeginCommand,
        handle: ExecutionWorkflowHandle,
        snapshot: ExecutionSnapshot,
    ) -> ExecutionAdoption | None:
        """Publish a begin handle only while its reservation is current."""

        if not self._owns_begin(command):
            return None
        transition = self.accept_snapshot(command.runtime, snapshot)
        if transition is None:
            return None
        run_id = snapshot.run_id
        if run_id is None or run_id != handle.run_id:
            self._context = replace(
                self._context,
                run_id=run_id,
                begin_pending=False,
            )
            transition = replace(transition, context=self._context)
            return ExecutionAdoption(
                lease=None,
                transition=transition,
                contract_matches=False,
            )
        self._context = replace(
            self._context,
            run_id=run_id,
            handle=handle,
            begin_pending=False,
        )
        transition = replace(transition, context=self._context)
        lease = ExecutionWorkflowLease(
            generation=self._context.generation,
            runtime=command.runtime,
            run_id=run_id,
            handle=handle,
        )
        return ExecutionAdoption(
            lease=lease,
            transition=transition,
            contract_matches=True,
        )

    def reconcile_begin_failure(
        self,
        command: ExecutionBeginCommand,
        snapshot: ExecutionSnapshot,
    ) -> BeginFailureReconciliation | None:
        """Restore the previous owner or adopt a run created before failure."""

        if not self._owns_begin(command):
            return None
        if snapshot.run_id is None or snapshot.run_id == command.previous_run_id:
            self._context = replace(
                self._context,
                run_id=command.previous_run_id,
                handle=None,
                originating_session=command.previous_session,
                snapshot=command.previous_snapshot,
                begin_pending=False,
            )
            return BeginFailureReconciliation(False, None)
        transition = self.accept_snapshot(command.runtime, snapshot)
        if transition is not None:
            self._context = replace(
                self._context,
                run_id=snapshot.run_id,
                begin_pending=False,
            )
            transition = replace(transition, context=self._context)
        return BeginFailureReconciliation(
            adopted_new_run=transition is not None,
            transition=transition,
        )

    def capture_command(
        self,
        runtime: ExecutionWorkflowRuntime,
    ) -> ExecutionCommand | None:
        """Freeze pause/resume ownership before invoking the runtime."""

        context = self._context
        if not context.active or runtime is not context.runtime or context.run_id is None:
            return None
        return ExecutionCommand(
            generation=context.generation,
            runtime=runtime,
            run_id=context.run_id,
        )

    def accept_command_result(
        self,
        command: ExecutionCommand,
        snapshot: ExecutionSnapshot,
    ) -> ExecutionTransition | None:
        """Accept a command result only for its unchanged runtime and run."""

        if not self._owns_command(command):
            return None
        return self.accept_snapshot(command.runtime, snapshot)

    def capture_stop(
        self,
        runtime: ExecutionWorkflowRuntime,
        *,
        close_training_session: bool,
    ) -> ExecutionStopCommand | None:
        """Freeze the owned run before the confirmation dialog opens."""

        context = self._context
        if not context.active or runtime is not context.runtime or context.run_id is None:
            return None
        return ExecutionStopCommand(
            generation=context.generation,
            runtime=runtime,
            run_id=context.run_id,
            lease=self.current_lease(),
            close_training_session=close_training_session,
        )

    def stop_is_current(
        self,
        command: ExecutionStopCommand,
        runtime: ExecutionWorkflowRuntime | None,
        snapshot: ExecutionSnapshot,
    ) -> bool:
        """Reject confirmation after unmount, runtime replacement, or new run."""

        context = self._context
        return bool(
            context.active
            and command.generation == context.generation
            and command.runtime is runtime
            and runtime is context.runtime
            and command.run_id == context.run_id
            and snapshot.run_id == command.run_id
        )

    def accept_stop_result(
        self,
        command: ExecutionStopCommand,
        snapshot: ExecutionSnapshot,
    ) -> ExecutionTransition | None:
        """Accept a stop result while the captured command remains current."""

        if not self.stop_is_current(command, command.runtime, snapshot):
            return None
        return self.accept_snapshot(command.runtime, snapshot)

    def observe_completion(
        self,
        lease: ExecutionWorkflowLease,
        snapshot: ExecutionSnapshot,
    ) -> ExecutionTransition | None:
        """Accept a runtime-owned handle result and decide observer release."""

        if not self._owns_lease(lease):
            return None
        transition = self.accept_snapshot(lease.runtime, snapshot)
        if transition is None:
            return None
        release_observer = transition.projection.task_release_eligible
        if release_observer:
            self._context = replace(self._context, handle=None)
            transition = ExecutionTransition(
                context=self._context,
                projection=transition.projection,
                release_observer=True,
            )
        return transition

    def accept_snapshot(
        self,
        runtime: ExecutionWorkflowRuntime,
        snapshot: ExecutionSnapshot,
        *,
        selected_session: str | None = None,
    ) -> ExecutionTransition | None:
        """Project a snapshot for the active runtime and tracked run."""

        context = self._context
        if not context.active or runtime is not context.runtime:
            return None
        projection = self._projection_service.project(
            snapshot,
            tracked_run_id=context.run_id,
        )
        if projection is None:
            return None
        originating_session = context.originating_session
        if (
            not context.begin_pending
            and context.run_id is None
            and projection.tracked_run_id is not None
        ):
            originating_session = originating_session or selected_session
        self._context = replace(
            context,
            run_id=(context.run_id if context.begin_pending else projection.tracked_run_id),
            originating_session=originating_session,
            snapshot=projection.snapshot,
        )
        return ExecutionTransition(self._context, projection)

    def claim_terminal(
        self,
        snapshot: ExecutionSnapshot,
        *,
        fallback_session: str | None,
    ) -> TerminalSessionEffect | None:
        """Claim terminal persistence exactly once for the originating run."""

        context = self._context
        claim = (snapshot.run_id, snapshot.state)
        if (
            not context.active
            or snapshot != context.snapshot
            or snapshot.state not in _TERMINAL_STATES
            or claim == context.terminal_claim
        ):
            return None
        self._context = replace(context, terminal_claim=claim)
        return TerminalSessionEffect(
            state=snapshot.state,
            session_name=context.originating_session or fallback_session,
        )

    def current_lease(self) -> ExecutionWorkflowLease | None:
        """Return the current handle lease, when one has been adopted."""

        context = self._context
        if (
            not context.active
            or context.runtime is None
            or context.run_id is None
            or context.handle is None
        ):
            return None
        return ExecutionWorkflowLease(
            generation=context.generation,
            runtime=context.runtime,
            run_id=context.run_id,
            handle=context.handle,
        )

    def _owns_begin(self, command: ExecutionBeginCommand) -> bool:
        """Return the owns begin value owned by this control panel execution workflow coordinator."""
        context = self._context
        return bool(
            context.active
            and command.generation == context.generation
            and command.runtime is context.runtime
            and context.run_id is None
            and context.begin_pending
        )

    def _owns_command(self, command: ExecutionCommand) -> bool:
        """Return the owns command value owned by this control panel execution workflow
        coordinator.
        """
        context = self._context
        return bool(
            context.active
            and command.generation == context.generation
            and command.runtime is context.runtime
            and command.run_id == context.run_id
        )

    def _owns_lease(self, lease: ExecutionWorkflowLease) -> bool:
        """Return the owns lease value owned by this control panel execution workflow coordinator."""
        context = self._context
        return bool(
            context.active
            and lease.generation == context.generation
            and lease.runtime is context.runtime
            and lease.run_id == context.run_id
            and lease.handle is context.handle
        )
