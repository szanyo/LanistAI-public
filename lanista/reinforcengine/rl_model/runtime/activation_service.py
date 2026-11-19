# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Application command for atomic active-runtime replacement."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Thread

from lanista.reinforcengine.application_errors import (
    OperationResult,
    RuntimeUnavailableError,
    exception_from_failure,
)
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationSafetyService,
)
from lanista.reinforcengine.rl_model.execution_contracts import RLModelState
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.model import BaseModelRuntime
from lanista.reinforcengine.rl_model.runtime.events import (
    ModelChangedEvent,
    RuntimeEvent,
    RuntimeEventPublisher,
)
from lanista.reinforcengine.rl_model.runtime.load_coordinator import (
    RuntimeLoadCoordinator,
    RuntimeLoadRequest,
)
from lanista.reinforcengine.rl_model.runtime.load_finalization_service import (
    LoadFinalizationService,
)
from lanista.reinforcengine.rl_model.runtime.load_service import LoadOutcome, LoadService
from lanista.reinforcengine.rl_model.runtime.requests import RuntimeInitializationRequest
from lanista.reinforcengine.rl_model.runtime.retirement_service import RuntimeRetirementService
from lanista.reinforcengine.rl_model.runtime_slot_commit_service import RuntimeSlotCommitService
from lanista.reinforcengine.rl_model.runtime_slot_store import (
    RuntimeSlotState,
    RuntimeSlotStore,
)
from lanista.reinforcengine.rl_model.storage.recovery_service import (
    StorageRecoveryReceipt,
    StorageRecoveryResult,
    StorageRecoveryService,
)


async def _finish_cancelled_recovery(
    worker: Thread,
    completion: asyncio.Future[StorageRecoveryResult],
) -> None:
    """Keep yielding while a cancelled recovery still owns storage."""

    current = asyncio.current_task()
    while worker.is_alive() or not completion.done():
        if current is not None:
            current.uncancel()
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            continue
    worker.join()
    if not completion.cancelled():
        completion.exception()


async def _recover_storage_off_event_loop(
    service: StorageRecoveryService,
    manifest: ModelManifest,
) -> StorageRecoveryResult:
    """Run recovery off-loop while retaining the load lease on cancellation."""

    loop = asyncio.get_running_loop()
    completion = loop.create_future()

    def recover() -> None:
        """Perform the recover step within the enclosing recover storage off event loop workflow."""
        try:
            receipt = service.recover_manifest(manifest)
        except BaseException as error:
            loop.call_soon_threadsafe(completion.set_exception, error)
        else:
            loop.call_soon_threadsafe(completion.set_result, receipt)

    worker = Thread(
        target=recover,
        name="lanistai-storage-recovery",
        daemon=True,
    )
    worker.start()
    try:
        return await asyncio.shield(completion)
    except asyncio.CancelledError:
        # A recovery thread may still own the storage lock. Finish it before
        # cancellation lets RuntimeLoadCoordinator release the load lease.
        await _finish_cancelled_recovery(worker, completion)
        raise
    finally:
        if completion.done():
            worker.join()


@dataclass(frozen=True, slots=True)
class _PreparedRuntimeCandidate:
    """Keep recovery evidence bound to the candidate it authorized."""

    load_outcome: LoadOutcome
    model_manifest: ModelManifest
    storage_recovery_receipt: StorageRecoveryReceipt | None


_RuntimeCandidate = _PreparedRuntimeCandidate | None


class _RequestBoundEventPublisher:
    """Drop application events emitted by a superseded runtime load."""

    def __init__(
        self,
        publisher: RuntimeEventPublisher | None,
        coordinator: RuntimeLoadCoordinator[_RuntimeCandidate, BaseModelRuntime | None],
        request: RuntimeLoadRequest,
        failure_logger: Callable[[str, Exception], None],
    ) -> None:
        """Initialize the request bound event publisher and its owned state."""
        self._publisher = publisher
        self._coordinator = coordinator
        self._request = request
        self._failure_logger = failure_logger

    def publish(self, event: RuntimeEvent) -> None:
        """Publish the operation owned by this request bound event publisher."""
        if self._publisher is None or not self._coordinator.is_current(self._request):
            return
        try:
            self._publisher.publish(event)
        except Exception as error:
            self._failure_logger(self._request.operation_id, error)


@dataclass(frozen=True, slots=True)
class RuntimeActivationRequest:
    """Select the manifest and optional framework for one slot transition."""

    model_manifest: ModelManifest | None
    selected_framework_uuid: str | None = None


@dataclass(frozen=True, slots=True)
class RuntimeActivationResult:
    """Report whether the successful transition left an active runtime."""

    active_runtime_available: bool


@dataclass(frozen=True, slots=True)
class _ActivationContext:
    """Carry the runtime context for activation."""

    request: RuntimeLoadRequest
    relay: _RequestBoundEventPublisher
    previous: RuntimeSlotState


class RuntimeActivationService:
    """Prepare, commit, publish, and retire one active-runtime transition."""

    def __init__(
        self,
        *,
        load_service: LoadService,
        finalization_service: LoadFinalizationService,
        slot_commit_service: RuntimeSlotCommitService,
        retirement_service: RuntimeRetirementService,
        administration_safety: AdministrationSafetyService,
        storage_recovery_service: StorageRecoveryService,
        slot_store: RuntimeSlotStore,
        initialization_request_factory: Callable[
            [ModelManifest], RuntimeInitializationRequest
        ],
        progress_reporter: Callable[
            [RuntimeEventPublisher | None, str, str, float, float],
            None,
        ],
        info_logger: Callable[[str], None],
        error_logger: Callable[[str], None],
        debug_logger: Callable[[str], None],
        event_publish_failure_logger: Callable[[str, Exception], None],
        storage_recovery_runner: Callable[
            [StorageRecoveryService, ModelManifest],
            Awaitable[StorageRecoveryResult],
        ] = _recover_storage_off_event_loop,
        native_gui_mutation_guard: Callable[[], None] = lambda: None,
    ) -> None:
        """Initialize the runtime activation service and its owned state."""
        self._load_service = load_service
        self._finalization_service = finalization_service
        self._slot_commit_service = slot_commit_service
        self._retirement_service = retirement_service
        self._administration_safety = administration_safety
        self._storage_recovery_service = storage_recovery_service
        self._slot_store = slot_store
        self._initialization_request_factory = initialization_request_factory
        self._progress_reporter = progress_reporter
        self._info_logger = info_logger
        self._error_logger = error_logger
        self._debug_logger = debug_logger
        self._event_publish_failure_logger = event_publish_failure_logger
        self._storage_recovery_runner = storage_recovery_runner
        self._native_gui_mutation_guard = native_gui_mutation_guard
        self._coordinator = RuntimeLoadCoordinator[
            _RuntimeCandidate,
            BaseModelRuntime | None,
        ]()
        self._last_issue: str | None = None

    def get_last_issue(self) -> str | None:
        """Return the latest current-request failure message, if any."""

        return self._last_issue

    async def execute(
        self,
        request: RuntimeActivationRequest,
        *,
        event_publisher: RuntimeEventPublisher | None = None,
    ) -> OperationResult[RuntimeActivationResult]:
        """Initialize a candidate and swap the active runtime only on success."""

        self._native_gui_mutation_guard()
        load_request = self._coordinator.begin()
        relay = _RequestBoundEventPublisher(
            event_publisher,
            self._coordinator,
            load_request,
            self._event_publish_failure_logger,
        )
        context = _ActivationContext(
            request=load_request,
            relay=relay,
            previous=self._slot_store.snapshot(),
        )
        self._last_issue = None
        relay.publish(ModelChangedEvent(context.previous.model, RLModelState.LOADING))
        self._report_progress(relay, "Preparing import", 5.0)
        try:
            if request.model_manifest is None:
                internal_result = await self._clear_runtime(context)
            else:
                internal_result = await self._replace_runtime(
                    context,
                    request.model_manifest,
                    selected_framework_uuid=request.selected_framework_uuid,
                )
        except asyncio.CancelledError:
            self._publish_cancelled(context)
            raise
        return self._to_public_result(internal_result)

    def _publish_cancelled(self, context: _ActivationContext) -> None:
        """Restore a terminal projection for a cancelled current request."""

        if not self._coordinator.is_current(context.request):
            return
        self._last_issue = "Runtime activation was cancelled."
        self._report_progress(context.relay, "Cancelled", 100.0)
        context.relay.publish(
            ModelChangedEvent(self._slot_store.snapshot().model, RLModelState.FAILED)
        )
        self._debug_logger(
            "Runtime activation cancelled before completion "
            f"[operation_id={context.request.operation_id}]."
        )

    @staticmethod
    def _to_public_result(
        result: OperationResult[BaseModelRuntime | None],
    ) -> OperationResult[RuntimeActivationResult]:
        """Convert this runtime activation service's value to public result."""
        if result.failure is not None:
            return OperationResult.failed(result.failure)
        return OperationResult.success(
            RuntimeActivationResult(active_runtime_available=result.value is not None),
            operation_id=result.operation_id,
        )

    async def _clear_runtime(
        self,
        context: _ActivationContext,
    ) -> OperationResult[BaseModelRuntime | None]:
        """Clear the runtime owned by this runtime activation service."""
        result = await self._coordinator.execute(
            context.request,
            prepare=self._prepare_clear,
            commit=lambda candidate: self._commit_clear(context, candidate),
            after_commit=lambda committed: self._retire_or_degrade(
                context.previous.model,
                replacement=committed,
                operation_id=context.request.operation_id,
            ),
            load_lease_factory=self._administration_safety.acquire_runtime_load_lease,
        )
        if not self._coordinator.is_current(context.request):
            return result
        if result.succeeded:
            self._report_progress(context.relay, "No manifest selected", 100.0)
            context.relay.publish(ModelChangedEvent(None, RLModelState.IDLE))
            self._debug_logger("Current model cleared.")
        else:
            self._last_issue = result.failure.message if result.failure is not None else None
            context.relay.publish(
                ModelChangedEvent(context.previous.model, RLModelState.FAILED)
            )
        return result

    @staticmethod
    async def _prepare_clear() -> None:
        """Prepare the clear required by this runtime activation service."""
        return None

    def _commit_clear(
        self,
        context: _ActivationContext,
        candidate: _RuntimeCandidate,
    ) -> BaseModelRuntime | None:
        """Commit the clear owned by this runtime activation service."""
        if candidate is not None:
            raise RuntimeUnavailableError("A clear transition received a runtime candidate.")
        self._slot_commit_service.clear(
            expected_revision=context.previous.revision,
            transition_owner_id=context.request.operation_id,
        )
        return None

    async def _replace_runtime(
        self,
        context: _ActivationContext,
        model_manifest: ModelManifest,
        *,
        selected_framework_uuid: str | None,
    ) -> OperationResult[BaseModelRuntime | None]:
        """Replace the runtime owned by this runtime activation service."""
        result = await self._coordinator.execute(
            context.request,
            prepare=lambda: self._prepare_candidate(
                context,
                model_manifest,
                selected_framework_uuid=selected_framework_uuid,
            ),
            commit=lambda candidate: self._commit_candidate(
                context,
                candidate,
            ),
            discard=lambda candidate: self._discard_candidate(context, candidate),
            after_commit=lambda committed: self._retire_or_degrade(
                context.previous.model,
                replacement=committed,
                operation_id=context.request.operation_id,
            ),
            load_lease_factory=self._administration_safety.acquire_runtime_load_lease,
        )
        return self._finalize_replacement(context, model_manifest, result)

    async def _prepare_candidate(
        self,
        context: _ActivationContext,
        model_manifest: ModelManifest,
        *,
        selected_framework_uuid: str | None,
    ) -> _RuntimeCandidate:
        """Prepare the candidate required by this runtime activation service."""
        recovery_receipt: StorageRecoveryReceipt | None = None
        candidate_manifest = model_manifest
        if self._administration_safety.snapshot().storage_recovery_required:
            self._report_progress(context.relay, "Recovering storage", 10.0)
            recovery = await self._storage_recovery_runner(
                self._storage_recovery_service,
                model_manifest,
            )
            recovery_receipt = recovery.receipt
            candidate_manifest = recovery.manifest
        loaded = await self._load_service.load(
            candidate_manifest,
            runtime_request=self._initialization_request_factory(
                candidate_manifest
            ),
            event_publisher=context.relay,
            selected_framework_uuid=selected_framework_uuid,
            requested_session_name=context.previous.session_name,
            requested_checkpoint_name=context.previous.checkpoint_name,
        )
        if loaded.model is not None:
            return _PreparedRuntimeCandidate(
                load_outcome=loaded,
                model_manifest=candidate_manifest,
                storage_recovery_receipt=recovery_receipt,
            )
        if loaded.failure is not None:
            raise exception_from_failure(loaded.failure)
        raise RuntimeUnavailableError(
            loaded.last_issue or "The model runtime could not be loaded."
        )

    def _commit_candidate(
        self,
        context: _ActivationContext,
        candidate: _RuntimeCandidate,
    ) -> BaseModelRuntime | None:
        """Commit the candidate owned by this runtime activation service."""
        if candidate is None or candidate.load_outcome.model is None:
            raise RuntimeUnavailableError("The prepared runtime candidate is empty.")
        loaded = candidate.load_outcome
        candidate_manifest = loaded.model_manifest or candidate.model_manifest
        recovery_receipt = candidate.storage_recovery_receipt
        finalization = self._finalization_service.build_successful_load(
            model_manifest=candidate_manifest,
            load_outcome=loaded,
            current_runtime_framework_uuid=loaded.current_runtime_framework_uuid,
        )
        committed_model = finalization.model or loaded.model
        return self._slot_commit_service.commit_loaded(
            RuntimeSlotState(
                model=committed_model,
                manifest=candidate_manifest,
                framework_uuid=finalization.current_runtime_framework_uuid,
                session_name=finalization.selected_session_name,
                checkpoint_name=finalization.selected_checkpoint_name,
                last_successful_model_name=finalization.last_successful_model_name,
                last_successful_framework_uuid=(
                    finalization.last_successful_framework_uuid
                ),
                last_successful_session_name=finalization.selected_session_name,
                last_successful_checkpoint_name=finalization.selected_checkpoint_name,
                revision=context.previous.revision,
            ),
            expected_revision=context.previous.revision,
            after_commit=lambda: self._finalization_service.persist_successful_load(
                finalization
            ),
            transition_owner_id=context.request.operation_id,
            storage_recovery_receipt=recovery_receipt,
        )

    async def _discard_candidate(
        self,
        context: _ActivationContext,
        candidate: _RuntimeCandidate,
    ) -> None:
        """Run the discard candidate responsibility for this runtime activation service
        asynchronously.
        """
        if candidate is None:
            return
        await self._retirement_service.retire(
            candidate.load_outcome.model,
            replacement=self._slot_store.snapshot().model,
            operation_id=context.request.operation_id,
            require_quiescence=True,
        )

    async def _retire_or_degrade(
        self,
        runtime: BaseModelRuntime | None,
        *,
        replacement: BaseModelRuntime | None,
        operation_id: str,
    ) -> None:
        """Run the retire or degrade responsibility for this runtime activation service
        asynchronously.
        """
        try:
            await self._retirement_service.retire(
                runtime,
                replacement=replacement,
                operation_id=operation_id,
                require_quiescence=True,
            )
        except Exception:
            self._administration_safety.mark_degraded(
                reload_required=True,
                storage_recovery_required=False,
            )
            raise

    def _finalize_replacement(
        self,
        context: _ActivationContext,
        model_manifest: ModelManifest,
        result: OperationResult[BaseModelRuntime | None],
    ) -> OperationResult[BaseModelRuntime | None]:
        """Return the finalize replacement value owned by this runtime activation service."""
        if not self._coordinator.is_current(context.request):
            return result
        model_name = str(
            getattr(model_manifest.get_configuration(), "model_name", "") or ""
        )
        if result.succeeded:
            self._report_progress(context.relay, "Completed", 100.0)
            context.relay.publish(ModelChangedEvent(result.value, RLModelState.IDLE))
            self._info_logger(f"Model import completed: '{model_name}'")
        else:
            self._last_issue = result.failure.message if result.failure is not None else None
            context.relay.publish(
                ModelChangedEvent(context.previous.model, RLModelState.FAILED)
            )
            failure_code = result.failure.code if result.failure is not None else "unknown"
            self._error_logger(
                f"Model '{model_name}' could not be loaded [{failure_code}]."
            )
        return result

    def _report_progress(
        self,
        publisher: RuntimeEventPublisher | None,
        step: str,
        progress: float,
    ) -> None:
        """Report the progress owned by this runtime activation service."""
        self._progress_reporter(publisher, "Loading model", step, progress, 100.0)
