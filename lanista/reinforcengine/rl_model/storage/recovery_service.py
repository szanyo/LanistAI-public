# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Root-confined storage recovery with single-use ownership receipts."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    OperationConflict,
    PersistenceError,
)
from lanista.reinforcengine.rl_model.metadata.configuration_schema import ConfigurationSchema
from lanista.reinforcengine.rl_model.metadata.model_manifest import ModelManifest
from lanista.reinforcengine.rl_model.metadata.revision_store import (
    read_existing_payload,
    require_regular_file,
)
from lanista.reinforcengine.rl_model.metadata.schema import MetadataKind
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_lock import StorageMutationLock
from lanista.reinforcengine.rl_model.storage.mutation_scope import (
    ModelsStorageMutationScope,
    StorageMutationLockFactory,
)
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard
from lanista.reinforcengine.rl_model.storage.transaction import StorageTransaction

ModelsRootProvider = Callable[[], str | os.PathLike[str]]
_VALIDATED_STORAGE_RECOVERY_MARKER = object()


class ValidatedStorageRecovery:
    """Opaque, context-bound proof accepted by the runtime safety layer."""

    __slots__ = ("_active", "_marker")

    _active: bool
    _marker: object

    def __init__(self) -> None:
        """Initialize the validated storage recovery and its owned state."""
        raise TypeError("Validated storage recovery is issued only by StorageRecoveryService.")

    def __setattr__(self, name: str, value: object) -> None:
        """Assign an attribute while preserving this validated storage recovery's invariants."""
        raise AttributeError("Validated storage recovery proof is immutable.")


def _new_validated_storage_recovery() -> ValidatedStorageRecovery:
    """Return the new validated storage recovery value owned by this storage recovery service
    module.
    """
    proof = object.__new__(ValidatedStorageRecovery)
    object.__setattr__(proof, "_marker", _VALIDATED_STORAGE_RECOVERY_MARKER)
    object.__setattr__(proof, "_active", True)
    return proof


def _invalidate_validated_storage_recovery(
    proof: ValidatedStorageRecovery,
) -> None:
    """Invalidate the validated storage recovery owned by this storage recovery service module."""
    object.__setattr__(proof, "_active", False)


def is_validated_storage_recovery(value: object) -> bool:
    """Return whether ``value`` is an active service-issued recovery proof."""

    if not isinstance(value, ValidatedStorageRecovery):
        return False
    try:
        return value._marker is _VALIDATED_STORAGE_RECOVERY_MARKER and value._active
    except AttributeError:
        return False


@dataclass(frozen=True, slots=True)
class StorageRecoveryReceipt:
    """Single-use proof that one model root passed recovery and readback."""

    receipt_id: str
    model_root: Path
    model_uuid: str
    model_manifest_digest: str
    model_configuration_digest: str


@dataclass(frozen=True, slots=True)
class StorageRecoveryResult:
    """Recovered evidence plus a fresh manifest projection from readback."""

    receipt: StorageRecoveryReceipt
    manifest: ModelManifest


@dataclass(frozen=True, slots=True)
class _StorageRecoveryRevision:
    """Carry the data required for storage recovery revision."""

    model_uuid: str
    model_manifest_digest: str
    model_configuration_digest: str


class StorageRecoveryService:
    """Recover one model under the shared models-root mutation lock."""

    def __init__(
        self,
        models_root_provider: ModelsRootProvider,
        *,
        lock_factory: StorageMutationLockFactory = StorageMutationLock,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialize the storage recovery service and its owned state."""
        self._models_root_provider = models_root_provider
        self._lock_factory = lock_factory
        self._timeout_seconds = max(0.0, float(timeout_seconds))
        self._receipt_lock = Lock()
        self._latest_receipts: dict[Path, StorageRecoveryReceipt] = {}

    def recover_manifest(self, manifest: ModelManifest) -> StorageRecoveryResult:
        """Recover and read back a model before issuing ownership evidence."""

        try:
            scope = ModelsStorageMutationScope.from_models_root(self._models_root_provider())
            with scope.lock(
                lock_factory=self._lock_factory,
                timeout_seconds=self._timeout_seconds,
            ):
                return self._recover_manifest_locked(manifest, scope.models_root)
        except PersistenceError:
            raise
        except Exception as error:
            raise PersistenceError("Model storage recovery or readback failed.") from error

    @contextmanager
    def validated_receipt_operation(
        self,
        receipt: StorageRecoveryReceipt,
        manifest: ModelManifest,
    ) -> Iterator[ValidatedStorageRecovery]:
        """Validate once and retain the storage lock for the caller's commit."""

        try:
            scope = ModelsStorageMutationScope.from_models_root(self._models_root_provider())
            storage_lock = scope.lock(
                lock_factory=self._lock_factory,
                timeout_seconds=self._timeout_seconds,
            )
            storage_lock.__enter__()
        except Exception as error:
            raise PersistenceError(
                "Storage recovery validation could not acquire its mutation scope."
            ) from error

        proof: ValidatedStorageRecovery | None = None
        try:
            proof = self._validate_and_claim_receipt_locked(
                receipt,
                manifest,
                models_root=scope.models_root,
            )
            try:
                yield proof
            finally:
                _invalidate_validated_storage_recovery(proof)
        finally:
            storage_lock.__exit__(None, None, None)

    def _recover_manifest_locked(
        self,
        manifest: ModelManifest,
        models_root: Path,
    ) -> StorageRecoveryResult:
        """Return the recover manifest locked value owned by this storage recovery service."""
        configured_root, configured_uuid = self._manifest_owner(manifest)
        model_root = self._canonical_model_root(
            configured_root,
            models_root=models_root,
        )
        self._invalidate_receipt(model_root)

        recovery = StorageTransaction.recover(model_root)
        if recovery.failed:
            raise PersistenceError("Model storage recovery did not complete successfully.")

        revision = self._readback_revision(model_root)
        current_root_text, current_uuid = self._manifest_owner(manifest)
        current_root = self._canonical_model_root(
            current_root_text,
            models_root=models_root,
        )
        if (
            current_root != model_root
            or current_uuid != configured_uuid
            or revision.model_uuid != configured_uuid
        ):
            raise PersistenceError("Recovered metadata does not match the selected model owner.")

        recovered_manifest = ModelManifest(str(model_root))
        try:
            recovered_manifest.load()
        except ApplicationError as error:
            raise PersistenceError(
                "Recovered model metadata could not be projected for runtime loading."
            ) from error
        recovered_root_text, recovered_uuid = self._manifest_owner(recovered_manifest)
        recovered_root = self._canonical_model_root(
            recovered_root_text,
            models_root=models_root,
        )
        if recovered_root != model_root or recovered_uuid != configured_uuid:
            raise PersistenceError("Recovered model projection does not match its storage owner.")

        receipt = StorageRecoveryReceipt(
            receipt_id=uuid4().hex,
            model_root=model_root,
            model_uuid=configured_uuid,
            model_manifest_digest=revision.model_manifest_digest,
            model_configuration_digest=revision.model_configuration_digest,
        )
        with self._receipt_lock:
            self._latest_receipts[model_root] = receipt
        return StorageRecoveryResult(
            receipt=receipt,
            manifest=recovered_manifest,
        )

    @staticmethod
    def _manifest_owner(manifest: ModelManifest) -> tuple[str, str]:
        """Return the manifest owner value owned by this storage recovery service."""
        configuration: object = manifest.get_configuration()
        model_root = str(getattr(configuration, "model_folder_path", "") or "").strip()
        model_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        if not model_root or not model_uuid:
            raise PersistenceError("The selected model does not expose a stable storage owner.")
        return model_root, model_uuid

    @staticmethod
    def _canonical_model_root(
        model_root: str | os.PathLike[str],
        *,
        models_root: Path,
    ) -> Path:
        """Return the canonical model root value owned by this storage recovery service."""
        ModelsStorageMutationScope.for_model_root(
            model_root,
            models_root=models_root,
        )
        return StoragePathGuard(models_root).require_descendant(
            model_root,
            must_exist=True,
        )

    @staticmethod
    def _readback_revision(model_root: Path) -> _StorageRecoveryRevision:
        """Return the readback revision value owned by this storage recovery service."""
        guard = StoragePathGuard(model_root)
        manifest_path = require_regular_file(
            guard,
            model_root / MODEL_MANIFEST_FILE_NAME,
        )
        configuration_path = require_regular_file(
            guard,
            model_root / MODEL_CONFIGURATION_FILE_NAME,
        )
        manifest_payload, manifest_digest = read_existing_payload(
            manifest_path,
            MetadataKind.MODEL_MANIFEST,
        )
        configuration_payload, configuration_digest = read_existing_payload(
            configuration_path,
            MetadataKind.MODEL_CONFIGURATION,
        )
        ConfigurationSchema.from_model_configuration_payload(
            configuration_payload,
            model_folder_path=str(model_root),
        )

        model_section = manifest_payload.get("model")
        if not isinstance(model_section, Mapping):
            raise PersistenceError("Recovered model metadata has no owner section.")
        model_uuid = str(model_section.get("uuid") or "").strip()
        if not model_uuid:
            raise PersistenceError("Recovered model metadata has no owner UUID.")
        return _StorageRecoveryRevision(
            model_uuid=model_uuid,
            model_manifest_digest=manifest_digest,
            model_configuration_digest=configuration_digest,
        )

    def _validate_and_claim_receipt_locked(
        self,
        receipt: StorageRecoveryReceipt,
        manifest: ModelManifest,
        *,
        models_root: Path,
    ) -> ValidatedStorageRecovery:
        """Validate the and claim receipt locked required by this storage recovery service."""
        current_root, current_uuid = self._receipt_owner(
            manifest,
            models_root=models_root,
        )
        if receipt.model_root != current_root or receipt.model_uuid != current_uuid:
            raise OperationConflict("The storage recovery receipt belongs to another model.")

        with self._receipt_lock:
            latest = self._latest_receipts.get(current_root)
            if latest != receipt:
                raise OperationConflict(
                    "The storage recovery receipt is stale or was already consumed."
                )
            del self._latest_receipts[current_root]

        revision = self._readback_revision(current_root)
        if (
            revision.model_uuid != receipt.model_uuid
            or revision.model_manifest_digest != receipt.model_manifest_digest
            or revision.model_configuration_digest != receipt.model_configuration_digest
        ):
            raise OperationConflict("Model metadata changed after storage recovery completed.")
        return _new_validated_storage_recovery()

    def _receipt_owner(
        self,
        manifest: ModelManifest,
        *,
        models_root: Path,
    ) -> tuple[Path, str]:
        """Return the receipt owner value owned by this storage recovery service."""
        try:
            configured_root, configured_uuid = self._manifest_owner(manifest)
            current_root = self._canonical_model_root(
                configured_root,
                models_root=models_root,
            )
        except Exception as error:
            raise OperationConflict(
                "The storage recovery receipt no longer matches the selected model."
            ) from error
        return current_root, configured_uuid

    def _invalidate_receipt(self, model_root: Path) -> None:
        """Invalidate the receipt owned by this storage recovery service."""
        with self._receipt_lock:
            self._latest_receipts.pop(model_root, None)


__all__ = [
    "StorageRecoveryReceipt",
    "StorageRecoveryResult",
    "StorageRecoveryService",
    "ValidatedStorageRecovery",
    "is_validated_storage_recovery",
]
