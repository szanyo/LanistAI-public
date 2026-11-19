# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Typed TensorBoard run naming and filesystem layout policy."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.rl_metrics.tensorboard_schema import (
    TENSORBOARD_SCHEMA_VERSION,
    resolve_tensorboard_profile,
)
from lanista.reinforcengine.rl_model.name_formatting_service import (
    NameFormattingService,
)
from lanista.reinforcengine.rl_model.sessions.configuration_wrapper import (
    SessionConfigurationWrapper,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    SESSION_MANIFEST_FILE_NAME,
    TENSORBOARD_DIR_NAME,
    TENSORBOARD_EVALUATION_DIR_NAME,
    TENSORBOARD_RAY_DIR_NAME,
    TENSORBOARD_SB3_DIR_NAME,
    TENSORBOARD_TRAININGS_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.names import (
    InvalidStorageName,
    SessionStorageName,
    StorageName,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.tensorboard_artifact_normalizer import (
    TensorboardArtifactNormalizationRequest,
    TensorboardArtifactNormalizationResult,
    TensorboardArtifactNormalizer,
)

_LOGGER = logging.getLogger(__name__)


class TensorboardLayoutMode(Enum):
    """Filesystem branches shared by runtime writers and application views."""

    TRAINING = TENSORBOARD_TRAININGS_DIR_NAME
    EVALUATION = TENSORBOARD_EVALUATION_DIR_NAME


class TensorboardLayoutLogger(Protocol):
    """Logging boundary for observable, non-fatal layout degradation."""

    def warning(self, message: str) -> None:
        """Publish a warning record through this tensorboard layout logger."""
        ...

    def error(self, message: str) -> None:
        """Publish a error record through this tensorboard layout logger."""
        ...


class TensorboardRunNameResolutionCode(StrEnum):
    """Stable success and failure codes for read-only run-name resolution."""

    LEGACY_DIRECTORY = "legacy_directory"
    CACHED_METADATA = "cached_metadata"
    DERIVED_METADATA = "derived_metadata"
    INVALID_REQUEST = "invalid_request"
    UNSAFE_STORAGE = "unsafe_storage"
    MODEL_ROOT_MISSING = "model_root_missing"
    SESSION_MISSING = "session_missing"
    MANIFEST_MISSING = "manifest_missing"
    INVALID_MANIFEST = "invalid_manifest"
    INVALID_RUN_NAME = "invalid_run_name"


@dataclass(frozen=True, slots=True)
class TensorboardRunNameResolution:
    """Read-only result that makes metadata degradation observable."""

    code: TensorboardRunNameResolutionCode
    run_name: str | None = None
    message: str | None = None

    @property
    def succeeded(self) -> bool:
        """Return whether a safe run directory name was resolved."""

        return self.run_name is not None


@dataclass(frozen=True, slots=True)
class _TensorboardSessionRunMetadata:
    """Minimal validated session projection used by naming policy."""

    display_name: str
    session_name: str
    created_at: str
    cached_run_name: str


@dataclass(frozen=True, slots=True)
class TensorboardRunNameRequest:
    """Storage coordinates needed to resolve one session's run name."""

    model_folder_path: str
    session_name: str


@dataclass(frozen=True, slots=True)
class TensorboardRuntimeLayoutRequest:
    """Inputs for the writer-owned, session-specific runtime directory."""

    model_folder_path: str
    session_name: str
    framework_uuid: str
    mode: TensorboardLayoutMode
    evaluation_run_name: str | None = None


@dataclass(frozen=True, slots=True)
class TensorboardApplicationLayoutRequest:
    """Inputs for the broader directory exposed to the TensorBoard process."""

    model_folder_path: str
    selected_session: str | None
    framework_uuid: str | None
    mode: TensorboardLayoutMode


class TensorboardRuntimeLayoutService:
    """Resolve names and normalize TensorBoard layout without owning writers."""

    def __init__(
        self,
        *,
        artifact_normalizer: TensorboardArtifactNormalizer | None = None,
    ) -> None:
        """Initialize the tensorboard runtime layout service and its owned state."""
        self._artifact_normalizer = artifact_normalizer or TensorboardArtifactNormalizer()

    def resolve_run_name(
        self,
        request: TensorboardRunNameRequest,
    ) -> TensorboardRunNameResolution:
        """Resolve one safe run name from a validated schema-v1 manifest."""

        try:
            session_name = SessionStorageName.parse(request.session_name)
            guard, model_root = self._validated_model_root(request.model_folder_path)
            manifest_path = self._validated_session_manifest_path(
                guard,
                model_root,
                session_name,
            )
        except InvalidStorageName:
            return self._failed_run_name(
                TensorboardRunNameResolutionCode.INVALID_REQUEST,
                "TensorBoard requires one valid session storage name.",
            )
        except FileNotFoundError as error:
            return self._missing_path_result(error)
        except (StoragePathViolation, ValueError) as error:
            self._log_resolution_exception(
                TensorboardRunNameResolutionCode.UNSAFE_STORAGE,
                error,
            )
            return self._failed_run_name(
                TensorboardRunNameResolutionCode.UNSAFE_STORAGE,
                "TensorBoard session storage coordinates are unsafe.",
            )

        metadata = self._read_session_run_metadata(manifest_path, session_name)
        if isinstance(metadata, TensorboardRunNameResolution):
            return metadata

        legacy_resolution = self._legacy_run_name(
            guard,
            model_root,
            session_name,
        )
        if legacy_resolution is not None:
            return legacy_resolution

        generated_run_name = self._generated_run_name(
            metadata,
            str(session_name),
        )
        cached_run_name = metadata.cached_run_name
        if not generated_run_name and cached_run_name:
            return self._validated_run_name_result(
                cached_run_name,
                TensorboardRunNameResolutionCode.CACHED_METADATA,
            )
        if generated_run_name:
            return self._validated_run_name_result(
                generated_run_name,
                TensorboardRunNameResolutionCode.DERIVED_METADATA,
            )
        return self._failed_run_name(
            TensorboardRunNameResolutionCode.INVALID_RUN_NAME,
            "TensorBoard run name cannot be derived from the validated session manifest.",
        )

    def resolve_runtime_log_dir(
        self,
        request: TensorboardRuntimeLayoutRequest,
        logger: TensorboardLayoutLogger,
    ) -> str | None:
        """Create and return the session-specific writer directory."""

        resolution = self.resolve_run_name(
            TensorboardRunNameRequest(
                model_folder_path=request.model_folder_path,
                session_name=request.session_name,
            )
        )
        if not resolution.succeeded:
            self._report_run_name_failure(
                logger,
                resolution,
                request.session_name,
            )
            return None
        run_name = resolution.run_name
        assert run_name is not None

        framework_root = self.framework_log_dir(
            model_folder_path=request.model_folder_path,
            mode=request.mode,
            framework_uuid=request.framework_uuid,
        )
        run_dir = self._confined_run_directory(
            model_folder_path=request.model_folder_path,
            framework_root=framework_root,
            run_name=run_name,
        )
        if run_dir is None:
            logger.error(
                "TensorBoard run directory was rejected because its storage coordinates are unsafe."
            )
            return None
        normalization = self.normalize_framework_root_artifacts(
            TensorboardArtifactNormalizationRequest(
                model_folder_path=request.model_folder_path,
                framework_root=framework_root,
                run_dir=run_dir,
                ensure_run_directory=True,
            )
        )
        if not normalization.coordinates_validated:
            logger.error(
                "TensorBoard run directory validation failed "
                f"(operation_id={normalization.operation_id})."
            )
            return None
        if normalization.recovery_required:
            logger.error(
                "TensorBoard storage recovery is required before opening a "
                "new writer "
                f"(operation_id={normalization.operation_id})."
            )
            return None
        if (
            request.mode is TensorboardLayoutMode.EVALUATION
            and request.evaluation_run_name
        ):
            return self._create_evaluation_log_dir(
                model_folder_path=request.model_folder_path,
                run_dir=run_dir,
                evaluation_run_name=request.evaluation_run_name,
                logger=logger,
            )
        self._ensure_schema_marker(run_dir, logger)
        return run_dir

    def _create_evaluation_log_dir(
        self,
        *,
        model_folder_path: str,
        run_dir: str,
        evaluation_run_name: str,
        logger: TensorboardLayoutLogger,
    ) -> str | None:
        """Create one root-confined evaluation directory."""

        evaluation_dir = self._confined_evaluation_directory(
            model_folder_path=model_folder_path,
            session_run_dir=run_dir,
            evaluation_run_name=evaluation_run_name,
        )
        if evaluation_dir is None:
            logger.error(
                "TensorBoard evaluation directory was rejected because "
                "its storage coordinates are unsafe."
            )
            return None
        try:
            Path(evaluation_dir).mkdir(parents=False, exist_ok=True)
        except OSError as error:
            logger.error(
                "TensorBoard evaluation directory could not be created "
                f"({type(error).__name__}: {error})."
            )
            return None
        self._ensure_schema_marker(evaluation_dir, logger)
        return evaluation_dir

    @staticmethod
    def _ensure_schema_marker(
        log_dir: str,
        logger: TensorboardLayoutLogger,
    ) -> None:
        """Create one non-event schema marker without replacing prior evidence."""

        marker_path = Path(log_dir) / "tensorboard-schema.json"
        if marker_path.exists():
            return
        try:
            with marker_path.open("x", encoding="utf-8") as marker:
                json.dump(
                    {
                        "schema_version": TENSORBOARD_SCHEMA_VERSION,
                        "profile": resolve_tensorboard_profile().value,
                    },
                    marker,
                    indent=2,
                    sort_keys=True,
                )
                marker.write("\n")
        except FileExistsError:
            return
        except OSError as error:
            logger.warning(
                "TensorBoard schema marker could not be created "
                f"({type(error).__name__}: {error})."
            )

    def resolve_application_log_dir(
        self,
        request: TensorboardApplicationLayoutRequest,
    ) -> str | None:
        """Return the framework directory viewed by the TensorBoard process.

        Manifest metadata remains read-only on this path. Training-mode legacy
        artifacts may still be normalized into the resolved session directory.
        """

        if not request.model_folder_path:
            return None
        framework_root = self.framework_log_dir(
            model_folder_path=request.model_folder_path,
            mode=request.mode,
            framework_uuid=request.framework_uuid,
        )
        if (
            self._confined_framework_directory(
                model_folder_path=request.model_folder_path,
                framework_root=framework_root,
            )
            is None
        ):
            return None
        if request.mode is TensorboardLayoutMode.TRAINING:
            selected_session = str(request.selected_session or "").strip()
            if not selected_session:
                return framework_root
            resolution = self.resolve_run_name(
                TensorboardRunNameRequest(
                    model_folder_path=request.model_folder_path,
                    session_name=selected_session,
                )
            )
            if not resolution.succeeded:
                return None
            run_name = resolution.run_name
            assert run_name is not None
            run_dir = self._confined_run_directory(
                model_folder_path=request.model_folder_path,
                framework_root=framework_root,
                run_name=run_name,
            )
            if run_dir is None:
                return None
            self.normalize_framework_root_artifacts(
                TensorboardArtifactNormalizationRequest(
                    model_folder_path=request.model_folder_path,
                    framework_root=framework_root,
                    run_dir=run_dir,
                )
            )
        return framework_root

    @staticmethod
    def framework_log_dir(
        *,
        model_folder_path: str,
        mode: TensorboardLayoutMode,
        framework_uuid: str | None,
    ) -> str:
        """Return the framework root selected by mode and engine."""

        normalized_framework = str(framework_uuid or "").strip().lower()
        framework_directory = (
            TENSORBOARD_RAY_DIR_NAME if normalized_framework == "ray" else TENSORBOARD_SB3_DIR_NAME
        )
        return os.path.join(
            model_folder_path,
            TENSORBOARD_DIR_NAME,
            mode.value,
            framework_directory,
        )

    def normalize_framework_root_artifacts(
        self,
        request: TensorboardArtifactNormalizationRequest,
    ) -> TensorboardArtifactNormalizationResult:
        """Move known framework-root artifacts as one recoverable batch."""

        return self._artifact_normalizer.normalize(request)

    @staticmethod
    def _validated_model_root(
        model_folder_path: str,
    ) -> tuple[StoragePathGuard, Path]:
        """Return the validated model root value owned by this tensorboard runtime layout service.
        """
        if not str(model_folder_path or "").strip():
            raise ValueError("The model storage root is required.")
        declared_root = Path(model_folder_path).expanduser().absolute()
        guard = StoragePathGuard(declared_root)
        if declared_root != guard.root or declared_root.is_symlink():
            raise StoragePathViolation("Model storage aliases are forbidden.")
        if not os.path.lexists(declared_root) or not declared_root.is_dir():
            raise FileNotFoundError("model_root")
        return guard, declared_root

    @staticmethod
    def _validated_session_manifest_path(
        guard: StoragePathGuard,
        model_root: Path,
        session_name: SessionStorageName,
    ) -> Path:
        """Return the validated session manifest path value owned by this tensorboard runtime
        layout service.
        """
        lexical_training_root = model_root / TRAINING_SESSIONS_DIR_NAME
        if not os.path.lexists(lexical_training_root):
            raise FileNotFoundError("session")
        training_root = guard.require_descendant(
            lexical_training_root,
            must_exist=True,
        )
        if (
            lexical_training_root != training_root
            or lexical_training_root.is_symlink()
            or not training_root.is_dir()
        ):
            raise StoragePathViolation("Training-session storage is aliased.")

        lexical_session_path = training_root / str(session_name)
        if not os.path.lexists(lexical_session_path):
            raise FileNotFoundError("session")
        session_path = guard.require_descendant(
            lexical_session_path,
            must_exist=True,
        )
        if (
            lexical_session_path != session_path
            or lexical_session_path.is_symlink()
            or not session_path.is_dir()
        ):
            raise StoragePathViolation("Session storage is aliased.")

        lexical_manifest_path = session_path / SESSION_MANIFEST_FILE_NAME
        if not os.path.lexists(lexical_manifest_path):
            raise FileNotFoundError("manifest")
        manifest_path = guard.require_descendant(
            lexical_manifest_path,
            must_exist=True,
        )
        if (
            lexical_manifest_path != manifest_path
            or lexical_manifest_path.is_symlink()
            or not manifest_path.is_file()
        ):
            raise StoragePathViolation("Session manifest is aliased.")
        return manifest_path

    @staticmethod
    def _confined_framework_directory(
        *,
        model_folder_path: str,
        framework_root: str,
    ) -> str | None:
        """Return the confined framework directory value owned by this tensorboard runtime layout
        service.
        """
        try:
            guard, declared_root = TensorboardRuntimeLayoutService._validated_model_root(
                model_folder_path
            )
            lexical_framework = Path(framework_root).expanduser().absolute()
            validated_framework = guard.require_descendant(lexical_framework)
            if (
                lexical_framework != validated_framework
                or lexical_framework.is_symlink()
                or lexical_framework.parent.parent.parent != declared_root
            ):
                raise StoragePathViolation("TensorBoard framework aliases are forbidden.")
        except (FileNotFoundError, StoragePathViolation, ValueError):
            return None
        return str(validated_framework)

    @staticmethod
    def _confined_run_directory(
        *,
        model_folder_path: str,
        framework_root: str,
        run_name: str,
    ) -> str | None:
        """Return the confined run directory value owned by this tensorboard runtime layout
        service.
        """
        try:
            run_component = StorageName.parse(run_name)
            guard = StoragePathGuard(model_folder_path)
            declared_root = Path(model_folder_path).expanduser().absolute()
            if declared_root != guard.root or declared_root.is_symlink():
                raise StoragePathViolation("Model storage aliases are forbidden.")
            lexical_framework = Path(framework_root).expanduser().absolute()
            validated_framework = guard.require_descendant(lexical_framework)
            if lexical_framework != validated_framework or lexical_framework.is_symlink():
                raise StoragePathViolation("TensorBoard framework aliases are forbidden.")
            lexical_run_dir = lexical_framework / str(run_component)
            validated_run_dir = guard.require_descendant(lexical_run_dir)
            if (
                lexical_run_dir != validated_run_dir
                or lexical_run_dir.is_symlink()
                or validated_run_dir.parent != validated_framework
            ):
                raise StoragePathViolation("TensorBoard run aliases are forbidden.")
        except (InvalidStorageName, StoragePathViolation, ValueError):
            return None
        return str(validated_run_dir)

    @staticmethod
    def _confined_evaluation_directory(
        *,
        model_folder_path: str,
        session_run_dir: str,
        evaluation_run_name: str,
    ) -> str | None:
        """Return one safe child directory under a validated session run."""

        try:
            evaluation_component = StorageName.parse(evaluation_run_name)
            guard = StoragePathGuard(model_folder_path)
            lexical_session_dir = Path(session_run_dir).expanduser().absolute()
            validated_session_dir = guard.require_descendant(
                lexical_session_dir,
                must_exist=True,
            )
            if (
                lexical_session_dir != validated_session_dir
                or lexical_session_dir.is_symlink()
                or not validated_session_dir.is_dir()
            ):
                raise StoragePathViolation(
                    "TensorBoard evaluation session aliases are forbidden."
                )
            lexical_evaluation_dir = lexical_session_dir / str(evaluation_component)
            validated_evaluation_dir = guard.require_descendant(
                lexical_evaluation_dir
            )
            if (
                lexical_evaluation_dir != validated_evaluation_dir
                or lexical_evaluation_dir.is_symlink()
                or validated_evaluation_dir.parent != validated_session_dir
            ):
                raise StoragePathViolation(
                    "TensorBoard evaluation aliases are forbidden."
                )
        except (
            FileNotFoundError,
            InvalidStorageName,
            StoragePathViolation,
            ValueError,
        ):
            return None
        return str(validated_evaluation_dir)

    @classmethod
    def _read_session_run_metadata(
        cls,
        manifest_path: Path,
        session_name: SessionStorageName,
    ) -> _TensorboardSessionRunMetadata | TensorboardRunNameResolution:
        """Read the session run metadata without changing this tensorboard runtime layout service's
        state.
        """
        try:
            configuration = SessionConfigurationWrapper(str(manifest_path)).load_configuration()
            session_section = getattr(configuration, "session", None)
            declared_storage_name = cls._configuration_text(
                session_section,
                "storage_name",
            )
            if (
                declared_storage_name
                and SessionStorageName.parse(declared_storage_name) != session_name
            ):
                raise ValueError("Session manifest storage_name does not match its directory.")
        except FileNotFoundError:
            return cls._failed_run_name(
                TensorboardRunNameResolutionCode.MANIFEST_MISSING,
                "TensorBoard session manifest is missing.",
            )
        except Exception as error:
            cls._log_resolution_exception(
                TensorboardRunNameResolutionCode.INVALID_MANIFEST,
                error,
            )
            return cls._failed_run_name(
                TensorboardRunNameResolutionCode.INVALID_MANIFEST,
                "TensorBoard session manifest is invalid or uses an unsupported schema.",
            )

        return _TensorboardSessionRunMetadata(
            display_name=(
                cls._configuration_text(configuration, "display_name")
                or cls._configuration_text(session_section, "display_name")
            ),
            session_name=(
                cls._configuration_text(configuration, "session_name")
                or cls._configuration_text(session_section, "name")
            ),
            created_at=(
                cls._configuration_text(configuration, "created_at")
                or cls._configuration_text(session_section, "created_at")
            ),
            cached_run_name=cls._configuration_text(
                configuration,
                "tensorboard_run_name",
            ),
        )

    @classmethod
    def _generated_run_name(
        cls,
        metadata: _TensorboardSessionRunMetadata,
        session_name: str,
    ) -> str | None:
        """Return the generated run name value owned by this tensorboard runtime layout service."""
        generated_run_name = NameFormattingService.sanitize_name(
            metadata.display_name or metadata.session_name,
            limit=80,
            fallback="",
        )
        if generated_run_name:
            return generated_run_name

        return NameFormattingService.build_run_name(
            session_name,
            metadata.created_at or None,
            invalid_strategy="sanitize",
            fallback="unknown-time",
        )

    @classmethod
    def _legacy_run_name(
        cls,
        guard: StoragePathGuard,
        model_root: Path,
        session_name: SessionStorageName,
    ) -> TensorboardRunNameResolution | None:
        """Return the legacy run name value owned by this tensorboard runtime layout service."""
        tensorboard_root = model_root / TENSORBOARD_DIR_NAME
        try:
            if os.path.lexists(tensorboard_root):
                validated_root = guard.require_descendant(
                    tensorboard_root,
                    must_exist=True,
                )
                if (
                    tensorboard_root != validated_root
                    or tensorboard_root.is_symlink()
                    or not tensorboard_root.is_dir()
                ):
                    raise StoragePathViolation("TensorBoard storage is aliased.")

            legacy_run_path = tensorboard_root / str(session_name)
            if not os.path.lexists(legacy_run_path):
                return None
            validated_legacy_path = guard.require_descendant(
                legacy_run_path,
                must_exist=True,
            )
            if (
                legacy_run_path != validated_legacy_path
                or legacy_run_path.is_symlink()
                or not legacy_run_path.is_dir()
            ):
                raise StoragePathViolation("Legacy TensorBoard storage is aliased.")
        except (FileNotFoundError, StoragePathViolation, ValueError) as error:
            cls._log_resolution_exception(
                TensorboardRunNameResolutionCode.UNSAFE_STORAGE,
                error,
            )
            return cls._failed_run_name(
                TensorboardRunNameResolutionCode.UNSAFE_STORAGE,
                "TensorBoard session storage coordinates are unsafe.",
            )
        return TensorboardRunNameResolution(
            code=TensorboardRunNameResolutionCode.LEGACY_DIRECTORY,
            run_name=str(session_name),
        )

    @classmethod
    def _validated_run_name_result(
        cls,
        run_name: str,
        code: TensorboardRunNameResolutionCode,
    ) -> TensorboardRunNameResolution:
        """Return the validated run name result value owned by this tensorboard runtime layout
        service.
        """
        try:
            run_component = StorageName.parse(run_name)
        except InvalidStorageName:
            return cls._failed_run_name(
                TensorboardRunNameResolutionCode.INVALID_RUN_NAME,
                "TensorBoard run-name metadata is not one safe storage component.",
            )
        return TensorboardRunNameResolution(
            code=code,
            run_name=str(run_component),
        )

    @staticmethod
    def _missing_path_result(
        error: FileNotFoundError,
    ) -> TensorboardRunNameResolution:
        """Return the missing path result value owned by this tensorboard runtime layout service."""
        marker = error.args[0] if error.args else None
        if marker == "model_root":
            return TensorboardRuntimeLayoutService._failed_run_name(
                TensorboardRunNameResolutionCode.MODEL_ROOT_MISSING,
                "TensorBoard model storage root is unavailable.",
            )
        if marker == "session":
            return TensorboardRuntimeLayoutService._failed_run_name(
                TensorboardRunNameResolutionCode.SESSION_MISSING,
                "TensorBoard session storage does not exist.",
            )
        return TensorboardRuntimeLayoutService._failed_run_name(
            TensorboardRunNameResolutionCode.MANIFEST_MISSING,
            "TensorBoard session manifest is missing.",
        )

    @staticmethod
    def _failed_run_name(
        code: TensorboardRunNameResolutionCode,
        message: str,
    ) -> TensorboardRunNameResolution:
        """Return the failed run name value owned by this tensorboard runtime layout service."""
        return TensorboardRunNameResolution(
            code=code,
            message=message,
        )

    @staticmethod
    def _report_run_name_failure(
        logger: TensorboardLayoutLogger,
        resolution: TensorboardRunNameResolution,
        session_name: str,
    ) -> None:
        """Report the run name failure owned by this tensorboard runtime
layout service."""
        message = resolution.message or "TensorBoard run name could not be resolved."
        logger.error(f"{message} (code={resolution.code.value}, session='{session_name}')")

    @staticmethod
    def _log_resolution_exception(
        code: TensorboardRunNameResolutionCode,
        error: Exception,
    ) -> None:
        """Log the resolution exception owned by this tensorboard runtime layout service."""
        _LOGGER.warning(
            "tensorboard_run_name_resolution_failed "
            f"code={code.value} exception_type={type(error).__name__}",
            exc_info=(type(error), error, error.__traceback__),
        )

    @staticmethod
    def _configuration_text(
        configuration: object,
        *attribute_names: str,
    ) -> str:
        """Return the configuration text value owned by this tensorboard runtime layout service."""
        for attribute_name in attribute_names:
            value = str(getattr(configuration, attribute_name, "") or "").strip()
            if value:
                return value
        return ""
