# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Coordinate tensorboard pack."""

import os
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import Protocol
from uuid import uuid4

from lanista.reinforcengine.application_errors import (
    ApplicationError,
    OperationResult,
    OperationTimeout,
    RetryPolicy,
    failure_from_exception,
)
from lanista.reinforcengine.rl_model.administration_safety_service import (
    AdministrationCompletionGuard,
)
from lanista.reinforcengine.rl_model.comparison.command_service import (
    ComparisonPackCommandService,
)
from lanista.reinforcengine.rl_model.comparison.contracts import (
    ComparisonPackCreationOutcome,
    ComparisonPackMutationOutcome,
)
from lanista.reinforcengine.rl_model.comparison.materializer import (
    ComparisonPackMaterializer,
)
from lanista.reinforcengine.rl_model.comparison.repository import (
    ComparisonPackRepository,
)
from lanista.reinforcengine.rl_model.comparison.tensorboard_entry import (
    TensorboardPackEntry,
    TensorboardRunMetadata,
)
from lanista.reinforcengine.rl_model.comparison.tensorboard_run_metadata import (
    TensorboardRunMetadataReader,
)
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.name_formatting_service import (
    NameFormattingService,
    NameSanitizer,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    TENSORBOARD_DIR_NAME,
    TENSORBOARD_EVALUATION_DIR_NAME,
    TENSORBOARD_TRAININGS_DIR_NAME,
)


class _ComparisonPackManifest(Protocol):
    """Define the typed boundary for comparison pack manifest."""

    def get_configuration(self) -> object:
        """Return the configuration exposed by this comparison pack manifest."""
        ...


class _ComparisonPackModelCandidate(Protocol):
    """Define the typed boundary for comparison pack model candidate."""

    @property
    def manifest(self) -> _ComparisonPackManifest:
        """Return the manifest exposed by this comparison pack model candidate."""
        ...

    @property
    def display_name(self) -> str:
        """Return the display name exposed by this comparison pack model candidate."""
        ...

    @property
    def effective_version(self) -> str:
        """Return the effective version exposed by this comparison pack model candidate."""
        ...


class _ComparisonPackCommandPort(Protocol):
    """Define the typed boundary for comparison pack command."""

    def create_comparison_pack(
        self,
        pack_name: str,
        source_paths: list[str],
    ) -> ComparisonPackCreationOutcome:
        """Create the comparison pack owned by this comparison pack command port."""
        ...

    def rename_comparison_pack(
        self,
        pack_name: str | None,
        renamed_pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Rename the comparison pack owned by this comparison pack command port."""
        ...

    def delete_comparison_pack(
        self,
        pack_name: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Delete the comparison pack through this comparison pack command port's safety boundary.
        """
        ...

    def update_comparison_pack_annotation(
        self,
        pack_name: str | None,
        annotation: str | None,
    ) -> ComparisonPackMutationOutcome:
        """Update the comparison pack annotation owned by this comparison pack command port."""
        ...

    def replace_comparison_pack_sources(
        self,
        pack_name: str | None,
        source_paths: list[str],
    ) -> ComparisonPackMutationOutcome:
        """Replace the comparison pack sources owned by this comparison pack command port."""
        ...


class TensorboardPackService:
    """Lists TensorBoard packs and delegates comparison pack lifecycle."""

    def __init__(
        self,
        configuration_provider: Callable[[], object | None],
        current_model_display_provider: Callable[[], str | None],
        model_candidates_provider: Callable[
            [],
            tuple[
                Sequence[_ComparisonPackModelCandidate],
                Sequence[object],
            ],
        ],
        sanitize_name: NameSanitizer,
        logger: object,
        models_root_provider: Callable[[], str] | None = None,
        administration_operation_factory: Callable[
            [], AbstractContextManager[AdministrationCompletionGuard]
        ]
        | None = None,
        comparison_pack_command: _ComparisonPackCommandPort | None = None,
        metadata_reader: TensorboardRunMetadataReader | None = None,
    ) -> None:
        """Initialize the tensorboard pack service and its owned state."""
        self._configuration_provider = configuration_provider
        self._current_model_display_provider = current_model_display_provider
        self._model_candidates_provider = model_candidates_provider
        self._logger = logger
        self._metadata_reader = metadata_reader or TensorboardRunMetadataReader()
        self._comparison_pack_repository = ComparisonPackRepository(
            active_model_folder_path_provider=self._active_model_folder_path,
            logger=logger,
        )
        materializer = ComparisonPackMaterializer(
            source_entries_provider=self.list_sources_for_comparison,
            sanitize_name=sanitize_name,
        )
        self._comparison_pack_command = comparison_pack_command or (
            ComparisonPackCommandService(
                active_model_folder_path_provider=self._active_model_folder_path,
                repository=self._comparison_pack_repository,
                materializer=materializer,
                sanitize_name=sanitize_name,
                logger=logger,
                models_root_provider=models_root_provider,
                administration_operation_factory=administration_operation_factory,
            )
        )

    @staticmethod
    def build_tensorboard_run_name(session_uuid: str, created_at_raw: str | None) -> str | None:
        """Build the tensorboard run name required by this tensorboard pack service."""
        return NameFormattingService.build_run_name(
            session_uuid,
            created_at_raw,
            invalid_strategy="sanitize",
            fallback="unknown-time",
        )

    @staticmethod
    def sanitize_tensorboard_name(value: str, limit: int = 40) -> str:
        """Return the sanitize tensorboard name value owned by this tensorboard pack service."""
        return NameFormattingService.sanitize_name(value, limit=limit, fallback="")

    def _active_model_folder_path(self) -> str:
        """Return the active model folder path value owned by this tensorboard pack service."""
        configuration = self._configuration_provider()
        if configuration is None:
            return ""
        return str(getattr(configuration, "model_folder_path", "") or "")

    def list_for_current_model(self) -> list[TensorboardPackEntry]:
        """Return the for current model available to this tensorboard pack service."""
        model_folder_path = self._active_model_folder_path()
        if not model_folder_path:
            return []

        model_display_name = (
            self._current_model_display_provider()
            or os.path.basename(model_folder_path.rstrip(os.sep))
        )
        return self.list_for_model(model_folder_path, model_display_name)

    def list_for_model(
        self,
        model_folder_path: str,
        model_display_name: str,
    ) -> list[TensorboardPackEntry]:
        """List inspection targets for one explicitly selected model."""

        raw_path = str(model_folder_path or "").strip()
        if not raw_path:
            return []
        normalized_path = os.path.abspath(raw_path)
        if not os.path.isdir(normalized_path):
            return []

        entries = self.list_training_evaluation_packs_for_model(
            normalized_path,
            model_display_name,
        )
        active_raw_path = str(self._active_model_folder_path() or "").strip()
        active_path = os.path.abspath(active_raw_path) if active_raw_path else ""
        repository = (
            self._comparison_pack_repository
            if normalized_path == active_path
            else ComparisonPackRepository(
                active_model_folder_path_provider=lambda: normalized_path,
                logger=self._logger,
            )
        )
        model_name = os.path.basename(normalized_path.rstrip(os.sep)) or model_display_name
        for comparison_pack in repository.list_packs():
            source_paths = list(comparison_pack.source_paths) or None
            metadata = self._metadata_reader.summarize_comparison(
                comparison_pack.path,
                declared_source_count=len(source_paths or ()),
            )
            entries.append(
                TensorboardPackEntry(
                    pack_type="comparison",
                    framework_uuid="mixed",
                    run_name=comparison_pack.name,
                    path=comparison_pack.path,
                    search_text=(
                        f"{model_display_name} comparison mixed "
                        f"{comparison_pack.name} "
                        f"{comparison_pack.annotation or ''} "
                        f"{self._metadata_search_text(metadata)}"
                    ).lower(),
                    annotation=comparison_pack.annotation,
                    source_paths=source_paths,
                    model_name=model_name,
                    model_display_name=model_display_name,
                    metadata=metadata,
                )
            )

        entries.sort(
            key=lambda entry: (entry.pack_type, entry.framework_uuid, entry.run_name), reverse=False
        )
        return entries

    def list_sources_for_comparison(
        self, include_all_models: bool = True
    ) -> list[TensorboardPackEntry]:
        """Return the sources for comparison available to this tensorboard pack service."""
        if not include_all_models:
            return [
                entry
                for entry in self.list_for_current_model()
                if entry.pack_type in {"training", "evaluation"}
            ]

        active_model_path = (
            os.path.abspath(self._active_model_folder_path())
            if self._active_model_folder_path()
            else ""
        )

        candidates: list[tuple[str, str]] = []
        seen_model_paths: set[str] = set()

        if active_model_path:
            active_display = self._current_model_display_provider() or os.path.basename(
                active_model_path.rstrip(os.sep)
            )
            candidates.append((active_model_path, active_display))
            seen_model_paths.add(active_model_path)

        valid_entries, _ = self._model_candidates_provider()
        for discovered in valid_entries:
            manifest = discovered.manifest
            configuration = manifest.get_configuration()
            model_folder_path = str(getattr(configuration, "model_folder_path", "") or "").strip()
            if not model_folder_path:
                continue
            normalized_path = os.path.abspath(model_folder_path)
            if normalized_path in seen_model_paths:
                continue
            model_name = str(
                getattr(configuration, "model_name", "") or discovered.display_name or ""
            ).strip()
            model_version = str(
                getattr(configuration, "model_version", "") or discovered.effective_version or ""
            ).strip()
            display_name = (
                EmptyModelGenerator.build_model_display_name(model_name, model_version)
                or model_name
                or discovered.display_name
            )
            candidates.append((normalized_path, display_name))
            seen_model_paths.add(normalized_path)

        source_entries: list[TensorboardPackEntry] = []
        for model_folder_path, model_display_name in candidates:
            source_entries.extend(
                self.list_training_evaluation_packs_for_model(
                    model_folder_path,
                    model_display_name,
                )
            )

        source_entries.sort(
            key=lambda entry: (
                str(entry.model_display_name or "").lower(),
                entry.pack_type,
                entry.framework_uuid,
                entry.run_name,
            )
        )
        return source_entries

    def list_training_evaluation_packs_for_model(
        self,
        model_folder_path: str,
        model_display_name: str,
    ) -> list[TensorboardPackEntry]:
        """Return the training evaluation packs for model available to this tensorboard pack
        service.
        """
        tensorboard_root = os.path.join(model_folder_path, TENSORBOARD_DIR_NAME)
        pack_roots = (
            ("training", os.path.join(tensorboard_root, TENSORBOARD_TRAININGS_DIR_NAME)),
            ("evaluation", os.path.join(tensorboard_root, TENSORBOARD_EVALUATION_DIR_NAME)),
        )

        model_name = os.path.basename(model_folder_path.rstrip(os.sep)) or model_display_name
        entries: list[TensorboardPackEntry] = []
        for pack_type, pack_root in pack_roots:
            for framework_uuid in ("ray", "sb3"):
                framework_root = os.path.join(pack_root, framework_uuid)
                if not os.path.isdir(framework_root):
                    continue
                for child in os.scandir(framework_root):
                    if not child.is_dir(follow_symlinks=False):
                        continue
                    session_run_name = str(child.name or "").strip()
                    if not session_run_name:
                        continue
                    run_directories = (
                        self._evaluation_run_directories(
                            session_run_name,
                            child.path,
                        )
                        if pack_type == "evaluation"
                        else [
                            (
                                session_run_name,
                                child.path,
                                self._metadata_reader.summarize_run(child.path),
                                True,
                            )
                        ]
                    )

                    for run_name, run_path, metadata, recursive_copy in run_directories:
                        search_text = (
                            f"{model_display_name} {pack_type} {framework_uuid} {run_name} "
                            f"{self._metadata_search_text(metadata)}"
                        ).lower()
                        entries.append(
                            TensorboardPackEntry(
                                pack_type=pack_type,
                                framework_uuid=framework_uuid,
                                run_name=run_name,
                                path=run_path,
                                search_text=search_text,
                                model_name=model_name,
                                model_display_name=model_display_name,
                                metadata=metadata,
                                recursive_copy=recursive_copy,
                            )
                        )
        return entries

    def _evaluation_run_directories(
        self,
        session_run_name: str,
        session_run_path: str,
    ) -> list[tuple[str, str, TensorboardRunMetadata, bool]]:
        """List isolated evaluation runs plus an optional direct legacy run."""

        nested_runs: list[tuple[str, str, TensorboardRunMetadata, bool]] = []
        for child in os.scandir(session_run_path):
            if not child.is_dir(follow_symlinks=False):
                continue
            metadata = self._metadata_reader.summarize_run(child.path)
            if metadata.event_file_count:
                nested_runs.append(
                    (
                        f"{session_run_name}/{child.name}",
                        child.path,
                        metadata,
                        True,
                    )
                )
        if not nested_runs:
            return [
                (
                    session_run_name,
                    session_run_path,
                    self._metadata_reader.summarize_run(session_run_path),
                    True,
                )
            ]

        direct_metadata = self._metadata_reader.summarize_direct_run(
            session_run_path
        )
        if direct_metadata.event_file_count:
            nested_runs.insert(
                0,
                (
                    f"{session_run_name}/legacy",
                    session_run_path,
                    direct_metadata,
                    False,
                ),
            )
        return nested_runs

    @staticmethod
    def _metadata_search_text(metadata: TensorboardRunMetadata) -> str:
        """Project event statistics into the existing case-insensitive filter."""

        duration_seconds = int(metadata.duration_seconds or 0)
        maximum_step = metadata.maximum_step if metadata.maximum_step is not None else 0
        return (
            f"duration {duration_seconds} seconds charts tags {metadata.chart_count} "
            f"scalars {metadata.scalar_tag_count} histograms {metadata.histogram_tag_count} "
            f"steps {maximum_step} events {metadata.event_record_count} "
            f"files {metadata.event_file_count} bytes {metadata.log_size_bytes}"
        )

    def create_comparison_pack(
        self,
        pack_name: str,
        source_paths: list[str],
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ComparisonPackCreationOutcome]:
        """Create the comparison pack owned by this tensorboard pack service."""
        return self._execute_command(
            operation="create",
            operation_id=operation_id,
            command=lambda: self._comparison_pack_command.create_comparison_pack(
                pack_name,
                source_paths,
            ),
        )

    def get_comparison_pack_root(self) -> str | None:
        """Return the comparison pack root exposed by this tensorboard pack service."""
        return self._comparison_pack_repository.get_root()

    def rename_comparison_pack(
        self,
        pack_name: str | None,
        renamed_pack_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Rename the comparison pack owned by this tensorboard pack service."""
        return self._execute_command(
            operation="rename",
            operation_id=operation_id,
            command=lambda: self._comparison_pack_command.rename_comparison_pack(
                pack_name,
                renamed_pack_name,
            ),
        )

    def delete_comparison_pack(
        self,
        pack_name: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Delete the comparison pack through this tensorboard pack service's safety boundary."""
        return self._execute_command(
            operation="delete",
            operation_id=operation_id,
            command=lambda: self._comparison_pack_command.delete_comparison_pack(pack_name),
        )

    def update_comparison_pack_annotation(
        self,
        pack_name: str | None,
        annotation: str | None,
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Update the comparison pack annotation owned by this tensorboard pack service."""
        return self._execute_command(
            operation="annotate",
            operation_id=operation_id,
            command=lambda: (
                self._comparison_pack_command.update_comparison_pack_annotation(
                    pack_name,
                    annotation,
                )
            ),
        )

    def replace_comparison_pack_sources(
        self,
        pack_name: str | None,
        source_paths: list[str],
        *,
        operation_id: str | None = None,
    ) -> OperationResult[ComparisonPackMutationOutcome]:
        """Replace the comparison pack sources owned by this tensorboard pack service."""
        return self._execute_command(
            operation="replace-sources",
            operation_id=operation_id,
            command=lambda: (
                self._comparison_pack_command.replace_comparison_pack_sources(
                    pack_name,
                    source_paths,
                )
            ),
        )

    def _execute_command[OutcomeT](
        self,
        *,
        operation: str,
        operation_id: str | None,
        command: Callable[[], OutcomeT],
    ) -> OperationResult[OutcomeT]:
        """Execute the command owned by this tensorboard pack service."""
        correlation_id = operation_id or uuid4().hex
        try:
            outcome = command()
        except Exception as error:
            failure = failure_from_exception(
                error,
                operation_id=correlation_id,
                retry_policy=(RetryPolicy.SAFE if isinstance(error, OperationTimeout) else None),
            )
            self._warning(
                "comparison_pack_command_failed "
                f"operation_id={correlation_id} operation={operation} "
                f"failure_code={failure.code} "
                f"exception_type={type(error).__name__}"
            )
            if not isinstance(error, ApplicationError) or error.__cause__ is not None:
                self._exception(error.__cause__ or error)
            return OperationResult.failed(failure)
        return OperationResult.success(
            outcome,
            operation_id=correlation_id,
        )

    def _warning(self, message: str) -> None:
        """Publish a warning record through this tensorboard pack service."""
        warning = getattr(self._logger, "warning", None)
        if not callable(warning):
            return
        try:
            warning(message)
        except Exception:
            return

    def _exception(self, error: BaseException) -> None:
        """Publish a exception record through this tensorboard pack service."""
        exception = getattr(self._logger, "exception", None)
        if not callable(exception):
            return
        exception_info = (type(error), error, error.__traceback__)
        try:
            exception(error, exc_info=exception_info)
        except TypeError:
            try:
                exception(error)
            except Exception:
                return
        except Exception:
            return
