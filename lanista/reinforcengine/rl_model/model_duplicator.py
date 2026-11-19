# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Duplicate model."""

import os
import shutil
import uuid
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.empty_model_generator import EmptyModelGenerator
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.model_clone_preflight import ModelClonePreflight
from lanista.reinforcengine.rl_model.name_formatting_service import NameFormattingService
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import ModelPluginNamespace
from lanista.reinforcengine.rl_model.sessions.training_session_manager import (
    TrainingSessionManager,
)
from lanista.reinforcengine.rl_model.storage.layout import (
    CLONE_MANIFEST_FILE_NAME,
    FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    RAY_RESOURCES_DIR_NAME,
    RESOURCES_DIR_NAME,
    SB3_RESOURCES_DIR_NAME,
    TENSORBOARD_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
    ensure_tensorboard_structure,
    find_existing_model_manifest_path,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import ModelsStorageMutationScope
from lanista.reinforcengine.rl_model.storage.names import ModelStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)
from lanista.reinforcengine.rl_model.storage.transaction import FailureInjector, StorageTransaction


@dataclass(frozen=True, slots=True)
class _ClonePlan:
    """Validated source, target, and metadata used by one clone transaction."""

    models_root: str
    source_folder: Path
    target_folder: Path
    target_model_name: str
    target_model_version: str
    source_description: str
    source_framework: str
    parent_model_uuid: str | None
    selected_relative_paths: tuple[str, ...] | None


class ModelDuplicator:
    """Clone validated model packages through an atomic storage transaction."""

    _NON_CLONABLE_ROOT_DIRS = {
        TENSORBOARD_DIR_NAME,
        TRAINING_SESSIONS_DIR_NAME,
    }
    _GENERATED_RELATIVE_PATHS = {
        MODEL_MANIFEST_FILE_NAME,
        MODEL_CONFIGURATION_FILE_NAME,
        CLONE_MANIFEST_FILE_NAME,
        "__init__.py",
        f"{RESOURCES_DIR_NAME}/__init__.py",
    }
    _IGNORED_FILE_PATTERNS = ("__pycache__", "*.pyc", "*.pyo")

    @dataclass
    class DuplicateResult:
        """Report the result of duplicate."""

        target_model_name: str
        target_model_folder_path: str
        parent_model_uuid: str | None
        initial_session_uuid: str | None
        auto_load: bool = True

    @staticmethod
    def _sanitize_session_name(value: str, limit: int = 48) -> str:
        """Return the sanitize session name value owned by this model duplicator."""
        return NameFormattingService.sanitize_name(value, limit=limit, fallback="session")

    @classmethod
    def _normalize_relative_path(cls, value: str) -> str:
        """Normalize the relative path accepted by this model duplicator."""
        normalized = str(value or "").strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        normalized = normalized.rstrip("/")
        return normalized

    @classmethod
    def _is_non_clonable_relative_path(cls, relative_path: str) -> bool:
        """Return whether non clonable relative path applies to this model duplicator."""
        normalized = cls._normalize_relative_path(relative_path)
        if not normalized:
            return False
        if normalized in cls._GENERATED_RELATIVE_PATHS:
            return True
        root = normalized.split("/", 1)[0]
        return root in cls._NON_CLONABLE_ROOT_DIRS

    @classmethod
    def _is_ignored_entry_name(cls, name: str) -> bool:
        """Return whether ignored entry name applies to this model duplicator."""
        return any(fnmatch(name, pattern) for pattern in cls._IGNORED_FILE_PATTERNS)

    def _copytree_ignore_filter(
        self, source_model_folder_path: str, directory: str, names: list[str]
    ) -> set[str]:
        """Return the copytree ignore filter value owned by this model duplicator."""
        ignored: set[str] = set()
        for name in names:
            if self._is_ignored_entry_name(name):
                ignored.add(name)
                continue
            source_path = os.path.join(directory, name)
            relative_path = self._normalize_relative_path(
                os.path.relpath(source_path, source_model_folder_path)
            )
            if self._is_non_clonable_relative_path(relative_path):
                ignored.add(name)
        return ignored

    def _copy_all_cloneable_content(
        self, source_model_folder_path: str, target_model_folder_path: str
    ) -> None:
        """Copy the all cloneable content owned by this model duplicator."""
        shutil.copytree(
            source_model_folder_path,
            target_model_folder_path,
            ignore=lambda directory, names: self._copytree_ignore_filter(
                source_model_folder_path, directory, names
            ),
            dirs_exist_ok=True,
        )

    def _copy_selected_clone_content(
        self,
        source_model_folder_path: str,
        target_model_folder_path: str,
        selected_relative_paths: list[str],
    ) -> None:
        """Copy the selected clone content owned by this model duplicator."""
        os.makedirs(target_model_folder_path, exist_ok=True)

        normalized_paths = {
            self._normalize_relative_path(relative_path)
            for relative_path in selected_relative_paths
            if self._normalize_relative_path(relative_path)
        }

        for relative_path in sorted(normalized_paths):
            if self._is_non_clonable_relative_path(relative_path):
                continue

            source_path = os.path.join(source_model_folder_path, relative_path)
            if not os.path.exists(source_path):
                continue

            target_path = os.path.join(target_model_folder_path, relative_path)
            if os.path.isdir(source_path):
                shutil.copytree(
                    source_path,
                    target_path,
                    ignore=lambda directory, names: self._copytree_ignore_filter(
                        source_model_folder_path, directory, names
                    ),
                    dirs_exist_ok=True,
                )
                continue

            entry_name = os.path.basename(source_path)
            if self._is_ignored_entry_name(entry_name):
                continue
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.copy2(source_path, target_path)

    def _targetize_framework_manifest_configurations(
        self, target_model_folder_path: str, *, logical_target_model_folder_path: str | None = None
    ) -> None:
        """Retarget framework manifest configurations to the duplicated model."""
        target_model_package = os.path.basename(
            str(logical_target_model_folder_path or target_model_folder_path).rstrip(os.sep)
        )
        for framework_dir_name in (RAY_RESOURCES_DIR_NAME, SB3_RESOURCES_DIR_NAME):
            relative_path = os.path.join(
                RESOURCES_DIR_NAME,
                framework_dir_name,
                FRAMEWORK_MODEL_CONFIGURATION_FILE_NAME,
            )
            target_path = os.path.join(target_model_folder_path, relative_path)
            if not os.path.isfile(target_path):
                continue

            target_cfg = Configuration()
            target_cfg.set_config_file_path(target_path)
            target_cfg.load()
            model_implementation = getattr(target_cfg, "model_implementation", None)
            if model_implementation is None:
                continue

            if hasattr(model_implementation, "get_data"):
                model_implementation_data = dict(model_implementation.get_data())
            else:
                model_implementation_data = dict(model_implementation)
            model_implementation_data["module"] = ModelPluginNamespace.create(
                model_package=target_model_package,
                framework_uuid=framework_dir_name,
            ).module("rl_model")
            target_cfg.model_implementation = model_implementation_data
            target_cfg.save()

    @classmethod
    def _validate_selected_paths(
        cls,
        source_folder: Path,
        selected_relative_paths: list[str] | None,
    ) -> tuple[str, ...] | None:
        """Validate the selected paths required by this model duplicator."""
        if selected_relative_paths is None:
            return None

        source_guard = StoragePathGuard(source_folder)
        validated_paths: list[str] = []
        for relative_path in selected_relative_paths:
            normalized_path = cls._normalize_relative_path(relative_path)
            if not normalized_path:
                continue
            validated_path = source_guard.resolve(normalized_path, must_exist=True)
            validated_paths.append(validated_path.relative_to(source_folder).as_posix())
        return tuple(validated_paths)

    def _prepare_clone(
        self,
        *,
        source_model_folder_path: str,
        target_model_name: str,
        target_model_version: str | None,
        selected_relative_paths: list[str] | None,
        models_root: str | None,
    ) -> _ClonePlan:
        """Prepare the clone required by this model duplicator."""
        resolved_models_root = os.path.abspath(
            str(models_root or os.path.dirname(os.path.abspath(source_model_folder_path)))
        )
        models_guard = StoragePathGuard(resolved_models_root)
        source_folder = models_guard.require_descendant(
            source_model_folder_path,
            must_exist=True,
        )
        if source_folder.parent != models_guard.root:
            raise StoragePathViolation(
                "Source models must be direct children of the model storage root."
            )
        normalized_target_name = str(target_model_name).strip()
        if not normalized_target_name:
            raise ValueError("target_model_name is required.")

        target_folder = models_guard.resolve(ModelStorageName.parse(normalized_target_name))
        if os.path.lexists(target_folder):
            raise FileExistsError(f"Target model folder already exists: {target_folder}")

        validated_paths = self._validate_selected_paths(
            source_folder,
            selected_relative_paths,
        )
        source_metadata = ModelClonePreflight(source_folder).inspect(validated_paths)
        resolved_target_version = (
            str(target_model_version or source_metadata.model_version or "0.1.0").strip() or "0.1.0"
        )
        return _ClonePlan(
            models_root=resolved_models_root,
            source_folder=source_folder,
            target_folder=target_folder,
            target_model_name=normalized_target_name,
            target_model_version=resolved_target_version,
            source_description=source_metadata.description,
            source_framework=source_metadata.preferred_framework_uuid,
            parent_model_uuid=source_metadata.model_uuid,
            selected_relative_paths=validated_paths,
        )

    @staticmethod
    def _reset_generated_storage(staging_path: str) -> None:
        """Reset the generated storage owned by this model duplicator."""
        for directory_name in (TRAINING_SESSIONS_DIR_NAME, TENSORBOARD_DIR_NAME):
            generated_root = os.path.join(staging_path, directory_name)
            if os.path.isdir(generated_root):
                shutil.rmtree(generated_root)
        ensure_tensorboard_structure(staging_path)

    def _create_initial_session(
        self,
        *,
        staging_path: str,
        target_model_name: str,
        description: str,
    ) -> str:
        """Create the initial session owned by this model duplicator."""
        session_uuid = str(uuid.uuid4())
        session_manager = TrainingSessionManager(
            staging_path,
            # ``duplicate`` already owns the models-root lock while this
            # transaction-private staging tree is populated.
            storage_lock_already_held=True,
        )
        session_manager.ensure_session_structure(session_uuid)
        created_at = utc_now_iso()
        initializer: dict[str, object] = {
            "session": {
                "name": self._sanitize_session_name(target_model_name),
                "uuid": session_uuid,
                "description": str(description or "").strip(),
                "tags": [],
                "created_at": created_at,
                "modified_at": created_at,
                "parent_session_uuid": None,
                "child_session_uuids": [],
            }
        }
        session_manager.create_session_configuration(session_uuid, initializer)
        return session_uuid

    def _populate_staging(
        self,
        plan: _ClonePlan,
        staging_path: str,
        *,
        create_initial_session: bool,
        initial_session_description: str,
    ) -> str | None:
        """Return the populate staging value owned by this model duplicator."""
        logical_target_path = str(plan.target_folder)
        source_path = str(plan.source_folder)
        EmptyModelGenerator._create_directories(staging_path)
        EmptyModelGenerator._write_model_manifest(
            staging_path,
            plan.target_model_name,
            plan.target_model_version,
            plan.source_description,
        )
        EmptyModelGenerator._write_model_configuration(
            staging_path,
            framework_uuid=plan.source_framework,
            logical_model_folder_path=logical_target_path,
        )
        EmptyModelGenerator._write_framework_runtime_scaffolds(
            staging_path,
            logical_model_folder_path=logical_target_path,
        )
        if plan.selected_relative_paths is None:
            self._copy_all_cloneable_content(source_path, staging_path)
        else:
            self._copy_selected_clone_content(
                source_path,
                staging_path,
                list(plan.selected_relative_paths),
            )
        self._targetize_framework_manifest_configurations(
            staging_path,
            logical_target_model_folder_path=logical_target_path,
        )

        target_cfg = Configuration()
        target_cfg.set_config_file_path(find_existing_model_manifest_path(staging_path))
        target_cfg.load()
        created_at = utc_now_iso()
        target_cfg.model = {
            "name": plan.target_model_name,
            "uuid": str(uuid.uuid4()),
            "version": plan.target_model_version,
            "description": plan.source_description,
            "created_at": created_at,
            "modified_at": created_at,
            "parent_model_uuid": plan.parent_model_uuid,
            "archived": False,
        }
        self._reset_generated_storage(staging_path)
        initial_session_uuid = (
            self._create_initial_session(
                staging_path=staging_path,
                target_model_name=plan.target_model_name,
                description=initial_session_description,
            )
            if create_initial_session
            else None
        )
        target_cfg.save()
        return initial_session_uuid

    def duplicate(
        self,
        source_model_folder_path: str,
        target_model_name: str,
        target_model_version: str | None = None,
        initial_session_description: str = "",
        create_initial_session: bool = False,
        selected_relative_paths: list[str] | None = None,
        *,
        models_root: str | None = None,
        failure_injector: FailureInjector | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> DuplicateResult:
        """Clone a root-confined model through staging and atomic publication."""

        resolved_models_root = os.path.abspath(
            str(models_root or os.path.dirname(os.path.abspath(source_model_folder_path)))
        )
        scope = ModelsStorageMutationScope.from_models_root(resolved_models_root)
        with scope.lock(timeout_seconds=lock_timeout_seconds):
            plan = self._prepare_clone(
                source_model_folder_path=source_model_folder_path,
                target_model_name=target_model_name,
                target_model_version=target_model_version,
                selected_relative_paths=selected_relative_paths,
                models_root=os.fspath(scope.models_root),
            )
            with StorageTransaction(
                plan.models_root,
                operation="clone-model",
                failure_injector=failure_injector,
            ) as transaction:
                staging_folder = transaction.stage_directory(plan.target_folder)
                initial_session_uuid = self._populate_staging(
                    plan,
                    str(staging_folder),
                    create_initial_session=create_initial_session,
                    initial_session_description=initial_session_description,
                )
                transaction.publish(staging_folder, plan.target_folder)
                transaction.commit()

            return self.DuplicateResult(
                target_model_name=plan.target_model_name,
                target_model_folder_path=str(plan.target_folder),
                parent_model_uuid=plan.parent_model_uuid,
                initial_session_uuid=initial_session_uuid,
                auto_load=True,
            )
