# Copyright (c) 2025-2026 Benedek Szanyó.
# Copyright (c) 2025-2026 HUN-REN SZTAKI.
# All rights reserved.

"""Generate empty model scaffolds for a selected RL framework."""

import os
import re
import shutil
import uuid
from dataclasses import dataclass

from lanista.foundation.configuration import Configuration
from lanista.foundation.logging import LoggerInterface
from lanista.locations import get_model_templates_path, get_rl_model_storage_path
from lanista.reinforcengine.rl_model.metadata.configuration_schema import (
    ConfigurationSchema,
    FrameworkResourcePaths,
)
from lanista.reinforcengine.rl_model.metadata.timestamps import utc_now_iso
from lanista.reinforcengine.rl_model.runtime.plugin_namespace import ModelPluginNamespace
from lanista.reinforcengine.rl_model.storage.layout import (
    MODEL_CONFIGURATION_FILE_NAME,
    MODEL_MANIFEST_FILE_NAME,
    RAY_RESOURCES_DIR_NAME,
    RESOURCES_DIR_NAME,
    SB3_RESOURCES_DIR_NAME,
    TENSORBOARD_COMPARISON_DIR_NAME,
    TENSORBOARD_DIR_NAME,
    TENSORBOARD_EVALUATION_DIR_NAME,
    TENSORBOARD_RAY_DIR_NAME,
    TENSORBOARD_SB3_DIR_NAME,
    TENSORBOARD_TRAININGS_DIR_NAME,
    TRAINING_SCENARIOS_DIR_NAME,
    TRAINING_SESSIONS_DIR_NAME,
)
from lanista.reinforcengine.rl_model.storage.mutation_scope import ModelsStorageMutationScope
from lanista.reinforcengine.rl_model.storage.names import ModelStorageName
from lanista.reinforcengine.rl_model.storage.path_guard import StoragePathGuard
from lanista.reinforcengine.rl_model.storage.persistence import atomic_write_configuration_file
from lanista.reinforcengine.rl_model.storage.transaction import FailureInjector, StorageTransaction


class EmptyModelGenerator(LoggerInterface):
    """Generate empty model artifacts."""

    @dataclass
    class GenerateResult:
        """Report the result of generate."""

        target_model_name: str
        target_model_folder_path: str
        auto_load: bool = True

    _BASE_MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*_project$")
    _MODEL_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9]*_project$")
    _COMPOSITE_VERSION_PATTERN = re.compile(r"^v?([0-9A-Za-z-]+)\.([0-9]+)\.([0-9]+)$")
    _MANUAL_MAJOR_PATTERN = re.compile(r"^[0-9A-Za-z-]+$")

    @classmethod
    def parse_composite_version(cls, version_text: str) -> tuple[str, int, int] | None:
        """Parse the composite version owned by this empty model generator."""
        text = str(version_text or "").strip()
        match = cls._COMPOSITE_VERSION_PATTERN.fullmatch(text)
        if not match:
            return None
        return match.group(1), int(match.group(2)), int(match.group(3))

    @classmethod
    def is_valid_manual_major(cls, version_text: str) -> bool:
        """Return whether valid manual major applies to this empty model generator."""
        return bool(cls._MANUAL_MAJOR_PATTERN.fullmatch(str(version_text or "").strip()))

    @staticmethod
    def build_composite_version(manual_major: str, model_number: int, session_number: int) -> str:
        """Build the composite version required by this empty model generator."""
        return f"v{str(manual_major or '').strip()}.{int(model_number)}.{int(session_number)}"

    @staticmethod
    def shorten_uuid(value: str, limit: int = 8) -> str:
        """Return the shorten uuid value owned by this empty model generator."""
        text = str(value or "").strip()
        return text[:limit] if text else ""

    @classmethod
    def suggest_default_version(cls, models_root: str, manual_major: str = "1") -> str:
        """Return the suggest default version value owned by this empty model generator."""
        next_model_number = cls._get_next_generated_model_number(models_root)
        return cls.build_composite_version(manual_major, next_model_number, 0)

    @classmethod
    def _normalize_base_model_name(cls, model_name: str) -> str:
        """Normalize the base model name accepted by this empty model generator."""
        normalized = str(model_name or "").strip()
        if not normalized:
            return ""

        project_index = normalized.find("_project")
        if project_index >= 0:
            normalized = normalized[: project_index + len("_project")]
        elif not normalized.endswith("_project"):
            normalized = f"{normalized}_project"

        return normalized

    @classmethod
    def _get_next_generated_model_number(cls, models_root: str) -> int:
        """Return the next generated model number exposed by this empty model generator."""
        max_model_number = 0
        if os.path.isdir(models_root):
            for entry in os.listdir(models_root):
                manifest_path = os.path.join(models_root, str(entry), MODEL_MANIFEST_FILE_NAME)
                parsed = None
                if os.path.isfile(manifest_path):
                    cfg = Configuration()
                    cfg.set_config_file_path(manifest_path)
                    try:
                        cfg.load()
                        model_section = getattr(cfg, "model", None)
                        parsed = cls.parse_composite_version(
                            str(getattr(model_section, "version", "") or "")
                        )
                    except (OSError, AttributeError, TypeError, ValueError):
                        parsed = None
                if parsed is None:
                    parsed = cls.parse_composite_version("_".join(str(entry).split("_")[-3:]))
                if parsed is None:
                    continue
                _, model_number, _ = parsed
                max_model_number = max(max_model_number, model_number)
        return max_model_number + 1

    @classmethod
    def resolve_model_version(
        cls,
        version_text: str,
        *,
        existing_version: str = "",
        models_root: str = "",
        default_session_number: int = 0,
    ) -> str:
        """Resolve the model version required by this empty model generator."""
        raw_version = str(version_text or "").strip()
        if not raw_version:
            return ""

        parsed_raw = cls.parse_composite_version(raw_version)
        if parsed_raw is not None:
            return raw_version

        if not cls.is_valid_manual_major(raw_version):
            return ""

        parsed_existing = cls.parse_composite_version(existing_version)
        if parsed_existing is not None:
            _, existing_model_number, existing_session_number = parsed_existing
            return cls.build_composite_version(
                raw_version, existing_model_number, existing_session_number
            )

        next_model_number = cls._get_next_generated_model_number(models_root) if models_root else 1
        return cls.build_composite_version(raw_version, next_model_number, default_session_number)

    @staticmethod
    def normalize_model_name(model_name: str, model_version: str = "") -> str:
        """Normalize the model name accepted by this empty model generator."""
        return EmptyModelGenerator._normalize_base_model_name(model_name)

    @classmethod
    def get_model_folder_name(cls, model_name: str) -> str:
        """Return the model folder name exposed by this empty model generator."""
        return cls._normalize_base_model_name(model_name)

    @staticmethod
    def build_model_display_name(model_name: str, model_version: str = "") -> str:
        """Build the model display name required by this empty model generator."""
        base_name = EmptyModelGenerator._normalize_base_model_name(model_name)
        version = str(model_version or "").strip()
        if base_name and version:
            return f"{base_name} [{version}]"
        return base_name or version

    def generate(
        self,
        model_name: str,
        model_version: str,
        model_description: str = "",
        framework_uuid: str = "ray",
        *,
        models_root: str | None = None,
        failure_injector: FailureInjector | None = None,
        lock_timeout_seconds: float = 5.0,
    ) -> GenerateResult:
        """Build a model in transaction-owned staging and publish it atomically."""

        resolved_models_root = os.path.abspath(
            str(models_root or os.path.join(get_rl_model_storage_path(), "models"))
        )
        os.makedirs(resolved_models_root, exist_ok=True)
        scope = ModelsStorageMutationScope.from_models_root(resolved_models_root)
        with scope.lock(timeout_seconds=lock_timeout_seconds):
            return self._generate_locked(
                model_name=model_name,
                model_version=model_version,
                model_description=model_description,
                framework_uuid=framework_uuid,
                models_root=resolved_models_root,
                failure_injector=failure_injector,
            )

    def _generate_locked(
        self,
        *,
        model_name: str,
        model_version: str,
        model_description: str,
        framework_uuid: str,
        models_root: str,
        failure_injector: FailureInjector | None,
    ) -> GenerateResult:
        """Resolve uniqueness and commit while the shared models lock is held."""

        model_version = self.resolve_model_version(
            model_version,
            models_root=models_root,
            default_session_number=0,
        )
        model_name = self.normalize_model_name(model_name, model_version)
        model_description = str(model_description or "").strip()

        if not model_name:
            raise ValueError("Model name is required.")
        if not self._MODEL_NAME_PATTERN.fullmatch(model_name):
            raise ValueError("Model name must resolve to a unique '<name>_project' value.")
        if not model_version:
            raise ValueError("Model version is required.")

        guard = StoragePathGuard(models_root)
        model_storage_name = ModelStorageName.parse(self.get_model_folder_name(model_name))
        target_model_folder = guard.resolve(model_storage_name)
        if os.path.lexists(target_model_folder):
            raise FileExistsError(f"Model '{model_name}' already exists.")

        with StorageTransaction(
            models_root,
            operation="create-model",
            failure_injector=failure_injector,
        ) as transaction:
            staging_folder = transaction.stage_directory(target_model_folder)
            staging_path = str(staging_folder)
            logical_target_path = str(target_model_folder)

            self._create_directories(staging_path)
            self._write_model_manifest(
                staging_path,
                model_name,
                model_version,
                model_description,
            )
            self._write_model_configuration(
                staging_path,
                framework_uuid=framework_uuid,
                logical_model_folder_path=logical_target_path,
            )
            self._write_framework_runtime_scaffolds(
                staging_path,
                logical_model_folder_path=logical_target_path,
            )
            transaction.publish(staging_folder, target_model_folder)
            transaction.commit()

        return self.GenerateResult(
            target_model_name=model_name,
            target_model_folder_path=str(target_model_folder),
            auto_load=True,
        )

    @staticmethod
    def _create_directories(model_folder_path: str) -> None:
        """Create the directories owned by this empty model generator."""
        os.makedirs(
            os.path.join(model_folder_path, RESOURCES_DIR_NAME, TRAINING_SCENARIOS_DIR_NAME),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(model_folder_path, RESOURCES_DIR_NAME, RAY_RESOURCES_DIR_NAME),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(model_folder_path, RESOURCES_DIR_NAME, SB3_RESOURCES_DIR_NAME),
            exist_ok=True,
        )
        os.makedirs(os.path.join(model_folder_path, TRAINING_SESSIONS_DIR_NAME), exist_ok=True)
        os.makedirs(
            os.path.join(
                model_folder_path,
                TENSORBOARD_DIR_NAME,
                TENSORBOARD_TRAININGS_DIR_NAME,
                TENSORBOARD_RAY_DIR_NAME,
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                model_folder_path,
                TENSORBOARD_DIR_NAME,
                TENSORBOARD_TRAININGS_DIR_NAME,
                TENSORBOARD_SB3_DIR_NAME,
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                model_folder_path,
                TENSORBOARD_DIR_NAME,
                TENSORBOARD_EVALUATION_DIR_NAME,
                TENSORBOARD_RAY_DIR_NAME,
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                model_folder_path,
                TENSORBOARD_DIR_NAME,
                TENSORBOARD_EVALUATION_DIR_NAME,
                TENSORBOARD_SB3_DIR_NAME,
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(model_folder_path, TENSORBOARD_DIR_NAME, TENSORBOARD_COMPARISON_DIR_NAME),
            exist_ok=True,
        )
        for init_dir in (
            model_folder_path,
            os.path.join(model_folder_path, RESOURCES_DIR_NAME),
            os.path.join(model_folder_path, RESOURCES_DIR_NAME, RAY_RESOURCES_DIR_NAME),
            os.path.join(model_folder_path, RESOURCES_DIR_NAME, SB3_RESOURCES_DIR_NAME),
        ):
            init_path = os.path.join(init_dir, "__init__.py")
            if not os.path.isfile(init_path):
                with open(init_path, "w", encoding="utf-8") as handle:
                    handle.write("")

    @staticmethod
    def _write_model_manifest(
        model_folder_path: str, model_name: str, model_version: str, model_description: str
    ) -> None:
        """Persist the model manifest owned by this empty model generator."""
        timestamp = utc_now_iso()
        cfg = Configuration()
        cfg.set_config_file_path(os.path.join(model_folder_path, MODEL_MANIFEST_FILE_NAME))
        cfg.schema_version = 1
        cfg.model = {
            "name": model_name,
            "uuid": str(uuid.uuid4()),
            "version": model_version,
            "description": model_description,
            "created_at": timestamp,
            "modified_at": timestamp,
            "parent_model_uuid": None,
            "archived": False,
        }
        atomic_write_configuration_file(
            cfg, os.path.join(model_folder_path, MODEL_MANIFEST_FILE_NAME)
        )

    @staticmethod
    def _write_model_configuration(
        model_folder_path: str,
        framework_uuid: str = "ray",
        *,
        logical_model_folder_path: str | None = None,
    ) -> None:
        """Persist the model configuration owned by this empty model generator."""
        normalized_framework_uuid = str(framework_uuid or "ray").strip().lower()
        if normalized_framework_uuid not in ("ray", "sb3"):
            normalized_framework_uuid = "ray"
        schema = ConfigurationSchema(
            preferred_framework_uuid=normalized_framework_uuid,
            active_runtime_framework_uuid=normalized_framework_uuid,
            selected_default_model_name="",
            selected_default_checkpoint_name="",
            framework_resource_paths=FrameworkResourcePaths(
                ray=f"{RESOURCES_DIR_NAME}/{RAY_RESOURCES_DIR_NAME}",
                sb3=f"{RESOURCES_DIR_NAME}/{SB3_RESOURCES_DIR_NAME}",
            ),
        )
        cfg = Configuration()
        cfg.set_config_file_path(os.path.join(model_folder_path, MODEL_CONFIGURATION_FILE_NAME))
        payload = schema.to_model_configuration_payload(
            model_folder_path=logical_model_folder_path or model_folder_path,
        )
        cfg.schema_version = payload["schema_version"]
        cfg.model_configuration = payload["model_configuration"]
        atomic_write_configuration_file(
            cfg, os.path.join(model_folder_path, MODEL_CONFIGURATION_FILE_NAME)
        )

    @staticmethod
    def _write_framework_runtime_scaffolds(
        model_folder_path: str,
        *,
        logical_model_folder_path: str | None = None,
    ) -> None:
        """Persist the framework runtime scaffolds owned by this empty model generator."""
        EmptyModelGenerator._materialize_framework_template(
            model_folder_path=model_folder_path,
            framework_uuid="ray",
            object_name="TemplateModel",
            running_profile="single",
            logical_model_folder_path=logical_model_folder_path,
        )
        EmptyModelGenerator._materialize_framework_template(
            model_folder_path=model_folder_path,
            framework_uuid="sb3",
            object_name="TemplateModel",
            running_profile="single",
            logical_model_folder_path=logical_model_folder_path,
        )

    @staticmethod
    def _build_framework_entry_module_path(model_folder_path: str, framework_uuid: str) -> str:
        """Build the framework entry module path required by this empty model generator."""
        model_folder_name = os.path.basename(str(model_folder_path).rstrip(os.sep))
        return ModelPluginNamespace.create(
            model_package=model_folder_name,
            framework_uuid=framework_uuid,
        ).module("rl_model")

    @staticmethod
    def _get_framework_template_dir(framework_uuid: str) -> str:
        """Return the framework template dir exposed by this empty model generator."""
        return os.path.join(
            get_model_templates_path(),
            str(framework_uuid or "").strip().lower(),
        )

    @staticmethod
    def _render_template_text(template_text: str, replacements: dict[str, str]) -> str:
        """Return the render template text value owned by this empty model generator."""
        rendered = str(template_text)
        for key, value in replacements.items():
            rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
        return rendered

    @classmethod
    def _materialize_framework_template(
        cls,
        model_folder_path: str,
        framework_uuid: str,
        object_name: str,
        running_profile: str,
        *,
        logical_model_folder_path: str | None = None,
    ) -> None:
        """Materialize the framework template owned by this empty model
generator."""
        framework_dir_name = (
            RAY_RESOURCES_DIR_NAME if framework_uuid == "ray" else SB3_RESOURCES_DIR_NAME
        )
        framework_dir_path = os.path.join(model_folder_path, RESOURCES_DIR_NAME, framework_dir_name)
        template_dir = cls._get_framework_template_dir(framework_uuid)
        if not os.path.isdir(template_dir):
            raise FileNotFoundError(f"Framework template is missing: {template_dir}")

        replacements = {
            "FRAMEWORK_UUID": framework_uuid,
            "MODEL_MODULE_PATH": cls._build_framework_entry_module_path(
                logical_model_folder_path or model_folder_path,
                framework_uuid,
            ),
            "MODEL_OBJECT": object_name,
            "RUNNING_PROFILE": running_profile,
            "SCENARIOS_FOLDER_PATH": f"{RESOURCES_DIR_NAME}/{TRAINING_SCENARIOS_DIR_NAME}",
            "SCENARIO_CONFIG_PATH": f"{RESOURCES_DIR_NAME}/{framework_dir_name}/scenario_manager_configuration.yaml",
        }

        for entry_name in os.listdir(template_dir):
            source_path = os.path.join(template_dir, entry_name)
            target_path = os.path.join(framework_dir_path, entry_name)
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
                continue
            with open(source_path, encoding="utf-8") as handle:
                template_text = handle.read()
            rendered = cls._render_template_text(template_text, replacements)
            with open(target_path, "w", encoding="utf-8") as handle:
                handle.write(rendered)
