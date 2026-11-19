# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Build validated, immutable targets for application-owned inspection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from lanista.reinforcengine.application_errors import ApplicationValidationError
from lanista.reinforcengine.rl_model.model import RLModelRunningModes
from lanista.reinforcengine.rl_model.runtime.tensorboard_layout_service import (
    TensorboardApplicationLayoutRequest,
    TensorboardLayoutMode,
    TensorboardRuntimeLayoutService,
)
from lanista.reinforcengine.rl_model.storage.path_guard import (
    StoragePathGuard,
    StoragePathViolation,
)


class InspectionTargetKind(Enum):
    """Distinguish an ordinary catalog target from an explicit pack target."""

    ORDINARY = "ordinary"
    PACK = "pack"


@dataclass(frozen=True, slots=True)
class InspectionTarget:
    """Carry one fully validated selection into the process-lifecycle owner."""

    kind: InspectionTargetKind
    model_uuid: str
    model_root: str
    framework_uuid: str
    mode: RLModelRunningModes
    native_logdir: str
    selected_session: str | None = None
    selected_checkpoint: str | None = None
    pack_type: str | None = None
    pack_name: str | None = None

    @property
    def catalog_pack_type(self) -> str:
        """Return the dashboard pack-type filter for this target."""

        if self.kind is InspectionTargetKind.PACK:
            return "all"
        return "evaluation" if self.mode is RLModelRunningModes.EVALUATION else "training"

    @property
    def display_name(self) -> str:
        """Return a compact status label without exposing absolute paths."""

        if self.kind is InspectionTargetKind.PACK:
            return f"{self.pack_type or 'pack'} / {self.pack_name or 'selection'}"
        focus = self.selected_session or "catalog"
        return f"{self.mode.value} / {self.framework_uuid} / {focus}"


class InspectionTargetService:
    """Normalize GUI selections without accepting unguarded process paths."""

    _FRAMEWORKS = frozenset({"ray", "sb3"})
    _PACK_TYPES = frozenset({"training", "evaluation", "comparison"})

    def __init__(self, *, layout_service: TensorboardRuntimeLayoutService | None = None) -> None:
        """Initialize the target service with the canonical layout resolver."""

        self._layout_service = layout_service or TensorboardRuntimeLayoutService()

    def ordinary(
        self,
        *,
        configuration: object,
        mode: RLModelRunningModes,
        framework_uuid: str,
        selected_session: str | None,
        selected_checkpoint: str | None,
    ) -> InspectionTarget:
        """Build an ordinary model/framework/mode catalog target."""

        model_uuid, model_root = self._validated_model(configuration)
        framework = self._validated_framework(framework_uuid)
        layout_mode = self._layout_mode(mode)
        native_logdir = self._layout_service.resolve_application_log_dir(
            TensorboardApplicationLayoutRequest(
                model_folder_path=model_root,
                selected_session=self._clean_optional(selected_session),
                framework_uuid=framework,
                mode=layout_mode,
            )
        )
        if not native_logdir or not Path(native_logdir).is_dir():
            raise ApplicationValidationError(
                "Inspection requires an existing TensorBoard log directory for the selected scope."
            )
        guarded_logdir = self._validated_descendant(model_root, native_logdir)
        return InspectionTarget(
            kind=InspectionTargetKind.ORDINARY,
            model_uuid=model_uuid,
            model_root=model_root,
            framework_uuid=framework,
            mode=mode,
            native_logdir=guarded_logdir,
            selected_session=self._clean_optional(selected_session),
            selected_checkpoint=self._clean_optional(selected_checkpoint),
        )

    def pack(
        self,
        *,
        configuration: object,
        pack_type: str,
        framework_uuid: str,
        pack_name: str,
        logdir: str,
    ) -> InspectionTarget:
        """Build an explicit pack target after root and type validation."""

        model_uuid, model_root = self._validated_model(configuration)
        normalized_type = str(pack_type or "").strip().lower()
        if normalized_type not in self._PACK_TYPES:
            raise ApplicationValidationError("Inspection requires a recognized TensorBoard pack type.")
        framework = str(framework_uuid or "").strip().lower()
        supported_frameworks = self._FRAMEWORKS | ({"mixed"} if normalized_type == "comparison" else set())
        if framework not in supported_frameworks:
            raise ApplicationValidationError("Inspection requires a valid framework selection.")
        normalized_name = str(pack_name or "").strip()
        if not normalized_name:
            raise ApplicationValidationError("Inspection requires a named TensorBoard pack.")
        guarded_logdir = self._validated_descendant(model_root, logdir, must_exist=True)
        mode = (
            RLModelRunningModes.EVALUATION
            if normalized_type == "evaluation"
            else RLModelRunningModes.TRAINING
        )
        return InspectionTarget(
            kind=InspectionTargetKind.PACK,
            model_uuid=model_uuid,
            model_root=model_root,
            framework_uuid=framework,
            mode=mode,
            native_logdir=guarded_logdir,
            pack_type=normalized_type,
            pack_name=normalized_name,
        )

    @classmethod
    def _validated_model(cls, configuration: object) -> tuple[str, str]:
        """Return the validated model identity and canonical storage root."""

        model_uuid = str(getattr(configuration, "model_uuid", "") or "").strip()
        raw_root = str(getattr(configuration, "model_folder_path", "") or "").strip()
        if not model_uuid or not raw_root:
            raise ApplicationValidationError(
                "Inspection requires a validated model identity and storage root."
            )
        declared_root = Path(raw_root).expanduser().absolute()
        try:
            guard = StoragePathGuard(declared_root)
        except StoragePathViolation as error:
            raise ApplicationValidationError("The selected model storage root is unsafe.") from error
        if (
            declared_root != guard.root
            or declared_root.is_symlink()
            or not declared_root.is_dir()
        ):
            raise ApplicationValidationError("The selected model storage root is unavailable or aliased.")
        return model_uuid, str(declared_root)

    @classmethod
    def _validated_framework(cls, framework_uuid: str) -> str:
        """Return one supported framework identifier."""

        framework = str(framework_uuid or "").strip().lower()
        if framework not in cls._FRAMEWORKS:
            raise ApplicationValidationError("Inspection requires a valid framework selection.")
        return framework

    @staticmethod
    def _layout_mode(mode: RLModelRunningModes) -> TensorboardLayoutMode:
        """Map executable modes onto persisted TensorBoard layout modes."""

        if mode is RLModelRunningModes.TRAINING:
            return TensorboardLayoutMode.TRAINING
        if mode is RLModelRunningModes.EVALUATION:
            return TensorboardLayoutMode.EVALUATION
        raise ApplicationValidationError("Inference does not produce an inspection target.")

    @staticmethod
    def _clean_optional(value: str | None) -> str | None:
        """Normalize an optional user selection."""

        return str(value or "").strip() or None

    @staticmethod
    def _validated_descendant(
        model_root: str,
        candidate: str,
        *,
        must_exist: bool = True,
    ) -> str:
        """Return one canonical model-root descendant directory."""

        try:
            path = StoragePathGuard(model_root).require_descendant(
                str(candidate or "").strip(),
                must_exist=must_exist,
            )
        except (FileNotFoundError, StoragePathViolation, ValueError) as error:
            raise ApplicationValidationError(
                "The selected TensorBoard log directory is unavailable or unsafe."
            ) from error
        if path.is_symlink() or not path.is_dir():
            raise ApplicationValidationError(
                "The selected TensorBoard log directory is unavailable or aliased."
            )
        return str(path)


__all__ = [
    "InspectionTarget",
    "InspectionTargetKind",
    "InspectionTargetService",
]
