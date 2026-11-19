# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Expose selected-model evaluation libraries to application callers."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lanista.foundation.configuration import Configuration
from lanista.reinforcengine.rl_model.evaluation.contracts import LibraryEvaluationScope
from lanista.reinforcengine.rl_model.evaluation.model_paths import ModelEvaluationPaths
from lanista.reinforcengine.rl_model.storage.layout import (
    find_existing_model_manifest_path,
    get_framework_model_configuration_path,
    get_model_configuration_path,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    manifest_digest,
    validate_library_id,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.generation import (
    EvaluationLibraryGenerator,
    SourceFile,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.publication import (
    EVALUATION_LIBRARY_MANIFEST_FILENAME,
    read_verified_library,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryPathViolation,
    EvaluationLibraryStorageConfiguration,
)
from lanista.reinforcengine.utils.scenarios.route_entity_contracts import (
    RouteEntitiesProvider,
)

if TYPE_CHECKING:
    from lanista.reinforcengine.utils.scenarios.evaluation_library.generation import (
        GeneratedEvaluationLibrary,
        GroupedEvaluationLibraryParameters,
    )


@dataclass(frozen=True, slots=True)
class EvaluationLibraryCatalogItem:
    """Show one grouped library's verified identity or a diagnostic row."""

    library_id: str
    name: str
    created_at: str | None
    group_count: int
    scenario_count: int
    manifest_sha256: str | None
    valid: bool
    diagnostic: str | None = None


@dataclass(frozen=True, slots=True)
class ResolvedLibrarySelection:
    """Carry one reverified grouped library and its model-local run parent."""

    model_root: Path
    library_storage: EvaluationLibraryStorageConfiguration
    run_parent: Path
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _FrozenRouteEntitiesProvider:
    """Give every scenario the same route-entity lines observed at start."""

    lines: tuple[str, ...]

    def route_entity_lines(self) -> list[str]:
        return list(self.lines)


class _CheckedMaterializer:
    """Abort staged generation if its selected-model identity changes."""

    def __init__(self, materializer: object, check: Callable[[], None]) -> None:
        self._materializer = materializer
        self._check = check

    def materialize(self, request: object) -> object:
        self._check()
        result = self._materializer.materialize(request)
        self._check()
        return result


def _source(root: Path, configured: str | Path, kind: str) -> SourceFile:
    """Return a regular, root-confined configured generation source."""

    path = Path(configured)
    if ".." in path.parts:
        raise EvaluationLibraryPathViolation(f"Generation source traverses its model: {path!s}.")
    candidate = path if path.is_absolute() else root / path
    if (
        not candidate.is_relative_to(root)
        or candidate.resolve(strict=False) != candidate
        or not candidate.is_file()
    ):
        raise EvaluationLibraryPathViolation(
            f"Generation source is missing or outside the selected model: {candidate!s}."
        )
    return SourceFile(
        kind=kind,
        source_path=candidate,
        manifest_path=candidate.relative_to(root).as_posix(),
    )


def _configured_sources(
    configuration: Configuration,
    model_root: Path,
) -> tuple[object, SourceFile, SourceFile, SourceFile, tuple[SourceFile, ...]]:
    """Resolve the current model's actual generation settings and source files."""

    from lanista.reinforcengine.utils.scenarios.manager.scenario_manager_configuration_wrapper import (
        ScenarioManagerConfigurationWrapper,
    )

    scenario_file = _source(
        model_root,
        str(configuration.scenario_manager_configuration_file_path),
        "scenario_manager_configuration",
    )
    scenario_configuration = ScenarioManagerConfigurationWrapper()
    scenario_configuration.set_config_file_path(scenario_file.source_path)
    scenario_configuration.load()
    scenario_configuration.model_folder_path = str(model_root)

    network = _source(model_root, scenario_configuration.get_network_filename(), "network")
    # The scenario-manager YAML configures route and SUMO-file generation. Neither
    # output is a pre-existing template file in the current model resources.
    route_configuration = _source(
        model_root,
        scenario_file.source_path,
        "route_template",
    )
    sumo_configuration = _source(
        model_root,
        scenario_file.source_path,
        "sumo_configuration",
    )
    gui_settings = _source(
        model_root,
        scenario_configuration.get_gui_settings_filename(),
        "gui_settings",
    )
    model_manifest = _source(
        model_root,
        find_existing_model_manifest_path(str(model_root)),
        "model_manifest_configuration",
    )
    model_configuration = _source(
        model_root,
        get_model_configuration_path(str(model_root)),
        "model_configuration",
    )
    framework_uuid = str(getattr(configuration.rl_engine_framework, "uuid", "") or "")
    framework_configuration = _source(
        model_root,
        get_framework_model_configuration_path(str(model_root), framework_uuid),
        "framework_configuration",
    )
    return (
        scenario_configuration,
        network,
        route_configuration,
        sumo_configuration,
        (scenario_file, gui_settings, model_manifest, model_configuration, framework_configuration),
    )


class EvaluationLibraryApplicationService:
    """Resolve catalog, generation, selection, and report paths for the active model."""

    def __init__(
        self,
        *,
        active_configuration_provider: Callable[[], Configuration | None],
        models_root_provider: Callable[[], str],
        route_entities_provider: Callable[[], RouteEntitiesProvider | None],
        report_generator: Callable[[Path], object] | None = None,
    ) -> None:
        self._active_configuration_provider = active_configuration_provider
        self._models_root_provider = models_root_provider
        self._route_entities_provider = route_entities_provider
        self._report_generator = report_generator

    def _current(self) -> tuple[Configuration, ModelEvaluationPaths]:
        configuration = self._active_configuration_provider()
        if configuration is None:
            raise RuntimeError("No model is selected for evaluation libraries.")
        model_root = Path(str(configuration.model_folder_path))
        training_path = Path(str(configuration.scenarios_folder_path))
        training_root = (
            training_path if training_path.is_absolute() else model_root / training_path
        )
        paths = ModelEvaluationPaths.resolve(
            model_root=model_root,
            models_root=Path(self._models_root_provider()),
            training_root=training_root,
        )
        return configuration, paths

    def current_model_root(self) -> Path:
        """Expose the validated model identity for one GUI draft lifetime."""

        _, paths = self._current()
        return paths.model_root

    def list_for_current_model(self) -> tuple[EvaluationLibraryCatalogItem, ...]:
        """Read verified grouped libraries and retain corrupt rows for diagnosis."""

        _, paths = self._current()
        root = paths.library_storage.library_root
        if not root.exists():
            return ()
        paths.require_current_roots()
        if not root.is_dir():
            raise EvaluationLibraryPathViolation("Evaluation-library root is not a directory.")
        rows: list[EvaluationLibraryCatalogItem] = []
        for entry in sorted(root.iterdir(), key=lambda item: item.name):
            if entry.name.startswith(".") or (not entry.is_dir() and not entry.is_symlink()):
                continue
            try:
                validate_library_id(entry.name)
            except ValueError:
                continue
            marker = entry / EVALUATION_LIBRARY_MANIFEST_FILENAME
            if marker.is_file() and not marker.is_symlink():
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, ValueError):
                    payload = None
                if isinstance(payload, dict) and payload.get("schema_version") == 1:
                    continue
            try:
                manifest = read_verified_library(paths.library_storage, entry.name)
                if manifest.schema_version != 2:
                    continue
                rows.append(
                    EvaluationLibraryCatalogItem(
                        library_id=entry.name,
                        name=manifest.name,
                        created_at=manifest.created_at,
                        group_count=len(manifest.groups),
                        scenario_count=len(manifest.scenarios),
                        manifest_sha256=manifest_digest(manifest),
                        valid=True,
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                rows.append(
                    EvaluationLibraryCatalogItem(
                        library_id=entry.name,
                        name=entry.name,
                        created_at=None,
                        group_count=0,
                        scenario_count=0,
                        manifest_sha256=None,
                        valid=False,
                        diagnostic=str(exc),
                    )
                )
        return tuple(rows)

    def resolve_selection(self, scope: LibraryEvaluationScope) -> ResolvedLibrarySelection:
        """Verify a grouped selection again immediately before dispatch."""

        _, paths = self._current()
        manifest = read_verified_library(
            paths.library_storage,
            scope.library_id,
            expected_manifest_sha256=scope.manifest_sha256,
        )
        if manifest.schema_version != 2:
            raise ValueError("Only grouped evaluation libraries may be selected here.")
        paths.require_current_roots()
        return ResolvedLibrarySelection(
            model_root=paths.model_root,
            library_storage=paths.library_storage,
            run_parent=paths.run_parent,
            manifest_sha256=scope.manifest_sha256,
        )

    def generate_for_current_model(
        self,
        parameters: GroupedEvaluationLibraryParameters,
        *,
        expected_model_root: Path | None = None,
    ) -> GeneratedEvaluationLibrary:
        """Generate one grouped library from the current model's configured inputs."""

        from lanista.reinforcengine.utils.scenarios.evaluation_library.configured_materializer import (
            ConfiguredScenarioMaterializer,
        )
        from lanista.reinforcengine.utils.scenarios.evaluation_library.generation import (
            GroupedEvaluationLibraryGenerationRequest,
        )

        configuration, paths = self._current()
        if expected_model_root is not None and Path(expected_model_root) != paths.model_root:
            raise RuntimeError("Selected model changed before library generation.")
        provider = self._route_entities_provider()
        if provider is None:
            raise RuntimeError("Load the selected model before generating a library.")
        route_lines = tuple(provider.route_entity_lines())
        if not route_lines or any(not isinstance(line, str) for line in route_lines):
            raise ValueError("The model route-entities provider returned invalid lines.")
        frozen_provider = _FrozenRouteEntitiesProvider(route_lines)
        scenario_configuration, network, route_config, sumo_config, inputs = (
            _configured_sources(configuration, paths.model_root)
        )

        def require_same_model() -> None:
            _, current_paths = self._current()
            paths.require_current_roots()
            if current_paths.model_root != paths.model_root:
                raise RuntimeError("Selected model changed during library generation.")
            if tuple(provider.route_entity_lines()) != route_lines:
                raise RuntimeError("Route-entity source changed during library generation.")

        request = GroupedEvaluationLibraryGenerationRequest(
            parameters=parameters,
            network=network,
            route_template=route_config,
            sumo_configuration=sumo_config,
            generator_inputs=inputs,
            route_entity_lines=route_lines,
            pre_publish_check=require_same_model,
        )
        materializer = ConfiguredScenarioMaterializer(
            configuration=scenario_configuration,
            route_entities_provider=frozen_provider,
        )
        require_same_model()
        return EvaluationLibraryGenerator(paths.library_storage).generate_grouped(
            request,
            _CheckedMaterializer(materializer, require_same_model),
        )

    def publish_report(self, run_root: Path) -> None:
        """Publish a completed grouped run beneath the selected model only."""

        _, paths = self._current()
        paths.require_current_roots()
        candidate = Path(run_root)
        if (
            candidate.parent != paths.run_parent
            or candidate.is_symlink()
            or not candidate.is_dir()
        ):
            raise EvaluationLibraryPathViolation("Evaluation run is outside the selected model.")
        generator = self._report_generator
        if generator is None:
            from lanista.reinforcengine.rl_metrics.evaluation_report import (
                EvaluationReportService,
            )

            generator = EvaluationReportService().generate
        generator(candidate)


__all__ = [
    "EvaluationLibraryApplicationService",
    "EvaluationLibraryCatalogItem",
    "ResolvedLibrarySelection",
]
