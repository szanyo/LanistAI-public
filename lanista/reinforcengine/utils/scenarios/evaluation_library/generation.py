# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Materialise and atomically publish one deterministic evaluation library."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    EVALUATION_LIBRARY_SCHEMA_VERSION,
    GROUPED_EVALUATION_LIBRARY_SCHEMA_VERSION,
    AnyEvaluationLibraryManifest,
    EvaluationGroupDefinition,
    EvaluationLibraryManifest,
    EvaluationScenarioEntry,
    GroupedEvaluationLibraryManifest,
    GroupedEvaluationScenarioEntry,
    FileDigest,
    SourceIdentity,
    manifest_digest,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.planning import (
    SEED_DERIVATION_VERSION,
    GROUPED_SEED_DERIVATION_VERSION,
    PlannedEvaluationScenario,
    PlannedGroupedScenario,
    build_generation_plan,
    build_grouped_generation_plan,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.publication import (
    EvaluationLibraryIntegrityError,
    EvaluationLibraryPublication,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import (
    EvaluationLibraryStorageConfiguration,
)


@dataclass(frozen=True, slots=True)
class SourceFile:
    """Bind one generation input to its stable manifest path."""

    kind: str
    source_path: Path
    manifest_path: str


@dataclass(frozen=True, slots=True)
class EvaluationLibraryGenerationRequest:
    """Carry all frozen inputs required to generate one library."""

    library_id: str
    created_at: str
    generator_version: str
    purpose: str
    vehicle_levels: tuple[int, ...]
    pedestrian_levels: tuple[int, ...]
    replicate_count: int
    master_seed: int
    network: SourceFile
    route_template: SourceFile
    sumo_configuration: SourceFile
    generator_inputs: tuple[SourceFile, ...]


@dataclass(frozen=True, slots=True)
class GroupedEvaluationLibraryParameters:
    """Describe a grouped library without exposing filesystem sources to the GUI."""

    library_id: str
    name: str
    description: str | None
    created_at: str
    groups: tuple[EvaluationGroupDefinition, ...]
    master_seed: int
    generator_version: str = "lanistai-grouped-v2"
    purpose: str = "held-out evaluation"


@dataclass(frozen=True, slots=True)
class GroupedEvaluationLibraryGenerationRequest:
    """Bind grouped parameters to one frozen model-wide generation input set."""

    parameters: GroupedEvaluationLibraryParameters
    network: SourceFile
    route_template: SourceFile
    sumo_configuration: SourceFile
    generator_inputs: tuple[SourceFile, ...]
    route_entity_lines: tuple[str, ...]
    pre_publish_check: Callable[[], None] | None = None

    @property
    def created_at(self) -> str:
        return self.parameters.created_at


@dataclass(frozen=True, slots=True)
class ScenarioMaterializationRequest:
    """Describe one planned scenario and its confined output directory."""

    scenario_id: str
    output_directory: Path
    requested_vehicle_count: int
    requested_pedestrian_count: int
    replicate_index: int
    seed: int
    library_created_at: str
    network_path: Path
    route_template_path: Path
    sumo_configuration_path: Path
    generator_input_paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class MaterializedScenario:
    """Report measured populations after materialisation."""

    realised_vehicle_count: int
    realised_pedestrian_count: int
    realisation_note: str | None = None


class ScenarioMaterializer(Protocol):
    """Materialise one scenario below the directory supplied by the generator."""

    def materialize(self, request: ScenarioMaterializationRequest) -> MaterializedScenario:
        """Create one complete scenario and return measured populations."""
        ...


@dataclass(frozen=True, slots=True)
class GeneratedEvaluationLibrary:
    """Identify one successfully published immutable library."""

    root: Path
    manifest: AnyEvaluationLibraryManifest
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _GenerationSources:
    files: tuple[SourceFile, ...]
    identities: tuple[SourceIdentity, ...]
    network: SourceIdentity
    route_template: SourceIdentity
    sumo_configuration: SourceIdentity
    generator_inputs: tuple[SourceIdentity, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(source: SourceFile) -> SourceIdentity:
    path = Path(source.source_path)
    if path.is_symlink() or not path.is_file():
        raise EvaluationLibraryIntegrityError(f"Generation source is missing or unsafe: {path!s}.")
    return SourceIdentity(
        kind=source.kind,
        path=source.manifest_path,
        sha256=_sha256_file(path),
    )


def _scenario_file_digests(root: Path) -> tuple[FileDigest, ...]:
    files: list[FileDigest] = []
    for current_root, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(current_root)
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                raise EvaluationLibraryIntegrityError(
                    "Scenario materialisation must not contain symbolic links."
                )
        for file_name in file_names:
            path = current / file_name
            if path.is_symlink() or not path.is_file():
                raise EvaluationLibraryIntegrityError(
                    "Scenario materialisation may contain only regular files."
                )
            files.append(
                FileDigest(
                    relative_path=path.relative_to(root).as_posix(),
                    byte_size=path.stat().st_size,
                    sha256=_sha256_file(path),
                )
            )
    if not files:
        raise EvaluationLibraryIntegrityError(
            "A materialised scenario must contain at least one regular file."
        )
    return tuple(sorted(files, key=lambda item: item.relative_path))


def _validated_generation_sources(
    request: EvaluationLibraryGenerationRequest | GroupedEvaluationLibraryGenerationRequest,
) -> _GenerationSources:
    required_kinds = (
        (request.network, "network"),
        (request.route_template, "route_template"),
        (request.sumo_configuration, "sumo_configuration"),
    )
    for source, expected_kind in required_kinds:
        if source.kind != expected_kind:
            raise ValueError(f"Expected {expected_kind!r} source, received {source.kind!r}.")
    files = (
        request.network,
        request.route_template,
        request.sumo_configuration,
        *request.generator_inputs,
    )
    identities = tuple(_source_identity(source) for source in files)
    network, route_template, sumo_configuration, *generator_inputs = identities
    ordered_inputs = tuple(sorted(generator_inputs, key=lambda item: (item.kind, item.path)))
    generator_input_keys = tuple((item.kind, item.path) for item in ordered_inputs)
    if len(generator_input_keys) != len(set(generator_input_keys)):
        raise ValueError("Generator input identities must be unique.")
    return _GenerationSources(
        files=files,
        identities=identities,
        network=network,
        route_template=route_template,
        sumo_configuration=sumo_configuration,
        generator_inputs=ordered_inputs,
    )


def _materialize_scenarios(
    *,
    staging_root: Path,
    plan: tuple[PlannedEvaluationScenario, ...] | tuple[PlannedGroupedScenario, ...],
    request: EvaluationLibraryGenerationRequest | GroupedEvaluationLibraryGenerationRequest,
    materializer: ScenarioMaterializer,
) -> tuple[EvaluationScenarioEntry, ...]:
    scenarios_root = staging_root / "scenarios"
    scenarios_root.mkdir()
    entries: list[EvaluationScenarioEntry] = []
    for planned in plan:
        relative_path = f"scenarios/{planned.scenario_id}"
        output_directory = staging_root / relative_path
        output_directory.mkdir()
        materialized = materializer.materialize(
            ScenarioMaterializationRequest(
                scenario_id=planned.scenario_id,
                output_directory=output_directory,
                requested_vehicle_count=planned.requested_vehicle_count,
                requested_pedestrian_count=planned.requested_pedestrian_count,
                replicate_index=planned.replicate_index,
                seed=planned.seed,
                library_created_at=request.created_at,
                network_path=request.network.source_path,
                route_template_path=request.route_template.source_path,
                sumo_configuration_path=request.sumo_configuration.source_path,
                generator_input_paths=tuple(
                    source.source_path for source in request.generator_inputs
                ),
            )
        )
        entry = EvaluationScenarioEntry(
            scenario_id=planned.scenario_id,
            relative_path=relative_path,
            requested_vehicle_count=planned.requested_vehicle_count,
            requested_pedestrian_count=planned.requested_pedestrian_count,
            replicate_index=planned.replicate_index,
            seed=planned.seed,
            realised_vehicle_count=materialized.realised_vehicle_count,
            realised_pedestrian_count=materialized.realised_pedestrian_count,
            realisation_note=materialized.realisation_note,
            files=_scenario_file_digests(output_directory),
        )
        if isinstance(planned, PlannedGroupedScenario):
            entries.append(
                GroupedEvaluationScenarioEntry.model_validate(
                    {
                        **entry.model_dump(),
                        "group_id": planned.group_id,
                        "group_scenario_index": planned.group_scenario_index,
                    }
                )
            )
        else:
            entries.append(entry)
    return tuple(entries)


class EvaluationLibraryGenerator:
    """Coordinate deterministic generation without touching training storage."""

    def __init__(self, storage: EvaluationLibraryStorageConfiguration) -> None:
        self._storage = storage

    def generate(
        self,
        request: EvaluationLibraryGenerationRequest,
        materializer: ScenarioMaterializer,
    ) -> GeneratedEvaluationLibrary:
        """Generate every planned scenario and atomically publish the complete library."""

        sources = _validated_generation_sources(request)
        plan = build_generation_plan(
            vehicle_levels=request.vehicle_levels,
            pedestrian_levels=request.pedestrian_levels,
            replicate_count=request.replicate_count,
            master_seed=request.master_seed,
        )
        with EvaluationLibraryPublication(self._storage, request.library_id) as publication:
            scenario_entries = _materialize_scenarios(
                staging_root=publication.staging_root,
                plan=plan,
                request=request,
                materializer=materializer,
            )
            current_identities = tuple(_source_identity(source) for source in sources.files)
            if current_identities != sources.identities:
                raise RuntimeError("A generation source changed during materialisation.")
            manifest = EvaluationLibraryManifest(
                schema_version=EVALUATION_LIBRARY_SCHEMA_VERSION,
                library_id=request.library_id,
                created_at=request.created_at,
                generator_version=request.generator_version,
                purpose=request.purpose,
                master_seed=request.master_seed,
                seed_derivation_version=SEED_DERIVATION_VERSION,
                network=sources.network,
                route_template=sources.route_template,
                sumo_configuration=sources.sumo_configuration,
                generator_inputs=sources.generator_inputs,
                expected_scenario_count=len(plan),
                complete=True,
                scenarios=tuple(scenario_entries),
            )
            root = publication.publish(manifest)

        return GeneratedEvaluationLibrary(
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_digest(manifest),
        )


    def generate_grouped(
        self,
        request: GroupedEvaluationLibraryGenerationRequest,
        materializer: ScenarioMaterializer,
    ) -> GeneratedEvaluationLibrary:
        """Generate ordered named groups and publish one complete v2 library."""

        parameters = request.parameters
        if not request.route_entity_lines or any(
            not isinstance(line, str) or not line.strip() for line in request.route_entity_lines
        ):
            raise ValueError("Grouped generation requires frozen route-entity lines.")
        sources = _validated_generation_sources(request)
        plan = build_grouped_generation_plan(
            groups=parameters.groups,
            master_seed=parameters.master_seed,
        )
        with EvaluationLibraryPublication(self._storage, parameters.library_id) as publication:
            scenario_entries = _materialize_scenarios(
                staging_root=publication.staging_root,
                plan=plan,
                request=request,
                materializer=materializer,
            )
            if request.pre_publish_check is not None:
                request.pre_publish_check()
            if tuple(_source_identity(source) for source in sources.files) != sources.identities:
                raise RuntimeError("A generation source changed during materialisation.")
            manifest = GroupedEvaluationLibraryManifest(
                schema_version=GROUPED_EVALUATION_LIBRARY_SCHEMA_VERSION,
                library_id=parameters.library_id,
                name=parameters.name,
                description=parameters.description,
                created_at=parameters.created_at,
                generator_version=parameters.generator_version,
                purpose=parameters.purpose,
                master_seed=parameters.master_seed,
                seed_derivation_version=GROUPED_SEED_DERIVATION_VERSION,
                network=sources.network,
                route_template=sources.route_template,
                sumo_configuration=sources.sumo_configuration,
                generator_inputs=sources.generator_inputs,
                route_entity_lines=request.route_entity_lines,
                groups=parameters.groups,
                expected_scenario_count=len(plan),
                complete=True,
                scenarios=tuple(scenario_entries),
            )
            root = publication.publish(manifest)
        return GeneratedEvaluationLibrary(
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_digest(manifest),
        )


__all__ = [
    "EvaluationLibraryGenerationRequest",
    "EvaluationLibraryGenerator",
    "GroupedEvaluationLibraryGenerationRequest",
    "GroupedEvaluationLibraryParameters",
    "GeneratedEvaluationLibrary",
    "MaterializedScenario",
    "ScenarioMaterializationRequest",
    "ScenarioMaterializer",
    "SourceFile",
]
