# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Define the immutable schema for one held-out scenario library."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

EVALUATION_LIBRARY_SCHEMA_VERSION = 1
GROUPED_EVALUATION_LIBRARY_SCHEMA_VERSION = 2
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
NonNegativeInt = Annotated[int, Field(strict=True, ge=0)]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def _validated_relative_path(value: str) -> str:
    """Return one normalized POSIX path that cannot leave its declared root."""

    raw_value = str(value)
    path = PurePosixPath(raw_value)
    if (
        not raw_value
        or raw_value != path.as_posix()
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in raw_value
    ):
        raise ValueError(f"Path must be a normalized relative POSIX path: {value!r}.")
    return raw_value


def validate_library_id(value: str) -> str:
    """Require one portable, non-padded filesystem component."""

    raw_value = str(value)
    if (
        not raw_value
        or raw_value != raw_value.strip()
        or raw_value in {".", ".."}
        or PurePosixPath(raw_value).name != raw_value
        or "\\" in raw_value
    ):
        raise ValueError(f"Identifier must be one relative path component: {value!r}.")
    return raw_value


def validate_utc_timestamp(value: str) -> str:
    """Require a second-precision UTC timestamp with the canonical ``Z`` suffix."""

    raw_value = str(value)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", raw_value) is None:
        raise ValueError("Timestamp must be second-precision UTC with a Z suffix.")
    try:
        datetime.strptime(raw_value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("Timestamp is not a valid ISO-8601 UTC instant.") from exc
    return raw_value


class _FrozenContract(BaseModel):
    """Apply strict immutable behavior shared by persisted library contracts."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FileDigest(_FrozenContract):
    """Identify one regular file relative to its owning scenario directory."""

    relative_path: str
    byte_size: NonNegativeInt
    sha256: Sha256

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return _validated_relative_path(value)


class SourceIdentity(_FrozenContract):
    """Record one input identity used to materialise the library."""

    kind: str = Field(min_length=1)
    path: str
    sha256: Sha256

    @field_validator("kind")
    @classmethod
    def _validate_kind(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != value:
            raise ValueError("Source kind must not contain surrounding whitespace.")
        return normalized

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        return _validated_relative_path(value)


class EvaluationScenarioEntry(_FrozenContract):
    """Describe one ordered scenario and its reproducibility evidence."""

    scenario_id: str
    relative_path: str
    requested_vehicle_count: NonNegativeInt
    requested_pedestrian_count: NonNegativeInt
    replicate_index: NonNegativeInt
    seed: NonNegativeInt
    realised_vehicle_count: NonNegativeInt
    realised_pedestrian_count: NonNegativeInt
    realisation_note: str | None = None
    files: tuple[FileDigest, ...] = Field(min_length=1)

    @field_validator("scenario_id")
    @classmethod
    def _validate_scenario_id(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return _validated_relative_path(value)

    @field_validator("realisation_note")
    @classmethod
    def _normalize_realisation_note(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_realisation_and_files(self) -> EvaluationScenarioEntry:
        population_differs = (
            self.requested_vehicle_count != self.realised_vehicle_count
            or self.requested_pedestrian_count != self.realised_pedestrian_count
        )
        if population_differs and self.realisation_note is None:
            raise ValueError("Requested and realised population differences require a note.")
        file_paths = tuple(file.relative_path for file in self.files)
        if len(file_paths) != len(set(file_paths)):
            raise ValueError("Scenario file paths must be unique.")
        if file_paths != tuple(sorted(file_paths)):
            raise ValueError("Scenario files must use canonical path ordering.")
        return self


class EvaluationGroupDefinition(_FrozenContract):
    """Define one named, ordered demand group within a grouped library."""

    group_id: str
    name: str
    vehicle_count: NonNegativeInt
    pedestrian_count: NonNegativeInt
    scenario_count: PositiveInt

    @field_validator("group_id")
    @classmethod
    def _validate_group_id(cls, value: str) -> str:
        if re.fullmatch(r"group-\d{3,}", value) is None or int(value[6:]) < 1:
            raise ValueError("Group ID must be a stable group-001 style identifier.")
        return value

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value or value != value.strip():
            raise ValueError("Group name must be nonempty and trimmed.")
        return value


class GroupedEvaluationScenarioEntry(EvaluationScenarioEntry):
    """Bind one scenario to its stable group and zero-based local index."""

    group_id: str
    group_scenario_index: NonNegativeInt

    @field_validator("group_id")
    @classmethod
    def _validate_group_id(cls, value: str) -> str:
        return EvaluationGroupDefinition._validate_group_id(value)

    @model_validator(mode="after")
    def _validate_group_identity(self) -> GroupedEvaluationScenarioEntry:
        expected_id = f"{self.group_id}-scenario-{self.group_scenario_index + 1:04d}"
        if self.scenario_id != expected_id:
            raise ValueError("Grouped scenario ID must match its group and local index.")
        if self.relative_path != f"scenarios/{expected_id}":
            raise ValueError("Grouped scenario path must match its scenario ID.")
        if self.replicate_index != self.group_scenario_index:
            raise ValueError("Replicate index must equal group scenario index.")
        return self


class EvaluationLibraryManifest(_FrozenContract):
    """Validate the complete persistent identity of one scenario library."""

    schema_version: Literal[1]
    library_id: str
    created_at: str
    generator_version: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    master_seed: NonNegativeInt
    seed_derivation_version: str = Field(min_length=1)
    network: SourceIdentity
    route_template: SourceIdentity
    sumo_configuration: SourceIdentity
    generator_inputs: tuple[SourceIdentity, ...]
    expected_scenario_count: PositiveInt
    complete: bool
    scenarios: tuple[EvaluationScenarioEntry, ...]

    @field_validator("library_id")
    @classmethod
    def _validate_library_id(cls, value: str) -> str:
        return validate_library_id(value)

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, value: str) -> str:
        return validate_utc_timestamp(value)

    @field_validator("generator_version", "purpose", "seed_derivation_version")
    @classmethod
    def _validate_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if normalized != value or not normalized:
            raise ValueError("Required manifest text must not be empty or padded.")
        return normalized

    @model_validator(mode="after")
    def _require_canonical_scenario_order(self) -> EvaluationLibraryManifest:
        input_order = tuple((item.kind, item.path) for item in self.generator_inputs)
        if input_order != tuple(sorted(input_order)):
            raise ValueError("Generator inputs must use canonical kind and path ordering.")
        scenario_ids = tuple(item.scenario_id for item in self.scenarios)
        scenario_paths = tuple(item.relative_path for item in self.scenarios)
        scenario_seeds = tuple(item.seed for item in self.scenarios)
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("Scenario identifiers must be unique.")
        if len(scenario_paths) != len(set(scenario_paths)):
            raise ValueError("Scenario paths must be unique.")
        if len(scenario_seeds) != len(set(scenario_seeds)):
            raise ValueError("Scenario seeds must be unique.")
        parsed_roots = tuple(PurePosixPath(path) for path in scenario_paths)
        for index, scenario_root in enumerate(parsed_roots):
            for other_root in parsed_roots[index + 1 :]:
                if scenario_root in other_root.parents or other_root in scenario_root.parents:
                    raise ValueError("Scenario roots must not overlap.")
        expanded_paths = tuple(
            (PurePosixPath(scenario.relative_path) / file.relative_path).as_posix()
            for scenario in self.scenarios
            for file in scenario.files
        )
        if len(expanded_paths) != len(set(expanded_paths)):
            raise ValueError("Expanded scenario artifact paths must be unique.")
        canonical_order = tuple(
            sorted(
                self.scenarios,
                key=lambda item: (
                    item.requested_vehicle_count,
                    item.requested_pedestrian_count,
                    item.replicate_index,
                    item.scenario_id,
                ),
            )
        )
        if self.scenarios != canonical_order:
            raise ValueError("Scenarios must use canonical factor and replicate ordering.")
        if len(self.scenarios) > self.expected_scenario_count:
            raise ValueError("A library cannot exceed its expected scenario count.")
        if self.complete and len(self.scenarios) != self.expected_scenario_count:
            raise ValueError("A complete library must contain its expected scenario count.")
        return self


class GroupedEvaluationLibraryManifest(EvaluationLibraryManifest):
    """Validate one grouped version-2 library and all group memberships."""

    schema_version: Literal[2]
    name: str
    description: str | None = None
    route_entity_lines: tuple[str, ...]
    groups: tuple[EvaluationGroupDefinition, ...] = Field(min_length=1)
    scenarios: tuple[GroupedEvaluationScenarioEntry, ...]

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return EvaluationGroupDefinition._validate_name(value)

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str | None) -> str | None:
        if value is not None and (not value or value != value.strip()):
            raise ValueError("Description must be nonempty and trimmed when present.")
        return value

    @model_validator(mode="after")
    def _require_canonical_scenario_order(self) -> GroupedEvaluationLibraryManifest:
        input_order = tuple((item.kind, item.path) for item in self.generator_inputs)
        if input_order != tuple(sorted(set(input_order))):
            raise ValueError("Generator inputs must use unique canonical ordering.")

        group_ids = tuple(group.group_id for group in self.groups)
        if group_ids != tuple(sorted(set(group_ids), key=lambda value: int(value[6:]))):
            raise ValueError("Groups must have unique IDs in creation order.")
        names = tuple(unicodedata.normalize("NFKC", group.name).casefold() for group in self.groups)
        if len(names) != len(set(names)):
            raise ValueError("Group names must be unique after case-insensitive normalization.")

        expected_count = sum(group.scenario_count for group in self.groups)
        if self.expected_scenario_count != expected_count:
            raise ValueError("Expected scenario count must equal the sum of group counts.")
        expected_order = tuple(
            (group.group_id, index)
            for group in self.groups
            for index in range(group.scenario_count)
        )
        actual_order = tuple(
            (scenario.group_id, scenario.group_scenario_index)
            for scenario in self.scenarios
        )
        if actual_order != expected_order:
            raise ValueError("Scenarios must use contiguous group-local indexes in group order.")

        groups_by_id = {group.group_id: group for group in self.groups}
        for scenario in self.scenarios:
            group = groups_by_id[scenario.group_id]
            if (
                scenario.requested_vehicle_count != group.vehicle_count
                or scenario.requested_pedestrian_count != group.pedestrian_count
            ):
                raise ValueError("Scenario demand must match its group definition.")

        ids = tuple(scenario.scenario_id for scenario in self.scenarios)
        paths = tuple(scenario.relative_path for scenario in self.scenarios)
        seeds = tuple(scenario.seed for scenario in self.scenarios)
        if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
            raise ValueError("Scenario identifiers and paths must be unique.")
        if len(seeds) != len(set(seeds)):
            raise ValueError("Scenario seeds must be unique.")
        if any(seed < 1 or seed > 2**31 - 1 for seed in seeds):
            raise ValueError("Scenario seed is outside SUMO's positive range.")
        roots = tuple(PurePosixPath(path) for path in paths)
        for index, root in enumerate(roots):
            for other in roots[index + 1 :]:
                if root in other.parents or other in root.parents:
                    raise ValueError("Scenario roots must not overlap.")
        return self


AnyEvaluationLibraryManifest = EvaluationLibraryManifest | GroupedEvaluationLibraryManifest


def validate_manifest_payload(payload: Mapping[str, object]) -> AnyEvaluationLibraryManifest:
    """Validate an external manifest using its explicit version."""

    if payload.get("schema_version") == GROUPED_EVALUATION_LIBRARY_SCHEMA_VERSION:
        return GroupedEvaluationLibraryManifest.model_validate(dict(payload))
    return EvaluationLibraryManifest.model_validate(dict(payload))


def canonical_manifest_bytes(manifest: AnyEvaluationLibraryManifest) -> bytes:
    """Serialize a validated manifest to stable UTF-8 JSON bytes."""

    payload = manifest.model_dump(mode="json", exclude_none=False)
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def manifest_digest(manifest: AnyEvaluationLibraryManifest) -> str:
    """Return the SHA-256 identity of the canonical manifest representation."""

    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


__all__ = [
    "EVALUATION_LIBRARY_SCHEMA_VERSION",
    "EvaluationLibraryManifest",
    "EvaluationScenarioEntry",
    "FileDigest",
    "SourceIdentity",
    "canonical_manifest_bytes",
    "manifest_digest",
    "validate_library_id",
    "validate_manifest_payload",
    "validate_utc_timestamp",
]
