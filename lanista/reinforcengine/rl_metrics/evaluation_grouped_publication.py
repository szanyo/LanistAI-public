# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Verify frozen grouped-run inputs and publish a complete report directory."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationSweepDescriptor,
    RunConfigurationSnapshot,
)
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    LIBRARY_MANIFEST_SNAPSHOT_FILENAME,
    RUN_CONFIGURATION_FILENAME,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    GroupedEvaluationLibraryManifest,
    canonical_manifest_bytes,
    validate_manifest_payload,
)

_REPORT_FILENAMES = frozenset(
    {"scenario-summary.csv", "group-summary.csv", "grid-summary.csv", "library-summary.json"}
)


def _regular_bytes(path: Path) -> bytes:
    """Read one immutable regular file without accepting a symlink alias."""

    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Missing or unsafe frozen evaluation evidence: {path.name}.")
    return path.read_bytes()


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def read_grouped_run_context(
    run_root: Path,
    descriptor: EvaluationSweepDescriptor,
) -> GroupedEvaluationLibraryManifest:
    """Prove the snapshotted manifest and effective configuration match the run."""

    if descriptor.run_configuration_sha256 is None:
        raise RuntimeError("Grouped run lacks its configuration digest.")
    manifest_bytes = _regular_bytes(run_root / LIBRARY_MANIFEST_SNAPSHOT_FILENAME)
    if hashlib.sha256(manifest_bytes).hexdigest() != descriptor.library_manifest_sha256:
        raise RuntimeError("Frozen library manifest digest does not match the run.")
    try:
        manifest = validate_manifest_payload(json.loads(manifest_bytes))
    except (ValueError, TypeError) as error:
        raise RuntimeError("Frozen library manifest is invalid.") from error
    if (
        not isinstance(manifest, GroupedEvaluationLibraryManifest)
        or not manifest.complete
        or canonical_manifest_bytes(manifest) != manifest_bytes
        or manifest.library_id != descriptor.library_id
        or manifest.expected_scenario_count != descriptor.expected_scenario_count
    ):
        raise RuntimeError("Frozen library manifest does not match the completed run.")

    configuration_bytes = _regular_bytes(run_root / RUN_CONFIGURATION_FILENAME)
    if hashlib.sha256(configuration_bytes).hexdigest() != descriptor.run_configuration_sha256:
        raise RuntimeError("Frozen run configuration digest does not match the run.")
    try:
        configuration = RunConfigurationSnapshot.model_validate_json(configuration_bytes)
    except ValueError as error:
        raise RuntimeError("Frozen run configuration is invalid.") from error
    if _canonical_json_bytes(configuration.model_dump(mode="json", exclude_none=False)) != configuration_bytes:
        raise RuntimeError("Frozen run configuration is not canonical.")
    matching_fields = (
        "model_id",
        "framework",
        "session_id",
        "checkpoint_id",
        "checkpoint_sha256",
        "library_id",
        "library_manifest_sha256",
        "deterministic_inference",
        "repetitions_per_scenario",
        "software_versions",
    )
    if any(getattr(configuration, name) != getattr(descriptor, name) for name in matching_fields):
        raise RuntimeError("Frozen run configuration does not match its descriptor.")
    return manifest


def publish_grouped_report(run_root: Path, files: Mapping[str, bytes]) -> Path:
    """Make four validated artifacts visible together; never overwrite a partial report."""

    if set(files) != _REPORT_FILENAMES:
        raise ValueError("Grouped publication requires exactly four report artifacts.")
    summary_root = run_root / "summary"
    if os.path.lexists(summary_root):
        if summary_root.is_symlink() or not summary_root.is_dir():
            raise RuntimeError("Evaluation summary root is unsafe.")
        existing = tuple(summary_root.iterdir())
        if {path.name for path in existing} != _REPORT_FILENAMES or any(
            path.is_symlink() or not path.is_file() for path in existing
        ):
            raise RuntimeError("Existing grouped report is incomplete or unsafe.")
        if any((summary_root / name).read_bytes() != content for name, content in files.items()):
            raise RuntimeError("Existing grouped report differs from frozen evidence.")
        return summary_root

    with tempfile.TemporaryDirectory(prefix=".summary-stage-", dir=run_root) as temporary:
        staged = Path(temporary) / "summary"
        staged.mkdir()
        for name, content in files.items():
            with (staged / name).open("xb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        if any((staged / name).read_bytes() != content for name, content in files.items()):
            raise RuntimeError("Staged grouped report failed readback validation.")
        if os.path.lexists(summary_root):
            raise RuntimeError("Evaluation summary root appeared during publication.")
        os.rename(staged, summary_root)
    return summary_root


__all__ = ["publish_grouped_report", "read_grouped_run_context"]
