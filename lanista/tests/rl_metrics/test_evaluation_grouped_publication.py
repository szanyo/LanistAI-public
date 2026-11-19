# Copyright (c) 2026 Benedek Szanyó.
# Copyright (c) 2026 HUN-REN SZTAKI.
# All rights reserved.

"""Check frozen grouped-run evidence and atomic publication reports."""

from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lanista.reinforcengine.rl_metrics.evaluation_report import EvaluationReportService
from lanista.reinforcengine.rl_model.evaluation.contracts import (
    EvaluationScenarioExecutionRecord,
    EvaluationSweepDescriptor,
    RunConfigurationSnapshot,
)
from lanista.reinforcengine.rl_model.evaluation.coordinator import (
    EVALUATION_LIBRARY_RUN_FILENAME,
    SCENARIO_RESULT_FILENAME,
)
from lanista.reinforcengine.utils.scenarios.evaluation_library.contracts import (
    EvaluationGroupDefinition,
    FileDigest,
    GroupedEvaluationLibraryManifest,
    GroupedEvaluationScenarioEntry,
    SourceIdentity,
    canonical_manifest_bytes,
)

_TIME = "2026-09-30T00:00:00Z"


def _canonical_bytes(payload: object) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _fixture(root: Path) -> None:
    source = SourceIdentity(kind="network", path="inputs/network.net.xml", sha256="a" * 64)
    groups = (
        EvaluationGroupDefinition(
            group_id="group-001", name="low", vehicle_count=15, pedestrian_count=10,
            scenario_count=1,
        ),
        EvaluationGroupDefinition(
            group_id="group-002", name="high", vehicle_count=30, pedestrian_count=10,
            scenario_count=3,
        ),
    )
    scenarios = tuple(
        GroupedEvaluationScenarioEntry(
            group_id=group.group_id,
            group_scenario_index=index,
            scenario_id=f"{group.group_id}-scenario-{index + 1:04d}",
            relative_path=f"scenarios/{group.group_id}-scenario-{index + 1:04d}",
            requested_vehicle_count=group.vehicle_count,
            requested_pedestrian_count=group.pedestrian_count,
            realised_vehicle_count=group.vehicle_count,
            realised_pedestrian_count=group.pedestrian_count,
            replicate_index=index,
            seed=10 + sequence,
            files=(FileDigest(relative_path="route.rou.xml", byte_size=1, sha256="b" * 64),),
        )
        for sequence, (group, index) in enumerate(
            (group, index) for group in groups for index in range(group.scenario_count)
        )
    )
    manifest = GroupedEvaluationLibraryManifest(
        schema_version=2,
        library_id="library-a",
        name="Study A",
        description=None,
        created_at=_TIME,
        generator_version="test",
        purpose="evaluation",
        master_seed=1,
        seed_derivation_version="sha256-group-v1",
        network=source,
        route_template=source.model_copy(update={"kind": "route-template"}),
        sumo_configuration=source.model_copy(update={"kind": "sumo-configuration"}),
        generator_inputs=(),
        route_entity_lines=("<vType id='test'/>",),
        expected_scenario_count=4,
        complete=True,
        groups=groups,
        scenarios=scenarios,
    )
    manifest_bytes = canonical_manifest_bytes(manifest)
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    _write(root / "library-manifest.json", manifest_bytes)
    snapshot = RunConfigurationSnapshot(
        model_id="model-a", framework="sb3", session_id="session-a",
        checkpoint_id="checkpoint-a", checkpoint_sha256="c" * 64,
        library_id="library-a", library_manifest_sha256=manifest_digest,
        deterministic_inference=True, repetitions_per_scenario=2,
        headless_pacing={}, effective_runtime_configuration={}, software_versions={},
    )
    snapshot_bytes = _canonical_bytes(snapshot.model_dump(mode="json", exclude_none=False))
    _write(root / "run-configuration.json", snapshot_bytes)
    descriptor = EvaluationSweepDescriptor(
        schema_version=1, run_id="run-a", framework="sb3", model_id="model-a",
        session_id="session-a", checkpoint_id="checkpoint-a",
        checkpoint_sha256="c" * 64, library_id="library-a",
        library_manifest_sha256=manifest_digest,
        run_configuration_sha256=hashlib.sha256(snapshot_bytes).hexdigest(),
        deterministic_inference=True, repetitions_per_scenario=2,
        started_at=_TIME, ended_at=_TIME, status="completed",
        expected_scenario_count=4, completed_scenario_count=4,
        failed_scenario_count=0, software_versions={},
    )
    _write(root / EVALUATION_LIBRARY_RUN_FILENAME, descriptor.model_dump_json().encode())
    for sequence, scenario in enumerate(scenarios):
        evidence_root = root / "scenarios" / scenario.scenario_id
        evidence_root.mkdir(parents=True)
        record = EvaluationScenarioExecutionRecord(
            schema_version=1, sequence_index=sequence, scenario_id=scenario.scenario_id,
            requested_vehicle_count=scenario.requested_vehicle_count,
            requested_pedestrian_count=scenario.requested_pedestrian_count,
            replicate_index=scenario.replicate_index, seed=scenario.seed,
            status="completed", started_at=_TIME, ended_at=_TIME,
            evidence_files=("evaluation-run.json", "evaluation_traces/a/evaluation_trace.csv"),
            terminal_outcome="route_end",
        )
        _write(evidence_root / SCENARIO_RESULT_FILENAME, record.model_dump_json().encode())
        _write(evidence_root / "evaluation-run.json", b"{}\n")
        trace = evidence_root / "evaluation_traces" / "a" / "evaluation_trace.csv"
        trace.parent.mkdir(parents=True)
        with trace.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("episode_index", "termination_reason", "actual_speed_mps"),
                lineterminator="\n",
            )
            writer.writeheader()
            speed = 0 if sequence == 0 else 10
            writer.writerow({"episode_index": 0, "termination_reason": "route_end", "actual_speed_mps": speed})
            writer.writerow({"episode_index": 1, "termination_reason": "route_end", "actual_speed_mps": speed})


class GroupedPublicationTests(unittest.TestCase):
    def test_scenario_weighted_report_is_complete_and_repeatable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root)
            first = EvaluationReportService().generate(root)
            files = {path.name: path.read_bytes() for path in first.summary_root.iterdir()}
            second = EvaluationReportService().generate(root)
            payload = json.loads((first.summary_root / "library-summary.json").read_text())
            self.assertEqual(files, {path.name: path.read_bytes() for path in second.summary_root.iterdir()})
            self.assertEqual(
                {"scenario-summary.csv", "group-summary.csv", "grid-summary.csv", "library-summary.json"},
                set(files),
            )
            self.assertEqual(4, len(first.scenarios))
            self.assertEqual(2, len(first.groups))
            self.assertEqual(7.5, first.library.metrics["mean_speed_mps"].mean)
            self.assertEqual(4, first.library.metrics["mean_speed_mps"].count)
            self.assertEqual(7.5, payload["metrics"]["mean_speed_mps"]["mean"])
            with (first.summary_root / "scenario-summary.csv").open(newline="") as handle:
                rows = tuple(csv.DictReader(handle))
            self.assertEqual(4, len(rows))
            self.assertEqual("low", rows[0]["group_name"])
            self.assertEqual("15", rows[0]["realised_vehicle_count"])

    def test_tampered_frozen_manifest_does_not_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _fixture(root)
            with (root / "library-manifest.json").open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                EvaluationReportService().generate(root)
            self.assertFalse((root / "summary").exists())


if __name__ == "__main__":
    unittest.main()
