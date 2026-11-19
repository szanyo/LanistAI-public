<!--
Copyright (c) 2026 Benedek Szanyó.
Copyright (c) 2026 HUN-REN SZTAKI.
All rights reserved.
-->

# Held-out evaluation-library workflow

This runbook evaluates one frozen Ray RLlib or Stable-Baselines3 checkpoint
over the complete Susan library and regenerates its run-level tables. The
[reproduction guide](../reproduction.md) identifies the paper inputs and
explains which statistical calculations are available in this snapshot.

## Verify the immutable library

Use `publication-upperhalf-4x25-20260930` with master seed `20260930`.
Each of the four requested-demand groups has 25 independently seeded cases.

| Group | Requested vehicles | Requested pedestrians |
|---|---:|---:|
| group-001 | 114 | 14 |
| group-002 | 143 | 17 |
| group-003 | 171 | 21 |
| group-004 | 200 | 25 |

Use the [library verification command](../reproduction.md#evaluate-the-held-out-cases)
to check the manifest and its file digests. The manifest records scenario IDs,
seeds and realised populations. Keep the evaluation library outside the fixed
training scenario directory. `EvaluationLibraryStorageConfiguration` rejects
overlapping and symlink-aliased roots.

## Select one checkpoint and execute the complete sweep

1. Load Susan and open Configure with F1.
2. Choose Evaluation, the framework, the session and an explicit checkpoint.
3. Select the Library evaluation profile and the paper library. Use one
   repetition per scenario for the paper protocol.
4. Apply the configuration and use Start. Library inference is deterministic
   and runs headless at maximum throughput.

The runtime revalidates the active framework, session, checkpoint, checkpoint
bytes, library manifest, and repetition count before launch. SB3 restores one
cached `PPO` and calls `predict()`. Ray restores one cached evaluation-mode
`RLModule` and calls `forward_inference()`. Both use the local evaluation
environment and run without training.

The coordinator visits each manifest entry once, in manifest order. The
application stores the run in Susan's writable model directory.

```text
tensorboard/evaluation/libraries/<run-id>/
  evaluation-library-run.json
  scenarios/<scenario-id>/
    scenario-result.json
```

Use the per-scenario evidence inventory to locate each trace and TensorBoard
subdirectory.

## Cancellation, failure, and retry policy

A stop request is cooperative. It prevents a new scenario from starting or
cancels the active adapter at its cancellation point, then closes its
environment, trace, and TensorBoard writers. The top-level descriptor is marked
`cancelled`. Use it only for diagnosis.

Scenario failures are persisted as explicit failed rows and traversal continues
where possible. A grouped sweep with a failure has status `failed` and can
retain a partial summary. After correcting the issue, start a new run with
the same library and checkpoint and keep the preceding run for diagnosis.

## Regenerate the offline report

From the repository root with the Python environment active, pass the path
containing `evaluation-library-run.json`.

```bash
python -m lanista.reinforcengine.rl_metrics.evaluation_report_cli \
  /absolute/path/to/library-run
```

For a completed grouped run the command writes:

```text
summary/
  scenario-summary.csv
  group-summary.csv
  grid-summary.csv
  library-summary.json
```

Report generation verifies the frozen configuration, evidence and library
manifest identities. Repeating the command accepts an identical existing
summary and errors if the existing summary differs.

## Hand off evidence to the manuscript workflow

Treat one scenario row as the statistical unit. Use `scenario-summary.csv`
for per-scenario values and `group-summary.csv` for descriptive statistics
by requested-demand group. Keep the run descriptor, library manifest and
summary files together so the available values and input identities remain
auditable. Keep missing values explicit. The
[reproduction scope](../reproduction.md#scope-of-the-statistical-reports)
covers the paper's separate paired analysis.
