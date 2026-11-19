# Reproducing the Susan experiment

## Source and inputs

The source snapshot uses commit
`d63984a3e22e7ccb585a168ae4d9ba5e683d6a47` dated 30 September 2026.
`PUBLICATION-SOURCE.json` records the source and library commits.
The held-out input is `publication-upperhalf-4x25-20260930` with
master seed `20260930`. Its manifest and all 100 scenarios are included.

The paper reports two trained checkpoints.

| Framework | Session | Checkpoint | Session model version |
| --- | --- | --- | --- |
| SB3 | `2026-09-30_09-49-18` | `scenario_20_checkpoint_2936010_steps.zip` | `v1.3.231` |
| Ray RLlib | `2026-09-29_19-08-33` | `scenario_20_checkpoint_18593792_steps` | `v1.3.230` |

The paper records the checkpoint digests below.

- SB3 `a311d08cd84bdd92d58933ee85d24eb6fc291240c2f643dedd1c59cfa480b00e`
- Ray RLlib `b54b0f275288f9ef9847aa55cb1587b78308a9ba5cf7ca570dd781997c23fd8b`

These sessions predate the selected source snapshot, which identifies itself as
Susan `v1.3.232`. The paper provenance does not identify the exact source
checkout. This repository contains source and scenario inputs. The original
trained weights and original generated training routes are separate artifacts.
A fresh run uses newly generated traffic and learns a new policy.

## Train Susan

Complete the [installation](runbooks/installation.md), then follow the
[training steps](guide/using-the-application.md#start-a-training-session).
Select SB3 or Ray RLlib and create a fresh session.

The runtime generates 21 training cases on first use according to the
framework's `scenario_manager_configuration.yaml`. The endpoints of the
traffic schedule are 10 vehicles and one pedestrian and 200 vehicles and
25 pedestrians. SUMO advances in 0.05-second steps with up to 7000 active policy
steps after warmup. Curriculum rules can impose shorter limits. The route
generator invokes SUMO randomTrips with `--random`.
The shared PPO settings are in `resources/utils/training/ppo_config.py`.
Record the generated routes, framework, run configuration, hardware,
software versions and snapshot digest alongside each run.

The historical jitter reward uses the largest of the local, trend and
comfort estimates after weighting. The comfort estimate is scaled by the
ratio of the trend window length to the comfort window length, capped at
one. Each estimate is multiplied by 1.5 when its command history contains
both strictly positive and strictly negative applied acceleration commands.
The resulting maximum is divided by 3.0 and negated. The jitter penalty uses
discrete differences between applied commands.

Sessions and checkpoints are stored under Susan in the writable data
directory. Stop the run using the application control panel.

## Evaluate the held-out cases

Follow the [held-out evaluation runbook](runbooks/evaluation-library.md) to
select a compatible checkpoint, run the four groups of 25 cases and locate
the run evidence under Susan's `tensorboard/evaluation/libraries/`.

Verify the held-out scenario files before evaluating. The temporary training
directory in this command keeps the verification independent of generated routes.

```bash
python - <<'PY'
from pathlib import Path
from tempfile import TemporaryDirectory
from lanista.reinforcengine.utils.scenarios.evaluation_library.publication import read_verified_library
from lanista.reinforcengine.utils.scenarios.evaluation_library.storage import EvaluationLibraryStorageConfiguration

resources = Path('lanista/rl_model_storage/models/susan_project/resources')
with TemporaryDirectory() as training_root:
    storage = EvaluationLibraryStorageConfiguration(
        training_root=training_root,
        library_root=resources / 'evaluation_libraries',
    )
    library = read_verified_library(
        storage,
        'publication-upperhalf-4x25-20260930',
        expected_manifest_sha256='16a2cd1f2ac7c3677a8eff37b248a64af4dfe57fa82a0c0c743b1f9df44533e1',
    )
    print(library.library_id, len(library.groups), len(library.scenarios))
PY
```

The expected counts are four groups and 100 scenarios. Compare frameworks on
the same library and report the checkpoints, training exposures and
contributing episode counts.

## Scope of the statistical reports

This snapshot produces run-level summaries. The paper's paired IQM,
group-stratified bootstrap intervals and post-departure stop counts use a
separate analysis workflow. Those tables cannot be regenerated with the
included report generator alone.

The snapshot's `control_deviation_mean` averages the absolute difference
between the applied and supervised command fields on each trace row with two
finite values. Those fields refer to different control intervals. The
measurement therefore includes changes between intervals. A later analysis
uses the absolute recorded policy-to-QP correction within one control
interval. Keep the two definitions distinct when comparing reports.
