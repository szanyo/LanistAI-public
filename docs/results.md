# Paper results and data

The experiment compares two fixed PPO checkpoints on the same 100 held-out
traffic scenarios. The four traffic groups contain 25 cases each. The
[scenario manifest](../lanista/rl_model_storage/models/susan_project/resources/evaluation_libraries/publication-upperhalf-4x25-20260930/evaluation-library.json)
lists the requested and realised traffic counts, scenario IDs, seeds and
file digests. The [reproduction guide](reproduction.md) lists the training
sessions and checkpoint digests.

## Results at a glance

| Held-out measure | Paired IQM, Ray minus SB3 | 95% interval |
|---|---:|---:|
| Episode return | +219.33 | [+192.07, +260.74] |
| Mean absolute applied command (m/s²) | −0.056 | [−0.068, −0.044] |
| Realised acceleration RMS (m/s²) | −0.032 | [−0.040, −0.023] |
| Stopped-step fraction (percentage points) | −1.60 | [−3.93, +0.59] |
| Post-departure stops per scenario | 0.00 | [−0.04, +0.10] |

The IQM trims 25% from each tail of the scenario differences. The intervals
use 10,000 paired bootstrap draws stratified by traffic group. They describe
scenario-sampling uncertainty for the two checkpoints. The source values and
method are in the [paired statistics](../results/extended_evaluation/paired-continuous-statistics.csv)
and [statistical provenance](../results/extended_evaluation/statistical-provenance.json).

## Source files

| Figure or table | Data files |
|---|---|
| Training return by environment step and elapsed wall time | [Training progression](../results/data/training-progression.csv) |
| Framework-native execution rates | [Native execution rate](../results/data/native-execution-rate.csv) |
| Requested and realised traffic by load group | [Traffic realisation](../results/data/traffic-realisation.csv) |
| Vehicle-level means, dispersion and held-out returns | [Platform statistics](../results/data/heldout-platform-statistics.csv) and [extended platform statistics](../results/extended_evaluation/platform-continuous-statistics.csv) |
| Paired return, command demand and stopped-time distributions | [Case-level paired values](../results/paired-scenario-values.csv) and [descriptive paired differences](../results/data/heldout-paired-differences.csv) |
| Paired IQM and confidence intervals | [Paired continuous statistics](../results/extended_evaluation/paired-continuous-statistics.csv) |
| Trip completion and traffic-event rates | [Paired event rates](../results/extended_evaluation/paired-event-rates.csv) |
| Post-departure stops | [Scenario stop counts](../results/extended_evaluation/scenario-stop-counts.csv) |
| Spatial speed and speed-limit profiles | [Scenario and route-bin values](../results/data/speed-route-profile-scenarios.csv), [profile statistics](../results/data/speed-route-profile-statistics.csv) and [profile provenance](../results/data/speed-route-profile-provenance.json) |
| Single-scenario speed trace in the paper | **group-003-scenario-0001**. The two original trace paths and SHA-256 values are listed in the [profile provenance](../results/data/speed-route-profile-provenance.json). |

In the evaluation CSVs, `group_id=library` denotes the whole held-out
library. Use `scenario_id` and `seed` to join case-level files. The
scenario and route-bin CSV is best opened in a local spreadsheet or a CSV
reader. The native rates retain each framework's own metric definition.
The event flags describe recorded simulator and control conditions.

## Provenance and coverage

The 14 CSV and JSON files under `results/data/` and
`results/extended_evaluation/` are byte-for-byte copies of the manuscript
data exports. [Snapshot provenance](../PUBLICATION-SOURCE.json) records
their source revision. The [comparison provenance](../results/data/comparison-provenance.json)
records the original run IDs, checkpoint hashes, library digest and metric
scope. The profile provenance identifies the 200 input traces by relative
path and SHA-256.

[Case-level paired values](../results/paired-scenario-values.csv) contain
seven verified metrics for each of the 100 paired scenarios. They are a
selection from the archived 3 October 2026 reanalysis of the same two runs
(`paired-iqm-stops-001`). The source CSV has SHA-256
`9069e1e4b6bc8954d7a004809801356f9ce94dfc761ad4c6abd79f89672a7aab`.
Recomputing IQMs and intervals from these rows reproduces the corresponding
paper CSV values. The two full per-step trace sets and the original TensorBoard
events remain separate artifacts. The [reproduction guide](reproduction.md)
identifies the included checkpoints and explains how to evaluate them.
